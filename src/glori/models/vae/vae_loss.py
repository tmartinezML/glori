import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

import glori.models.vae.vae_utils as vae_utils


# Functions from taming transformers codebase
# taming.modules.losses.vqperceptual
def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


def wasserstein_d_loss(logits_real, logits_fake):
    d_loss = torch.mean(logits_fake) - torch.mean(logits_real)
    return d_loss


def gp_loss(
    real_images,
    fake_images,
    discriminator,
    lambda_gp=10.0,
):
    """
    Gradient penalty for WGAN-GP
    """
    batch_size = real_images.shape[0]
    alpha = torch.rand(batch_size, 1, 1, 1).to(real_images.device)
    interpolates = alpha * real_images + (1 - alpha) * fake_images
    interpolates = interpolates.requires_grad_()
    is_training = discriminator.training
    if not is_training:
        discriminator.train()
    # Context necessary for this to work in validation step
    with torch.enable_grad():
        d_interpolates = discriminator(interpolates).contiguous()

    if not is_training:
        discriminator.eval()

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    # Epsilon for numerical stability
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
    gradient_penalty = lambda_gp * ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty, gradients_norm.mean()


# Functions from taming transformers codebase
# taming.modules.discriminator.model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


class ActNorm(nn.Module):
    def __init__(
        self, num_features, logdet=False, affine=True, allow_reverse_init=False
    ):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm="batch"):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        match norm:
            case "batch":
                norm_layer = nn.BatchNorm2d
                use_bias = False

            case "instance":
                norm_layer = nn.InstanceNorm2d
                use_bias = False

            case "actnorm":
                norm_layer = ActNorm
                use_bias = True

            case _:
                raise ValueError(
                    f"Unsupported normalization layer: {norm}. "
                    "Supported options are: 'batch', 'instance', 'actnorm'."
                )

        kw = 4
        padw = 1
        sequence = [
            # nn.Tanh(),
            nn.Identity(),  # no activation at the start
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            # norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class L1WithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        disc_ramp_steps=1000,
        logvar_init=0.0,
        kl_weight=1.0,
        kl_I_weight=1.0,
        pixelloss_weight=1.0,
        unscaled_pixelloss_weight=0.0,
        disc_num_layers=3,
        disc_in_channels=1,
        disc_factor=1.0,
        disc_weight=1.0,
        pretrain_disc=False,
        disc_norm="batch",
        use_adapt_weight=True,
        disc_conditional=False,
        reg_loss="kl",  # kl, kl-enhanced,
        disc_loss="hinge",
        hpf_sigma=0.0,
    ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "wasserstein"]
        self.kl_weight = kl_weight
        self.kl_I_weight = kl_I_weight
        self.pixel_weight = pixelloss_weight
        self.unsc_pixel_weight = unscaled_pixelloss_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels, n_layers=disc_num_layers, norm=disc_norm
        ).apply(weights_init)
        self.high_pass_filter = nn.Identity()
        if hpf_sigma > 0.0:
            self.high_pass_filter = vae_utils.GaussianHighPassFilter(sigma=hpf_sigma)

        self.discriminator_iter_start = disc_start
        self.disc_ramp_steps = disc_ramp_steps
        self.loss_type = disc_loss
        self.reg_loss_type = reg_loss
        self.disc_loss = (
            hinge_d_loss
            if disc_loss == "hinge"
            else vanilla_d_loss if disc_loss == "vanilla" else wasserstein_d_loss
        )
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.use_adapt_weight = use_adapt_weight
        self.pretrain_disc = pretrain_disc

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def disc_ramp_weight(self, global_step):
        """
        Ramp up the discriminator weight from 0 to 1 over the first 1000 iterations.
        """
        if self.disc_ramp_steps == 0:
            return 1.0
        f = ((global_step - self.discriminator_iter_start) / self.disc_ramp_steps) ** 3
        return max(0.0, min(1.0, f))

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train",
        weights=None,
        scale_fn=None,
    ):
        # Autoencoder loss
        if optimizer_idx == 0:
            # L1 Loss
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

            nll_loss_unscaled = torch.tensor(0, dtype=torch.float32)
            rec_loss_unscaled = torch.tensor(0, dtype=torch.float32)
            if self.unsc_pixel_weight > 0.0:
                assert (
                    scale_fn is not None
                ), "scale_fn must be provided for unscaled pixel loss."
                rec_loss_unscaled = torch.abs(
                    scale_fn(inputs.contiguous())
                    - scale_fn(reconstructions.contiguous())
                )
                nll_loss_unscaled = (
                    rec_loss_unscaled / torch.exp(self.logvar) + self.logvar
                )
                nll_loss_unscaled = (
                    torch.sum(nll_loss_unscaled) / nll_loss_unscaled.shape[0]
                )

            # KL loss
            match self.reg_loss_type:
                case "kl":
                    kl_loss = posteriors.kl()
                    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                case "kl-enhanced":
                    kl_G_loss = posteriors.kl_G()
                    kl_I_loss = posteriors.kl_I()
                    kl_I_loss = torch.sum(kl_I_loss) / kl_I_loss.shape[0]
                    kl_I_delta = posteriors.kl_I_delta()
                    kl_I_delta = torch.sum(kl_I_delta) / kl_I_delta.shape[0]
                    assert (
                        len(set([t.shape for t in [kl_G_loss, kl_I_loss, kl_I_delta]]))
                        == 1
                    ), (
                        "All kl-loss tensors must have the same shape, "
                        f"got {kl_G_loss.shape}, {kl_I_loss.shape}"
                        f" and {kl_I_delta.shape}."
                    )
                    kl_loss = (
                        kl_G_loss + self.kl_I_weight * kl_I_loss
                    )  # + 0.01 * kl_I_delta

                case _:
                    raise ValueError(
                        f"Unsupported regularization loss type: {self.reg_loss_type}. "
                        "Supported options are: 'kl', 'kl-enhanced'."
                    )

            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(
                    self.high_pass_filter(reconstructions).contiguous()
                )
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat(
                        (self.high_pass_filter(reconstructions).contiguous(), cond),
                        dim=1,
                    )
                )
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = (
                        self.calculate_adaptive_weight(
                            nll_loss, g_loss, last_layer=last_layer
                        )
                        if self.use_adapt_weight
                        else torch.tensor(self.discriminator_weight)
                    )
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            disc_factor *= self.disc_ramp_weight(global_step)

            ae_loss = (
                self.pixel_weight * nll_loss
                + self.unsc_pixel_weight * nll_loss_unscaled
                + self.kl_weight * kl_loss
                + d_weight * disc_factor * g_loss
            )

            device = self.logvar.device
            log = {
                "{}/total_loss".format(split): ae_loss.clone().detach().mean(),
                "{}/logvar".format(split): self.logvar.detach(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/nll_loss_unscaled".format(split): nll_loss_unscaled.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/rec_loss_unscaled".format(split): rec_loss_unscaled.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach().to(device),
                "{}/disc_factor".format(split): torch.tensor(disc_factor).to(device),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }
            if self.reg_loss_type == "kl-enhanced":
                log.update(
                    {
                        "{}/kl_G_loss".format(split): kl_G_loss.detach(),
                        "{}/kl_I_loss".format(split): kl_I_loss.detach(),
                        "{}/kl_I_delta".format(split): kl_I_delta.detach(),
                    }
                )
            return ae_loss, log

        # Discriminator update
        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(
                    self.high_pass_filter(inputs).contiguous().detach()
                )
                logits_fake = self.discriminator(
                    self.high_pass_filter(reconstructions).contiguous().detach()
                )
            else:
                logits_real = self.discriminator(
                    torch.cat(
                        (self.high_pass_filter(inputs).contiguous().detach(), cond),
                        dim=1,
                    )
                )
                logits_fake = self.discriminator(
                    torch.cat(
                        (
                            self.high_pass_filter(reconstructions)
                            .contiguous()
                            .detach(),
                            cond,
                        ),
                        dim=1,
                    )
                )

            disc_factor = adopt_weight(
                self.disc_factor,
                global_step,
                threshold=self.discriminator_iter_start,
                value=int(
                    self.pretrain_disc
                ),  # Changed this so discriminator is pre-trained during warmup
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            # d_loss = self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }

            # Add gradient penalty if using Wasserstein loss
            if self.loss_type == "wasserstein":
                gp, gradients = gp_loss(
                    inputs.detach(),
                    reconstructions.detach(),
                    self.discriminator,
                    lambda_gp=10.0,
                )
                gp = gp * disc_factor

                log.update(
                    {
                        "{}/gradient_penalty".format(split): gp.detach(),
                        "{}/gradients".format(split): gradients.detach(),
                    }
                )
                d_loss = d_loss + gp

            return d_loss, log
