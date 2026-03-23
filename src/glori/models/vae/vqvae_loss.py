import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from glori.models.vae.vae_loss import (
    NLayerDiscriminator,
    hinge_d_loss,
    vanilla_d_loss,
    weights_init,
)


def hinge_d_loss_with_exemplar_weights(logits_real, logits_fake, weights):
    assert weights.shape[0] == logits_real.shape[0] == logits_fake.shape[0]
    loss_real = torch.mean(F.relu(1.0 - logits_real), dim=[1, 2, 3])
    loss_fake = torch.mean(F.relu(1.0 + logits_fake), dim=[1, 2, 3])
    loss_real = (weights * loss_real).sum() / weights.sum()
    loss_fake = (weights * loss_fake).sum() / weights.sum()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def measure_perplexity(predicted_indices, n_embed):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use


def l1(x, y):
    return torch.abs(x - y)


def l2(x, y):
    return torch.pow((x - y), 2)


class VQLossWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        unsc_pixel_weight=0.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        disc_norm="instance",
        disc_conditional=False,
        disc_loss="hinge",
        n_classes=None,
        pixel_loss="l1",
        pretrain_disc=False,
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        assert pixel_loss in ["l1", "l2"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.unsc_pixel_weight = unsc_pixel_weight
        self.pretrain_disc = pretrain_disc

        if pixel_loss == "l1":
            self.pixel_loss = l1
        else:
            self.pixel_loss = l2

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            norm=disc_norm,
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLossWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.n_classes = n_classes

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

    def forward(
        self,
        codebook_loss,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train",
        predicted_indices=None,
        scale_fn=None,
    ):
        # VAE optimization
        if optimizer_idx == 0:

            # Reguzlarization loss
            if codebook_loss is None:
                codebook_loss = torch.tensor([0.0]).to(inputs.device)

            # Reconstruction loss
            rec_loss = self.pixel_loss(
                inputs.contiguous(), reconstructions.contiguous()
            )
            nll_loss = torch.mean(rec_loss)

            # Unscaled pixel loss
            nll_loss_unscaled = torch.tensor(0, dtype=torch.float32)
            rec_loss_unscaled = torch.tensor(0, dtype=torch.float32)
            if self.unsc_pixel_weight > 0.0:
                assert (
                    scale_fn is not None
                ), "scale_fn must be provided for unscaled pixel loss."
                rec_loss_unscaled = self.pixel_loss(
                    scale_fn(inputs.contiguous()),
                    scale_fn(reconstructions.contiguous()),
                )
                nll_loss_unscaled = torch.mean(rec_loss_unscaled)

            # Generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1)
                )

            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(
                    nll_loss, g_loss, last_layer=last_layer
                )
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )

            # Calculate total loss
            loss = (
                nll_loss
                + self.unsc_pixel_weight * nll_loss_unscaled
                + d_weight * disc_factor * g_loss
                + self.codebook_weight * codebook_loss.mean()
            )

            # Log the losses
            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/nll_loss_unscaled".format(split): nll_loss_unscaled.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach().to(inputs.device),
                "{}/disc_factor".format(split): torch.tensor(disc_factor).to(
                    inputs.device
                ),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }

            # Measure perplexity and cluster usage
            if predicted_indices is not None:
                assert self.n_classes is not None
                with torch.no_grad():
                    perplexity, cluster_usage = measure_perplexity(
                        predicted_indices, self.n_classes
                    )
                log[f"{split}/perplexity"] = perplexity
                log[f"{split}/cluster_usage"] = cluster_usage
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous().detach(), cond), dim=1)
                )
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1)
                )

            disc_factor = adopt_weight(
                self.disc_factor,
                global_step,
                threshold=self.discriminator_iter_start,
                value=int(self.pretrain_disc),  # if pretraining true, use 1
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log
