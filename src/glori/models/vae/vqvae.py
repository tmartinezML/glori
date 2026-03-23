from typing import Any, Literal
from copy import deepcopy
from functools import partial

import torch
import wandb
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from vqtorch.nn import VectorQuant

from glori.data.trf.scalers import LOFARScaler
import glori.models.vae.vae_utils as vae_utils
from glori.models.vae.vqvae_loss import VQLossWithDiscriminator
from glori.models.vae.quantize import VectorQuantizer
from glori.models.networks.modules import (
    WeightStandardizedConv2d,
    configModuleBaseLightning,
    zero_module,
)
from glori.models.networks.vae_modules import *


class VQVAE(configModuleBaseLightning):
    """
    Variational Autoencoder (VAE) class.

    This class implements a VAE architecture with an encoder and decoder.
    The encoder maps input images to a latent space, and the decoder reconstructs
    images from the latent space.

    Parameters
    ----------
    init_channels : int
        Number of initial channels in the encoder/decoder.
    z_channels : int
        Number of channels in the latent space.
    image_channels : int
        Number of channels in the input images.
    channel_mults : tuple, optional
        Multipliers for the number of channels at each level (default: (1, 2, 4)).
    norm_groups : int, optional
        Number of groups for normalization (default: 32).
    dropout : float, optional
        Dropout rate (default: 0.0).
    num_res_blocks : int, optional
        Number of residual blocks at each level (default: 2).
    attention_levels : int, optional
        Number of levels where attention is applied (default: 0).
    attention_heads : int, optional
        Number of attention heads (default: 4).
    attention_head_channels : int, optional
        Number of channels per attention head (default: 32).
    """

    def __init__(
        self,
        *,
        init_channels,
        z_channels,
        n_embed,
        emb_dim,
        image_channels,
        channel_mults=(
            1,
            2,
            4,
        ),
        norm_groups=32,
        dropout=0.0,
        num_res_blocks=2,
        attention_levels=0,
        attention_heads=4,
        attention_head_channels=32,
        lr=1e-6,
        lr_disc=1e-4,
        pixel_loss="l1",
        pixelloss_weight=1.0,
        unsc_pixel_weight=0.0,
        codebook_weight=1,
        codebook_start=0,
        vq_kwargs={},
        disc_loss="hinge",
        disc_weight=0.5,
        disc_iter_start=0,
        pretrain_disc=False,
        disc_update_rate=1,
        dynamic_freezing=False,
        disc_num_layers=3,
        disc_norm="batch",
        train_mode: Literal[
            "standard",
            "discriminator",
        ] = "standard",
        overfit_batch=False,
        scaler: str = None,
    ):
        # Initialize Architecture
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(
            init_channels,
            z_channels,
            image_channels,
            channel_mults=channel_mults,
            norm_groups=norm_groups,
            dropout=dropout,
            num_res_blocks=num_res_blocks,
            attention_levels=attention_levels,
            attention_heads=attention_heads,
            attention_head_channels=attention_head_channels,
            variational=False,  # VQVAE does not use variational encoding
        )
        self.decoder = Decoder(
            init_channels,
            z_channels,
            image_channels,
            channel_mults=channel_mults[::-1],
            norm_groups=norm_groups,
            dropout=dropout,
            num_res_blocks=num_res_blocks,
            attention_levels=attention_levels,
            attention_heads=attention_heads,
            attention_head_channels=attention_head_channels,
        )

        self.z_channels = z_channels
        self.emb_dim = emb_dim
        self.n_embed = n_embed
        """
        self.quantize = VectorQuantizer(
            n_embed,
            emb_dim,
            beta=vq_beta,
        )
        """
        default_kw = dict(
            beta=0.98,
            kmeans_init=True,
            affine_lr=10,
            sync_nu=0.2,
            replace_freq=20,
        )
        default_kw.update(vq_kwargs)
        self.quantize = VectorQuant(
            feature_size=emb_dim,
            num_codes=n_embed,
            dim=1,  # Quantization dimension
            **default_kw,
        )
        self.quant_conv = WeightStandardizedConv2d(z_channels, emb_dim, 1)
        self.post_quant_conv = WeightStandardizedConv2d(emb_dim, z_channels, 1)

        # Initialize loss function and training parameters
        self.lr = lr
        self.lr_disc = lr_disc
        # self.lr_min = lr_min
        # self.lr_max = lr_max
        self.codebook_weight = codebook_weight
        self.codebook_start = codebook_start
        # self.use_scheduler = use_scheduler
        # self.scheduler = scheduler
        self.disc_iter_start = disc_iter_start
        self.loss = VQLossWithDiscriminator(
            self.disc_iter_start,
            codebook_weight=codebook_weight,
            pixelloss_weight=pixelloss_weight,
            unsc_pixel_weight=unsc_pixel_weight,
            disc_in_channels=image_channels,
            disc_loss=disc_loss,
            disc_weight=disc_weight,
            pretrain_disc=pretrain_disc,
            disc_num_layers=disc_num_layers,
            disc_norm=disc_norm,
            pixel_loss=pixel_loss,
            n_classes=n_embed,
        )

        if scaler is not None:
            self.scaler = LOFARScaler.load(scaler)
        self.automatic_optimization = False
        self.train_mode = train_mode
        self.overfit_batch = overfit_batch
        self.dissc_update_rate = disc_update_rate
        self.dynamic_freezing = dynamic_freezing
        self.vae_freeze = False

    def encode(self, x: torch.Tensor):
        """
        Encode input images to latent space.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_channels, height, width).

        Returns
        -------
        torch.Tensor
            Latent representation of shape (batch_size, z_channels, height/8, width/8).
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, info = self.quantize(h)
        emb_loss = info["loss"]
        return quant, emb_loss, info

    def encode_to_prequant(self, x: torch.Tensor):
        """
        Encode input images to latent space without quantization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_channels, height, width).

        Returns
        -------
        torch.Tensor
            Latent representation of shape (batch_size, z_channels, height/8, width/8).

        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        if self.quantize.norm_before_grouping:
            h = self.quantize.norm_layer(h)
        return h

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruct images.

        Parameters
        ----------
        quant : torch.Tensor
            Latent representation of shape (batch_size, z_channels, height/8, width/8).

        Returns
        -------
        torch.Tensor
            Reconstructed images of shape (batch_size, input_channels, height, width).
        """
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latent codes to reconstruct images.

        Parameters
        ----------
        code_b : torch.Tensor
            Quantized latent codes of shape (batch_size, n_embed, height/8, width/8).

        Returns
        -------
        torch.Tensor
            Reconstructed images of shape (batch_size, input_channels, height, width).
        """
        code_b = self.quantize.to_canonical_group_format(code_b, self.quantize.groups)
        if not self.quantize.norm_before_grouping:
            code_b = self.quantize.norm_layer(code_b)
        quant_b, _, _ = self.quantize.quantize(self.quantize.codebook.weight, code_b)
        quant_b = self.quantize.to_original_format(quant_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, x: torch.Tensor, return_pred_indices=False) -> torch.Tensor:
        """
        Forward pass through the VQVAE.
        Encodes input images to latent space and decodes them back to reconstruct images.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_channels, height, width).
        return_pred_indices : bool, optional
            If True, returns the indices of the quantized latent codes (default: False).

        Returns
        -------
        torch.Tensor
            Reconstructed images of shape (batch_size, input_channels, height, width).
        torch.Tensor
            Difference between input and reconstructed images.
        tuple
            Indices of the quantized latent codes if `return_pred_indices` is True.
        """
        # quant, diff, (_, _, ind) = self.encode(x)
        quant, diff, info = self.encode(x)
        ind = info["q"]
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Training step for the VAE with averaged loss and log_dict logging over gradient accumulation steps.
        """
        # Get the number of batches to accumulate gradients
        if not hasattr(self, "train_step"):
            self.train_step = int(deepcopy(self.global_step) / 2)
        self.train_step += 1
        self.log("train_step", self.train_step, prog_bar=True, logger=True)
        self.log("trainer/train_step", self.train_step, prog_bar=True, logger=True)
        self.log("trainer/batch_idx", batch_idx, prog_bar=True, logger=True)

        # Get batch and pass through network
        x = batch[0]  # Batch is a tuple

        # Get the optimizers
        opt_ae, opt_disc = self.optimizers()

        # Autoencoder warmup
        if self.train_step < self.codebook_start:
            # Pass without quantization
            h = self.encode_to_prequant(x)
            xrec = self.decode(h)
            nll_loss = self.loss.pixel_loss(x.contiguous(), xrec.contiguous())
            nll_loss = nll_loss.mean()
            self.log(
                "train/warmup_nll_loss",
                nll_loss.detach().mean(),
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=False,
            )

            # Optimization
            if self.train_mode == "standard":
                # Optimize AE
                opt_ae.zero_grad()
                self.manual_backward(nll_loss)
                opt_ae.step()
            return

        # Pass through network
        xrec, qloss, ind = self(x, return_pred_indices=True)

        # Autoencoder optimization
        if (
            (self.train_step < self.loss.discriminator_iter_start)
            or (self.train_step + 1) % self.dissc_update_rate == 0
            and not self.vae_freeze
        ):
            aeloss, log_dict_ae = self.loss(
                qloss,
                x,
                xrec,
                0,  # optimizer idx
                self.train_step,
                last_layer=self.get_last_layer(),
                split="train",
                scale_fn=(
                    partial(self.scaler.inverse_scale, use_torch=True)
                    if self.scaler is not None
                    else None
                ),
                predicted_indices=ind,
            )

            # Perform backward pass
            if self.train_mode == "standard":
                # Optimize AE
                opt_ae.zero_grad()
                self.manual_backward(aeloss)
                opt_ae.step()

            # Log the loss
            self.log(
                "aeloss",
                aeloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=False,
            )
            self.log_dict(
                log_dict_ae,
                on_step=True,
                prog_bar=False,
                on_epoch=False,
                logger=True,
            )

            # If generator loss is < 0, freeze the VAE
            if (
                self.train_mode == "standard"
                and self.dynamic_freezing
                and log_dict_ae["train/g_loss"] < 0
            ):
                self.vae_freeze = True

        # Discriminator optimization
        discloss, log_dict_disc = self.loss(
            qloss,
            x,
            xrec,
            1,  # optimizer idx
            self.train_step,
            last_layer=self.get_last_layer(),
            split="train",
        )

        # Log discriminator weights and graidents
        total_grad_norm = torch.norm(
            torch.stack(
                [
                    (
                        p.grad.norm(2)
                        if p.grad is not None
                        else torch.tensor(0, dtype=p.dtype)
                    )
                    for p in self.loss.discriminator.parameters()
                ]
            ),
            2,
        )
        total_weight_norm = torch.norm(
            torch.stack([p.data.norm(2) for p in self.loss.discriminator.parameters()]),
            2,
        )
        self.log_dict(
            {
                "discriminator/weight_norm": total_weight_norm,
                "discriminator/grad_norm": total_grad_norm,
                "discriminator/train_step": self.train_step,
            },
            logger=True,
            on_step=True,
            prog_bar=False,
            on_epoch=False,
        )

        # Perform optimizer step for discriminator
        if self.train_mode in ["standard", "discriminator"]:
            opt_disc.zero_grad()
            self.manual_backward(discloss)
            opt_disc.step()

        # Log the loss
        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True)
        self.log_dict(
            log_dict_disc,
            logger=True,
            on_step=True,
            prog_bar=False,
            on_epoch=False,
        )

        # Log status of vae freeze
        self.log(
            "train/vae_freeze",
            self.vae_freeze,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        # If the fake logits are back to negative, unfreeze the vae
        if (
            self.train_mode == "standard"
            and self.dynamic_freezing
            and log_dict_disc["train/logits_fake"] < 0
        ):
            self.vae_freeze = False

        # Log images and latents if overfit batch
        if self.overfit_batch and (self.train_step % self.train_img_log_interval == 0):
            img_grid = vae_utils.vae_log_image_grid(x, xrec)

            # Log images to wandb
            self.logger.experiment.log(
                {
                    "train_batch": wandb.Image(
                        img_grid,
                        mode="F",
                        caption="Single Element, top to bottom: input, reconstruction, difference",
                    ),
                }
            )

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Validation step for the VAE.

        Parameters
        ----------
        batch : Any
            Input batch of data.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Loss value for the validation step.
        """
        x = batch[0]  # Batch is a tuple

        xrec, qloss, ind = self(x, return_pred_indices=True)

        if not hasattr(self, "train_step"):
            self.train_step = int(deepcopy(self.global_step) / 2)

        if self.train_step < self.codebook_start:
            # Pass without quantization
            h = self.encode_to_prequant(x)
            xrec = self.decode(h)
            nll_loss = self.loss.pixel_loss(x.contiguous(), xrec.contiguous())
            nll_loss = nll_loss.mean()
            self.log(
                "val/nll_loss",
                nll_loss.detach().mean(),
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
            )
            if batch_idx == 0:
                img_grid = vae_utils.vae_log_image_grid(x, xrec)

            # Log the loss for the optimizer
            self.log(
                "val_loss",
                nll_loss.detach().mean(),
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            return

        aeloss, log_dict_ae = self.loss(
            qloss,
            x,
            xrec,
            0,
            self.train_step,
            last_layer=self.get_last_layer(),
            split="val",
            scale_fn=(
                partial(self.scaler.inverse_scale, use_torch=True)
                if self.scaler is not None
                else None
            ),
        )

        discloss, log_dict_disc = self.loss(
            qloss,
            x,
            xrec,
            1,
            self.train_step,
            last_layer=self.get_last_layer(),
            split="val",
        )
        self.log(
            "val/train_step",
            self.train_step,
            prog_bar=False,
            on_epoch=True,
            sync_dist=True,
            logger=True,
        )
        self.log_dict(
            log_dict_ae, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True
        )
        self.log_dict(
            log_dict_disc, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True
        )

        # Log images and latents in first batch
        if batch_idx == 0:
            img_grid = vae_utils.vae_log_image_grid(x, xrec)

            # Log pixel distributions to wandb
            fig, ax = plt.subplots(
                1,
                1,
            )
            ax.hist(
                x.flatten().cpu().to(torch.float32).numpy(),
                bins=256,
                density=False,
                label="input",
                histtype="step",
            )
            ax.hist(
                xrec.flatten().cpu().to(torch.float32).numpy(),
                bins=256,
                density=False,
                label="reconstructed",
                histtype="step",
            )
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Counts")
            ax.set_yscale("log")

            self.logger.experiment.log(
                {
                    "val/pixel_distributions": wandb.Image(fig),
                }
            )
            plt.close(fig)

            # Log pixel scatter plot to wandb
            fig, axs = plt.subplots(
                (n := int(np.sqrt(x.shape[0]))),
                n + np.ceil((x.shape[0] - n**2) / n).astype(int),
                tight_layout=True,
            )
            for i, ax in enumerate(axs.flat):
                if i >= x.shape[0]:
                    ax.axis("off")
                    continue
                ax.scatter(
                    (x_unsc := self.scaler.inverse_scale(x[i].cpu().flatten())),
                    self.scaler.inverse_scale(xrec[i].cpu().flatten()),
                    alpha=0.6,
                    s=0.5,
                )
                ax.axline(
                    [
                        x_unsc.mean(),
                    ]
                    * 2,
                    slope=1,
                    ls="--",
                    color="red",
                    alpha=0.3,
                )
                ax.grid(alpha=0.3)
                # Set equal limits for x and y
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                minlim, maxlim = min(xmin, ymin), max(xmax, ymax)
                ax.set_xlim(minlim, maxlim)
                ax.set_ylim(minlim, maxlim)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            self.logger.experiment.log(
                {
                    "val/pixel_scatter": wandb.Image(fig),
                }
            )
            plt.close(fig)

            # Log images to wandb
            self.logger.experiment.log(
                {
                    "val_batch": wandb.Image(
                        img_grid,
                        mode="F",
                        caption="Single Element, top to bottom: input, reconstruction, difference",
                    ),
                }
            )

        # Log the loss for the optimizer
        self.log(
            "val_loss",
            aeloss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return self.log_dict

    def configure_optimizers(self):
        """
        Configure the optimizer for the VAE.

        Returns
        -------
        torch.optim.Optimizer
            Optimizer for training the VAE.
        """

        optimizer_ae = torch.optim.AdamW(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantize.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=self.lr,
        )
        optimizer_disc = torch.optim.AdamW(
            self.loss.discriminator.parameters(),
            lr=self.lr_disc,
        )

        return [optimizer_ae, optimizer_disc], []

    def get_last_layer(self):
        """
        Get the last layer of the VAE.

        Returns
        -------
        torch.nn.Module
            Last layer of the VAE.
        """
        return self.decoder.out[-1].weight

    def load_discriminator_state(self, path: str):
        """
        Load VAE weights from a checkpoint file.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location="cpu")

        # Load the discriminator weights
        disc_weights = {
            k: v for k, v in checkpoint["state_dict"].items() if "discriminator" in k
        }
        self.loss.discriminator.load_state_dict(disc_weights)

        # Load optimizer state
        optimizer_state = checkpoint["optimizer_states"][1]
        self.optimizers()[1].load_state_dict(optimizer_state)


class VQVAEInterface(VQVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if force_not_quantize:
            quant = h
        else:
            quant, _ = self.quantize(h)
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
