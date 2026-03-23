from typing import Any, Literal
from copy import deepcopy
from functools import partial

import torch
import wandb
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from data.trf.scalers import LOFARScaler
import models.vae.vae_utils as vae_utils
from models.vae.vae_loss import L1WithDiscriminator
from models.networks.modules import (
    WeightStandardizedConv2d,
    configModuleBaseLightning,
    zero_module,
)
from models.networks.vae_modules import *


class VAE(configModuleBaseLightning):
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
        pixelloss_weight=1.0,
        unscaled_pixelloss_weight=0.0,
        reg_loss="kl",
        kl_weight=1e-6,
        kl_I_weight=1.0,
        disc_loss="hinge",
        disc_weight=0.5,
        use_adapt_weight=True,
        disc_iter_start=0,
        disc_ramp_steps=0,
        pretrain_disc=False,
        disc_update_rate=1,
        dynamic_freezing=False,
        disc_num_layers=3,
        disc_norm="batch",
        disc_hpf_sigma=0.0,
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
        self.emb_conv = WeightStandardizedConv2d(2 * z_channels, 2 * z_channels, 1)
        self.post_emb_conv = WeightStandardizedConv2d(z_channels, z_channels, 1)

        # Initialize loss function and training parameters
        self.lr = lr
        self.lr_disc = lr_disc
        # self.lr_min = lr_min
        # self.lr_max = lr_max
        self.kl_weight = kl_weight
        # self.use_scheduler = use_scheduler
        # self.scheduler = scheduler
        self.disc_iter_start = disc_iter_start
        self.loss = L1WithDiscriminator(
            self.disc_iter_start,
            disc_ramp_steps=disc_ramp_steps,
            kl_weight=self.kl_weight,
            kl_I_weight=kl_I_weight,
            pixelloss_weight=pixelloss_weight,
            unscaled_pixelloss_weight=unscaled_pixelloss_weight,
            disc_in_channels=image_channels,
            reg_loss=reg_loss,
            disc_loss=disc_loss,
            disc_weight=disc_weight,
            pretrain_disc=pretrain_disc,
            hpf_sigma=disc_hpf_sigma,
            disc_num_layers=disc_num_layers,
            disc_norm=disc_norm,
            use_adapt_weight=use_adapt_weight,
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
        z = self.encoder(x)
        moments = self.emb_conv(z)
        # Clamp for numerical stability
        # moments = torch.clamp(moments, -255.0, 255.0)
        posterior = vae_utils.DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruct images.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation of shape (batch_size, z_channels, height/8, width/8).

        Returns
        -------
        torch.Tensor
            Reconstructed images of shape (batch_size, input_channels, height, width).
        """
        z = self.post_emb_conv(z)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

    def forward(self, x: torch.Tensor, sample_posterior=True) -> torch.Tensor:
        """
        Forward pass through the VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, input_channels, height, width).
        """
        # Encode input
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        if not z.dtype == self.dtype:
            z = z.to(self.dtype)
        dec = self.decode(z)
        return dec, posterior

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
        x_reconstructed, posterior = self(x)

        # Get the optimizers
        opt_ae, opt_disc = self.optimizers()

        # Autoencoder optimization
        if (
            (self.train_step < self.loss.discriminator_iter_start)
            or (self.train_step + 1) % self.dissc_update_rate == 0
            and not self.vae_freeze
        ):
            aeloss, log_dict_ae = self.loss(
                x,
                x_reconstructed,
                posterior,
                0,  # optimizer idx
                self.train_step,
                last_layer=self.get_last_layer(),
                split="train",
                scale_fn=(
                    partial(self.scaler.inverse_scale, use_torch=True)
                    if self.scaler is not None
                    else None
                ),
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
            x,
            x_reconstructed,
            posterior,
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
            img_grid = vae_utils.vae_log_image_grid(x, x_reconstructed)

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

            z_grid = vae_utils.vae_log_z_grid(posterior)
            # Log latents to wandb
            self.logger.experiment.log(
                {
                    "train_latents": wandb.Image(
                        z_grid,
                        mode="F",
                        caption="Latent space representation. Single element: Rows represent latent channels, left column: mean, right column: logvar",
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
        x_reconstructed, posterior = self(x)

        if not hasattr(self, "train_step"):
            self.train_step = int(deepcopy(self.global_step) / 2)

        aeloss, log_dict_ae = self.loss(
            x,
            x_reconstructed,
            posterior,
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
            x,
            x_reconstructed,
            posterior,
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
            img_grid = vae_utils.vae_log_image_grid(x, x_reconstructed)

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
                x_reconstructed.flatten().cpu().to(torch.float32).numpy(),
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

            z_grid = vae_utils.vae_log_z_grid(posterior)
            # Log latents to wandb
            self.logger.experiment.log(
                {
                    "val_latents": wandb.Image(
                        z_grid,
                        mode="F",
                        caption="Latent space representation. Single element: Rows represent latent channels, left column: mean, right column: logvar",
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
            + list(self.emb_conv.parameters())
            + list(self.post_emb_conv.parameters()),
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
