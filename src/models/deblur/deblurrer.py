from typing import Any, Literal
from copy import deepcopy

import torch
import wandb
import numpy as np
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from torchvision.utils import make_grid

import models.deblur.utils as dbutils
from models.networks.unet import Unet
from models.networks.modules import configModuleBaseLightning
from models.vae.vae_utils import vae_log_image_grid


class ResidualDeblurrer(configModuleBaseLightning):
    def __init__(
        self,
        resid_weight=0.1,
        lr=2e-5,
        blur_kernel_fwhm=4,  # in pixels
        init_channels=128,
        channel_mults=(1, 2, 4, 8),
        image_channels=1,
        norm_groups=32,
        dropout=0,
        num_res_blocks=2,
        attention_levels=3,
        attention_heads=4,
        attention_head_channels=32,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.model = Unet(
            init_channels=init_channels,
            out_channels=2,
            channel_mults=channel_mults,
            image_channels=image_channels,
            norm_groups=norm_groups,
            dropout=dropout,
            num_res_blocks=num_res_blocks,
            attention_levels=attention_levels,
            attention_heads=attention_heads,
            attention_head_channels=attention_head_channels,
        )

        self.lr = lr
        self.resid_weight = resid_weight

        kernel_sigma = blur_kernel_fwhm / 2.355
        kernel_size = 2 * np.ceil(5 * kernel_sigma) + 1
        kernel = np.zeros((int(kernel_size), int(kernel_size)))
        kernel[int(kernel_size) // 2, int(kernel_size) // 2] = 1
        kernel = gaussian_filter(kernel, sigma=kernel_sigma)
        kernel = kernel / np.sum(kernel)
        kernel = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)

        # Initialize blur
        self.blur = nn.Conv2d(
            in_channels=image_channels,
            out_channels=image_channels,
            kernel_size=int(kernel_size),
            padding="same",
            groups=image_channels,
            bias=False,
        )
        # Make parameters not trained
        self.blur.weight.data = kernel
        self.blur.weight.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_channels, height, width).
        """
        x = self.model(x)
        return torch.clamp(x, -1, None)

    def loss(self, x, x_hat):
        """
        Compute the loss between the input and output tensors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).
        x_hat : torch.Tensor
            Output tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            The computed loss value.
        """
        # Rescale the input and output tensors to [0, 1]
        x = (x + 1) / 2
        x_hat = (x_hat + 1) / 2
        model_img = x_hat[:, 0:1, :, :]
        resid = x_hat[:, 1:2, :, :]
        # Blur the output
        model_img_blurred = self.blur(model_img)
        # Compute the L1 loss between the blurred output and the input
        rec_loss = torch.sum(torch.abs(x - (model_img_blurred + resid)) / x.shape[0])
        resid_loss = torch.sum(torch.abs(resid)) / x.shape[0]
        loss = rec_loss + self.resid_weight * resid_loss

        return (loss, rec_loss, resid_loss), model_img_blurred, resid

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Parameters
        ----------
        batch : dict
            A dictionary containing the input data and target labels.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            The loss value for the current batch.
        """
        x = batch

        # Forward pass
        x_hat = self(x)

        # Compute loss
        losses, _, _ = self.loss(x, x_hat)
        loss, rec_loss, resid_loss = losses

        # Log loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Log reconstruction loss
        self.log(
            "train_rec_loss",
            rec_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Log residual loss
        self.log(
            "train_resid_loss",
            resid_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Parameters
        ----------
        batch : dict
            A dictionary containing the input data and target labels.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            The loss value for the current batch.
        """
        x = batch

        # Forward pass
        x_hat = self(x)

        # Compute loss
        losses, model_img_blurred, resid = self.loss(x, x_hat)
        loss, rec_loss, resid_loss = losses

        # Log loss
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        # Log reconstruction loss
        self.log(
            "val_rec_loss",
            rec_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        # Log residual loss
        self.log(
            "val_resid_loss",
            resid_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Log images in the first batch
        if batch_idx == 0:
            grid = dbutils.residual_deblurrer_model_image_grid(
                x, x_hat[:, 0:1, :, :], nrow=4, padding=4
            )
            self.logger.experiment.log(
                {"val_images": wandb.Image(grid, caption="Validation Images", mode="F")}
            )
            grid = dbutils.residual_deblurrer_log_image_grid(
                (x + 1) / 2, model_img_blurred, resid, nrow=4, padding=4
            )
            self.logger.experiment.log(
                {
                    "val_images_blurred": wandb.Image(
                        grid, caption="Validation Images", mode="F"
                    )
                }
            )
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler for the model.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer for the model.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
        )

        return optimizer


class Deblurrer(configModuleBaseLightning):
    def __init__(
        self,
        lr=2e-5,
        blur_kernel_fwhm=4,  # in pixels
        rec_loss="l1",
        smooth_weight=0.1,
        smooth_loss="total_variation_loss",
        init_channels=128,
        channel_mults=(1, 2, 4, 8),
        image_channels=1,
        n_labels=0,
        context_dim=0,
        label_dropout=0,
        context_dropout=0,
        norm_groups=32,
        dropout=0,
        num_res_blocks=2,
        attention_levels=3,
        attention_heads=4,
        attention_head_channels=32,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.model = Unet(
            init_channels=init_channels,
            out_channels=1,
            channel_mults=channel_mults,
            image_channels=image_channels,
            norm_groups=norm_groups,
            dropout=dropout,
            num_res_blocks=num_res_blocks,
            attention_levels=attention_levels,
            attention_heads=attention_heads,
            attention_head_channels=attention_head_channels,
        )

        self.lr = lr
        self.smooth_weight = smooth_weight
        assert rec_loss in ["l1", "l2"], f"Invalid reconstruction loss type: {rec_loss}"
        self.rec_loss_type = rec_loss

        # Assert that smooth_loss describes a function in dbutils
        assert smooth_loss in dir(dbutils), f"Invalid smooth loss type: {smooth_loss}"
        self.smooth_loss_fn = getattr(dbutils, smooth_loss)

        # Initialize blur
        kernel_sigma = blur_kernel_fwhm / 2.355
        kernel_size = 2 * np.ceil(5 * kernel_sigma) + 1
        kernel = np.zeros((int(kernel_size), int(kernel_size)))
        kernel[int(kernel_size) // 2, int(kernel_size) // 2] = 1
        kernel = gaussian_filter(kernel, sigma=kernel_sigma)
        kernel = kernel / np.sum(kernel)
        kernel = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
        self.blur = nn.Conv2d(
            in_channels=image_channels,
            out_channels=image_channels,
            kernel_size=int(kernel_size),
            padding="same",
            groups=image_channels,
            bias=False,
        )
        self.blur.weight.data = kernel
        # Make parameters not trained
        self.blur.weight.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_channels, height, width).
        """
        x = self.model(x)
        return torch.clamp(x, -1, None)

    def loss(self, x, x_hat):
        """
        Compute the loss between the input and output tensors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).
        x_hat : torch.Tensor
            Output tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            The computed loss value.
        """
        # Rescale the input and output tensors to [0, 1]
        x = (x + 1) / 2
        x_hat = (x_hat + 1) / 2
        # Blur the output
        x_hat_blurred = self.blur(x_hat)
        # Compute the L1 loss between the blurred output and the input
        rec_loss = (
            torch.sum(
                (torch.abs if self.rec_loss_type == "l1" else torch.square)(
                    x - x_hat_blurred
                )
            )
            / x.shape[0]
        )
        smooth_loss = 0
        if self.smooth_weight > 0:
            smooth_loss = self.smooth_loss_fn(x_hat_blurred, x_hat) / x.shape[0]

        loss = rec_loss + self.smooth_weight * smooth_loss

        return loss, rec_loss, smooth_loss, x_hat_blurred

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Parameters
        ----------
        batch : dict
            A dictionary containing the input data and target labels.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            The loss value for the current batch.
        """
        x = batch

        # Forward pass
        x_hat = self(x)

        # Compute loss
        loss, rec_loss, smooth_loss, _ = self.loss(x, x_hat)

        # Log loss
        self.log_dict(
            {
                "train_loss": loss,
                "train_rec_loss": rec_loss,
                "train_smooth_loss": smooth_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Parameters
        ----------
        batch : dict
            A dictionary containing the input data and target labels.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            The loss value for the current batch.
        """
        x = batch

        # Forward pass
        x_hat = self(x)

        # Compute loss
        loss, rec_loss, smooth_loss, x_hat_blurred = self.loss(x, x_hat)

        # Log loss
        self.log_dict(
            {
                "val_loss": loss,
                "val_rec_loss": rec_loss,
                "val_smooth_loss": smooth_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Log images in the first batch
        if batch_idx == 0:
            grid = vae_log_image_grid(x, x_hat, nrow=4, padding=4)
            self.logger.experiment.log(
                {"val_images": wandb.Image(grid, caption="Validation Images", mode="F")}
            )
            grid = vae_log_image_grid((x + 1) / 2, x_hat_blurred, nrow=4, padding=4)
            self.logger.experiment.log(
                {
                    "val_images_blurred": wandb.Image(
                        grid, caption="Validation Images", mode="F"
                    )
                }
            )
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler for the model.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer for the model.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
        )

        return optimizer
