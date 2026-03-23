from functools import partial

import torch
import numpy as np
from torchvision.utils import make_grid

import torch
from torch.optim.lr_scheduler import _LRScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianHighPassFilter(nn.Module):
    """
    A high-pass filter that applies a Gaussian blur to an image and returns the residual.
    This implementation is compatible with PyTorch backpropagation.

    Parameters
    ----------
    kernel_size : int
        The size of the Gaussian kernel (must be odd).
    sigma : float
        The standard deviation of the Gaussian kernel.
    """

    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = int(np.ceil(3 * sigma)) * 2 + 1
        self.gaussian_kernel = self._create_gaussian_kernel(self.kernel_size, sigma)

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """
        Create a 2D Gaussian kernel.

        Parameters
        ----------
        kernel_size : int
            The size of the Gaussian kernel (must be odd).
        sigma : float
            The standard deviation of the Gaussian kernel.

        Returns
        -------
        torch.Tensor
            A 2D Gaussian kernel of shape (1, 1, kernel_size, kernel_size).
        """
        # Create a 1D Gaussian kernel
        x = torch.arange(kernel_size) - kernel_size // 2
        gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()

        # Create a 2D Gaussian kernel by outer product
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        gauss_2d = gauss_2d / gauss_2d.sum()

        # Reshape to (1, 1, kernel_size, kernel_size) for convolution
        return gauss_2d.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        """
        Apply the high-pass filter to the input image.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            High-pass filtered image of the same shape as the input.
        """
        # Ensure the Gaussian kernel is on the same device as the input
        kernel = self.gaussian_kernel.to(x.device, dtype=x.dtype)

        # Apply Gaussian blur using depthwise convolution
        blurred = F.conv2d(x, kernel, padding=self.kernel_size // 2, groups=x.shape[1])

        # Compute the residual (high-pass filtered image)
        high_pass = x - blurred

        return high_pass


class LinearInterpolationLR(_LRScheduler):
    """
    Custom PyTorch scheduler that linearly interpolates between two learning rates
    over a specified number of steps.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer for which to schedule the learning rate.
    start_lr : float
        The initial learning rate.
    end_lr : float
        The final learning rate.
    total_steps : int
        The number of steps over which to interpolate.
    last_epoch : int, optional
        The index of the last epoch. Default is -1 (start from scratch).
    """

    """
    Note: This class uses torch.linspace to create a tensor of learning rates.
    To avoid the warning about copying tensors, ensure that the tensor is detached
    and not requiring gradients.
    """

    def __init__(self, optimizer, start_lr, end_lr, total_steps, last_epoch=-1):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.total_steps = total_steps

        self.lrs = (self.end_lr - self.start_lr) * torch.linspace(
            0, 1, total_steps
        ).detach() + self.start_lr
        self.counter = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute the current learning rate based on the step.

        Returns
        -------
        list of float
            The learning rates for each parameter group.
        """
        if self.counter >= self.total_steps:
            return [self.end_lr for _ in self.base_lrs]
        current_lr = self.lrs[self.counter]
        self.counter += 1
        return [current_lr for _ in self.base_lrs]


def vae_log_image_grid(
    x,
    x_recon,
    nrow=2,
    padding=4,
):
    inp = x.cpu()
    rec = x_recon.cpu()
    diff = rec - inp
    # Prepare a 3 * N row image, where each "meta-row" is 3*nrow images tall

    B, C, H, W = inp.shape

    # Concatenate vertically for each sample → (B, C, 3*H, W)
    tall_images = torch.cat([inp, rec, diff], dim=2)

    # Compute number of columns = ceil(B / num_meta_rows)
    # At some point between here and the wandb panel,
    # it seems the notion of rows and columns is
    # flipped, so we need to be careful about the order.
    ncol = B // nrow + B % nrow

    # Now we can pass these tall images to make_grid
    grid = make_grid(
        tall_images,
        nrow=ncol,
        padding=padding,
        normalize=True,
        scale_each=True,
        pad_value=0,
    )

    return grid


def vae_log_z_grid(
    posterior,
    nrow=2,
    padding=4,
):
    mean = posterior.mean.cpu()
    var = posterior.var.cpu()
    logvar = posterior.logvar.cpu()
    # Prepare a 3 * N row image, where each "meta-row" is 3*nrow images tall

    # Get shape
    B, C, H, W = mean.shape

    # Concatenate channels along the height dimension → (B, 1, C*H, W)
    # Also, min-max scale mean and logvar individually
    tall_mean = mean.view(B, 1, C * H, W)
    tall_var = var.view(B, 1, C * H, W)
    tall_logvar = logvar.view(B, 1, C * H, W)
    # Concatenate images along width dimension → (B, 1, C*H, 2*W)
    tall_images = torch.cat([tall_mean, tall_var], dim=3)

    # Compute number of columns = ceil(B / num_meta_rows)
    # At some point between here and the wandb panel,
    # it seems the notion of rows and columns is
    # flipped, so we need to be careful about the order.
    ncol = B // nrow + B % nrow

    # Now we can pass these tall images to make_grid
    grid = make_grid(
        tall_images,
        nrow=ncol,
        padding=padding,
        normalize=True,
        scale_each=True,
        pad_value=0,
    )

    return grid


def VAELoss(
    x,
    recon_x,
    posteriors,
    kl_weight,
):
    # L2 reconstruction loss
    rec_loss = (recon_x - x).pow(2).sum() / x.shape[0]

    # KL loss
    kl_loss = posteriors.kl().sum() / x.shape[0]
    loss = rec_loss + kl_weight * kl_loss
    return loss, rec_loss, kl_loss


def VAELossFunction(kl_weight=1e-6):
    return partial(
        VAELoss,
        kl_weight=kl_weight,
    )


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def kl_G(self, other=None):
        # Similar to kl but over batch
        if other is not None:
            raise NotImplementedError(
                "kl_G is not implemented for DiagonalGaussianDistribution with other"
            )
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            bsize = self.mean.shape[0]
            mean_bar = torch.mean(self.mean.view(bsize, -1), dim=0, keepdim=True)
            z = self.sample().view(bsize, -1)
            var_bar = torch.mean(torch.pow((z - mean_bar), 2), dim=0, keepdim=True)
            return 0.5 * torch.sum(
                torch.pow(mean_bar, 2) + var_bar - 1 - torch.log(var_bar),
                # dim=1,
            )

    def kl_I(self, other=None):
        # Like kl but no mean term
        if other is not None:
            raise NotImplementedError(
                "kl_I is not implemented for DiagonalGaussianDistribution with other"
            )
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            return 0.5 * torch.sum(
                self.var - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )

    def kl_I_delta(self, other=None):
        # Measure difference in kl_I between latent dimensions (i.e. channels)
        if other is not None:
            raise NotImplementedError(
                "kl_delta is not implemented for DiagonalGaussianDistribution with other"
            )
        if self.deterministic:
            return torch.Tensor([0.0])

        else:
            return torch.abs(
                torch.diff(
                    0.5
                    * torch.sum(
                        self.var - 1.0 - self.logvar,
                        dim=[2, 3],
                    ),
                    dim=1,
                )
            )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean
