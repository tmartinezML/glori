import torch
from torchvision.utils import make_grid


def halo_loss(
    img: torch.Tensor, orig_img: torch.Tensor, eps: float = 5e-4, reduction: str = "sum"
) -> torch.Tensor:

    halo_map = img / (orig_img**2 + eps)

    if reduction == "mean":
        return halo_map.mean()
    elif reduction == "sum":
        return halo_map.sum()
    else:
        return halo_map


def adaptive_fft_loss(
    img: torch.Tensor, orig_img: torch.Tensor, eps: float = 1e-3, reduction: str = "sum"
) -> torch.Tensor:
    """
    Adaptive frequency-domain smoothness loss.
    1. FFT -> apply frequency weight (HPF-like) -> IFFT.
    2. Penalize magnitude more in low-brightness areas.

    Args:
        img: Tensor (N, C, H, W), expected in [0,1] or [0,255]
        eps: small constant for stability
        reduction: 'mean' | 'sum' | 'none'

    Returns:
        Scalar tensor if reduction != 'none'
    """
    N, C, H, W = img.shape

    # Brightness map (grayscale proxy)
    brightness = img.mean(dim=1, keepdim=True)  # (N,1,H,W)

    # ---- FFT ----
    fft = torch.fft.fft2(img, dim=(-2, -1))
    fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))

    # Frequency weighting mask (radial ramp)
    yy, xx = torch.meshgrid(
        torch.linspace(-0.5, 0.5, H, device=img.device),
        torch.linspace(-0.5, 0.5, W, device=img.device),
        indexing="ij",
    )
    freq_radius = torch.sqrt(xx**2 + yy**2)  # 0 at center, max ~0.7 at corners
    freq_weight = freq_radius.unsqueeze(0).unsqueeze(0) ** 0.5  # shape (1,1,H,W)
    freq_weight /= freq_weight.max()

    # Apply high-pass weighting
    fft_filtered = fft_shift * freq_weight

    # Undo shift & inverse FFT back to spatial domain
    fft_unshift = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
    high_freq_map = torch.fft.ifft2(fft_unshift, dim=(-2, -1)).real

    # Penalize high frequencies more in dark areas
    penalty_map = torch.abs(high_freq_map) * (1.0 / (brightness + eps))

    # Reduce
    if reduction == "mean":
        return penalty_map.mean()
    elif reduction == "sum":
        return penalty_map.sum()
    else:
        return penalty_map


def spectral_loss(img: torch.Tensor, _, reduction: str = "sum") -> torch.Tensor:
    """
    Penalizes high-frequency energy of an image batch using FFT.

    Args:
        img: Tensor of shape (N, C, H, W)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar spectral loss unless reduction='none'
    """
    # Compute Fourier transform over height and width
    fft = torch.fft.fft2(img, dim=(-2, -1))
    fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))  # center low frequencies

    # Compute magnitude spectrum
    mag = torch.abs(fft_shift)

    # Create frequency weighting mask that grows with distance from center
    N, C, H, W = img.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-0.5, 0.5, H, device=img.device),
        torch.linspace(-0.5, 0.5, W, device=img.device),
        indexing="ij",
    )
    freq_radius = torch.sqrt(xx**2 + yy**2)  # 0 at center, ~0.7 at corners
    freq_weight = freq_radius.unsqueeze(0).unsqueeze(0)  # shape (1,1,H,W)

    # Apply weight to magnitudes
    spec_energy = mag * freq_weight

    # Compute loss
    if reduction == "mean":
        return spec_energy.mean()
    elif reduction == "sum":
        return spec_energy.sum()
    else:
        return spec_energy


def total_variation_loss(img: torch.Tensor, _, reduction: str = "sum") -> torch.Tensor:
    """
    Computes isotropic total variation loss for a batch of images.

    Args:
        img: Tensor of shape (N, C, H, W) with pixel values.
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar tensor with the TV loss (unless reduction='none').
    """
    # Differences between neighboring pixels in horizontal and vertical directions
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]

    tv = torch.abs(dx).flatten(-2, -1) + torch.abs(dy).flatten(-2, -1)

    if reduction == "mean":
        return tv.mean()
    elif reduction == "sum":
        return tv.sum()
    else:  # 'none'
        return tv


def residual_deblurrer_model_image_grid(
    x,
    model_img,
    nrow=2,
    padding=4,
):

    inp = x.cpu()
    model_img = model_img.cpu()

    # Prepare a 3 * N row image, where each "meta-row" is 3*nrow images tall
    B, C, H, W = inp.shape

    # Concatenate for each sample → (B, C, 2*H, W)
    tall_images = torch.cat([inp, model_img], dim=2)

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


def residual_deblurrer_log_image_grid(
    x,
    model_blr,
    resid,
    nrow=2,
    padding=4,
):
    inp = x.cpu()
    model_blr = model_blr.cpu()
    resid = resid.cpu()
    diff = model_blr + resid - inp

    # Prepare a 3 * N row image, where each "meta-row" is 3*nrow images tall
    B, C, H, W = inp.shape

    # Concatenate for each sample → (B, C, 2*H, 2*W)
    left_col = torch.cat([inp, diff], dim=2)
    right_col = torch.cat([model_blr, resid], dim=2)
    tall_images = torch.cat([left_col, right_col], dim=3)

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
