from functools import partial

import numpy as np
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF


import random
from data.trf.scalers import ContextScaler


def random_side_masking(img, f_center, p=[0.1, 0.1, 0.01]):
    h, w = img.shape[-2:]
    center_size_h = h // f_center
    center_size_w = w // f_center
    top = (h - center_size_h) // 2
    left = (w - center_size_w) // 2
    if img.ndim == 3:
        img[:, top : top + center_size_h, left : left + center_size_w] *= 0
        return img

    elif img.ndim == 4:
        img[:, :, top : top + center_size_h, left : left + center_size_w] *= 0
        return img

    raise ValueError(f"Expected 3D or 4D tensor, got {len(img.shape)}D")


def zero_center(img, f_center):
    h, w = img.shape[-2:]
    center_size_h = h // f_center
    center_size_w = w // f_center
    top = (h - center_size_h) // 2
    left = (w - center_size_w) // 2
    if img.ndim == 3:
        img[:, top : top + center_size_h, left : left + center_size_w] *= 0
        return img

    elif img.ndim == 4:
        img[:, :, top : top + center_size_h, left : left + center_size_w] *= 0
        return img

    raise ValueError(f"Expected 3D or 4D tensor, got {len(img.shape)}D")


def random_rotate_90(img):
    return TF.rotate(img, random.choice([0, 90, 180, 270]))


def max_scale_batch(batch):
    match batch:
        case torch.Tensor():
            return batch / batch.amax(dim=(-1, -2), keepdim=True)
        case np.ndarray():
            return batch / batch.max(axis=(-1, -2), keepdims=True)
        case _:
            raise TypeError(f"Unsupported type: {type(batch)}")


def minmax_scale_batch(batch):
    match batch:
        case torch.Tensor():
            mx = batch.amax(dim=(-1, -2), keepdim=True)
            mn = batch.amin(dim=(-1, -2), keepdim=True)
        case np.ndarray():
            mx = batch.max(axis=(-1, -2), keepdims=True)
            mn = batch.min(axis=(-1, -2), keepdims=True)
        case _:
            raise TypeError(f"Unsupported type: {type(batch)}")

    return (batch - mn) / (mx - mn)


def train_scale(img):
    return img * 2 - 1


def train_scale_present(transform):
    # Check if minmax_scale is part of the composed transform.
    # Used for automatically setting vmin in plotting function.
    return any(
        isinstance(t, T.Lambda) and t.lambd == train_scale for t in transform.transforms
    )


def minmax_scale_masked(img, mask):
    """
    Scale the image to [0, 1] using the min and max values of the masked region.
    Everything outsitde is set to 0.
    """
    mx = img[mask].max()
    mn = img[mask].min()
    if mx == mn:
        return torch.zeros_like(img)
    return ((img - mn) / (mx - mn)) * mask


def minmax_scale(img):
    if img.max() == img.min():
        return torch.zeros_like(img)

    return (img - img.min()) / (img.max() - img.min())


def add_channel_dim(img):
    if img.ndim == 2:
        return img[None, :, :]
    elif img.ndim == 3:
        return img
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {len(img.shape)}D")


def single_channel(img):

    if len(img.shape) == 3:
        return img[:1, :, :]

    elif len(img.shape) == 2:
        return img.unsqueeze(0) if type(img) == torch.Tensor else img[None, :, :]


def stack_quadrants(image):
    """
    Stack the four quadrants of the input image along channel dimension.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor of shape (c, h, w) or (b, c, h, w).

    Returns
    -------
    torch.Tensor
        Stacked image tensor of shape (c * 4, h // 2, w // 2) or (b, c * 4, h // 2, w // 2).

    Raises
    ------
    ValueError
        If the input image tensor has an unsupported shape.

    """
    # Handle both single images (c, h, w) and batches (b, c, h, w)
    if len(image.shape) == 3:
        # Single image case: (c, h, w)
        c, h, w = image.shape
        out = (
            image.view(c, 2, h // 2, 2, w // 2)  # (c, 2, h//2, 2, w//2)
            .permute(0, 1, 3, 2, 4)  # (c, 2, 2, h//2, w//2)
            .reshape(c * 4, h // 2, w // 2)  # (c * 4, h//2, w//2)
        )
    elif len(image.shape) == 4:
        # Batch case: (b, c, h, w)
        b, c, h, w = image.shape
        out = (
            image.view(b, c, 2, h // 2, 2, w // 2)  # (b, c, 2, h//2, 2, w//2)
            .permute(0, 1, 2, 4, 3, 5)  # (b, c, 2, 2, h//2, w//2)
            .reshape(b, c * 4, h // 2, w // 2)  # (b, c * 4, h//2, w//2)
        )
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {len(image.shape)}D")

    return out


def unstack_quadrants(image):
    """
    Unstack the four quadrants of the input image, stacked along channel
    dimension, and rearrange them as a 2x2 grid into a single-channel image.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor of shape (c, h, w) or (b, c, h, w).

    Returns
    -------
    torch.Tensor
        Unstacked image tensor of shape (c // 4, 2, 2, h, w) or (b, c // 4, 2, 2, h, w).

    Raises
    ------
    ValueError
        If the input image tensor has an unsupported shape.
    """
    # Handle both single stacked encodings (c, h, w) and batches (b, c, h, w)
    if len(image.shape) == 3:
        # Single encoding case: (c, h, w)
        c, h, w = image.shape
        out = (
            image.view(c // 4, 2, 2, h, w)  # Reshape to (c//4, 2, 2, h, w)
            .permute(0, 1, 3, 2, 4)  # Rearrange dimensions: (c//4, 2, h, 2, w)
            .reshape(c // 4, 2 * h, 2 * w)  # Flatten to final shape
        )
    elif len(image.shape) == 4:
        # Batch case: (b, c, h, w)
        b, c, h, w = image.shape
        out = (
            image.view(b, c // 4, 2, 2, h, w)  # Reshape to (b, c//4, 2, 2, h, w)
            .permute(0, 1, 2, 4, 3, 5)  # Rearrange dimensions: (b, c//4, 2, h, 2, w)
            .reshape(b, c // 4, 2 * h, 2 * w)  # Flatten to final shape
        )
    else:
        raise ValueError(
            f"Expected 3D or 4D tensor with shape (c,h,w) or (b,c,h,w), got {len(image.shape)}D"
        )

    return out


def unpack_quadrants(image):
    """
    Unpack the four quadrants of the input image, stacked along additional
    dimension, and rearrange them as a 2x2 grid into a single-channel image.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor of shape (q, c, h, w) or (b, q, c, h, w).

    Returns
    -------
    torch.Tensor
        Unpacked image tensor of shape (c, h * 2, w * 2) or (b, c, h * 2, w * 2).

    Raises
    ------
    ValueError
        If the input image tensor has an unsupported shape.

    """
    # Handle both single stacked encodings (q, c, h, w) and batches (b, q, c, h, w)
    if len(image.shape) == 4:
        # Single encoding case: (q, c, h, w)
        q, c, h, w = image.shape
        out = (
            image.contiguous()
            .view(q // 2, q // 2, c, h, w)  # Split q=4 into 2x2 grid
            .permute(2, 0, 3, 1, 4)  # Rearrange dimensions: (c, q // 2, h, q // 2, w)
            .reshape(c, h * 2, w * 2)  # Flatten to final shape
        )
    elif len(image.shape) == 5:
        # Batch case: (b, q, c, h, w)
        b, q, c, h, w = image.shape
        out = (
            image.contiguous()
            .view(b, q // 2, q // 2, c, h, w)  # Split q=4 into 2x2 grid
            .permute(
                0, 3, 1, 4, 2, 5
            )  # Rearrange dimensions: (b, c, q // 2, h, q // 2, w)
            .reshape(b, c, h * 2, w * 2)  # Flatten to final shape
        )
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {len(image.shape)}D")

    return out


def pack_quadrants(image):
    """
    Pack the input image into four quadrants, stacked along extra dimension.


    """
    h, w = image.shape[-2:]

    if len(image.shape) == 2:
        out = (
            image.reshape(2, h // 2, 2, w // 2)
            .permute(0, 2, 1, 3)
            .reshape(4, h // 2, w // 2)
            .unsqueeze(1)
        )

    if len(image.shape) == 3:
        # Single image case: (c, h, w)
        c, h, w = image.shape
        out = (
            image.reshape(c, 2, h // 2, 2, w // 2)
            .permute(0, 1, 3, 2, 4)
            .reshape(c, 4, h // 2, w // 2)
            .permute(1, 0, 2, 3)  # Move channels to the 2nd dimension
        )

    if len(image.shape) == 4:
        # Batch case: (b, c, h, w)
        b, c, h, w = image.shape
        out = (
            image.reshape(b, c, 2, h // 2, 2, w // 2)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(b, c, 4, h // 2, w // 2)
            .permute(0, 2, 1, 3, 4)  # Move channels to the 2nd dimension
            .reshape(b * 4, c, h // 2, w // 2)  # Flatten batch and quadrants
        )

    return out


def half_size_random_crop(image):
    _, h, w = image.shape
    return T.RandomCrop((h // 2, w // 2))(image)


def catalog_context_pos_rescale(x):
    if len(x.shape) == 3:
        x[0] = x[0] * 2 - 1
    elif len(x.shape) == 4:
        x[:, 0] = x[:, 0] * 2 - 1
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {len(x.shape)}D")
    return x


def scale_fn(channel, mask, fn):
    channel[mask] = fn(channel[mask])
    return channel


def catalog_context_value_scale(x, scale_fns):
    assert (
        len(scale_fns) == 3
    ), f"Scale functions must be a tuple of length 3, got {len(scale_fns)}: {scale_fns}"

    if len(x.shape) == 3:
        mask = x[0] > 0  # Precompute mask
        [x[i + 1].masked_scatter_(mask, scale_fns[i](x[i + 1][mask])) for i in range(3)]

    elif len(x.shape) == 4:
        mask = x[:, 0] > 0  # Precompute mask
        [
            x[:, i + 1].masked_scatter_(mask, scale_fns[i](x[:, i + 1][mask]))
            for i in range(3)
        ]

    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {len(x.shape)}D")

    return x


def make_catalog_context_value_scale(
    scalers=[
        "ctxt_scaler_ftot",
        "ctxt_scaler_fpeak",
        "ctxt_scaler_maj",
    ],
    inverse=False,
):
    scale_fns = [
        getattr(ContextScaler.load(s), "scale" if not inverse else "inverse_scale")
        for s in scalers
    ]
    return partial(catalog_context_value_scale, scale_fns=scale_fns)
