from functools import partial, reduce
import random

import torch
from torchvision.transforms.v2.functional import crop


def compose(*functions):
    return reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)


def make_post(fn_dict, copy=False):
    """
    Returns a function that applies a series of transformations to the input sample.

    Parameters
    ----------
    fn_dict : dict
        A dictionary where keys are the names of the transformations and values are
        dictionaries of keyword arguments for each transformation.

    Returns
    -------
    function
        A function that applies the specified transformations in sequence.
    """

    def transform(sample):
        sample = conditional_copy(sample, copy=copy)
        for key, fn in fn_dict.items():
            try:
                sample[key] = fn(sample[key])
            except ValueError as e:
                print(f"ValueError in {key} for sample {sample}")
                raise e
        return sample

    return transform


def ctxt_to_artifact(
    x,
):
    x = x[:, 0:3:2]
    h, w = x.shape[-2:]
    x[:, :, h // 4 : -h // 4, w // 4 : -w // 4] *= 0
    x[:, 1] = sparse_convolve_batch_vectorized(x[:, 1], artifact_maker)
    return x


def artifact_maker(r, phi, omega=50):
    """Example: Gaussian with angular modulation"""
    return torch.sin(omega * phi)


def sparse_convolve_batch_vectorized(images, f):
    """
    Fully vectorized sparse convolution for batch of images using PyTorch.

    Args:
        images: 3D tensor (batch, height, width)
        f: function(r, phi) that defines the kernel

    Returns:
        3D tensor (batch, height, width)
    """
    batch_size, h, w = images.shape

    # Find non-zero pixels for all images
    mask = images >= 1
    batch_idx, y_coords, x_coords = torch.where(mask)
    values = images[batch_idx, y_coords, x_coords]

    # Create coordinate grids
    y_grid = torch.arange(h, device=images.device, dtype=torch.float32).view(-1, 1)
    x_grid = torch.arange(w, device=images.device, dtype=torch.float32).view(1, -1)

    # Calculate distances from each non-zero pixel (shape: n_pixels, h, w)
    dy = y_grid[None, :, :] - y_coords[:, None, None].float()
    dx = x_grid[None, :, :] - x_coords[:, None, None].float()

    r = torch.sqrt(dx**2 + dy**2)
    phi = torch.atan2(dy, dx)

    # Evaluate kernel function
    kernels = f(r, phi) * values[:, None, None]  # (n_pixels, h, w)

    # Sum kernels for each image in batch using index_add
    result = torch.zeros((batch_size, h, w), device=images.device, dtype=images.dtype)
    result.index_add_(0, batch_idx, kernels)

    return result


def make_ctxt_to_topk(*, key, copy=False, **kwargs):
    """
    Returns a function that converts a catalog context tensor to a top-k representation.

    Parameters
    ----------
    key : str
        The key in the sample dictionary to apply the transformation to.
    k : int
        The number of top elements to select.
    sort_channel : bool, optional
        Whether to sort the selected elements by their values, by default True.

    Returns
    -------
    function
        A function that performs the top-k conversion on the specified key.
    """

    def transform(sample):
        sample = conditional_copy(sample, copy=copy)
        try:
            sample[key] = ctxt_to_topk_functional(sample[key], **kwargs)
        except ValueError as e:
            print(f"ValueError in ctxt_to_topk_functional for sample {sample}")
            raise e

        return sample

    return transform


def ctxt_to_topk_functional(x, k=10, sort_channel=2, min_val=10, pad=1):
    """
    Convert a catalog context tensor to a top-k representation.

    Parameters
    ----------
    x : torch.Tensor
        The input context tensor of shape (C, H, W) or (B, C, H, W).
    k : int
        The number of top elements to select.
    sort_channel : bool, optional
        Whether to sort the selected elements by their values, by default True.

    Returns
    -------
    torch.Tensor
        The top-k representation of shape (C+2, k), where the last two channels are the
        normalized y and x coordinates.
    """
    batched = x.ndim == 4
    if not batched:
        x = x.unsqueeze(0)  # Add batch dimension

    h, w = x.shape[-2:]
    # Mask out center
    x[:, :, h // 4 : -h // 4, w // 4 : -w // 4] *= 0
    # Mask out borders based on pad. Pad is fraction of distance between center
    # and border, which is also half the inner image width.
    # It's defined from the inside but applied from the outside, hence 1 - pad
    if pad < 1:
        pad_h = int(h // 4 * (1 - pad))
        pad_w = int(w // 4 * (1 - pad))
        x[:, :, :pad_h, :] *= 0
        x[:, :, -pad_h:, :] *= 0
        x[:, :, :, :pad_w] *= 0
        x[:, :, :, -pad_w:] *= 0
    # Mask out low values
    x = x * (x[:, sort_channel].unsqueeze(1) >= min_val)

    ii = torch.topk(x[:, sort_channel].flatten(start_dim=1), k=k, dim=1).indices
    out = torch.gather(
        # x.flatten(start_dim=2)[:, 1:], 2, ii.unsqueeze(1).expand(-1, x.shape[1] - 1, -1)
        # Edit: use only peak flux
        x.flatten(start_dim=2)[:, sort_channel : sort_channel + 1],
        2,
        ii.unsqueeze(1).expand(-1, 1, -1),
    )

    # Turn flattened indices into relative x-y indices
    dxy = ii.unsqueeze(1).expand(-1, 2, -1).to(torch.float32)  # (B, 2, k)
    h_norm = 2.0 / h  # Computed ONCE
    w_norm = 2.0 / w  # Computed ONCE
    dxy[:, 0] = (dxy[:, 0] // w) * h_norm - 1.0  # Uses pre-computed h_norm
    dxy[:, 1] = (dxy[:, 1] % w) * w_norm - 1.0  # Uses pre-computed w_norm
    # Set unused entries to 0
    dxy = dxy * (out[:, 0:1, :] > 0).to(torch.float32)
    out = torch.cat((dxy, out), dim=1).permute(0, 2, 1)  # (B, k, C+2)

    if not batched:
        return out.squeeze(0)  # Remove batch dimension

    # Safety check for nan values
    if torch.isnan(out).any():
        raise ValueError("NaN values found in top-k output tensor.")

    return out


def dependent_random_crop(sample, *, crop_size, keys, f=1, s=1, copy=False):
    """
    Perform a dependent random crop on the input sample.

    Parameters
    ----------
    sample : torch.Tensor
        The input sample to be cropped. Expected shape is (C, H, W).
    crop_size : int
        The size of the crop to be extracted.
    f : int, optional
        The downsampling factor, by default 1.

    Returns
    -------
    torch.Tensor
        The cropped sample.
    """
    assert keys is not None, "keys must be provided"
    if isinstance(f, (int, float)):
        f = [f] * (len(keys) - 1)
    if isinstance(s, (int, float)):
        s = [s] * (len(keys) - 1)
    if len(keys) > 1:
        assert (
            len(f) == len(s) == len(keys) - 1
        ), f"Length of f ({len(f)}) and s ({len(s)}) must match keys-1 ({len(keys)-1})."

    img_shape = sample[keys[0]].shape
    crop_params = random_crop_generator(
        img_shape, crop_size, max(f + [1]), max(s + [1])
    )
    # Create new dictionary instead of mutating
    # Choose copy vs mutation based on parameter
    result = conditional_copy(sample, copy=copy)
    result[keys[0]] = crop(sample[keys[0]], *crop_params)

    for i in range(len(keys) - 1):
        crop_params_f = crop_params_f_transform(crop_params, f[i], s[i], img_shape)
        result[keys[i + 1]] = crop(sample[keys[i + 1]], *crop_params_f)

    return result


def make_dependent_random_crop(**kwargs):
    """
    Returns a function that performs a dependent random crop on the input sample.

    Parameters
    ----------
    crop_size : int
        The size of the crop to be extracted.
    f : int, optional
        The downsampling factor, by default 1.

    Returns
    -------
    function
        A function that performs the dependent random crop.
    """

    return partial(dependent_random_crop, **kwargs)


def dependent_center_crop(sample, *, crop_size, keys=None, f=1, s=1, copy=False):
    """
    Perform a dependent center crop on the input sample.

    Parameters
    ----------
    sample : torch.Tensor
        The input sample to be cropped. Expected shape is (C, H, W).
    crop_size : int
        The size of the crop to be extracted.
    f : int, optional
        The downsampling factor, by default 1.

    Returns
    -------
    torch.Tensor
        The cropped sample.
    """
    assert keys is not None, "keys must be provided"
    if len(keys) > 1:
        if isinstance(f, (int, float)):
            f = [f] * (len(keys) - 1)
        if isinstance(s, (int, float)):
            s = [s] * (len(keys) - 1)
        assert (
            len(f) == len(s) == len(keys) - 1
        ), f"Length of f ({len(f)}) and s ({len(s)}) must match keys-1 ({len(keys)-1})."
    else:
        f = s = None

    img_shape = sample[keys[0]].shape
    t = (img_shape[-2] - crop_size) // 2
    l = (img_shape[-1] - crop_size) // 2
    crop_params = int(t), int(l), int(crop_size), int(crop_size)
    # Choose copy vs mutation based on parameter
    result = conditional_copy(sample, copy=copy)
    result[keys[0]] = crop(sample[keys[0]], *crop_params)

    for i in range(len(keys) - 1):
        crop_params_f = crop_params_f_transform(crop_params, f[i], s[i], img_shape)
        result[keys[i + 1]] = crop(sample[keys[i + 1]], *crop_params_f)

    return result


def make_dependent_center_crop(**kwargs):
    """
    Returns a function that performs a dependent center crop on the input sample.

    Parameters
    ----------
    crop_size : int
        The size of the crop to be extracted.
    f : int, optional
        The downsampling factor, by default 1.

    Returns
    -------
    function
        A function that performs the dependent center crop.
    """

    return partial(dependent_center_crop, **kwargs)


def conditional_copy(x, copy=False):
    if copy:
        return x.copy()
    return x


def crop_params_f_transform(crop_params, f, s, img_shape):
    # f is how much bigger the cutout should be (angular size), as compared to original cutout.
    # s is how much higher the resolution is, as compared to original cutout.
    if f == 1 and s == 1:
        return crop_params
    t, l, h, w = crop_params
    H, W = img_shape[-2:]
    t_ = s * ((H - h) // 2 * (1 - 1 / f) + t // f)
    l_ = s * ((W - w) // 2 * (1 - 1 / f) + l // f)
    return int(t_), int(l_), int(h * s), int(w * s)


def random_crop_generator(img_shape, crop_size, f=1, s=1):
    t = random.randint(0, int((img_shape[-2] - crop_size) / f * s)) * f / s
    l = random.randint(0, int((img_shape[-1] - crop_size) / f * s)) * f / s
    return int(t), int(l), int(crop_size), int(crop_size)
