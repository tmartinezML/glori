from copy import deepcopy

import torch
import numpy as np
from torch.utils.data import DataLoader


def weighted_random_quadrant_mask(
    img_batch_shape,
    weights=[0.1, 0.1, 0.1, 0.7],
    make_ext_ctxt_mask=False,
    ext_ctxt_p=(0.9, 0.1, 0.1),
    mask_coverage=0.5,
):
    b, _, h, w = img_batch_shape
    assert (
        h == w
    ), f"Height and width must be equal for quadrant masking, got {h=} and {w=}."
    mask_codes = [[1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]]
    sampled_idxs = np.random.choice(len(mask_codes), size=(b,), p=weights)
    mask_code_batch = torch.tensor([mask_codes[i] for i in sampled_idxs]).float()

    # mask = torch.nn.functional.interpolate(mask_code_batch.reshape(b, 1, 2, 2), scale_factor=h // 2, mode="nearest")
    mask = torch.zeros(h, w, b)
    m1, m2, m3, m4 = mask_code_batch.permute(1, 0).reshape(4, 1, 1, b)
    w_cov, h_cov = int(w * mask_coverage), int(h * mask_coverage)
    mask[:h_cov, :w_cov] = m1
    mask[:h_cov, w_cov:] = m2
    mask[h_cov:, :w_cov] = m3
    mask[h_cov:, w_cov:] = m4
    mask = mask.permute(2, 0, 1).unsqueeze(1)

    if make_ext_ctxt_mask:
        apply_left_mask = torch.from_numpy((sampled_idxs == 0) | (sampled_idxs == 2))
        apply_top_mask = torch.from_numpy((sampled_idxs == 0) | (sampled_idxs == 1))
        ext_ctxt_mask = get_ext_ctxt_mask(
            img_batch_shape,
            apply_left_mask=apply_left_mask,
            apply_top_mask=apply_top_mask,
            p=ext_ctxt_p,
        )
        return mask, ext_ctxt_mask

    return mask


def get_ext_ctxt_mask(
    img_batch_shape,
    apply_left_mask=True,
    apply_top_mask=True,
    f_inner=2,
    p=(0.9, 0.1, 0.1),
):
    """
    Generate an extended context mask that masks the top and/or left regions of the input image batch.

    Parameters
    ----------
    img_batch_shape : tuple
        Shape of the input image batch, typically (batch_size, channels, height, width).
    left_mask : bool, optional
        Whether to apply a mask to the left region, by default True.
    top_mask : bool, optional
        Whether to apply a mask to the top region, by default True.
    f_inner : int, optional
        The scaling factor for the inner region, by default 2.

    Returns
    -------
    torch.Tensor
        A mask tensor with the same shape as the input image batch, where the specified regions are
        masked.

    """
    b, _, h, w = img_batch_shape
    p_general, p_bottom, p_right = p

    ext_mask = torch.ones((b, 1, h * f_inner, w * f_inner))
    randoms = torch.rand(b) < p_general

    # Top mask
    size_top = (h * (f_inner - 1)) // 2
    ext_mask[:, :, :size_top, :] *= 1 - torch.logical_and(
        apply_top_mask, randoms
    ).int().reshape(b, 1, 1, 1)

    # Left mask
    size_left = (w * (f_inner - 1)) // 2
    ext_mask[:, :, :, :size_left] *= 1 - torch.logical_and(
        apply_left_mask, randoms
    ).int().reshape(b, 1, 1, 1)

    # Bottom mask
    apply_bottom_mask = (torch.rand(b) < p_bottom) & torch.logical_not(apply_top_mask)
    ext_mask[:, :, -size_top:, :] *= 1 - torch.logical_and(
        apply_bottom_mask, randoms
    ).int().reshape(b, 1, 1, 1)

    # Right mask
    apply_right_mask = (torch.rand(b) < p_right) & torch.logical_not(apply_left_mask)
    ext_mask[:, :, :, -size_left:] *= 1 - torch.logical_and(
        apply_right_mask, randoms
    ).int().reshape(b, 1, 1, 1)
    return ext_mask


def random_quadrant_mask(img_batch_shape):
    """
    Generate a random mask that selects any from none to all of the four quadrants of the input image batch.
    Separate choice for each batch item, so each item can have a different quadrant selected.

    Parameters
    ----------
    img_batch_shape : tuple
        Shape of the input image batch, typically (batch_size, channels, height, width).

    Returns
    -------
    torch.Tensor
        A mask tensor with the same shape as the input image batch, where one quadrant is selected.
    """
    b, _, h, w = img_batch_shape
    assert (
        h == w
    ), f"Height and width must be equal for quadrant masking, got {h=} and {w=}."
    # Shape (b, 1, 2, 2), indicating which quadrants are selected:
    t = torch.randint(0, 2, (b, 1, 2, 2)).float()
    # Upscale to shape (b, 1, h, w):
    mask = torch.nn.functional.interpolate(t, scale_factor=h // 2, mode="nearest")
    return mask
