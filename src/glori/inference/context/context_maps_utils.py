from copy import deepcopy
import math
import itertools

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import pil_to_tensor

import itertools
from tqdm import trange
import numpy as np
import numba as nb
from numba.np.unsafe.ndarray import to_fixed_tuple


@nb.njit
def _sample_histdd_nb(
    countsdd: np.ndarray, bins_list: np.ndarray, n_samples: int
) -> np.ndarray:
    # countsdd: ndarray (ndim)
    # bins_list: list of arrays, each of length n_bins+1
    ndim = countsdd.ndim
    out = np.empty((n_samples, ndim), dtype=np.float64)

    # precompute bin centers and widths
    centers = [0.5 * (bins[:-1] + bins[1:]) for bins in bins_list]
    widths = [bins[1:] - bins[:-1] for bins in bins_list]

    # flatten weights
    w = countsdd.ravel()
    total = w.sum()

    for i in range(n_samples):
        r = np.random.rand() * total
        acc = 0.0
        flat_idx = 0
        for k in range(w.size):
            acc += w[k]
            if acc >= r:
                flat_idx = k
                break

        # unravel index
        idx_nd = np.empty(ndim, dtype=np.int64)
        tmp = flat_idx
        for d in range(ndim - 1, -1, -1):
            dim = countsdd.shape[d]
            idx_nd[d] = tmp % dim
            tmp //= dim

        # map to centers + jitter
        for d in range(ndim):
            c = centers[d][idx_nd[d]]
            wbin = widths[d][idx_nd[d]]
            out[i, d] = c + (np.random.rand() - 0.5) * wbin

    return out


def _histdd_marginal_loop(
    marginal_counts: np.ndarray,
    free_axes: np.ndarray,
    samples: np.ndarray,
    bins_list: list,
) -> np.ndarray:
    n_samples = marginal_counts.shape[0]
    for i in range(n_samples):
        marginal = marginal_counts[i]
        if marginal.sum() == 0:
            raise ValueError(f"No samples available for input combination.")
        sampled = _sample_histdd_nb(
            marginal,
            [bins_list[ax] for ax in free_axes],
            n_samples=1,
        )
        for j in range(free_axes.size):
            samples[i, free_axes[j]] = sampled[0, j]
    return samples


def _marginal_counts_3d(
    countsdd: np.ndarray, fixed_axes: np.ndarray, fixed_bins: np.ndarray
) -> np.ndarray:
    """
    countsdd: (nb, nb, nb)
    fixed_axes: array-like of length 1 or 2
    fixed_bins: shape (n_samples, len(fixed_axes))

    returns:
      - if 1 fixed axis: (n_samples, nb, nb)
      - if 2 fixed axes: (n_samples, nb)
    with free axes in the original axis order
    """
    fixed_axes = np.asarray(fixed_axes)
    if fixed_axes.size not in (1, 2):
        raise ValueError("fixed_axes must have length 1 or 2")

    free_axes = np.array([ax for ax in range(3) if ax not in fixed_axes])
    perm = np.empty(3, dtype=np.int64)
    perm[: fixed_axes.size] = fixed_axes
    perm[fixed_axes.size :] = free_axes
    inv_perm = np.argsort(perm)
    # c = np.transpose(countsdd, to_fixed_tuple(perm, 3))
    c = np.transpose(countsdd, perm)

    if fixed_axes.size == 1:
        marginal = c[fixed_bins[:, 0], :, :]  # (n_samples, nb, nb)
        # free axes are currently in order free_axes; restore original order

    elif fixed_axes.size == 2:
        marginal = np.empty(
            (fixed_bins.shape[0], fixed_bins.shape[1], c.shape[-1]), dtype=c.dtype
        )
        for i in range(fixed_bins.shape[0]):
            marginal[i, :] = c[fixed_bins[i, 0], fixed_bins[i, 1], :]

    # Bring back to original axis order
    marginal = np.transpose(marginal, inv_perm)

    return marginal


def _enforce_ranges(
    countsdd,
    bins_list,
    n_samples,
    ranges=(None, None, None),
    inputs=(None, None, None),
):
    if all(r is None for r in ranges):
        return countsdd, bins_list, inputs

    inputs_enf = list(inputs)
    ranges_enf = list(ranges)
    for i, r in enumerate(ranges):
        if r is None:
            continue
        try:
            rmin, rmax = r if len(r) == 2 else r * 2
        except ValueError as e:
            print(r)
            raise e
        if rmin == rmax:
            inputs_enf[i] = np.full(n_samples, rmin)
            ranges_enf[i] = None  # no need to enforce range if it's a single value

    inputs = tuple(inputs_enf)
    ranges = tuple(ranges_enf)

    assert all(
        len(r) == 2 for r in ranges if r is not None
    ), f"Each range must be a tuple of (min, max), got {ranges}"

    bins_list = deepcopy(bins_list)
    for i, (r, bins) in enumerate(zip(ranges, bins_list)):
        # Act only if range is specified for this axis
        if r is None:
            continue

        # Unpack and validate range
        rmin, rmax = r
        if rmin >= rmax:
            raise ValueError(
                f"Invalid range for axis {i}: min {rmin} must be less than max {rmax}."
            )

        # Set selection mask and validate that it includes at least one bin
        mask_min = bins[:-1] >= rmin if rmin is not None else True
        mask_max = bins[1:] <= rmax if rmax is not None else True
        mask = mask_min & mask_max
        if not mask.any():
            raise ValueError(f"No bins within range {r} for axis {i}.")
        valid_idxs = np.where(mask)[0]

        # Apply mask
        countsdd = countsdd.take(valid_idxs, axis=i)
        # Explanation of +2: upper slice limit is exclusive (+1),
        # and we need to include the upper edge of the last valid bin (+1),
        bins_list[i] = bins_list[i][valid_idxs[0] : valid_idxs[-1] + 2]
    return countsdd, bins_list, inputs, ranges


def _validate_inputs(
    countsdd,
    bins_list,
    inputs,
    n_samples=None,
):
    assert (
        len(bins_list) == countsdd.ndim
    ), "Number of different sets of bins must match number of dimensions"
    assert all(
        len(bins_list[i]) == countsdd.shape[i] + 1 for i in range(countsdd.ndim)
    ), f"Number of bin edges must be one more than number of bins, got {[len(bins) for bins in bins_list]} for bins and {countsdd.shape} for counts."
    assert len(inputs) == countsdd.ndim, "Input length must match number of dimensions"
    assert any(
        x is not None for x in inputs + (n_samples,)
    ), "At least one of inputs must be provided"
    assert (
        len(inp_len := set(len(x) for x in filter(lambda x: x is not None, inputs)))
        <= 1
    ), "All non-None inputs must have same length"
    return inp_len.pop() if n_samples is None else n_samples


def sample_histdd_marginal(
    countsdd,
    bins_list,
    inputs=(None, None, None),
    ranges=(None, None, None),
):
    n_samples = _validate_inputs(countsdd, bins_list, inputs)

    countsdd, bins_list, inputs, ranges = _enforce_ranges(
        countsdd,
        bins_list,
        n_samples,
        ranges,
        inputs,
    )
    input_vals = np.full((n_samples, countsdd.ndim), np.nan)
    for j, inp in enumerate(inputs):
        if inp is not None:
            input_vals[:, j] = inp

    samples = input_vals.copy()
    input_flag = input_flag = ~np.isnan(input_vals[0])

    fixed_axes = np.where(input_flag)[0]
    free_axes = np.where(~input_flag)[0]

    # precompute fixed bin indices per sample
    fixed_bins = np.stack(
        [
            np.clip(
                np.digitize(samples[:, j], bins_list[j]) - 1,
                0,
                bins_list[j].shape[0] - 2,
            )
            for j in fixed_axes
        ],
        axis=1,
    )
    marginal_counts = _marginal_counts_3d(countsdd, fixed_axes, fixed_bins)

    if (
        zero_flag := marginal_counts.sum(axis=tuple(range(1, marginal_counts.ndim)))
        == 0
    ).any():
        raise ValueError(
            f"No samples available for input combinations."
        )

    samples = _histdd_marginal_loop(
        marginal_counts,
        free_axes,
        samples,
        bins_list,
    )
    return samples


def sample_histdd(
    countsdd,
    bins_list,
    n_samples=None,
    inputs=(None, None, None),
    ranges=(None, None, None),
):
    n_samples_inferred = _validate_inputs(countsdd, bins_list, inputs, n_samples)

    countsdd, bins_list, inputs, ranges = _enforce_ranges(
        countsdd,
        bins_list,
        n_samples_inferred,
        ranges,
        inputs,
    )

    if all(x is None for x in inputs):
        return _sample_histdd_nb(countsdd, bins_list, n_samples)

    return sample_histdd_marginal(countsdd, bins_list, inputs, ranges)


def auto_fontsize(text, image_size=(64, 64), margin=0.05):
    """
    Calculate font size directly from character metrics without test rendering.

    Parameters:
    -----------
    text : str
        Text to render
    target_area : float
        Target area in pixels²
    image_size : tuple
        Image dimensions
    font_path : str
        Font file path

    Returns:
    --------
    int : Calculated font size
    """
    # Get text dimensions - count actual characters
    lines = text.split("\n")
    n_lines = len(lines)
    max_chars_per_line = max(len(line) for line in lines)

    # For monospace fonts, we can predict dimensions directly
    # Typical character metrics for monospace fonts:
    char_width_ratio = 0.6  # Character width ≈ 0.6 * font_size
    char_height_ratio = 0.8  # Character height ≈ 0.8 * font_size
    line_spacing = 0.2  # Line spacing = int(0.2 * font_size)

    # See whether height or width is the limiting factor
    if (n_lines * (char_height_ratio + line_spacing)) / (
        max_chars_per_line * char_width_ratio
    ) > (image_size[1] / image_size[0]):
        # Height is limiting factor. Choose font size based on image height
        font_size = (
            (1 - 2 * margin)
            * image_size[1]
            / (n_lines * (char_height_ratio + line_spacing))
        )
    else:
        # Width is limiting factor. Choose font size based on image width
        font_size = (
            (1 - 2 * margin) * image_size[0] / (max_chars_per_line * char_width_ratio)
        )
    # print(font_size)
    return max(1, font_size)


def text_symbol_to_mask(text, size=(64, 64), font_size="auto"):
    # Create image
    img = Image.new("L", size, 0)  # 'L' for grayscale
    draw = ImageDraw.Draw(img)

    if font_size == "auto":
        font_size = auto_fontsize(text, image_size=size)

    # Try to use a font, fallback to default
    try:
        font = ImageFont.truetype("FreeMonoBold.ttf", font_size)
    except Exception as e:
        print(e)
        font = ImageFont.load_default(font_size)

    draw.text(
        tuple(s // 2 for s in size),
        text,
        fill=256,
        font=font,
        anchor="mm",
        align="center",
        spacing=max(1, int(0.2 * font_size)),
    )
    return pil_to_tensor(img).squeeze() > 128


def find_best_grid(N, H, W):
    aspect = H / W
    c_est = math.sqrt(N / aspect)

    best_rc = None
    best_score = float("inf")

    for c in range(max(1, int(c_est) - 5), int(c_est) + 6):
        r = math.ceil(N / c)
        if r * c >= N:
            # Try to make cells as square as possible: minimize distortion
            cell_aspect = (H / r) / (W / c)  # = (H * c) / (W * r)
            distortion = abs(
                math.log(cell_aspect)
            )  # log scale = distortion from square
            if distortion < best_score:
                best_score = distortion
                best_rc = (r, c)

    return best_rc


def best_point_grid(N, H, W):
    grid = torch.zeros((H, W))
    r, c = find_best_grid(N, H, W)
    h_step, h_offset = H // r, H % r
    w_step, w_offset = W // c, W % c
    for k, (i, j) in enumerate(itertools.product(range(r), range(c))):
        # if i * h_step + h_step // 2 < H and j * w_step + w_step // 2 < W:
        if k < N:
            grid[
                i * h_step + (h_step + h_offset) // 2,
                j * w_step + (w_step + w_offset) // 2,
            ] = 1
    return grid


def cartesian_outer_prod_nd(a1, a2):
    """
    Generate a cartesian product of two N-D arrays.
    Product is taken over first dimensions, other dimensions are concatenated.
    """
    assert a1.ndim == a2.ndim, "Both arrays must have the same number of dimensions."
    assert a1.ndim >= 2, "Input arrays must have at least two dimensions."
    a1 = a1.unsqueeze(0).repeat(a2.shape[0], *(1,) * a1.ndim).transpose(1, 0)
    a2 = a2.unsqueeze(0).repeat(a1.shape[0], *(1,) * a2.ndim)
    # print("outer prod. shapes: ", a1.shape, a2.shape)
    return torch.cat((a1, a2), dim=2).flatten(0, 1)


def dice_pos_mask(enc_size, spacing="thirds"):
    x = torch.zeros((6, enc_size, enc_size))
    s = enc_size
    mid = s // 2
    match spacing:
        case "thirds":
            lo = s // 3
            hi = 2 * s // 3
        case "quarters":
            lo = s // 4
            hi = 3 * s // 4
        case _:
            raise ValueError("Invalid spacing option. Use 'thirds' or 'quarters'.")

    # One
    x[0, mid, mid] = 1

    # Two
    x[1, lo, lo] = 1
    x[1, hi, hi] = 1

    # Three
    x[2, lo, lo] = 1
    x[2, mid, mid] = 1
    x[2, hi, hi] = 1

    # Four
    x[3, lo, lo] = 1
    x[3, lo, hi] = 1
    x[3, hi, lo] = 1
    x[3, hi, hi] = 1

    # Five
    x[4, lo, lo] = 1
    x[4, lo, hi] = 1
    x[4, mid, mid] = 1
    x[4, hi, lo] = 1
    x[4, hi, hi] = 1

    # Six
    x[5, lo, lo] = 1
    x[5, lo, hi] = 1
    x[5, mid, lo] = 1
    x[5, mid, hi] = 1
    x[5, hi, lo] = 1
    x[5, hi, hi] = 1
    return x
