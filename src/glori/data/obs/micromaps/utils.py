import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from copy import deepcopy

import torch
import numpy as np
from astropy.coordinates import SkyCoord
from einops import reduce
from astropy.wcs import WCS
from astropy.io import fits
from tqdm import tqdm
from scipy import ndimage
from skimage.measure import label

import glori.settings.paths as paths
from glori.data.trf.scalers import ContextScaler
from glori.maps.map_utils import get_map_image


def maps_to_encoding_path(maps_path):
    if is_str := isinstance(maps_path, str):
        maps_path = Path(maps_path)

    encoding_path = maps_path.with_name(
        maps_path.name.replace("micromaps", "micromap_encodings")
    )
    return str(encoding_path) if is_str else encoding_path


def get_model_image(
    pointing, model_dir=(paths.LOFAR_DATA_PARENT / "pointings"), which="PyBDSF"
):
    """
    Get the model image for a given pointing.
    """
    match which:
        case "DDF":
            model_file = (
                model_dir
                / pointing
                / "images/image_full_ampphase_di_m.NS.int.model.fits"
            )
        case "PyBDSF":
            model_file = model_dir / pointing / "mosaic.bdsf_model.fits"
        case _:
            raise ValueError(f"Unknown model type: <{which}>. Supported: DDF, PyBDSF")

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file {model_file} does not exist. Please check the model data path."
        )

    return get_map_image(model_file, get_wcs=False).squeeze()


def get_mosaic_shape(pointing, mosaic_dir=paths.MOSAIC_DIR_DR2):
    """
    Get the shape of the mosaic image for a given pointing.
    """
    mosaic_file = Path(mosaic_dir) / pointing / "mosaic-blanked.fits"
    if not mosaic_file.exists():
        raise FileNotFoundError(f"Mosaic file not found: {mosaic_file}")
    with fits.open(mosaic_file, memmap=True) as hdul:
        data = hdul[0].data
    shape = deepcopy(data.shape)
    del data  # Close the memmap
    return shape


def fill_nan_pixels(image, max_size=8, early_stop=True):
    """
    Fill NaN pixels in small islands with mean of their 8-connectivity neighbors.

    Parameters:
        image: Input image array
        max_size: Maximum size of NaN island to fill (default: 5)

    Returns:
        filled_image: Image with small NaN islands filled
        n_filled: Number of pixels that were filled
    """
    filled_image = image.copy()

    # Find NaN islands once
    nan_mask = np.isnan(filled_image)
    if not nan_mask.any():
        return filled_image

    # Label connected components (8-connectivity)
    island_labels, _ = label(nan_mask, return_num=True)

    # Count pixels in each island
    island_sizes = np.bincount(island_labels.ravel())[1:]  # Skip background (0)

    # Process each small island
    for island_id, size in enumerate(island_sizes, start=1):
        if size > max_size:
            if early_stop:
                break
            continue

        # Get all pixels in this island
        island_coords = np.where((island_labels == island_id))

        # Fill each pixel in the island
        for y, x in zip(*island_coords):
            # Get 8-connectivity neighborhood bounds & extract neighborhood
            y_start, y_end = max(0, y - 1), min(filled_image.shape[0], y + 2)
            x_start, x_end = max(0, x - 1), min(filled_image.shape[1], x + 2)
            neighborhood = filled_image[y_start:y_end, x_start:x_end]

            # Get non-NaN neighbors (excluding center pixel)
            valid_mask = ~np.isnan(neighborhood)
            valid_mask[y - y_start, x - x_start] = False

            # If we have valid neighbors, fill with their mean
            if valid_mask.sum():
                valid_neighbors = neighborhood[valid_mask]
                filled_image[y, x] = valid_neighbors.mean()

    return filled_image


def center_cutout(x, cutout_size_px):
    """
    Get a centered cutout of the image (or wcs).
    """
    cby2 = cutout_size_px // 2
    cpx = (x.pixel_shape if isinstance(x, WCS) else x.shape)[-1] // 2
    sl = slice(cpx - cby2, cpx + cby2)
    return x[sl, sl]


def assemble_micromaps(micromaps, centers, map_shape=None):
    h, w = micromaps.shape[-2:]
    H, W = map_shape or centers.max(axis=0) + np.array([h, w]) // 2
    assembled = np.empty((H, W)) * np.nan
    for m, c in tqdm(
        zip(micromaps, centers), total=len(micromaps), desc="Assembling micromaps"
    ):
        x, y = c
        assembled[x - h // 2 : x + h // 2, y - w // 2 : y + w // 2] = m
    return assembled


def context_map_by_wcs(
    wcs,
    catalog,
    scalers=["ctxt_scaler_ftot", "ctxt_scaler_fpeak", "ctxt_scaler_maj"],
    scale_output=False,
):
    if scale_output:
        if type(scalers[0]) is str:
            scalers = [ContextScaler.load(scaler) for scaler in scalers]
        assert (types := set([type(scaler) for scaler in scalers])) == {
            ContextScaler
        }, f"All scalers must be of type ContextScaler, got {types}"

    # Get sub-catalog filtered by WCS, i.e. all sources on mosaic
    sub_cat = filter_catalog_by_wcs(catalog, wcs)

    # Get pixel coordinates of sources in the sub-catalog
    ix, iy = wcs.wcs_world2pix(sub_cat.RA.values, sub_cat.DEC.values, 0)
    ix = ix.astype(int)
    iy = iy.astype(int)

    # Make context array
    ctxt = np.zeros((4, *list(reversed(wcs.pixel_shape))))
    ctxt[0][iy, ix] = 1
    qtys = ["Total_flux", "Peak_flux", "Maj"]
    for i, (qty, scaler) in enumerate(zip(qtys, scalers), start=1):
        ctxt[i][iy, ix] = (
            scaler.scale(sub_cat[qty].values) if scale_output else sub_cat[qty].values
        )
    return ctxt, sub_cat


def expand_context_map(
    ctxt_arr,
    f_upscale=4,
):
    """
    Expand context map by adding zero pixels.
    """
    # Get current shape
    shape = ctxt_arr.shape
    is_torch = isinstance(ctxt_arr, torch.Tensor)

    match len(shape):
        case 3:
            c, h, w = shape
            new_shape = (c, h * f_upscale, w * f_upscale)
            sl = (slice(None),) + (slice(None, None, f_upscale),) * 2
        case 4:
            b, c, h, w = shape
            new_shape = (b, c, h * f_upscale, w * f_upscale)
            sl = (slice(None),) * 2 + (slice(None, None, f_upscale),) * 2
        case _:
            raise ValueError(f"Unsupported array dimension: {len(shape)}")

    # Create new array
    expanded = (torch if is_torch else np).zeros(new_shape, dtype=ctxt_arr.dtype)
    # Fill in the original context map using the multi-dimensional slice
    expanded[sl] = ctxt_arr

    return expanded


def reduce_context_map(
    ctxt_arr,
    scalers=["ctxt_scaler_ftot", "ctxt_scaler_fpeak", "ctxt_scaler_maj"],
    f_downscale=4,
    input_scaled=False,
    scale_output=False,
):
    """
    Reduce context map by averaging over the specified scalers.
    """
    if input_scaled or scale_output:
        if type(scalers[0]) is str:
            scalers = [ContextScaler.load(scaler) for scaler in scalers]
        assert (types := set([type(scaler) for scaler in scalers])) == {
            ContextScaler
        }, f"All scalers must be of type ContextScaler, got {types}"

    is_torch = isinstance(ctxt_arr, torch.Tensor)
    match (ndim := ctxt_arr.ndim):
        case 3:
            ctxt_arr = ctxt_arr.unsqueeze(0) if is_torch else ctxt_arr[np.newaxis, ...]
        case 4:
            pass
        case _:
            raise ValueError(f"Unsupported array dimension: {ctxt_arr.ndim}")

    # Invert the scaling before the sum
    arr_inv_sc = ctxt_arr.clone() if is_torch else ctxt_arr.copy()
    arr_mask = arr_inv_sc[:, 0] > 0
    if input_scaled:
        for i, scaler in enumerate(scalers, start=1):
            arr_inv_sc[:, i] = scaler.inverse_scale(arr_inv_sc[:, i]) * arr_mask

    # Reduce the array by summing f_downscale-neighboring pixels
    arr_red = reduce(
        arr_inv_sc,
        "b c (h p1) (w p2) -> b c h w",
        "sum",
        p1=f_downscale,
        p2=f_downscale,
    )
    # Positional context should still be binary map.
    arr_mask = (
        (arr_red[:, 0] > 0).to(torch.float32)
        if is_torch
        else (arr_red[:, 0] > 0).astype(np.float32)
    )
    arr_red[:, 0] = arr_mask
    # Apply the scaling again to the summed values
    # Also, set -inf to zero (we get -inf where unscaled values are 0)
    for i, scaler in enumerate(scalers, start=1):
        arr_red[:, i] = (torch.where if is_torch else np.where)(
            (torch if is_torch else np).isfinite(arr_red[:, i]),
            scaler.scale(arr_red[:, i]) if scale_output else arr_red[:, i],
            0,
        )

    if ndim == 3:
        arr_red = arr_red.squeeze(0)

    return arr_red


@contextmanager
def graceful_shutdown(executor):
    """Context manager for graceful shutdown of ThreadPoolExecutor."""

    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        executor.shutdown(wait=False)
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        yield executor
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down...")
        executor.shutdown(wait=False)
        sys.exit(0)
    finally:
        executor.shutdown(wait=True)


def coords_in_wcs_footprint(ra_coords, dec_coords, wcs, axes=None, mask=None):
    """
    Check if (RA, DEC) coordinates are within the WCS footprint.
    Optimized for your LOFAR data processing.
    """
    # This is needed to later construct a binary mask of valid size
    cat_len = len(ra_coords)
    # If dec_coords are sorted, we can quickly exclude those outside the RA
    # range of the WCS footprint
    # (this works only with dec because it is -90, 90, RA will not work since it
    # goes all around the sphere)
    # If ra_coords are sorted, we can quickly exclude those outside the RA range of the WCS footprint
    bl, tl, tr, br = wcs.calc_footprint(axes=axes)  # bottom-left, top-right corners
    dec_min, dec_max = min(bl[1], br[1]), max(tl[1], tr[1])
    i_left = np.searchsorted(dec_coords, dec_min, side="left")
    i_right = i_left + np.searchsorted(dec_coords[i_left:], dec_max, side="right")
    ra_coords = ra_coords[i_left:i_right]
    dec_coords = dec_coords[i_left:i_right]
    # Convert to SkyCoord
    coords = SkyCoord(ra_coords, dec_coords, unit="deg")

    # Convert to pixel coordinates
    try:
        pix_x, pix_y = wcs.world_to_pixel(coords)
    except Exception:
        print(
            "Error transforming coordinates to pixel space. Check WCS and coordinates."
        )
        # If transformation fails, return all False
        return np.zeros(len(ra_coords), dtype=bool)

    # Get image shape
    if axes is not None:
        ny, nx = axes[1], axes[0]
    elif getattr(wcs, "array_shape", None) is not None:
        ny, nx = wcs.array_shape
    else:
        # Try to get from pixel_shape or set default
        try:
            ny, nx = wcs.pixel_shape[1], wcs.pixel_shape[0]
        except:
            # You might need to pass image shape explicitly
            raise ValueError("Cannot determine image shape from WCS")

    # Make binary mask of original coordinate length
    valid_coords = np.zeros(cat_len, dtype=bool)

    # Check bounds
    valid_coords[i_left:i_right] = (
        (pix_x >= 0)
        & (pix_x < nx)
        & (pix_y >= 0)
        & (pix_y < ny)
        & np.isfinite(pix_x)
        & np.isfinite(pix_y)
    )

    # If a mask is passed, remove all coords that are not covered by the mask
    if mask is not None and valid_coords[i_left:i_right].any():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (ny, nx):
            raise ValueError(
                f"Mask shape {mask.shape} does not match WCS image shape {(ny, nx)}"
            )
        valid_coords[valid_coords] &= mask[
            pix_y[valid_coords[i_left:i_right]].astype(int),
            pix_x[valid_coords[i_left:i_right]].astype(int),
        ]

    return valid_coords


# Example usage with your existing code:
def filter_catalog_by_wcs(catalog, wcs, axes=None, mask=None):
    """Filter catalog entries that fall within the WCS footprint"""
    valid_mask = coords_in_wcs_footprint(
        catalog["RA"].values, catalog["DEC"].values, wcs, mask=mask, axes=axes
    )
    return catalog[valid_mask]


def single_micromap_cutout(map_arr, micromap_size, center, wcs=None, fill_excess=False):
    """
    Get a single micromap cutout from a map array.
    """
    x0, y0 = center - micromap_size // 2
    x1, y1 = center + micromap_size // 2
    # Ensure the cutout is within the bounds of the map array.
    # If not, give error.
    if x0 < 0 or y0 < 0 or x1 > map_arr.shape[0] or y1 > map_arr.shape[1]:
        if not fill_excess or wcs is not None:
            raise ValueError(
                f"Cutout center {center} with size {micromap_size} is out of bounds for the map array with shape {map_arr.shape}"
            )
        else:
            # Fill the excess with zeroes
            # Offsets are defined as positive values:
            dx0 = -min(x0, 0)
            dy0 = -min(y0, 0)
            dx1 = max(x1 - map_arr.shape[0], 0)
            dy1 = max(y1 - map_arr.shape[1], 0)
            # Pad the map_arr
            # print(f"Map arr shape for padding: {map_arr.shape}")
            padded_map = np.pad(
                map_arr,
                (
                    (dx0, dx1),
                    (dy0, dy1),
                )
                + ((0, 0),) * (map_arr.ndim - 2),
                mode="constant",
                constant_values=0,
            )
            # We add the lower offsets to the original coordinates.
            # In the case of the lower coordinates, if the lower offset is > 0,
            # it means the coordinates were negative and we need to shift them to the right by the offset to get the correct cutout.
            out = padded_map[x0 + dx0 : x1 + dx0, y0 + dy0 : y1 + dy0]
            assert out.shape[:2] == (micromap_size, micromap_size), (
                f"Cutout shape {out.shape} does not match micromap size {micromap_size}"
                f" after padding with ({dx0}, {dx1}), ({dy0}, {dy1})\n"
                f"Coordinates: x0={x0}, y0={y0}, x1={x1}, y1={y1}, center={center}\n"
                f"Map array shape: {map_arr.shape}, padded map shape: {padded_map.shape}"
            )
            return out, None

    if wcs is not None:
        return map_arr[x0:x1, y0:y1], wcs[x0:x1, y0:y1]

    return map_arr[x0:x1, y0:y1], None


def get_optimal_micromap_centers(map_arr, micromap_size, spacing=1):
    val_flag = np.isfinite(map_arr)

    # Diameter is maximum distance between valid pixels along one axis
    i_valid = np.argwhere(val_flag).T
    x_min, x_max = i_valid[1].min(), i_valid[1].max()
    safe_width = x_max - x_min
    # print(f"Array shape: {map_arr.shape}, micromap_size: {micromap_size}")
    # print(f"r_h: {r_h}, r_w: {r_w}, H: {H}, W: {W}")

    # Generate ceners in x-direction as regular steps, symmetric around center
    x_steps = np.arange(
        x_min
        + (safe_width % (spacing * micromap_size)) // 2
        + (spacing * micromap_size) // 2,
        x_max,
        spacing * micromap_size,
    )
    # p = np.array(np.meshgrid(*(steps,) * 2)).reshape(2, -1).T

    # print(f"x_steps: {x_steps}")

    # Loop through steps, starting from center, to fill in y-direction
    p = []
    for i, x_step in enumerate(x_steps):
        # Current column slice
        sl_col = slice(x_step - micromap_size // 2, x_step + micromap_size // 2)
        col = val_flag[:, sl_col]
        # Upper and lower boundaries for fitting a square:
        col_valids = [np.argwhere(c) for c in col.T if np.any(c)]
        # There can be columns with no valid pixels
        if not len(col_valids):
            # This happens when the mosaic has separate contiguous valid areas
            # with big gaps in between.
            continue
        y_min = max(c.min() for c in col_valids)
        y_max = min(c.max() for c in col_valids)

        # Total safe height to fit squares into
        safe_height = y_max - y_min
        # print(f"{i}: x_step: {x_step}, y_min: {y_min}, width: {row_width}, r_w: {r_w}")

        # Square centers in y-direction
        y_steps = np.arange(
            y_min
            + (safe_height % (spacing * micromap_size)) // 2
            + (spacing * micromap_size) // 2,
            y_max,
            spacing * micromap_size,
        )
        p += [[y, x_step] for y in y_steps]

    p = np.array(p)
    return np.round(p).astype(int)


def get_optimal_micromap_centers_circle(map_arr, micromap_size, spacing=1):
    val_flag = np.isfinite(map_arr)

    # Diameter is maximum distance between valid pixels along one axis
    i_valid = np.argwhere(val_flag).T
    r_h, r_w = (i_valid.max(axis=1) - i_valid.min(axis=1)) // 2
    H, W = map_arr.shape
    # print(f"Array shape: {map_arr.shape}, micromap_size: {micromap_size}")
    # print(f"r_h: {r_h}, r_w: {r_w}, H: {H}, W: {W}")

    # Generate ceners in x-direction as regular steps, symmetric around center
    x_steps = np.arange(
        (W % (spacing * micromap_size)) // 2 + (spacing * micromap_size) // 2,
        W,
        spacing * micromap_size,
    )
    # p = np.array(np.meshgrid(*(steps,) * 2)).reshape(2, -1).T

    # print(f"x_steps: {x_steps}")

    # Loop through steps, starting from center, to fill in y-direction
    p = []
    for i, x_step in enumerate(x_steps[len(x_steps) // 2 :]):
        # Number of row-squares in current x-direction
        n_row_squares = 2 * (i + int(len(x_steps) % 2 == 0)) + (len(x_steps) % 2)
        row_width = micromap_size * n_row_squares
        if row_width > 2 * r_w:
            # print(f"Stopping at i={i}, x_step={x_step}")
            break
        # Lowest safe point inside the circle to fit a square:
        try:
            y_min = np.ceil(r_h - np.sqrt(r_h**2 - (row_width // 2) ** 2))
        except RuntimeWarning as e:
            print(
                f"RuntimeWarning at i={i}, x_step={x_step}, row_width={row_width}, r_h={r_h}, r_w={r_w}"
            )
            raise e
        # Total safe height to fit squares into
        safe_height = 2 * (r_h - y_min)
        # print(f"{i}: x_step: {x_step}, y_min: {y_min}, width: {row_width}, r_w: {r_w}")
        # Square centers in y-direction
        y_steps = np.arange(
            (H - 2 * r_h) // 2
            + y_min
            + (safe_height % (spacing * micromap_size)) // 2
            + (spacing * micromap_size) // 2,
            H // 2 + r_h - y_min,
            spacing * micromap_size,
        )
        p += [[x_step, y] for y in y_steps]
        if not (i == 0 and len(x_steps) % 2 == 1):
            x_step_mirror = x_steps[len(x_steps) // 2 - i - int(len(x_steps) % 2 == 0)]
            p += [[x_step_mirror, y] for y in y_steps]

    p = np.array(p)

    # Keep those that are within the 'safe' radius
    # p = p[((p - r_map) ** 2).sum(axis=1) < (r_map - pad) ** 2]
    return np.round(p).astype(int)


def get_micromap_centers(map_arr, micromap_size, spacing=1, symmetric=False):
    val_flag = np.isfinite(map_arr)

    # Diameter is maximum sum along one axis
    r_map = np.max(np.sum(val_flag, axis=0)) // 2
    pad = np.ceil(np.sqrt(2) * micromap_size) / 2

    # Generate micromap centers
    steps = np.arange(
        ((map_arr.shape[-1] % micromap_size) // 2) if symmetric else 0,
        map_arr.shape[-1],
        spacing * micromap_size,
    )
    p = np.array(np.meshgrid(*(steps,) * 2)).reshape(2, -1).T

    # Keep those that are within the 'safe' radius
    p = p[((p - r_map) ** 2).sum(axis=1) < (r_map - pad) ** 2]
    return np.round(p).astype(int)


def get_micromaps(
    map_arr,
    micromap_size,
    spacing=1,
    centers=None,
    wcs=None,
    pbar=True,
    verbose=False,
    remove_nans=True,
    max_nan_fill_size=8,
):
    """
    Get micromaps from a map array.
    """
    # Get micromap centers
    if centers is None:
        centers = get_optimal_micromap_centers(map_arr, micromap_size, spacing)

    # Fill NaN pixels for small islands
    # map_arr = fill_nan_pixels(map_arr, max_size=max_nan_fill_size, early_stop=False)

    # Get micromaps
    micromaps = np.empty((centers.shape[0], micromap_size, micromap_size))
    wcss = []
    nan_flag = np.zeros(centers.shape[0], dtype=bool)
    # Make progress bar optional
    if pbar:
        pbar = tqdm(total=centers.shape[0], desc="Micromaps", unit="micromap")
    for i, center in enumerate(centers):
        # Get micromap cutout
        mm, microwcs = single_micromap_cutout(map_arr, micromap_size, center, wcs=wcs)
        mm = fill_nan_pixels(mm, max_size=max_nan_fill_size, early_stop=True)
        if np.any(np.isnan(mm)) and remove_nans:
            nan_flag[i] = True
        else:
            micromaps[i] = mm
            wcss.append(microwcs)
        # Update progress bar
        if pbar:
            pbar.update(1)

    if verbose:
        print(
            f"Found {nan_flag.sum()} micromaps with NaNs out of {centers.shape[0]} total."
        )
    if not remove_nans:
        if verbose:
            print("Returning all micromaps, including those with NaNs.")
        return micromaps, centers, (wcss if wcs is not None else None)

    return (
        micromaps[~nan_flag],
        centers[~nan_flag],
        (wcss if wcs is not None else None),
    )
