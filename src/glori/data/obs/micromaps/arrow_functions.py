from glori.data.obs.micromaps.arrow import logger
from glori.data.obs.micromaps.utils import (
    context_map_by_wcs,
    reduce_context_map,
    single_micromap_cutout,
    get_micromaps,
)
import tempfile
import pickle
from copy import deepcopy

import numpy as np
import pyarrow as pa
from astropy.io import fits
from astropy.wcs import WCS
from datasets.utils.tqdm import disable_progress_bars, enable_progress_bars

from glori.maps.map_utils import get_map_image

from functools import partial
import glori.settings.paths as paths

from datasets.arrow_writer import OptimizedTypedSequence
from datasets.features.features import Features


def get_samples_from_mosaic(
    mosaic_file,
    micromap_size,
    spacing,
    centers=None,
    splits=None,
):
    pointing = mosaic_file.parent.name
    # Get the map array
    map_arr, wcs = get_map_image(mosaic_file, get_wcs=True)
    if centers is not None:
        centers = np.array(centers)

    if centers is not None and splits is not None:
        assert len(centers) == len(
            splits
        ), f"Length of centers and splits must match. Got {len(centers)} and {len(splits)}."

    # Get micromaps and metadata
    logger.debug(f"Getting micromaps for {pointing}...")
    try:
        micromaps, centers, wcss = get_micromaps(
            map_arr, micromap_size, spacing, wcs=wcs, pbar=False, centers=centers
        )
    except Exception as e:
        logger.error(f"Error getting micromaps for {pointing}: {e}")
        return None

    # Get center coordinates in ra/dec from pixel coordinates
    center_coords = wcs.pixel_to_world(*centers.T)
    center_coords = np.stack([center_coords.ra.deg, center_coords.dec.deg]).T
    # Get the WCS information in string format
    wcss = np.array([wcs.to_header_string() for wcs in wcss])

    # Create template dict to store samples
    s = {
        "__key__": [],
        "npy": [],
        "center_coord": [],
        "wcs": [],
    }
    samples = {k: deepcopy(s) for k in ["train", "val", "test"]}
    for i in range(len(micromaps)):
        sample = {
            "__key__": f"{pointing}-{centers[i,0]}-{centers[i,1]}",
            "npy": micromaps[i].astype(np.float32),
            "center_coord": center_coords[i].astype(np.float32),
            "wcs": wcss[i].encode("utf-8"),
        }
        # Assign split if provided
        if splits is not None:
            split = splits[i]
        # If not, assign train, test or val with 0.8, 0.1, 0.1 proportions
        else:
            split = np.random.choice(["train", "val", "test"], p=[0.8, 0.1, 0.1])
        for k, v in sample.items():
            samples[split][k].append(v)

    # Convert to arrow table, make them None if empty
    for split in samples.keys():
        if len(samples[split]["__key__"]) == 0:
            samples[split] = None
        else:
            # Adapted from datasets.arrow_writer.write_batch
            cols = list(samples[split])
            inferred_features = Features()
            arrays = []
            for col in cols:
                col_values = samples[split][col]
                typed_sequence = OptimizedTypedSequence(col_values, col=col)
                arrays.append(pa.array(typed_sequence))
                inferred_features[col] = typed_sequence.get_inferred_type()
            schema = inferred_features.arrow_schema
            table = pa.Table.from_arrays(arrays, schema=schema)
            samples[split] = table

    del map_arr, wcs, micromaps, centers, wcss, center_coords, s
    return samples


def add_context_to_sample(sample, ctxt, mosaic, img_size, f_ctxt_size, f_downscale):
    """Extract context cutout for a single sample."""
    img_key = sample["__key__"]
    x, y = [int(s) for s in img_key.replace(f"{mosaic}-", "").split("-")]
    micro_ctxt = single_micromap_cutout(
        ctxt.transpose(1, 2, 0),
        int(img_size * f_ctxt_size),
        np.array([x, y]),
        fill_excess=(f_ctxt_size > 1),
    )[0].transpose(2, 0, 1)

    # Validate shape
    expected_shape = (int(img_size * f_ctxt_size),) * 2
    assert micro_ctxt.shape[-2:] == expected_shape, (
        f"Context shape {micro_ctxt.shape[-2:]} does not match expected "
        f"shape {expected_shape} for key {img_key}."
    )

    micro_ctxt_downscaled = reduce_context_map(
        micro_ctxt,
        f_downscale=f_downscale,
        input_scaled=False,
        scale_output=False,
    )

    ctxt_key = (
        "context_downscaled.npy"
        if (f_ctxt_size == 1 and f_downscale == 4)
        else f"context_downscaled_d={f_downscale}_f={f_ctxt_size}.npy"
    )
    sample[ctxt_key] = micro_ctxt_downscaled
    return sample


def add_context_process_mosaic(
    mosaic,
    enc_ids,
    encs_dset,
    cat,
    mosaic_dir,
    img_size,
    f_ctxt_size,
    f_downscale,
    debug,
):
    """Process a single mosaic ID."""
    # Filter datasets for this mosaic
    idxs = np.argwhere(enc_ids == mosaic).flatten()
    if len(idxs) == 0:
        logger.warning(f"No encodings found for mosaic {mosaic}. Skipping.")
        return None
    encs_subset = encs_dset.select(idxs)

    if debug:
        encs_subset = encs_subset.select(range(min(2, len(encs_subset))))
        logger.debug(f"Debug mode: limiting to first 2 samples for mosaic {mosaic}.")

    # Load mosaic WCS
    mosaic_file = mosaic_dir / f"{mosaic}/mosaic-blanked.fits"
    assert mosaic_file.exists(), f"Mosaic file for {mosaic} does not exist."
    with fits.open(mosaic_file) as hdul:
        wcs = WCS(hdul[0].header)

    ctxt, _ = context_map_by_wcs(wcs, cat, scale_output=False)

    # Use partial to bind parameters
    add_context_fn = partial(
        add_context_to_sample,
        ctxt=ctxt,
        mosaic=mosaic,
        img_size=img_size,
        f_ctxt_size=f_ctxt_size,
        f_downscale=f_downscale,
    )

    # Execute the mapping
    if not debug:
        disable_progress_bars()
    encs_subset = encs_subset.map(add_context_fn, num_proc=min(4, len(encs_subset)))
    if not debug:
        enable_progress_bars()

    return encs_subset
