import numpy as np
import pandas as pd
import shutil
import json
from tqdm import trange
from itertools import chain


from ..bdsf_analysis import (
    load_multistep_output,
)
import glori.settings.paths as paths
from glori.infra.logging import get_logger

logger = get_logger(__name__)


def load_sampling_result(run_name, parent=None, missing_is_error={}):

    # Identify results directory based on run name and parent directory
    if parent is None:
        parent = paths.ANALYSIS_PARENT / "ldm"
    parent_candidates = sorted(parent.glob(f"*{run_name}*"))
    if len(parent_candidates) == 0:
        raise FileNotFoundError(
            f"No results found for run name: {run_name} in {parent}."
        )
    res_parent = parent_candidates[-1]
    if not res_parent.exists():
        raise FileNotFoundError(f"Results directory not found: {res_parent}")
    del parent, parent_candidates

    logger.info(f"Loading results for <{run_name}>...")

    # Check entries in missing_is_error dict
    valid_keys = {"summary", "bdsf", "original_bdsf"}
    if isinstance(missing_is_error, bool):
        missing_is_error = {key: missing_is_error for key in valid_keys}
    if any(k not in valid_keys for k in missing_is_error.keys()):
        raise ValueError(
            f"Invalid keys in missing_is_error: {missing_is_error.keys()}. Valid keys are: {valid_keys}."
        )

    # Load summary
    if (summary_npy := res_parent / "npy/extended_summary.npy").exists():
        summary = np.load(summary_npy, allow_pickle=True).item()
    elif (summary_json := res_parent / "summary.json").exists():
        summary = json.loads(summary_json.read_text())
    elif missing_is_error.get("summary", True):
        raise FileNotFoundError(
            f"No summary file found in {res_parent}. Expected 'npy/extended_summary.npy' or 'summary.json'."
        )
    else:
        logger.warning(f"No summary file found in {res_parent}.")
        summary = {}

    # Load sampled images
    img_batch = np.concatenate(
        [np.load(f) for f in sorted((res_parent / "npy").glob("img_batch_*.npy"))]
    )

    # Load context maps
    patterns = ["ctxt_map_*.npy", "context_*.npy"]
    ctxt_map = np.concatenate(
        [
            np.load(f)
            for f in sorted(
                chain.from_iterable(
                    (res_parent / "npy").glob(pattern) for pattern in patterns
                )
            )
        ]
    )

    # Reshape context map if needed
    if ctxt_map.ndim == 3:
        # We have n context vectors concatenated along channel axis. We want to add
        # batch dimension and repeat each one for the batch size
        n_chan = 4
        batch_size = summary["batch_size"]
        ctxt_stack = np.split(ctxt_map, ctxt_map.shape[0] // n_chan, axis=0)
        ctxt_batch = []
        for c in ctxt_stack:
            c_b = np.expand_dims(c, 0).repeat(batch_size, axis=0)
            ctxt_batch.append(c_b)
        ctxt_map = np.concatenate(ctxt_batch, axis=0)

    # Load originals if available
    orig_files = sorted((res_parent / "npy").glob("original_imgs_*.npy"))
    original_imgs = None
    if len(orig_files):
        original_imgs = np.stack([np.load(f) for f in orig_files])

    # Load BDSF results
    if len(list((res_parent / "bdsf").glob("batch-*"))):
        prefix = "batch"
    elif len(list((res_parent / "bdsf").glob("img-*"))):
        prefix = "img"
    elif missing_is_error.get("bdsf", True):
        raise FileNotFoundError(f"No BDSF results found in {res_parent / 'bdsf'}.")
    else:
        logger.info(
            f"No BDSF results found in {res_parent / 'bdsf'}. Continuing without error."
        )
        prefix = "img"  # Default
    bdsf_results = [
        load_multistep_output(
            res_parent / f"bdsf/{prefix}-{img_idx:04d}",
            empty_is_err=missing_is_error.get("bdsf", True),
        )
        for img_idx in trange(len(img_batch), desc="Loading BDSF results")
    ]

    # Load BDSF results of original images if available
    orig_bdsf_results = []
    prefix = "original"
    if len(list((res_parent / "bdsf").glob(f"{prefix}-*"))):
        orig_bdsf_results = [
            load_multistep_output(
                res_parent / f"bdsf/{prefix}-{img_idx:04d}",
                empty_is_err=missing_is_error.get("original_bdsf", True),
            )
            for img_idx in trange(len(img_batch), desc="Loading original BDSF results")
        ]

    # Load context catalogs if available
    if any(res_parent.glob("ctxt_cat_*.parquet")):
        input_catalogs = pd.concat(
            [pd.read_parquet(f) for f in sorted(res_parent.glob("ctxt_cat_*.parquet"))],
            ignore_index=True,
        )
        try:
            pos_masks = np.concatenate(
                [
                    np.load(f)
                    for f in sorted((res_parent / "npy").glob("pos_mask_*.npy"))
                ]
            )
        except ValueError:
            pos_masks = None
    else:
        logger.info("No input catalogs found.")
        input_catalogs = None
        pos_masks = None

    # Load wcs and srls
    wcs = bdsf_results[0]["wcs"]
    srls = [result["catalogs"]["srl"] for result in bdsf_results]

    logger.info("Done.")

    return {
        "img_batch": img_batch,
        "ctxt_map": ctxt_map,
        "bdsf_results": bdsf_results,
        "orig_bdsf_results": orig_bdsf_results,
        "summary": summary,
        "wcs": wcs,
        "srls": srls,
        "input_catalogs": input_catalogs,
        "pos_masks": pos_masks,
        "original_imgs": original_imgs,
        "res_parent": res_parent,
    }


def prepare_directory(out_folder, override=False):
    if out_folder.exists() and override:
        logger.info(f"Output folder {out_folder} already exists. Deleting...")
        shutil.rmtree(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    sub_folder_names = "bdsf", "images", "npy"
    sub_folders = [out_folder / name for name in sub_folder_names]
    for sub_folder in sub_folders:
        sub_folder.mkdir(exist_ok=True)
    return sub_folders
