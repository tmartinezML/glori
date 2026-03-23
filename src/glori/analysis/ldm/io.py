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


def load_sweep_result(run_name, sampling_type="check", bdsf_empty_is_err=True):
    assert sampling_type in [
        "ICM",
        "LDM",
        "check",
    ], "sampling_type must be 'ICM', 'LDM', or 'check'"

    try:
        if sampling_type == "check":
            res_parent = sorted(
                # (paths.ANALYSIS_PARENT / "ldm").glob(f"*-sweep_*{run_name}")
                (paths.ANALYSIS_PARENT / "ldm").glob(f"*{run_name}*")
            )[-1]
        else:
            # res_parent = paths.ANALYSIS_PARENT / f"ldm/{sampling_type}-sweep_{run_name}"
            res_parent = sorted(
                (paths.ANALYSIS_PARENT / "ldm").glob(
                    f"{sampling_type}-sweep_*{run_name}"
                )
            )[-1]
    except IndexError:
        raise FileNotFoundError(
            f"No results found for run name: {run_name} (sampling_type={sampling_type})"
        )

    print(f"Loading results for <{run_name}>...")

    if not res_parent.exists():
        raise FileNotFoundError(f"Results directory not found: {res_parent}")

    # Load summary
    if (summary_npy := res_parent / "npy/extended_summary.npy").exists():
        summary = np.load(summary_npy, allow_pickle=True).item()
    elif (summary_json := res_parent / "summary.json").exists():
        summary = json.loads(summary_json.read_text())
    else:
        raise FileNotFoundError(
            f"No summary file found in {res_parent}. Expected 'npy/extended_summary.npy' or 'summary.json'."
        )

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
    elif bdsf_empty_is_err:
        raise FileNotFoundError(f"No BDSF results found in {res_parent / 'bdsf'}.")
    else:
        print(
            f"No BDSF results found in {res_parent / 'bdsf'}. Continuing without error."
        )
        prefix = "img"  # Default
    bdsf_results = [
        load_multistep_output(
            res_parent / f"bdsf/{prefix}-{img_idx:04d}", empty_is_err=bdsf_empty_is_err
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
                empty_is_err=bdsf_empty_is_err,
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
        print("No input catalogs found.")
        input_catalogs = None
        pos_masks = None

    # Load wcs and
    wcs = bdsf_results[0]["wcs"]
    srls = [result["catalogs"]["srl"] for result in bdsf_results]

    print("Done.")

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
        print(f"Output folder {out_folder} already exists. Deleting...")
        shutil.rmtree(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    sub_folder_names = "bdsf", "images", "npy"
    sub_folders = [out_folder / name for name in sub_folder_names]
    for sub_folder in sub_folders:
        sub_folder.mkdir(exist_ok=True)
    return sub_folders
