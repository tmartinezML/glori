import os
import shutil
import psutil
import concurrent.futures
from datetime import datetime
from functools import partial

import json
import pandas as pd
import numpy as np
import skimage.measure as skm
from tqdm import tqdm, trange
from webdataset import TarWriter

import glori.settings.paths as paths
from glori.infra.logging import get_logger, add_file_handler
from . import prototypes_utils as putil
from .cutouts import cutout_from_catalog
from glori.data.load import load_lotss_catalog
from glori.data.trf.segment import smooth_mask
from glori.data.trf.image_utils import apply_restoring_beam
from glori.data.trf.functional import minmax_scale_batch

logger = get_logger(__name__)


def log_memory_usage(logger, tag=""):
    """Log memory usage with an optional tag to identify the location."""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024**2  # Convert to MB
    logger.info(f"Memory Usage {tag}: {memory_mb:.2f} MB")


# Wrapper function for creating model cuts, used for parallelization
def extract_model(i, sub_cat):
    try:
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / 1024 / 1024

        model, _ = cutout_from_catalog(
            sub_cat,
            i,
            parent_dir=paths.LOFAR_DATA_PARENT / "pointings",
            file_name="images/image_full_ampphase_di_m.NS.int.model.fits",
            mask_nan=False,
            size_px=256,
            opt_c=True,
        )

        end_mem = process.memory_info().rss / 1024 / 1024
        mem_diff = end_mem - start_mem

        if mem_diff > 10000:  # Log if memory increased by more than 100MB
            logger.warning(
                f"Large memory increase in worker {os.getpid()}, source {sub_cat.iloc[i].Source_Name}: {mem_diff:.2f} MB"
            )

    except Exception as e:
        logger.error(f"Error extracting model for index {i}: {e}")
        return i, str(e)

    return i, model


def create_webdataset(
    dset_name,
    override=False,
    extract_cutouts=True,
    SNR_threshold=5,
    mask_threshold=5e-5,
    debug=False,
    nsrc_debug=100,
    out_parent=paths.LOFAR_DATA_PARENT / "model_prototypes",
    res_cat=paths.LOFAR_RES_CAT,
    cmp_cat=paths.LOFAR_DATA_PARENT / "combined-components-v1.1.fits",
    pointing_parent=paths.LOFAR_DATA_PARENT / "pointings",
):
    # Handle output directory
    out_path = out_parent / dset_name
    check_existing = False
    if override and not extract_cutouts:
        logger.warning(
            "'override' is True and 'extract_cutouts' is False. Aborting for safety."
        )
        return

    if out_path.exists():
        if override:
            logger.info(f"Removing existing files in \n\t{out_path}")
            shutil.rmtree(out_path)
            out_path.mkdir(parents=True, exist_ok=True)
        elif not extract_cutouts:
            check_existing = True

    # Prepare summary, will be saved to json at the end
    summary = {}
    summary.update(
        dict(
            out_path=str(out_path),
            SNR_threshold=SNR_threshold,
            mask_threshold=mask_threshold,
            cutouts_extracted=extract_cutouts,
            override=override,
        )
    )

    # Prepare log output
    add_file_handler(logger, out_path / "creation.log")
    logger.divider(n_lines=3)
    logger.info(f"Creating webdataset at {out_path}")

    # Add memory usage at the start
    log_memory_usage(logger, "Start")

    # This will be used whenever tqdm is called
    tqdm_kw = dict(dynamic_ncols=True, colour="green")

    # Load catalogs
    logger.info(f"Loading resolved sources catalog \n\t<{res_cat}>...")
    res_cat = load_lotss_catalog(path=res_cat, select_cols=None)

    logger.info(f"Loading component catalog \n\t<{cmp_cat}>...")
    cmp_cat = load_lotss_catalog(
        path=cmp_cat, select_cols=["Component_Name", "RA", "DEC", "Parent_Source"]
    )

    if extract_cutouts:
        # Filter for pointings that have existing model files loaded
        all_pointings = sorted(list(pointing_parent.glob("*")))
        logger.info(f"Total pointings found: {len(all_pointings)}")
        flag_dict = {
            p.name: (p / "images").exists()
            for p in tqdm(all_pointings, desc="Checking pointings", **tqdm_kw)
        }
        flag = np.array([flag_dict[m] for m in res_cat["Mosaic_ID"]])
        sub_cat = res_cat[flag].reset_index(drop=True)
        logger.info(
            f"Catalog filtered to {len(sub_cat):_} sources with valid model data."
        )

        # Further filter out sources that are already in the output dataset
        if check_existing:
            logger.info("Filtering out existing sources in dataset.")
            try:
                prv_cat = pd.read_parquet(out_path / "initial_catalog.parquet")
                prv_sources = set(prv_cat["Source_Name"])
                sub_cat = sub_cat[~sub_cat["Source_Name"].isin(prv_sources)]
                logger.info(f"Filtered out {len(prv_sources):_} existing sources.")
                if len(sub_cat) == 0:
                    logger.info("No new sources to process. Exiting.")
                    return
            except Exception as e:
                logger.error(f"Error loading previous catalog: {e}")

        if debug:
            logger.info(f"Debug mode: limiting to {nsrc_debug} sources.")
            sub_cat = sub_cat.iloc[:nsrc_debug]

        # Update summary
        summary.update(
            dict(
                initial_n_src=len(sub_cat),
                n_pointings=len(all_pointings),
                n_valid_pointings=int(flag.sum()),
            )
        )

        # Make pre-selection based on SNR
        SNR_tot = sub_cat.Total_flux / sub_cat.E_Total_flux
        filter_flag = SNR_tot.values >= SNR_threshold
        sub_cat = sub_cat[filter_flag]
        init_cat = sub_cat.copy()
        n_src = len(sub_cat)
        logger.info(f"Selected {n_src:_} sources with SNR >= {SNR_threshold}.")

        logger.divider()

        # Run extraction of model cutouts in parallel
        log_memory_usage(logger, "Before model extraction")
        logger.info("Extracting model cutouts in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=512) as executor:
            model_cuts = list(
                tqdm(
                    executor.map(partial(extract_model, sub_cat=sub_cat), range(n_src)),
                    total=n_src,
                    desc="Model cutouts",
                    smoothing=0,
                    **tqdm_kw,
                )
            )

        log_memory_usage(logger, "After model extraction")

        # Filter out any errors
        ii = [i for i, cut in model_cuts if not isinstance(cut, str)]
        errors = {
            sub_cat.iloc[i].Source_Name: cut
            for i, cut in model_cuts
            if isinstance(cut, str)
        }
        model_cuts = [cut for _, cut in model_cuts if not isinstance(cut, str)]
        logger.info(
            f"Successfully extracted {len(model_cuts):_} model cutouts. Errors for {len(errors)} sources."
        )
        summary.update(
            dict(
                n_extracted_models=len(model_cuts),
                n_errors=len(errors),
                extraction_errors=errors,
            )
        )
        sub_cat = sub_cat.iloc[ii].reset_index(drop=True)

    else:
        logger.info("Loading existing cutouts from file...")
        model_cuts = np.array(
            putil.load_by_extension("model.npy", out_path / "images.tar")
        )
        init_cat = pd.read_parquet(out_path / "initial_catalog.parquet")
        final_cat_prv = pd.read_parquet(out_path / "catalog.parquet")
        sub_cat = init_cat[
            init_cat.Source_Name.isin(final_cat_prv.Source_Name)
        ].reset_index(drop=True)
        assert len(model_cuts) == len(sub_cat), (
            f"Number of model cutouts does not match catalog entries: "
            f"{len(model_cuts)} != {len(sub_cat)}"
        )
        del final_cat_prv
        logger.info(f"Loaded {len(model_cuts):_} model cutouts.")

    logger.divider()

    # Make masks based on thresholding the model images
    logger.info("Making masks...")
    model_cuts = np.clip(np.array(model_cuts), 0, None)
    model_cuts_beam = apply_restoring_beam(model_cuts)
    masks = model_cuts_beam > mask_threshold
    masks = np.array(
        [smooth_mask(mask) for mask in tqdm(masks, desc="Smoothing masks", **tqdm_kw)]
    )

    # Associate islands with components to produce clean masks
    logger.info("Associating components...")
    profiles = []
    clean_masks = np.empty_like(masks)
    # Loop through all selected images to make clean masks
    for i in tqdm(range(len(masks)), desc="Making clean masks...", **tqdm_kw):

        clean_mask, profile = putil.make_clean_mask(
            masks[i], sub_cat.iloc[i], cmp_cat=cmp_cat
        )
        profiles.append(profile)
        clean_masks[i] = clean_mask

    # In some cases there will be no good island, those are 'problems':
    # (Either host is not on any masked island, or all masks are contaminated)
    problem_flag = clean_masks.sum(axis=(1, 2)) == 0
    summary.update(
        dict(
            n_problems=int(problem_flag.sum()),
            problem_sources=sub_cat.Source_Name[problem_flag].tolist(),
        )
    )
    logger.info(f"Identified {problem_flag.sum():_} sources with no valid masks.")
    # Remove problematic sources
    clean_masks = clean_masks[~problem_flag]
    model_cuts = model_cuts[~problem_flag] * clean_masks
    # model_cuts_beam = model_cuts_beam[~problem_flag] * clean_masks
    sub_cat = sub_cat[~problem_flag].reset_index(drop=True)
    n_src = len(sub_cat)
    logger.info(f"Final sample size: {n_src:_} sources.")
    summary.update(dict(final_n_src=n_src))

    logger.divider()

    # Center model prototypes
    logger.info("Centering model prototypes...")
    cntsrc = [
        putil.center_source(model_cuts[i], clean_masks[i])
        for i in trange(n_src, desc="Centering Models")
    ]
    model_cuts = np.array([c[0] for c in cntsrc])
    clean_masks = np.array([c[1] for c in cntsrc])
    circle_radii = np.array([c[2] for c in cntsrc])
    model_cuts_beam = apply_restoring_beam(model_cuts)

    # Determine sizes
    logger.info("Determining sizes...")
    # Major and Minor axis as determined from central image moments.
    # The pre-factor is from conversion to FWHM, for consistency with LoTSS catalog.
    # The sqrt comes from the definition of the moments.
    maj_min_img = (
        2
        * np.sqrt(2 * np.log(2))
        * np.sqrt(
            np.array(
                [
                    skm.inertia_tensor_eigvals(x * m)
                    for x, m in tqdm(
                        zip(model_cuts_beam, clean_masks),
                        desc="Image moments",
                        total=n_src,
                        **tqdm_kw,
                    )
                ]
            ).T
        )
    )
    # Size as defined in previous prototypes version: largest distance between mask
    # pixels. Probably still useful.
    mask_sizes = np.array(
        [
            # regionprops_table(m, properties=('axis_major_length',))
            skm.regionprops(m)[0].feret_diameter_max
            for m in tqdm(
                clean_masks.astype(int), desc="Mask sizes", total=n_src, **tqdm_kw
            )
        ]
    )

    # Add properties to catalog
    logger.info("Adding properties to catalog...")
    sub_cat["pt_maj_axis"] = maj_min_img[0]
    sub_cat["pt_min_axis"] = maj_min_img[1]
    sub_cat["pt_mask_size"] = mask_sizes
    sub_cat["pt_circle_radius"] = circle_radii
    sub_cat["pt_total_flux"] = model_cuts.sum(axis=(1, 2))
    sub_cat["pt_total_flux_minmax"] = minmax_scale_batch(model_cuts).sum(axis=(1, 2))
    sub_cat["pt_peak_flux"] = model_cuts.max(axis=(1, 2))
    sub_cat["pt_peak_flux_beam"] = model_cuts_beam.max(axis=(1, 2))

    logger.divider()
    # Save results
    logger.info("Saving results...")
    # Image data to tar file
    samples = [
        {
            "__key__": sub_cat.Source_Name[i].replace(".", "p"),
            "model.npy": model_cuts[i].astype(np.float32),
            "model_beam.npy": model_cuts_beam[i].astype(np.float32),
            "mask.npy": clean_masks[i].astype(np.uint8),
            "maj_min.npy": maj_min_img[:, i].astype(np.float32),
            "mask_size.npy": np.array(mask_sizes.astype(np.float32)[i]),
            "circle_radius.npy": np.array(circle_radii.astype(np.float32)[i]),
        }
        for i in trange(n_src, desc="Collecting results")
    ]
    putil.append_to_tar(out_path / "images.tar", samples, keep_backup=check_existing)
    # Catalogs to parquet files
    putil.append_to_parquet(out_path / "catalog.parquet", sub_cat)
    putil.append_to_parquet(out_path / "initial_catalog.parquet", init_cat)

    logger.info(f"Webdataset saved to {out_path}.")

    logger.divider()
    # Save summary to json
    with open(out_path / "summary.json", "a" if check_existing else "w") as f:
        d = {datetime.now().strftime("%Y-%m-%d %H:%M"): summary}
        if check_existing:
            # Read data
            with open(out_path / "summary.json", "r") as fr:
                existing_summary = json.load(fr)
            # Update with new data
            d.update(existing_summary)
        json.dump(d, f, indent=4)

    logger.info(f"Summary saved to {out_path / 'summary.json'}")
    logger.info("Done.")
    logger.divider(n_lines=3)


if __name__ == "__main__":
    create_webdataset(
        "model_prototypes",
        extract_cutouts=True,
        override=True,
        debug=False,
        nsrc_debug=20,
    )
