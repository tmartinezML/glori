import json
import art
import torch
import numpy as np
from skimage.transform import rescale
import contextlib
import concurrent.futures
from datetime import datetime

import randomname
from glori.plotting.images import plot_image_grid
import glori.settings.paths as paths
from glori.data.trf.functional import catalog_context_pos_rescale
from glori.data.load import load_mosaic, load_fits_catalog, load_lotss_catalog
from glori.infra.logging import get_logger

from glori.analysis.bdsf_analysis import *
from glori.models.load import parse_lightning_ckpt
from glori.inference.swiit_sampler import SWIITSampler
from glori.data.obs.micromaps.utils import (
    context_map_by_wcs,
    get_model_image,
    reduce_context_map,
)

# Read first argument --debug for debug mode
debug_mode = len(sys.argv) > 1 and sys.argv[1] == "--debug"

# Limit Python to use only the first N CPUs
Ncpu = 16
os.sched_setaffinity(0, set(range(Ncpu)))

# Check the CPUs available to the process
print("CPUs available:", os.sched_getaffinity(0))

logger = get_logger("LDM-Resample")
logger.divider()
print(art.text2art("LDM Resample", font="cybermedium"))
logger.divider()

# Model settings
denoiser = "LDM-Denoiser-WnetCC-v4"
denoiser_ckpt = "best"
uncond_denoiser = "LDM-Denoiser-Uncond-128"
uncond_denoiser_ckpt = "best"
vae = "VQ-VAE-256"
sampling_type = "ICM"  # "LDM or "ICM"
ctxt_type = "cat"  # "model" or "cat"

# Sampling settings
latent_size = 128
f_vae = 4
image_size = latent_size * f_vae
guidance_strength = 0.2
timesteps = 25
f_inner = 1
f_ext_ctxt = 2
resample_first = False
drop_inpainting_ctxt_at = -1
do_inpainting = True
use_inpainting_replacement = True
catalog_mode = "combined"  # options: "combined", "separate"
scale_ctxt = False
save_intermediates = False
device = "cuda:1"

# Sampling blueprint settings
sampling_steps = (5, 5)
mask_coverage = 0.5
mask_overlap = int(mask_coverage * latent_size)
stride = latent_size - mask_overlap
img_mask_overlap = int(mask_coverage * image_size)
img_stride = image_size - img_mask_overlap
latent_map_size = tuple(latent_size + (s - 1) * stride for s in sampling_steps)
img_map_size = tuple(s * f_vae for s in latent_map_size)
latent_map_height, latent_map_width = latent_map_size
mosaics = None
batch_size = 8
batch_repeated = False
max_beam_arcsec = 6

# Experiment settings
seed = 69
noise_seed = 42
n_iter = 4
bdsf_workers = 8  # Number of parallel workers for BDSF analysis
do_bdsf = True
do_bdsf_originals = True  # Whether to run BDSF on the original cutouts as well (in addition to the sampled images)


# Settings for output folder
out_folder_lbl = ""
comments = ""

# Apply debug mode settings
if debug_mode:
    sampling_steps = (2, 2)
    latent_map_size = tuple(latent_size + (s - 1) * stride for s in sampling_steps)
    img_map_size = tuple(s * f_vae for s in latent_map_size)
    latent_map_height, latent_map_width = latent_map_size
    n_iter = 1
    batch_size = 2
    timesteps = 2

logger = get_logger(__name__)
if debug_mode:
    logger.setLevel("DEBUG")
logger.divider()
logger.debug("Debug mode is ON. Using reduced settings for quick testing.")

# Prepare output folders
# Save the output image
# Set output paths
resample_name = randomname.get_name() if not debug_mode else "debug"
logger.info(f"Resample name: {resample_name}")
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
out_folder = out_parent = paths.ANALYSIS_PARENT / (
    f"ldm/ICM-resample-{img_map_size[0]}x{img_map_size[1]}"
    f"{'-' + out_folder_lbl if out_folder_lbl is not None and len(out_folder_lbl) > 0 else ''}"
    f"-{timestamp}-{resample_name}"
)
if debug_mode:
    out_folder = out_parent = paths.ANALYSIS_PARENT / "ldm/ICM-resample-debug"
    if out_folder.exists():
        logger.debug(f"Debug output folder {out_folder} already exists. Deleting...")
        shutil.rmtree(out_folder)
out_folder.mkdir(parents=True, exist_ok=True)
sub_folder_names = "bdsf", "images", "npy"
sub_folders = bdsf_folder, img_folder, npy_folder = [
    out_folder / name for name in sub_folder_names
]
for sub_folder in sub_folders:
    sub_folder.mkdir(exist_ok=True)
if save_intermediates:
    (interm_folder := (out_folder / "intermediates")).mkdir(exist_ok=True)


# Define list of mosaics
logger.info("Gathering mosaics...")

# Gather all mosaics available
mosaic_list = sorted([p.name for p in paths.MOSAIC_DIR_DR3.glob("*") if p.is_dir()])

# If desired, filter mosaics by max beam size
if max_beam_arcsec is not None:
    pointing_info_df = pd.read_csv(
        paths.LOFAR_DATA_PARENT / "DR3_pointing_lookup.csv", index_col="mosaic"
    )
    good_pointings = np.array(
        pointing_info_df.index[
            pointing_info_df["Beam"] <= np.round(max_beam_arcsec / 3600, 5)
        ],
        dtype=np.str_,
    )
    select_idxs = np.argwhere([p in good_pointings for p in mosaic_list]).flatten()
    logger.info(
        f"Filtering mosaics by max beam {max_beam_arcsec} arcsec:"
        f" {len(mosaic_list) - len(select_idxs)} of {len(mosaic_list)} mosaics removed."
        f" Remaining mosaics: {len(select_idxs)}."
    )
    mosaic_list = [mosaic_list[i] for i in select_idxs]

# Define list of selected mosaics
match mosaics:

    # Specified indices:
    case [int(), *rest]:
        if batch_repeated:
            assert (
                len(mosaics) == n_iter
            ), f"Length of mosaics list {len(mosaics)} must match n_iter {n_iter}."
        else:
            assert (
                len(mosaics) == n_iter * batch_size
            ), f"Length of mosaics list {len(mosaics)} must match n_iter * batch_size {n_iter * batch_size}."
        mosaics = [mosaic_list[i] for i in mosaics]

    # Specified names:
    case [str(), *rest]:
        if batch_repeated:
            assert (
                len(mosaics) == n_iter
            ), f"Length of mosaics list {len(mosaics)} must match n_iter {n_iter}."
        else:
            assert (
                len(mosaics) == n_iter * batch_size
            ), f"Length of mosaics list {len(mosaics)} must match n_iter * batch_size {n_iter * batch_size}."
        for mosaic in mosaics:
            if mosaic not in mosaic_list:
                raise ValueError(f"Mosaic {mosaic} not found in available mosaics.")

    # No specification: Select randomly
    case None:
        np.random.seed(seed)
        n_samples = n_iter if batch_repeated else n_iter * batch_size
        mosaics = [np.random.choice(mosaic_list) for _ in range(n_samples)]

    # Invalid type
    case _:
        raise ValueError(f"Invalid mosaic type: {type(mosaics)}")


# Load the catalog
if ctxt_type == "cat":
    logger.info("Using catalog-based context. Loading catalog...")
    cat = load_lotss_catalog(
        select_cols=[
            "Source_Name",
            "RA",
            "DEC",
            "Total_flux",
            "Peak_flux",
            "Maj",
            # "Mosaic_ID",
        ]
    )

# Load the sampler
ckpt_filename = str(parse_lightning_ckpt(denoiser_ckpt, model_name=denoiser))
icm_sampler = SWIITSampler(
    denoiser=denoiser,
    denoiser_ckpt=denoiser_ckpt,
    uncond_denoiser=uncond_denoiser,
    uncond_denoiser_ckpt=uncond_denoiser_ckpt,
    vae=vae,
    device=device,
    image_size=image_size,
    latent_size=latent_size,
    div_stride=f_inner,
    f_ext_ctxt=f_ext_ctxt,
    mask_coverage=mask_coverage,
)


# Convenience function for getting the slice of the central cutout
def get_slices(img, cutout_size_px):
    match cutout_size_px:
        case int():
            cutout_size_px = (cutout_size_px, cutout_size_px)
        case (int(), int()) | [int(), int()]:
            pass
        case _:
            raise ValueError(
                f"Invalid cutout_size_px: {cutout_size_px} of type {type(cutout_size_px)}"
            )
    out = []
    for s in cutout_size_px:
        cby2, cpx = s // 2, img.shape[-1] // 2
        sl = slice(cpx - cby2, cpx + cby2)
        out.append(sl)

    return tuple(out)


logger.info(f"Selected mosaics: {mosaics}")
logger.info(f"Running {len(mosaics)} iterations.")

img_batches = []
img_cut_batches = []
for itr in range(n_iter):

    logger.divider(n_lines=2)
    logger.info(f"Iteration {itr + 1}/{n_iter}")

    logger.info(f"Loading mosaics for current iterations...")

    # Get current batch of mosaics
    mosaics_itr = (
        mosaics[itr : itr + 1]
        if batch_repeated
        else mosaics[itr * batch_size : (itr + 1) * batch_size]
    )

    # Prepare containers for batched data
    f_downscale_latent = image_size // latent_size
    img_cut_batch = np.zeros((batch_size, *img_map_size), dtype=np.float32)
    ctxt_red_batch = np.zeros(
        (
            batch_size,
            4,
            img_map_size[0] // f_downscale_latent,
            img_map_size[1] // f_downscale_latent,
        ),
        dtype=np.float32,
    )

    for i, mosaic in tqdm(
        enumerate(mosaics_itr), desc="Loading mosaics", total=len(mosaics_itr)
    ):
        img, wcs = load_mosaic(mosaic)
        sl1, sl2 = get_slices(img, img_map_size)
        img_cut, wcs_cut = img[sl1, sl2], wcs[sl1, sl2]

        match ctxt_type:
            case "cat":
                ctxt, ctxt_cat = context_map_by_wcs(
                    wcs_cut,
                    cat,
                    scale_output=False,
                )
                ctxt_red = reduce_context_map(
                    ctxt,
                    scale_output=scale_ctxt,
                    input_scaled=False,
                )
                # Scale location map to [-1, 1]
                ctxt_red[0] = ctxt_red[0] * 2 - 1

            case "model":
                logger.info("Using model-based context.")
                ctxt = get_model_image(mosaic, which="PyBDSF") * 1e3
                sl1, sl2 = get_slices(ctxt, img_map_size)
                ctxt = np.expand_dims(ctxt[sl1, sl2], axis=0)
                ctxt_red = rescale(ctxt, 1 / 4, anti_aliasing=False, channel_axis=0)
            case _:
                raise ValueError(f"Unknown context type: {ctxt_type}")

        img_cut_batch[i] = img_cut
        ctxt_red_batch[i] = ctxt_red

    if batch_repeated:
        # If batch is repeated, extract first entry.
        # Repetition is handled in the sampler
        img_cut_batch = img_cut_batch[0]
        ctxt_red_batch = ctxt_red_batch[0]

    # Store context sub-catalog
    ctxt_cat.to_parquet(out_folder / f"ctxt_cat_{itr + 1:04d}.parquet")

    # Set torch random state for reproducibility
    torch.manual_seed(noise_seed + itr)

    logger.info("Running LDM sampling...")
    sampling_output = icm_sampler.sample(
        batch_size=batch_size,
        ctxt_map=torch.from_numpy(ctxt_red_batch).to(torch.float32),
        guidance_strength=guidance_strength,
        save_intermediates=save_intermediates,
        timesteps=timesteps,
        drop_inpainting_ctxt_at=drop_inpainting_ctxt_at,
        do_inpainting=do_inpainting,
        use_inpainting_replacement=use_inpainting_replacement,
        catalog_mode=catalog_mode,
    )
    if save_intermediates:
        img_out, _, intermediates = sampling_output
    else:
        img_out, _ = sampling_output

    img_out_unsc = icm_sampler.scaler.inverse_scale(img_out)
    img_batches.append(img_out_unsc)
    img_cut_batches.append(img_cut_batch)

    # Save results of this iteration
    logger.divider()
    np.save(npy_folder / f"img_batch_{itr + 1:04d}.npy", img_out)
    np.save(npy_folder / f"samples_unscaled_{itr + 1:04d}.npy", img_out_unsc)
    np.save(
        npy_folder / f"context_{itr + 1:04d}.npy",
        ctxt_red_batch,
    )
    np.save(npy_folder / f"original_imgs_{itr + 1:04d}.npy", img_cut_batch)

    logger.info(f"Saved output images to {npy_folder}. Saving png images...")
    for img_idx, img in tqdm(
        enumerate(img_out),
        desc="Saving png images",
        total=len(img_out),
    ):
        # Save sampled image
        fig, _ = plot_image_grid([img], img_side_len=8)
        fig.savefig(img_folder / f"img_{itr + 1:04d}_{img_idx:04d}.png")
        plt.close(fig)

    # Save original
    if batch_repeated:
        mosaic = mosaics_itr[0]
        img_cut_batch = np.expand_dims(img_cut_batch, 0)

    for img_idx, orig in tqdm(
        enumerate(img_cut_batch),
        desc="Saving original png images",
        total=len(img_cut_batch),
    ):
        fig, _ = plot_image_grid([icm_sampler.scaler.scale(orig)], img_side_len=8)
        suffix = f"{mosaic}" if batch_repeated else f"{img_idx:04d}"
        fig.savefig(img_folder / f"img_original_{itr + 1:04d}_{suffix}.png")
        plt.close(fig)

    if save_intermediates:
        logger.info("Saving intermediates...")
        torch.save(
            intermediates,
            interm_folder / f"intermediates_{itr + 1:04d}_{mosaic}.pt",
        )
        rows, cols = (intermediates["quadrant_indices"].max(dim=0).values + 1).tolist()
        for batch_idx, imgs in tqdm(
            enumerate(intermediates["decoded_images"].transpose(1, 0)),
            desc="Saving intermediate png images",
            total=len(intermediates["decoded_images"]),
        ):
            fig, _ = plot_image_grid(
                imgs.squeeze(),
                n_rows=rows,
                n_cols=cols,
            )
            fig.savefig(
                interm_folder
                / f"decoded_intermediates_{itr + 1:04d}_{batch_idx:04d}_{mosaic}.png"
            )
            plt.close(fig)

# Make big img batch
big_img_batch = np.concatenate(img_batches, axis=0)
big_img_cut_batch = np.concatenate(img_cut_batches, axis=0)


def process_img(img_idx, img_batch, out_folder, file_prefix="img"):
    """Process a single batch with BDSF analysis"""
    try:
        # Extract single image from batch
        img = img_batch[img_idx].squeeze()

        # Run BDSF analysis with output suppressed
        with (
            contextlib.redirect_stdout(open(os.devnull, "w")),
            contextlib.redirect_stderr(open(os.devnull, "w")),
        ):
            result = tiered_bdsf_wrapper(img, quiet=True)

        # Save result
        save_multistep_output(
            result, f"{file_prefix}-{img_idx:04d}", out_parent=out_folder
        )

        return img_idx, True, None
    except Exception as e:
        return img_idx, False, str(e)


if do_bdsf:
    # Run tiered bdsf wrapper in parallel
    logger.divider()
    logger.info("Running BDSF analysis in parallel...")

    # Set number of workers - ThreadPoolExecutor can handle more workers since it's lighter
    max_workers = min(
        os.cpu_count(), bdsf_workers
    )  # Cap at 8 to avoid overwhelming the system
    logger.info(f"Using {max_workers} workers for BDSF analysis")

    # Run parallel processing with ThreadPoolExecutor
    results = {}
    img_indices = list(range(len(big_img_batch)))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_batch = {
            executor.submit(process_img, img_idx, big_img_batch, bdsf_folder): img_idx
            for img_idx in img_indices
        }

        # Process results with progress bar
        with tqdm(total=len(img_indices), desc="BDSF Analysis") as pbar:
            for future in concurrent.futures.as_completed(future_to_batch):
                result = future.result()
                img_idx, success, error = result
                results[img_idx] = result
                pbar.update(1)

                # Log any errors
                if not success:
                    logger.warning(f"Image {img_idx} failed: {error}")

    # Get results into correct order
    results = [results[i] for i in range(len(img_indices))]

    # Collect successful results
    successful_images = [r[0] for r in results if r[1]]
    failed_images = [r[0] for r in results if not r[1]]

    logger.info(
        f"BDSF analysis complete: {len(successful_images)} successful, {len(failed_images)} failed"
    )


if do_bdsf_originals:
    logger.divider()
    logger.info("Running BDSF analysis on original cutouts...")

    # Run parallel processing with ThreadPoolExecutor
    original_results = {}
    original_img_indices = list(range(len(big_img_cut_batch)))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_batch = {
            executor.submit(
                process_img,
                img_idx,
                big_img_cut_batch,
                bdsf_folder,
                file_prefix="original",
            ): img_idx
            for img_idx in original_img_indices
        }

        # Process results with progress bar
        with tqdm(
            total=len(original_img_indices), desc="BDSF Analysis on Originals"
        ) as pbar:
            for future in concurrent.futures.as_completed(future_to_batch):
                result = future.result()
                img_idx, success, error = result
                original_results[img_idx] = result
                pbar.update(1)

                # Log any errors
                if not success:
                    logger.warning(f"Original image {img_idx} failed: {error}")

    # Get results into correct order
    original_results = [original_results[i] for i in range(len(original_img_indices))]

    # Collect successful results
    successful_original_images = [r[0] for r in original_results if r[1]]
    failed_original_images = [r[0] for r in original_results if not r[1]]

    logger.info(
        f"BDSF analysis on originals complete: {len(successful_original_images)} successful, {len(failed_original_images)} failed"
    )


logger.info("Saving summary...")
summary = {}
summary.update(
    dict(
        denoiser=denoiser,
        denoiser_ckpt=denoiser_ckpt,
        denoiser_ckpt_filename=ckpt_filename,
        uncond_denoiser=uncond_denoiser,
        uncond_denoiser_ckpt=uncond_denoiser_ckpt,
        vae=vae,
        device=device,
        image_size=image_size,
        latent_size=latent_size,
        quadrant_size=sampling_steps,
        batch_size=batch_size,
        n_iter=n_iter,
        cutout_size_px=img_map_size,
        context_type=ctxt_type,
        out_folder_lbl=out_folder_lbl,
        guidance_strength=guidance_strength,
        f_inner=f_inner,
        f_ext_ctxt=f_ext_ctxt,
        drop_inpainting_ctxt_at=drop_inpainting_ctxt_at,
        use_latent_ctxt=do_inpainting,
        use_inpainting_replacement=use_inpainting_replacement,
        catalog_mode=catalog_mode,
        save_intermediates=save_intermediates,
        do_bdsf=do_bdsf,
        bdsf_workers=bdsf_workers if do_bdsf else None,
        total_images=len(big_img_batch) if do_bdsf else None,
        successful_images=len(successful_images) if do_bdsf else None,
        failed_images=len(failed_images) if do_bdsf else None,
        do_bdsf_originals=do_bdsf_originals,
        total_original_images=len(img_cut_batch) if do_bdsf_originals else None,
        successful_original_images=(
            len(successful_original_images) if do_bdsf_originals else None
        ),
        failed_original_images=(
            len(failed_original_images) if do_bdsf_originals else None
        ),
        comments=comments,
    )
)
with open(out_folder / "summary.json", "w") as f:
    json.dump(summary, f, indent=4)
np.save(npy_folder / "extended_summary.npy", summary)

logger.info(f"Results saved to: {out_folder}")
logger.divider()
