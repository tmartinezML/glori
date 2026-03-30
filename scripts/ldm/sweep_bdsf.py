import os
import art
import json
import contextlib
import concurrent.futures
from functools import partial
from math import prod
from datetime import datetime

import torch
import randomname
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import rescale

import utils.paths as paths
import data.trf.transforms as T
from data.utils import load_lotss_catalog
from glori.inference.swiit_sampler import SWIITSampler
from glori.inference.ldm_sampler import LDMSampler
from analysis.bdsf_analysis import tiered_bdsf_wrapper, save_multistep_output
from utils.my_logging import get_logger, pretty_print_config
from plotting.images import plot_image_grid
from data.obs.models import recreate_model_image


# Limit Python to use only the first N CPUs
Ncpu = 16
os.sched_setaffinity(0, set(range(Ncpu)))

# Check the CPUs available to the process
print("CPUs available:", os.sched_getaffinity(0))

logger = get_logger("LDM-Sweep")
logger.divider()
print(art.text2art("LDM Sampling Sweep", font="cybermedium"))
logger.divider()

# Model settings
denoiser = "LDM-Denoiser-WnetCC"
denoiser_ckpt = "best"
uncond_denoiser = "LDM-Denoiser-Uncond-128"
uncond_denoiser_ckpt = "best-last-10"
vae = "VQ-VAE-256"
sampling_type = "ICM"  # "LDM or "ICM"
ctxt_type = "cat"  # "model" or "cat"

# Settings for the sweep
size_quadrants = (5, 5)
src_per_q_len = 2
batch_size = 8
n_iter = 4
bdsf_workers = 8
shuffle_positions = True

# Sampling settings
device = "cuda:2"
guidance_strength = 0.2
image_size = 512
latent_size = 128
f_inner = 1
f_ext_ctxt = (2, 2)

# For even sampling across range of selected quantity
qty = "Peak_flux"  # Total_flux, Peak_flux, Maj
qty_bin_args = 0.8, 1e5, 200  # Min, Max, N_bins
scale_ctxt = False  # Whether to scale context values
min_fpeak = 1  # Min. for sampling. Will be ignored if qty == 'Peak_flux'

# Comments for summary file
comments = (
    "Unweighted training. "
    "Source positions shuffled. "
    "Used ICMSampler v2, after removing sigma_max=3 from class constructor."
)

# Some assertions about the settings
assert sampling_type in (
    "LDM",
    "ICM",
), f"sampling_type must be 'LDM' or 'ICM', got {sampling_type}"
assert ctxt_type in (
    "model",
    "cat",
), f"ctxt_type must be 'model' or 'cat', got {ctxt_type}"
assert qty in (
    "Peak_flux",
    "Total_flux",
    "Maj",
), f"qty must be 'Peak_flux', 'Total_flux', or 'Maj', got {qty}"

# Set output paths
sweep_name = randomname.get_name()
# sweep_name = "debug"
logger.info(f"Sweep name: {sweep_name}")
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
out_folder = paths.ANALYSIS_PARENT / (
    f"ldm/{sampling_type}-sweep_{timestamp}_{sweep_name}"
)
out_folder.mkdir(parents=True, exist_ok=True)
sub_folder_names = "bdsf", "images", "npy"
sub_folders = bdsf_folder, img_folder, npy_folder = [
    out_folder / name for name in sub_folder_names
]
for sub_folder in sub_folders:
    sub_folder.mkdir(exist_ok=True)


# Load sampler
if sampling_type == "ICM":
    logger.info("Loading ICM sampler...")
    sampler = SWIITSampler(
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
    )
elif sampling_type == "LDM":
    logger.info("Loading LDM sampler...")
    sampler = LDMSampler(
        denoiser=denoiser,
        denoiser_ckpt=denoiser_ckpt,
        uncond_denoiser=uncond_denoiser,
        uncond_denoiser_ckpt=uncond_denoiser_ckpt,
        vae=vae,
        device=device,
        image_size=image_size,
        latent_size=latent_size,
    )


# Load DR3 cat
logger.info("Loading DR3 catalog...")
dr3_cat = load_lotss_catalog(
    select_cols=[
        "Source_Name",
        "RA",
        "DEC",
        "Total_flux",
        "Peak_flux",
        "Maj",
        "Mosaic_ID",
    ]
)

# Define log-spaced bins for balanced sampling:
logger.info("Defining log-spaced bins for balanced sampling...")
# Get approximately equivalent amount of image data if LDM sampling:
if sampling_type == "LDM":
    f_batch = int(np.ceil(prod(size_quadrants) / 4))
    logger.info(
        f"LDM sampling: converting quadrant size {size_quadrants} to {f_batch} images."
    )
    logger.info(f"Changing batch size from {batch_size} to {f_batch * batch_size}.")
    size_quadrants = (2, 2)
    batch_size *= f_batch
n_per_img = prod(size_quadrants) * src_per_q_len**2
n_per_it = n_per_img * batch_size
mn, mx, nb = qty_bin_args
if qty == "Peak_flux":
    mn = max(mn, min_fpeak)
qty_bins = np.geomspace(mn, mx, min(nb, n_per_it))
fpeak_flag = dr3_cat["Peak_flux"] >= min_fpeak
dr3_cat_fpeak = dr3_cat[fpeak_flag].reset_index(drop=True)
qty_bin_idxs = np.digitize(dr3_cat_fpeak[qty].values, qty_bins)

img_batches = []
for itr in range(n_iter):
    # Number of samples
    # Make ctxt map
    logger.divider(n_lines=2)
    logger.info(f"Iteration {itr + 1}/{n_iter}: Making context map...")

    # Fill with values sampled from dr3 catalog,
    # ensuring balanced sampling across log-bins in peak flux.
    logger.info("Sampling context values from DR3 catalog...")

    # Pick equal number of samples from every bin:
    n_samples = n_per_it
    unique_bins = np.unique(qty_bin_idxs)
    n_per_bin = max(int(np.ceil(n_samples / len(unique_bins))), 1)
    logger.info(
        f"Sampling {n_samples} sources for context from {len(unique_bins)} bins,"
        f" {n_per_bin} per bin."
    )
    idx = []
    for b in unique_bins:
        bin_idxs = np.argwhere(qty_bin_idxs == b).ravel()
        chosen = np.random.choice(
            bin_idxs, size=n_per_bin, replace=(len(bin_idxs) <= n_per_bin)
        )
        idx.extend(chosen.tolist())
    assert len(idx) >= n_samples, (
        f"Not enough samples drawn ({len(idx)}) to fill batch ({n_samples}).\n"
        f"Numbers: n_per_it={n_per_it}, unique_bins={len(unique_bins)},"
        f" n_per_bin={n_per_bin}"
    )

    idx = torch.tensor(idx)[torch.randperm(len(idx))][:n_samples]
    # Sort by quantity
    if not shuffle_positions:
        idx = idx[torch.argsort(torch.tensor(dr3_cat_fpeak.iloc[idx][qty].values))]

    # Make context for catalog-conditioned model
    if ctxt_type == "cat":
        cols = ["Total_flux", "Peak_flux", "Maj"]
        ctxt_vals = (
            torch.tensor(
                [[dr3_cat_fpeak.iloc[i.item()][col] for col in cols] for i in idx]
            )
            .reshape(batch_size, n_per_img, 3)
            .permute(0, 2, 1)
            .to(torch.float32)
        )

        # Dummy map:
        ctxt_map = (
            SWIITSampler.explorer_context_map(
                np.ones((prod(size_quadrants) * src_per_q_len**2,)),
                1,
                1,
                size_quadrants=size_quadrants,
                latent_size=latent_size,
                logger=logger,
                i_product=None,
            )
            .unsqueeze(0)
            .repeat(batch_size, 1, 1, 1)
        )
        # Fill:
        ctxt_map[:, 1:, ctxt_map[0, 0] > 0] = ctxt_vals

        if scale_ctxt:
            sc_trf = T.make_catalog_context_value_scale()
            ctxt_map = sc_trf(ctxt_map)

    # Make context for model-image-conditioned model
    elif ctxt_type == "model":
        comp_cat = load_lotss_catalog(
            paths.LOFAR_DATA_PARENT / "LoTSS_DR3_v0.1_srl.fits",
            select_cols=["RA", "DEC", "Peak_flux", "Maj", "Min", "Source_Name", "PA"],
        )
        pos_mask = (
            SWIITSampler.explorer_context_map(
                np.ones((prod(size_quadrants) * src_per_q_len**2,)),
                1,
                1,
                size_quadrants=size_quadrants,
                latent_size=latent_size,
                logger=logger,
                i_product=None,
            )[0]
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        i_pos = torch.argwhere(pos_mask > 0)
        ctxt_map = torch.zeros_like(pos_mask)

        # Make model images in parallel
        model_imgs = []

        def process_idx(i):
            return recreate_model_image(
                dr3_cat_fpeak.iloc[i.item()], cmp_cat=comp_cat, wcs=None
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(process_idx, i) for i in idx]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Recreating model images",
            ):
                model_imgs.append(future.result())

        for j, i in tqdm(enumerate(idx), desc="Making context map", total=len(idx)):
            model = model_imgs[j]
            model = rescale(model, 1 / 4, anti_aliasing=False)
            model = torch.from_numpy(model).to(torch.float32)
            iz, iy, ix = i_pos[j]
            # Make sure we don't exceed any edges ever
            try:
                ctxt_map[
                    iz,
                    max(0, iy - model.shape[0] // 2) : min(
                        ctxt_map.shape[1], iy + model.shape[0] // 2 + model.shape[0] % 2
                    ),
                    max(0, ix - model.shape[1] // 2) : min(
                        ctxt_map.shape[2], ix + model.shape[1] // 2 + model.shape[1] % 2
                    ),
                ] += model[
                    max(0, model.shape[0] // 2 - iy) : min(
                        model.shape[0],
                        model.shape[0] // 2 + (ctxt_map.shape[1] - iy),
                    ),
                    max(0, model.shape[1] // 2 - ix) : min(
                        model.shape[1],
                        model.shape[1] // 2 + (ctxt_map.shape[2] - ix),
                    ),
                ]
            except RuntimeError as e:
                logger.error(
                    f"Error placing model image for index {i}.\n"
                    f"Model shape: {model.shape}, ctxt_map shape: {ctxt_map.shape}, "
                    f"position: {(iz, iy, ix)}"
                )
                raise e

        # Add channel dim and bring to mJy/beam
        ctxt_map = ctxt_map.unsqueeze(1) * 1e3

        # Save position mask for reference in analysis
        np.save(npy_folder / f"pos_mask_{itr + 1}.npy", pos_mask.numpy())

    # Store context sub-catalog
    ctxt_cat = dr3_cat_fpeak.iloc[idx].reset_index(drop=True)
    ctxt_cat.to_parquet(out_folder / f"ctxt_cat_{itr + 1}.parquet")

    # Sample
    logger.divider()
    logger.info(f"Iteration {itr + 1}/{n_iter}: Sampling...")
    if sampling_type == "ICM":
        sampling_kw = dict(
            size_quadrants=size_quadrants,
            batch_size=batch_size,
            ctxt_map=ctxt_map,
            guidance_strength=guidance_strength,
        )
    elif sampling_type == "LDM":
        # logger.info(f"Ctxt map shape: {ctxt_map.shape}")
        sampling_kw = dict(
            context=ctxt_map,
            rescale=False,
            return_latents=True,
            # Hard-coded workaround for EmbCC model:
            catalog_context=torch.zeros((ctxt_map.shape[0], 10, 3)),
            guidance_strength=guidance_strength,
        )
    img_batch, latent_batch = sampler.sample(**sampling_kw)

    # Save results
    logger.divider()
    logger.info(f"Iteration {itr + 1}/{n_iter}: Saving results...")
    np.save(npy_folder / f"img_batch_{itr + 1}.npy", img_batch)
    np.save(npy_folder / f"latent_batch_{itr + 1}.npy", latent_batch)
    np.save(npy_folder / f"ctxt_map_{itr + 1}.npy", ctxt_map)

    for img_idx, img in tqdm(
        enumerate(img_batch), desc="Saving png images", total=len(img_batch)
    ):
        fig, _ = plot_image_grid([img], img_side_len=8)
        fig.savefig(img_folder / f"img_{itr + 1}_{img_idx:04d}.png")
        plt.close(fig)

    # Scale batch
    scaler = sampler.scaler if sampling_type == "ICM" else sampler.px_scaler
    img_batch = scaler.inverse_scale(img_batch)
    img_batches.append(img_batch)

# Make big img batch
big_img_batch = np.concatenate(img_batches, axis=0)

# Run tiered bdsf wrapper in parallel
logger.divider()
logger.info("Running BDSF analysis in parallel...")


def process_batch(batch_idx, img_batch, out_folder):
    """Process a single batch with BDSF analysis"""
    try:
        # Extract single image from batch
        img = img_batch[batch_idx].squeeze()

        # Run BDSF analysis with output suppressed
        with contextlib.redirect_stdout(
            open(os.devnull, "w")
        ), contextlib.redirect_stderr(open(os.devnull, "w")):
            result = tiered_bdsf_wrapper(img, quiet=True)

        # Save result
        save_multistep_output(result, f"batch-{batch_idx:04d}", out_parent=bdsf_folder)

        return batch_idx, True, None
    except Exception as e:
        return batch_idx, False, str(e)


# Set number of workers - ThreadPoolExecutor can handle more workers since it's lighter
max_workers = min(
    os.cpu_count(), bdsf_workers
)  # Cap at 8 to avoid overwhelming the system
logger.info(f"Using {max_workers} workers for BDSF analysis")

# Run parallel processing with ThreadPoolExecutor
results = []
batch_indices = list(range(len(big_img_batch)))

with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks
    future_to_batch = {
        executor.submit(process_batch, batch_idx, big_img_batch, bdsf_folder): batch_idx
        for batch_idx in batch_indices
    }

    # Process results with progress bar
    with tqdm(total=len(batch_indices), desc="BDSF Analysis") as pbar:
        for future in concurrent.futures.as_completed(future_to_batch):
            result = future.result()
            results.append(result)
            pbar.update(1)

            # Log any errors
            batch_idx, success, error = result
            if not success:
                logger.warning(f"Batch {batch_idx} failed: {error}")

# Collect successful results
successful_images = [r[0] for r in results if r[1]]
failed_images = [r[0] for r in results if not r[1]]

logger.info(
    f"BDSF analysis complete: {len(successful_images)} successful, {len(failed_images)} failed"
)

# Save summary
summary = {
    "comments": comments,
    "quantity": qty,
    "quantity_bin_args": qty_bin_args,
    "sampling_type": sampling_type,
    "ctxt_type": ctxt_type,
    "scale_ctxt": scale_ctxt,
    "min_fpeak": min_fpeak,
    "total_images": len(big_img_batch),
    "successful_images": len(successful_images),
    "failed_images": len(failed_images),
    "sweep_name": sweep_name,
    "parameters": {
        "size_quadrants": size_quadrants,
        "src_per_q_len": src_per_q_len,
        "batch_size": batch_size,
        "n_iter": n_iter,
        "guidance_strength": guidance_strength,
    },
    "models": {
        "denoiser": denoiser,
        "denoiser_ckpt": denoiser_ckpt,
        "uncond_denoiser": uncond_denoiser,
        "uncond_denoiser_ckpt": uncond_denoiser_ckpt,
        "vae": vae,
    },
}
logger.info("Summary:")
pretty_print_config(summary)
# Save as json
with open(out_folder / "summary.json", "w") as f:
    json.dump(summary, f, indent=4)
# Add sweep values so those are saved as well in the npy
np.save(npy_folder / "extended_summary.npy", summary)

logger.info(f"Results saved to: {out_folder}")
logger.divider()
