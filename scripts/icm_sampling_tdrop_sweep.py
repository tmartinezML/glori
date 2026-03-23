import shutil
import sys

import matplotlib.pyplot as plt
import torch
import os
import randomname
from datetime import datetime
from pathlib import Path
from importlib import reload
import numpy as np

import json
import utils.paths as paths

from importlib import import_module, reload
import pandas as pd
from tqdm import tqdm


from data.utils import load_mosaic
from data.obs.micromaps.utils import (
    context_map_by_wcs,
    reduce_context_map,
)
from utils.my_logging import get_logger, add_file_handler
from plotting.images import plot_image_grid


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


# Convenience function for loading mosaic names
# with control over random seed, number and max beam size
def load_mosaic_names(n_mosaics=1, seed=None, max_beam_arcsec=None):
    # Get list of mosaics based on directory contents
    mosaic_list = sorted([p.name for p in paths.MOSAIC_DIR_DR3.glob("*") if p.is_dir()])

    # Filter by max beam if specified
    if max_beam_arcsec is not None:
        # Load pointing info, which includes beam sizes
        pointing_info_df = pd.read_csv(
            paths.LOFAR_DATA_PARENT / "DR3_pointing_lookup.csv", index_col="mosaic"
        )
        # Select good pointings, i.e. pointings with beam <= max_beam_arcsec
        good_pointings = np.array(
            pointing_info_df.index[
                pointing_info_df["Beam"] <= np.round(max_beam_arcsec / 3600, 5)
            ],
            dtype=np.str_,
        )
        # Filter mosaics by good pointings
        select_idxs = np.argwhere([p in good_pointings for p in mosaic_list]).flatten()
        mosaic_list = [mosaic_list[i] for i in select_idxs]
        # Pick randomly from remaining mosaics
        np.random.seed(seed)
        mosaics = [np.random.choice(mosaic_list) for _ in range(n_mosaics)]
        return mosaics


# Convenience function for getting mosaic central cutout and context map
def get_mosaic_blueprint(mosaic, cutout_size_px, cat, f_downscale=4):
    # Get mosaic and wcs data
    img, wcs = load_mosaic(mosaic)

    # Get central cutout from both image and wcs
    sl1, sl2 = get_slices(img, cutout_size_px)
    img_cut, wcs_cut = img[sl1, sl2], wcs[sl1, sl2]

    # Create context map from catalog
    ctxt, ctxt_cat = context_map_by_wcs(
        wcs_cut,
        cat,
        scale_output=False,
    )
    # Downsample context map like training data
    ctxt_red = reduce_context_map(
        ctxt,
        scale_output=False,
        input_scaled=False,
    )
    # Scale location map to [-1, 1]
    ctxt_red[0] = ctxt_red[0] * 2 - 1
    return img_cut, ctxt_red, ctxt_cat


# Read first argument --debug for debug mode
debug_mode = len(sys.argv) > 1 and sys.argv[1] == "--debug"

# Get logger
logger = get_logger("script")
if debug_mode:
    logger.setLevel("DEBUG")
    logger.debug("Debug mode enabled: setting log level to DEBUG")

# Prepare output directory
resample_name = randomname.get_name() if not debug_mode else "debug"
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
output_dir = (
    paths.ANALYSIS_PARENT
    / "icm_sampling_tdrop_sweep"
    / (f"{timestamp}_{resample_name}" if not debug_mode else "debug")
)
if debug_mode and output_dir.exists():
    logger.debug(f"Debug output directory {output_dir} already exists. Removing it.")
    shutil.rmtree(output_dir)

output_dir.mkdir(parents=True, exist_ok=True)
log_file = output_dir / "experiment.log"
add_file_handler(logger, str(log_file))

logger.info(f"Experiment name: {resample_name}")
logger.info(f"Output directory:\n\t{output_dir}")

# Model settings
denoiser = "LDM-Denoiser-WnetCC-v4"
denoiser_ckpt = "best"
uncond_denoiser = "LDM-Denoiser-Uncond-128"
uncond_denoiser_ckpt = "best"
vae = "VQ-VAE-256"

# Sampling settings
latent_size = 128
f_vae = 4
image_size = latent_size * f_vae
guidance_strength = 0.2
timesteps = 25
f_inner = 1
f_ext_ctxt = 2
resample_first = False
catalog_mode = "combined"  # options: "combined", "separate"
decoding_stride = None
use_inpainting_replacement = True
do_inpainting = True
device = "cuda:1"


# Sampling blueprint settings
sampling_steps = (4, 4)
mask_coverage = 0.5
mask_overlap = int(mask_coverage * latent_size)
stride = latent_size - mask_overlap
img_mask_overlap = int(mask_coverage * image_size)
img_stride = image_size - img_mask_overlap
latent_map_size = tuple(latent_size + (s - 1) * stride for s in sampling_steps)
img_map_size = tuple(s * f_vae for s in latent_map_size)
latent_map_height, latent_map_width = latent_map_size
batch_size = 4
single_mosaic = True
max_beam_arcsec = 6
seed = 70
noise_seed = 42
t_drop_values = np.array([25, *np.flip(np.arange(0, 25, step=4)), -1])
# t_drop_values = [26, 18, 15, 12, -1]

if debug_mode:
    sampling_steps = (2, 2)
    latent_map_size = tuple(latent_size + (s - 1) * stride for s in sampling_steps)
    img_map_size = tuple(s * f_vae for s in latent_map_size)
    latent_map_height, latent_map_width = latent_map_size
    batch_size = 1
    single_mosaic = True
    t_drop_values = [26, -1]
    timesteps = 2

# Load LoTSS DR3 catalog
from data.utils import load_lotss_catalog

lotss_cat = load_lotss_catalog(
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

# Load mosaic name
mosaics = load_mosaic_names(
    n_mosaics=1 if single_mosaic else batch_size,
    seed=seed,
    max_beam_arcsec=max_beam_arcsec,
)
logger.info(f"Selected mosaics: {mosaics}")

# Extract blueprints from selected mosaics
logger.info("Loading blueprints...")
original_images = []
ctxt_red_arrays = []
ext_ctxt_red_arrays = []
for mosaic in tqdm(mosaics):
    # Get mosaic blueprint
    img_cut, ctxt_red, _ = get_mosaic_blueprint(mosaic, img_map_size, cat=lotss_cat)

    # Get extended context map
    extended_ctxt_size = tuple(s + image_size * (f_ext_ctxt - 1) for s in img_map_size)
    _, ext_ctxt_red, _ = get_mosaic_blueprint(mosaic, extended_ctxt_size, cat=lotss_cat)

    # Store results
    original_images.append(img_cut)
    ctxt_red_arrays.append(ctxt_red)
    ext_ctxt_red_arrays.append(ext_ctxt_red)

# Stack into one array for batch sampling
img_cut = np.stack(original_images)
ctxt_red = np.stack(ctxt_red_arrays)
ext_ctxt_red = np.stack(ext_ctxt_red_arrays)
if single_mosaic:
    # If only one mosaic, repeat it to fill the batch
    img_cut = np.repeat(img_cut, batch_size, axis=0)
    ctxt_red = np.repeat(ctxt_red, batch_size, axis=0)
    ext_ctxt_red = np.repeat(ext_ctxt_red, batch_size, axis=0)
# Verify correct shape
assert ctxt_red.shape == (
    batch_size,
    4,
    *latent_map_size,
), f"ctxt_red shape {ctxt_red.shape} does not match expected {(batch_size, 4, *latent_map_size)}"
assert ext_ctxt_red.shape == (
    batch_size,
    4,
    *[s // f_vae for s in extended_ctxt_size],
), f"ext_ctxt_red shape {ext_ctxt_red.shape} does not match expected {(batch_size, 4, *[s // f_vae for s in extended_ctxt_size])}"

# Load ICM Sampler
from glori.inference.swiit_sampler import SWIITSampler

logger.info("Initializing ICM sampler...")
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

# Set up sampling loop
torch.manual_seed(noise_seed)
seed_noise_map = torch.randn(batch_size, 3, *latent_map_size)

sampling_kw = dict(
    ctxt_map=torch.from_numpy(ctxt_red).to(torch.float32),
    guidance_strength=guidance_strength,
    save_intermediates=True,
    timesteps=timesteps,
    seed_noise_map=seed_noise_map,
    resample_first=resample_first,
    verbose=False,
    decoding_stride=decoding_stride,
    catalog_mode=catalog_mode,
    use_inpainting_replacement=use_inpainting_replacement,
)

# Sample baseline image
logger.info("Sampling baseline (no inpainting context)...")
baseline_map, _, baseline_intermediates = icm_sampler.sample(
    do_inpainting=False,
    **sampling_kw,
)

samples = []
intermediates_list = []
for t_drop in tqdm(t_drop_values, desc="parameter Loop", colour="blue"):
    map_sampled_exp, _, intermediates = icm_sampler.sample(
        drop_inpainting_ctxt_at=t_drop,
        do_inpainting=True,
        **sampling_kw,
    )
    samples.append(map_sampled_exp.reshape(batch_size, *map_sampled_exp.shape[-2:]))
    intermediates_list.append(intermediates)


# Two metrics for every sampling step:
# 1) Baseline difference: Difference to same step at baseline (no inpainting ctxt)
#       --> How similar is it to the "no inpainting context" case?
# 2) Overlap difference: Difference between overlapping region with inpainting context
#       --> How similar is the sampled image to the original image in the region where inpainting context is used?

# We need to decode the intermediate inpainting contexts, which at this point are
# still in latent space.

logger.info("Sampling done. Computing metrics...")
steps_sampled = len(intermediates_list[0]["decoded_images"])
baseline_diffs = np.zeros(
    (len(samples), steps_sampled, batch_size, image_size, image_size)
)
overlap_diffs = np.zeros(
    (len(samples), steps_sampled, batch_size, image_size, image_size)
)
replaced_intermediate_images = np.zeros(
    (len(samples), steps_sampled, batch_size, image_size, image_size)
)
# Loop over iterations of the experiment:
for i, intermediates in tqdm(
    enumerate(intermediates_list),
    desc="Computing metrics",
    total=len(intermediates_list),
):
    intermediate_images = intermediates["decoded_images"]
    # Loop over sampling iterations of a single map:
    for j, intermediate_img in enumerate(intermediate_images):
        # We need to decode the intermediate inpainting contexts, which at this point are
        # still in latent space.

        # Get corresponding intermediate from baseline case
        baseline_intermediate = baseline_intermediates["decoded_images"][j]

        # Compute baseline difference
        baseline_diff = (intermediate_img - baseline_intermediate).numpy().squeeze()
        baseline_diffs[i, j] = baseline_diff

        # At this point the intermediate inpainting context is still in latent space.
        # We replace the sapled latent with the inpainting context at the
        # overlapping region, decode it to pixel space and compute the difference
        # to the original image in that region.
        inp_ctxt = intermediates["inpainting_contexts"][j]
        intermediate_latent = intermediates["latents"][j]
        sampling_mask = intermediates["sampling_masks"][j]
        replaced_latent = torch.where(
            sampling_mask.to(bool), intermediate_latent, inp_ctxt
        ).to(device)
        icm_sampler.vae.to(device)
        with torch.no_grad():
            replaced_image = icm_sampler.vae.decode(replaced_latent).cpu()
        del replaced_latent
        icm_sampler.vae.to("cpu")
        replaced_intermediate_images[i, j] = replaced_image.numpy().squeeze()

        # Compute overlap difference
        overlap_diff = (intermediate_img - replaced_image).numpy().squeeze()
        sampling_mask = (
            torch.nn.functional.interpolate(
                sampling_mask, scale_factor=f_vae, mode="nearest"
            )
            .to(bool)
            .squeeze()
            .numpy()
        )
        # Only meaningful for non-sampled parts
        overlap_diff[sampling_mask] = np.nan
        overlap_diffs[i, j] = overlap_diff


# Save results
logger.info("Saving results...")
np.savez(
    output_dir / "results.npz",
    samples=np.array(samples),
    original=img_cut[0:1] if single_mosaic else img_cut,
    ctxt_red=ctxt_red[0:1] if single_mosaic else ctxt_red,
    ext_ctxt_red=ext_ctxt_red[0:1] if single_mosaic else ext_ctxt_red,
    seed_noise_map=seed_noise_map.cpu().numpy(),
    baseline_diffs=baseline_diffs,
    overlap_diffs=overlap_diffs,
    parameters=t_drop_values,
    baseline_intermediates=baseline_intermediates,
    intermediates_list=intermediates_list,
    t_drop_values=t_drop_values,
    replaced_intermediate_images=replaced_intermediate_images,
)

# Write all settings to json file
logger.info("Saving settings...")
settings = {
    "model_settings": {
        "denoiser": denoiser,
        "denoiser_ckpt": denoiser_ckpt,
        "uncond_denoiser": uncond_denoiser,
        "uncond_denoiser_ckpt": uncond_denoiser_ckpt,
        "vae": vae,
    },
    "sampling_settings": {
        "latent_size": latent_size,
        "f_vae": f_vae,
        "image_size": image_size,
        "guidance_strength": guidance_strength,
        "timesteps": timesteps,
        "f_inner": f_inner,
        "f_ext_ctxt": f_ext_ctxt,
        "resample_first": resample_first,
        "decoding_stride": decoding_stride,
        "use_inpainting_replacement": use_inpainting_replacement,
        "catalog_mode": catalog_mode,
        "device": device,
    },
    "sampling_blueprint_settings": {
        "sampling_steps": sampling_steps,
        "mask_coverage": mask_coverage,
        "mask_overlap": mask_overlap,
        "stride": stride,
        "img_mask_overlap": img_mask_overlap,
        "img_stride": img_stride,
        "latent_map_size": latent_map_size,
        "img_map_size": img_map_size,
    },
    "experiment_settings": {
        "single_mosaic": single_mosaic,
        "max_beam_arcsec": max_beam_arcsec,
        "seed": seed,
        "noise_seed": noise_seed,
    },
}
with open(output_dir / "settings.json", "w") as f:
    json.dump(settings, f, indent=4)

# Plot sampled images
logger.info("Plotting results...")
img_dir = output_dir / "image_plots"
img_dir.mkdir(exist_ok=True)
for i in range(len(samples) + 1):
    if i == 0:
        sample = baseline_map
        title = "baseline"
    else:
        sample = samples[i - 1]
        title = f"t_drop={t_drop_values[i - 1]}"
    fig, axs = plot_image_grid(sample, n_cols=1, img_side_len=8)
    fig.suptitle(title)
    fig.savefig(img_dir / f"{title}.png")
    plt.close(fig)
title = "original"
fig, axs = plot_image_grid(icm_sampler.scaler.scale(img_cut), n_cols=1, img_side_len=8)
fig.suptitle(title)
fig.savefig(img_dir / f"{title}.png")
plt.close(fig)


logger.info(f"Experiment completed. Results saved to\n\t{output_dir}.")
