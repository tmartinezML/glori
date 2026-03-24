import json
from datetime import datetime
from functools import partial

import art
import torch
import randomname
import numpy as np

import glori.settings.paths as paths
import glori.data.trf.post as post
import glori.data.trf.transforms as T
import glori.analysis.ldm.io as ldmio
import glori.analysis.ldm.bdsf as ldmbdsf
from glori.analysis.bdsf_analysis import *
from glori.plotting.images import plot_image_grid
from glori.infra.logging import get_logger, add_file_handler
from glori.data.trf.functional import zero_center
from glori.data.sets.micromaps import MicromapDatasetHF
from glori.models.load import parse_lightning_ckpt
from glori.inference.ldm_sampler import LDMSampler

# Read first argument --debug for debug mode
debug_mode = len(sys.argv) > 1 and sys.argv[1] == "--debug"

# Limit Python to use only the first N CPUs
Ncpu = 16
os.sched_setaffinity(0, set(range(Ncpu)))

# Check the CPUs available to the process
print("CPUs available:", os.sched_getaffinity(0))

logger = get_logger("LDM-Resample")
logger.setLevel(level="DEBUG" if debug_mode else "INFO")
logger.divider()
print(art.text2art("LDM Resample", font="cybermedium"))
logger.divider()

# Model settings
denoiser = "LDM-Denoiser-WnetCC-v5"
denoiser_ckpt = "last-best"
uncond_denoiser = None
uncond_denoiser_ckpt = "best-last-10"
vae = "VQ-VAE-256-DR3opt-FT"
sampling_type = "LDM"  # "LDM or "SWIIT"

# Sampling settings
latent_size = 128
f_vae = 4
image_size = latent_size * f_vae
guidance_strength = 0.2
timesteps = 25
use_inpainting_replacement = True
catalog_mode = "combined"  # options: "combined", "separate"
device = "cuda:1"

# Settings for the sampled images
n_images = 512
batch_size = 16
assert (
    n_images % batch_size == 0
), f"n_images must be divisible by batch_size, got {n_images} and {batch_size}"
n_iter = n_images // batch_size
dataset = "micromap-encodings-DR3-opt-1024px-spacing=1"
dataset_lookup = paths.MICROMAP_SUBSETS_ARROW_HOPPER
weights_file = None
max_beam_arcsec = 6
seed = 42
noise_seed = 42
bdsf_workers = 16  # Number of parallel workers for BDSF analysis
do_bdsf = True

# Apply debug mode settings
if debug_mode:
    logger.info("Debug mode enabled. Using reduced settings for quick testing.")
    n_images = 4
    batch_size = 2
    n_iter = n_images // batch_size
    dataset = "micromap-encodings-DR3-opt-1024"
    timesteps = 2

# Settings for output folder
out_folder_lbl = ""
comments = ""

logger.divider()

# Prepare output folders
resample_name = randomname.get_name() if not debug_mode else "debug"
logger.info(f"Resample name: {resample_name}")
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
out_folder = out_parent = paths.ANALYSIS_PARENT / (
    f"ldm/LDM-resample_cutouts-{n_images=}"
    f"{'-' + out_folder_lbl if out_folder_lbl is not None and len(out_folder_lbl) > 0 else ''}"
    f"-{timestamp}-{resample_name}"
)
if debug_mode:
    out_folder = out_parent = paths.ANALYSIS_PARENT / "ldm/ICM-resample-debug"
bdsf_folder, img_folder, npy_folder = ldmio.prepare_directory(
    out_folder, override=debug_mode
)

# Add log file
log_file = out_folder / "resample.log"
add_file_handler(logger, log_file, level="DEBUG" if debug_mode else "INFO")
logger.info(f"Logging to file: {log_file}")

# Load datasets
logger.info("Loading encodings dataset...")
match catalog_mode:
    case "combined":
        output_tuple = (
            "npy",
            "context_downscaled_d=4_f=2.npy",
        )
        ctxt_transform = {
            "context_downscaled_d=4_f=2.npy": T.CatalogContextTransform(scale_fn=None),
        }
    case "separate":
        output_tuple = (
            "npy",
            "context_downscaled.npy",
            "context_downscaled_d=4_f=2.npy",
        )
        ctxt_transform = {
            "context_downscaled.npy": T.CatalogContextTransform(scale_fn=None),
            "context_downscaled_d=4_f=2.npy": T.CatalogContextTransform(scale_fn=None),
        }
    case _:
        raise ValueError(f"Invalid catalog_mode: {catalog_mode}")
post_transforms = []
post_transforms.append(
    post.make_dependent_center_crop(
        crop_size=latent_size,
        f=[1, 2] if catalog_mode == "separate" else [2],
        s=[1, 2] if catalog_mode == "separate" else [2],
        keys=output_tuple,
    )
)
if catalog_mode == "separate":
    post_transforms.append(
        post.make_post(
            {
                "context_downscaled_d=4_f=2.npy": partial(zero_center, f_center=2),
            }
        )
    )
# Load encodings, needed for the catalog context.
dset = MicromapDatasetHF(
    dset=dataset,
    dset_lookup=dataset_lookup,
    split="test",
    output_tuple=output_tuple,
    weights_file=weights_file,
    ctxt_transform=ctxt_transform,
    post_transform=post.compose(*post_transforms),
    max_beam_arcsec=max_beam_arcsec,
)
# Load maps, needed for visual comparison
logger.info("Loading maps dataset...")
maps_dset = MicromapDatasetHF(
    dset=dataset.replace("micromap-encodings", "micromaps"),
    dset_lookup=dataset_lookup,
    split="test",
    output_tuple=("npy",),
    weights_file=weights_file,
    mode="maps-scaled",
    post_transform=post.make_dependent_center_crop(crop_size=image_size, keys=("npy",)),
    max_beam_arcsec=max_beam_arcsec,
)

# Make sub-selection
logger.info(f"Selecting {n_images} cutouts from dataset...")
np.random.seed(seed)
sampling_weights = None
if weights_file is not None:
    sampling_weights = np.array(
        [np.max(dset.weights_dict[k]) for k in dset.dataset["__key__"]],
        dtype=np.float32,
    )
    sampling_weights /= sampling_weights.sum()
selected_indices = np.random.choice(
    len(dset), size=n_images, replace=False, p=sampling_weights
)
dset = dset.select(selected_indices)
maps_dset = maps_dset.select(selected_indices)
assert (
    dset.dataset["__key__"] == maps_dset.dataset["__key__"]
), "Datasets are not aligned after selection!"

# Get dataloader
dataloader = dset.get_dataloader(batch_size=batch_size, shuffle=False)
dl_it = iter(dataloader)

# Load the sampler
ckpt_filename = str(parse_lightning_ckpt(denoiser_ckpt, model_name=denoiser))
ldm_sampler = LDMSampler(
    denoiser=denoiser,
    denoiser_ckpt=denoiser_ckpt,
    uncond_denoiser=uncond_denoiser,
    uncond_denoiser_ckpt=uncond_denoiser_ckpt,
    vae=vae,
    device=device,
    image_size=image_size,
    latent_size=latent_size,
)

# Prepare torch noise seed
torch.manual_seed(noise_seed)

# Start sampling loop
logger.info(f"Running {n_iter} iterations.")

img_batches = []
for itr in range(n_iter):

    # Log progress
    logger.divider(n_lines=2)
    logger.info(f"Iteration {itr + 1}/{n_iter}")

    # Read batch
    batch = next(dl_it)
    match catalog_mode:
        case "combined":
            enc, ext_ctxt = batch
            ctxt = None
        case "separate":
            enc, ctxt, ext_ctxt = batch

    # Prepare seed noise for this batch
    batch_seed = seed + itr  # Different seed for each batch
    torch.manual_seed(batch_seed)
    seed_noise = torch.randn_like(enc)

    # Sample images with LDM sampler
    logger.info("Running LDM sampling...")
    img_out, enc_out = ldm_sampler.sample(
        img_context=ctxt,
        catalog_context=ext_ctxt,
        guidance_strength=guidance_strength,
        timesteps=timesteps,
        rescale=False,
        return_latents=True,
        seed_noise=seed_noise,
        use_inpainting_replacement=use_inpainting_replacement,
    )

    # Inverse scale the output images and save to list for later BDSF analysis
    img_out_unsc = ldm_sampler.scaler.inverse_scale(img_out)
    img_batches.append(img_out_unsc)

    # Save arrays of this iteration
    logger.divider()
    logger.info(f"Saving numpy arrays for iteration {itr + 1:04d}...")
    np.save(npy_folder / f"img_batch_{itr + 1:04d}.npy", img_out)
    np.save(npy_folder / f"samples_unscaled_{itr + 1:04d}.npy", img_out_unsc)
    # Save center crop of extended context map.
    # This is useful for later input-output comparison.
    if catalog_mode == "combined":
        padw = ext_ctxt.shape[-1] // 4
        ctxt = ext_ctxt.clone()[:, :, padw:-padw, padw:-padw]
    np.save(npy_folder / f"context_{itr + 1:04d}.npy", ctxt.numpy())
    np.save(npy_folder / f"extended_context_{itr + 1:04d}.npy", ext_ctxt.numpy())

    # Save png images for this iteration
    logger.info(f"Saved output images to {npy_folder}. Saving png images...")
    fig, _ = plot_image_grid(img_out.squeeze())
    fig.savefig(img_folder / f"img_{itr + 1:04d}.png")
    plt.close(fig)

    # Save original array and png
    (original_imgs,) = maps_dset.get_batch(itr, batch_size=batch_size)
    np.save(npy_folder / f"original_imgs_{itr + 1:04d}.npy", original_imgs.numpy())
    fig, _ = plot_image_grid(original_imgs.squeeze())
    fig.savefig(img_folder / f"img_{itr + 1:04d}_original.png")
    plt.close(fig)

    logger.info(f"Saved original images.")

# Make big img batch
big_img_batch = np.concatenate(img_batches, axis=0)


if do_bdsf:
    results, n_successful_images, n_failed_images = ldmbdsf.run_bdsf_parallel(
        big_img_batch, bdsf_folder, logger=logger, max_workers=bdsf_workers
    )

logger.info("Saving summary...")
summary = {}
summary.update(
    dict(
        sampling_script="ldm_resample_cutouts.py",
        denoiser=denoiser,
        denoiser_ckpt=denoiser_ckpt,
        denoiser_ckpt_filename=ckpt_filename,
        uncond_denoiser=uncond_denoiser,
        uncond_denoiser_ckpt=uncond_denoiser_ckpt,
        vae=vae,
        device=device,
        n_iter=n_iter,
        batch_size=batch_size,
        n_cutouts=n_images,
        dataset=dataset,
        weights_file=weights_file,
        max_beam_arcsec=max_beam_arcsec,
        seed=seed,
        bdsf_workers=bdsf_workers,
        do_bdsf=do_bdsf,
        image_size=image_size,
        latent_size=latent_size,
        out_folder_lbl=out_folder_lbl,
        guidance_strength=guidance_strength,
        timesteps=timesteps,
        total_images=len(big_img_batch) if do_bdsf else None,
        successful_images=n_successful_images if do_bdsf else None,
        failed_images=n_failed_images if do_bdsf else None,
        comments=comments,
    )
)
with open(out_folder / "summary.json", "w") as f:
    json.dump(summary, f, indent=4)
np.save(npy_folder / "summary.npy", summary)

logger.info(f"Results saved to: {out_folder}")
logger.divider()
