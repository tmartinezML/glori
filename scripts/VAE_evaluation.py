import os
import argparse
from functools import partial

import torch
import numpy as np
from tqdm import tqdm, trange

import utils.paths as paths
from models.vae.vqvae import VQVAE
from utils.my_logging import get_logger
from data.trf.scalers import LOFARScaler
from data.sets.micromaps import MicromapDatasetHF
from data.trf.transforms import WebdatasetTransformCrop

# Add argument for debug configuration
parser = argparse.ArgumentParser()
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug mode",
)
args = parser.parse_args()

debug = args.debug

# Get logger
logger = get_logger("script")
logger.info("Setting up...")

# Settings
image_size = 512
f_vae = 4
latent_size = image_size // f_vae
num_workers = 8
device = "cuda:3"
batch_size = 16
vae_model = "VQ-VAE-256-DR3opt-FT"

# Limit cpu usage
os.sched_setaffinity(0, set(range(2 * num_workers)))

# Prepare output directory
logger.info("Preparing path...")
out_path_sub = paths.ANALYSIS_PARENT / "paper_results_III/VAE"
out_path_sub.mkdir(exist_ok=True)

# Load dataset
logger.info("Loading test split...")
cutouts_set = MicromapDatasetHF(
    "micromaps-DR3-opt-1024",
    split="test",
    max_beam_arcsec=6,
    dset_lookup=paths.MICROMAP_SUBSETS_ARROW_HOPPER,
    custom_transform=WebdatasetTransformCrop(crop_size=image_size, random_crop=False),
)
# In debug mode, limit number of samples
if debug:
    cutouts_set = cutouts_set.select(np.arange(128))

# Prepare dataloader
dl = cutouts_set.get_dataloader(
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
)

# Load scaler
logger.info("Loading scaler...")
scaler = LOFARScaler.load("LOFAR_scaler_II")

# Load VAE
logger.info("Loading VQ-VAE...")
vae = VQVAE.load(vae_model, ckpt="best")

# Prepare VAE for decoding
latent_dim = vae.emb_dim
vae.eval()
vae.to(device)

# Prepare tensors to hold data
cutouts_shape = (len(cutouts_set), 1, image_size, image_size)
encodings_shape = (len(cutouts_set), latent_dim, latent_size, latent_size)
cutouts = torch.zeros(cutouts_shape, dtype=torch.float32)
cutouts_scaled = cutouts.clone()
reconstructions = cutouts.clone()
reconstructions_unscaled = cutouts.clone()
encodings = torch.zeros(encodings_shape, dtype=torch.float32)

# Run dataloading and decoding loop
with torch.no_grad():
    for i, batch in enumerate(tqdm(dl, total=len(dl))):
        # Get batch
        (batch_cutouts,) = batch
        batch_cutouts_scaled = scaler.scale(batch_cutouts)

        # Place inputs
        sl = slice(i * batch_size, i * batch_size + batch_cutouts.shape[0])
        cutouts[sl] = batch_cutouts
        cutouts_scaled[sl] = batch_cutouts_scaled

        # Decode batch
        batch_encodings = vae.encode_to_prequant(batch_cutouts_scaled.to(device))
        batch_reconstructions = vae.decode_code(batch_encodings).cpu()
        batch_encodings = batch_encodings.cpu()
        encodings[sl] = batch_encodings
        reconstructions[sl] = batch_reconstructions

        # Un-scale outputs
        reconstructions_unscaled[sl] = scaler.inverse_scale(batch_reconstructions)

# Make everything numpy
cutouts = cutouts.numpy().squeeze()
cutouts_scaled = cutouts_scaled.numpy().squeeze()
encodings = encodings.numpy().squeeze()
reconstructions = reconstructions.numpy().squeeze()
reconstructions_unscaled = reconstructions_unscaled.numpy().squeeze()

# Release GPU
vae.to("cpu")
del batch_encodings, batch_reconstructions, vae
torch.cuda.empty_cache()

logger.info("Calculating delta...")
rec_delta = reconstructions - cutouts_scaled

# Save everything to npy
logger.info("Saving...")
payload = {
    "cutouts": cutouts,
    "cutouts_scaled": cutouts_scaled,
    "reconstructions": reconstructions,
    "reconstructions_unscaled": reconstructions_unscaled,
    "rec_delta": rec_delta,
    "encodings": encodings,
}
for k, v in tqdm(payload.items(), desc="Saving arrays"):
    np.save(out_path_sub / f"{k}.npy", v)
