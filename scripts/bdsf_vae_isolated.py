import torch
import numpy as np

import utils.paths as paths
from data.trf.transforms import MicroMapTransformLDM
from data.utils import load_mosaic, load_fits_catalog
from utils.my_logging import get_logger
from torch.utils.data import TensorDataset, DataLoader

from analysis.bdsf_analysis import *
from data.obs.micromaps.utils import get_micromaps, assemble_micromaps
from data.trf.scalers import LOFARScaler
from models.vae.vqvae import VQVAE
import models.utils as mutil

# Some settings
cutout_size_px = 4096  # Size of the cutout in pixels
mmpsize = 256  # Micromap size in pixels
batch_size = 64
device = "cuda:3"

logger = get_logger(__name__)
logger.divider()

logger.info("Preparing input image...")
mosaic_list = sorted([p.name for p in paths.MOSAIC_DIR_DR3.glob("*") if p.is_dir()])
mosaic = mosaic_list[-1]
map_img, wcs = load_mosaic(mosaic, paths.MOSAIC_DIR_DR3)
micromaps, centers, _ = get_micromaps(
    map_img, mmpsize, 1, wcs=wcs, pbar=True, verbose=True, remove_nans=True
)
assembled = assemble_micromaps(micromaps, centers, map_shape=map_img.shape)

logger.divider()
logger.info("Preparing dataset...")
scaler = LOFARScaler.load("LOFAR_scaler_II")
dset = TensorDataset(torch.from_numpy(scaler.scale(micromaps)).float().unsqueeze(1))
loader = DataLoader(
    dset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
)

logger.divider()
logger.info("Preparing VAE...")
ckpt = mutil.parse_lightning_ckpt("best", model_name="VQ-VAE-256")
logger.info(f"Loading VAE:\t{ckpt.parent.parent.name}\nfrom:\t{ckpt.name}")
vae = VQVAE.load_from_checkpoint(ckpt, map_location="cpu")
vae.eval()

logger.divider()
logger.info("Encoding micromaps with VAE...")
mmaps_rec = torch.empty(micromaps.shape)
vae = vae.to(device)
with torch.no_grad():
    for i, batch in tqdm(
        enumerate(loader), total=len(loader), desc="Encoding micromaps"
    ):
        mmaps_rec[i * batch_size : (i + 1) * batch_size] = scaler.inverse_scale(
            vae(batch[0].to(device))[0].cpu().squeeze(1)
        )
vae = vae.cpu()

logger.divider()
logger.info("Assembling reconstructed map...")
assembled_rec = assemble_micromaps(mmaps_rec.numpy(), centers, map_shape=map_img.shape)

logger.info("Saving maps...")
out_folder = out_parent = (
    paths.ANALYSIS_PARENT
    / f"ldm/VAE-Isolated_{mosaic}-{cutout_size_px}x{cutout_size_px}"
)
out_folder.mkdir(exist_ok=True)
np.save(out_folder / "assembled.npy", assembled)
np.save(out_folder / "assembled_rec.npy", assembled_rec)

logger.divider()
logger.info("Running BDSF on reconstructed map cutout...")
cby2, cpx = cutout_size_px // 2, assembled_rec.shape[-1] // 2
sl = slice(cpx - cby2, cpx + cby2)
ass_rec_cut, wcs_cut = assembled_rec[sl, sl], wcs[sl, sl]

res = tiered_bdsf_wrapper(ass_rec_cut, wcs_cut)

logger.info("BDSF analysis completed. Saving...")
save_multistep_output(res, f"bdsf_result_rec", out_parent=out_folder)

logger.divider()
logger.info("BDSF analysis complete.")
