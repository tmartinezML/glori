from pathlib import Path
import torch
from models.diffusion.denoiser import Denoiser
import train.utils as tutils

# Load Nan checkpoint
nan_ckpt_path = Path(
    "/hs/fs08/data/group-brueggen/tmartinez/diffusion/model_results/LDM-Denoiser-EmbCC-128/lightning/ckpt_nan_debug_00266639.pt"
)
ckpt_dict = torch.load(nan_ckpt_path, map_location="cpu")
ckpt_dict.keys()

# Load model from checkpoint
model = Denoiser.from_preset("LDM-Denoiser-EmbCC")
model.model.load_state_dict(ckpt_dict["model_state_dict"])

# Prepare inputs
device = "cuda:0"
model.to(device)

batch = ckpt_dict["batch"].copy()
x, ctxt_dict = model.read_batch(batch)
x = x.to(device)
ctxt_dict = {k: (v.to(device) if v is not None else None) for k, v in ctxt_dict.items()}
mask = tutils.weighted_random_quadrant_mask(x.shape, weights=[0.1, 0.1, 0.1, 0.7]).to(
    device
)

# Sample sigmas
sigmas = tutils.sample_sigmas(x, model.train_config.P_mean, model.train_config.P_std)
n = torch.randn_like(x) * sigmas

# Call model
loss = tutils.edm_loss(
    model,
    x,
    **ctxt_dict,
    sigma_data=model.model.sigma_data,
    noise=n,
    mask=mask,
    return_output=False
)
