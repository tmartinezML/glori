"""
Download pretrained glori checkpoints and scalers from the public Hugging Face
Hub repository. This is the entry point for anyone without access to the
original group storage (STORAGE_PARENT in settings/paths.py) -- e.g. an
external scientist installing this package standalone.

If the environment variable GLORI_PRETRAINED_DIR points to a local copy of the
repository (same file layout), files found there are used instead of downloading.
"""

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO_ID = "astrokevin/glori-ldm-swiit"

_FILES = {
    "denoiser": "LDM-Denoiser-WnetCC-v5.ckpt",
    "denoiser-rms": "LDM-Denoiser-WnetCC-rms-v1.ckpt",
    "vae": "VQ-VAE-256-DR3opt-FT.ckpt",
    "scaler": "scalers/LOFAR_scaler_II.pt",
    "ctxt_scaler_ftot": "scalers/ctxt_scaler_ftot.pt",
    "ctxt_scaler_fpeak": "scalers/ctxt_scaler_fpeak.pt",
    "ctxt_scaler_maj": "scalers/ctxt_scaler_maj.pt",
    # data of the noise-band tools (glori.inference.noise_sampler / noise_map_sampler)
    "dr3_snr_pool": "noise/dr3_snr_pool.npz",
    "dr3_fields": "noise/dr3_fields.csv",
}


def download_pretrained(component: str, repo_id: str = DEFAULT_REPO_ID) -> Path:
    """
    Download one of glori's pretrained files from the Hugging Face Hub.

    Downloads are cached in huggingface_hub's normal local cache
    (`~/.cache/huggingface/hub` by default), so repeated calls after the first
    are free and work offline.

    Parameters
    ----------
    component : str
        Which file to fetch. One of: "denoiser", "denoiser-rms", "vae", "scaler",
        "ctxt_scaler_ftot", "ctxt_scaler_fpeak", "ctxt_scaler_maj",
        "dr3_snr_pool", "dr3_fields".
    repo_id : str, optional
        Hugging Face Hub repo to download from. Defaults to the reference
        pretrained models for this codebase.

    Returns
    -------
    Path
        Local path to the downloaded file.

    Raises
    ------
    ValueError
        If `component` is not a recognized file.
    """
    if component not in _FILES:
        raise ValueError(
            f"Unknown component {component!r}. Expected one of {sorted(_FILES)}."
        )
    local = os.environ.get("GLORI_PRETRAINED_DIR")
    if local and (Path(local) / _FILES[component]).exists():
        return Path(local) / _FILES[component]
    return Path(hf_hub_download(repo_id, _FILES[component]))
