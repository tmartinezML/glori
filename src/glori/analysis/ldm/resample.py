from importlib import import_module, reload
from types import SimpleNamespace
import inspect
import pandas as pd
import numpy as np

from glori.data.utils import load_mosaic
from glori.data.obs.micromaps.utils import (
    context_map_by_wcs,
    reduce_context_map,
)
import glori.settings.paths as paths

DEFAULT_SETTINGS_DICT = {
    # Model settings
    "denoiser": "LDM-Denoiser-WnetCC-v4.1",
    "denoiser_ckpt": "best",
    "uncond_denoiser": "LDM-Denoiser-Uncond-128",
    "uncond_denoiser_ckpt": "best",
    "vae": "VQ-VAE-256",
    # Sampling settings (inputs only; no derived fields here)
    "latent_size": 128,
    "f_vae": 4,
    "guidance_strength": 0.2,
    "timesteps": 25,
    "f_inner": 1,
    "f_ext_ctxt": 2,
    "resample_first": False,
    "device": "cuda:2",
    "catalog_mode": "combined",  # options: "combined", "separate"
    "decoding_stride": None,
    "do_inpainting": True,
    "use_inpainting_replacement": True,
    "drop_inpainting_ctxt_at": -1,
    # Sampling blueprint input settings
    "sampling_steps": (2, 2),
    "mask_coverage": 0.5,
    "batch_size": 8,
    "max_beam_arcsec": 6,
    "seed": 42,
    "noise_seed": 42,
}


class ResampleSettings(SimpleNamespace):
    def __init__(self, **kw):
        data = {**DEFAULT_SETTINGS_DICT, **kw}
        super().__init__(**data)
        self._refresh_derived()

    def _refresh_derived(self):
        self.image_size = self.latent_size * self.f_vae

        self.mask_overlap = int(self.mask_coverage * self.latent_size)
        self.stride = self.latent_size - self.mask_overlap

        self.img_mask_overlap = int(self.mask_coverage * self.image_size)
        self.img_stride = self.image_size - self.img_mask_overlap

        self.latent_map_size = tuple(
            self.latent_size + (s - 1) * self.stride for s in self.sampling_steps
        )
        self.img_map_size = tuple(s * self.f_vae for s in self.latent_map_size)

        self.latent_map_height = self.latent_map_size[0]
        self.latent_map_width = self.latent_map_size[1]

        self.extended_ctxt_size = tuple(
            s + self.image_size * (self.f_ext_ctxt - 1) for s in self.img_map_size
        )

    def set(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self._refresh_derived()
        return self

    def copy(self, **kw):
        data = {**vars(self), **kw}
        return ResampleSettings(**data)


DEFAULT_SETTINGS = ResampleSettings()


def auto_kwargs(func, namespace, run=True, **other_kwargs):
    """
    Extract matching arguments from a SimpleNamespace based on a function's signature.
    Works with both functions and classes (inspects __init__ for classes).

    Args:
        func: A function or class constructor to inspect
        namespace: A SimpleNamespace containing potential arguments

    Returns:
        A dict with argument names and values from namespace that match func's parameters
    """
    sig = inspect.signature(func)
    namespace_dict = vars(namespace)
    kwargs = {}

    for param_name in sig.parameters:
        if param_name in namespace_dict:
            kwargs[param_name] = namespace_dict[param_name]

    if run:
        kwargs.update(other_kwargs)
        return func(**kwargs)

    return kwargs


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
