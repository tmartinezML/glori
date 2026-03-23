from typing import Literal
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from bdsf.functions import gaussian_fcn

import glori.settings.paths as paths
import glori.data.utils as dutil
from glori.data.obs.prototypes_utils import make_wcs


def recreate_model_image(
    src: pd.Series,
    cmp_cat: pd.DataFrame | Path,
    min_size_arcsec: float = 96,
    wcs: WCS | None = None,
) -> None:
    match cmp_cat:
        case pd.DataFrame():
            cmp_cat = cmp_cat
        case Path():
            if not cmp_cat.exists():
                raise FileNotFoundError(f"Comparison catalog file not found: {cmp_cat}")
            cmp_cat = dutil.read_fits_catalog(cmp_cat)
        case _:
            raise ValueError(
                f"cmp_cat must be a DataFrame or a Path to a FITS file, got {type(cmp_cat)}"
            )

    # Filter for parent source
    cmp_cat = cmp_cat[cmp_cat["Source_Name"] == src["Source_Name"]]

    # Make empty image
    if wcs is not None:
        img = np.zeros(wcs.array_shape, dtype=np.float32)
    else:
        s = max(int(src["Maj"] * 2), min_size_arcsec) // 1.5
        # Make sure it's even after rescaling
        s += s % 8
        img = np.zeros((int(s),) * 2, dtype=np.float32)
        wcs = make_wcs(img, src)
    y, x = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]

    # Fill with Gaussians
    for _, row in cmp_cat.iterrows():
        """
        # Position in pixels, relative to source center
        x0 = (
            -(src_coord.ra.arcsec - row["RA"] * 3600) / 1.5 + img.shape[0] // 2
        )  # arcsec
        y0 = (src_coord.dec.arcsec - row["DEC"] * 3600) / 1.5 + img.shape[
            1
        ] // 2  # arcsec
        """
        x0, y0 = wcs.all_world2pix([[row["RA"], row["DEC"]]], 0)[0]

        g = [
            row["Peak_flux"] * 1e-3,  # Amplitude
            x0,  # x center
            y0,  # y center
            row["Maj"] / 1.5,  # x stddev (FWHM to stddev)
            row["Min"] / 1.5,  # y stddev
            row["PA"],  # theta in deg
        ]

        img += gaussian_fcn(g, x, y)

    return img


def get_model_image(
    src: pd.Series, which: Literal["DDF", "PyBDSF"] = "DDF", min_size_arcsec=96
) -> None:
    """Get the model image for a source from the specified model type.

    Parameters
    ----------
    src : pd.Series
        A row from the LOFAR catalog containing source information.
    which : Literal["DDF", "PyBDSF"]
        The type of model to retrieve. "DDF" for direction-dependent calibration model,
        "PyBDSF" for PyBDSF Gaussian model.

    Returns
    -------
    np.ndarray or None
        The model image array if available, otherwise None.
    """
    match which:
        case "DDF":
            path = (
                paths.MODEL_DIR_DR2
                / src["Mosaic_ID"]
                / "images/image_full_ampphase_di_m.NS.int.model.fits"
            )
        case "PyBDSF":
            path = paths.MODEL_DIR_DR2 / src["Mosaic_ID"] / "mosaic.bdsf_model.fits"
        case _:
            raise ValueError(f"Unknown model type: {which}")

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    img, wcs = dutil.load_fits_image(path, get_wcs=True)
    s = max(int(src["Maj"] * 2), min_size_arcsec)
    cutout = Cutout2D(
        img.squeeze(),
        position=SkyCoord(ra=src["RA"], dec=src["DEC"], unit="deg"),
        size=u.Quantity((s, s), u.arcsec),
        wcs=wcs,
    )
    return cutout.data, cutout.wcs
