import logging
import os
import sys
import signal
import shutil
import psutil
import pickle
import argparse
import warnings
import tempfile
import traceback
import multiprocessing as mp
from pathlib import Path
from numbers import Number
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor as PPEx


import bdsf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from scipy.optimize import linear_sum_assignment
from photutils.aperture import SkyEllipticalAperture
from astropy import units as u

from utils.devices import is_jupyter
import utils.paths as paths
from analysis.sourcefind import run_tiered_bdsf, flatten
from data.utils import (
    load_fits_image,
    load_fits_catalog,
)


if is_jupyter():
    bdsf.functions.getTerminalSize = lambda: (80, 24)  # Mock terminal size


import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import linear_sum_assignment

DEFAULT_KWARGS = dict(
    thresh_isl=3.0,
    thresh_pix=4.5,  # 4.0 in the paper
    thresh=None,
    rms_box=(150, 15),
    rms_map=True,
    mean_map="zero",
    ini_method="intensity",
    adaptive_rms_box=True,
    adaptive_thresh=150,
    rms_box_bright=(60, 15),
    group_by_isl=False,
    group_tol=10.0,
    output_opts=False,
    output_all=False,
    atrous_do=True,
    atrous_jmax=4,
    flagging_opts=True,
    flag_maxsize_fwhm=0.5,
    advanced_opts=True,
    blank_limit=None,
    frequency=144e6,
    debug=False,
    quiet=False,
)


def match_catalogs(cat1_df, cat2_df, thr_arcsec=50, ra_col="RA", dec_col="DEC"):
    """
    Match two catalogs (pandas DataFrames) using Hungarian algorithm, avoiding duplicate matches.

    Parameters
    ----------
    cat1_df, cat2_df : pandas.DataFrame
        Must contain RA/Dec columns (in degrees).
    max_radius : Quantity
        Maximum allowed separation for a match.
    ra_col, dec_col : str
        Column names for RA/Dec in the dataframes.

    Returns
    -------
    dict
        {
            "matched_cat1": matched_cat1_df,
            "matched_cat2": matched_cat2_df,
            "unmatched_cat1": unmatched_cat1_df,
            "unmatched_cat2": unmatched_cat2_df,
            "match_mask": match_mask,
            "idx": idx,
            "d2d": d2d
        }
    """
    # Convert to SkyCoord
    cat1_coords = SkyCoord(
        cat1_df[ra_col].values * u.deg, cat1_df[dec_col].values * u.deg
    )
    cat2_coords = SkyCoord(
        cat2_df[ra_col].values * u.deg, cat2_df[dec_col].values * u.deg
    )

    # Compute separation matrix in arcsec
    sep_matrix = cat1_coords[:, None].separation(cat2_coords[None, :]).arcsec

    # Pad to square for Hungarian algorithm
    n1, n2 = sep_matrix.shape
    max_size = max(n1, n2)
    cost_matrix = np.full((max_size, max_size), 1e9)
    cost_matrix[:n1, :n2] = sep_matrix

    # Hungarian solve
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Initialize match arrays
    idx = np.full(n1, np.nan)  # index in cat2 for each cat1
    d2d = np.full(n1, np.nan)  # separation in arcsec

    for r, c in zip(row_ind, col_ind):
        if r < n1 and c < n2:
            idx[r] = c
            d2d[r] = sep_matrix[r, c]

    # Match mask based on thr_arcsec
    match_mask = d2d <= thr_arcsec

    # Matched DataFrames
    matched_cat1 = cat1_df[match_mask].reset_index(drop=True)
    matched_idx = idx[match_mask].astype(int)
    matched_cat2 = cat2_df.iloc[matched_idx].reset_index(drop=True)

    # Unmatched DataFrames
    unmatched_cat1 = cat1_df[~match_mask].reset_index(drop=True)
    unmatched_idx_cat2 = np.setdiff1d(np.arange(n2), matched_idx)
    unmatched_cat2 = cat2_df.iloc[unmatched_idx_cat2].reset_index(drop=True)

    # Convert d2d to astropy Quantity (arcsec)
    d2d_quantity = d2d * u.arcsec

    return {
        "matched_cat1": matched_cat1,
        "matched_cat2": matched_cat2,
        "unmatched_cat1": unmatched_cat1,
        "unmatched_cat2": unmatched_cat2,
        "match_mask": match_mask,
        "idx": idx,
        "d2d": d2d_quantity,
    }


def match_catalogs_nn(cat1, cat2, thr_arcsec=12):
    """
    Match two catalogs based on RA and DEC within a given threshold in arcseconds.
    Returns a DataFrame with matched sources and their differences.
    """
    cat1_coords = SkyCoord(ra=cat1.RA.values, dec=cat1.DEC.values, unit="deg")
    cat2_coords = SkyCoord(ra=cat2.RA.values, dec=cat2.DEC.values, unit="deg")

    idx, d2d, _ = cat1_coords.match_to_catalog_sky(cat2_coords)
    match_mask = d2d.arcsec < thr_arcsec
    # If there are duplicates, keep the one with the smallest distance
    i, c = np.unique(idx[match_mask], return_counts=True)
    if np.any(c > 1):
        # If there are duplicates, keep the one with the smallest distance
        match_mask[np.argwhere(np.isin(idx, i[c > 1]))] = False
        i_smallest = np.array(
            [np.argwhere(idx == ii)[np.argmin(d2d[ii])] for ii in i[c > 1]]
        )
        # Remove duplicates from the match mask
        match_mask[i_smallest] = True

    match_mask_cat2 = np.zeros(len(cat2), dtype=bool)
    match_mask_cat2[idx[match_mask]] = True
    matched_cat1 = cat1[match_mask]
    matched_cat2 = cat2[match_mask_cat2]
    unmatched_cat1 = cat1[~match_mask]
    unmatched_cat2 = cat2[~match_mask_cat2]
    return {
        "matched_cat1": matched_cat1,
        "matched_cat2": matched_cat2,
        "unmatched_cat1": unmatched_cat1,
        "unmatched_cat2": unmatched_cat2,
        "match_mask": match_mask,
        "idx": idx,
        "d2d": d2d,
    }


def tiered_bdsf_wrapper(img, wcs=None, tmpdir=paths.ANALYSIS_PARENT / "tmp", **kwargs):
    """
    Wrapper for the tiered BDSF source finding algorithm.
    """
    # Create output directory
    with tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".fits") as tmpfile:

        img_fpath = tmpfile.name

        # Save image to fits file
        hdu = fits.PrimaryHDU(data=img, header=fits.Header(make_header_dict(wcs=wcs)))
        hdu.writeto(img_fpath, overwrite=True)

        # Run BDSF on the image
        srl_f, gaul_f, model_f, resid_f = run_tiered_bdsf(
            img_fpath, img_fpath, **kwargs
        )

        # Make output dict
        srl = load_fits_catalog(srl_f)
        gaul = load_fits_catalog(gaul_f)
        # Bring to desired units
        for cat in [srl, gaul]:
            cat["Total_flux"] *= 1e3
            cat["Peak_flux"] *= 1e3
            cat["Maj"] *= 3600  # Convert to arcsec

        with fits.open(model_f) as hdul:
            hdu_flat = flatten(hdul)
            model_img = hdu_flat.data
            wcs = WCS(hdu_flat.header, naxis=2)

        resid_img, _ = load_fits_image(resid_f, get_wcs=False)

        output = {
            "input_image": img,
            "catalogs": {"srl": srl, "gaul": gaul},
            "model_gaus_arr": model_img.squeeze(),
            "resid_gaus_arr": resid_img.squeeze(),
            "wcs": wcs,
        }

    # Remove files
    os.remove(srl_f)
    os.remove(gaul_f)
    os.remove(model_f)
    os.remove(resid_f)
    for f in Path(tmpdir).glob(f"{Path(img_fpath).stem}*"):
        f.unlink()

    return output


def multistep_bdsf_on_image(
    img,
    snr_thresh_bright=150,
    mean_rms_maps=None,
    tmpdir=paths.ANALYSIS_PARENT / "tmp",
    wcs=None,
    **kw,
):
    # Replicated algorithm from Shimwell+25, App. A
    default_kwargs = DEFAULT_KWARGS.copy()
    default_kwargs.update(kw)

    # Step 1 and 2 are in order to obtain the deep rms map
    if mean_rms_maps is None:
        # Step 1: Run bdsf with settings on image
        res1 = bdsf_on_image(img, wcs=wcs, **default_kwargs)

        # Step 2: Make residual image that keeps bright sources:
        # -- A: Construct mask for bright sources
        bright_mask = np.zeros_like(res1.resid_gaus_arr)
        if wcs is None:
            wcs = WCS(res1.wcs_obj.to_header())
        for src in res1.sources:
            if src.peak_flux_max / src.rms_isl >= snr_thresh_bright:
                for g in src.gaussians:
                    # Get gaussian paremeters
                    ra, dec = g.centre_sky
                    maj, min, pa = g.size_sky
                    # Construct mask and add to bright_mask image
                    position = SkyCoord(ra=ra, dec=dec, unit="deg")
                    aper = SkyEllipticalAperture(
                        position,
                        1.5 * maj * u.degree,
                        1.5 * min * u.degree,
                        pa * u.degree,
                    )
                    bright_mask += (
                        aper.to_pixel(wcs).to_mask().to_image(bright_mask.shape[-2:])
                    )
        bright_mask = (bright_mask != 0).astype(np.float32).T
        # -- B: Remove bright sources from model image
        model_no_bright = res1.model_gaus_arr.T * (1 - bright_mask)
        # -- C: Remove that from the original image to obtain residual image
        #       with bright sources in it.
        resid_map_with_bright = res1.image_arr.squeeze().T - model_no_bright

        # Step 3: Run bdsf on the residual image with bright sources,
        # which gives a deeper rms map
        rms_deep_res = run_rms_map(resid_map_with_bright, wcs=wcs, **default_kwargs)
        rms_map_deep = rms_deep_res.rms_arr.T
        # mean_map_deep = rms_deep_res.mean_arr.T
        mean_map_deep = np.zeros_like(rms_map_deep)

    else:
        mean_map_deep, rms_map_deep = (m.squeeze() for m in mean_rms_maps)

    # Step 4: Run bdsf on the image with the deeper rms map
    res2 = bdsf_on_image(
        img,
        wcs=wcs,
        tmpdir=tmpdir,
        mean_rms_maps=(mean_map_deep, rms_map_deep),
        **default_kwargs,
    )

    # Step 5: Run bdsf again on deep residual image
    resid_map_deep = res2.resid_gaus_arr.copy()
    default_kwargs.update(dict(thresh_pix=10, flag_maxsize_bm=100))
    res3 = bdsf_on_image(
        resid_map_deep.T,
        wcs=wcs,
        tmpdir=tmpdir,
        mean_rms_maps=(mean_map_deep, rms_map_deep),
        **default_kwargs,
    )

    # Step 6: Return list of sources and catalogs
    catalogs = {
        cat_type: pd.concat(
            [catalogs_from_bdsf(r)[cat_type] for r in [res2, res3]], ignore_index=True
        ).reset_index(drop=True)
        for cat_type in ["gaul", "srl"]
    }

    # Put everything into output dict
    output = {
        "catalogs": catalogs,
        "model_gaus_arr": res2.model_gaus_arr + res3.model_gaus_arr,
        "resid_gaus_arr": res3.resid_gaus_arr,
        "wcs": wcs,
    }
    return output


def save_multistep_output(output, name, out_parent=paths.ANALYSIS_PARENT / "bdsf"):
    out_dir = out_parent / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save image as npy
    np.save(out_dir / "input_image.npy", output["input_image"])

    # Save products as fits
    for key in ["model_gaus_arr", "resid_gaus_arr"]:
        fpath = out_dir / f"{key}.fits"
        hdu = fits.PrimaryHDU(data=output[key], header=output["wcs"].to_header())
        hdu.writeto(fpath, overwrite=True)

    # Save catalogs as parquet
    for cat_type, catalog in output["catalogs"].items():
        fpath = out_dir / f"catalog_{cat_type}.parquet"
        if fpath.exists():
            fpath.unlink()
        catalog.to_parquet(fpath, index=False)


def load_multistep_output(
    name, out_parent=paths.ANALYSIS_PARENT / "bdsf", empty_is_err=True
):
    # If name is absolute path, out_parent will be ignored in the following line
    out_dir = out_parent / name
    if not out_dir.exists():
        if empty_is_err:
            raise FileNotFoundError(f"Output directory {out_dir} does not exist.")
        else:
            # Return dict with all entries None
            d = {
                k: None
                for k in [
                    "input_image",
                    "model_gaus_arr",
                    "resid_gaus_arr",
                    "wcs",
                ]
            }
            d["catalogs"] = {cat_type: None for cat_type in ["gaul", "srl"]}
            return d

    output = {}

    # Load input image
    img_fpath = out_dir / "input_image.npy"
    if not img_fpath.exists():
        raise FileNotFoundError(f"Input image {img_fpath} does not exist.")
    output["input_image"] = np.load(img_fpath)

    # Load images from fits
    for key in ["model_gaus_arr", "resid_gaus_arr"]:
        fpath = out_dir / f"{key}.fits"
        with fits.open(fpath) as hdul:
            output[key] = hdul[0].data

    # Load catalogs from parquet
    output["catalogs"] = {
        cat_type: pd.read_parquet(out_dir / f"catalog_{cat_type}.parquet")
        for cat_type in ["gaul", "srl"]
    }

    # Load WCS from fits header
    with fits.open(out_dir / "model_gaus_arr.fits") as hdul:
        output["wcs"] = WCS(hdul[0].header)

    return output


def run_rms_map(
    img,
    tmpdir=paths.ANALYSIS_PARENT / "tmp",
    beam_size_arcsec=6,
    wcs=None,
    set_quiet=True,
    **bdsf_kwargs,
):

    img = img.squeeze()

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".fits", dir=tmpdir) as f:

        # Write the hdu to tmp fits file
        write_to_fits(img, f.name, wcs=wcs)

        # make bdsf op list
        op_list = [
            bdsf.readimage.Op_readimage,
            bdsf.collapse.Op_collapse,
            bdsf.preprocess.Op_preprocess,
            bdsf.rmsimage.Op_rmsimage,
        ]

        img = bdsf.image.Image({"filename": f.name})
        logging.USERINFO = logging.INFO + 1
        img.log = str(Path(f.name).with_suffix(".pybdsf.log"))
        img.opts.quiet = set_quiet
        img.opts.stopat = "isl"
        img.opts.beam = (beam_size_arcsec / 3600, beam_size_arcsec / 3600, 0)
        if len(bdsf_kwargs) > 0:
            bdsf.interface.set_pars(img, **bdsf_kwargs)
        bdsf._run_op_list(img, op_list)

        # os.remove(f"{f.name}.pybdsf.log")

    return img





def bdsf_on_image(
    img: np.ndarray,
    wcs=None,
    mean_rms_maps=None,
    px_size_arcsec=1.5,
    beam_size_arcsec=6,  # 6 arcsec
    id=None,
    tmpdir=paths.ANALYSIS_PARENT / "tmp",
    logdir=None,
    add_noise=False,
    **bdsf_kwargs,
):
    """
    Run bdsf on a single image.
    """
    img = img.squeeze()

    if mean_rms_maps is not None:
        mean_map, rms_map = (m.squeeze() for m in mean_rms_maps)

    # Add small amount of noise, otherwise sigma-clipping algorithm called
    # within bdsf.process_image (functions.bstat) might not converge
    if add_noise:
        z = np.random.normal(0, scale=min(img.max(), 1) * 1e-2, size=img.shape)
        img += z

    # Create a temporary file
    with (
        tempfile.NamedTemporaryFile(prefix=id, suffix=".fits", dir=tmpdir) as f,
        tempfile.NamedTemporaryFile(
            prefix=id, suffix=".mean.fits", dir=tmpdir
        ) as f_mean,
        tempfile.NamedTemporaryFile(prefix=id, suffix=".rms.fits", dir=tmpdir) as f_rms,
    ):

        # Write the hdu to tmp fits file
        write_to_fits(img, f.name, wcs=wcs, px_size_arcsec=px_size_arcsec)

        beam_size = beam_size_arcsec / 3600
        kwargs = {
            "beam": (beam_size, beam_size, 0),
            "thresh_isl": 5,
            "thresh_pix": 0.5,
            "mean_map": "const",
            "rms_map": True,
            "thresh": "hard",
            "quiet": True,
            "debug": True,
            "frequency": 144e6,  # Default frequency in Hz
        }

        if mean_rms_maps is not None:
            # Write mean and rms maps to temporary files
            write_to_fits(mean_map, f_mean.name, wcs=wcs, px_size_arcsec=px_size_arcsec)
            write_to_fits(rms_map, f_rms.name, wcs=wcs, px_size_arcsec=px_size_arcsec)
            kwargs["rmsmean_map_filename"] = [
                os.path.basename(f_mean.name),
                os.path.basename(f_rms.name),
            ]
        kwargs.update(bdsf_kwargs)

        beam_size = beam_size_arcsec / 3600
        img = bdsf.process_image(
            f.name,
            **kwargs,
        )

        # (Re)move log file
        if logdir is not None:
            os.rename(tmpdir / f"{f.name}.pybdsf.log", logdir / f"{id}.pybdsf.log")
        else:
            os.remove(f"{f.name}.pybdsf.log")

    # Return bdsf image object
    return img


def write_to_fits(img, fpath, wcs=None, **header_kw):
    hdu = fits.PrimaryHDU(
        data=img,
        header=(
            wcs.to_header()
            if wcs is not None
            else fits.Header(make_header_dict(**header_kw))
        ),
    )
    fits.HDUList([hdu]).writeto(fpath, overwrite=True)


def make_header_dict(
    px_size_arcsec=1.5, freq_hz=144e6, wcs=None, beam_size_arcsec=6, **header_kw
):
    """
    Create a header dictionary for bdsf.
    """
    d = {
        "CDELT1": -px_size_arcsec / 3600,  # Pixel size in deg (1.5 arcsec)
        "CUNIT1": "deg",
        "CTYPE1": "RA---SIN",
        "CDELT2": px_size_arcsec / 3600,
        "CUNIT2": "deg",
        "CTYPE2": "DEC--SIN",
        "CRVAL4": freq_hz,  # Frequency in Hz
        "CUNIT4": "HZ",
        "CTYPE4": "FREQ",
        "HISTORY": "Created manually by utils.analysis.bdsf_analysis.make_header_dict",
        "BMAJ": beam_size_arcsec / 3600,
        "BMIN": beam_size_arcsec / 3600,
        "BPA": 90,
    }
    if wcs is not None:
        d.update(wcs.to_header())
    d.update(header_kw)
    return d


def catalogs_from_bdsf(
    img: bdsf.image.Image,
    tmpdir=paths.ANALYSIS_PARENT / "tmp",
):
    cat_dict = {}

    # Loop through both cat types
    for cat_type in ["gaul", "srl"]:

        # Create a temporary csv file
        with tempfile.NamedTemporaryFile(suffix=".csv", dir=tmpdir) as f:

            # Write the catalog to the csv file
            with open(os.devnull, "w") as devnull:
                # Supress all output from this process.
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    img.write_catalog(
                        outfile=f.name,
                        clobber=True,
                        catalog_type=cat_type,
                        format="csv",
                    )
                finally:
                    sys.stdout = old_stdout

            # Read the csv file
            try:
                catalog = pd.read_csv(f.name, skiprows=5, skipinitialspace=True)

            except pd.errors.EmptyDataError:
                catalog = pd.DataFrame({})

            # Append to list
            cat_dict[cat_type] = catalog

    return cat_dict


def dict_from_bdsf(img: bdsf.image.Image):
    # Get all attributes of the bdsf image object
    # that are numbers or numpy arrays
    d = {
        k: v
        for k, v in img.__dict__.items()
        if isinstance(v, Number) or isinstance(v, np.ndarray)
    }
    return d


def append_to_pickle(obj, fpath):
    if fpath.exists():
        with open(fpath, "rb") as f:
            data = pickle.load(f)
    else:
        data = []
    data.append(obj)
    with open(fpath, "wb") as f:
        pickle.dump(data, f)


def write_to_pickle(obj, fpath):
    Image_id = obj["Image_id"]
    with open(fpath / f"{Image_id}.pkl", "wb") as f:
        pickle.dump(obj, f)


def append_to_csv(df, fpath):
    df.to_csv(fpath, index=False, mode="a", header=not fpath.exists())


def writer(queue, q_pbar, fpaths, write_fns):
    print("Writer started.")
    log_file = fpaths[0].parent / "writer_log.txt"
    i = 0

    def signal_handler(sig, frame):
        print("Keyboard interrupt ignored by writer.")

    signal.signal(signal.SIGINT, signal_handler)
    while True:
        items = queue.get()
        if items == -1:
            print("Closing writer.")
            break

        elif items == 0:
            i += 1
            q_pbar.put(1)
            continue

        else:
            for item, fpath, write_fn in zip(items, fpaths, write_fns):
                if item is None:
                    continue
                write_fn(item, fpath)

            # Write log
            img_id = items[0]["Image_id"]
            with open(log_file, "a" if i else "w") as f:
                f.write(f"{i},{img_id}\n")

            i += 1
            q_pbar.put(1)


def kill_child_processes(parent_pid, sig=signal.SIGTERM):

    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)

    for process in children:
        try:
            process.send_signal(sig)
        except psutil.NoSuchProcess:
            pass


def bdsf_worker_func(img, image_id, q, bdsf_kwargs):

    # Handle keyboard interrupts
    def signal_handler(sig, frame):
        pid = os.getpid()
        print(f"Keyboard interrupt handeled by worker process {pid}.")
        return

    signal.signal(signal.SIGINT, signal_handler)

    # Run bdsf on the image
    bdsf_img = bdsf_on_image(img, id=image_id, **bdsf_kwargs)

    # bdsf dict:
    # Retrieve the desired attributes of the bdsf image object as dict
    attr_dict = dict_from_bdsf(bdsf_img)
    bdsf_dict = {
        k: attr_dict[k]
        for k in [
            "model_gaus_arr",
            "total_flux_gaus",
            "island_labels",
            "ngaus",
            "nsrc",
        ]
    }
    # Add image_id to the dict
    bdsf_dict["Image_id"] = image_id

    # Catalogs:
    # Retrieve the catalogs of the bdsf image object
    cats = catalogs_from_bdsf(bdsf_img)  # cat_g, cat_s
    for cat in cats:
        if not cat.empty:
            cat["Image_id"] = image_id

    # Append to writer queue
    q.put([bdsf_dict, *cats])


def bdsf_wrapper(*args):
    try:
        bdsf_worker_func(*args)
    except Exception as e:
        args[2].put(0)
        raise e


def iterable_func_caller(func, args):
    return func(*args)


def progress_bar_worker(q_pbar, n_tot):
    print("Progress bar worker started.")

    def signal_handler(sig, frame):
        print("Keyboard interrupt ignored by progress bar worker.")
        q_pbar.put(-1)

    signal.signal(signal.SIGINT, signal_handler)
    with tqdm(total=n_tot, smoothing=0.1) as pbar:
        while True:
            sig = q_pbar.get()
            match sig:
                case 0:
                    print("Test signal received.")
                case -1:
                    print("Closing progress bar.")
                    break
                case 1:
                    pbar.update(1)
                case _:
                    print(f"Unknown signal: {sig}")


def bdsf_run(
    imgs: Iterable,
    out_folder: str | Path,
    out_parent=paths.ANALYSIS_PARENT,
    names: Iterable[str] = None,
    override=False,
    max_workers=96,
    **bdsf_kwargs,
):
    """
    Run bdsf on a set of images and save the resulting list as pickle file.
    """
    # Set tmp directory
    tmp_dir = paths.ANALYSIS_PARENT / "tmp"
    os.environ["TMPDIR"] = str(tmp_dir)
    warnings.filterwarnings("ignore")

    # Set up the output folder
    out_path = out_parent / out_folder / "bdsf"
    out_path.mkdir(exist_ok=True)

    # Set up the output file paths
    dicts_path = out_path / f"dicts"
    logs_path = out_path / f"logs"
    err_path = out_path / f"errors"
    gaul_path = out_path / f"bdsf_gaul.csv"
    srl_path = out_path / f"bdsf_srl.csv"

    # If override is True, delete the files if they exist
    if override:
        for fpath in [
            dicts_path,
            logs_path,
            err_path,
            gaul_path,
            srl_path,
        ]:
            if fpath.exists():
                fpath.unlink() if fpath.is_file() else shutil.rmtree(fpath)

    # Look for images already processed
    else:
        print("Looking for images already processed...")

        if dicts_path.exists():
            # Open pickle files in dicts_path
            pkl_files = list(dicts_path.glob("*.pkl"))
            ids = np.array([f.stem for f in pkl_files])
            print(f"Removing {len(ids)} processed images from the list.")
            mask = np.in1d(names, ids, invert=True)
            imgs = imgs[mask]
            names = names[mask]
        else:
            print("No images processed yet.")

    logs_path.mkdir(exist_ok=True)
    err_path.mkdir(exist_ok=True)
    dicts_path.mkdir(exist_ok=True)

    # This will assign an id to every image/job
    print("Getting image ids...")
    image_ids = names if names is not None else range(len(imgs))
    image_ids = [str(i) for i in image_ids]

    # Prepare helper processes: writer and progress bar
    manager = mp.Manager()
    q = manager.Queue()
    q_pbar = manager.Queue()
    helper_pool = mp.Pool(2)

    # Start the writer process
    w_res = helper_pool.apply_async(
        writer,
        (
            q,
            q_pbar,
            [dicts_path, gaul_path, srl_path],
            [write_to_pickle, append_to_csv, append_to_csv],
        ),
    )

    # Start the progress bar process
    pbar_res = helper_pool.apply_async(progress_bar_worker, (q_pbar, len(imgs)))

    # Launch the worker pool
    with PPEx(max_workers=max_workers) as worker_pool:

        # Handle keyboard interrupts
        def signal_handler(sig, frame):
            print("Keyboard interrupt. Closing pools.")
            worker_pool.shutdown(cancel_futures=True, wait=False)
            q.put(-1)
            q_pbar.put(-1)
            helper_pool.close()
            helper_pool.join()
            kill_child_processes(os.getpid())
            for f in tmp_dir.iterdir():
                if f.suffix in [".log", ".fits"]:
                    f.unlink()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Add log dir to bdsf kwargs
        bdsf_kwargs["logdir"] = logs_path

        # Run bdsf on the images
        print(f"Running bdsf on images. Parent process: {os.getpid()}.")
        """
        res = worker_pool.map(
            partial(iterable_func_caller, bdsf_worker_func),
            zip(imgs, image_ids, [q] * len(imgs), [bdsf_kwargs] * len(imgs)),
            chunksize=10,
        )
        """
        fut = []
        for img, image_id in zip(imgs, image_ids):
            f = worker_pool.submit(bdsf_wrapper, img, image_id, q, bdsf_kwargs)
            fut.append(f)

        # If any exceptions were raised, print to out file
        err_count = 0
        for i, f in enumerate(fut):
            res = f.exception()
            if res is not None:
                with open(err_path / f"{image_ids[i]}.txt", "w") as f:
                    traceback.print_exception(res, file=f)
                err_count += 1
            else:
                # Check if there is an error file from previous runs,
                # if so remove.
                err_file = err_path / f"{image_ids[i]}.txt"
                if err_file.exists():
                    err_file.unlink()

        print(f"{err_count} errors were encountered - check error directory.")

        # [print(r := f.result()) for f in fut if r is not None]
        # [print(r.successful()) for r in res if r is not None]

    # Finish helper processes
    print("Closing pools.")
    q.put(-1)
    q_pbar.put(-1)
    helper_pool.close()
    helper_pool.join()

    # Check if writer was successful
    s = w_res.successful()
    print(f"Writer success: {s}")
    if not s:
        print(w_res.get())

    # Check if progress bar worker was successful
    s = pbar_res.successful()
    print(f"Progress bar worker success: {s}")
    if not s:
        print(pbar_res.get())

    print("Done.")


def bdsf_plot(img, keys=["ch0_arr", "resid_gaus_arr", "model_gaus_arr"]):
    fig = plt.figure(figsize=(10, 15))
    for i, key in enumerate(keys):
        # add subplot
        ax = fig.add_subplot(1, len(keys), i + 1)
        image = getattr(img, key)
        ax.imshow(image.T)
        ax.set_title(key)
        ax.set_axis_off()
    fig.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--override",
        required=False,
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    arguments = parser.parse_args()

    data_path = paths.LOFAR_SUBSETS["200p"]
    out_folder = paths.ANALYSIS_PARENT / data_path.stem
    out_folder.mkdir(exist_ok=True)

    # Load the dataset
    # TODO: Update the dataset to use the new EvaluationDataset class
    # dataset = EvaluationDataset(data_path, img_size=200)

    # For testing: deterministic subset (use None for all images)
    n = None

    bdsf_kwargs = {
        "thresh_isl": 3.5,
        "thresh_pix": 1,
        # 'shapelet_do': True,
        # 'atrous_do': True,
    }

    # Run bdsf on the images
    bdsf_out = bdsf_run(
        # Call dataset for Transform
        np.array([dataset[i] for i in range(n or len(dataset))]),
        out_folder=out_folder,
        names=dataset.names[:n],
        override=arguments.override,
        max_workers=96,
        ang_size=120,
        **bdsf_kwargs,
    )
