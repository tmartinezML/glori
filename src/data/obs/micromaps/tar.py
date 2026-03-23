import io
import gc
import os
import sys
import shutil
import json
import tarfile
import tempfile
import ctypes
import ctypes.util
import traceback
from pathlib import Path
from functools import partial
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import h5py
import torch
import numpy as np
from tqdm import tqdm
from einops import reduce
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from webdataset import WebLoader
from webdataset.writer import TarWriter
from skimage.transform import rescale

from data.obs.micromaps.utils import (
    get_micromaps,
    single_micromap_cutout,
    filter_catalog_by_wcs,
    graceful_shutdown,
    reduce_context_map,
    context_map_by_wcs,
    get_mosaic_shape,
)
import utils.paths as paths
import analysis.bdsf_analysis as bdsf_analysis
import utils.my_logging as my_logging
from models.vae.vqvae import VQVAE
from data.utils import parse_dset_path
from data.trf.scalers import ContextScaler
from maps.map_utils import get_map_image
from deprecated.micromap_webdataset import MicromapPreparationDataset
from data.trf.image_utils import apply_restoring_beam, center_crop_numpy


logger = my_logging.get_logger(__name__)

# Limit Python to use only the first N CPUs
Ncpu = 32
os.sched_setaffinity(0, set(range(Ncpu)))

# Check the CPUs available to the process
print("CPUs available:", os.sched_getaffinity(0))


def make_model_contexts(
    dset,
    debug=False,
    model_dir=paths.LOFAR_DATA_PARENT / "pointings",
    mosaic_dir=paths.MOSAIC_DIR_DR2,
    which="DDF",  # 'DDF or 'PyBDSF'
    f_downscale=4,
    img_size=None,
    spacing=None,
    url_istart=0,
    max_workers=64,
):
    logger.divider()
    logger.setLevel("DEBUG" if debug else "INFO")
    logger.info(f"Adding model contexts to {dset} encodings.\n")
    logger.info(f"Parallel processing with {max_workers} workers.")
    logger.debug(
        "Debug mode enabled. Will process only one url from each split, and only two images from each url."
    )

    assert which in [
        "DDF",
        "PyBDSF",
    ], f"Invalid <which> parameter: {which}. Must be 'DDF' or 'PyBDSF'."

    # Get dataset parent directory and create contexts output parent from it
    dset_parent = parse_dset_path(dset, lookup=paths.MICROMAP_SUBSETS)
    enc_parent = dset_parent.with_name(
        dset_parent.name.replace("micromaps", "micromap_encodings")
    )
    out_parent = enc_parent
    if debug:
        out_parent = enc_parent.with_name(
            enc_parent.name.replace(
                "micromap_encodings", "micromap_encodings_contexts_debug"
            )
        )
        logger.debug(f"Debug mode: output directory will be created as {out_parent}.")
        if out_parent.exists():
            logger.debug(
                f"Recursively removing existing directory: \n\t{out_parent}.\n"
            )
            shutil.rmtree(out_parent)
        out_parent.mkdir()

    if not out_parent.exists():
        raise FileNotFoundError(
            f"Input dataset directory {out_parent} does not exist. Please check the dataset path."
        )

    # Get image size from dataset name
    # (used when extracting cutouts from big context array)
    if img_size is None:
        try:
            img_size = int(dset.split("-")[-1].replace("px", ""))
        except (IndexError, ValueError) as e:
            logger.error(
                f"Cannot infer image size from dataset name <{dset}>. Please provide it explicitly."
            )
            raise e
    if spacing is None:
        try:
            spacing = int(dset_parent.name.split("-")[-1].split("=")[-1])
        except (IndexError, ValueError) as e:
            logger.error(
                f"Cannot infer spacing from dataset name <{dset}>. Please provide it explicitly."
            )
            raise e

    # Get all urls from the dataset
    urls = sorted(dset_parent.glob("*/*.tar"))
    if len(urls) == 0:
        raise FileNotFoundError(
            f"No tar files found in {dset_parent}. Please check the dataset path."
        )
    logger.info(f"Found {len(urls)} tar files in\n\t{dset_parent}.")

    if debug:
        logger.debug(f"Debug mode enabled. Processing only one url from each split.")
        urls = []
        urls.append(sorted(dset_parent.glob("train/*.tar"))[0])
        urls.append(sorted(dset_parent.glob("val/*.tar"))[0])
        urls.append(sorted(dset_parent.glob("test/*.tar"))[0])
        url_istart = 0  # Start from the beginning in debug mode

    def process_url(url, i_url):

        # Extract center coordinates for every key into a dictionary
        ctxt_dict = {}
        with tarfile.open(str(url), "r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    # The key is the base name without extension, e.g. "pointing-x-y"
                    key = member.name.split(".")[0]
                    ext = ".".join(member.name.split(".")[1:])
                    if ext == "center.npy":
                        ctxt_dict[key] = np.load(
                            io.BytesIO(tar.extractfile(member).read())
                        )
        if len(ctxt_dict) == 0:
            return RuntimeError(
                f"No context data found in {url} (iteration {i_url}). Something went wrong there."
            )
        centers = np.array(list(ctxt_dict.values())).astype(int)

        # Get the model image for the pointing
        pointing = url.stem
        pointing_dir = model_dir / pointing
        if not pointing_dir.exists():
            raise FileNotFoundError(
                f"Pointing directory {pointing_dir} does not exist. Please check the model data path."
            )
        match which:
            case "DDF":
                model_file = (
                    pointing_dir / "images/image_full_ampphase_di_m.NS.int.model.fits"
                )
            case "PyBDSF":
                model_file = pointing_dir / "mosaic.bdsf_model.fits"

        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file {model_file} does not exist. Please check the model data path."
            )

        model_img = get_map_image(model_file, get_wcs=False).squeeze()
        mshape = get_mosaic_shape(pointing, mosaic_dir=mosaic_dir)
        model_img = center_crop_numpy(model_img, mshape)  # Crop to size of mosaic

        # Extract the model micromaps
        model_mmaps, _, _ = get_micromaps(
            model_img,
            img_size,
            centers=centers,
            pbar=False,
            spacing=spacing,
        )
        assert len(model_mmaps) == len(
            ctxt_dict
        ), f"Number of model micromaps {len(model_mmaps)} does not match number of contexts {len(ctxt_dict)}."

        # Downscale the model micromaps
        model_mmaps_downscaled = rescale(
            model_mmaps,
            1 / f_downscale,
            anti_aliasing=False,
            channel_axis=0,
        )

        # For the DDF model, apply restoring beam
        if which == "DDF":
            model_mmaps_beam = apply_restoring_beam(model_mmaps)
            model_mmaps_beam_downscaled = rescale(
                model_mmaps_beam,
                1 / f_downscale,
                anti_aliasing=False,
                channel_axis=0,
            )

        # Put in a dictionary
        model_dict = {}
        for i, key in enumerate(ctxt_dict.keys()):
            if which == "DDF":
                model_dict[key] = {"model.npy": model_mmaps_downscaled[i]}
                model_dict[key]["model_beam.npy"] = model_mmaps_beam_downscaled[i]
            elif which == "PyBDSF":
                model_dict[key] = {"pybdsf_model.npy": model_mmaps_downscaled[i]}

        # Read all data from encodings in order to add the model micromaps to it
        data_dict = {}
        enc_file = enc_parent / url.relative_to(dset_parent)
        with tarfile.open(str(enc_file), "r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    # The key is the base name without extension, e.g. "pointing-x-y"
                    key = member.name.split(".")[0]
                    ext = ".".join(member.name.split(".")[1:])
                    if key not in data_dict:
                        data_dict[key] = {}
                    fileobj = tar.extractfile(member)
                    if fileobj is not None:
                        if ext.endswith("npy"):
                            arr = np.load(io.BytesIO(fileobj.read()))
                            data_dict[key][ext] = arr
                        else:
                            raise ValueError(
                                f"Unsupported file extension: {ext} for key {key}"
                            )

        # Now append the context to each entry and re-write
        for i, img_key in enumerate(data_dict):
            if img_key not in model_dict:
                raise ValueError(
                    f"Context for key {img_key} not found in context dictionary. "
                    f"Something went wrong with the context extraction."
                )
            data_dict[img_key].update(model_dict[img_key])

        # Write to a temporary file first
        out_file = out_parent / url.relative_to(dset_parent)
        if not out_file.parent.exists():
            out_file.parent.mkdir()

        with tempfile.NamedTemporaryFile(dir=out_file.parent, delete=False) as tmpfile:
            tmp_path = Path(tmpfile.name)
            with TarWriter(str(tmp_path)) as writer:
                for key, data in data_dict.items():
                    sample = {"__key__": key}
                    for ext, arr in data.items():
                        sample[ext] = arr.astype(np.float32)
                    writer.write(sample)

        # Atomically move the temp file to the final destination
        tmp_path.replace(out_file)
        return

    # Process URLs in parallel with progress bar
    logger.info(f"Parallel processing urls.")
    logger.info(f"Starting at index {url_istart} of {len(urls)} urls.")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with graceful_shutdown(executor):
            futures = {
                executor.submit(process_url, url, i + url_istart): url
                for i, url in enumerate(urls[url_istart:])
            }

            for future in tqdm(
                as_completed(futures),
                desc="Processing URLs",
                total=len(urls),
                initial=url_istart,
                dynamic_ncols=True,
                smoothing=0.1,
            ):
                try:
                    result = future.result()
                    # logger.debug(result)
                except Exception as e:
                    if debug:
                        raise e
                    url = futures[future]
                    logger.error(f"Error processing {url}: {e}")
                    if debug:
                        raise e

    logger.divider()
    logger.info(f"Done creating micromaps encodings with contexts dataset.\n")

    return


def add_encodings_contexts_parallel(
    dset,
    debug=False,
    f_downscale=4,
    f_ctxt_size=1,
    img_size=None,
    input_scaled=False,
    scale_context=False,
    url_istart=0,
    url_file=None,
    max_workers=64,
):
    logger.divider()
    logger.setLevel("DEBUG" if debug else "INFO")
    logger.info(f"Adding micromap contexts from dataset {dset} to encodings.\n")
    logger.info(f"Parallel processing with {max_workers} workers.")
    logger.debug(
        "Debug mode enabled. Will process only one url from each split, and only two images from each url."
    )

    # Get dataset parent directory and create contexts output parent from it
    dset_parent = parse_dset_path(dset, lookup=paths.MICROMAP_SUBSETS)
    enc_parent = dset_parent.with_name(
        dset_parent.name.replace("micromaps", "micromap_encodings")
    )
    out_parent = enc_parent
    if debug:
        out_parent = enc_parent.with_name(
            enc_parent.name.replace(
                "micromap_encodings", "micromap_encodings_contexts_debug"
            )
        )
        logger.debug(f"Debug mode: output directory will be created as {out_parent}.")
        if out_parent.exists():
            logger.debug(
                f"Recursively removing existing directory: \n\t{out_parent}.\n"
            )
            shutil.rmtree(out_parent)
        out_parent.mkdir()

    if not out_parent.exists():
        raise FileNotFoundError(
            f"Input dataset directory {out_parent} does not exist. Please check the dataset path."
        )

    # Initialize the scalers
    scaler_ftot = ContextScaler.load("ctxt_scaler_ftot")
    scaler_fpeak = ContextScaler.load("ctxt_scaler_fpeak")
    scaler_maj = ContextScaler.load("ctxt_scaler_maj")
    scalers = [scaler_ftot, scaler_fpeak, scaler_maj]

    # Get image size from dataset name
    # (used when extracting cutouts from big context array)
    if img_size is None:
        try:
            img_size = int(dset.split("-")[-1].replace("px", ""))
        except (IndexError, ValueError) as e:
            logger.error(
                f"Cannot infer image size from dataset name <{dset}>. Please provide it explicitly."
            )
            raise e

    # Get all urls from the dataset
    urls = sorted(dset_parent.glob("*/*.tar"))
    if len(urls) == 0:
        raise FileNotFoundError(
            f"No tar files found in {dset_parent}. Please check the dataset path."
        )
    logger.info(f"Found {len(urls)} tar files in\n\t{dset_parent}.")

    if url_file is not None:
        # Load urls from file
        if not (dset_parent / url_file).exists():
            raise FileNotFoundError(
                f"URL file {dset_parent / url_file} does not exist. Please check the file path."
            )
        with open(dset_parent / url_file, "r") as f:
            url_list = [line.strip() for line in f.readlines() if line.strip()]
        urls = [Path(url) for url in url_list]
        logger.info(f"Loaded {len(urls)} urls from file {url_file} to be processed.")

    if debug:
        logger.debug(f"Debug mode enabled. Processing only one url from each split.")
        urls = []
        urls.append(sorted(dset_parent.glob("train/*.tar"))[0])
        urls.append(sorted(dset_parent.glob("val/*.tar"))[0])
        urls.append(sorted(dset_parent.glob("test/*.tar"))[0])
        url_istart = 0  # Start from the beginning in debug mode

    def process_url(url, i_url):
        # Get contexts from file
        # Extract context data for every key into a dictionary
        ctxt_dict = {}
        with tarfile.open(str(url), "r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    # The key is the base name without extension, e.g. "pointing-x-y"
                    key = member.name.split(".")[0]
                    ext = ".".join(member.name.split(".")[1:])
                    if ext == (
                        "context.npy"
                        if f_ctxt_size == 1
                        else f"context_f={f_ctxt_size}.npy"
                    ):
                        ctxt_dict[key] = np.load(
                            io.BytesIO(tar.extractfile(member).read())
                        )

        if len(ctxt_dict) == 0:
            return RuntimeError(
                f"No context data found in {url} (iteration {i_url}). Something went wrong there."
            )

        # Downsample the contexts
        ctxt_dict_downscaled = {}
        for key, arr in ctxt_dict.items():
            ctxt_dict_downscaled[key] = reduce_context_map(
                arr,
                scalers=scalers,
                f_downscale=f_downscale * f_ctxt_size,
                input_scaled=input_scaled,
                scale_output=scale_context,
            )

        # Read the contents of the encodings tar file
        enc_file = enc_parent / url.relative_to(dset_parent)
        if not enc_file.exists():
            raise FileNotFoundError(
                f"Encodings file {enc_file} does not exist. Please check the dataset path."
            )

        data_dict = {}
        with tarfile.open(str(enc_file), "r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    # The key is the base name without extension, e.g. "pointing-x-y"
                    key = member.name.split(".")[0]
                    ext = ".".join(member.name.split(".")[1:])
                    if key not in data_dict:
                        data_dict[key] = {}
                    fileobj = tar.extractfile(member)
                    if fileobj is not None:
                        if ext.endswith("npy"):
                            arr = np.load(io.BytesIO(fileobj.read()))
                            data_dict[key][ext] = arr
                        else:
                            raise ValueError(
                                f"Unsupported file extension: {ext} for key {key}"
                            )

        # Now append the context to each entry and re-write
        for img_key in data_dict:
            if img_key not in ctxt_dict_downscaled:
                raise ValueError(
                    f"Context for key {img_key} not found in context dictionary. "
                    f"Something went wrong with the context extraction."
                )
            data_dict[img_key][
                (
                    "context_downscaled.npy"
                    if (f_ctxt_size == 1 and f_downscale == 4)
                    else f"context_downscaled_d={f_downscale}_f={f_ctxt_size}.npy"
                )
            ] = ctxt_dict_downscaled[img_key]

        # Write to a temporary file first
        out_file = out_parent / url.relative_to(dset_parent)
        if not out_file.parent.exists():
            out_file.parent.mkdir()

        with tempfile.NamedTemporaryFile(dir=out_file.parent, delete=False) as tmpfile:
            tmp_path = Path(tmpfile.name)
            with TarWriter(str(tmp_path)) as writer:
                for key, data in data_dict.items():
                    sample = {"__key__": key}
                    for ext, arr in data.items():
                        sample[ext] = arr.astype(np.float32)
                    writer.write(sample)

        # Atomically move the temp file to the final destination
        tmp_path.replace(out_file)
        return

    # Process URLs in parallel with progress bar
    logger.info(f"Parallel processing urls.")
    logger.info(f"Starting at index {url_istart} of {len(urls)} urls.")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with graceful_shutdown(executor):
            futures = {
                executor.submit(process_url, url, i + url_istart): url
                for i, url in enumerate(urls[url_istart:])
            }

            for future in tqdm(
                as_completed(futures),
                desc="Processing URLs",
                total=len(urls),
                initial=url_istart,
                dynamic_ncols=True,
                smoothing=0.1,
            ):
                try:
                    result = future.result()
                    logger.debug(result)
                except Exception as e:
                    url = futures[future]
                    logger.error(f"Error processing {url}: {e}")

    logger.divider()
    logger.info(f"Done creating micromaps encodings with contexts dataset.\n")

    return


def create_micromaps_contexts_parallel(
    dset,
    debug=False,
    catalog=paths.LOTSS_DR2_CAT,
    mosaic_dir=paths.MOSAIC_DIR_DR2,
    img_size=None,
    f_ctxt_size=1,
    scale_context=False,
    url_istart=0,
    url_file=None,
    max_workers=64,
):
    logger.divider()
    logger.setLevel("DEBUG" if debug else "INFO")
    logger.info(f"Creating micromaps contexts from dataset {dset}\n")
    logger.info(f"Parallel processing with {max_workers} workers.")
    logger.debug(
        "Debug mode enabled. Will process only one url from each split, and only two images from each url."
    )

    # Get dataset parent directory
    in_parent = parse_dset_path(dset, lookup=paths.MICROMAP_SUBSETS)
    out_parent = in_parent
    if debug:
        # Replace output parent
        out_parent = in_parent.with_name(
            in_parent.name.replace("micromaps", "micromap_contexts_debug")
        )
        logger.debug(f"Debug mode: output directory will be created as {out_parent}.")
        if out_parent.exists():
            logger.debug(
                f"Recursively removing existing directory: \n\t{out_parent}.\n"
            )
            shutil.rmtree(out_parent)
        out_parent.mkdir()
        url_istart = 0  # Start from the beginning in debug mode

    # Raise error if not existing
    if not out_parent.exists():
        raise FileNotFoundError(
            f"Input dataset directory {out_parent} does not exist. Please check the dataset path."
        )

    # Initialize the scalers
    scaler_ftot = ContextScaler.load("ctxt_scaler_ftot")
    scaler_fpeak = ContextScaler.load("ctxt_scaler_fpeak")
    scaler_maj = ContextScaler.load("ctxt_scaler_maj")

    # Get image size from dataset name
    # (used when extracting cutouts from big context array)
    if img_size is None:
        try:
            img_size = int(dset.split("-")[-1].replace("px", ""))
        except (IndexError, ValueError) as e:
            logger.error(
                f"Cannot infer image size from dataset name <{dset}>. Please provide it explicitly."
            )
            raise e

    # Load catalog
    logger.info(f"Loading catalog from: \n\t{catalog}.\n")
    cat = Table.read(catalog).to_pandas()
    # Keep only relevant columns for faster processing
    names = ["RA", "DEC", "Total_flux", "Peak_flux", "Maj"]
    cat = cat[names]
    # Sort by DEC for faster processing
    logger.info(f"Sorting catalog by DEC...")
    cat.sort_values(by="DEC", inplace=True)

    # Get all urls from the dataset
    urls = sorted(in_parent.glob("*/*.tar"))
    if len(urls) == 0:
        raise FileNotFoundError(
            f"No tar files found in {in_parent}. Please check the dataset path."
        )
    logger.info(f"Found {len(urls)} tar files in\n\t{in_parent}.")

    if url_file is not None:
        # Load urls from file
        if not (in_parent / url_file).exists():
            raise FileNotFoundError(
                f"URL file {in_parent / url_file} does not exist. Please check the file path."
            )
        with open(in_parent / url_file, "r") as f:
            url_list = [line.strip() for line in f.readlines() if line.strip()]
        urls = [Path(url) for url in url_list]
        logger.info(f"Loaded {len(urls)} urls from file {url_file} to be processed.")

    if debug:
        logger.debug(f"Debug mode enabled. Processing only one url from each split.")
        urls = []
        urls.append(sorted(in_parent.glob("train/*.tar"))[0])
        urls.append(sorted(in_parent.glob("val/*.tar"))[0])
        urls.append(sorted(in_parent.glob("test/*.tar"))[0])

    def process_url(url):
        # Get image and wcs from mosaic
        mosaic = url.stem
        mosaic_file = mosaic_dir / f"{mosaic}/mosaic-blanked.fits"
        assert mosaic_file.exists(), f"Mosaic file for {mosaic} does not exist."
        with fits.open(mosaic_file) as hdul:
            # img = hdul[0].data
            wcs = WCS(hdul[0].header)

        # Get context map by WCS
        ctxt, sub_cat = context_map_by_wcs(
            wcs,
            cat,
            scalers=[scaler_ftot, scaler_fpeak, scaler_maj],
            scale_output=scale_context,
        )

        # Extract all data for every key into a dictionary
        data_dict = {}
        with tarfile.open(str(url), "r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    # The key is the base name without extension, e.g. "pointing-x-y"
                    key = member.name.split(".")[0]
                    ext = ".".join(member.name.split(".")[1:])
                    if key not in data_dict:
                        data_dict[key] = {}
                    fileobj = tar.extractfile(member)
                    if fileobj is not None:
                        if ext.endswith("npy"):
                            arr = np.load(io.BytesIO(fileobj.read()))
                            data_dict[key][ext] = arr
                        elif ext == "wcs":
                            # Read as utf-8 string
                            data_dict[key][ext] = fileobj.read().decode("utf-8")
                        else:
                            raise ValueError(
                                f"Unsupported file extension: {ext} for key {key}"
                            )

        # Now append the context to each entry and re-write
        if len(data_dict) == 0:
            logger.warning(
                f"No data found in {url}. Something went wrong there. Skipping this file."
            )
            return
        for img_key in data_dict:
            x, y = [int(s) for s in img_key.replace(f"{mosaic}-", "").split("-")]
            micro_ctxt = single_micromap_cutout(
                ctxt.transpose(1, 2, 0),
                int(img_size * f_ctxt_size),
                np.array([x, y]),
                fill_excess=(f_ctxt_size > 1),
            )[0].transpose(2, 0, 1)
            # Make sure shape is correct
            assert micro_ctxt.shape[-2:] == (int(img_size * f_ctxt_size),) * 2, (
                f"Context shape {micro_ctxt.shape[-2:]} does not match expected "
                f"shape {(int(img_size * f_ctxt_size),) * 2} for key {img_key} in url {url}."
            )
            data_dict[img_key][
                ("context.npy" if f_ctxt_size == 1 else f"context_f={f_ctxt_size}.npy")
            ] = micro_ctxt
        logger.debug(
            f"Key: {img_key}, micro ctxt stats: min: {micro_ctxt.min()}, max: {micro_ctxt.max()}, mean: {micro_ctxt.mean()}"
        )

        # Write the data_dict to a new tar file
        out_file = out_parent / url.relative_to(in_parent)
        if not out_file.parent.exists():
            out_file.parent.mkdir()

        # Write to a temporary file first
        with tempfile.NamedTemporaryFile(dir=out_file.parent, delete=False) as tmpfile:
            tmp_path = Path(tmpfile.name)
            with TarWriter(str(tmp_path)) as writer:
                for key, data in data_dict.items():
                    sample = {"__key__": key}
                    for ext, arr in data.items():
                        if ext == "wcs":
                            sample[ext] = arr.encode("utf-8")
                        else:
                            sample[ext] = arr.astype(np.float32)
                    writer.write(sample)
        # Atomically move the temp file to the final destination
        tmp_path.replace(out_file)

    # Process URLs in parallel with progress bar
    logger.info(f"Parallel processing urls.")
    if url_istart > 0:
        logger.info(f"Starting at index {url_istart} of {len(urls)} urls.")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with graceful_shutdown(executor):
            futures = {
                executor.submit(process_url, url): url for url in urls[url_istart:]
            }

            for future in tqdm(
                as_completed(futures),
                desc="Processing URLs",
                total=len(urls),
                initial=url_istart,
                dynamic_ncols=True,
                smoothing=0.1,
            ):
                try:
                    _ = future.result()
                except Exception as e:
                    url = futures[future]
                    logger.error(f"Error processing {url}: {e}")
                    traceback.print_exc()

    logger.divider()
    logger.info(f"Done creating micromaps contexts dataset.\n")


def create_micromaps_encodings(
    dset,
    model_name,
    name_suffix="",
    override=False,
    debug=False,
    batch_size=64,
    num_workers=8,
    model_class=VQVAE,
    ckpt="best",
    device="cuda:1",
):
    logger.divider()
    if debug:
        logger.setLevel("DEBUG")
    logger.info(f"Creating encodings from dataset {dset} with model {model_name}\n")

    # Get dataset parent directory and create encodings output parent from it
    out_parent = (
        dset_parent := parse_dset_path(dset, lookup=paths.MICROMAP_SUBSETS)
    ).with_name(
        dset_parent.name.replace("micromaps", "micromap_encodings")
        + (f"-{name_suffix}" if len(name_suffix) > 0 else "")
    )

    # Check if the output directory already exists
    if out_parent.exists():
        if override:
            logger.info(f"Recursively removing existing directory: \n\t{out_parent}.\n")
            shutil.rmtree(out_parent)
        else:
            logger.info(f"Micromap dataset already exists. Aborting for safety.")
            sys.exit(0)

    # If not, create it
    logger.info(f"Creating output directory: \n\t{out_parent}.\n")
    out_parent.mkdir(parents=True, exist_ok=True)
    (out_parent / "metadata").mkdir(exist_ok=True)

    # Parse model checkpoint file
    model_parent = paths.MODEL_PARENT / model_name
    match ckpt:
        case "best":
            try:
                ckpt_path = sorted((model_parent / "lightning").glob("best-*.ckpt"))[-1]
            except IndexError:
                raise FileNotFoundError(
                    f"No best checkpoint found in {model_parent / 'lightning'}."
                )
        case "last":
            try:
                ckpt_path = sorted(
                    (model_parent / "lightning").glob("train_step-*.ckpt")
                )[-1]
            except IndexError:
                raise FileNotFoundError(
                    f"No last checkpoint found in {model_parent / 'lightning'}."
                )
        case str() if "/" in ckpt:
            ckpt_path = ckpt

        case str():
            ckpt_path = model_parent / "lightning" / ckpt

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")

    # Load model
    logger.info(f"Loading model from checkpoint: \n\t{ckpt_path}.\n")
    model = model_class.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    model.to(device)

    for split in ["train", "val", "test"]:
        logger.divider()
        logger.info(f"Processing split: {split}\n")
        logger.debug("--- debug mode ---")

        # Load dataset, initialize dataloader
        dataset = MicromapPreparationDataset(dset, split=split)
        dataloader = WebLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
        )

        # Initialize split directory
        (out_parent / split).mkdir(exist_ok=True)
        (out_parent / "metadata" / split).mkdir(exist_ok=True)

        # Loop through dataloader
        logger.info("Looping through dataloader...")
        outp_batches = []
        # stupid debugging
        if debug:
            i_debug = 0
        for batch in tqdm(
            dataloader,
            desc=f"Processing {split} split",
            unit="batch",
            ncols=80,
            total=(
                np.ceil(dataset._len / num_workers / batch_size) * num_workers
            ).astype(int),
        ):
            # Get micromaps and metadata from batch
            x, keys, urls = batch

            # Because of the splitting, the micromaps have shape
            # (B, 4, 1, H, W),
            # reshape to (4*B, 1, H, W)
            # Edit: I changed this on July 1st 25 because I stopped using
            # the quadrant stuff
            #
            # x = x.reshape(-1, *x.shape[-3:])

            # Encode micromaps
            with torch.no_grad():
                x = x.to(device)
                encodings = model.encode_to_prequant(x)

            # Move encodings to CPU and reshape to (B, 4, h, w)
            #
            # Edit: I changed this on July 1st 25 because I stopped using
            # the quadrant stuff
            #
            # encodings = encodings.cpu().reshape(-1, 4, *encodings.shape[-3:])
            encodings = encodings.cpu().numpy().astype(np.float32)
            outp_batches.append([list(keys), list(urls), encodings])

            if debug:
                i_debug += 1
                if i_debug >= 2:
                    logger.debug(f"Stopping after {i_debug} batches.")
                    break

        # Put
        logger.info(f"Saving encodings to .tar files...")
        outp_data = {}
        for b in outp_batches:
            keys, urls, encodings = b
            for i, url in enumerate(urls):
                pointing = Path(url).stem
                if pointing not in outp_data:
                    outp_data[pointing] = {}
                outp_data[pointing][keys[i]] = encodings[i]

        # Containers for later writing metadata
        pointings = set()
        names = set()
        for i, pointing in enumerate(outp_data.keys()):
            tar_path = out_parent / split / f"{pointing}.tar"

            with TarWriter(str(tar_path)) as writer:
                for key in outp_data[pointing].keys():
                    encoding = outp_data[pointing][key]
                    sample = {
                        "__key__": key,
                        "npy": encoding,
                    }
                    writer.write(sample)
                    names.add(key)

            pointings.add(tar_path.stem)

        # Save metadata to json files
        logger.info(f"Saving pointings metadata...")
        for pointing in sorted(list(pointings)):
            pointing_keys = [k for k in names if k.split("-")[0] == pointing]
            metadata = {
                "pointing": pointing,
                "names": pointing_keys,
            }
            with open(out_parent / "metadata" / split / f"{pointing}.json", "w") as f:
                json.dump(metadata, f, indent=4)

    # Save metadata to json file
    logger.info(f"Saving encodings metadata...")
    metadata = {
        "dataset": dset,
        "model": model_name,
        "model_class": model_class.__name__,
        "ckpt": str(ckpt_path),
    }
    with open(out_parent / "metadata" / f"encodings.json", "w") as f:
        json.dump(metadata, f, indent=4)

    logger.divider()
    logger.info(f"Done creating micromap encodings dataset.\n")


def create_micromap_webdataset(
    micromap_size,
    spacing=1,
    override=False,
    debug=False,
    prefix="micromaps",
    mosaic_dir=paths.MOSAIC_DIR_DR2,
):
    logger.divider()
    if debug:
        logger.setLevel("DEBUG")
    logger.info(
        f"Creating micromaps with size {micromap_size} and spacing {spacing}.\n"
    )
    out_parent = paths.MICROMAP_DIR / f"{prefix}-{micromap_size}px-{spacing=}"
    logger.info(f"Saving micromaps as .tar archive to: \n\t{out_parent}.\n")

    # Check if the output file already exists
    if out_parent.exists():
        if override:
            logger.info(f"Recursively removing existing directory: \n\t{out_parent}.\n")
            shutil.rmtree(out_parent)
        else:
            logger.info(f"Micromap dataset already exists. Aborting for safety.")
            sys.exit(0)

    # If not, create it
    out_parent.mkdir()
    (out_parent / "metadata").mkdir()
    for split in ["train", "val", "test"]:
        (out_parent / split).mkdir()
        (out_parent / f"metadata/{split}").mkdir()

    # Look for mosaics
    logger.info(f"Looking for mosaics in \n\t{mosaic_dir}...")
    pointings = sorted([p.name for p in mosaic_dir.iterdir() if p.is_dir()])
    logger.info(f"Found {len(pointings)} mosaics.")

    for i, pointing in tqdm(
        enumerate(pointings),
        desc="Pointings",
        unit="pointing",
        total=len(pointings),
        ncols=80,
    ):
        # Get the map array
        logger.debug(f"Loading {pointing}...")
        map_arr, wcs = get_map_image(
            mosaic_dir / pointing / f"mosaic-blanked.fits", get_wcs=True
        )

        # Get micromaps and metadata
        logger.debug(f"Getting micromaps for {pointing}...")
        micromaps, centers, wcss = get_micromaps(
            map_arr, micromap_size, spacing, wcs=wcs, pbar=False
        )

        # Get center coordinates in ra/dec from pixel coordinates
        center_coords = wcs.pixel_to_world(*centers.T)
        center_coords = np.stack([center_coords.ra.deg, center_coords.dec.deg]).T
        # Get the WCS information in string format
        wcss = np.array([wcs.to_header_string() for wcs in wcss])

        # Split into 80% train, 10% val, 10% test
        all_idxs = np.arange(len(micromaps))
        flg = np.random.choice(
            all_idxs, np.round(0.2 * len(micromaps)).astype(int), replace=False
        )
        val_flg, test_flg = np.array_split(flg, 2)
        # Make them binary flags
        val_flg = np.isin(all_idxs, val_flg)
        test_flg = np.isin(all_idxs, test_flg)
        train_flg = np.logical_not(val_flg | test_flg)

        logger.debug(f"Saving micromaps for {pointing}...")
        for split, flag in zip(
            ["train", "val", "test"], [train_flg, val_flg, test_flg]
        ):
            logger.debug(f"Saving {split} split...")

            # Save micromaps
            names = []
            with TarWriter(str(out_parent / f"{split}/{pointing}.tar")) as writer:
                for mm, center, center_crd, wcs in zip(
                    micromaps[flag], centers[flag], center_coords[flag], wcss[flag]
                ):
                    name = f"{pointing}-{center[0]}-{center[1]}"
                    sample = {
                        "__key__": name,
                        "npy": mm.astype(np.float32),
                        "center.npy": center.astype(np.int32),
                        "center_radec.npy": center_crd.astype(np.float32),
                        "wcs": wcs.encode("utf-8"),
                    }
                    writer.write(sample)
                    names.append(name)
            logger.debug(f"Saved {pointing} with {micromaps.shape[0]} micromaps.")

            # Save metadata to json file
            metadata = {
                "pointing": pointing,
                "names": names,
                "centers": centers[flag].tolist(),
                "center_coords": center_coords[flag].tolist(),
                "wcss": wcss[flag].tolist(),
            }
            with open(out_parent / f"metadata/{split}/{pointing}.json", "w") as f:
                json.dump(metadata, f, indent=4)
            logger.debug(f"Saved metadata for {pointing}.")

        # Stupid debugging
        if debug:
            if i >= 2:
                logger.debug(f"Stopping after {i} pointings.")
                break

    logger.info(f"Done creating micromaps dataset.")


if __name__ == "__main__":
    # Use this to create micromaps from the mosaics.
    # -------------------------------------------------------------
    if False:
        micromap_size = 1024
        create_micromap_webdataset(
            micromap_size,
            spacing=1,
            override=True,
            prefix="micromaps-DR3",
            mosaic_dir=paths.MOSAIC_DIR_DR3,
        )

    # Use this to create micromap encodings from the micromaps.
    # -------------------------------------------------------------
    if True:
        create_micromaps_encodings(
            dset="micromaps-DR3-512",
            model_name="VQ-VAE-256",
            override=True,
            debug=False,
            batch_size=16,
            num_workers=16,
            model_class=VQVAE,
            ckpt="best",
            device="cuda:0",
        )

    cpu_count = os.cpu_count()
    # Use this to create micromap contexts from the micromaps.
    # -------------------------------------------------------------
    if True:
        create_micromaps_contexts_parallel(
            dset="micromaps-DR3-512",
            mosaic_dir=paths.MOSAIC_DIR_DR3,
            catalog=paths.LOTSS_DR3_CAT,
            debug=False,
            f_ctxt_size=2,
            url_istart=5494,
            # url_file="faulty_urls.txt",
            max_workers=cpu_count // 4,
        )

    # Use this to add contexts to micromap encodings.
    # -------------------------------------------------------------
    if True:
        add_encodings_contexts_parallel(
            dset="micromaps-DR3-512",
            debug=False,
            f_downscale=2,
            f_ctxt_size=2,
            max_workers=cpu_count // 4,
            # url_file="faulty_urls.txt",
        )

    # Use this to add model micromaps to micromap encodings with contexts.
    # -------------------------------------------------------------
    if False:
        make_model_contexts(
            dset="micromaps-256",
            which="PyBDSF",
            debug=False,
            f_downscale=4,
            max_workers=64,
        )
