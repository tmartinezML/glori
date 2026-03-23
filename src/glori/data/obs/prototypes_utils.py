import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.wcs import WCS
from skimage import measure
from glori.data.obs.micromaps.utils import filter_catalog_by_wcs
from glori.data.trf.segment import get_circle


def make_wcs(img, src):
    # Make wcs object
    center_coord = src.RA, src.DEC
    if hasattr(src, "optRA") and not np.isnan(src.optRA):
        center_coord = src.optRA, src.optDec

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [img.shape[1] // 2, img.shape[0] // 2]
    wcs.pixel_shape = [img.shape[1], img.shape[0]]
    wcs.wcs.cdelt = [-1.5 / 3600, 1.5 / 3600]  # degrees per pixel (1.5 arcsec)
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.crval = center_coord
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    return wcs


def get_pixel_locations(wcs, cat):
    return wcs.all_world2pix(np.array([cat.RA, cat.DEC]).T, 0).astype(int)


def source_profile(mask, src, cmp_cat):

    # Get WCS:
    wcs = make_wcs(mask, src)

    # All components on WCS:
    components = filter_catalog_by_wcs(cmp_cat, wcs)

    # Host means src is parent source
    host_cmp = components[components.Parent_Source == src.Source_Name]
    host_pxcoords = get_pixel_locations(wcs, host_cmp)

    # Contaminating means different parent source
    cont_cmp = components[components.Parent_Source != src.Source_Name]
    cont_pxcoords = get_pixel_locations(wcs, cont_cmp)

    # Get regions and labels of mask
    regions, labels = measure.label(mask, return_num=True, connectivity=2)

    # Initialize dictionaries to store results
    host_is_on_mask = {}
    guests_on_mask = {}

    # Iterate through each region and check for host and guest sources
    for l in range(1, labels + 1):
        region_mask = regions == l

        # Check if host is in the region
        host_is_on_mask[l] = False
        for j in range(len(host_pxcoords)):
            # Boolean value:
            if region_mask[host_pxcoords[j, 1], host_pxcoords[j, 0]]:
                host_is_on_mask[l] = True
                break

        # Check for guests in the region
        guests_on_region = []
        for j in range(len(cont_pxcoords)):
            if region_mask[cont_pxcoords[j, 1], cont_pxcoords[j, 0]]:
                # Source name
                guests_on_region.append(cont_cmp["Component_Name"].values[j])

        guests_on_mask[l] = guests_on_region

    return {"host_is_on_mask": host_is_on_mask, "guests_on_mask": guests_on_mask}


def make_clean_mask(mask, src, profile=None, cmp_cat=None):

    if profile is None:
        assert (
            cmp_cat is not None
        ), "Need component catalog if no source profile is passed."
        profile = source_profile(mask, src, cmp_cat)

    # Remove contaminating islands from mask:
    clean_mask = np.zeros_like(mask)

    # Get island labels
    regions, labels = measure.label(mask, return_num=True, connectivity=2)

    # Loop through labels
    for label in range(1, labels + 1):

        # Add only those islands with the host source and no contaminating components
        if profile["host_is_on_mask"][label] and not len(
            profile["guests_on_mask"][label]
        ):
            clean_mask[regions == label] = 1

    return clean_mask, profile


def center_source(img, mask):
    x, y, r = get_circle(mask)

    # Return new image and mask, translated so that x, y is at center
    img_c = np.zeros_like(img)
    mask_c = np.zeros_like(mask)
    mask_coords = np.flip(np.argwhere(mask), axis=1)
    mask_coords_c = (
        mask_coords
        - np.array([x, y])
        + np.array([img.shape[-1] // 2, img.shape[-2] // 2])
    )
    # Remove coordinates that are out of bounds
    valid = np.all(
        (mask_coords_c >= 0) & (mask_coords_c < np.array([img.shape[1], img.shape[0]])),
        axis=1,
    )
    mask_coords_c = mask_coords_c[valid]
    mask_c[mask_coords_c[:, 1], mask_coords_c[:, 0]] = 1
    img_c[mask_coords_c[:, 1], mask_coords_c[:, 0]] = img[
        mask_coords[:, 1][valid], mask_coords[:, 0][valid]
    ]
    return img_c, mask_c, r


import tempfile
import shutil
import webdataset as wds
from pathlib import Path

import io
import tempfile
import shutil
import tarfile
import numpy as np
from pathlib import Path
from webdataset import TarWriter


def append_to_tar(tar_path: Path, new_samples: list[dict], keep_backup=True):
    """
    Safely append new samples to an existing tar file using a temporary file.

    Parameters
    ----------
    tar_path : Path
        Path to the existing tar file
    new_samples : list[dict]
        List of new samples to add. Each sample should be a dict with '__key__' and data
    keep_backup : bool, optional
        Whether to keep a .bak file of the original tar, by default True
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(
            suffix=".tar", delete=False, dir=tar_path.parent
        ) as tmp:
            temp_path = Path(tmp.name)

            # Write existing content plus new samples to temp file
            with TarWriter(str(temp_path)) as writer:
                # First copy existing content if tar exists
                samples = []
                if tar_path.exists():
                    with tarfile.open(tar_path, "r") as tar:
                        for member in tqdm(
                            tar.getmembers(), desc="Extracting existing samples"
                        ):
                            if member.isfile():
                                # Extract key and extension
                                key = member.name.split(".")[0]
                                ext = ".".join(member.name.split(".")[1:])

                                # Read the data
                                fileobj = tar.extractfile(member)
                                if fileobj is not None:
                                    data = fileobj.read()
                                    match ext.split(".")[-1]:
                                        case "npy":
                                            sample = {
                                                "__key__": key,
                                                ext: np.load(io.BytesIO(data)),
                                            }
                                        case _:
                                            sample = {
                                                "__key__": key,
                                                ext: data.decode(),
                                            }
                                    samples.append(sample)

                # Then add new samples
                for sample in tqdm(
                    samples + new_samples, desc="Writing samples to tar"
                ):
                    writer.write(sample)

            # Backup original if requested
            if keep_backup and tar_path.exists():
                backup_path = tar_path.with_suffix(".tar.bak")
                shutil.copy2(tar_path, backup_path)

            # Move temp file to target location
            shutil.move(temp_path, tar_path)

    except Exception as e:
        # Clean up temp file in case of error
        if "temp_path" in locals():
            temp_path.unlink(missing_ok=True)
        raise e


def append_to_parquet(parquet_path: Path, new_cat: pd.DataFrame):
    """
    Safely append new entries to an existing parquet file.

    Parameters
    ----------
    parquet_path : Path
        Path to the existing parquet file
    new_cat : pd.DataFrame
        New catalog entries to append
    """
    try:
        if parquet_path.exists():
            # Load existing catalog
            existing_cat = pd.read_parquet(parquet_path)
            # Concatenate and drop duplicates
            combined_cat = (
                pd.concat([existing_cat, new_cat])
                .drop_duplicates()
                .reset_index(drop=True)
            )
        else:
            combined_cat = new_cat

        # Write back to parquet
        combined_cat.to_parquet(parquet_path, index=False)

    except Exception as e:
        raise e


def load_by_extension(ext: str, tar_path: Path):
    """
    Load all images selected by their extension.
    """
    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(ext)]
        if not members:
            raise KeyError(f"Extension {ext} not found in tar {tar_path}")
        sample = []
        for member in tqdm(members, desc=f"Loading", leave=False):
            fileobj = tar.extractfile(member)
            if fileobj is not None:
                data = fileobj.read()
                ext = ".".join(member.name.split(".")[1:])
                match ext.split(".")[-1]:
                    case "npy":
                        sample.append(np.load(io.BytesIO(data)))
                    case _:
                        sample.append(data.decode())
        return sample
