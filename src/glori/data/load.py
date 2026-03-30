from pprint import pformat
from pathlib import Path
from copy import deepcopy
from functools import cache

import pandas as pd
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS

import glori.settings.paths as paths


def parse_weights_file(weights_file, parent=None):
    match weights_file:
        case str():
            if "/" in weights_file:
                weights_path = Path(weights_file)
            elif parent is not None:
                weights_path = parent / "metadata" / weights_file
        case Path():
            weights_path = weights_file
        case _:
            raise TypeError(
                f"weights_file must be str or Path, got {type(weights_file)} with parent {parent}"
            )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file not found in metadata:\n\t{weights_path}"
        )


def load_lotss_catalog(
    path=paths.LOTSS_DR3_CAT,
    select_cols=["RA", "DEC", "Total_flux", "Peak_flux", "Maj"],
    add_cols=[],
):
    select_cols = select_cols + add_cols if select_cols is not None else None
    match path.suffix:
        case ".parquet":
            cat = pd.read_parquet(path, columns=select_cols)
        case ".fits" | ".fit" | ".fts":
            cat = load_fits_catalog(path, select_cols=select_cols)
        case ".csv":
            cat = pd.read_csv(path, usecols=select_cols)
        case _:
            raise ValueError(f"Unsupported catalog format: {path.suffix}")

    if select_cols is None or "DEC" in select_cols:
        cat.sort_values(by="DEC", inplace=True)

    return cat


def list_mosaics(parent=paths.MOSAIC_DIR_DR3):
    """List all mosaic directories in the given parent directory."""
    return [d.name for d in parent.iterdir() if d.is_dir()]


def load_mosaic_header(mosaic_name, parent=paths.MOSAIC_DIR_DR3):
    """Load the header of a mosaic FITS file."""

    f = parent / mosaic_name / "mosaic-blanked.fits"
    return load_fits_header(f)


def load_mosaic(mosaic_name, parent=paths.MOSAIC_DIR_DR3, get_wcs=True):
    f = parent / mosaic_name / "mosaic-blanked.fits"
    img, wcs = load_fits_image(f, get_wcs=get_wcs)
    return img, wcs


def load_fits_header(file):
    with fits.open(file) as hdul:
        header = hdul[0].header
    return deepcopy(header)


def load_fits_image(file, get_wcs=True):

    with fits.open(file) as hdul:
        image = hdul[0].data
        wcs = WCS(hdul[0].header, naxis=2) if get_wcs else None

    return image.copy(), deepcopy(wcs)


def load_fits_catalog(catalog_path, select_cols=None):

    cat = Table.read(catalog_path, memmap=(select_cols is not None))

    if select_cols is not None:
        cat = cat[select_cols]

    cat = cat.to_pandas()

    # Format all columns that are obj to utf-8
    for col in cat.columns:
        if cat[col].dtype == "object":
            cat[col] = cat[col].str.decode("utf-8")

    return cat


def print_available_datasets():
    print(f"Available datasets in {paths.LOFAR_DATA_PARENT}:")
    for k, v in paths.LOFAR_SUBSETS.items():
        print(f"  {k} ({v.name})")


def parse_micromaps_path(dset):
    return parse_dset_path(dset, lookup=paths.MICROMAP_SUBSETS_ARROW)


def parse_dset_path(dset, lookup=paths.LOFAR_SUBSETS):
    match dset:
        case Path():
            return dset

        case str():
            if dset in lookup:
                return lookup[dset]

            elif Path(dset).exists():
                return Path(dset)

            else:
                parent = list(lookup.values())[0].parent
                candidates = list(filter(lambda p: dset in p.name, parent.iterdir()))
                if len(candidates) == 1:
                    return candidates[0]
                elif len(candidates) > 1:
                    matches = list(filter(lambda p: dset == p.name, candidates))
                    if len(matches) == 1:
                        return matches[0]
                    raise ValueError(
                        f"Multiple datasets match '{dset}': {[p.name for p in candidates]}"
                    )
                else:
                    raise FileNotFoundError(
                        f"File {dset} not found.\n\nAvailable datasets:\n{pformat(list(lookup.keys()))}"
                        f"\n\nAvailable files:\n{pformat([p.name for p in parent.iterdir()])}"
                    )

        case _:
            raise ValueError(f"Invalid argument type: {dset}")
