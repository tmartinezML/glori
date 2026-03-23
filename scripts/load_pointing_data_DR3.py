import requests
import tarfile
import time
import datetime
from urllib3.exceptions import InsecureRequestWarning
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
from pathlib import Path
import subprocess
import sys


import numpy as np
from tqdm import tqdm
from astropy.io import fits

import utils.paths as paths
from data.utils import load_lotss_catalog, load_fits_image, load_fits_header
from utils.my_logging import get_logger, add_file_handler

logger = get_logger(__name__)

# Disable SSL warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Thread lock for progress bar updates
pbar_lock = threading.Lock()

out_parent = Path("/hs/babbage/data/group-brueggen/tmartinez/pointings-dr3")
get_tnow = lambda: datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
# log_folder = out_parent / f"download_logs/{get_tnow()}"
# log_folder.mkdir(parents=True)
log_folder = out_parent / f"download_logs"
with open(log_folder / "file_stats.csv", "w") as f:
    f.write("tstart;present;offline;loading\n")
add_file_handler(logger, log_folder / "logger.log")


make_link = (
    lambda pointing: f"https://repository.surfsara.nl/datasets/lotss-dr2/{pointing.replace('+', '-')}/files/images.tar"
)
make_target = lambda pointing: out_parent / pointing / "images.tar"

wrap_command_ssh = lambda cmd: (
    f"ssh fs08 '/bin/bash -l -c \"\
    export PATH=$PATH:/opt/singularity/bin &&\
     singularity exec\
     --bind /hs/babbage/data/group-brueggen/tmartinez/pointings-dr3:/pointings-dr3\
     --home /hs/babbage/data/group-brueggen/tmartinez/pointings-dr3/utils:/pointings-dr3/utils\
     --pwd /pointings-dr3/utils\
     /hs/babbage/data/group-brueggen/tmartinez/pointings-dr3/utils/dl_img.sif\
     {cmd}\"'"
)

wrap_command = lambda cmd: (
    f"export PATH=$PATH:/opt/singularity/bin &&\
     singularity exec\
     --bind /hs/babbage/data/group-brueggen/tmartinez/pointings-dr3:/pointings-dr3\
     --home /hs/babbage/data/group-brueggen/tmartinez/pointings-dr3/utils:/pointings-dr3/utils\
     --pwd /pointings-dr3/utils\
     /hs/babbage/data/group-brueggen/tmartinez/pointings-dr3/utils/dl_img.sif\
     {cmd}"
)


def get_longlist_output(pointing: str, silent=False) -> str:
    """Check if a URL exists without downloading the file."""
    cmd = wrap_command(
        f"ada --tokenfile maca_sksp_tape_DDF_readonly.conf --longlist /archive/{pointing}/images.tar"
        # f"ada --tokenfile maca_sksp_tape_DDF_readonly.conf --longlist /archive/{pointing}"
    )
    with subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as proc:
        for line in proc.stdout:
            if not silent:
                print(line, end="")
    return line


def check_file_online(pointing: str) -> bool:
    """Check if a URL exists without downloading the file."""
    out = get_longlist_output(pointing, silent=True)
    return any([out.endswith(s) for s in ["NEARLINE", "ONLINE", "ONLINE_AND_NEARLINE"]])


def parallel_check_files_online(pointings):
    """Check multiple URLs in parallel."""
    with ThreadPoolExecutor(max_workers=64) as executor:
        checks = list(
            tqdm(
                executor.map(check_file_online, pointings),
                total=len(pointings),
                desc="Checking URLs",
            )
        )
    return checks


def request_file(pointing: str) -> requests.Response:
    """Request the file from the URL."""
    url = f"https://repository.surfsara.nl/api/objects/lotss-dr2/{pointing.replace('+', '-')}/stage/0"

    # You can try using just the key with None or an empty string
    data = {"share-token": ""}

    # You probably don't need cookies unless it's a private dataset
    # But if it's public, no auth is needed
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://repository.surfsara.nl",
        "Referer": make_link(pointing),
        "X-Requested-With": "XMLHttpRequest",
        "User-Agent": "Mozilla/5.0",  # Optional, just for good measure
    }

    try:
        response = requests.post(url, headers=headers, data=data, verify=False)
        response.raise_for_status()
        return response
    except (requests.RequestException, requests.ConnectionError) as e:
        logger.error(f"Error requesting file for {pointing}: {e}")
        return str(e)


def parallel_request_files(pointings):
    """Request multiple files in parallel."""
    with ThreadPoolExecutor(max_workers=64) as executor:
        responses = list(
            tqdm(
                executor.map(request_file, pointings),
                total=len(pointings),
                desc="Requesting files",
            )
        )
    return responses


def check_file_exists(pointing: str, suffix="model.fits") -> bool:
    """Check if the target file exists."""
    return (t := make_target(pointing).with_name("images")).exists() and any(
        t.glob(f"*.{suffix}")
    )


def download_and_extract_pointing(pointing, main_pbar):
    """Download and extract a single pointing."""
    link = make_link(pointing)
    target = make_target(pointing)
    dl_log_file = log_folder / "download_log.txt"

    try:
        # Check if already loaded
        if (t := target.with_name("images")).exists() and any(t.glob("*.fits")):
            with pbar_lock:
                main_pbar.update(1)
                main_pbar.set_postfix_str(f"Skipped {pointing}")
                main_pbar.refresh()
            outmsg = f"Skipped {pointing} (already exists)"
            with open(dl_log_file, "a") as logf:
                logf.write(outmsg + "\n")
            return 0, outmsg

        # Download if not present
        target.parent.mkdir(parents=True, exist_ok=True)

        # Use requests with SSL verification disabled
        response = requests.get(link, verify=False, stream=True)
        response.raise_for_status()

        # Get total file size for progress bar
        total_size = int(response.headers.get("content-length", 0))

        # Create progress bar for this specific download
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {pointing}",
            leave=False,  # This makes the bar disappear when done
            dynamic_ncols=True,
        ) as pbar:
            with open(target, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Extract with progress bar
        names_to_extract = [
            "image_full_ampphase_di_m.NS.int.restored.fits",
            "image_full_ampphase_di_m.NS.int.model.fits",
        ]
        with tarfile.open(target, "r") as tar:
            members = [m for m in tar.getmembers() if m.name in names_to_extract]
            with tqdm(
                total=len(members),
                unit="files",
                desc=f"Extracting {pointing}",
                leave=False,
                dynamic_ncols=True,
            ) as extract_pbar:
                for member in members:
                    tar.extract(member, path=target.with_name("images"))
                    extract_pbar.update(1)

        target.unlink()

        with pbar_lock:
            main_pbar.update(1)
            main_pbar.set_postfix_str(f"Completed {pointing}")
            main_pbar.refresh()

        outmsg = f"Completed {pointing}"
        with open(dl_log_file, "a") as logf:
            logf.write(outmsg + "\n")
        return 0, outmsg

    except Exception as e:
        with pbar_lock:
            main_pbar.update(1)
            main_pbar.set_postfix_str(f"Failed {pointing}")
            main_pbar.refresh()
        outmsg = f"Failed {pointing}: {str(e)}"
        with open(dl_log_file, "a") as logf:
            logf.write(outmsg + "\n")
        return 1, outmsg


def download_residual(pointing, main_pbar):
    """Download and extract a single pointing."""
    link = str(Path(make_link(pointing)).with_name("mosaic.resid.fits")).replace(
        ":/", "://"
    )
    target = make_target(pointing).with_name("mosaic.resid.fits")
    dl_log_file = log_folder / "resid_download_log.txt"

    try:
        # Check if already loaded
        if (t := target.with_name("images")).exists() and any(t.glob("*.resid.fits")):
            with pbar_lock:
                main_pbar.update(1)
                main_pbar.set_postfix_str(f"Skipped {pointing}")
                main_pbar.refresh()
            outmsg = f"Skipped {pointing} (already exists)"
            with open(dl_log_file, "a") as logf:
                logf.write(outmsg + "\n")
            return 0, outmsg

        # Download if not present
        target.parent.mkdir(parents=True, exist_ok=True)

        # Use requests with SSL verification disabled
        response = requests.get(link, verify=False, stream=True)
        response.raise_for_status()

        # Get total file size for progress bar
        total_size = int(response.headers.get("content-length", 0))

        # Create progress bar for this specific download
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {pointing}",
            leave=False,  # This makes the bar disappear when done
            dynamic_ncols=True,
        ) as pbar:
            with open(target, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        with pbar_lock:
            main_pbar.update(1)
            main_pbar.set_postfix_str(f"Completed {pointing}")
            main_pbar.refresh()

        outmsg = f"Completed {pointing}"
        with open(dl_log_file, "a") as logf:
            logf.write(outmsg + "\n")
        return 0, outmsg

    except Exception as e:
        with pbar_lock:
            main_pbar.update(1)
            main_pbar.set_postfix_str(f"Failed {pointing}")
            main_pbar.refresh()
        outmsg = f"Failed {pointing}: {str(e)}"
        with open(dl_log_file, "a") as logf:
            logf.write(outmsg + "\n")
        return 1, outmsg


def main_download_function():
    """Main function to download and extract pointings."""
    tnow = get_tnow()
    with open(log_folder / "download_log.txt", "a") as f:
        f.write("\n" + "=" * 40 + "\n")
        f.write(f"Download started at {tnow}\n")

    # Load pointing names
    logger.info("Loading pointing names from LoTSS catalog...")
    all_pointings = np.sort(
        np.unique(
            load_lotss_catalog(path=paths.LOTSS_DR2_CAT, select_cols=["Mosaic_ID"])[
                "Mosaic_ID"
            ].values
        )
    )
    pointings = all_pointings.copy()

    # Check which pointings are already loaded
    logger.info("Checking already loaded files...")
    pts_loaded = [check_file_exists(p) for p in tqdm(pointings)]
    with open(log_folder / "file_check_log.txt", "w") as f:
        f.write(
            f"Already loaded files at {tnow}: {sum(pts_loaded)} out of {len(pts_loaded)}\n"
        )
        for p, x in zip(pointings, pts_loaded):
            if x:
                f.write(f"{p}\n")
    pointings = pointings[~np.array(pts_loaded)]
    logger.info(
        f"Found {sum(pts_loaded)} out of {len(pts_loaded)} pointings already loaded. {len(pointings)} pointings to download and extract."
    )
    if len(pointings) == 0:
        logger.info("Everything loaded - we're done! Exiting.")
        return True

    # Check which pointings are offline
    logger.info("Checking offline files...")
    # url_offline = [not check_url_exists(p) for p in tqdm(pointings)]
    url_offline = [not x for x in parallel_check_files_online(pointings)]
    with open(log_folder / "url_check_log.txt", "w") as f:
        f.write(
            f"Offline files at {tnow}: {sum(url_offline)} out of {len(url_offline)}\n"
        )
        for p, offl in zip(pointings, url_offline):
            if offl:
                f.write(f"{p}\n")
    logger.info(
        f"Found {(s := sum(url_offline))} out of {(l := len(url_offline))} files offline. {s - l} online pointings to process."
    )
    with open(log_folder / "file_stats.csv", "a") as f:
        f.write(f"{tnow};{sum(pts_loaded)};{sum(url_offline)};{len(pointings)}\n")
    # Request the offline files for staging
    logger.info(f"Staging {sum(url_offline)} offline files...")
    responses = parallel_request_files(pointings[np.array(url_offline)])
    with open(log_folder / "request_log.txt", "w") as f:
        for p, r in zip(pointings[np.array(url_offline)], responses):
            f.write(f"{p}: {r if isinstance(r, str) else r.json()}\n")
    pointings = pointings[~np.array(url_offline)]
    if len(pointings) == 0:
        logger.info("No online pointings to download. Exiting.")
        return False

    logger.info(f"Starting download and extraction of {len(pointings)} pointings...")
    # Create main progress bar
    with tqdm(
        total=len(pointings), desc="Overall progress", position=0, dynamic_ncols=True
    ) as main_pbar:
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            futures = [
                executor.submit(download_and_extract_pointing, pointing, main_pbar)
                for pointing in pointings
            ]

            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    # Optional: print completion status
                    # print(result)
                except Exception as exc:
                    print(f"Task generated an exception: {exc}")
    return False


def download_loop(wait_time_minutes=10):
    # Keep trying until all files are downloaded.
    # Wait 1 hour between tries.
    while True:
        t0 = datetime.datetime.now()
        logger.divider(n_lines=2)
        logger.info(f"Starting download attempt at {t0.strftime('%Y-%m-%d %H:%M:%S')}")
        all_loaded = main_download_function()
        if all_loaded:
            logger.info("All files downloaded successfully.")
            break
        else:
            elapsed = datetime.datetime.now() - t0
            wait_time = max(0, wait_time_minutes * 60 - elapsed.total_seconds())
            logger.info(f"Waiting {wait_time/60:.1f} minutes before next attempt...")
            time.sleep(wait_time)


def load_bdsf_residuals():
    """Main function to download and extract pointings."""
    tnow = get_tnow()
    with open(log_folder / "resid_download_log.txt", "a") as f:
        f.write("\n" + "=" * 40 + "\n")
        f.write(f"Loading residuals\n")
        f.write(f"Download started at {tnow}\n")

    # Load pointing names
    logger.info("Loading pointing names from LoTSS catalog...")
    all_pointings = np.sort(
        np.unique(
            load_lotss_catalog(path=paths.LOTSS_DR2_CAT, select_cols=["Mosaic_ID"])[
                "Mosaic_ID"
            ].values
        )
    )
    pointings = all_pointings.copy()

    # Check which pointings are already loaded
    logger.info("Checking already loaded files...")
    pts_loaded = [check_file_exists(p, suffix="resid.fits") for p in tqdm(pointings)]
    with open(log_folder / "resid_file_check_log.txt", "w") as f:
        f.write(
            f"Already loaded files at {tnow}: {sum(pts_loaded)} out of {len(pts_loaded)}\n"
        )
        for p, x in zip(pointings, pts_loaded):
            if x:
                f.write(f"{p}\n")
    pointings = pointings[~np.array(pts_loaded)]
    logger.info(
        f"Found {sum(pts_loaded)} out of {len(pts_loaded)} pointings already loaded. {len(pointings)} pointings to download and extract."
    )
    if len(pointings) == 0:
        logger.info("Everything loaded - we're done! Exiting.")
        return True

    # Check which pointings are offline
    if False:
        logger.info("Checking offline files...")
        # url_offline = [not check_url_exists(p) for p in tqdm(pointings)]
        url_offline = [not x for x in parallel_check_files_online(pointings)]
        with open(log_folder / "resid_url_check_log.txt", "w") as f:
            f.write(
                f"Offline files at {tnow}: {sum(url_offline)} out of {len(url_offline)}\n"
            )
            for p, offl in zip(pointings, url_offline):
                if offl:
                    f.write(f"{p}\n")
        logger.info(
            f"Found {(s := sum(url_offline))} out of {(l := len(url_offline))} files offline. {s - l} online pointings to process."
        )
        # Request the offline files for staging
        logger.info(f"Staging {sum(url_offline)} offline files...")
        responses = parallel_request_files(pointings[np.array(url_offline)])
        with open(log_folder / "resid_request_log.txt", "w") as f:
            for p, r in zip(pointings[np.array(url_offline)], responses):
                f.write(f"{p}: {r if isinstance(r, str) else r.json()}\n")
        pointings = pointings[~np.array(url_offline)]
        if len(pointings) == 0:
            logger.info("No online pointings to download. Exiting.")
            return False

    logger.info(f"Starting of {len(pointings)} residuals...")
    # Create main progress bar
    with tqdm(
        total=len(pointings), desc="Overall progress", position=0, dynamic_ncols=True
    ) as main_pbar:
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            futures = [
                executor.submit(download_residual, pointing, main_pbar)
                for pointing in pointings
            ]

            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    # Optional: print completion status
                    # print(result)
                except Exception as exc:
                    print(f"Task generated an exception: {exc}")
    return False


def make_bdsf_model_images():
    # Load pointing names from mosaic directory and from pointing directory,
    # take the intersection
    pointings_mosaic = sorted(
        [p.name for p in (paths.MOSAIC_DIR_DR2).iterdir() if p.is_dir()]
    )
    pointings_pointing = sorted([p.name for p in (out_parent).iterdir() if p.is_dir()])
    pointings = list(set(pointings_mosaic).intersection(set(pointings_pointing)))
    logger.info(f"Found {len(pointings)} pointings with both mosaic and pointing data.")
    logger.info(
        f"Excluded {len(pointings_mosaic) - len(pointings)} pointings without pointing data."
    )
    logger.info(
        f"Excluded {len(pointings_pointing) - len(pointings)} pointings without mosaic data."
    )

    def process_pointing(pointing):
        # Load files:
        # Mosaic image and header
        mosaic_file = paths.MOSAIC_DIR_DR2 / pointing / "mosaic-blanked.fits"
        with fits.open(mosaic_file) as hdul:
            mosaic_img = hdul[0].data
            header = hdul[0].header
        # Residual image
        resid_file = out_parent / pointing / "mosaic.resid.fits"
        with fits.open(resid_file) as hdul:
            resid_img = hdul[0].data

        # Compute model
        model_img = mosaic_img - resid_img

        # Save model
        model_path = out_parent / pointing / "mosaic.bdsf_model.fits"

        fits.writeto(model_path, model_img, header=header, overwrite=True)

        del mosaic_img, resid_img, model_img, header

    logger.info("Starting BDSF model image creation...")
    # Create main progress bar
    with tqdm(
        total=len(pointings), desc="Overall progress", position=0, dynamic_ncols=True
    ) as main_pbar:
        # Use ThreadPoolExecutor for parallel model image creation
        with ThreadPoolExecutor(max_workers=32) as executor:
            # Submit all tasks
            futures = [
                executor.submit(process_pointing, pointing) for pointing in pointings
            ]

            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    # Optional: print completion status
                    # print(result)
                except Exception as exc:
                    print(f"Task generated an exception: {exc}")

                main_pbar.update(1)


if __name__ == "__main__":

    pointings = sorted(
        [
            d.name
            for d in paths.MOSAIC_DIR_DR3.iterdir()
            if d.is_dir() and d.name.startswith("P")
        ]
    )
    longlist_outfile = log_folder / "longlist_outputs.txt"
    if longlist_outfile.exists():
        longlist_outfile.unlink()

    # Get longlist outputs in parallel
    def process_pointing(pointing):
        # print(f"Processing {pointing}...")
        output = get_longlist_output(pointing, silent=True)
        with open(longlist_outfile, "a") as f:
            f.write(f"{pointing}: {output}")

    with tqdm(
        total=len(pointings), desc="Processing pointings", dynamic_ncols=True
    ) as pbar:
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = [executor.submit(process_pointing, p) for p in pointings]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    pbar.update(1)
                except Exception as exc:
                    print(f"Task generated an exception: {exc}")
                    pbar.update(1)

    logger.info("Sorting lists...")
    avail, missing = [], []
    with open(longlist_outfile, "r") as f:
        for line in f:
            if any([s in line for s in ["NEARLINE", "ONLINE", "ONLINE_AND_NEARLINE"]]):
                avail.append(line)
            else:
                missing.append(line)

    # write sorted into new files
    with open(longlist_outfile.with_name("longlist_available.txt"), "w") as f:
        for line in sorted(avail):
            f.write(line)
    with open(longlist_outfile.with_name("longlist_missing.txt"), "w") as f:
        for line in sorted(missing):
            f.write(line)
