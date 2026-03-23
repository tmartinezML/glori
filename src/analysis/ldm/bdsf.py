from tqdm import tqdm
import concurrent.futures


from analysis.bdsf_analysis import save_multistep_output, tiered_bdsf_wrapper


import contextlib
import os

from utils.my_logging import get_logger


def process_img(img_idx, img_batch, out_folder):
    """Process a single batch with BDSF analysis"""
    try:
        # Extract single image from batch
        img = img_batch[img_idx].squeeze()

        # Run BDSF analysis with output suppressed
        with (
            contextlib.redirect_stdout(open(os.devnull, "w")),
            contextlib.redirect_stderr(open(os.devnull, "w")),
        ):
            result = tiered_bdsf_wrapper(img, quiet=True)

        # Save result
        save_multistep_output(result, f"img-{img_idx:04d}", out_parent=out_folder)

        return img_idx, True, None
    except Exception as e:
        return img_idx, False, str(e)


def run_bdsf_parallel(img_batch, out_folder, logger=None, max_workers=16, **kwargs):
    if logger is None:
        logger = get_logger("bdsf-parallel")
    # Run tiered bdsf wrapper in parallel
    logger.divider()
    logger.info("Running BDSF analysis in parallel...")

    # Set number of workers - ThreadPoolExecutor can handle more workers since it's lighter
    max_workers = min(
        os.cpu_count(), max_workers
    )  # Cap at 8 to avoid overwhelming the system
    logger.info(f"Using {max_workers} workers for BDSF analysis")

    # Run parallel processing with ThreadPoolExecutor
    results = {}
    img_indices = list(range(len(img_batch)))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_batch = {
            executor.submit(process_img, img_idx, img_batch, out_folder): img_idx
            for img_idx in img_indices
        }

        # Process results with progress bar
        with tqdm(total=len(img_indices), desc="BDSF Analysis") as pbar:
            for future in concurrent.futures.as_completed(future_to_batch):
                result = future.result()
                img_idx, success, error = result
                results[img_idx] = result
                pbar.update(1)

                # Log any errors
                if not success:
                    logger.warning(f"Image {img_idx} failed: {error}")

    # Get results into correct order
    results = [results[i] for i in range(len(img_indices))]

    # Collect successful results
    n_successful_images = len([r[0] for r in results if r[1]])
    n_failed_images = len([r[0] for r in results if not r[1]])

    logger.info(
        f"BDSF analysis complete: {n_successful_images} successful, {n_failed_images} failed"
    )
    return results, n_successful_images, n_failed_images
