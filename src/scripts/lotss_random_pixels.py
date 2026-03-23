import utils.paths as paths
from maps.map_utils import get_map_image
import numpy as np


def get_pixels_sample(pointing, n_px=1000):
    p_file = paths.MOSAIC_DIR_DR3 / pointing / "mosaic-blanked.fits"
    map_arr = get_map_image(p_file, get_wcs=False).ravel()
    map_arr = map_arr[~np.isnan(map_arr)]
    random_pxs = np.random.choice(map_arr, n_px)
    max_val = np.max(map_arr)
    min_val = np.min(map_arr)
    return random_pxs, max_val, min_val


pointings = [p.name for p in paths.MOSAIC_DIR_DR3.iterdir() if p.is_dir()]

# Parallelize code with progress bar using process pool executor
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

px_per_map = 10_000
pxs = np.empty((len(pointings), px_per_map))
max_vals = np.empty(len(pointings))
min_vals = np.empty(len(pointings))

with ProcessPoolExecutor(max_workers=96) as executor:

    futures = [
        executor.submit(get_pixels_sample, pointing, px_per_map)
        for pointing in pointings
    ]

    with tqdm(total=len(pointings), desc="Processing...") as pbar:
        for i, future in enumerate(as_completed(futures)):
            pxs[i], max_vals[i], min_vals[i] = future.result()
            pbar.update(1)


# Save to file
import h5py

with h5py.File(paths.LOFAR_DATA_PARENT / "random_pixels.h5", "w") as f:
    f.create_dataset("pixels", data=pxs)
    f.create_dataset("max_vals", data=max_vals)
    f.create_dataset("min_vals", data=min_vals)