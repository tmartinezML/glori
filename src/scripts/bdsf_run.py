import utils.paths as paths
from data.utils import load_mosaic
from utils.my_logging import get_logger

from analysis.bdsf_analysis import *

logger = get_logger(__name__)
logger.divider()

logger.info("Starting BDSF analysis")
mosaic_list = sorted([p.name for p in paths.MOSAIC_DIR_DR3.glob("*") if p.is_dir()])
mosaic = mosaic_list[-1]
logger.info(f"Using mosaic: {mosaic}")
logger.divider()

img, wcs = load_mosaic(mosaic)
logger.info(f"Starting BDSF.")
res = tiered_bdsf_wrapper(img, wcs=wcs)
logger.divider()
logger.info("BDSF analysis completed. Saving...")
save_multistep_output(res, f"{mosaic}-full")
logger.divider()
logger.info("BDSF analysis complete.")

