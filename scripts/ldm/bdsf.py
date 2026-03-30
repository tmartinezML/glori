import sys
import json
from datetime import datetime
from functools import partial

import art
import torch
import randomname
import numpy as np

import glori.settings.paths as paths
import glori.data.trf.post as post
import glori.data.trf.transforms as T
import glori.analysis.ldm.io as ldmio
import glori.analysis.ldm.bdsf as ldmbdsf
from glori.analysis.bdsf_analysis import *
from glori.plotting.images import plot_image_grid
from glori.infra.logging import get_logger, add_file_handler
from glori.data.trf.functional import zero_center
from glori.data.sets.micromaps import MicromapDatasetHF
from glori.models.load import parse_lightning_ckpt
from glori.inference.ldm_sampler import LDMSampler
# Get logger
logger = get_logger(__name__)

# Read first argument as run name
if len(sys.argv) < 2:
    logger.error("Usage: python bdsf.py <run_name>")
    sys.exit(1)
run_name = sys.argv[1]

# Limit Python to use only the first N CPUs
Ncpu = 64
os.sched_setaffinity(0, set(range(Ncpu)))

logger.info(f"Starting BDSF analysis for run: {run_name}")

# Load result
result = ldmio.load_sampling_result(run_name, missing_is_error={"bdsf": False, "original_bdsf": False, "summary": False})
big_img_batch = result["img_batch"]

# Run bdsf
bdsf_workers = 8
results, n_successful_images, n_failed_images = ldmbdsf.run_bdsf_parallel(
    big_img_batch, bdsf_folder, logger=logger, max_workers=bdsf_workers
)