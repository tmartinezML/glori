import copy
import random
import logging
from pathlib import Path
from collections.abc import Iterable

import h5py
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import PowerTransformer
from torchvision.transforms.v2 import CenterCrop

from deprecated.micromap_webdataset import MicromapDataset
import utils.paths as paths
from data.sets.firstgalaxydata import FIRSTGalaxyData
from deprecated.image_path_dataset import ImagePathDataset, logger
from data.sets.micromaps import MicromapDatasetHF
import data.trf.transforms as T


class MicromapsDatasetVAE(MicromapDataset):
    def __init__(
        self,
        dset,
        split="train",
        mode="VAE-train",
        custom_transform=None,
        scaler="LOFAR_scaler_II",
        output_tuple=("npy",),
        **kwargs,
    ):
        super().__init__(
            dset,
            split=split,
            mode=mode,
            custom_transform=custom_transform,
            scaler=scaler,
            output_tuple=output_tuple,
            **kwargs,
        )


class MicromapsDatasetTest(MicromapDataset):
    def __init__(
        self,
        dset,
        split="train",
        mode=None,
        custom_transform=T.MicromapTransformTest,
        scaler="LOFAR_scaler",
        output_tuple=("npy",),
    ):
        super().__init__(
            dset,
            split=split,
            mode=mode,
            custom_transform=custom_transform,
            scaler=scaler,
            output_tuple=output_tuple,
        )


class LOFARDataset(ImagePathDataset):
    def __init__(self, path, img_size=80, **kwargs):
        super().__init__(
            path,
            mode_transforms={
                "train": T.TrainTransform(img_size),
                "eval": T.EvalTransform(img_size),
            },
            **kwargs,
        )


class SamplesDataset(LOFARDataset):
    def __init__(
        self, path_or_name, img_size=80, train_mode=False, key="samples", **kwargs
    ):

        match path_or_name:

            # Samples file
            case Path():
                path = path_or_name

            # Model name
            case str():
                path = (
                    paths.ANALYSIS_PARENT / f"{path_or_name}/{path_or_name}_samples.h5"
                )
                if not path.exists():
                    raise FileNotFoundError(f"File {path} not found.")

            # Invalid argument type
            case _:
                raise ValueError(f"Invalid argument type: {path_or_name}")

        super().__init__(
            path,
            img_size=img_size,
            key=key,
            train_mode=train_mode,
            **kwargs,
        )
        self.sampling_steps = self.data.numpy().copy()
        self.data = self.data[:, -1, :, :]


class LOFARPrototypesDataset(ImagePathDataset):
    def __init__(self, path, img_size=100, attributes=["masks"], **kwargs):
        super().__init__(
            path,
            mode_transforms={
                "train": T.TrainTransformPrototypes(img_size),
                "eval": T.EvalTransform(img_size),
            },
            attributes=attributes,
            **kwargs,
        )

        # Load mask metadata
        logger.info("Loading mask metadata...")
        self.mask_metadata = pd.read_hdf(self.path, key="mask_metadata")

        # Filter out sources with radius > 0.5 * img_size
        logger.info(
            f"Image size {img_size}: Removing sources with model_radius > {0.5 * img_size}..."
        )
        filter_flag = (self.mask_metadata["Model_Radius"] <= 0.5 * img_size).values
        self.index_slice(filter_flag)
        logger.info(
            f"Removed {(s := (~filter_flag).sum()):_} of {(l := len(filter_flag)):_} sources ({s/l*100:.1f}%)."
        )

        # Center crop images
        logger.info("Reshaping images...")
        self.data = CenterCrop(img_size)(self.data)

        # Bring masks to the same shape as images
        if hasattr(self, "masks"):
            logger.info("Reshaping masks...")
            self.masks = CenterCrop(img_size)(self.masks)

        # Box-Cox transform mask sizes
        self.mask_sizes = torch.Tensor(self.mask_metadata["feret_diameter_max"].values)
        self.box_cox_transform("mask_sizes")


class TrainDatasetFIRST(FIRSTGalaxyData):
    def __init__(self, img_size=80, **kwargs):
        super().__init__(
            selected_split=["train", "test", "valid"],
            is_balanced=True,
            transform=T.TrainTransform(img_size),
            **kwargs,
        )

    def set_context(self, *args):
        logger.warning("FIRSTGalaxyData has class labels as fixed context.")
        return


class CutoutsDataset(LOFARDataset):
    def __init__(self, path, img_size=80, **kwargs):
        super().__init__(path, img_size=img_size, key="cutouts", **kwargs)

        # Add catalog
        logger.info("Adding catalog...")
        cat = pd.read_hdf(path, key="catalog")
        self.catalog = cat

        # Remove problem sources: Flagged by all columns that contain 'Problem'
        problem_flag = cat["Problem_cutout"]
        # Only attributes, not image data, will contain problematic sources.
        for attr in self.__dict__.keys():
            if (
                hasattr(self, attr)
                and attr not in ["data", "catalog"]
                and isinstance(a := getattr(self, attr), Iterable)
                and len(a) == len(cat)
            ):
                setattr(self, attr, getattr(self, attr)[~problem_flag])
        self.catalog = cat[~problem_flag].reset_index(drop=True).iloc[self.subset_idxs]
