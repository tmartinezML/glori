import copy
import random
import json
from collections.abc import Iterable

import h5py
import torch
import wids
import tarfile
import numpy as np
import pandas as pd
import webdataset as wds
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import PowerTransformer

import data.trf.functional
import utils.my_logging
import utils.paths as paths
import data.trf.transforms as T
import data.utils as utils
from data.trf.scalers import LOFARScaler
from plotting.images import plot_image_grid

ALLOWED_KEYS = [
    "__key__",
    "model.npy",
    "model_beam.npy",
    "mask.npy",
    "maj_min.npy",
    "mask_size.npy",
    "circle_radius.npy",
]


class PrototypesDataset(wds.WebDataset):
    def __init__(
        self,
        dset,
        transform="TrainTransformPrototypesWDS",
        ctxt_transform={},
        output_tuple=("model.npy",),
        shuffle=2000,
        resampled=True,
        image_size=128,
        load_catalog=False,
        select=None,
        count=True,
    ):

        # Set logging
        self.logger = utils.my_logging.get_logger(self.__class__.__name__)
        # Set the path for the dataset
        self.path = (
            utils.parse_dset_path(dset, lookup=paths.LOFAR_SUBSETS) / "images.tar"
        )
        self.urls = [str(self.path)]
        if not self.path.exists():
            raise FileNotFoundError(
                f"File <{self.path}> not found. " "Please check the dataset path."
            )
        self.logger.info(f"Loading dataset from \n\t<{self.path}>.")

        # Check output tuple
        if not all(key in ALLOWED_KEYS for key in output_tuple):
            raise ValueError(
                f"Output tuple contains invalid keys. "
                f"Allowed keys are: {ALLOWED_KEYS}, "
                f"got {output_tuple}."
            )

        # Set transforms and behavior for different modes
        self.data_transforms = getattr(data.trf.transforms, transform)(
            image_size=image_size
        )
        self._context = []

        # Get length of dataset
        self._len = self._count_samples() if count else None
        if self._len is not None:
            self.logger.info(f"Dataset length: {self._len:_}")

        # initialize the dataset
        self.output_tuple = output_tuple
        super().__init__(
            self.urls,
            resampled=resampled,
            shardshuffle=False,
            empty_check=False,
            # nodesplitter=wds.shardlists.split_by_node,
        )
        if select is not None:
            self = self.select(select)
        if shuffle is not None and bool(shuffle):
            self.append(wds.shuffle(shuffle))
        self.append(wds.decode())
        self.append(
            wds.map_dict(**{"model.npy": self.data_transforms, **ctxt_transform})
        )
        self.append(wds.to_tuple(*self.output_tuple))

        # Load catalog if required
        if load_catalog:
            self.logger.info("Loading catalog...")
            self.cat = pd.read_parquet(self.path.parent / "catalog.parquet")

        self.logger.info("Data set initialized.")

    def index_slice(self, idx):
        # Slice all attributes that have the same shape as self.data
        for attr in self.__dict__.keys():
            if (
                hasattr(self, attr)
                and attr != "data"
                and isinstance(a := getattr(self, attr), Iterable)
                and len(a) == len(self.data)
            ):
                if isinstance(a, pd.DataFrame) and idx.dtype != bool:
                    setattr(self, attr, a.iloc[idx])

                else:
                    setattr(self, attr, a[idx])

        self.data = self.data[idx]

    def index_sliced(self, idx):
        subset = copy.deepcopy(self)
        subset.index_slice(idx)
        return subset

    def set_context(self, *args):
        assert all(hasattr(self, attr) for attr in args), (
            "Context attributes not found in dataset: "
            f"{[attr for attr in args if not hasattr(self, attr)]}"
        )
        assert all(len(getattr(self, attr)) == len(self.data) for attr in args), (
            f"Context attributes do not have the same length as data: ({len(self.data)})"
            f"{[(attr, len(getattr(self, attr))) for attr in args if len(getattr(self, attr)) != len(self.data)]}"
        )
        self._context = args

    def plot_image_grid(
        self, idxs=None, n_imgs=64, plot_masks=True, show_titles=True, **kwargs
    ):
        # pick n_imgs random images
        idxs = (
            idxs
            if idxs is not None
            else np.random.choice(len(self), n_imgs, replace=False)
        )
        masks = self.masks[idxs] if plot_masks and hasattr(self, "masks") else None
        titles = [f"{self.names[i]}\n({i})" for i in idxs] if show_titles else None
        vmin = -1 if data.trf.functional.train_scale_present(self.transforms) else 0

        # Plot
        return plot_image_grid(
            [self.transforms(self.data[i]) for i in idxs],
            titles=titles,
            masks=masks,
            vmin=vmin,
            **kwargs,
        )

    def _count_samples(self):
        self.logger.info("Counting samples in the dataset...")
        try:
            with tarfile.open(self.path, "r") as tar:

                self._len = len(
                    [
                        m
                        for m in tqdm(tar.getmembers(), desc="Counting", leave=False)
                        if m.name.endswith("model.npy")
                    ]
                )
                return self._len
        except Exception as e:
            self.logger.warning(
                f"Failed to count samples: {str(e)}. Setting length to None."
            )
            self._len = None
            return None
