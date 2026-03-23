import data.trf.functional
import data.trf.transforms as T
import data.utils as utils
import utils.my_logging
import utils.paths as paths
from data.trf.scalers import LOFARScaler
from plotting.images import plot_image_grid


import numpy as np
import pandas as pd
import webdataset as wds
from tqdm import tqdm


import copy
import json
from collections.abc import Iterable


class MicromapDataset(wds.WebDataset):
    def __init__(
        self,
        dset,
        split="train",
        mode="VAE-train",
        custom_transform=None,
        ctxt_transform={},
        post_transform=None,
        scaler="LOFAR_scaler_II",
        output_tuple=(
            "npy",
        ),  # Can contain: "npy", "__key__", "center.npy", "center_radec.npy", "wcs"
        shardshuffle=1000,
        shuffle=None,
        resampled=True,
        max_beam_arcsec=None,
        missing_is_error=True,
    ):

        # Set logging
        self.logger = utils.my_logging.get_logger("MMD")
        # Set the path for the dataset
        self.path = utils.parse_dset_path(dset, lookup=paths.MICROMAP_SUBSETS)
        self.split = split
        self.urls = sorted([str(p) for p in (self.path / self.split).glob("*.tar")])
        n_url = len(self.urls)
        if n_url == 0:
            raise FileNotFoundError(
                f"No tar files found in {self.path / self.split}. "
                "Please check the dataset path and split."
            )
        self.logger.info(
            f"Loading dataset from \n\t{self.path}\n\tFound {len(self.urls)} tar files."
        )
        if max_beam_arcsec is not None:
            self.logger.info(
                f"Max beam size (arcsec): {max_beam_arcsec}. Removing samples with larger beams."
            )
            pointing_info_df = pd.read_csv(
                paths.LOFAR_DATA_PARENT / "DR3_pointing_lookup.csv", index_col="mosaic"
            )
            self.urls = [
                url
                for url in self.urls
                if pointing_info_df.loc[url.split("/")[-1].split(".")[0], "Beam"]
                <= np.round(max_beam_arcsec / 3600, 5)
            ]
            self.logger.info(
                f"Removed {n_url - len(self.urls)} samples with beams larger than {max_beam_arcsec} arcsec."
            )
            n_url = len(self.urls)
            self.logger.info(f"Remaining urls: {n_url}")

        # Set transforms and behavior for different modes
        self.scaler = LOFARScaler.load(scaler) if scaler is not None else None
        self.mode = mode
        self.data_transforms = None
        if custom_transform is not None:
            self.data_transforms = custom_transform(
                scale_fn=self.scaler.scale if self.scaler is not None else None
            )
        else:
            self.set_transforms(mode)
        self._context = []

        # initialize the dataset
        self.output_tuple = output_tuple
        super().__init__(
            self.urls,
            shardshuffle=shardshuffle,
            resampled=resampled,
            nodesplitter=wds.shardlists.split_by_node,
        )
        if not missing_is_error:
            # Add filter for existing keys
            self.append(wds.select(lambda x: all(k in x for k in self.output_tuple)))
        self.append(wds.decode())
        # Assert that ctxt_transform keys are in output_tuple
        assert all(
            key in output_tuple for key in ctxt_transform.keys()
        ), f"ctxt_transform keys must be in output_tuple, got {ctxt_transform.keys()} and {output_tuple}."

        self.append(wds.map_dict(npy=self.data_transforms, **ctxt_transform))
        if post_transform is not None:
            self.append(wds.map(post_transform))
        if shuffle is not None and bool(shuffle):
            self.append(wds.shuffle(shuffle))
        self.append(wds.to_tuple(*self.output_tuple))
        # self.append(wds.with_epoch(self._len // n_gpu))
        # self.append(wds.map(lambda x: x[0]))

        # Get length of dataset
        self._len = None
        self._count_samples()
        self.logger.info(f"Dataset length: {self._len:_}")

        self.logger.info("Data set initialized.")

    def set_transforms(self, mode):
        """
        Set the mode for the dataset. This will change the transforms applied to the data.
        """
        scale_fn = self.scaler.scale if self.scaler is not None else None
        # Set shape of training images
        match mode:
            case "VAE-train" | "VAE-eval":
                self.data_transforms = T.MicroMapTransformVAE(scale_fn=scale_fn)

            case "LDM-train-mask" | "LDM-eval-mask":
                self.data_transforms = T.WebdatasetTransformRaw()
                # Changed on Aug. 9th 25 after implementing encodings-512
                # self.data_transforms = T.EncodingTransformLDM(scale_fn=scale_fn)

            case "test":
                self.data_transforms = T.MicromapTransformTest(scale_fn=scale_fn)
            case _:
                raise ValueError(f"Invalid mode: {mode}")

        self.mode = mode
        self.logger.info(f"Dataset mode set to '{mode}'.")

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
        # Count the number of samples in the dataset
        # meta_files = sorted((self.path / "metadata" / self.split).glob("*.json"))
        meta_files = sorted(
            [
                self.path
                / "metadata"
                / self.split
                / f"{f.split('/')[-1].split('.')[0]}.json"
                for f in self.urls
            ]
        )
        sum = 0
        for meta_file in tqdm(
            meta_files, desc="Counting samples", unit="file", ncols=80
        ):
            with open(meta_file, "r") as f:
                data = json.load(f)
                sum += len(data["names"])

        self._len = sum


class MicromapPreparationDataset(wds.WebDataset):
    def __init__(
        self,
        dset,
        split="train",
        scaler="LOFAR_scaler_II",
        custom_transform=None,
        output_tuple=(
            "npy",
            "__key__",
            "__url__",
        ),  # Can contain: "npy", "__key__", "__url__", "center.npy", "center_radec.npy", "wcs"
        max_beam_arcsec=None,
    ):

        # Set logging
        self.logger = utils.my_logging.get_logger(self.__class__.__name__)
        # Set the path for the dataset
        self.path = utils.parse_dset_path(dset, lookup=paths.MICROMAP_SUBSETS)
        self.split = split
        self.urls = sorted([str(p) for p in (self.path / self.split).glob("*.tar")])
        n_url = len(self.urls)
        if n_url == 0:
            raise FileNotFoundError(
                f"No tar files found in {self.path / self.split}. "
                "Please check the dataset path and split."
            )
        self.logger.info(
            f"Loading dataset from \n\t{self.path}\n\tFound {len(self.urls)} tar files."
        )

        # Get length of dataset
        self._len = None
        self._count_samples()
        self.logger.info(f"Dataset length: {self._len:_}")

        # Set transforms and behavior for different modes
        self.scaler = LOFARScaler.load(scaler) if scaler is not None else None
        trf_fn = custom_transform or T.MicroMapTransformLDM
        self.data_transforms = trf_fn(
            scale_fn=self.scaler.scale if self.scaler is not None else None
        )
        self._context = []

        # initialize the dataset
        self.output_tuple = output_tuple
        super().__init__(
            self.urls,
            resampled=False,
            shardshuffle=False,
            nodesplitter=wds.shardlists.split_by_node,
        )
        self.append(wds.decode())
        self.append(wds.map_dict(npy=self.data_transforms))
        self.append(wds.to_tuple(*self.output_tuple))

        self.logger.info("Data set initialized.")

    def _count_samples(self):
        # Count the number of samples in the dataset
        # meta_files = sorted((self.path / "metadata" / self.split).glob("*.json"))
        meta_files = sorted(
            [
                self.path
                / "metadata"
                / self.split
                / f"{f.split('/')[-1].split('.')[0]}.json"
                for f in self.urls
            ]
        )
        sum = 0
        for meta_file in tqdm(
            meta_files, desc="Counting samples", unit="file", ncols=80
        ):
            with open(meta_file, "r") as f:
                data = json.load(f)
                sum += len(data["names"])

        self._len = sum
