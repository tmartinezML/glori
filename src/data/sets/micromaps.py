import random
from io import BytesIO
import tarfile
from pathlib import Path

import h5py
import torch
import wids
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import PowerTransformer
from torch.utils.data.dataloader import default_collate
from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
    SequentialSampler,
)

import utils.paths as paths
from utils.my_logging import get_logger
import data.trf.transforms as T
import data.utils as utils
from data.trf.scalers import LOFARScaler


from datasets import load_from_disk, concatenate_datasets
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


class MicromapDatasetHF:
    """
    HuggingFace datasets-based weighted sampling for Micromap data.
    Loads arrow datasets and supports weighted sampling via WeightedRandomSampler.
    """

    def __init__(
        self,
        dset,
        dset_lookup=paths.MICROMAP_SUBSETS_ARROW,
        weights_file=None,
        weights_fn=lambda x: x.sum() ** 2 + x.prod(),
        split="train",
        mode="LDM-train-mask",
        custom_transform=None,
        ctxt_transform=None,
        post_transform=None,
        scaler="LOFAR_scaler_II",
        output_tuple=("npy",),
        missing_is_error=True,
        max_beam_arcsec=None,
    ):
        """
        Args:
            dset: Dataset name or path
            weights_file: Filename in metadata/ containing weights dict (npy file)
            weights_fn: Function to aggregate weights (default: np.sum)
            split: Dataset split ('train', 'val', 'test')
            mode: Transform mode ('VAE-train', 'LDM-train-mask', etc.)
            custom_transform: Custom transform function
            ctxt_transform: Dict of transforms for context keys
            post_transform: Transform applied after main transforms
            scaler: Scaler name or None
            output_tuple: Tuple of keys to return
            max_beam_arcsec: Filter by beam size (not implemented for arrow yet)
            missing_is_error: Whether to error on missing keys
        """
        self.logger = get_logger("MMDsHF")

        # Assume arrow datasets are in sibling directory
        self.path = utils.parse_dset_path(dset, lookup=dset_lookup)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Arrow dataset not found at {self.path}. "
                f"Please run tar2arrow.py first."
            )

        self.split = split
        self.mode = mode
        self.missing_is_error = missing_is_error

        # Load arrow dataset
        self.arrow_path = self.path / f"{split}.arrow"

        if not self.arrow_path.exists():
            raise FileNotFoundError(
                f"No arrow files found in {self.path / split}. "
                f"Please run tar2arrow.py first."
            )
        self.logger.info(f"Loading arrow dataset from\n\t{self.arrow_path}")
        sel = output_tuple
        if "__key__" not in sel:
            sel += ("__key__",)
        self.dataset = (
            load_from_disk(str(self.arrow_path))
            .sort("__key__")
            .select_columns(list(sel))
        )
        self.logger.info(f"Loaded {len(self.dataset):_} samples")
        # Housekeeping
        self.dataset.cleanup_cache_files()

        # Load weights dict
        if weights_file is not None:
            match weights_file:
                case str():
                    if "/" in weights_file:
                        weights_path = Path(weights_file)
                    else:
                        weights_path = self.path / "metadata" / weights_file
                case Path():
                    weights_path = weights_file
                case _:
                    raise TypeError(
                        f"weights_file must be str or Path, got {type(weights_file)}"
                    )

            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Weights file not found in metadata:\n\t{weights_path}"
                )
            self.logger.info(f"Loading weights from\n\t{weights_path}")
            self.weights_dict = np.load(weights_path, allow_pickle=True).item()
            self.weights_fn = weights_fn

            if len(self.weights_dict) < len(self.dataset):
                self.logger.info(
                    f"Filtering out {(len(self.dataset) - len(self.weights_dict)):_} samples without weights..."
                )
                keys = np.array(self.dataset["__key__"], dtype=np.str_)
                select_idxs = np.argwhere(
                    ~np.isin(
                        keys,
                        np.array(list(set(keys) - set(self.weights_dict.keys()))),
                        assume_unique=True,
                    )
                ).flatten()
                self.dataset = self.dataset.select(select_idxs.tolist())

            elif len(self.weights_dict) > len(self.dataset):
                self.logger.info(
                    f"Filtering weights_dict to match dataset of length {len(self.dataset):_}..."
                )
                self._filter_weights_dict()

            self.logger.info(f"Dataset has {len(self.dataset):_} samples.")

        else:
            self.weights_dict = None
            self.weights_fn = None

        # If desired, filter for resolution
        if max_beam_arcsec is not None:
            self.logger.info(
                f"Filtering for Max beam size (arcsec): {max_beam_arcsec}."
            )
            keys = np.array(self.dataset["__key__"], dtype=np.str_)
            pointing_info_df = pd.read_csv(
                paths.LOFAR_DATA_PARENT / "DR3_pointing_lookup.csv", index_col="mosaic"
            )
            good_pointings = np.array(
                pointing_info_df.index[
                    pointing_info_df["Beam"] <= np.round(max_beam_arcsec / 3600, 5)
                ],
                dtype=np.str_,
            )
            select_idxs = np.argwhere(
                np.isin(np.char.partition(keys, "-")[:, 0], good_pointings)
            ).flatten()
            self.dataset = self.dataset.select(select_idxs.tolist())

            self._filter_weights_dict()
            self.logger.info(f"Dataset has {len(self.dataset):_} samples.")

        # Setup transforms
        self.transforms_dict = {}
        self.scaler = LOFARScaler.load(scaler) if scaler is not None else None
        self.data_transforms = None
        if custom_transform is not None:
            self.data_transforms = custom_transform(
                scale_fn=self.scaler.scale if self.scaler is not None else None
            )
            self.transforms_dict["npy"] = self.data_transforms
        else:
            self.set_transforms(mode)

        # Store transform configs
        if ctxt_transform is None:
            ctxt_transform = {
                k: T.CatalogContextTransform() for k in output_tuple if "context" in k
            }
        assert all(
            key in output_tuple for key in ctxt_transform.keys()
        ), f"ctxt_transform keys must be in output_tuple"
        self.ctxt_transform = ctxt_transform
        self.transforms_dict.update(ctxt_transform)

        self.post_transform = (
            post_transform if post_transform is not None else lambda x: x
        )
        self.output_tuple = output_tuple

        # Set the dataset format to torch for efficient loading
        self.dataset.set_format(type="numpy")

        self.logger.info("HuggingFace dataset initialized.")

    def set_transforms(self, mode):
        """Set transforms based on mode."""
        scale_fn = self.scaler.scale if self.scaler is not None else None

        match mode:
            case "raw":
                self.data_transforms = T.WebdatasetTransformRaw()
            case "maps-scaled":
                self.data_transforms = T.WebdatasetTransformRaw(scale_fn=scale_fn)
            case "VAE-train" | "VAE-eval":
                self.data_transforms = T.MicroMapTransformVAE(scale_fn=scale_fn)
            case "LDM-train-mask" | "LDM-eval-mask":
                self.data_transforms = T.WebdatasetTransformRaw()
            case "test":
                self.data_transforms = T.MicromapTransformTest(scale_fn=scale_fn)
            case _:
                raise ValueError(f"Invalid mode: {mode}")

        self.mode = mode
        self.transforms_dict["npy"] = self.data_transforms
        self.logger.info(f"Dataset mode set to '{mode}'.")

    def _filter_weights_dict(self):
        """Filter weights_dict to match current dataset keys."""
        if self.weights_dict is not None:
            self.logger.info("Filtering weights_dict accordingly...")
            keys_after = set(self.dataset["__key__"])
            keys_before = set(self.weights_dict.keys())
            for key in keys_before - keys_after:
                del self.weights_dict[key]

            assert len(self.dataset) == len(self.weights_dict), (
                f"Weights dict length must match dataset length after filtering, got "
                f"{len(self.weights_dict):_} vs {len(self.dataset):_}"
            )

    def filter(self, *args, **kwargs):
        """Filter dataset using a filter function."""
        self.logger.info("Filtering dataset...")
        initial_len = len(self.dataset)
        self.dataset = self.dataset.filter(*args, **kwargs)
        final_len = len(self.dataset)
        self._filter_weights_dict()
        self.logger.info(
            f"Filtered dataset from {initial_len:_} to {final_len:_} samples."
        )
        return self

    def select(self, *args, **kwargs):
        """Select dataset using a select function."""
        self.logger.info("Selecting dataset...")
        initial_len = len(self.dataset)
        self.dataset = self.dataset.select(*args, **kwargs)
        final_len = len(self.dataset)
        self._filter_weights_dict()
        self.logger.info(
            f"Selected dataset from {initial_len:_} to {final_len:_} samples."
        )
        return self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get item with transforms applied."""
        sample = self.dataset[idx]

        # Apply transforms to specified keys
        sample_dict = {}
        for key in self.output_tuple:
            if key in ("__key__", "__url__"):
                # Metadata keys - pass through
                sample_dict[key] = sample[key]
            elif key in self.transforms_dict:
                # Apply transform
                sample_dict[key] = self.transforms_dict[key](sample[key])
            elif key in sample:
                # No transform, just pass through
                sample_dict[key] = sample[key]
            elif self.missing_is_error:
                raise KeyError(
                    f"Key '{key}' not found in sample {sample.get('__key__', idx)}"
                )
            else:
                self.logger.warning(f"Key '{key}' not found in sample, skipping")
                return None

        # Apply post-transform to entire sample dict
        sample_dict = self.post_transform(sample_dict)

        # Return as tuple in specified order
        return tuple(sample_dict[k] for k in self.output_tuple)

    def get_batch(self, batch_idx, batch_size):
        """
        Get a specific batch by index directly from the dataset.
        Works when using SequentialSampler (no shuffle, no weighted sampling).

        Args:
            batch_idx: Index of the batch (0-indexed)
            batch_size: Size of each batch

        Returns:
            Tuple of batched tensors matching output_tuple
        """
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(self.dataset))

        if start_idx >= len(self.dataset):
            raise IndexError(
                f"Batch index {batch_idx} out of range (dataset has {len(self)} samples)"
            )

        # Get samples directly
        samples = [self[i] for i in range(start_idx, end_idx)]
        return default_collate(samples)

    def get_samples(self, indices):
        """
        Get specific samples by indices directly from the dataset.

        Args:
            indices: List of sample indices
        Returns:
            Tuple of batched tensors matching output_tuple
        """
        samples = [self[i] for i in indices]
        return default_collate(samples)

    def get_dataloader(
        self,
        batch_size=64,
        num_workers=8,
        shuffle=True,
        num_samples=None,
        persistent_workers=True,
        **dataloader_kwargs,
    ):
        """
        Create DataLoader with weighted sampling.

        Args:
            batch_size: Batch size
            num_workers: Number of DataLoader workers
            shuffle: If True, use WeightedRandomSampler
            num_samples: Samples per epoch (default: len(dataset))
            persistent_workers: Keep workers alive between epochs
            **dataloader_kwargs: Additional DataLoader arguments

        Returns:
            torch.utils.data.DataLoader
        """
        sampler = None

        if shuffle and self.weights_dict is not None:
            self.logger.info("Building WeightedRandomSampler...")

            # Build weight tensor aligned with dataset
            self.logger.info("Getting keys and values...")
            weights = [
                self.weights_fn(self.weights_dict[key])
                for key in tqdm(self.dataset["__key__"])
            ]

            weights = torch.tensor(weights, dtype=torch.float32)

            # Normalize
            weights = weights / weights.sum()

            # Calculate number of samples
            num_samples = num_samples or len(self.dataset)
            num_samples = (num_samples // batch_size) * batch_size

            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=num_samples,
                replacement=True,
            )
            self.logger.info(
                f"WeightedRandomSampler created with {num_samples} samples/epoch"
            )

        # Build DataLoader kwargs
        kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            shuffle=(shuffle and sampler is None),  # Only shuffle if no sampler
            pin_memory=True,
            persistent_workers=(persistent_workers and num_workers > 0),
        )
        kwargs.update(dataloader_kwargs)

        return DataLoader(self, **kwargs)

    def get_key(self, idx):
        """Get the __key__ for a given index (for compatibility)."""
        return self.dataset[idx]["__key__"]
