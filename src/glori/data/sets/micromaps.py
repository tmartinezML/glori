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

import glori.settings.paths as paths
from glori.infra.logging import get_logger
from glori.config.micromaps_config import MicromapsConfig, resolve_micromap_kwargs
import glori.data.trf.transforms as T
import glori.data.load as load
from glori.data.trf.scalers import LOFARScaler


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
        dataset,
        dataset_lookup=paths.MICROMAP_SUBSETS_ARROW,
        split="train",
        output_tuple=("npy",),
        max_beam_arcsec=None,
        data_transform=None,
        ctxt_transform=None,
        post_transform=None,
        scaler="LOFAR_scaler_II",
        weights_file=None,
        weights_fn=None,
        missing_is_error=True,
    ):
        """
        Args:
            dset: Dataset name or path
            weights_file: Filename in metadata/ containing weights dict (npy file)
            weights_fn: Function to aggregate weights (default: np.sum)
            split: Dataset split ('train', 'val', 'test')
            mode: Transform mode ('VAE-train', 'LDM-train-mask', etc.)
            data_transform: Transform function for data
            ctxt_transform: Dict of transforms for context keys
            post_transform: Transform applied after main transforms
            scaler: Scaler name or None
            output_tuple: Tuple of keys to return
            max_beam_arcsec: Filter by beam size (not implemented for arrow yet)
            missing_is_error: Whether to error on missing keys
        """
        self.logger = get_logger("MMDsHF")

        # Assume arrow datasets are in sibling directory
        self.path = load.parse_dset_path(dataset, lookup=dataset_lookup)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Arrow dataset not found at {self.path}. "
                f"Please run tar2arrow.py first."
            )

        self.output_tuple = output_tuple
        self.missing_is_error = missing_is_error

        # Load arrow dataset
        self.arrow_path = self.path / f"{split}.arrow"
        if not self.arrow_path.exists():
            raise FileNotFoundError(
                f"No arrow files found in {self.path / split}. "
                f"Please run tar2arrow.py first."
            )
        self.logger.info(f"Loading arrow dataset from\n\t{self.arrow_path}")

        select_cols = self.output_tuple
        if "__key__" not in select_cols:
            select_cols += ("__key__",)
        self.dataset = (
            load_from_disk(str(self.arrow_path))
            .sort("__key__")
            .select_columns(list(select_cols))
        )
        self.logger.info(f"Loaded {len(self.dataset):_} samples")
        # Housekeeping
        self.dataset.cleanup_cache_files()

        # Load weights dict
        self.weights_dict = None
        self.weights_fn = None
        if weights_file is not None:
            self._load_weights(weights_file, weights_fn)

        # If desired, filter for resolution
        if max_beam_arcsec is not None:
            self._filter_for_max_beam(max_beam_arcsec)

        # Setup transforms
        self.transforms_dict = {}
        if data_transform is not None:
            self.transforms_dict["npy"] = data_transform
        if ctxt_transform is not None:
            self.transforms_dict.update(ctxt_transform)

        # Setup post-transform
        if post_transform is None:
            post_transform = lambda x: x
        self.post_transform = post_transform

        # Load scaler
        self.scaler = None
        if scaler is not None:
            self.scaler = LOFARScaler.load(scaler)

        # Set the dataset format to torch for efficient loading
        self.dataset.set_format(type="numpy")

        self.logger.info("HuggingFace dataset initialized.")

    @classmethod
    def from_config(cls, config: MicromapsConfig, **override) -> "MicromapDatasetHF":
        """Create a MicromapDatasetHF from a MicromapsConfig, allowing for overrides."""
        return cls(**resolve_micromap_kwargs(config, **override))

    @classmethod
    def from_preset(cls, preset: str | Path, **override) -> "MicromapDatasetHF":
        """Create a MicromapDatasetHF from a preset name or path, allowing for overrides."""
        return cls.from_config(MicromapsConfig.from_preset(preset), **override)

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

    def _load_weights(self, weights_file, weights_fn):
        weights_path = load.parse_weights_file(weights_file, parent=self.path)
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
            self.logger.info(f"Dataset has {len(self.dataset):_} samples.")

        elif len(self.weights_dict) > len(self.dataset):
            self.logger.info(
                f"Filtering weights_dict to match dataset of length {len(self.dataset):_}..."
            )
            self._filter_weights_dict()

    def _filter_for_max_beam(self, max_beam_arcsec):
        """Filter dataset for maximum beam size."""
        self.logger.info(f"Filtering for Max beam size (arcsec): {max_beam_arcsec}.")
        keys = np.array(self.dataset["__key__"], dtype=np.str_)
        # TODO: This should probably not be hard-coded here
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
