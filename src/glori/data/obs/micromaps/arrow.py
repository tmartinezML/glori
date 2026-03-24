import os
import gc
import sys
import shutil
import subprocess
from pathlib import Path
import pandas as pd
from astropy.wcs import WCS
from functools import partial
import traceback
import pickle

import datasets

from datasets import Dataset
from datasets.arrow_writer import ArrowWriter
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.compute as pc
import numpy as np
from tqdm import tqdm, trange
from datetime import datetime
from astropy.table import Table
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import glori.data.obs.micromaps.arrow_functions as afc
import glori.settings.paths as paths
import glori.data.obs.micromaps.arrow_utils as au
from glori.models.load import parse_lightning_ckpt
from glori.data.sets.micromaps import MicromapDatasetHF
from glori.infra.logging import get_logger, add_file_handler
from glori.data.load import load_lotss_catalog, parse_dset_path
from glori.data.obs.micromaps.utils import (
    get_micromaps,
    filter_catalog_by_wcs,
    graceful_shutdown,
    get_mosaic_shape,
    maps_to_encoding_path,
)
from glori.data.obs.micromaps.monitoring import monitor
from glori.data.trf.scalers import LOFARScaler


logger = get_logger("mm-arrow")

from multiprocessing import get_context

# ...existing code...

_SRC_LOOKUP = None
_SRC_FIELDS = None
_SRC_COLUMNS = None


def _init_keymerge_worker(src_lookup, src_fields, columns):
    global _SRC_LOOKUP, _SRC_FIELDS, _SRC_COLUMNS
    _SRC_LOOKUP = src_lookup
    _SRC_FIELDS = src_fields
    _SRC_COLUMNS = columns


def _merge_one_tgt_shard_by_key(
    tgt_path: Path,
    out_path: Path,
    require_key_match: bool,
    merge_batch_rows: int,
):
    # Uses globals initialized once per worker
    if _SRC_LOOKUP is None or _SRC_FIELDS is None or _SRC_COLUMNS is None:
        raise RuntimeError("Key-merge worker not initialized.")

    seen_keys = set() if require_key_match else None

    with pa.memory_map(str(tgt_path), "r") as s_tgt:
        r_tgt = ipc.open_stream(s_tgt)

        schema = r_tgt.schema
        for f in _SRC_FIELDS:
            schema = schema.append(f)

        with pa.OSFile(str(out_path), "wb") as sink:
            writer = ipc.new_stream(sink, schema)

            src_keys = _SRC_LOOKUP["__key__"]

            for batch_idx, b_tgt in enumerate(r_tgt):
                for row_start in range(0, b_tgt.num_rows, merge_batch_rows):
                    sub_len = min(merge_batch_rows, b_tgt.num_rows - row_start)
                    b_tgt_sub = b_tgt.slice(row_start, sub_len)
                    tgt_keys = b_tgt_sub.column("__key__")
                    idx = pc.index_in(tgt_keys, value_set=src_keys)

                    if require_key_match:
                        miss = pc.is_null(idx)
                        if pc.any(miss).as_py():
                            miss_np = miss.to_numpy(zero_copy_only=False)
                            sub_row_idx = int(np.flatnonzero(miss_np)[0])
                            bad_key = tgt_keys[sub_row_idx].as_py()
                            raise ValueError(
                                f"Target key not found in source: key={bad_key}, "
                                f"tgt={tgt_path}, batch={batch_idx}, row={row_start + sub_row_idx}"
                            )

                    if require_key_match:
                        seen_keys.update(tgt_keys.to_pylist())

                    append_arrays = [pc.take(_SRC_LOOKUP[c], idx) for c in _SRC_COLUMNS]
                    new_cols = [
                        b_tgt_sub.column(i) for i in range(b_tgt_sub.num_columns)
                    ] + append_arrays
                    writer.write_batch(
                        pa.RecordBatch.from_arrays(new_cols, schema.names)
                    )

            writer.close()

    return seen_keys


def _build_src_lookup_table(
    src_shards: list[Path],
    columns: list[str],
) -> tuple[dict[str, pa.ChunkedArray], list[pa.Field]]:
    key_chunks: list[pa.Array] = []
    col_chunks: dict[str, list[pa.Array]] = {c: [] for c in columns}
    src_fields: list[pa.Field] | None = None

    for src_path in tqdm(src_shards, desc="Reading source shards", dynamic_ncols=True):
        with pa.memory_map(str(src_path), "r") as s_src:
            r_src = ipc.open_stream(s_src)

            missing = [c for c in columns if c not in r_src.schema.names]
            if missing:
                raise ValueError(
                    f"Missing columns in source shard {src_path}: {missing}"
                )

            if src_fields is None:
                src_fields = [r_src.schema.field(c) for c in columns]

            for b_src in r_src:
                key_chunks.append(b_src.column("__key__"))
                for c in columns:
                    col_chunks[c].append(b_src.column(c))

    if src_fields is None:
        raise ValueError("No source shards found or source shards are empty.")

    src_lookup: dict[str, pa.ChunkedArray] = {"__key__": pa.chunked_array(key_chunks)}
    for field in src_fields:
        src_lookup[field.name] = pa.chunked_array(
            col_chunks[field.name], type=field.type
        )

    n_rows = len(src_lookup["__key__"])
    n_unique = pc.count_distinct(src_lookup["__key__"]).as_py()
    if n_rows != n_unique:
        raise ValueError(
            f"Duplicate __key__ in source: rows={n_rows}, unique={n_unique}"
        )

    return src_lookup, src_fields


def _merge_split_by_key(
    src_shards: list[Path],
    tgt_shards: list[Path],
    out_split: Path,
    columns: list[str],
    require_key_match: bool,
    max_workers: int = 8,
    merge_batch_rows: int = 8,
) -> None:
    src_lookup, src_fields = _build_src_lookup_table(src_shards, columns)
    seen_tgt_keys: set[str] = set()

    # Parallel over target shards
    mp_ctx = get_context("fork")  # Linux: share src_index via COW
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
        initializer=_init_keymerge_worker,
        initargs=(src_lookup, src_fields, columns),
    ) as ex:
        futures = {
            ex.submit(
                _merge_one_tgt_shard_by_key,
                tgt_path,
                out_split / tgt_path.name,
                require_key_match,
                merge_batch_rows,
            ): tgt_path
            for tgt_path in tgt_shards
        }

        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Merging target shards",
            dynamic_ncols=True,
        ):
            shard_seen = f.result()
            if require_key_match and shard_seen is not None:
                seen_tgt_keys.update(shard_seen)

    if require_key_match:
        extra_src = set(src_lookup["__key__"].to_pylist()) - seen_tgt_keys
        if extra_src:
            sample = list(sorted(extra_src))[:5]
            raise ValueError(
                f"Source has keys not present in target (count={len(extra_src)}). "
                f"Sample: {sample}"
            )


def import_columns_parallel(
    src_dset: Path | str,
    target_dset: Path | str,
    columns: list[str],
    out_dir: Path | None = None,
    dset_lookup=paths.MICROMAP_SUBSETS_ARROW,
    splits: list[str] = ["train", "val", "test"],
    require_key_match: bool = True,
    max_workers: int = 8,
    merge_batch_rows: int = 8,
) -> None:
    src_dir = parse_dset_path(src_dset, lookup=dset_lookup)
    target_dir = parse_dset_path(target_dset, lookup=dset_lookup)
    logger.info(
        f"Importing columns\n\t{columns}\n\tfrom {src_dir}\n\tto {target_dir}\n\tfor splits {splits}"
    )

    if out_dir is None:
        out_dir = target_dir.with_name(target_dir.name + "_ctxt-imported")

    for split in splits:
        logger.info(f"Processing split: {split}")

        src_shards = sorted(Path(src_dir, f"{split}.arrow").glob("data-*.arrow"))
        tgt_shards = sorted(Path(target_dir, f"{split}.arrow").glob("data-*.arrow"))

        out_split = Path(out_dir, f"{split}.arrow")
        out_split.mkdir(parents=True, exist_ok=True)

        logger.info(f"{split}: using key-based merge (row order independent)")
        _merge_split_by_key(
            src_shards=src_shards,
            tgt_shards=tgt_shards,
            out_split=out_split,
            columns=columns,
            require_key_match=require_key_match,
            max_workers=max_workers,  # pass through
            merge_batch_rows=merge_batch_rows,
        )

    logger.info(f"All done! Merged columns saved to {out_dir}")


@monitor(parse_args=True, logger=logger)
def make_encodings_dset(
    dset,
    dset_lookup=paths.MICROMAP_SUBSETS_ARROW,
    vae_model="VQ-VAE-256",
    vae_checkpoint="best",
    batch_size=128,
    debug=False,
    splits=["train", "val", "test"],
    device="cuda",
    override=False,
    compile_model=True,
    scaler="LOFAR_scaler_II",
):
    """
    Create a new Arrow dataset with VQ-VAE encodings from an existing micromap dataset.

    This function streams through the input dataset using .map(), encodes images
    in batches using a VQ-VAE model, and writes the encodings to a new dataset.

    Parameters
    ----------
    dset : str or Path
        Path or identifier of the input micromap dataset
    vae_model : str, optional
        Name of the VAE model (default: "VQ-VAE-256")
    vae_checkpoint : str or Path
        Path to the VQ-VAE checkpoint file to load (default: "best")
    batch_size : int, optional
        Batch size for encoding (default: 128)
    debug : bool, optional
        If True, processes only a subset of data (default: False)
    splits : list of str, optional
        Which splits to process (default: ["train", "val", "test"])
    device : str, optional
        Device to run the model on (default: "cuda")
    override : bool, optional
        If True, overwrite existing output directory (default: False)
    compile_model : bool, optional
        If True, compile the model with torch.compile for faster inference (default: True)

    Returns
    -------
    Path
        Path to the created encodings dataset directory
    """
    import torch
    from glori.models.vae.vqvae import VQVAE

    logger.divider()
    if debug:
        logger.setLevel("DEBUG")

    # Parse input dataset path
    maps_dset_path = parse_dset_path(dset, lookup=dset_lookup)
    logger.info(f"Loading micromap dataset from:\n\t{maps_dset_path}")

    # Create output directory
    out_parent = maps_to_encoding_path(maps_dset_path)
    if debug:
        out_parent = out_parent.with_name(out_parent.name + "_debug")
        logger.debug(f"Debug mode: output directory will be created as {out_parent}")

    if out_parent.exists():
        if debug or override:
            logger.warning(
                f"Output directory already exists and will be overridden:\n\t{out_parent}"
            )
            shutil.rmtree(out_parent)
        else:
            raise FileExistsError(
                f"Output directory already exists: {out_parent}. Use override=True to overwrite."
            )

    out_parent.mkdir(exist_ok=True)

    # Load VQ-VAE model
    logger.info(f"Loading VQ-VAE model from:\n\t{vae_checkpoint}")
    ckpt = parse_lightning_ckpt(vae_checkpoint, model_name=vae_model)
    vae = VQVAE.load_from_checkpoint(ckpt, map_location="cpu")
    vae.eval()
    vae.to(device)

    # Compile model for faster inference
    if compile_model and "cuda" in device:
        logger.info("Compiling model with torch.compile for faster inference...")
        vae.encode_to_prequant = torch.compile(
            vae.encode_to_prequant, mode="max-autotune"
        )

    # Load scaler
    logger.info(f"Loading scaler: {scaler}")
    scaler = LOFARScaler.load(scaler)

    logger.info(f"Model loaded successfully on {device}")

    # Define encoding function to be mapped over the dataset
    def encode_batch(batch):
        """Encode a batch of images using the VQ-VAE model."""
        # Stack images into a batch
        img_batch = np.stack(batch["npy"])

        # Apply scaling
        img_batch = scaler.scale(img_batch)

        # Convert to torch tensor and add channel dimension if needed
        img_tensor = torch.from_numpy(img_batch).to(device, dtype=torch.float32)
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(1)  # Add channel dimension

        # Encode with VQ-VAE
        with torch.no_grad():
            encoded = vae.encode_to_prequant(img_tensor).cpu().numpy()

        # Replace npy with encodings
        batch["npy"] = [enc.astype(np.float32) for enc in encoded]

        del img_tensor, encoded
        return batch

    # Process each split
    for split in splits:
        logger.separator(f"Processing split: {split}")

        split_dir = maps_dset_path / f"{split}.arrow"
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}, skipping...")
            continue

        # Load dataset
        logger.info(f"Loading dataset for split: {split}")
        maps_dset = datasets.load_from_disk(str(split_dir))

        if debug:
            logger.debug("Debug mode: selecting only first 50 samples")
            maps_dset = maps_dset.select(range(min(50, len(maps_dset))))

        logger.info(f"Dataset loaded with {len(maps_dset)} samples")

        # Apply encoding function using map
        logger.info(f"Encoding images...")
        encoded_dset = maps_dset.map(
            encode_batch,
            batched=True,
            batch_size=batch_size,
            desc=f"Encoding {split}",
        )

        # Determine number of shards
        num_shards = 2 ** int(np.log2(len(maps_dset.cache_files))) if not debug else 1
        logger.info(f"Saving encoded dataset with {num_shards} shards...")

        # Save to disk
        tmp_path = out_parent / f"{split}-tmp.arrow"
        encoded_dset.save_to_disk(
            str(tmp_path),
            num_shards=num_shards,
            num_proc=1,  # Single process to avoid model replication
        )

        # Cleanup
        logger.info("Cleaning up references...")
        encoded_dset.cleanup_cache_files()
        del encoded_dset
        maps_dset.cleanup_cache_files()
        del maps_dset

        # Garbage collection
        for i in range(3):
            collected = gc.collect()
            logger.debug(f"GC pass {i+1}: collected {collected} objects")

        # Move to final location
        final_path = out_parent / f"{split}.arrow"
        if final_path.exists():
            shutil.rmtree(final_path)
        tmp_path.rename(final_path)

        logger.info(f"Split '{split}' saved to:\n\t{final_path}")

    logger.info(f"Encodings dataset saved to:\n\t{out_parent}")
    logger.info("Done!")

    return out_parent


@monitor(parse_args=True, logger=logger)
def create_arrow_dataset_parallel(
    micromap_size,
    spacing=1,
    override=False,
    debug=False,
    prefix="micromaps",
    keys_from=None,
    mosaic_dir=paths.MOSAIC_DIR_DR3,
    max_workers=32,
    splits=["train", "val", "test"],
    raise_errors=False,
    max_shard_size_gb=1.0,
):
    """
    Create HF Arrow dataset from image cutouts with parallel extraction.

    Parameters
    ----------
    input_files : list[Path]
        List of files to extract cutouts from
    output_dir : Path
        Where to save the arrow dataset
    num_shards : int
        Number of shards to create
    max_workers : int
        Number of parallel workers for extraction
    shard_size_gb : float
        Approximate target size for each shard in GB
    """

    logger.divider()
    if debug:
        logger.setLevel("DEBUG")
    logger.info(
        f"Creating micromaps with size {micromap_size} and spacing {spacing}.\n"
    )
    out_parent = paths.MICROMAP_DIR_ARROW / f"{prefix}-{micromap_size}px-{spacing=}"
    logger.info(f"Saving micromaps as arrow dataset archive to: \n\t{out_parent}.\n")

    if debug:
        out_parent = out_parent.with_name(out_parent.name + "_debug")
        logger.debug(f"Debug mode: output directory will be created as {out_parent}.")
        raise_errors = True

    # Check if the output file already exists
    if out_parent.exists():
        if override or debug:
            logger.info(f"Recursively removing existing directory: \n\t{out_parent}.\n")
            shutil.rmtree(out_parent)
        else:
            logger.info(f"Micromap dataset already exists. Aborting for safety.")
            sys.exit(0)

    # If not, create it
    out_parent.mkdir()
    for split in splits:
        (out_parent / f"{split}.arrow").mkdir()

    # Look for mosaics
    logger.info(f"Looking for mosaics in \n\t{mosaic_dir}...")
    pointings = sorted([p.name for p in mosaic_dir.iterdir() if p.is_dir()])
    logger.info(f"Found {len(pointings)} mosaics.")
    input_files = [mosaic_dir / f"{mosaic}/mosaic-blanked.fits" for mosaic in pointings]

    # Lazy single-current-writer per split with size-based rolling.
    # Each split keeps only the active writer and its path; when the on-disk
    # file reaches max_shard_size_gb we finalize/close it and advance index.
    max_shard_bytes = int(max_shard_size_gb * 1024**3)
    shards = {s: {"path": None, "writer": None, "idx": 0} for s in splits}
    for split in splits:
        (out_parent / f"{split}.arrow").mkdir(exist_ok=True, parents=True)
    total_samples = {s: 0 for s in splits}

    # Read centers and assigned split from input if provided
    keys = {}
    if keys_from is not None:
        logger.info(f"Reading cutout centers and splits from {keys_from}...")
        keys = {p: ([], []) for p in pointings}
        for split in splits:
            col = au.extract_column(
                keys_from,
                split,
                "__key__",
            )
            kk, _, centers = np.char.partition(col, sep="-").T
            unique_kk, u_idx = np.unique(kk, return_inverse=True)
            centers = {
                unique_kk[i]: np.stack(np.char.split(centers[u_idx == i], "-", 2))
                .astype(int)
                .tolist()
                for i in trange(len(unique_kk), desc=split)
            }
            for k in centers.keys():
                keys[k][0].extend(centers[k])
                keys[k][1].extend([split] * len(centers[k]))

        # Remove empty entries (mosaics with no valid cutouts)
        keys = {k: v for k, v in keys.items() if len(v[0]) > 0}

        # Update input files to only those with keys
        input_files = [f for f in input_files if f.parent.name in keys.keys()]
        logger.info(
            f"After filtering, {len(input_files)} mosaics will be processed based on provided keys."
        )

    # In debug mode, reduce the number of files
    if debug:
        input_files = input_files[:4]
        logger.debug(
            f"Debug mode: limiting to first 4 mosaics: {[f.parent.name for f in input_files]}."
        )
        max_workers = 2

    get_args = lambda idx: (
        input_files[idx],
        micromap_size,
        spacing,
        *keys.get(input_files[idx].parent.name, (None, None)),
    )

    logger.info(f"Parallel processing with {max_workers} workers.")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit only this batch
        futures = {
            executor.submit(afc.get_samples_from_mosaic, *get_args(i)): input_files[i]
            for i in range(len(input_files))
        }

        # Process results as they complete. Use a named tqdm so we can update the
        # description to include current shard indices per split.
        pbar = tqdm(
            as_completed(futures),
            total=len(input_files),
            desc="Processing mosaics",
            unit="file",
            smoothing=0.01,
        )
        for future in pbar:
            try:
                samples = future.result()

                # Distribute samples across shards (single active writer per split).
                for split in splits:
                    split_samples = samples[split]
                    if split_samples is None:
                        continue
                    shards_split = shards[split]
                    idx = shards_split["idx"]

                    # ensure an active writer exists for this index
                    if shards_split["writer"] is None:
                        p = out_parent / f"{split}.arrow" / f"data-{idx:05d}.arrow"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        w = ArrowWriter(path=str(p), writer_batch_size=1000)
                        shards_split["path"] = p
                        shards_split["writer"] = w

                    # write sample
                    shards_split["writer"].write_table(split_samples)
                    total_samples[split] += 1

                    # check on-disk size and roll if needed
                    try:
                        size = (
                            shards_split["path"].stat().st_size
                            if shards_split["path"] is not None
                            else 0
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not get size for shard file {shards_split['path']} (assuming size 0): {e}"
                        )
                        size = 0
                    if size >= max_shard_bytes:
                        try:
                            shards_split["writer"].finalize()
                        except Exception as e:
                            logger.warning(
                                f"Could not finalize shard file {shards_split['path']}: {e}"
                            )
                            pass
                        # close and advance index; new writer will be created lazily
                        shards_split["writer"] = None
                        shards_split["path"] = None
                        shards_split["idx"] += 1

                # Explicitly delete to release memory immediately
                del samples
                futures.pop(future, None)
                gc.collect()

                # update progress bar description with current shard indices per split
                try:
                    desc = "Processing mosaics | " + " ".join(
                        f"{s[0]}:{shards[s]['idx']}" for s in splits
                    )
                    pbar.set_description(desc)
                except Exception as e:
                    logger.warning(f"Could not update progress bar description: {e}")

            except Exception as e:
                filepath = futures[future]
                logger.error(f"Error processing {filepath}: {e}")
                if raise_errors:
                    raise e
                traceback.print_exc()

    # Finalize any remaining open writer per split
    logger.info("Finalizing shards...")
    for split in splits:
        w = shards[split]["writer"]
        if w is not None:
            try:
                w.finalize()
            except Exception as e:
                logger.error(f"Error finalizing shard for split {split}: {e}")
                if raise_errors:
                    raise e
    for split in splits:
        logger.info(f"Split '{split}': total samples: {total_samples[split]}.")

    # Rename shard files to include total number of shards
    logger.info("Renaming shard files...")
    for split in splits:
        n_shards = shards[split]["idx"] + (
            1 if shards[split]["writer"] is not None else 0
        )
        split_dir = out_parent / f"{split}.arrow"
        for idx in range(n_shards):
            old_path = split_dir / f"data-{idx:05d}.arrow"
            new_path = split_dir / f"data-{idx:05d}-of-{n_shards:05d}.arrow"
            old_path.rename(new_path)

    logger.info(f"Dataset saved to {out_parent}")

    # Create dataset_info.json with metadata
    logger.info("Creating dataset info files...")
    for split in tqdm(splits):
        au.create_dataset_info(
            out_parent / f"{split}.arrow",
            out_parent.name,
            split=split,
            write=True,
        )
        au.create_state_json(out_parent / f"{split}.arrow")

    logger.info("Done!")

    return out_parent


@au.clean_bkp_decorator(lookup=paths.MICROMAP_SUBSETS_ARROW)
def make_contexts(
    dset,
    dset_lookup=paths.MICROMAP_SUBSETS_ARROW,
    debug=False,
    catalog=paths.LOTSS_DR3_CAT,
    mosaic_dir=paths.MOSAIC_DIR_DR3,
    img_size=None,
    f_ctxt_size=None,
    f_downscale=None,
    max_workers=64,
    splits=["train", "val", "test"],
    raise_errors=False,
):
    logger.divider()
    logger.setLevel("DEBUG" if debug else "INFO")
    logger.info(f"Creating micromaps contexts from dataset {dset}\n")
    logger.info(
        f"Working with {f_ctxt_size=}, {f_downscale=}, {img_size=}, {splits=}\n"
    )

    # Get dataset parent directory
    maps_dset_path = parse_dset_path(dset, lookup=dset_lookup)
    encs_dset_path = maps_to_encoding_path(maps_dset_path)
    out_parent = encs_dset_path
    if debug:
        # Replace output parent
        out_parent = encs_dset_path.with_name(
            encs_dset_path.name.replace("micromap_encodings", "contexts_debug")
        )
        logger.debug(f"Debug mode: output directory will be created as {out_parent}.")
        if out_parent.exists():
            logger.debug(
                f"Recursively removing existing directory: \n\t{out_parent}.\n"
            )
            shutil.rmtree(out_parent)
        out_parent.mkdir()
    # Raise error if not existing
    if not out_parent.exists():
        raise FileNotFoundError(
            f"Input dataset directory {out_parent} does not exist. Please check the dataset path."
        )
    log_dir = out_parent / "logs"
    log_dir.mkdir(exist_ok=True)
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = (
        log_dir / f"{'make_contexts' + ('' if not debug else '_debug')}_{now}.log"
    )
    add_file_handler(logger, str(log_file))

    logger.info(f"Parallel processing with {max_workers} workers.")
    logger.debug(
        "Debug mode enabled. Will process only one mosaic from each split, and only two images from each mosaic."
    )

    # Get image size from dataset name
    # (used when extracting cutouts from big context array)
    if img_size is None:
        try:

            img_size = int(dset.split("-")[-1].replace("px", ""))
            logger.info(f"Identified image size from dataset name: {img_size} px.")
        except (IndexError, ValueError, AttributeError) as e:
            logger.error(
                f"Cannot infer image size from dataset name <{dset}>. Please provide it explicitly."
            )
            raise e
    else:
        logger.info(f"Using provided image size: {img_size} px.")

    # Load catalog
    logger.info(f"Loading catalog from: \n\t{catalog}.\n")
    cat = load_lotss_catalog(
        catalog,
        # Keep only relevant columns for faster processing
        select_cols=["RA", "DEC", "Total_flux", "Peak_flux", "Maj"],
    )
    # Sort by DEC for faster processing
    logger.info(f"Sorting catalog by DEC...")
    cat.sort_values(by="DEC", inplace=True)

    # Assert splits is a list of valid strings
    assert isinstance(splits, list), f"Expected list for splits, got {type(splits)}"
    valid_splits = {"train", "val", "test"}
    for split in splits:
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split name: {split}. Must be one of {valid_splits}."
            )

    # For every split, extract the keys to get the mosaic IDs
    logger.info("Extracting mosaic IDs from dataset keys...")
    mosaic_ids = set()
    for split in tqdm(splits):
        ds = datasets.load_from_disk(str(encs_dset_path / f"{split}.arrow"))
        unique_ids = (
            ds.with_format("pandas")["__key__"].str.split("-", n=1).str.get(0).unique()
        )
        mosaic_ids.update(unique_ids.copy().tolist())
        ds.cleanup_cache_files()
        del ds
    mosaic_ids = np.array(sorted(list(mosaic_ids))).copy()

    if debug:
        mosaic_ids = mosaic_ids[:3]
        logger.debug(
            f"Debug mode: limiting to first 3 mosaic IDs: {mosaic_ids.tolist()}."
        )

    for split in splits:
        logger.separator(f"Processing split: {split}")

        encs_dset = datasets.load_from_disk(
            str(encs_dset_path / f"{split}.arrow")
        ).sort("__key__")

        enc_ids = (
            encs_dset.with_format("pandas")["__key__"].str.split("-", n=1).str.get(0)
        )

        # Create partial with all fixed arguments
        process_fn = partial(
            afc.add_context_process_mosaic,
            enc_ids=enc_ids,
            encs_dset=encs_dset,
            cat=cat,
            mosaic_dir=mosaic_dir,
            img_size=img_size,
            f_ctxt_size=f_ctxt_size,
            f_downscale=f_downscale,
            debug=debug,
        )

        # Parallelize processing of mosaic IDs
        logger.info(f"Processing {len(mosaic_ids)} mosaic IDs...")
        result_dsets = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_fn, mosaic): mosaic for mosaic in mosaic_ids
            }
            for future in tqdm(
                as_completed(futures),
                desc="Processing Mosaic IDs",
                total=len(mosaic_ids),
                dynamic_ncols=True,
                smoothing=0.1,
                position=0,
                colour="green",
            ):
                mosaic = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        result_dsets.append(result)
                except Exception as e:
                    logger.error(f"Error processing mosaic {mosaic}: {e}")
                    traceback.print_exc()
                    if debug or raise_errors:
                        raise e

        # Combine all processed datasets
        logger.info(f"Combining results for split {split}...")

        combined_dataset = datasets.concatenate_datasets(result_dsets)

        # Save dataset
        logger.info(f"Saving context dataset for split {split} to disk...")
        tmp_path = out_parent / f"{split}-tmp.arrow"
        combined_dataset.save_to_disk(
            str(tmp_path),
            num_shards=(
                2 ** int(np.log2(len(encs_dset.cache_files))) if not debug else 1
            ),
            # Careful with num_proc: It can clog disk bandwidth and it can
            # flood RAM
            num_proc=16 if not debug else 4,
        )

        # === CLEANUP ===
        logger.info("Cleaning up references...")

        del enc_ids
        del result
        del futures
        del future
        del result_dsets
        combined_dataset.cleanup_cache_files()
        del combined_dataset
        encs_dset.cleanup_cache_files()
        del encs_dset

        # Garbage collection
        for i in range(3):
            collected = gc.collect()
            logger.debug(f"GC pass {i+1}: {collected} objects collected")

        final_path = out_parent / f"{split}.arrow"
        bkp_path = out_parent / f"{split}-bkp.arrow"
        try:
            if bkp_path.exists():
                logger.warning(
                    f"Backup file from previous run found: {bkp_path}. Removing it to avoid confusion."
                )
                shutil.rmtree(bkp_path)
            if final_path.exists():
                final_path.rename(bkp_path)
                tmp_path.rename(final_path)
                shutil.rmtree(bkp_path)
            else:
                tmp_path.rename(final_path)

        except Exception as e:
            logger.error(
                f"Error during finalizing dataset files for split {split}:\n"
                f"{traceback.format_exc()}\n"
                f"\nPlease check the output directory {out_parent} for any temporary or backup files and resolve manually if needed."
            )

    logger.info("All done!")


def make_micromap_catalogs(
    dset: str,
    debug=False,
    splits=["train", "val", "test"],
):
    logger.info(f"Creating catalogs for dataset: {dset}")

    if debug:
        logger.setLevel("DEBUG")
        logger.debug("Debug mode enabled.")

    # Navigate to original dataset if encodings
    is_encodings = "encodings" in dset
    if is_encodings:
        logger.info("Reading WCSs from original dataset. Will later copy to encodings.")
        dset = dset.replace("micromap-encodings", "micromaps")

    # Make catalog output folder
    dset_path = parse_dset_path(dset, lookup=paths.MICROMAP_SUBSETS_ARROW)
    logger.info(f"Dataset path resolved to:\n\t{dset_path}")
    out_folder = dset_path / "catalogs"
    out_folder.mkdir(exist_ok=True, parents=True)

    # Retrieve image size from dset name
    img_size = int(dset.rsplit("-", 1)[-1])
    logger.info(f"Using image size: {img_size}")

    # Load LoTSS catalog
    logger.info("Loading LoTSS catalog...")
    cat = load_lotss_catalog()

    # define file for getting catalog for a given sample.
    # Will be mapped onto the dataset.
    def get_sub_catalog(batch):

        records = []
        for header, key in zip(batch["wcs"], batch["__key__"]):
            wcs = WCS(header=header)
            sub_cat = filter_catalog_by_wcs(cat, wcs, axes=(img_size,) * 2).copy()
            sub_cat["cutout_key"] = [
                key,
            ] * len(sub_cat)

            # Empty dataframe with same columns as cat if no sources
            if len(sub_cat) == 0:
                sub_cat = pd.DataFrame(
                    columns=cat.columns.tolist()
                    + [
                        "cutout_key",
                    ]
                )
            records.append(sub_cat.to_dict("records"))
        return {"sub_catalog": records}

    for split in splits:
        logger.separator(f"Processing split: {split}")
        logger.info("Loading dataset...")
        dataset = MicromapDatasetHF(dset, split=split, output_tuple=("__key__", "wcs"))

        if debug:
            logger.debug("Debug mode: selecting only first 10 samples")
            dataset.dataset = dataset.dataset.select(range(10))

        # Apply function with batched processing and parallelization
        result_dataset = dataset.dataset.map(
            get_sub_catalog,
            batched=True,
            batch_size=(32 if not debug else 10),
            num_proc=(64 if not debug else 1),
            remove_columns=dataset.dataset.column_names,
            desc=f"Extracting catalogs ({split})",
        )

        # Combine the outputs
        logger.info("Concatenating sub-catalogs...")
        combined_records = [
            r for records in result_dataset["sub_catalog"] for r in records
        ]
        combined_catalog = pd.DataFrame.from_records(
            combined_records, coerce_float=True
        ).astype({"cutout_key": "string"})

        # Save catalog
        logger.info("Saving catalog...")
        combined_catalog.to_parquet(
            out_folder / f"{split}.parquet",
            index=False,
        )

        if is_encodings:
            logger.info("Copying catalog to encodings dataset folder...")
            enc_out_folder = (
                out_folder.parent.with_name(
                    out_folder.parent.name.replace("micromaps", "micromap_encodings")
                )
                / out_folder.name
            )

            enc_out_folder.mkdir(exist_ok=True, parents=False)
            combined_catalog.to_parquet(
                enc_out_folder / f"{split}.parquet",
                index=False,
            )

        logger.info(f"Done with split: {split}")

    logger.info("All done!")


if __name__ == "__main__":

    cpu_count = os.cpu_count()
    dset_lookup = paths.MICROMAP_SUBSETS_ARROW_HOPPER

    # Use this to create micromaps from the mosaics.
    # -------------------------------------------------------------
    if False:
        create_arrow_dataset_parallel(
            micromap_size=1024,
            spacing=1,
            prefix="micromaps-DR3-opt",
            override=True,
            debug=True,
            keys_from=None,
            max_workers=16,
            max_shard_size_gb=1.0,
            raise_errors=True,
        )

    # Use this to create encodings dataset from micromaps.
    # -------------------------------------------------------------
    if False:
        make_encodings_dset(
            dset="micromaps-DR3-opt-1024",
            dset_lookup=dset_lookup,
            vae_model="VQ-VAE-256-DR3opt-FT",
            vae_checkpoint="best",
            batch_size=16,
            debug=False,
            override=True,
            compile_model=False,
            splits=["train", "val", "test"],
            device="cuda:0",
        )

    # Use this to add catalogs to a dataset.
    # -------------------------------------------------------------
    if False:
        make_micromap_catalogs(
            "micromaps-DR3-opt-1024", debug=False, splits=["train", "val", "test"]
        )

    # Use this to add contexts to micromap encodings.
    # -------------------------------------------------------------
    if True:
        dset = "micromaps-DR3-opt-1024"
        make_contexts(
            dset=dset,
            dset_lookup=dset_lookup,
            debug=False,
            f_ctxt_size=2,
            f_downscale=4,
            max_workers=16,
            raise_errors=True,
            img_size=1024,
            splits=["train", "val", "test"],
        )

    # Use this to import context from a previous dataset.
    # Shards and entries must match by __key__
    # -------------------------------------------------------------
    if False:
        import_columns_parallel(
            src_dset="micromap_encodings-DR3-opt-1024px-spacing=1_bkp",
            target_dset="micromap_encodings-DR3-opt-1024px-spacing=1_bkp",
            dset_lookup=paths.MICROMAP_SUBSETS_ARROW_HOPPER,
            columns=["context_downscaled_d=4_f=2.npy"],
            splits=["train", "val", "test"],
            require_key_match=True,
            max_workers=8,
        )
