import os
import gc
import sys
import json
import psutil
import shutil
import subprocess
import traceback
from pathlib import Path
from functools import wraps

from tqdm import tqdm
import numpy as np

import glori.settings.paths as paths
from glori.data.utils import parse_dset_path

import sys, io, traceback

import pyarrow as pa
import pyarrow.ipc as ipc


def create_state_json(output_dir):
    """Create minimal state.json for a manually created Arrow dataset."""
    output_dir = Path(output_dir)

    # Get all shard files
    shard_files = sorted(output_dir.glob("data-*.arrow"))

    state = {
        "_data_files": [{"filename": f.name} for f in shard_files],
        "_fingerprint": None,
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": "train",
    }

    with open(output_dir / "state.json", "w") as f:
        json.dump(state, f, indent=2)

    return state


def extract_column(
    dataset,
    split,
    column_name,
    output_file=None,
):
    """
    Extract a single column from Arrow shards without loading full data into memory.

    Returns a list of arrays (one per shard) or concatenates if output_file is given.
    """
    dataset_dir = (
        parse_dset_path(dataset, lookup=paths.MICROMAP_SUBSETS_ARROW) / f"{split}.arrow"
    )
    assert dataset_dir.exists(), f"Dataset directory {dataset_dir} does not exist."
    assert len(
        list(dataset_dir.glob("*.arrow"))
    ), f"No arrow files found in {dataset_dir}."
    shard_files = sorted(dataset_dir.glob("data-*.arrow"))

    column_data = []

    for shard_file in tqdm(shard_files, desc=f"Extracting {column_name}"):
        with pa.memory_map(str(shard_file), "r") as source:
            reader = ipc.open_stream(source)

            # Read only the specific column (avoids deserializing other columns)
            for batch in reader:
                # Get only the column you want
                col = batch.column(column_name)
                column_data.append(col.to_pylist())  # or .to_numpy() if numeric

    # Flatten if needed
    flattened = [item for sublist in column_data for item in sublist]

    if output_file:
        # Save as numpy or parque

        np.save(output_file, flattened)

    return np.array(flattened)


def create_dataset_info(
    output_dir,
    dataset_name,
    split="train",
    features=None,
    description="",
    license="",
    homepage="",
    write=True,
):
    """
    Create a valid dataset_info.json for HuggingFace datasets.

    Parameters
    ----------
    output_dir : Path
        Directory containing the arrow shards
    dataset_name : str
        Name of the dataset
    split : str
        Split name (e.g., "train", "test")
    features : dict, optional
        Feature schema. If None, will infer from first shard.
    description : str
        Dataset description
    license : str
        License (e.g., "CC-BY-4.0")
    homepage : str
        Dataset homepage URL
    write : bool
        Whether to write the JSON files to disk
    """

    output_dir = Path(output_dir)

    # Get all shard files
    shard_files = sorted(output_dir.glob("data-*.arrow"))

    if not shard_files:
        raise FileNotFoundError(f"No arrow shards found in {output_dir}")

    # Compute total size and shard lengths
    total_bytes = sum(f.stat().st_size for f in shard_files)
    shard_lengths = []
    total_examples = 0

    print(f"Processing {len(shard_files)} shards...")
    for shard_file in tqdm(shard_files, desc="Reading shard metadata"):
        # Read Arrow IPC Stream format (not File format!)
        with pa.memory_map(str(shard_file), "r") as source:
            reader = ipc.open_stream(source)  # Changed from open_file to open_stream

            # Read all batches and count rows
            total_rows = 0
            for batch in reader:
                total_rows += batch.num_rows

            shard_lengths.append(total_rows)
            total_examples += total_rows

    # Infer schema from first shard if not provided
    if features is None:
        print("Inferring schema from first shard...")
        with pa.memory_map(str(shard_files[0]), "r") as source:
            reader = ipc.open_stream(source)  # Changed from open_file to open_stream
            schema = reader.schema
            features = {}
            for field in schema:
                features[field.name] = _pyarrow_to_hf_feature(field.type)

        print(f"Inferred {len(features)} features:")
        for name in features.keys():
            print(f"  - {name}")

    # Create dataset_info
    dataset_info = {
        "builder_name": "arrow",
        "citation": "",
        "config_name": "default",
        "dataset_name": dataset_name,
        "dataset_size": total_bytes,
        "description": description,
        "download_checksums": {},
        "download_size": 0,
        "features": features,
        "homepage": homepage,
        "license": license,
        "size_in_bytes": total_bytes,
        "splits": {
            split: {
                "name": split,
                "num_bytes": total_bytes,
                "num_examples": total_examples,
                "shard_lengths": shard_lengths,
                "dataset_name": dataset_name,
            }
        },
        "version": {
            "version_str": "1.0.0",
            "major": 1,
            "minor": 0,
            "patch": 0,
        },
    }

    if write:
        print(f"\nWriting metadata to {output_dir}...")
        # Save dataset_info.json
        with open(output_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)

        # Also create state.json (required by HuggingFace)
        state = {
            "_data_files": [{"filename": f.name} for f in shard_files],
            "_fingerprint": None,
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": split,
        }

        with open(output_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"✅ Created dataset_info.json with {total_examples} total examples")
        print(f"✅ Created state.json with {len(shard_files)} shards")

    return dataset_info


def _pyarrow_to_hf_feature(pa_type):
    """Convert PyArrow type to HuggingFace feature schema."""
    import pyarrow as pa

    # Handle basic types
    if pa.types.is_string(pa_type):
        return {"dtype": "string", "_type": "Value"}
    elif pa.types.is_integer(pa_type):
        return {"dtype": str(pa_type), "_type": "Value"}
    elif pa.types.is_floating(pa_type):
        return {"dtype": str(pa_type), "_type": "Value"}
    elif pa.types.is_boolean(pa_type):
        return {"dtype": "bool", "_type": "Value"}

    # Handle lists (arrays)
    elif pa.types.is_list(pa_type):
        value_type = pa_type.value_type
        return {"feature": _pyarrow_to_hf_feature(value_type), "_type": "Sequence"}

    # Handle fixed-size lists (for known shapes)
    elif pa.types.is_fixed_size_list(pa_type):
        list_size = pa_type.list_size
        value_type = pa_type.value_type

        # If nested fixed-size list, infer shape
        if pa.types.is_fixed_size_list(value_type):
            # 2D array
            inner_size = value_type.list_size
            inner_type = value_type.value_type
            if pa.types.is_fixed_size_list(inner_type):
                # 3D array
                depth_size = inner_type.list_size
                dtype = str(inner_type.value_type)
                return {
                    "shape": [list_size, inner_size, depth_size],
                    "dtype": dtype.replace("double", "float64").replace(
                        "float", "float32"
                    ),
                    "_type": "Array3D",
                }
            else:
                dtype = str(inner_type)
                return {
                    "shape": [list_size, inner_size],
                    "dtype": dtype.replace("double", "float64").replace(
                        "float", "float32"
                    ),
                    "_type": "Array2D",
                }
        else:
            dtype = str(value_type)
            return {
                "shape": [list_size],
                "dtype": dtype.replace("double", "float64").replace("float", "float32"),
                "_type": "Array",
            }

    # Fallback
    return {"dtype": str(pa_type), "_type": "Value"}


class SpyOutput(io.StringIO):
    def write(self, data):
        # Print a traceback whenever something prints that looks like disassembly
        if "LOAD_" in data or "STORE_" in data or "BINARY_" in data:
            print("\n🔍 Disassembly-like output detected!\n")
            traceback.print_stack(limit=6)
        return super().write(data)


def clean_bkp_decorator(lookup=paths.MICROMAP_SUBSETS_ARROW):
    """
    Decorator that cleans up backup files after function execution.

    Args:
        lookup: Lookup dictionary for dataset paths

    Usage:
        @clean_bkp_decorator()
        def make_contexts(dset, ...):
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(dset, *args, **kwargs):
            try:
                # Execute the original function
                result = func(dset, *args, **kwargs)
                return result
            finally:
                # Always clean up backups, even if function raises
                try:
                    clean_bkp(dset, lookup=lookup)
                except Exception as e:
                    traceback.print_exc()

        return wrapper

    return decorator


def clean_bkp(dset, lookup=paths.MICROMAP_SUBSETS_ARROW):
    path = parse_dset_path(dset, lookup=lookup)
    for f in path.glob("*bkp*"):
        if f.is_dir():
            shutil.rmtree(f)
        elif f.is_file():
            f.unlink()
    return


def find_all_open_files(logger):
    """Find all Python objects with open file handles."""
    logger.info("Scanning for open file handles...")

    open_files = []
    for obj in gc.get_objects():
        try:
            # Check for file objects
            if hasattr(obj, "read") and hasattr(obj, "name"):
                if hasattr(obj, "closed") and not obj.closed:
                    open_files.append(
                        {
                            "type": type(obj).__name__,
                            "name": getattr(obj, "name", "<unknown>"),
                            "mode": getattr(obj, "mode", "<unknown>"),
                            "id": id(obj),
                        }
                    )

            # Check for Arrow datasets (memory-mapped files)
            elif hasattr(obj, "_data") and hasattr(obj._data, "files"):
                open_files.append(
                    {
                        "type": "ArrowDataset",
                        "name": f"Dataset with {len(obj._data.files)} files",
                        "files": obj._data.files,
                        "id": id(obj),
                    }
                )

            # Check for Arrow tables
            elif type(obj).__name__ in ["Table", "InMemoryTable", "MemoryMappedTable"]:
                if hasattr(obj, "files"):
                    open_files.append(
                        {
                            "type": type(obj).__name__,
                            "name": f"Arrow table with {len(obj.files) if hasattr(obj, 'files') else 0} files",
                            "id": id(obj),
                        }
                    )
        except Exception as e:
            # Silently skip objects that cause errors during inspection
            pass

    return open_files


def find_os_open_files(logger, path_filter=None):
    """Find open file descriptors at OS level."""
    logger.info("Checking OS-level file descriptors...")

    process = psutil.Process(os.getpid())
    open_files = process.open_files()

    if path_filter:
        open_files = [f for f in open_files if path_filter in f.path]

    return [{"fd": f.fd, "path": f.path} for f in open_files]


def close_all_file_handles(logger, path_filter=None):
    """Close all open Python file handles, optionally filtered by path."""
    logger.info("Attempting to close all open file handles...")

    closed_count = 0
    for obj in gc.get_objects():
        try:
            # Close regular file objects
            if (
                hasattr(obj, "close")
                and hasattr(obj, "name")
                and hasattr(obj, "closed")
            ):
                # Skip log file
                if str(obj.name).endswith(".log"):
                    continue

                if not obj.closed:
                    if path_filter is None or path_filter in str(obj.name):
                        obj.close()
                        closed_count += 1
                        logger.debug(f"Closed: {obj.name}")
        except Exception as e:
            logger.debug(f"Could not close object {obj.name}: {e}")

    logger.info(f"Closed {closed_count} file handles")
    return closed_count


# Add this right before your problematic shutil.rmtree() call:
def safe_remove_with_diagnostics(logger, path):
    """Remove directory with full diagnostics if it fails."""
    path = Path(path)

    # Step 1: Find what Python objects might be holding references
    logger.info("=== Python Object Analysis ===")
    python_files = find_all_open_files(logger)
    relevant_files = [f for f in python_files if str(path) in str(f.get("name", ""))]

    if relevant_files:
        logger.warning(
            f"Found {len(relevant_files)} Python objects referencing {path}:"
        )
        for f in relevant_files:
            logger.warning(f"  - {f['type']}: {f.get('name', 'unknown')}")

    # Step 2: Check OS-level file descriptors
    logger.info("=== OS-Level File Descriptors ===")
    os_files = find_os_open_files(logger, str(path))

    if os_files:
        logger.warning(f"Found {len(os_files)} OS-level open files in {path}:")
        for f in os_files:
            logger.warning(f"  - FD {f['fd']}: {f['path']}")

    # Step 3: Close all Python file handles related to this path
    # logger.info("=== Closing File Handles ===")
    # close_all_file_handles(logger, str(path))

    # Step 4: Force garbage collection multiple times
    logger.info("=== Garbage Collection ===")
    for i in range(3):
        collected = gc.collect()
        logger.info(f"GC pass {i+1}: collected {collected} objects")

    # Step 5: Wait for OS to release handles
    import time

    logger.info("Waiting for OS to release file handles...")
    time.sleep(2.0)

    # Step 6: Try to remove
    logger.info(f"=== Attempting to remove {path} ===")
    try:
        shutil.rmtree(path)
        logger.info(f"Successfully removed {path}")
        return True
    except OSError as e:
        logger.error(f"Failed to remove {path}: {e}")

        # Step 7: Final diagnostic - show what's still open
        logger.error("=== Final Diagnostic ===")
        result = subprocess.run(
            ["lsof", "+D", str(path)], capture_output=True, text=True
        )
        if result.stdout:
            logger.error(f"Still open:\n{result.stdout}")

        # Show Python objects still holding references
        remaining = find_all_open_files(logger)
        remaining_relevant = [
            f for f in remaining if str(path) in str(f.get("name", ""))
        ]
        if remaining_relevant:
            logger.error(f"Python objects still holding references:")
            for f in remaining_relevant:
                logger.error(f"  - {f}")

        return False
