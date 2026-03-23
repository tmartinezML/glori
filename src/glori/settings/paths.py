import urllib.request
from pathlib import Path
from indexed import IndexedOrderedDict

from glori.infra.logging import show_dl_progress


# Base directories for code base & storage
BASE_PARENT = Path(__file__).parent.parent.parent.parent

# CHANGE THIS IF DESIRED:
STORAGE_PARENT = Path("/hs/fs08/data/group-brueggen/tmartinez")
STORAE_PARENT_HOPPER = Path("/storage/tmartinez")

# Three main storage folders.
MODEL_PARENT = STORAGE_PARENT / "model_results"
ANALYSIS_PARENT = STORAGE_PARENT / "analysis_results"
IMG_DATA_PARENT = STORAGE_PARENT / "image_data"
IMG_DATA_PARENT_HOPPER = STORAE_PARENT_HOPPER / "image_data"

# Cache directory
CACHE_DIR = STORAGE_PARENT / ".cache"
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir()

# Model configuration presets
CONFIG_PARENT = BASE_PARENT / "configs/model_presets"
MODEL_CONFIGS = IndexedOrderedDict({f.stem: f for f in CONFIG_PARENT.glob("*.json")})

# Folders for different kinds of image data
LOFAR_DATA_PARENT = IMG_DATA_PARENT / "LOFAR"
LOFAR_DATA_PARENT_HOPPER = IMG_DATA_PARENT_HOPPER / "LOFAR"
FIRST_DATA_PARENT = IMG_DATA_PARENT / "FIRST"
for f in [LOFAR_DATA_PARENT, FIRST_DATA_PARENT]:
    if not f.exists():
        f.mkdir()
SKY_MAP_PARENT = STORAGE_PARENT / "sky_maps"

# Pretrained models
PRETRAINED_PARENT = MODEL_PARENT / "pretrained"
if not PRETRAINED_PARENT.exists():
    PRETRAINED_PARENT.mkdir()

# Train data subsets
LOFAR_SUBSETS = IndexedOrderedDict(
    {
        k: LOFAR_DATA_PARENT / f"{v}"
        for k, v in {
            "model-prototypes": "model_prototypes/model_prototypes",
            "prototypes": "subsets/LOFAR_prototypes.hdf5",
            "200p": "subsets/200p-SNR5-unclipped.hdf5",
            "0-clip": "subsets/0-clip.hdf5",
        }.items()
    }
)

# Micromap datasets
MICROMAP_SUBSETS = IndexedOrderedDict(
    {
        k: LOFAR_DATA_PARENT / f"micromaps/{v}"
        for k, v in {
            "micromap-encodings-DR3-opt-1024": "micromap_encodings-DR3-opt-1024px-spacing=1",
            "micromaps-DR3-opt-1024": "micromaps-DR3-opt-1024px-spacing=1",
            "micromaps-DR3-1024": "micromaps-DR3-1024px-spacing=1",
            "micromap-encodings-DR3-1024": "micromap_encodings-DR3-1024px-spacing=1",
            "micromap-encodings-DR3-512": "micromap_encodings-DR3-512px-spacing=1",
            "micromaps-DR3-512": "micromaps-DR3-512px-spacing=1",
            "micromap-encodings-DR3-256": "micromap_encodings-DR3-256px-spacing=1",
            "micromaps-DR3-256": "micromaps-DR3-256px-spacing=1",
            "micromap-encodings-256": "micromap_encodings-256px-spacing=1",
            "micromaps-512": "micromaps-512px-spacing=1",
            "micromaps-256": "micromaps-256px-spacing=1",
        }.items()
    }
)
MICROMAP_SUBSETS_ARROW = {
    k: v.parent.with_name("micromaps-arrow") / v.name
    for k, v in MICROMAP_SUBSETS.items()
}
MICROMAP_SUBSETS_ARROW_HOPPER = {
    k: LOFAR_DATA_PARENT_HOPPER / "micromaps-arrow" / v.name
    for k, v in MICROMAP_SUBSETS.items()
}
MICROMAP_SUBSETS_ARROW_BABBAGE = {
    k: Path(
        str(v)
        .replace("fs08", "babbage")
        .replace("image_data", "diffusion/image_data_local")
    )
    for k, v in MICROMAP_SUBSETS_ARROW.items()
}
# Paths for training data processing
MOSAIC_DIR_DR2 = Path(
    "/hs/fs05/data/AG_Brueggen/nicolasbp/RadioGalaxyImage/data/mosaics_public"
)
MOSAIC_DIR_DR3 = Path("/hs/babbage/data/group-brueggen/nbaron/lotss_dr3/mosaics/")
MODEL_DIR_DR2 = LOFAR_DATA_PARENT / "pointings"
CUTOUTS_DIR = LOFAR_DATA_PARENT / "cutouts"
MICROMAP_DIR = LOFAR_DATA_PARENT / "micromaps"
MICROMAP_DIR_ARROW = LOFAR_DATA_PARENT / "micromaps-arrow"
LOFAR_RES_CAT = LOFAR_DATA_PARENT / "6-LoTSS_DR2-public-resolved_sources.csv"
LOTSS_DR2_CAT = LOFAR_DATA_PARENT / "LoTSS_DR2_v110_masked.srl.fits"
LOTSS_DR3_CAT = LOFAR_DATA_PARENT / "LoTSS_DR3_v0.5.srl.parquet"

# Paths for map simulation files
MAP_SHELL_SCRIPTS = BASE_PARENT / "src/maps/shell_scripts"
MAP_DEFAULTS = BASE_PARENT / "src/maps/default_files"


def cast_to_Path(path):
    """
    Cast a string object to a Path object. If the input is already a Path object,
    return it as is. If not Path or str, raise a TypeError.

    Parameters
    ----------
    path : str or Path
        The path to be cast to a Path object.

    Returns
    -------
    Path
        The path as a Path object.

    Raises
    ------
    TypeError
        If the input is not a Path or a string.
    """
    match path:
        case Path():
            return path
        case str():
            return Path(path)
        case _:
            raise TypeError(f"Expected Path or str, got {type(path)}")


if __name__ == "__main__":

    print("Base directories for code base & storage")
    print(f"\tBASE_PARENT: {BASE_PARENT}")
    print(f"\tSTORAGE_PARENT: {STORAGE_PARENT}")

    print("\nThree main storage folders.")
    print(f"\tMODEL_PARENT: {MODEL_PARENT}")
    print(f"\tANALYSIS_PARENT: {ANALYSIS_PARENT}")
    print(f"\tIMG_DATA_PARENT: {IMG_DATA_PARENT}")

    print("\nFolders for different kinds of image data")
    print(f"\tLOFAR_DATA_PARENT: {LOFAR_DATA_PARENT}")
    print(f"\tFIRST_DATA_PARENT: {FIRST_DATA_PARENT}")

    print("\nTrain data subsets")
    for k, v in LOFAR_SUBSETS.items():
        print(f"\t{k}: {v}")
