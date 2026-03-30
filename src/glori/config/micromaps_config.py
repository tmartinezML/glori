import json
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from torchvision.transforms.v2._transform import Transform

import glori.data.trf.post as post
import glori.settings.paths as paths
import glori.data.trf.transforms as T
from glori.config.load import parse_preset
from glori.data.trf.functional import zero_center


@dataclass
class MicromapsConfig:
    dataset: str
    dataset_lookup: str | Path = "MICROMAP_SUBSETS_ARROW"
    split: str = "train"
    img_context: str | None = None
    catalog_context: str | None = None
    data_transform: str | None = "HFDatasetTransformRaw"
    ctxt_transform: dict[str, str] | None = None
    ctxt_scalers: dict[str, str] | None = None
    random_crop: int | None = None
    center_crop: int | None = None
    crop_fctxt: int | list[int] = 1
    crop_sctxt: int | list[int] = 1
    blank_center: dict[str, float] | None = None
    scaler: str | None = None
    max_beam_arcsec: float | None = 6.0
    weights_file: str | None = None
    weights_fn: str | None = None
    missing_is_error: bool = True

    @classmethod
    def from_preset(cls, preset: str | Path) -> "MicromapsConfig":
        config_file = parse_preset(preset, paths.DATASET_CONFIGS)
        return cls(**json.loads(config_file.read_text()))

    def __post_init__(self):
        if self.random_crop is not None and self.center_crop is not None:
            raise ValueError("Cannot specify both random_crop and center_crop.")


def resolve_micromap_kwargs(config: MicromapsConfig, **override) -> dict:
    """Create a MicromapsConfig from a dictionary, allowing for missing keys"""
    # Extract dictionary
    config_dict = config.__dict__.copy()
    config_dict.update(override)

    # Parse dataset_lookup if it's a string
    if (dataset_lookup := config_dict.get("dataset_lookup")) is not None:
        config_dict["dataset_lookup"] = getattr(paths, dataset_lookup)

    # Add context to output tuple if specified
    config_dict["output_tuple"] = ("npy",)
    if (img_ctxt := config_dict.pop("img_context", None)) is not None:
        config_dict["output_tuple"] += (img_ctxt,)

    if (catalog_ctxt := config_dict.pop("catalog_context", None)) is not None:
        config_dict["output_tuple"] += (catalog_ctxt,)

    # Parse data transform if specified
    if (data_trf_inp := config_dict.pop("data_transform", None)) is not None:
        data_trf_str, data_trf_kwargs = (
            (data_trf_inp, {}) if isinstance(data_trf_inp, str) else data_trf_inp
        )
        config_dict["data_transform"] = getattr(T, data_trf_str)(**data_trf_kwargs)

    # Parse context transforms if specified
    ctxt_transform = {}
    if (ctxt_trf_inp := config_dict.get("ctxt_transform")) is not None:
        scale_fns = {}
        # Extract context scalers if passed
        if (ctxt_scalers := config_dict.pop("ctxt_scalers", None)) is not None:
            scale_fns = {
                k: (T.make_catalog_context_value_scale(v))
                for k, v in ctxt_scalers.items()
            }
        # Initialize transforms for each context key, using kwargs if passed
        for k, v in ctxt_trf_inp.items():
            trf_str, trf_kwargs = (v, {}) if isinstance(v, str) else v
            trf_kwargs["scale_fn"] = scale_fns.get(k)
            ctxt_transform[k] = getattr(T, trf_str)(**trf_kwargs)
        config_dict["ctxt_transform"] = ctxt_transform
    config_dict.pop(
        "ctxt_scalers", None
    )  # Remove ctxt_scalers from config dict since it's now parsed

    # Parse post transforms if specified
    post_transform_list = []
    if (
        config_dict.get("random_crop") is not None
        and config_dict.get("center_crop") is not None
    ):
        raise ValueError("Cannot specify both random_crop and center_crop.")

    if (crop_size := config_dict.pop("random_crop", None)) is not None:
        post_transform_list.append(
            post.make_dependent_random_crop(
                crop_size=crop_size,
                f=config_dict.pop("crop_fctxt", 1),
                s=config_dict.pop("crop_sctxt", 1),
                keys=("npy", *list(ctxt_transform.keys())),
            )
        )
    elif (crop_size := config_dict.pop("center_crop", None)) is not None:
        post_transform_list.append(
            post.make_dependent_center_crop(
                crop_size=crop_size,
                f=config_dict.pop("crop_fctxt", 1),
                s=config_dict.pop("crop_sctxt", 1),
                keys=("npy", *list(ctxt_transform.keys())),
            )
        )
    config_dict.pop("crop_fctxt", None)
    config_dict.pop("crop_sctxt", None)
    if (blank_center := config_dict.pop("blank_center", None)) is not None:
        post_transform_list.append(
            post.make_post(
                {k: partial(zero_center, f_center=v) for k, v in blank_center.items()},
            )
        )
    config_dict["post_transform"] = post.compose(*post_transform_list)

    # Turn weights function from string to callable if specified
    if (weights_fn_str := config_dict.get("weights_fn")) is not None:
        config_dict["weights_fn"] = eval(weights_fn_str)

    return config_dict
