from pathlib import Path
from pprint import pformat
import json


def parse_preset(preset: Path | str, preset_lookup: dict[str, Path]) -> Path:
    match preset:
        case Path():
            out = preset

        case str():
            preset_path = preset_lookup.get(preset)
            if preset_path is None:
                raise ValueError(
                    f"Preset {preset} not found in {preset_lookup}. Available presets:\n{pformat(list(preset_lookup.keys()))}"
                )
            if not preset_path.exists():
                raise ValueError(
                    f"Preset {preset} not found in {preset_lookup} or as a file path."
                )
            out = preset_path
        case _:
            raise ValueError(
                f"Invalid preset identifier of type {type(preset)}: {preset}"
            )

    if not out.exists():
        raise ValueError(f"Preset file {out} does not exist.")

    return out
