"""
Build the large-map (SWIIT) noise calibration from 2048 px maps of DR3 fields at
several catalog depths T (maps.parquet: one row per map with target, field,
noise_uJy, texture_ok_frac, recovered, flux; from the eval_5_swiit calibration
run). Texture bands are taken from the single-image calibration, whose real DR3
reference is per 512 px cutout, the tile size the maps are checked in.

  python scripts/ldm/build_noise_map_calibration.py \
      --maps /export/data/group-brueggen/kevin/eval_5_swiit/calib/maps.parquet \
      --single configs/noise_calibration/rms-v1.json \
      --out configs/noise_calibration/rms-v1-swiit.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# The range is set by hand from the calibration: below ~50 uJy/beam (T=40) only
# 80 % of the 512 px tiles have a DR3-like texture (anisotropy), above 224 there
# is no calibration point.
VALID_NOISE = (50.0, 224.0)
CRIT = dict(
    max_rescale=1.3,  # generated map median vs. requested
    min_texture_frac=0.75,  # 512 px tiles with all texture metrics in the DR3 band
    min_bright_recovery=0.8,  # input peak >= 10 x noise found within 12"
    min_flux_sources=10,  # bright matched sources for the per-map flux factor
    flux_snr=20.0,  # input peak >= flux_snr x T used for the flux factor
)
# DR3 fields a map's real sources are taken from: native depth within the
# calibrated fields (36.7-87.2 uJy/beam), and no empty 64 x 64 cell block (6.4')
# in the map and its context margin (survey gaps or edge)
FIELDS = dict(max_native_uJy=90.0, block_cells=64, min_sources_per_block=1)
MAP_PX = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True)
    ap.add_argument("--single", required=True, help="single-image calibration (method, texture bands)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    maps = pd.read_parquet(args.maps)
    single = json.loads(Path(args.single).read_text())
    g = maps.groupby("target")
    tab = pd.DataFrame(
        dict(
            noise_p16=g.noise_uJy.quantile(0.16),
            noise_p50=g.noise_uJy.median(),
            noise_p84=g.noise_uJy.quantile(0.84),
            monotone=maps.sort_values("target").groupby("field").noise_uJy.apply(
                lambda s: bool(np.all(np.diff(s) > 0))).all(),
            texture_ok_frac=g.texture_ok_frac.mean(),
            maps_texture_ok=g.texture_ok_frac.apply(lambda s: float(np.mean(s >= CRIT["min_texture_frac"]))),
            recovered=g.recovered.median(),
            flux=g.flux.median(),
            flux_p16=g.flux.quantile(0.16),
            flux_p84=g.flux.quantile(0.84),
            n_maps=g.size(),
        )
    ).reset_index()
    if not tab.noise_p50.is_monotonic_increasing:
        raise SystemExit("map noise is not monotone in T, calibration unusable")
    t_lo, t_hi = (
        float(np.exp(np.interp(np.log(v), np.log(tab.noise_p50), np.log(tab.target)))) for v in VALID_NOISE
    )
    out = dict(
        description=(
            "Noise calibration for large maps (SWIIT, rms-v1): catalog depth T -> median of the "
            f"PyBDSF rms map of {MAP_PX} px maps, across {maps.field.nunique()} DR3 fields"
        ),
        map_px_calibrated=MAP_PX,
        method=single["method"],
        criteria=CRIT,
        fields=dict(FIELDS, native_uJy_calibrated=[round(float(v), 1) for v in
                                                   (maps.native_uJy.min(), maps.native_uJy.max())]),
        valid_range=dict(target=[round(t_lo, 2), round(t_hi, 2)],
                         noise_min_uJy=VALID_NOISE[0], noise_max_uJy=VALID_NOISE[1]),
        calibration=dict(target=tab.target.astype(float).tolist(),
                         **{k: tab[k].round(3).tolist() for k in ("noise_p16", "noise_p50", "noise_p84")}),
        # fallback when a map has too few bright sources for its own flux factor
        flux_correction=dict(target=tab.target.astype(float).tolist(), factor=tab.flux.round(4).tolist()),
        texture_bands=single["texture_bands"],
        table=tab.round(4).to_dict(orient="list"),
    )
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(tab.round(3).to_string(index=False))
    print(f"valid range: noise {VALID_NOISE[0]}-{VALID_NOISE[1]} uJy/beam = T {t_lo:.1f}-{t_hi:.1f}")


if __name__ == "__main__":
    main()
