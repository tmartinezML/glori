"""
Build the calibration file of glori.inference.noise_sampler from a calibration
run (generated images at catalog depths T, measured with PyBDSF) and real LoTSS
DR3 test cutouts measured the same way.

  python scripts/ldm/build_noise_calibration.py \
      --generated metrics_calib.parquet [metrics_calib_hi.parquet ...] \
      --reference metrics_reference_3000.parquet

Per depth T it records the generated noise (median, 16/84 %) and the residual
flux factor after rescaling to the requested noise, and decides whether T is
reliable:
  - noise rises to the next level for >= 95 % of the cutouts (same catalog cutout)
  - noise texture inside the band of real cutouts with similar noise for
    >= min(90 %, pass rate of real cutouts - 5 %) of the images
  - bright input sources (input peak >= 10 T) recovered (>= 80 %) in >= 95 % of images
  - after rescaling, median measured / input peak flux (input peak >= 20 T)
    within 1 +- 0.10
The valid range is the longest contiguous block of reliable depths; requests
are limited to the median noise at its ends.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from glori.inference.noise_sampler import NOISE_CALIBRATION, TEXTURE

WINDOW, MIN_REF = 0.25, 60  # real references within +-25 % noise, at least 60
CRIT = dict(monotone=0.95, realism=0.90, recovery=0.95, flux_tol=0.10,
            min_bright_recovery=0.8, max_rescale=1.3)
METHOD = dict(keep_native=8.0, drop_snr=3.0, density_a=1.97e5, density_b=-1.26, img_to_isl=1.07)


def ref_selection(ref, sigma):
    sel = ref[(ref.sigma_uJy >= sigma * (1 - WINDOW)) & (ref.sigma_uJy <= sigma * (1 + WINDOW))]
    if len(sel) < MIN_REF:
        sel = ref.iloc[np.argsort(np.abs(np.log(ref.sigma_uJy.values / sigma)))[:MIN_REF]]
    return sel


def ref_band(ref, sigma):
    sel = ref_selection(ref, sigma)
    return {m: (np.percentile(sel[m], 1), np.percentile(sel[m], 99)) for m in TEXTURE}


def texture_ok(row, band):
    return all(band[m][0] <= row[m] <= band[m][1] for m in TEXTURE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", nargs="+", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", default=str(NOISE_CALIBRATION))
    args = ap.parse_args()

    gen = pd.concat([pd.read_parquet(f) for f in args.generated], ignore_index=True)
    ref = pd.read_parquet(args.reference)
    ref = ref[np.isfinite(ref.sigma_uJy)].sort_values("sigma_uJy").reset_index(drop=True)
    T = np.array(sorted(gen.target.unique()))
    ref_pass = np.mean([texture_ok(r, ref_band(ref, r.sigma_uJy))
                        for _, r in ref.sample(600, random_state=0).iterrows()])

    # bright = input peak >= 10 T, flux factor from input peak >= 20 T (sources are dimmed
    # with the noise, so input peak / measured noise would overstate their SNR at high T);
    # older runs only have rec10 and fluxratio
    rec_col = "rec10T" if "rec10T" in gen else "rec10"
    piv = gen.pivot_table(index="key", columns="target", values="sigma_uJy")
    rows = []
    for k, t in enumerate(T):
        g = gen[gen.target == t]
        q = np.percentile(g.sigma_uJy, [16, 50, 84])
        rows.append(dict(
            target=float(t), noise_p16=q[0], noise_p50=q[1], noise_p84=q[2],
            monotone_next=float(np.mean(piv[T[k + 1]] > piv[t])) if k + 1 < len(T) else np.nan,
            realism=float(np.mean([texture_ok(r, ref_band(ref, r.sigma_uJy)) for _, r in g.iterrows()])),
            recovery=float(np.mean(g[rec_col].fillna(1) >= CRIT["min_bright_recovery"])),
            # sources scale with the noise: flux ratio after rescaling to noise T
            flux_rescaled=float(np.nanmedian(g.flux20T if "flux20T" in g else g.fluxratio * t / g.sigma_uJy)),
            within_rescale=float(np.mean(np.abs(np.log(g.sigma_uJy / q[1])) <= np.log(CRIT["max_rescale"]))),
        ))
    tab = pd.DataFrame(rows)
    tab["reliable"] = (
        ((tab.monotone_next >= CRIT["monotone"]) | tab.monotone_next.isna())
        & (tab.realism >= min(CRIT["realism"], ref_pass - 0.05))
        & (tab.recovery >= CRIT["recovery"])
        & (np.abs(tab.flux_rescaled - 1) <= CRIT["flux_tol"])
        & np.r_[np.diff(tab.noise_p50) > 0, True]
    )
    best, cur = (0, -1), None
    for i, r in enumerate(tab.reliable):
        cur = (i if cur is None else cur) if r else None
        if r and i - cur > best[1] - best[0]:
            best = (cur, i)
    lo, hi = best
    if hi < lo:
        raise RuntimeError("no reliable depth in the calibration run")
    use = tab.iloc[lo:hi + 1]

    grid = np.geomspace(ref.sigma_uJy.quantile(0.002), ref.sigma_uJy.quantile(0.998), 30)
    bands = [ref_band(ref, s) for s in grid]
    calib = dict(
        description="Noise calibration of glori.inference.noise_sampler: catalog depth T -> "
                    "PyBDSF noise of the generated image (median of the rms map, uJy/beam).",
        method=METHOD,
        criteria=dict(CRIT, real_texture_pass=round(float(ref_pass), 4)),
        valid_range=dict(target=[float(T[lo]), float(T[hi])],
                         noise_min_uJy=round(float(tab.noise_p50[lo]), 1),
                         noise_max_uJy=round(float(tab.noise_p50[hi]), 1)),
        calibration=use[["target", "noise_p16", "noise_p50", "noise_p84"]].round(3).to_dict(orient="list"),
        flux_correction=dict(target=use.target.tolist(), factor=use.flux_rescaled.round(4).tolist()),
        texture_bands=dict(noise_uJy=grid.round(2).tolist(),
                           **{m: dict(p_lo=[b[m][0] for b in bands], p_hi=[b[m][1] for b in bands])
                              for m in TEXTURE}),
        n_generated_cutouts=int(gen.key.nunique()), n_reference_cutouts=int(len(ref)),
        sources=dict(generated=[str(f) for f in args.generated], reference=str(args.reference)),
        table=tab.round(4).to_dict(orient="list"),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calib, indent=1))
    print(f"real cutouts pass their noise-matched texture bands: {ref_pass:.1%}")
    print(tab.round(3).to_string(index=False))
    print("valid range:", calib["valid_range"], "->", out)


if __name__ == "__main__":
    main()
