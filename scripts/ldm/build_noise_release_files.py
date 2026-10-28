"""
Build the two LoTSS-DR3 derived data files of the noise-band tools, released in
the Hugging Face repository under noise/ (see glori.hub):

  dr3_snr_pool.npz  (SNR, total/peak, Maj) of DR3 sources from typical-depth
                    regions (Isl_rms 40-140 uJy/beam), drawn for the synthetic
                    faint sources
  dr3_fields.csv    field centres drawn uniformly on the sky (Dec > 0) and kept
                    where DR3 covers a single image (1024 px context), with the
                    native depth and whether a 2048 px map with its margin is
                    covered as well

  python scripts/ldm/build_noise_release_files.py --out <release dir>/noise
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from glori.inference.noise_map_sampler import PAD
from glori.inference.noise_sampler import CTX_PX, CTX_SIZE, field_context, load_dr3_catalog

N_POOL = 200_000
N_FIELDS = 12_000  # uniform draws; outside the footprint they are dropped
BLOCK = 64  # coverage: no empty 64 x 64 cell (6.4') block
MAP_CELLS = 2048 // CTX_PX + 2 * PAD


def empty_block(ctx):
    src = ctx[0] > 0
    nb = src.shape[0] // BLOCK
    return bool((src[: nb * BLOCK, : nb * BLOCK].reshape(nb, BLOCK, nb, BLOCK).sum(axis=(1, 3)) == 0).any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cat = load_dr3_catalog()

    # source properties for the synthetic sources
    ok = (cat["Isl_rms"] >= 0.040) & (cat["Isl_rms"] < 0.140) & (cat["Peak_flux"] > 0)
    i = rng.choice(np.flatnonzero(ok), size=N_POOL, replace=False)
    pool = np.c_[cat["Peak_flux"][i] / cat["Isl_rms"][i], cat["Total_flux"][i] / cat["Peak_flux"][i], cat["Maj"][i]]
    np.savez(out / "dr3_snr_pool.npz", pool=pool.astype(np.float32))
    print(f"dr3_snr_pool.npz: {len(pool)} sources")

    # candidate fields: the map context (2048 px + margin) contains the single-image context
    rows = []
    lo = (MAP_CELLS - CTX_SIZE) // 2
    ras = rng.uniform(0, 360, N_FIELDS)
    decs = np.degrees(np.arcsin(rng.uniform(0, 1, N_FIELDS)))
    for k, (ra, dec) in enumerate(zip(ras, decs)):
        ctx = field_context(cat, ra, dec, MAP_CELLS)
        src = ctx[0] > 0
        if not src.any():
            continue
        rows.append(dict(
            ra=round(float(ra), 5), dec=round(float(dec), 5),
            native_uJy=round(float(np.median(ctx[4][src] / ((ctx[0][src] + 1) / 2))), 2),
            single_ok=not empty_block(ctx[:, lo:lo + CTX_SIZE, lo:lo + CTX_SIZE]),
            map_ok=not empty_block(ctx),
        ))
        if (k + 1) % 1000 == 0:
            print(f"{k + 1}/{N_FIELDS} fields", flush=True)
    f = pd.DataFrame(rows)
    f = f[f.single_ok].reset_index(drop=True)
    f.to_csv(out / "dr3_fields.csv", index=False)
    print(f"dr3_fields.csv: {len(f)} of {N_FIELDS} uniform draws with single-image coverage, {int(f.map_ok.sum())} with map coverage, "
          f"{int((f.map_ok & (f.native_uJy <= 90)).sum())} usable for maps (native <= 90 uJy/beam)")


if __name__ == "__main__":
    main()
