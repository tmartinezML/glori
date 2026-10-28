"""
Generate large LOFAR-like test maps with a median noise in a band, e.g. for
testing source finders on survey-like areas.

  python scripts/generate_noise_maps.py --noise 60-80 --n 8 --size 2048 --out maps_60-80

The noise of a map is the median of its PyBDSF rms map (LoTSS settings); it
varies across the map, the rms map is delivered with it. Output in --out:
  maps/map_#####.fits      size x size px, 1.5" px, 6" beam, Jy/beam (NOISE in header)
  maps/map_#####_rms.fits  PyBDSF rms map of the map [Jy/beam]
  truth_catalog.csv        input sources (map_id, x_px, y_px, peak/total flux [Jy],
                           Maj, synthetic or real LoTSS DR3, detected by PyBDSF)
  pybdsf_catalog.csv       PyBDSF sources of each map
  tiles.csv                noise and texture check per 512 px tile
  summary.csv              one row per map, report.json for the whole run
Uses one GPU (about 3-4 min per 2048 px map when --batch maps are sampled
together) and --workers single-core PyBDSF processes at low priority.
Models and data files are downloaded from Hugging Face on first use (or taken
from $GLORI_PRETRAINED_DIR), source catalogs from the public LoTSS-DR3 cone search.
"""

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noise", required=True, help="band of the map median noise in uJy/beam, e.g. 60-80")
    ap.add_argument("--n", type=int, required=True, help="number of maps")
    ap.add_argument("--size", type=int, default=2048, help="map size in px, 512 + k x 256 and >= 1024")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--gpu", default="0", help="the one GPU to use (CUDA index)")
    ap.add_argument("--batch", type=int, default=8, help="maps sampled together on the GPU")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel single-core PyBDSF processes (default: CPU count - 2, at most 24)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rescale", type=float, default=None,
                    help="largest factor between generated and requested noise (default: calibration)")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from glori.inference.noise_map_sampler import NoiseBandMapSampler

    lo, hi = (float(v) for v in args.noise.split("-"))
    sampler = NoiseBandMapSampler(device="cuda:0", batch_size=args.batch, bdsf_workers=args.workers)
    try:
        sampler.check_band(lo, hi)  # refuse before loading anything
        sampler.check_size(args.size)
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    report = sampler.sample(lo, hi, args.n, args.out, map_px=args.size, seed=args.seed,
                            max_rescale=args.max_rescale)
    print(f"{report['n_kept']}/{args.n} maps in {args.out} "
          f"({report['n_generated']} generated, rejected: {report['rejected']})")


if __name__ == "__main__":
    main()
