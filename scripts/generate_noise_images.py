"""
Generate LOFAR-like test images in a noise band, e.g. for testing source finders.

  python scripts/generate_noise_images.py --noise 40-55 --n 100 --out noise_40-55

Every image has a PyBDSF noise (median rms map, LoTSS settings) inside the band.
Output in --out:
  images/img_#####.fits   512x512, 1.5" px, 6" beam, Jy/beam (NOISE in header)
  truth_catalog.csv       input sources (x_px, y_px, peak/total flux [Jy], Maj,
                          synthetic or real LoTSS DR3)
  pybdsf_catalog.csv      PyBDSF sources of each image
  summary.csv             one row per image, report.json for the whole run
Uses one GPU and --workers single-core PyBDSF processes at low priority.
Models and data files are downloaded from Hugging Face on first use (or taken
from $GLORI_PRETRAINED_DIR), source catalogs from the public LoTSS-DR3 cone search.
"""

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noise", required=True, help="band in uJy/beam, e.g. 40-55")
    ap.add_argument("--n", type=int, required=True, help="number of images")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--gpu", default="0", help="the one GPU to use (CUDA index)")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel single-core PyBDSF processes (default: CPU count - 2, at most 24)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rescale", type=float, default=None,
                    help="largest factor between generated and requested noise (default: calibration)")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from glori.inference.noise_sampler import NoiseBandSampler

    lo, hi = (float(v) for v in args.noise.split("-"))
    sampler = NoiseBandSampler(device="cuda:0", bdsf_workers=args.workers)
    try:
        sampler.check_band(lo, hi)  # refuse before loading anything
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    report = sampler.sample(lo, hi, args.n, args.out, seed=args.seed, max_rescale=args.max_rescale)
    print(f"{report['n_kept']}/{args.n} images in {args.out} "
          f"({report['n_generated']} generated, rejected: {report['rejected']})")


if __name__ == "__main__":
    main()
