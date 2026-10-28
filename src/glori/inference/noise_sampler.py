"""
Generate LOFAR-like 512x512 images whose noise lies in a requested band, e.g.
40-55 uJy/beam, for testing source finders.

Noise is the median of the PyBDSF rms map (LoTSS settings, 6" beam, 1.5" px).
The rms-conditioned denoiser (LDM-Denoiser-WnetCC-rms-v1) cannot reach an
absolute noise from the rms channel alone, since the catalog itself encodes the
depth. Every image is therefore generated from a catalog consistent with a
depth T:
  - real LoTSS-DR3 sources of a random DR3 field (public cone search) with
    peak >= 8x its native noise (LoTSS is complete there) and >= 3 T
  - synthetic sources below that, with (SNR, total/peak, Maj) drawn from DR3,
    peak = SNR x T and a number following the DR3 density-depth relation
  - rms channel = T / 1.07 at every source pixel
Per image, a noise s is drawn log-uniformly from the requested band and T is
taken from the calibration so that the median generated noise is s. The image
is measured with PyBDSF and rescaled by s / measured noise, so its noise is
exactly s; the model scales sources and noise together, so the catalog fluxes
stay consistent (calibrated residual factor in the truth catalog). Images that
would need a rescale beyond `max_rescale` or fail the quality checks of the
calibration (noise texture like real DR3 cutouts of that noise, bright input
sources recovered) are dropped and replaced.

Calibration: configs/noise_calibration/rms-v1.json (built by
scripts/ldm/build_noise_calibration.py). Requests outside its valid range are
refused. The checkpoints, the scaler and two small DR3-derived data files (source
properties, field centres) come from the Hugging Face repository via glori.hub;
source catalogs are queried from the public LoTSS-DR3 cone search.
"""

import contextlib
import io
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import glori.infra.logging as my_logging
import glori.settings.paths as paths

NOISE_CALIBRATION = paths.CONFIG_PARENT / "noise_calibration" / "rms-v1.json"
DR3_CONE_URL = "https://vo.astron.nl/lotss_dr3/q/src_cone/scs.xml"
DR3_COLUMNS = ["RA", "DEC", "Total_flux", "Peak_flux", "Maj", "Isl_rms"]  # mJy, mJy/beam, arcsec
SAMPLE_KW = dict(guidance_strength=0.2, timesteps=25, use_inpainting_replacement=True)

IMAGE_SIZE, LATENT_SIZE = 512, 128
CTX_SIZE, CTX_PX = 256, 4  # context covers 1024 px, image is context[64:192, 64:192]
CELL_DEG = CTX_PX * 1.5 / 3600  # one context cell is 6"
MATCH_PX = 8.0  # 12", generated sources sit ~2-3 px off the catalog position
TEXTURE = ["grain_px", "neg5", "pos5", "aniso"]


# ---- released data, DR3 catalog and fields ------------------------------------
def snr_pool():
    """(SNR, total/peak, Maj) of DR3 sources from typical-depth regions (released file)."""
    from glori.hub import download_pretrained

    return np.load(download_pretrained("dr3_snr_pool"))["pool"]


def dr3_fields():
    """Field centres with DR3 coverage: ra, dec, native_uJy, single_ok, map_ok (released file)."""
    from glori.hub import download_pretrained

    return pd.read_csv(download_pretrained("dr3_fields"))


def load_dr3_catalog():
    """The full LoTSS-DR3 source catalog from group storage (only for building release files)."""
    from glori.data.load import load_lotss_catalog

    cat = load_lotss_catalog(select_cols=DR3_COLUMNS)
    return {k: cat[k].to_numpy() for k in cat.columns}


def fetch_dr3_catalog(ra, dec, radius_deg, max_rec=200_000, tries=3):
    """DR3 sources within `radius_deg` of (ra, dec) from the public cone search, sorted by DEC."""
    import requests
    from astropy.table import Table

    params = dict(RA=ra, DEC=dec, SR=radius_deg, MAXREC=max_rec)
    for k in range(tries):
        try:
            r = requests.get(DR3_CONE_URL, params=params, timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))
    t = Table.read(io.BytesIO(r.content), format="votable").to_pandas()
    if len(t) >= max_rec:
        raise RuntimeError(f"DR3 cone search truncated at {max_rec} sources, use a smaller area.")
    # the public "Majax" is the catalog's Maj (beam-included major axis)
    t = t.rename(columns={"Majax": "Maj"})[DR3_COLUMNS].dropna().sort_values("DEC")
    return {k: t[k].to_numpy(np.float64) for k in DR3_COLUMNS}


def field_context(cat, ra, dec, n):
    """
    (5, n, n) context of the DR3 sources around (ra, dec): 6" cells on a SIN
    projection, summed per cell like the training data (position channel
    2 x count - 1, fluxes [mJy], Maj ["], rms = Isl_rms [uJy]).
    """
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [-CELL_DEG, CELL_DEG]
    w.wcs.crpix = [n / 2 + 0.5, n / 2 + 0.5]
    r = n * CELL_DEG
    lo, hi = np.searchsorted(cat["DEC"], [dec - r, dec + r])
    sel = slice(lo, hi)
    d_ra = (cat["RA"][sel] - ra + 180) % 360 - 180
    near = np.abs(d_ra * np.cos(np.radians(dec))) < r
    c = {k: v[sel][near] for k, v in cat.items()}
    x, y = w.wcs_world2pix(c["RA"], c["DEC"], 0)
    ix, iy = np.round(x).astype(int), np.round(y).astype(int)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    ctx = np.zeros((5, n, n), np.float32)
    vals = [np.ones(ok.sum()), c["Total_flux"][ok], c["Peak_flux"][ok], c["Maj"][ok], c["Isl_rms"][ok] * 1e3]
    for ch, v in enumerate(vals):
        np.add.at(ctx[ch], (iy[ok], ix[ok]), np.asarray(v, np.float32))
    ctx[0] = np.where(ctx[0] > 0, 2 * ctx[0] - 1, -1)
    return ctx


def fetch_field_context(ra, dec, n):
    """Context of n x n cells around (ra, dec) from the public DR3 cone search."""
    radius = n * CELL_DEG / np.sqrt(2) * 1.02  # half diagonal
    return field_context(fetch_dr3_catalog(ra, dec, radius), ra, dec, n)


def code_version():
    """git commit of the code (with '+dirty' for local changes), else the package version."""
    here = Path(__file__).parent
    try:
        rev = subprocess.run(["git", "describe", "--always", "--dirty=+dirty"], cwd=here,
                             capture_output=True, text=True, timeout=10)
        if rev.returncode == 0 and rev.stdout.strip():
            return rev.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    from importlib.metadata import PackageNotFoundError, version

    try:
        return f"glori {version('glori')}"
    except PackageNotFoundError:
        return "unknown"


def load_scaler():
    """The image scaler (LOFAR_scaler_II) from the released files."""
    from glori.data.trf.scalers import LOFARScaler
    from glori.hub import download_pretrained

    return LOFARScaler(**torch.load(download_pretrained("scaler"), weights_only=False))


# ---- depth-matched catalog ---------------------------------------------------


def depth_matched_context(ctxt, depth, pool, rng, method):
    """
    ctxt: (5, H, W) catalog context of a real cutout (6" cells, (5, 256, 256)
    for single images), last channel = true rms [uJy] at source pixels. Returns
    the context for depth `depth` [uJy/beam] and a boolean map of the synthetic
    sources.
    """
    c = ctxt.copy()
    src = c[0] > 0
    native = float(np.median(c[4][src])) if src.any() else depth
    peak_uJy = c[2] * 1e3  # catalog channels are in mJy
    keep = (
        src
        & (peak_uJy >= method["keep_native"] * native)
        & (peak_uJy >= method["drop_snr"] * depth)
    )
    c[:, ~keep] = 0
    c[0][~keep] = -1  # empty pixels carry -1 in the position channel

    # catalogued sources below the real-complete limit at this depth
    split = method["keep_native"] * native
    area_deg2 = c.shape[1] * c.shape[2] * CELL_DEG**2
    n_all = method["density_a"] * depth ** method["density_b"] * area_deg2
    n_new = rng.poisson(n_all * float(np.mean(pool[:, 0] * depth < split)))
    draws = pool[rng.integers(0, len(pool), size=10 * n_new + 10)]
    draws = draws[draws[:, 0] * depth < split][:n_new]
    pix = rng.choice(np.flatnonzero(~keep.ravel()), size=len(draws), replace=False)
    yy, xx = np.unravel_index(pix, keep.shape)
    peak_mJy = draws[:, 0] * depth * 1e-3
    c[0][yy, xx] = 1
    c[1][yy, xx] = peak_mJy * draws[:, 1]
    c[2][yy, xx] = peak_mJy
    c[3][yy, xx] = draws[:, 2]
    c[4] = (c[0] > 0) * (depth / method["img_to_isl"])
    synthetic = np.zeros_like(keep)
    synthetic[yy, xx] = True
    return c, synthetic


def truth_catalog(ctxt, synthetic):
    """Sources of the image area: pixel position (x = column), fluxes [Jy], Maj."""
    lo, hi = (CTX_SIZE - IMAGE_SIZE // CTX_PX) // 2, (CTX_SIZE + IMAGE_SIZE // CTX_PX) // 2
    c = ctxt[:, lo:hi, lo:hi]
    rows, cols = np.nonzero(c[0] > 0)
    return pd.DataFrame(
        dict(
            x_px=cols * CTX_PX + (CTX_PX - 1) / 2,
            y_px=rows * CTX_PX + (CTX_PX - 1) / 2,
            peak_flux_input=c[2][rows, cols] * 1e-3,
            total_flux_input=c[1][rows, cols] * 1e-3,
            maj_arcsec=c[3][rows, cols],
            synthetic=synthetic[lo:hi, lo:hi][rows, cols],
        )
    )


# ---- measurement (runs in PyBDSF worker processes) -----------------------------
def _autocorr_halfwidth(img, sig, n_lag=12):
    a = np.clip(img / sig, -4, 4)
    a = a - a.mean()
    f = np.fft.fft2(a)
    ac = np.fft.ifft2(f * np.conj(f)).real
    ac = 0.5 * (ac[0, :n_lag] + ac[:n_lag, 0]) / ac[0, 0]
    k = np.argmax(ac < 0.5)
    return k - 1 + (ac[k - 1] - 0.5) / (ac[k - 1] - ac[k])


def _anisotropy(z, n_ang=36):
    """Max/median of the angular power in a mid-frequency annulus (stripes -> spikes)."""
    p = np.abs(np.fft.fftshift(np.fft.fft2(np.clip(z, -5, 5)))) ** 2
    n = z.shape[0]
    ky, kx = np.indices(p.shape) - n // 2
    k = np.hypot(kx, ky) / n
    ang = np.mod(np.arctan2(ky, kx), np.pi)
    sel = (k > 0.05) & (k < 0.25)
    bins = np.minimum((ang[sel] / np.pi * n_ang).astype(int), n_ang - 1)
    power = np.bincount(bins, weights=p[sel], minlength=n_ang) / np.bincount(
        bins, minlength=n_ang
    )
    return float(power.max() / np.median(power))


def measure(img, tmpdir):
    """PyBDSF noise, noise-texture metrics and source catalog (x, y [px]) of one image."""
    from astropy.wcs import WCS

    from glori.analysis.bdsf_analysis import (
        DEFAULT_KWARGS,
        bdsf_on_image,
        catalogs_from_bdsf,
        make_header_dict,
    )

    kw = dict(DEFAULT_KWARGS, quiet=True, debug=False, ncores=1)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
        res = bdsf_on_image(img.astype(np.float64), tmpdir=Path(tmpdir), **kw)
        srl = catalogs_from_bdsf(res, tmpdir=Path(tmpdir))["srl"]
    rms_map = np.asarray(res.rms_arr).squeeze().T
    resid = np.asarray(res.resid_gaus_arr).squeeze().T
    sigma = float(np.median(rms_map))
    z = resid / rms_map
    m = dict(
        noise_uJy=1e6 * sigma,
        grain_px=float(_autocorr_halfwidth(img, sigma)),
        neg5=float(np.mean(z < -5)),
        pos5=float(np.mean(z > 5)),
        aniso=_anisotropy(z),
    )
    cols = ["RA", "DEC", "Peak_flux", "E_Peak_flux", "Total_flux", "E_Total_flux", "Maj", "Min", "PA"]
    srl = srl[[c for c in cols if c in srl]].reset_index(drop=True)
    if len(srl):
        x, y = WCS(make_header_dict()).celestial.all_world2pix(srl.RA.values, srl.DEC.values, 0)
        srl.insert(0, "y_px", y)
        srl.insert(0, "x_px", x)
    else:
        srl = pd.DataFrame(columns=["x_px", "y_px"] + cols)
    return m, srl


def _measure_job(img, tmpdir):
    try:
        return measure(img, tmpdir), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _worker_init(nice):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.nice(nice)


def write_fits(img, fpath, **cards):
    """Image in the PyBDSF header convention, plus (value, comment) cards."""
    from astropy.io import fits

    from glori.analysis.bdsf_analysis import make_header_dict

    hdr = fits.Header(make_header_dict())
    hdr["BUNIT"] = "JY/BEAM"
    for k, v in cards.items():
        hdr[k] = v
    fits.PrimaryHDU(data=img, header=hdr).writeto(fpath, overwrite=True)


# ---- sampler ---------------------------------------------------------------------
class NoiseBandSampler:
    """
    sampler = NoiseBandSampler()
    report = sampler.sample(40, 55, n=100, out_dir="noise_40-55")
    """

    def __init__(
        self,
        calibration=NOISE_CALIBRATION,
        denoiser_ckpt=None,
        vae_ckpt=None,
        device="cuda:0",
        batch_size=16,
        bdsf_workers=None,
        bdsf_nice=10,
    ):
        """
        denoiser_ckpt, vae_ckpt: checkpoint paths, default: the released rms-v1
        denoiser and VQ-VAE (glori.hub). bdsf_workers: parallel single-core
        PyBDSF processes, default: CPU count - 2, at most 24.
        """
        self.logger = my_logging.get_logger(__name__)
        self.calib = json.loads(Path(calibration).read_text())
        self.calibration_file = Path(calibration)
        self._ckpts = dict(denoiser=denoiser_ckpt, vae=vae_ckpt)
        self.device, self.batch_size = device, batch_size
        self.bdsf_workers = bdsf_workers or max(1, min(24, (os.cpu_count() or 4) - 2))
        self.bdsf_nice = bdsf_nice
        self._sampler = self._base = self._pool = None

    def checkpoint(self, which):
        """Path of the denoiser or VAE checkpoint (downloaded on first use)."""
        if self._ckpts[which] is None:
            from glori.hub import download_pretrained

            self._ckpts[which] = download_pretrained({"denoiser": "denoiser-rms", "vae": "vae"}[which])
        return Path(self._ckpts[which])

    # -- calibration ----------------------------------------------------------
    @property
    def valid_range(self):
        """(lowest, highest) noise [uJy/beam] that can be requested."""
        v = self.calib["valid_range"]
        return v["noise_min_uJy"], v["noise_max_uJy"]

    def check_band(self, noise_min, noise_max):
        lo, hi = self.valid_range
        if not noise_min < noise_max:
            raise ValueError(f"Empty noise band {noise_min}-{noise_max} uJy/beam.")
        if noise_min < lo or noise_max > hi:
            raise ValueError(
                f"Noise band {noise_min}-{noise_max} uJy/beam is outside the validated "
                f"range {lo:.1f}-{hi:.1f} uJy/beam of {self.calibration_file.name}."
            )

    def depth_for_noise(self, noise_uJy):
        """Catalog depth T whose median generated noise is `noise_uJy`."""
        c = self.calib["calibration"]
        T = np.clip(
            np.exp(np.interp(np.log(noise_uJy), np.log(c["noise_p50"]), np.log(c["target"]))),
            *self.calib["valid_range"]["target"],
        )
        return T

    def flux_factor(self, depth, noise_uJy):
        """Expected measured / input peak flux in an image of depth T with noise `noise_uJy`."""
        f = self.calib["flux_correction"]
        return noise_uJy / depth * np.interp(depth, f["target"], f["factor"])

    def texture_band(self, noise_uJy):
        b = self.calib["texture_bands"]
        k = int(np.argmin(np.abs(np.log(np.array(b["noise_uJy"]) / noise_uJy))))
        return {m: (b[m]["p_lo"][k], b[m]["p_hi"][k]) for m in TEXTURE}

    # -- lazy resources ---------------------------------------------------------
    def _load_sampler(self):
        if self._sampler is None:
            from glori.inference.ldm_sampler import LDMSampler
            from glori.models.diffusion.denoiser import Denoiser
            from glori.models.vae.vqvae import VQVAE

            self._sampler = LDMSampler(
                denoiser=Denoiser.load_from_checkpoint(self.checkpoint("denoiser"), map_location="cpu"),
                vae=VQVAE.load_from_checkpoint(self.checkpoint("vae"), map_location="cpu"),
                px_scaler=load_scaler(),
                ctxt_scalers=[],  # the context is passed unscaled
                device=self.device,
                image_size=IMAGE_SIZE,
                latent_size=LATENT_SIZE,
                progress_bar=False,
            )
        return self._sampler

    def _load_base(self):
        """Candidate DR3 fields (and the source-property pool for the synthetic sources)."""
        if self._base is None:
            f = dr3_fields()
            self._base = f[f.single_ok].reset_index(drop=True)
            self._pool = snr_pool()
        return self._base

    # -- generation ---------------------------------------------------------------
    def _generate(self, n, band, rng):
        """n images, noise drawn log-uniformly over the band -> images, contexts, info."""
        fields = self._load_base()
        method = self.calib["method"]
        picks = fields.iloc[rng.integers(len(fields), size=n)]
        with ThreadPoolExecutor(4) as ex:  # a few cone searches at a time
            raw = list(ex.map(lambda rd: fetch_field_context(*rd, CTX_SIZE), zip(picks.ra, picks.dec)))
        ctxs, syn, info = [], [], []
        for (ra, dec), c in zip(zip(picks.ra, picks.dec), raw):
            noise = float(np.exp(rng.uniform(*np.log(band))))
            depth = float(self.depth_for_noise(noise))
            c, s = depth_matched_context(c, depth, self._pool, rng, method)
            ctxs.append(c)
            syn.append(s)
            info.append(dict(field_ra=float(ra), field_dec=float(dec), noise_request_uJy=noise, depth_uJy=depth))
        sampler = self._load_sampler()
        g = torch.Generator().manual_seed(int(rng.integers(2**63)))
        imgs = sampler.sample(
            catalog_context=torch.from_numpy(np.stack(ctxs)).float(),
            inpainting_context=(
                torch.ones((n, 1, LATENT_SIZE, LATENT_SIZE)),
                torch.zeros((n, 3, LATENT_SIZE, LATENT_SIZE)),
            ),
            seed_noise=torch.randn((n, 3, LATENT_SIZE, LATENT_SIZE), generator=g),
            **SAMPLE_KW,
        )
        imgs = imgs.cpu().numpy() if torch.is_tensor(imgs) else np.asarray(imgs)
        return imgs.reshape(n, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32), ctxs, syn, info

    def _judge(self, m, truth, srl, max_rescale):
        """
        Reason for rejection, or None if the image is kept (m: metrics after
        rescaling, truth: with the expected peak flux in the image).
        """
        if not 1 / max_rescale <= m["rescale"] <= max_rescale:
            return "rescale too large"
        tb = self.texture_band(m["noise_uJy"])
        if not all(tb[k][0] <= m[k] <= tb[k][1] for k in TEXTURE):
            return "noise texture unlike real DR3"
        bright = truth[truth.peak_flux * 1e6 >= 10 * m["noise_uJy"]]
        if len(bright):
            if len(srl):
                from scipy.spatial import cKDTree

                d, _ = cKDTree(srl[["x_px", "y_px"]].values).query(bright[["x_px", "y_px"]].values)
                rec = float(np.mean(d < MATCH_PX))
            else:
                rec = 0.0
            if rec < self.calib["criteria"]["min_bright_recovery"]:
                return "bright sources missing"
        return None

    def sample(self, noise_min, noise_max, n, out_dir, seed=0, max_rescale=None, max_attempts=None):
        """
        Generate `n` images with a PyBDSF noise in [noise_min, noise_max]
        uJy/beam that pass the quality checks. Writes FITS images, the truth and
        PyBDSF catalogs and a report to `out_dir`.
        """
        self.check_band(noise_min, noise_max)
        band = (float(noise_min), float(noise_max))
        max_rescale = max_rescale or self.calib["criteria"]["max_rescale"]
        max_attempts = max_attempts or 5 * n + 4 * self.batch_size
        out = Path(out_dir)
        (out / "images").mkdir(parents=True, exist_ok=True)
        tmpdir = Path(tempfile.mkdtemp(prefix="bdsf_", dir=out))
        rng = np.random.default_rng(seed)
        self.logger.info(
            f"Generating {n} images with noise {band[0]}-{band[1]} uJy/beam "
            f"(catalog depth {self.depth_for_noise(band[0]):.0f}-{self.depth_for_noise(band[1]):.0f})"
        )

        kept, truths, srls, rejected = [], [], [], {}
        n_tried, t0 = 0, time.time()
        pool = ProcessPoolExecutor(
            max_workers=self.bdsf_workers,
            mp_context=get_context("spawn"),
            initializer=_worker_init,
            initargs=(self.bdsf_nice,),
        )
        pending = {}
        try:
            while len(kept) < n:
                # keep the GPU busy while PyBDSF works on earlier batches
                need = n - len(kept) - len(pending)  # replacements follow rejections
                if n_tried < max_attempts and need > 0 and len(pending) < 2 * self.batch_size:
                    nb = min(self.batch_size, need, max_attempts - n_tried)
                    imgs, ctxs, syns, infos = self._generate(nb, band, rng)
                    for img, c, s, inf in zip(imgs, ctxs, syns, infos):
                        pending[pool.submit(_measure_job, img, tmpdir)] = (img, c, s, inf)
                    n_tried += nb
                    continue
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    img, c, s, inf = pending.pop(f)
                    (res, err) = f.result()
                    if err is not None:
                        rejected["PyBDSF failed"] = rejected.get("PyBDSF failed", 0) + 1
                        continue
                    m, srl = res
                    # PyBDSF is scale-invariant: rescaling the image scales its rms
                    # map and catalog fluxes by the same factor, detections stay
                    m["noise_generated_uJy"] = m["noise_uJy"]
                    m["rescale"] = inf["noise_request_uJy"] / m["noise_uJy"]
                    m["noise_uJy"] = inf["noise_request_uJy"]
                    img = img * np.float32(m["rescale"])
                    for col in ("Peak_flux", "E_Peak_flux", "Total_flux", "E_Total_flux"):
                        if col in srl:
                            srl[col] = srl[col] * m["rescale"]
                    truth = truth_catalog(c, s)
                    ff = float(self.flux_factor(inf["depth_uJy"], m["noise_uJy"]))
                    truth["peak_flux"] = truth.peak_flux_input * ff
                    truth["total_flux"] = truth.total_flux_input * ff
                    why = self._judge(m, truth, srl, max_rescale)
                    if why is not None or len(kept) >= n:
                        rejected[why or "surplus"] = rejected.get(why or "surplus", 0) + 1
                        continue
                    k = len(kept)
                    truth.insert(0, "image_id", k)
                    srl.insert(0, "image_id", k)
                    truths.append(truth)
                    srls.append(srl)
                    kept.append(dict(image_id=k, file=f"images/img_{k:05d}.fits", **inf, flux_factor=ff, **m))
                    write_fits(
                        img,
                        out / kept[-1]["file"],
                        NOISE=(round(m["noise_uJy"], 3), "PyBDSF median rms [uJy/beam]"),
                        RESCALE=(round(m["rescale"], 4), "factor applied to the generated image"),
                        DEPTH=(round(inf["depth_uJy"], 3), "catalog depth T [uJy/beam]"),
                        FIELDRA=(round(inf["field_ra"], 4), "DR3 field of the real sources [deg]"),
                        FIELDDEC=(round(inf["field_dec"], 4), "DR3 field of the real sources [deg]"),
                        MODEL=(self.checkpoint("denoiser").stem, "denoiser"),
                        CALIB=(self.calibration_file.name, "noise calibration"),
                        CODEVER=(code_version(), "glori code version"),
                    )
                if len(kept) and len(kept) % 10 == 0 or not pending:
                    self.logger.info(
                        f"{len(kept)}/{n} kept, {n_tried} generated, {time.time() - t0:.0f}s"
                    )
        finally:
            for f in pending:
                f.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            for f in tmpdir.glob("*"):
                f.unlink()
            tmpdir.rmdir()

        summary = pd.DataFrame(kept)
        summary.to_csv(out / "summary.csv", index=False)
        if truths:
            pd.concat(truths, ignore_index=True).to_csv(out / "truth_catalog.csv", index=False)
            pd.concat(srls, ignore_index=True).to_csv(out / "pybdsf_catalog.csv", index=False)
        report = dict(
            request=dict(noise_min_uJy=band[0], noise_max_uJy=band[1], n=n, seed=seed),
            n_kept=len(kept),
            n_generated=n_tried,
            rejected=rejected,
            noise_kept_uJy=dict(
                zip(["min", "p50", "max"], np.percentile(summary.noise_uJy, [0, 50, 100]).round(2).tolist())
            )
            if len(kept)
            else None,
            denoiser=str(self.checkpoint("denoiser")),
            calibration=str(self.calibration_file),
            code_version=code_version(),
            valid_range_uJy=list(self.valid_range),
            seconds=round(time.time() - t0),
        )
        (out / "report.json").write_text(json.dumps(report, indent=1))
        if len(kept) < n:
            self.logger.warning(
                f"Only {len(kept)}/{n} images in the band after {n_tried} attempts, see report.json"
            )
        return report
