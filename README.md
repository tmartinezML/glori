# glori — Generate LOFAR Radio Images

`glori` implements a **latent diffusion model (LDM)** that generates realistic, synthetic
radio-continuum survey maps — simulated LOFAR skies — with explicit control over the
positions, brightness, and size of the sources on the map. It also implements **SWIIT
sampling**, a sliding-window technique that stitches many LDM patches together (via
inpainting) into maps of arbitrary size, seamlessly.

This is the reference implementation for:

> T. Vicánek Martínez and M. Brüggen (2026), *"Generating radio continuum survey maps of
> arbitrary size with latent diffusion models"*, A&A, arXiv:[2609.06549](https://arxiv.org/abs/2609.06549)

If you use this code or the pretrained models, please cite that paper.

## How it works (short version)

1. A **VQ-VAE** (`glori.models.vae.vqvae.VQVAE`) compresses 256px image patches of LOFAR
   sky maps into a small discrete latent space (downsampling factor 4).
2. A **diffusion denoiser** (`glori.models.diffusion.denoiser.Denoiser`, a U-Net) is
   trained to generate new samples *in that latent space*, conditioned on an optional
   **catalog context vector** — a small image encoding source positions, total flux,
   peak flux, and size, derived from a real or synthetic source catalog.
3. **SWIIT sampling** (`glori.inference.swiit_sampler.SWIITSampler`) generates maps
   larger than a single patch by sliding a window across the target area, sampling each
   new patch conditioned on the already-sampled neighboring latents (via inpainting), and
   blending the decoded, overlapping patches together.

See Sections 2–5 of the paper for the full methodology. A worked example is in
[`tutorials/tutorial_getting_started.ipynb`](tutorials/tutorial_getting_started.ipynb), and a
conceptual walkthrough of what actually happens during sampling (noise → denoising loop →
decode) is in [`tutorials/how_sampling_works.ipynb`](tutorials/how_sampling_works.ipynb).

## Repository layout

```
src/glori/
  config/       preset-loading machinery (JSON configs -> model/dataset kwargs)
  data/         dataset classes, transforms, scalers, catalog/mosaic loading
  inference/    DMSampler (low-level), LDMSampler (single patch), SWIITSampler (arbitrary size)
  models/       VAE / VQ-VAE, diffusion denoiser (U-Net, DiT), model loading utilities
  settings/     paths.py — all storage locations (see "Storage locations" below)
  train/        Lightning training entry points (train.denoiser, train.vae, train.flow)
  plotting/     plotting helpers for maps, metrics, training curves
  analysis/     PyBDSF-based source finding and result analysis
  maps/         legacy T-RECS + telescope-simulation pipeline (see "What's not covered" below)
configs/
  model_presets/    JSON configs for VAE / VQ-VAE / Denoiser architectures & training
  dataset_presets/  JSON configs for dataset construction (crops, context, scalers, ...)
notebooks/      exploratory notebooks (mostly research scratchpads, not documentation)
tutorials/      polished, documented example notebooks -- start here
scripts/        one-off training/analysis scripts, of varying levels of maintenance
scripts/release/  tooling for preparing pretrained models for public release
```

## What's *not* covered by this README

`glori.maps.map_maker` + `glori.maps.telescope_simulator` implement an older, separate
pipeline that builds a sky model from a T-RECS source catalog and simulates a full
radio-interferometric observation of it (calibration, noise, PSF) by driving external
LOFAR software (DP3, DDF, `ddf-pipeline`, `losito`). It is **not** the pipeline described
in the paper above.

The first half — T-RECS catalog generation + assembling a clean, noise-free sky model
(`MapMaker`, no external LOFAR software needed) — **is** demonstrated in
[`tutorials/tutorial_legacy_mapmaker.ipynb`](tutorials/tutorial_legacy_mapmaker.ipynb).
The second half — `TelescopeSimulator`, converting that clean sky model into a realistic
simulated observation — needs a cluster environment with the external LOFAR software
installed and is not documented here; read `src/glori/maps/telescope_simulator.py`
directly, or ask a group member familiar with the LOFAR imaging stack.

This older pipeline is the one described in the group's *previous* paper:

> T. Vicánek Martínez, H. W. Edler, and M. Brüggen (2025), *"Simulating realistic radio
> continuum survey maps with diffusion models"*, A&A, arXiv:[2506.11715](https://arxiv.org/abs/2506.11715)

There, individual source images (sampled from a separate, size-conditioned diffusion
model — `model_results/Prototypes_Model_SizeCond`, for extended sources — or simple 2D
Gaussians for compact ones) are assembled onto a T-RECS catalog's source positions to
build a "sky model", which is then run through a full simulated LOFAR observation
(predict visibilities, add noise, CLEAN). This is a fundamentally different use of a
"catalog" than the catalog *context vector* used elsewhere in this repo (see "How it
works" above) — here the catalog drives literal per-source image placement and a real
interferometric simulation, rather than conditioning a single generative model that
produces a whole map directly.

## Installation

The package targets Python 3.11 and is installed in editable mode from `environment.yml`.
Standard routine with your own [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
(or `mamba`/`conda`):

```bash
git clone https://github.com/tmartinezML/glori.git
cd glori
git switch tutorial   # the tutorials and the install fixes live on this branch

micromamba env create -n glori -f environment.yml
micromamba activate glori
pip install -e .
```

### Alternative: uv

With [uv](https://docs.astral.sh/uv/) all dependencies come from `pyproject.toml`:

```bash
git clone https://github.com/tmartinezML/glori.git
cd glori
git switch tutorial   # the tutorials and the install fixes live on this branch

uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[notebooks]"
```

The `notebooks` extra adds `ipykernel`, which the tutorial notebooks need; leave it out
if you only use the package from scripts. Python must be 3.11 (the pinned `numpy` and
`torch` have no wheels for newer versions).

An editable install (`-e`) is recommended, but a regular one works too: the presets from
`configs/` are then shipped inside the package. Either run `uv pip install ".[notebooks]"`
in the clone (on the `tutorial` branch), or install straight from GitHub without cloning.
The part after the last `@` selects the branch:

```bash
uv pip install "glori[notebooks] @ git+https://github.com/tmartinezML/glori.git@tutorial"
```

### Set the storage directory

On a fresh machine you have to point `glori` to a storage directory of your own.
Either set the environment variable `GLORI_STORAGE_PARENT` before importing `glori`:

```bash
export GLORI_STORAGE_PARENT=$HOME/glori_storage
```

or open `src/glori/settings/paths.py` and change `STORAGE_PARENT` there (only possible
with an editable install). Every path this package uses (model checkpoints, cache, image
data, ...) is derived from it, and the whole directory tree is created automatically the
first time you import `glori` — but only under whatever `STORAGE_PARENT` currently
points to.

## Storage locations

All data/model paths are centralized in `src/glori/settings/paths.py`, keyed off a single
`STORAGE_PARENT` constant.

Importing `glori.settings.paths`
creates the directory tree under `STORAGE_PARENT` automatically if it doesn't exist yet.

**You don't need any of this machinery just to run inference**, though. The easiest
option for inference is `glori.hub.download_pretrained(...)` (see Quickstart), which
fetches straight from the public Hugging Face repo and doesn't use `STORAGE_PARENT`
for anything — it still creates the (empty, harmless) `STORAGE_PARENT` directory tree.

## Pretrained models

Two checkpoints are needed to reproduce the paper's main results (patch size 512px /
latent size 128px):

| Component | Checkpoint | Notes |
|---|---|---|
| VQ-VAE | `VQ-VAE-256-DR3opt-FT` | image <-> latent compression, downsampling factor 4 |
| Denoiser | `LDM-Denoiser-WnetCC-v5` | catalog-context-conditioned diffusion U-Net |

Public download: **[huggingface.co/astrokevin/glori-ldm-swiit](https://huggingface.co/astrokevin/glori-ldm-swiit)**.
(Stripped-down, inference-only versions of these checkpoints — weights only, no optimizer
state — are prepared with `scripts/release/strip_checkpoint.py`.)

**License:** the pretrained weights are released under **CC BY-SA 4.0**, separately from
this repo's MIT-licensed code. The models were trained on [LoTSS-DR3](https://lofar-surveys.org/dr3.html)
data, which is itself [CC BY-SA 3.0](https://vo.astron.nl/lotss_dr3/q/src_cone/howtocite) — the share-alike clause carries over to derivative
works like these weights, so if you build on them (fine-tune, redistribute, etc.), your
result needs to stay under a compatible CC BY-SA license too. If you use the weights,
please cite both the paper above and Shimwell et al. (2026), *"The LOFAR Two-metre Sky
Survey. VII. Third data release"*, A&A 707, A198, [10.1051/0004-6361/202557749](https://doi.org/10.1051/0004-6361/202557749).

For inference, you don't need to place checkpoints under `STORAGE_PARENT` at all — every
sampler accepts an explicit file path in place of a symbolic checkpoint name. See the
Quickstart below.

## Quickstart

Minimal unconditional sampling — generates one 512x512px synthetic sky patch from pure
noise, no source catalog needed. Checkpoints download automatically from the public
Hugging Face repo the first time this runs (and are cached locally afterwards) — no
cluster access or manual download required:

```python
import torch
from glori.config.swiit_config import SWIITSamplerConfig
from glori.inference.swiit_sampler import SWIITSampler
from glori.data.trf.scalers import LOFARScaler
from glori.hub import download_pretrained

DENOISER_CKPT = download_pretrained("denoiser")
VAE_CKPT = download_pretrained("vae")
SCALER_PATH = download_pretrained("scaler")

config = SWIITSamplerConfig(
    denoiser="unused",  # ignored: an explicit "/" path in *_ckpt below bypasses name lookup
    denoiser_ckpt=DENOISER_CKPT,
    vae="unused",
    vae_ckpt=VAE_CKPT,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)
sampler = SWIITSampler(config=config)
sampler.scaler = LOFARScaler(**torch.load(SCALER_PATH, weights_only=False))

map_image, latent_map = sampler.sample(sampling_steps=(2, 2), timesteps=25)
physical_map = sampler.scaler.inverse_scale(map_image)  # back to Jy/beam-like units
```

For the full walkthrough — including catalog-conditioned generation (controlling where
sources appear), sampling larger maps, and understanding the sampling parameters — see
[`tutorials/tutorial_getting_started.ipynb`](tutorials/tutorial_getting_started.ipynb).

## Test images in a noise band (for source finders)

`scripts/generate_noise_images.py` generates LOFAR-like 512x512 px images (1.5" px,
6" beam, 144 MHz, Jy/beam) whose noise lies in a requested band, together with the
catalog of the sources put into them:

```bash
python scripts/generate_noise_images.py --noise 40-55 --n 100 --out noise_40-55 --gpu 0
```

or from Python:

```python
from glori.inference.noise_sampler import NoiseBandSampler
NoiseBandSampler(device="cuda:0").sample(40, 55, n=100, out_dir="noise_40-55")
```

**Requirements.** One GPU, a working PyBDSF (`bdsf`) installation and internet access.
On first use, the noise-conditioned denoiser (`LDM-Denoiser-WnetCC-rms-v1`), the VQ-VAE,
the scaler and two small LoTSS-DR3 derived files (source properties for the synthetic
sources, a list of field centres) are downloaded from
[huggingface.co/astrokevin/glori-ldm-swiit](https://huggingface.co/astrokevin/glori-ldm-swiit)
and cached; set `GLORI_PRETRAINED_DIR` to a local copy of that repository to use it instead
(on the group cluster: `/hs/fs08/data/group-brueggen/kevin/glori_paper_release`). Source
catalogs are queried from the public [LoTSS-DR3 cone search](https://vo.astron.nl/lotss_dr3/q/src_cone/info)
for every image, a few at a time.

**Noise** is the median of the PyBDSF rms map with the LoTSS settings of
`glori.analysis.bdsf_analysis.DEFAULT_KWARGS`. Every written image has a noise inside
the band (stored as `NOISE` in the FITS header), drawn log-uniformly over it.

**Valid range: 40–224 µJy/beam.** Requests outside it are refused. The range is the one in
which the calibration (`configs/noise_calibration/rms-v1.json`, 96 cutouts x 23 depths,
3000 real DR3 cutouts as reference) found every criterion met: the noise rises with the
requested depth, the noise texture and artefacts look like real DR3 cutouts of the same
noise, bright input sources are recovered, and source fluxes stay consistent within 10 %.
Below ~37 µJy/beam the model cannot go deeper. Above ~224 µJy/beam the noise still rises
(up to ~370 µJy/beam tested) with consistent fluxes, but bright sources are missed slightly
more often than the criterion allows and real reference cutouts become sparse. Real 6"
DR3 cutouts span ~55–200 µJy/beam (5–95 %).

**How an image is made.** The model (`LDM-Denoiser-WnetCC-rms-v1`) reads the depth of
an image mostly from its input catalog, so each image gets a catalog consistent with a
depth T: the real sources of a random DR3 field (centres drawn uniformly over the
survey footprint) above its completeness limit (8x its native noise), plus synthetic faint sources with DR3-like SNR, sizes and
total/peak ratios down to the depth T, in DR3-like numbers. T comes from the calibration
so that the generated noise is close to the requested one; the image is then measured
with PyBDSF and rescaled to the requested noise exactly. PyBDSF results scale linearly
with the image, and the model scales sources and noise together, so the rescale
(typically within ±15 %, at most x1.3) changes the flux scale only, not the LOFAR noise
texture. Images that would need a larger rescale, whose noise texture falls outside
the range of real DR3 cutouts, or that miss bright input sources are dropped and
regenerated (`report.json` counts them).

**Output** in `--out`:

| File | Content |
|---|---|
| `images/img_#####.fits` | image [Jy/beam]; header: `NOISE` [µJy/beam], `RESCALE`, `DEPTH` (T), `FIELDRA`/`FIELDDEC` (DR3 field of the real sources), `CODEVER` (git commit of the code) |
| `truth_catalog.csv` | all input sources per `image_id`: `x_px`, `y_px` (0-based, x = column), `peak_flux`, `total_flux` [Jy], `maj_arcsec`, `synthetic` |
| `pybdsf_catalog.csv` | PyBDSF sources of each image (same pixel convention) |
| `summary.csv`, `report.json` | one row per image (noise, rescale, texture metrics), and the run: request, rejections, runtime |

**Known limitations of the truth catalog**
- Positions are on the 4 px grid of the catalog context (6"), and generated sources sit
  ~2–3 px (3–4.5") off their catalog position; match with a radius of ~12".
- `peak_flux`/`total_flux` are the input fluxes times the calibrated image flux factor.
  The PyBDSF peak flux of single sources (>= 20 sigma) scatters by ~±25 % around it; per
  image the median ratio is 0.94–1.01 across the range. Above ~150 µJy/beam, peak fluxes
  of bright extended sources (Maj >= 8") come out ~10 % low.
- Faint emission the model adds on its own (below the catalog depth) is not listed.

**Resources.** One GPU (`--gpu`) and `--workers` single-core PyBDSF processes at low
priority (default: CPU count - 2, at most 24). About 1 min per 16 images on an A100.

### Large maps in a noise band

`scripts/generate_noise_maps.py` makes survey-like maps of any size (default 2048x2048 px
= 51'; 512 + k x 256 px, at least 1024) with SWIIT, the tiled sampler, from the same kind
of depth-matched catalog:

```bash
python scripts/generate_noise_maps.py --noise 60-80 --n 8 --size 2048 --out maps_60-80 --gpu 0
```

```python
from glori.inference.noise_map_sampler import NoiseBandMapSampler
NoiseBandMapSampler(device="cuda:0").sample(60, 80, n=8, out_dir="maps_60-80", map_px=2048)
```

**Noise of a map** is the median of its PyBDSF rms map; each map is rescaled as a whole
so that this median lies in the band. The noise still varies across a map: 256 px blocks
span a factor ~1.2–1.4 (p95/p5) at low noise and ~1.4–2.0 at high noise, somewhat more
than in real DR3 fields (~1.1–1.3). The PyBDSF rms map is therefore delivered with every
map; use it for local SNR.

**Valid range: 50–224 µJy/beam** for the map median; requests outside it are refused.
The calibration (`configs/noise_calibration/rms-v1-swiit.json`, 8 DR3 fields of native depth
37–87 µJy/beam x 7 depths, 2048 px) found the map noise rising with the depth in every
field, no tile seams, ~99 % of bright input sources recovered and 85–96 % of the 512 px
tiles with a DR3-like noise texture. Below ~50 µJy/beam only ~80 % of the tiles pass
(striped, too anisotropic noise), so that part is excluded.

**How a map is made.** Real sources come from a random DR3 field with native depth
<= 90 µJy/beam and full survey coverage over the map and a 256 px margin (the margin is
the context SWIIT sees beyond the map edge); the catalog is made consistent with depth T
as for single images. Maps that would need a rescale above x1.3, whose 512 px tiles have
a DR3-like texture in fewer than 75 % of cases, or that miss more than 20 % of the bright
input sources are dropped and regenerated.

**Fluxes.** SWIIT maps come out with source fluxes 5–20 % below the catalog, depending on
the field. The truth catalog is therefore calibrated per map: `peak_flux`/`total_flux` are
the input fluxes times the median measured / input peak flux of the map's own sources
with input peak >= 20 T (`FLUXFAC` in the header; the calibrated factor of the depth is
used if a map has fewer than 10 such sources). Checked by re-running PyBDSF on the written
maps: sources at 20–100 sigma come out at 0.98–1.08 of the truth flux per map; sources above
100 sigma (mostly bright extended ones) ~10 % low, with a larger scatter.

**Output** in `--out`:

| File | Content |
|---|---|
| `maps/map_#####.fits` | map [Jy/beam]; header: `NOISE` (median rms) [µJy/beam], `RESCALE`, `DEPTH` (T), `FLUXFAC`, `FIELDRA`/`FIELDDEC` (DR3 field of the real sources), `CODEVER` |
| `maps/map_#####_rms.fits` | PyBDSF rms map of the map [Jy/beam] |
| `truth_catalog.csv` | input sources per `map_id`: as for single images, plus `detected` (PyBDSF source within 12") |
| `pybdsf_catalog.csv` | PyBDSF sources of each map |
| `tiles.csv` | per 512 px tile: noise, texture metrics, `texture_ok` |
| `summary.csv`, `report.json` | one row per map (noise, rescale, flux factor, rms spread, texture fraction, bright-source recovery), and the run |

The real sources can come from anywhere in DR3, including areas whose catalogs were seen
in training; the maps themselves are newly generated.

**Resources.** Requirements as for single images; one cone search per map (checked complete
up to 3° radius, i.e. maps up to 8192 px). One GPU and `--workers` single-core PyBDSF processes at low priority.
Sampling 8 maps of 2048 px together (`--batch 8`) takes ~26 min on an A100 (~3.3 min per
map; ~4.3 min per map in batches of 4); time grows with the map area. PyBDSF takes ~2–3 min
per 2048 px map and runs while the next batch is sampled.

## Known issues / gotchas

- **`STORAGE_PARENT`** in `settings/paths.py` is hardcoded to the original author's
  cluster path. Change it for your own setup or set `GLORI_STORAGE_PARENT` (see
  "Installation"/"Storage locations" above), or use the explicit-checkpoint-path pattern above to bypass it entirely for
  inference. The directory tree under it is created automatically on import, so you
  don't need to `mkdir` anything yourself once it points somewhere valid — but on a
  fresh machine you do need to change it *before* the first `import glori...`, or the
  auto-creation will target the original hardcoded path instead.
- **`glori.models.load.load_model()` is dead code** — it unconditionally raises
  `NotImplementedError`. The real loading path (used everywhere in this codebase) is
  `SomeLightningClass.load_from_checkpoint(parse_lightning_ckpt(...))`.
- **Old checkpoints can fail to load** with a `TypeError` about an unexpected `distance_fn`
  keyword argument in the VQ-VAE's vector-quantization layer. This happens for checkpoints
  saved before the `vqtorch` dependency was pinned to a newer commit (see commit
  `412cf25`). Fix: `del ckpt["hyper_parameters"]["vq_kwargs"]["distance_fn"]` before
  loading, or re-save the checkpoint after removing that key.
- **Several loose scripts under `scripts/`** (e.g. `sample_from_model.py`,
  `sample_intermediates.py`, `map_simulation_streamline.py`) still use a pre-refactor flat
  import layout (`models.*`, `data.*`, `utils.*` instead of `glori.models.*`, etc.) and
  will fail to import as-is. `scripts/ldm/*.py` and
  `tutorials/tutorial_getting_started.ipynb` are kept working; treat everything else under
  `scripts/` as reference material rather than ready-to-run code.
- `tests/unit_tests.py` only checks that every module *imports* without error — it does
  not exercise any sampling/training behavior.

## License

Code: MIT (see `pyproject.toml`). Pretrained model weights: **CC BY-SA 4.0**, see
"Pretrained models" above — this is a separate, stricter license required by the
LoTSS-DR3 training data's own share-alike terms.
