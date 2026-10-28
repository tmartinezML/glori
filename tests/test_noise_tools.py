"""
CPU smoke tests of the noise-band tools (no GPU, network or group storage):
calibration lookups, request checks and the catalog/context construction.

  pytest tests/test_noise_tools.py
"""

import os
import sys

import numpy as np
import pytest

main_folder = os.path.join(os.path.dirname(__file__), "..", "src")
if main_folder not in sys.path:
    sys.path.insert(0, main_folder)

from glori.inference import noise_map_sampler as nm  # noqa: E402
from glori.inference import noise_sampler as ns  # noqa: E402


@pytest.fixture(scope="module")
def samplers():
    return ns.NoiseBandSampler(device="cpu"), nm.NoiseBandMapSampler(device="cpu")


def test_valid_ranges(samplers):
    single, maps = samplers
    assert single.valid_range == pytest.approx((39.9, 223.6), abs=0.1)
    assert maps.valid_range == (50.0, 224.0)


@pytest.mark.parametrize("band", [(30, 50), (100, 300), (60, 50), (50, 50)])
def test_refused_bands(samplers, band):
    with pytest.raises(ValueError):
        samplers[1].check_band(*band)


@pytest.mark.parametrize("px,ok", [(512, False), (1000, False), (1024, True), (2048, True), (4096, True)])
def test_map_sizes(px, ok):
    if ok:
        nm.NoiseBandMapSampler.check_size(px)
    else:
        with pytest.raises(ValueError):
            nm.NoiseBandMapSampler.check_size(px)


def test_depth_for_noise_monotone(samplers):
    for s in samplers:
        lo, hi = s.valid_range
        t = s.depth_for_noise(np.geomspace(lo, hi, 20))
        assert np.all(np.diff(t) > 0)
        assert s.calib["valid_range"]["target"][0] - 1e-6 <= t.min()
        assert t.max() <= s.calib["valid_range"]["target"][1] + 1e-6


def toy_catalog(ra=150.0, dec=50.0, n=400, seed=0):
    """Random sources around (ra, dec) in the column convention of the DR3 catalog (mJy, arcsec)."""
    rng = np.random.default_rng(seed)
    d = dict(
        RA=ra + rng.uniform(-0.2, 0.2, n) / np.cos(np.radians(dec)),
        DEC=dec + rng.uniform(-0.2, 0.2, n),
        Total_flux=rng.uniform(0.2, 20, n),
        Maj=rng.uniform(6, 20, n),
        Isl_rms=np.full(n, 0.06),
    )
    d["Peak_flux"] = d["Total_flux"] * 0.8
    order = np.argsort(d["DEC"])
    return {k: v[order] for k, v in d.items()}


def test_field_context_centre():
    cat = {k: np.array([v]) for k, v in dict(RA=150.0, DEC=50.0, Total_flux=5.0, Peak_flux=4.0, Maj=8.0,
                                             Isl_rms=0.06).items()}
    c = ns.field_context(cat, 150.0, 50.0, ns.CTX_SIZE)
    assert c.shape == (5, ns.CTX_SIZE, ns.CTX_SIZE)
    rows, cols = np.nonzero(c[0] > 0)
    assert len(rows) == 1 and abs(rows[0] - ns.CTX_SIZE / 2) <= 1 and abs(cols[0] - ns.CTX_SIZE / 2) <= 1
    assert c[1:, rows[0], cols[0]] == pytest.approx([5.0, 4.0, 8.0, 60.0])
    assert (c[0] == -1).sum() == ns.CTX_SIZE**2 - 1


def test_depth_matched_context(samplers):
    method = samplers[0].calib["method"]
    c = ns.field_context(toy_catalog(), 150.0, 50.0, ns.CTX_SIZE)
    pool = np.c_[np.geomspace(1, 1000, 5000), np.ones(5000), np.full(5000, 8.0)].astype(np.float32)
    depth = 50.0
    d, syn = ns.depth_matched_context(c, depth, pool, np.random.default_rng(1), method)
    src = d[0] > 0
    real = src & ~syn
    native = 60.0  # Isl_rms of the toy catalog
    assert np.all(d[2][real] * 1e3 >= method["keep_native"] * native)
    assert np.all(d[2][syn] * 1e3 < method["keep_native"] * native)
    assert syn.sum() > 0 and real.sum() > 0
    assert d[4][src] == pytest.approx(np.full(src.sum(), depth / method["img_to_isl"]))
    truth = ns.truth_catalog(d, syn)
    assert truth.x_px.between(0, ns.IMAGE_SIZE).all() and truth.y_px.between(0, ns.IMAGE_SIZE).all()


def test_field_ok(samplers):
    fields = samplers[1].calib["fields"]
    c = ns.field_context(toy_catalog(n=20000), 150.0, 50.0, 256)
    assert nm.field_ok(c, fields) == pytest.approx(60.0)
    c[0, :64, :64] = -1  # an empty 6.4' block: survey gap
    assert nm.field_ok(c, fields) is None


def test_map_truth_catalog():
    c = -np.ones((5, 8, 8), np.float32)
    c[1:] = 0
    c[:, 2, 5] = [1, 3.0, 2.0, 9.0, 60.0]
    t = nm.map_truth_catalog(c, np.zeros((8, 8), bool))
    assert (t.x_px.item(), t.y_px.item()) == (5 * ns.CTX_PX + 1.5, 2 * ns.CTX_PX + 1.5)
    assert (t.peak_flux_input.item(), t.total_flux_input.item()) == pytest.approx((2e-3, 3e-3))
