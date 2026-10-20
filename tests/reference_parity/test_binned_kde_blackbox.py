"""Black-box parity of the binned Gaussian KDE against R.

``statspai.core._binned_kde`` is a clean-room implementation (spec:
``docs/dev/cleanroom/binned_kde.md``). The fixture was written by running
R's ``bw.nrd0`` / ``density()`` and ``dineq::rif`` as black boxes
(``_fixtures/_generate_binned_kde_blackbox_R.R``).

Tolerances: bandwidths 1e-13 relative (closed form, only rounding);
densities 1e-12 relative to the peak (an FFT convolution on both sides,
observed about 1e-14); the RIF density 1e-12 relative (observed 3e-15).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from statspai.core._binned_kde import (
    binned_density_at,
    binned_gaussian_kde,
    silverman_bandwidth,
)
from statspai.decomposition._common import (
    _bw_nrd0,
    influence_function,
    kde_at_dineq,
    weighted_quantile_hmisc,
)
from statspai.regression.interflex import r_density

FIX = json.loads(
    (Path(__file__).parent / "_fixtures" / "binned_kde_blackbox_R.json").read_text(
        encoding="utf-8"
    )
)
DENS = FIX["density"]
RIF = FIX["rif"]


def _thin(v):
    return np.asarray(v)[::7]


def _rel_to_peak(a, b):
    b = np.asarray(b, float)
    return float(np.max(np.abs(np.asarray(a, float) - b)) / np.max(b))


@pytest.mark.parametrize("name", sorted(DENS))
def test_silverman_bandwidth_matches_r(name):
    rec = DENS[name]
    x = np.asarray(rec["x"], float)
    assert silverman_bandwidth(x) == pytest.approx(rec["bw"], rel=1e-13)
    assert _bw_nrd0(x) == pytest.approx(rec["bw"], rel=1e-13)


@pytest.mark.parametrize("name", sorted(DENS))
def test_binned_density_matches_r(name):
    rec = DENS[name]
    x = np.asarray(rec["x"], float)
    pts, y, bw = r_density(x)
    assert bw == pytest.approx(rec["bw"], rel=1e-13)
    np.testing.assert_allclose(
        _thin(pts), rec["x512"], rtol=0, atol=1e-12 * (1 + np.ptp(x))
    )
    assert _rel_to_peak(_thin(y), rec["y512"]) < 1e-12
    for nn in (100, 1000):
        pts, y, _ = r_density(x, n_user=nn)
        assert len(pts) == nn
        np.testing.assert_allclose(
            _thin(pts), rec[f"x{nn}"], rtol=0, atol=1e-12 * (1 + np.ptp(x))
        )
        assert _rel_to_peak(_thin(y), rec[f"y{nn}"]) < 1e-12
    assert _rel_to_peak(_thin(r_density(x, cut=1.0)[1]), rec["y_cut1"]) < 1e-12
    assert _rel_to_peak(_thin(r_density(x, bw=0.37)[1]), rec["y_bw037"]) < 1e-12
    _, yw, _ = binned_gaussian_kde(x, weights=np.asarray(rec["w"], float))
    assert _rel_to_peak(_thin(yw), rec["y_w"]) < 1e-12


@pytest.mark.parametrize("name", sorted(RIF))
def test_density_at_quantile_matches_dineq(name):
    rec = RIF[name]
    x = np.asarray(rec["x"], float)
    w = None if rec["w"] is None else np.asarray(rec["w"], float)
    tau = float(rec["tau"])
    rif_r = np.asarray(rec["rif"], float)
    q = float(weighted_quantile_hmisc(x, tau, w=w))
    # dineq's RIF is q + (tau - 1{y < q}) / f(q); recover f(q) from it.
    f_r = (tau - (x < q)) / (rif_r - q)
    assert np.ptp(f_r) / np.median(f_r) < 1e-12
    assert kde_at_dineq(x, q, w=w) == pytest.approx(float(np.median(f_r)), rel=1e-12)
    np.testing.assert_allclose(
        influence_function(x, "quantile", tau=tau, w=w, quantile_convention="dineq"),
        rif_r,
        rtol=1e-12,
    )


def test_density_integrates_to_one_and_mass_outside_window_is_dropped():
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    pts, y, _ = binned_gaussian_kde(x, n=2048, cut=8.0)
    assert np.trapezoid(y, pts) == pytest.approx(1.0, abs=1e-6)
    # A far-away cluster cannot reach a point more than ext*bw from it.
    far = np.r_[x, np.full(400, 1e3)]
    masses = np.full(far.size, 1.0 / far.size)
    bw = silverman_bandwidth(x)
    near = binned_density_at(far, 0.0, weights=masses, bw=bw)
    alone = binned_density_at(x, 0.0, weights=masses[:400], bw=bw)
    assert near == pytest.approx(alone, rel=1e-12)


def test_bandwidth_fallbacks():
    # Coinciding quartiles: the standard deviation is used.
    x = np.r_[-20.0, np.zeros(98), 20.0]
    assert silverman_bandwidth(x) == pytest.approx(
        0.9 * np.std(x, ddof=1) * 100**-0.2, rel=1e-14
    )
    # Constant data: |x[0]|, then 1.
    assert silverman_bandwidth(np.full(10, -3.0)) == pytest.approx(0.9 * 3 * 10**-0.2)
    assert silverman_bandwidth(np.zeros(5)) == pytest.approx(0.9 * 5**-0.2)
    # One observation has no spread.
    assert silverman_bandwidth(np.array([2.0])) == pytest.approx(1.8)
