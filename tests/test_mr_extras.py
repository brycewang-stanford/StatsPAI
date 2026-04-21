"""Tests for MR mode-based estimator, F-statistic, and plot helpers."""

from __future__ import annotations

import numpy as np
import pytest

import statspai as sp


@pytest.fixture
def clean_mr_data():
    rng = np.random.default_rng(21)
    n_snps = 20
    bx = rng.uniform(0.05, 0.25, n_snps)
    sx = np.full(n_snps, 0.02)
    sy = np.full(n_snps, 0.03)
    true_beta = 0.5
    by = true_beta * bx + rng.normal(0, sy)
    return dict(bx=bx, by=by, sx=sx, sy=sy, true_beta=true_beta)


def test_mr_mode_weighted(clean_mr_data):
    d = clean_mr_data
    r = sp.mr_mode(d["bx"], d["by"], d["sx"], d["sy"],
                   method="weighted", n_boot=200, seed=1)
    assert abs(r.estimate - d["true_beta"]) < 0.3
    assert r.n_snps == len(d["bx"])
    assert r.bandwidth > 0
    assert r.ci[0] < r.estimate < r.ci[1]


def test_mr_mode_simple(clean_mr_data):
    d = clean_mr_data
    r = sp.mr_mode(d["bx"], d["by"], d["sx"], d["sy"],
                   method="simple", n_boot=200, seed=2)
    assert np.isfinite(r.estimate)
    assert "simple" in r.method


def test_mr_mode_rejects_invalid_method(clean_mr_data):
    d = clean_mr_data
    with pytest.raises(ValueError):
        sp.mr_mode(d["bx"], d["by"], d["sx"], d["sy"], method="bogus")


def test_mr_f_statistic_strong_instruments(clean_mr_data):
    d = clean_mr_data
    r = sp.mr_f_statistic(d["bx"], d["sx"])
    # With bx in [0.05, 0.25] and se = 0.02, F ≈ (bx/se)^2 = 6.25 ~ 156
    assert r.f_min > 0
    assert r.f_mean > r.f_min
    assert r.f_mean < r.f_max + 1e-9
    assert r.per_snp_F.shape == (len(d["bx"]),)


def test_mr_f_statistic_flags_weak():
    # Simulate a weak instrument
    bx = np.array([0.05, 0.03, 0.04])
    sx = np.array([0.05, 0.05, 0.05])  # F < 10
    r = sp.mr_f_statistic(bx, sx)
    assert r.weak_instrument_risk is True


def test_mr_f_statistic_summary_runs(clean_mr_data):
    d = clean_mr_data
    r = sp.mr_f_statistic(d["bx"], d["sx"])
    text = r.summary()
    assert "Mean per-SNP F" in text


def test_mr_funnel_plot_smoke(clean_mr_data):
    d = clean_mr_data
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    ax = sp.mr_funnel_plot(d["bx"], d["by"], d["sy"])
    assert ax is not None


def test_mr_scatter_plot_smoke(clean_mr_data):
    d = clean_mr_data
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    ax = sp.mr_scatter_plot(d["bx"], d["by"], d["sx"], d["sy"])
    assert ax is not None
