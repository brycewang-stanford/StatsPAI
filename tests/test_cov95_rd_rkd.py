"""Coverage tests for statspai.rd.rkd (Regression Kink Design).

Sharp (reduced-form) and fuzzy RKD, kernels, polynomial orders, clustering,
auto/manual bandwidth, summary/detail/plot, and error paths. Real synthetic
kink data; properties asserted (kink magnitude near truth, SE positivity).
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")

import statspai as sp


def _make_sharp_kink(n=2500, slope_change=0.8, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, n)
    Y = 0.5 * X + slope_change * np.maximum(X, 0) + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y": Y, "x": X})


def _make_fuzzy_kink(n=3000, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, n)
    T = 1.0 * X + 2.0 * np.maximum(X, 0) + rng.normal(0, 0.3, n)
    Y = 0.4 * T + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": Y, "x": X, "treat": T})


def test_rkd_sharp_properties():
    df = _make_sharp_kink()
    # auto-bandwidth: property checks only (RKD derivative estimation noisy)
    res = sp.rkd(df, y="y", x="x", c=0)
    assert res.se > 0
    assert 0.0 <= res.pvalue <= 1.0
    assert res.model_info["design"] == "Sharp (Reduced-Form)"
    assert res.model_info["bandwidth"] > 0
    assert res.model_info["n_left"] > 0 and res.model_info["n_right"] > 0
    assert np.isfinite(res.estimate)
    # with a reasonable manual bandwidth the kink (slope change) is recovered
    res_h = sp.rkd(df, y="y", x="x", c=0, h=1.5)
    assert abs(res_h.estimate - 0.8) < 0.4


def test_rkd_fuzzy_ratio():
    df = _make_fuzzy_kink()
    res = sp.rkd(df, y="y", x="x", c=0, treatment="treat", h=1.5)
    assert res.model_info["design"] == "Fuzzy"
    assert res.se > 0
    assert "kink_treatment" in res.model_info
    # LATE ~ 0.4
    assert abs(res.estimate - 0.4) < 0.4


@pytest.mark.parametrize("kernel", ["triangular", "epanechnikov", "uniform"])
def test_rkd_kernels(kernel):
    df = _make_sharp_kink()
    res = sp.rkd(df, y="y", x="x", c=0, kernel=kernel, h=1.0)
    assert res.se > 0
    assert res.model_info["kernel"] == kernel


def test_rkd_manual_bandwidth_and_order2():
    df = _make_sharp_kink()
    res = sp.rkd(df, y="y", x="x", c=0, h=1.2, p=2)
    assert res.model_info["bandwidth"] == pytest.approx(1.2)
    assert res.model_info["polynomial_order"] == 2


def test_rkd_cluster_sharp_and_fuzzy():
    df = _make_fuzzy_kink()
    df["g"] = (np.arange(len(df)) // 30).astype(int)
    sharp = sp.rkd(df, y="y", x="x", c=0, cluster="g", h=1.0)
    fuzzy = sp.rkd(df, y="y", x="x", c=0, treatment="treat", cluster="g", h=1.0)
    assert sharp.se > 0 and fuzzy.se > 0


def test_rkd_summary_and_detail():
    df = _make_sharp_kink()
    res = sp.rkd(df, y="y", x="x", c=0)
    txt = res.summary()
    assert "Regression Kink Design" in txt
    assert res.detail is not None
    assert "Kink (outcome)" in res.detail["term"].values
    # fuzzy summary path
    dff = _make_fuzzy_kink()
    rf = sp.rkd(dff, y="y", x="x", c=0, treatment="treat")
    txt_f = rf.summary()
    assert "First stage kink" in txt_f
    assert "Kink (treatment)" in rf.detail["term"].values


def test_rkd_plot_returns_figure():
    df = _make_sharp_kink()
    res = sp.rkd(df, y="y", x="x", c=0)
    fig = res.plot(show=False)
    assert fig is not None


def test_rkd_errors():
    df = _make_sharp_kink()
    with pytest.raises(ValueError, match="kernel"):
        sp.rkd(df, y="y", x="x", c=0, kernel="bad")
    with pytest.raises(ValueError, match="p must be"):
        sp.rkd(df, y="y", x="x", c=0, p=0)
    with pytest.raises(ValueError, match="Too few"):
        sp.rkd(df.head(10), y="y", x="x", c=0)


def test_rkd_insufficient_within_bandwidth():
    df = _make_sharp_kink()
    with pytest.raises(ValueError, match="Insufficient"):
        sp.rkd(df, y="y", x="x", c=0, h=1e-4)


def test_rkd_zero_first_stage_kink_raises():
    # Treatment with no kink (pure linear) -> fuzzy denominator ~ 0
    rng = np.random.default_rng(3)
    n = 2000
    X = rng.uniform(-2, 2, n)
    T = 1.0 * X + rng.normal(0, 0.01, n)  # no slope change at 0
    Y = 0.4 * T + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": Y, "x": X, "treat": T})
    # may raise on zero kink; if it estimates, SE must still be finite
    try:
        res = sp.rkd(df, y="y", x="x", c=0, treatment="treat", h=1.5)
        assert np.isfinite(res.estimate)
    except ValueError as e:
        assert "first-stage kink" in str(e).lower() or "zero" in str(e).lower()
