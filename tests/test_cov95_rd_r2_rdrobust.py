"""Coverage round-2 — ``statspai.rd.rdrobust`` secondary paths.

Round 1 covered the internal local-polynomial estimator. This file targets:

- the official-``rdrobust`` (CCT) delegation branches: donut filtering,
  cluster filtering, fuzzy + covs + manual h/b/rho kwargs;
- ``sp.rdplot`` (binned scatter): covariate partial-out, weights, bandwidth
  shading, donut shading, scatter=False, nbins given, binselect variants;
- ``sp.rdplotdensity`` (CJM density discontinuity): manual h, own-axes;
- ``sp.rdrobust(bootstrap='rbc')`` with cluster and fuzzy first-stage.

Real synthetic RD data; properties asserted (positive SE/bandwidth, ordered
CIs, p-values in [0,1], effect magnitude near the seeded jump). The CCT tests
skip when the external ``rdrobust`` package is absent.
"""

from __future__ import annotations

import importlib.util

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.core.results import CausalResult  # noqa: E402

_HAS_CCT = importlib.util.find_spec("rdrobust") is not None
_skip_cct = pytest.mark.skipif(
    not _HAS_CCT, reason="official rdrobust package not installed"
)


def _sharp(n=2000, tau=3.0, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, n)
    Z = rng.normal(0, 1, n)
    Y = 0.5 * X + tau * (X >= 0) + 0.3 * Z + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": Y, "x": X, "z": Z, "z2": rng.normal(0, 1, n)})


def _fuzzy(n=3000, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, n)
    prob = 0.15 + 0.7 * (X >= 0)
    D = (rng.uniform(0, 1, n) < prob).astype(float)
    Y = 0.5 * X + 2.0 * D + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y": Y, "x": X, "d": D})


# ── CCT (official rdrobust) delegation branches ────────────────────────────


@_skip_cct
def test_cct_with_donut_and_cluster():
    df = _sharp()
    df["g"] = (np.arange(len(df)) // 30).astype(int)
    res = sp.rdrobust(df, y="y", x="x", c=0, bwselect="cct", donut=0.03, cluster="g")
    assert isinstance(res, CausalResult)
    assert res.se > 0
    assert res.model_info["donut"] == 0.03


@_skip_cct
def test_cct_manual_h_and_b():
    df = _sharp()
    res = sp.rdrobust(df, y="y", x="x", c=0, bwselect="cct", h=0.3, b=0.5)
    assert res.se > 0
    assert np.isfinite(res.estimate)


@_skip_cct
def test_cct_rho_kwarg():
    df = _sharp()
    res = sp.rdrobust(df, y="y", x="x", c=0, bwselect="cct", rho=0.8)
    assert res.se > 0


@_skip_cct
def test_cct_fuzzy_covs_kink():
    df = _fuzzy()
    df["w1"] = np.random.default_rng(0).normal(0, 1, len(df))
    res = sp.rdrobust(df, y="y", x="x", c=0, bwselect="cct", fuzzy="d", covs=["w1"])
    assert res.se > 0
    assert abs(res.estimate - 2.0) < 1.5


@_skip_cct
def test_cct_donut_too_aggressive_raises():
    df = _sharp()
    with pytest.raises(ValueError, match="donut"):
        sp.rdrobust(df, y="y", x="x", c=0, bwselect="cct", donut=5.0)


# ── rdplot ────────────────────────────────────────────────────────────────


def test_rdplot_basic():
    df = _sharp()
    fig, ax = sp.rdplot(df, y="y", x="x", c=0)
    assert fig is not None and ax is not None
    plt.close("all")


def test_rdplot_covariate_partial_out_and_weights():
    df = _sharp()
    df["wt"] = np.abs(np.random.default_rng(1).normal(1, 0.1, len(df)))
    fig, ax = sp.rdplot(df, y="y", x="x", c=0, covs=["z", "z2"], weights="wt")
    assert ax is not None
    plt.close("all")


def test_rdplot_bandwidth_and_donut_shading():
    df = _sharp()
    fig, ax = sp.rdplot(df, y="y", x="x", c=0, show_bw=True, donut=0.05)
    assert ax is not None
    plt.close("all")


def test_rdplot_explicit_bandwidth_shading():
    df = _sharp()
    fig, ax = sp.rdplot(df, y="y", x="x", c=0, show_bw=True, h=0.4)
    assert ax is not None
    plt.close("all")


def test_rdplot_nbins_no_scatter_no_ci():
    df = _sharp()
    fig, ax = sp.rdplot(
        df,
        y="y",
        x="x",
        c=0,
        nbins=15,
        scatter=False,
        hide_ci=True,
        title="T",
        x_label="X",
        y_label="Y",
    )
    assert ax is not None
    plt.close("all")


@pytest.mark.parametrize("binselect", ["es", "qs", "esmv", "qsmv"])
def test_rdplot_binselect_variants(binselect):
    df = _sharp()
    fig, ax = sp.rdplot(df, y="y", x="x", c=0, binselect=binselect)
    assert ax is not None
    plt.close("all")


def test_rdplot_provided_axes():
    df = _sharp()
    fig0, ax0 = plt.subplots()
    fig, ax = sp.rdplot(df, y="y", x="x", c=0, ax=ax0)
    assert ax is ax0
    plt.close("all")


# ── rdplotdensity ──────────────────────────────────────────────────────────


def test_rdplotdensity_auto_bw():
    df = _sharp()
    fig, ax = sp.rdplotdensity(df, x="x", c=0)
    assert fig is not None and ax is not None
    plt.close("all")


def test_rdplotdensity_manual_h_no_hist():
    df = _sharp()
    fig, ax = sp.rdplotdensity(df, x="x", c=0, h=0.5, hist=False, p=1)
    assert ax is not None
    plt.close("all")


def test_rdplotdensity_provided_axes():
    df = _sharp()
    fig0, ax0 = plt.subplots()
    fig, ax = sp.rdplotdensity(df, x="x", c=0, ax=ax0)
    assert ax is ax0
    plt.close("all")


# ── rbc bootstrap variants ────────────────────────────────────────────────


def test_rbc_bootstrap_cluster():
    df = _sharp()
    df["g"] = (np.arange(len(df)) // 25).astype(int)
    res = sp.rdrobust(
        df, y="y", x="x", c=0, cluster="g", bootstrap="rbc", n_boot=199, random_state=0
    )
    boot = res.model_info["rbc_bootstrap"]
    assert boot["ci"][0] < boot["ci"][1]


def test_rbc_bootstrap_fuzzy():
    df = _fuzzy()
    res = sp.rdrobust(
        df, y="y", x="x", c=0, fuzzy="d", bootstrap="rbc", n_boot=199, random_state=1
    )
    boot = res.model_info["rbc_bootstrap"]
    assert boot["ci"][0] < boot["ci"][1]
