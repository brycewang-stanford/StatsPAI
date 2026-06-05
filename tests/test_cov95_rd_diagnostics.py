"""Coverage tests for statspai.rd.diagnostics.

Exercises rdbwsensitivity, rdbalance, rdplacebo (with auto cutoffs and the
verbose print/plot paths) and the rdsummary battery with verbose printing,
full extended diagnostics, and the multi-panel plot. Real synthetic RD data.
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statspai as sp


def _make_sharp(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, n)
    Z = rng.normal(0, 1, n)
    Y = 0.5 * X + 3.0 * (X >= 0) + 0.3 * Z + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": Y, "x": X, "z": Z, "z2": rng.normal(0, 1, n)})


def test_rdbwsensitivity_grid():
    df = _make_sharp()
    out = sp.rdbwsensitivity(df, y="y", x="x", c=0, n_grid=6)
    assert isinstance(out, pd.DataFrame)
    assert (out["bandwidth"] > 0).all()
    assert ((out["pvalue"] >= 0) & (out["pvalue"] <= 1)).all()
    plt.close("all")


def test_rdbwsensitivity_explicit_grid():
    df = _make_sharp()
    out = sp.rdbwsensitivity(df, y="y", x="x", c=0,
                             bw_grid=[0.2, 0.4, 0.6])
    assert len(out) <= 3
    plt.close("all")


def test_rdbalance_default_and_explicit_covs():
    df = _make_sharp()
    out = sp.rdbalance(df, x="x", c=0, covs=["z", "z2"])
    assert set(["covariate", "estimate", "se", "pvalue"]).issubset(out.columns)
    # auto-detect covariates (all numeric except x)
    out2 = sp.rdbalance(df, x="x", c=0)
    assert len(out2) >= 1


def test_rdplacebo_auto_cutoffs():
    df = _make_sharp()
    out = sp.rdplacebo(df, y="y", x="x", c=0, n_placebo=8, side="both")
    assert "is_true_cutoff" in out.columns
    assert out["is_true_cutoff"].any()
    plt.close("all")


@pytest.mark.parametrize("side", ["left", "right", "both"])
def test_rdplacebo_sides(side):
    df = _make_sharp()
    out = sp.rdplacebo(df, y="y", x="x", c=0, n_placebo=6, side=side)
    assert isinstance(out, pd.DataFrame)
    plt.close("all")


def test_rdplacebo_explicit_cutoffs():
    df = _make_sharp()
    out = sp.rdplacebo(df, y="y", x="x", c=0,
                       placebo_cutoffs=[-0.5, -0.3, 0.3, 0.5])
    assert len(out) >= 1
    plt.close("all")


def test_rdsummary_verbose_basic():
    df = _make_sharp()
    res = sp.rdsummary(df, y="y", x="x", c=0, verbose=True)
    assert "estimate" in res
    assert "density_test" in res
    assert "bw_sensitivity" in res
    plt.close("all")


def test_rdsummary_with_covs_verbose():
    df = _make_sharp()
    res = sp.rdsummary(df, y="y", x="x", c=0, covs=["z", "z2"], verbose=True)
    assert res["balance"] is not None
    plt.close("all")


def test_rdsummary_full_with_plot():
    df = _make_sharp()
    res = sp.rdsummary(df, y="y", x="x", c=0, covs=["z"],
                       full=True, verbose=True, plot=True)
    assert "honest_ci" in res
    assert "power" in res
    assert "placebos" in res
    assert "bandwidth_comparison" in res
    assert "figure" in res
    plt.close("all")


def test_rdsummary_full_plot_no_covs_placebo_panel():
    # No covariates -> balance is None, so the diagnostic plot's 4th panel
    # falls through to the placebo-cutoff branch.
    df = _make_sharp()
    res = sp.rdsummary(df, y="y", x="x", c=0, full=True,
                       verbose=False, plot=True)
    assert res["balance"] is None
    assert "figure" in res
    plt.close("all")
