"""Coverage tests for statspai.rd.diagnostics.

Exercises rdbwsensitivity, rdbalance, rdplacebo (with auto cutoffs and the
verbose print/plot paths) and the rdsummary battery with verbose printing,
full extended diagnostics, and the multi-panel plot. Real synthetic RD data.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

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
    # Bandwidth grid and point estimates are reproducible across platforms;
    # the bias-corrected SE depends on a data-driven pilot bandwidth whose
    # discrete selection can flip under different BLAS backends (macOS
    # Accelerate vs Linux OpenBLAS), so it is checked only for validity, not
    # pinned to the dB. Numerical parity is guarded by tests/reference_parity/.
    head = out[["bandwidth", "estimate", "se"]].head(3).to_numpy()
    np.testing.assert_allclose(
        head[:, :2],
        # Re-pinned after the WP-1 CCT bandwidth fix. These are REGRESSION
        # GUARDS, not correctness evidence: rdbwsensitivity has no R
        # counterpart. What is verified is the input -- sp.rdrobust's h, b and
        # conventional estimate now match rdrobust 4.0.0 across a 36-cell grid
        # (tests/reference_parity/test_rdrobust_parity.py).
        np.array([[0.177517, 2.80895], [0.284027, 2.886383], [0.390537, 2.925874]]),
        rtol=1e-4,
        atol=1e-6,
    )
    assert np.isfinite(head[:, 2]).all() and (head[:, 2] > 0).all()
    plt.close("all")


def test_rdbwsensitivity_explicit_grid():
    df = _make_sharp()
    out = sp.rdbwsensitivity(df, y="y", x="x", c=0, bw_grid=[0.2, 0.4, 0.6])
    assert len(out) <= 3
    plt.close("all")


def test_rdbalance_default_and_explicit_covs():
    df = _make_sharp()
    out = sp.rdbalance(df, x="x", c=0, covs=["z", "z2"])
    assert set(["covariate", "estimate", "se", "pvalue"]).issubset(out.columns)
    # Balance point estimates are platform-stable; the SE/p-value depend on a
    # data-driven bandwidth that can flip under different BLAS backends, so
    # they are checked for validity rather than pinned. Parity is guarded by
    # tests/reference_parity/.
    vals = out[["estimate", "se", "pvalue"]].to_numpy()
    np.testing.assert_allclose(
        # Re-pinned after the WP-1 CCT bandwidth fix. REGRESSION GUARD, not
        # correctness evidence: this diagnostic has no R counterpart. Its
        # input is verified -- sp.rdrobust's h/b/conventional now match
        # rdrobust 4.0.0 across a 36-cell grid (reference_parity).
        vals[:, 0],
        np.array([-0.041551, -0.096642]),
        rtol=1e-4,
        atol=1e-6,
    )
    assert np.isfinite(vals[:, 1]).all() and (vals[:, 1] > 0).all()
    assert ((vals[:, 2] >= 0) & (vals[:, 2] <= 1)).all()
    # auto-detect covariates (all numeric except x)
    out2 = sp.rdbalance(df, x="x", c=0)
    assert len(out2) >= 1


def test_rdplacebo_auto_cutoffs():
    df = _make_sharp()
    out = sp.rdplacebo(df, y="y", x="x", c=0, n_placebo=8, side="both")
    assert "is_true_cutoff" in out.columns
    assert out["is_true_cutoff"].any()
    true = out.loc[out["is_true_cutoff"]].iloc[0]
    # Cutoff and point estimate are platform-stable; the SE depends on a
    # data-driven bandwidth that can flip under different BLAS backends, so it
    # is checked for validity rather than pinned. Parity is guarded by
    # tests/reference_parity/.
    np.testing.assert_allclose(
        # Re-pinned after the WP-1 CCT bandwidth fix. REGRESSION GUARD, not
        # correctness evidence: this diagnostic has no R counterpart. Its
        # input is verified -- sp.rdrobust's h/b/conventional now match
        # rdrobust 4.0.0 across a 36-cell grid (reference_parity).
        [true["cutoff"], true["estimate"]],
        [0.0, 2.924306],
        rtol=1e-4,
        atol=1e-6,
    )
    assert np.isfinite(true["se"]) and true["se"] > 0
    assert 0.0 <= true["pvalue"] <= 1.0
    plt.close("all")


@pytest.mark.parametrize("side", ["left", "right", "both"])
def test_rdplacebo_sides(side):
    df = _make_sharp()
    out = sp.rdplacebo(df, y="y", x="x", c=0, n_placebo=6, side=side)
    assert isinstance(out, pd.DataFrame)
    plt.close("all")


def test_rdplacebo_explicit_cutoffs():
    df = _make_sharp()
    out = sp.rdplacebo(df, y="y", x="x", c=0, placebo_cutoffs=[-0.5, -0.3, 0.3, 0.5])
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
    res = sp.rdsummary(
        df, y="y", x="x", c=0, covs=["z"], full=True, verbose=True, plot=True
    )
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
    res = sp.rdsummary(df, y="y", x="x", c=0, full=True, verbose=False, plot=True)
    assert res["balance"] is None
    assert "figure" in res
    plt.close("all")
