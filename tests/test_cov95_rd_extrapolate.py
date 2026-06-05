"""Coverage tests for statspai.rd.extrapolate.

Covers Angrist-Rokkanen extrapolation (ols / ipw / doubly_robust),
fuzzy variant, explicit eval_points, multi-cutoff extrapolation
(linear / polynomial / weighted), external validity diagnostics with
covariate overlap, and the extrapolation plot helper. Properties asserted.
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statspai as sp
from statspai.rd.extrapolate import _extrapolation_plot
from statspai.core.results import CausalResult


def _make_ar_data(n=2000, tau=3.0, seed=42):
    """Conditional-independence data: Y depends on Z, X via Z."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, n)
    X = Z + rng.normal(0, 0.5, n)
    D = (X >= 0).astype(int)
    Y = 1.0 + 2.0 * Z + tau * D + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": Y, "x": X, "z": Z, "d": D})


def _make_multi_cutoff(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 4, n)
    tau_true = 2.0 + 0.5 * X
    D = ((X >= 1) | (X >= 3)).astype(int)
    Y = 0.5 * X + tau_true * D + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": Y, "x": X})


@pytest.mark.parametrize("method", ["ols", "ipw", "doubly_robust"])
def test_rd_extrapolate_methods(method):
    df = _make_ar_data()
    res = sp.rd_extrapolate(df, y="y", x="x", c=0, covs=["z"],
                            n_eval=6, method=method)
    assert isinstance(res, CausalResult)
    assert res.detail is not None
    assert set(["x_value", "cate", "se", "ci_lower", "ci_upper"]).issubset(
        res.detail.columns)
    ci = res.model_info["conditional_independence_test"]
    assert 0.0 <= ci["control_side"]["p_value"] <= 1.0
    assert 0.0 <= ci["treated_side"]["p_value"] <= 1.0
    assert "local_rd_estimate" in res.model_info


def test_rd_extrapolate_explicit_eval_points_and_fuzzy():
    df = _make_ar_data()
    eval_pts = np.linspace(-1.5, 1.5, 8)
    res = sp.rd_extrapolate(df, y="y", x="x", c=0, covs=["z"],
                            treatment="d", eval_points=eval_pts, method="ols")
    assert len(res.detail) == 8
    np.testing.assert_allclose(res.detail["x_value"].values, eval_pts)


def test_rd_extrapolate_errors():
    df = _make_ar_data()
    with pytest.raises(ValueError, match="covs is required"):
        sp.rd_extrapolate(df, y="y", x="x", c=0, covs=None)
    with pytest.raises(ValueError, match="method must be"):
        sp.rd_extrapolate(df, y="y", x="x", c=0, covs=["z"], method="bad")


def test_rd_extrapolate_insufficient_one_side():
    # All observations on one side -> control or treated < 5
    rng = np.random.default_rng(0)
    n = 60
    X = rng.uniform(0.5, 2, n)  # all >= 0 -> no controls
    Z = rng.normal(0, 1, n)
    Y = Z + 3 * (X >= 0) + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": Y, "x": X, "z": Z})
    # one side is empty: either the local rdrobust or the extrapolate-specific
    # guard raises a ValueError.
    with pytest.raises(ValueError):
        sp.rd_extrapolate(df, y="y", x="x", c=0, covs=["z"], n_eval=4)


@pytest.mark.parametrize("method", ["linear", "polynomial", "weighted"])
def test_rd_multi_extrapolate_methods(method):
    df = _make_multi_cutoff()
    res = sp.rd_multi_extrapolate(df, y="y", x="x", cutoffs=[1.0, 3.0],
                                  method=method,
                                  eval_points=np.linspace(0, 4, 6))
    assert isinstance(res, CausalResult)
    assert "cate_extrapolated" in res.detail.columns
    assert res.model_info["n_cutoffs"] >= 2
    het = res.model_info["heterogeneity_test"]
    assert het is None or 0.0 <= het["p_value"] <= 1.0


def test_rd_multi_extrapolate_polynomial_three_cutoffs():
    df = _make_multi_cutoff()
    res = sp.rd_multi_extrapolate(df, y="y", x="x", cutoffs=[0.0, 1.0, 3.0],
                                  method="polynomial")
    assert res.model_info["degree"] >= 1
    assert len(res.model_info["coefficients"]) == res.model_info["degree"] + 1


def test_rd_multi_extrapolate_errors():
    df = _make_multi_cutoff()
    with pytest.raises(ValueError, match="At least 2 cutoffs"):
        sp.rd_multi_extrapolate(df, y="y", x="x", cutoffs=[1.0])
    with pytest.raises(ValueError, match="method must be"):
        sp.rd_multi_extrapolate(df, y="y", x="x", cutoffs=[1.0, 3.0], method="bad")


def test_rd_external_validity_with_covs():
    df = _make_ar_data()
    diag = sp.rd_external_validity(df, y="y", x="x", c=0, covs=["z"],
                                   target_x_range=(-2.0, 2.0))
    assert "recommendation" in diag
    assert diag["local_estimate"] is not None
    assert diag["ci_test"] is not None
    # overlap diagnostics should be computed
    assert diag["overlap"] is not None
    cov_diag = diag["overlap"]["covariate_diagnostics"]
    assert "z" in cov_diag
    assert 0.0 <= cov_diag["z"]["overlap_coefficient"] <= 1.0
    assert 0.0 <= cov_diag["z"]["ks_pvalue"] <= 1.0


def test_rd_external_validity_ci_holds_triggers_extrapolation():
    # Y has no direct X effect given Z -> conditional independence holds,
    # so the extrapolated estimate branch executes.
    rng = np.random.default_rng(1)
    n = 2500
    Z = rng.normal(0, 1, n)
    X = Z + rng.normal(0, 1.0, n)
    D = (X >= 0).astype(int)
    Y = 1.0 + 2.0 * Z + 3.0 * D + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": Y, "x": X, "z": Z})
    diag = sp.rd_external_validity(df, y="y", x="x", c=0, covs=["z"])
    assert diag["ci_test"]["ci_holds"] is True
    assert diag["extrapolated_estimate"] is not None
    assert "RECOMMENDATION" in diag["recommendation"]


def test_rd_external_validity_no_covs():
    df = _make_ar_data()
    diag = sp.rd_external_validity(df, y="y", x="x", c=0, covs=None)
    assert diag["ci_test"] is None
    assert diag["overlap"] is None
    assert "not recommended" in diag["recommendation"].lower() or \
        "cannot be tested" in diag["recommendation"].lower()


def test_extrapolation_plot_single_cutoff():
    df = _make_ar_data()
    res = sp.rd_extrapolate(df, y="y", x="x", c=0, covs=["z"],
                            n_eval=10, method="ols")
    ax = _extrapolation_plot(res)
    assert ax is not None
    plt.close("all")


def test_extrapolation_plot_multi_cutoff():
    df = _make_multi_cutoff()
    res = sp.rd_multi_extrapolate(df, y="y", x="x", cutoffs=[1.0, 3.0],
                                  method="linear")
    fig, ax = plt.subplots()
    out = _extrapolation_plot(res, ax=ax, show_ci=True)
    assert out is not None
    plt.close("all")
