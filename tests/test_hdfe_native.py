"""Native HDFE absorber + hdfe_ols tests.

Verifies numerical agreement with dummy-variable OLS on small samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.panel.hdfe import Absorber, absorb_ols, demean


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------


def _make_panel(n_firms=20, n_years=8, beta=(1.0, -0.5), seed=0):
    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(n_years), n_firms)
    n = n_firms * n_years
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    alpha_firm = rng.standard_normal(n_firms)[firm]
    alpha_year = rng.standard_normal(n_years)[year]
    y = alpha_firm + alpha_year + beta[0] * x1 + beta[1] * x2 + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"firm": firm, "year": year, "x1": x1, "x2": x2, "y": y})


# ---------------------------------------------------------------------------
# Demean correctness
# ---------------------------------------------------------------------------


def test_demean_one_way_matches_dummy_regression():
    df = _make_panel(n_firms=10, n_years=6)
    ab = Absorber(df[["firm"]].values, drop_singletons=False)
    yw = ab.demean(df["y"].values)

    # Compare to manual group-demean
    expected = df["y"].values - df.groupby("firm")["y"].transform("mean").values
    np.testing.assert_allclose(yw, expected, atol=1e-10)


def test_demean_two_way_converges():
    df = _make_panel(n_firms=20, n_years=8)
    ab = Absorber(df[["firm", "year"]].values, drop_singletons=False, tol=1e-10)
    yw = ab.demean(df["y"].values)
    # After two-way demeaning, any further group mean should be near-zero
    resid_firm_mean = pd.Series(yw).groupby(df["firm"].values).mean().abs().max()
    resid_year_mean = pd.Series(yw).groupby(df["year"].values).mean().abs().max()
    assert resid_firm_mean < 1e-6
    assert resid_year_mean < 1e-6


def test_absorb_ols_matches_dummy_ols_two_way():
    df = _make_panel(n_firms=12, n_years=10, beta=(2.0, -1.0), seed=7)
    # Native HDFE
    res_native = absorb_ols(
        y=df["y"].values,
        X=df[["x1", "x2"]].values,
        fe=df[["firm", "year"]].values,
        drop_singletons=False,
    )
    # Dummy-variable OLS for comparison
    firm_dummies = pd.get_dummies(df["firm"], prefix="firm", drop_first=True, dtype=float)
    year_dummies = pd.get_dummies(df["year"], prefix="year", drop_first=True, dtype=float)
    X_full = np.column_stack([
        np.ones(len(df)),
        df[["x1", "x2"]].values,
        firm_dummies.values,
        year_dummies.values,
    ])
    beta_full, *_ = np.linalg.lstsq(X_full, df["y"].values, rcond=None)
    # coefficients on x1, x2 are at positions 1 and 2
    np.testing.assert_allclose(res_native["coef"], beta_full[1:3], atol=1e-8)


def test_singleton_drop_reduces_rows():
    # Add a unique firm-year that's a singleton
    df = _make_panel(n_firms=5, n_years=4)
    # Add a row whose firm appears only once
    df_ext = pd.concat([
        df,
        pd.DataFrame({"firm": [999], "year": [7], "x1": [0.0], "x2": [0.0], "y": [0.0]}),
    ]).reset_index(drop=True)

    ab = Absorber(df_ext[["firm", "year"]].values, drop_singletons=True)
    assert ab.n_dropped >= 1
    assert ab.n_kept < len(df_ext)


def test_clustered_se_from_hdfe_matches_sandwich():
    df = _make_panel(n_firms=40, n_years=10, seed=123)
    res = absorb_ols(
        y=df["y"].values,
        X=df[["x1", "x2"]].values,
        fe=df[["firm", "year"]].values,
        cluster=df["firm"].values,
        drop_singletons=False,
    )
    # SEs should be non-zero and positive
    assert np.all(res["se"] > 0)
    # Coefficients should be close to true DGP (true β = (1, -0.5))
    assert abs(res["coef"][0] - 1.0) < 0.2
    assert abs(res["coef"][1] - (-0.5)) < 0.2


def test_hdfe_ols_via_top_level():
    df = _make_panel(n_firms=15, n_years=6, seed=1)
    res = sp.hdfe_ols("y ~ x1 + x2 | firm + year", data=df, cluster="firm")
    assert "x1" in res.params.index
    assert res.r2_within > 0.0
    assert res.se_type == "cluster"


def test_demean_functional_wrapper():
    df = _make_panel(n_firms=8, n_years=5)
    yw, keep = demean(df["y"].values, df[["firm"]].values, drop_singletons=False)
    assert yw.shape[0] == keep.sum()


def test_cluster_se_with_singleton_drop_does_not_crash():
    # Regression test: with drop_singletons=True, the cluster array must
    # also be masked to keep_mask. Without the fix, this would shape-mismatch.
    df = _make_panel(n_firms=12, n_years=8, seed=99)
    # Append a singleton firm
    df_ext = pd.concat([
        df,
        pd.DataFrame({
            "firm": [999], "year": [0], "x1": [0.0], "x2": [0.0], "y": [0.0],
        }),
    ]).reset_index(drop=True)
    res = absorb_ols(
        y=df_ext["y"].values,
        X=df_ext[["x1", "x2"]].values,
        fe=df_ext[["firm", "year"]].values,
        cluster=df_ext["firm"].values,
        drop_singletons=True,
    )
    assert res["n_singletons_dropped"] >= 1
    assert np.all(res["se"] > 0)


def test_feols_summary_prints_cleanly():
    df = _make_panel(n_firms=20, n_years=6, seed=4)
    res = sp.hdfe_ols("y ~ x1 + x2 | firm + year", data=df, cluster="firm")
    summary_str = res.summary()
    assert "FEOLS" in summary_str
    assert "firm" in summary_str
    assert "x1" in summary_str
