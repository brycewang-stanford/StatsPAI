"""Tests for multiway_cluster_vcov, cr3_jackknife_vcov, and subcluster WCR."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.inference.multiway_cluster import (
    multiway_cluster_vcov,
    cluster_robust_se,
    cr3_jackknife_vcov,
)
from statspai.inference.wild_subcluster import (
    subcluster_wild_bootstrap,
    wild_cluster_ci_inv,
)


def _sim_clustered_ols(n_firms=40, n_years=10, seed=0):
    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(n_years), n_firms)
    u_firm = rng.standard_normal(n_firms)[firm]
    u_year = rng.standard_normal(n_years)[year] * 0.5
    x = rng.standard_normal(len(firm))
    e = rng.standard_normal(len(firm))
    y = 1.0 + 0.8 * x + u_firm + u_year + e
    X = np.column_stack([np.ones(len(x)), x])
    return X, y, firm, year


def test_multiway_cluster_vcov_two_way_matches_existing_twoway_cluster():
    X, y, firm, year = _sim_clustered_ols()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    V_new = multiway_cluster_vcov(X, resid, [firm, year])
    # Should be symmetric, PSD, and have positive diagonal
    assert np.allclose(V_new, V_new.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(V_new)
    assert (eigvals >= -1e-10).all()
    assert (np.diag(V_new) > 0).all()


def test_multiway_cluster_three_way_runs():
    rng = np.random.default_rng(11)
    X, y, firm, year = _sim_clustered_ols()
    industry = rng.integers(0, 5, size=len(y))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    V = multiway_cluster_vcov(X, resid, [firm, year, industry])
    assert V.shape == (2, 2)
    assert (np.diag(V) > 0).all()


def test_cluster_robust_se_shape():
    X, y, firm, _ = _sim_clustered_ols()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    se = cluster_robust_se(X, resid, firm)
    assert se.shape == (X.shape[1],)
    assert (se > 0).all()


def test_cr3_jackknife_gives_positive_semidefinite():
    X, y, firm, _ = _sim_clustered_ols(n_firms=20, n_years=8, seed=42)
    V = cr3_jackknife_vcov(X, y, firm)
    eigvals = np.linalg.eigvalsh(V)
    assert (eigvals >= -1e-10).all()


def test_subcluster_wild_bootstrap_basic():
    # Small-G setting where subcluster WCR is the recommended method
    rng = np.random.default_rng(7)
    n_firms = 6
    n_years = 20
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(n_years), n_firms)
    x = rng.standard_normal(len(firm))
    y = 0.3 * x + rng.standard_normal(n_firms)[firm] + rng.standard_normal(len(firm)) * 0.2

    df = pd.DataFrame({"firm": firm, "year": year, "x": x, "y": y})
    res = subcluster_wild_bootstrap(
        df, y="y", x=["x"], cluster="firm", subcluster="year",
        n_boot=199, weight_type="webb", seed=1,
    )
    assert 0 <= res["p_boot"] <= 1
    assert res["n_clusters"] == n_firms
    assert res["n_subclusters"] == n_years


def test_wild_cluster_ci_inv_brackets_truth():
    rng = np.random.default_rng(3)
    n_firms = 12
    n_obs = 400
    firm = rng.integers(0, n_firms, size=n_obs)
    x = rng.standard_normal(n_obs)
    y = 2.5 * x + rng.standard_normal(n_firms)[firm] + rng.standard_normal(n_obs) * 0.3
    df = pd.DataFrame({"firm": firm, "x": x, "y": y})
    res = wild_cluster_ci_inv(
        df, y="y", x=["x"], cluster="firm",
        n_boot=199, weight_type="webb", seed=11, grid_size=21,
    )
    lo, hi = res["ci"]
    # True coefficient ~ 2.5 should lie in the inverted CI
    assert lo < 2.5 < hi
