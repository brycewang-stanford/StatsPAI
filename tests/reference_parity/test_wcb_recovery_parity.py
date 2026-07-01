"""Analytical parity: sp.wild_cluster_bootstrap / sp.subcluster_wild_bootstrap.

Both are seeded Rademacher-cluster bootstrap estimators (deterministic for a
fixed seed). Verified against:

  1. **OLS point-estimate recovery** on an unbiased linear DGP — the wild
     cluster bootstrap point estimate equals OLS ``beta`` exactly.
  2. **Deterministic self-consistency** under a fixed seed.

Seeded determinism lets us pin the entire bootstrap distribution exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def unbiased_dgp():
    rng = np.random.default_rng(2026)
    n = 600
    x = rng.normal(0, 1, n)
    d = rng.integers(0, 2, n)  # random (unbiased, no selection)
    y = 1.0 + 0.5 * d + 0.8 * x + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "d": d, "x": x, "clust": rng.integers(0, 30, n)})


@pytest.mark.parametrize("B", [99, 999])
def test_wcb_recovers_ols_coefficient(unbiased_dgp, B):
    res = sp.wild_cluster_bootstrap(
        unbiased_dgp, y="y", x=["d"], cluster="clust", n_boot=B, seed=42
    )
    # OLS coefficient on d
    X = np.column_stack([np.ones(len(unbiased_dgp)), unbiased_dgp["d"].values])
    ols = np.linalg.lstsq(X, unbiased_dgp["y"].values, rcond=None)[0][1]
    assert res["beta_hat"] == pytest.approx(ols, abs=1e-10)


@pytest.mark.parametrize("B", [99, 999])
def test_subcluster_wcb_recovers_ols_coefficient(unbiased_dgp, B):
    res = sp.subcluster_wild_bootstrap(
        unbiased_dgp, y="y", x=["d"], cluster="clust", n_boot=B, seed=42
    )
    X = np.column_stack([np.ones(len(unbiased_dgp)), unbiased_dgp["d"].values])
    ols = np.linalg.lstsq(X, unbiased_dgp["y"].values, rcond=None)[0][1]
    assert res["beta_hat"] == pytest.approx(ols, abs=1e-10)


def test_wcb_deterministic_under_seed(unbiased_dgp):
    r1 = sp.wild_cluster_bootstrap(
        unbiased_dgp, y="y", x=["d"], cluster="clust", n_boot=999, seed=42
    )
    r2 = sp.wild_cluster_bootstrap(
        unbiased_dgp, y="y", x=["d"], cluster="clust", n_boot=999, seed=42
    )
    assert r1["beta_hat"] == r2["beta_hat"]
    assert r1["t_stat"] == r2["t_stat"]
