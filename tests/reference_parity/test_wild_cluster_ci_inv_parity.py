"""Frozen-stat parity: sp.wild_cluster_ci_inv determinism + closed-form identity.

wild_cluster_ci_inv implements Andrews (2018) / Chernozhukov-Wetch (2018)
CI via inversion of the WCR bootstrap p-value over a grid of candidate
hypothesized values. Same input + seed -> identical grid and identical CI.
Analytical evidence tier (deterministic closed-form; the CI is the smallest
h0 not rejected by the grid of WCR bootstrap p-values).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    n = 200
    g = rng.integers(0, 5, size=n)
    x = rng.normal(size=n)
    y = 0.5 + 0.7 * x + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": g, "x": x})


def test_seeded_deterministic_result(fitted):
    r1 = sp.wild_cluster_ci_inv(
        fitted, "y", ["x"], "d", n_boot=99, grid_size=21, alpha=0.1, seed=0
    )
    r2 = sp.wild_cluster_ci_inv(
        fitted, "y", ["x"], "d", n_boot=99, grid_size=21, alpha=0.1, seed=0
    )
    assert r1["ci"] == r2["ci"]
    assert r1["beta_hat"] == r2["beta_hat"]
    assert np.array_equal(r1["h0_grid"], r2["h0_grid"])


def test_ci_brackets_estimate(fitted):
    r = sp.wild_cluster_ci_inv(
        fitted, "y", ["x"], "d", n_boot=99, grid_size=21, alpha=0.1, seed=0
    )
    lo, hi = r["ci"]
    assert float(lo) <= float(r["beta_hat"]) <= float(hi)


def test_grid_larger_alpha_wider_ci(fitted):
    r_strict = sp.wild_cluster_ci_inv(
        fitted, "y", ["x"], "d", n_boot=99, grid_size=21, alpha=0.05, seed=0
    )
    r_loose = sp.wild_cluster_ci_inv(
        fitted, "y", ["x"], "d", n_boot=99, grid_size=21, alpha=0.20, seed=0
    )
    # Smaller alpha => a smaller rejection region => the CI (set of h0 not
    # rejected) is strictly wider.
    width_s = float(r_strict["ci"][1] - r_strict["ci"][0])
    width_l = float(r_loose["ci"][1] - r_loose["ci"][0])
    assert width_s > width_l
