"""Tests for overlap_weights and CBPS."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.matching.overlap_weights import overlap_weights
from statspai.matching.cbps import cbps


def _sim_obs(n=500, true_ate=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    logit = 0.5 * X[:, 0] - 0.3 * X[:, 1]
    ps = 1 / (1 + np.exp(-logit))
    T = (rng.uniform(size=n) < ps).astype(int)
    Y = true_ate * T + 0.8 * X[:, 0] + 0.2 * X[:, 2] + rng.standard_normal(n) * 0.5
    return pd.DataFrame({
        "y": Y, "t": T, "x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2],
    })


# --------------------------------------------------------------------- #
# Overlap weights
# --------------------------------------------------------------------- #


def test_overlap_weights_close_to_true_ate():
    df = _sim_obs(n=800, true_ate=1.0, seed=0)
    res = overlap_weights(
        df, y="y", treat="t", covariates=["x1", "x2", "x3"],
        estimand="ATO", n_bootstrap=100, seed=1,
    )
    # ATO is not exactly ATE, but under near-linear DGP it's close
    assert abs(res.estimate - 1.0) < 0.4
    assert res.se > 0


def test_overlap_weights_ate_variant():
    df = _sim_obs(n=500, true_ate=0.5, seed=11)
    res = overlap_weights(
        df, y="y", treat="t", covariates=["x1", "x2", "x3"],
        estimand="ATE", n_bootstrap=100, seed=2,
    )
    assert abs(res.estimate - 0.5) < 0.5


def test_overlap_weights_tilt_matching():
    df = _sim_obs(n=400, seed=3)
    res = overlap_weights(
        df, y="y", treat="t", covariates=["x1", "x2"],
        estimand="MATCHING", n_bootstrap=50, seed=5,
    )
    ess = res.model_info["effective_sample_size"]
    assert ess > 0


# --------------------------------------------------------------------- #
# CBPS
# --------------------------------------------------------------------- #


def test_cbps_exact_variant_recovers_ate():
    df = _sim_obs(n=800, true_ate=1.0, seed=0)
    res = cbps(
        df, y="y", treat="t", covariates=["x1", "x2", "x3"],
        variant="exact", estimand="ATE", n_bootstrap=50, seed=1,
    )
    assert abs(res.estimate - 1.0) < 0.4


def test_cbps_over_variant_runs():
    df = _sim_obs(n=500, true_ate=0.8, seed=2)
    res = cbps(
        df, y="y", treat="t", covariates=["x1", "x2"],
        variant="over", estimand="ATE", n_bootstrap=30, seed=4,
    )
    assert res.estimate is not None
    assert res.model_info["model_type"].startswith("CBPS")


def test_cbps_att_variant():
    df = _sim_obs(n=500, true_ate=0.6, seed=7)
    res = cbps(
        df, y="y", treat="t", covariates=["x1", "x2", "x3"],
        variant="exact", estimand="ATT", n_bootstrap=30, seed=3,
    )
    # ATT tends to be close to ATE in a linear DGP
    assert abs(res.estimate - 0.6) < 0.6


def test_cbps_balance_diagnostics_present():
    df = _sim_obs(n=400, seed=5)
    res = cbps(
        df, y="y", treat="t", covariates=["x1", "x2"],
        variant="exact", n_bootstrap=20, seed=9,
    )
    smd = res.model_info["std_mean_diff_after"]
    assert isinstance(smd, dict)
    # Balance should be near-zero after CBPS weighting
    for var, val in smd.items():
        if var == "_intercept":
            continue
        assert abs(val) < 0.2
