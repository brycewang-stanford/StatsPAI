"""Tests for GRF-style causal forest inference (test_calibration, rate,
honest_variance)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.forest.forest_inference import calibration_test, rate, honest_variance


def _sim_hte(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    # Heterogeneous treatment effect: τ(x) = 1 + 2 * x1
    tau = 1.0 + 2.0 * X[:, 0]
    T = rng.integers(0, 2, size=n)
    Y0 = X[:, 1] + rng.standard_normal(n) * 0.5
    Y1 = Y0 + tau
    Y = np.where(T == 1, Y1, Y0)
    return X, T, Y


def _fit_forest():
    X, T, Y = _sim_hte(n=400, seed=1)
    cf = sp.causal_forest(Y=Y, T=T, X=X, n_estimators=50, random_state=42)
    return cf, X, T, Y


def test_calibration_returns_dataframe():
    cf, X, T, Y = _fit_forest()
    out = calibration_test(cf, X=X, Y=Y, T=T)
    assert isinstance(out, pd.DataFrame)
    assert "coef" in out.columns
    assert "se" in out.columns
    assert "p" in out.columns
    assert len(out) == 2
    # Differential forest prediction coefficient should be positive
    # (forest captures real heterogeneity)
    diff = out.loc["differential_forest_prediction"]
    assert np.isfinite(diff["coef"])


def test_rate_returns_autoc_estimate():
    cf, X, T, Y = _fit_forest()
    out = rate(cf, X=X, Y=Y, T=T, target="AUTOC", q_grid=50, seed=3)
    assert "estimate" in out
    assert "se" in out
    assert "ci_low" in out
    assert "toc_curve" in out
    assert out["toc_curve"].shape[1] == 2


def test_rate_qini_variant_runs():
    cf, X, T, Y = _fit_forest()
    out = rate(cf, X=X, Y=Y, T=T, target="QINI", q_grid=30, seed=4)
    assert np.isfinite(out["estimate"])


def test_honest_variance_reports_ci():
    cf, X, _, _ = _fit_forest()
    out = honest_variance(cf, X=X, n_splits=10, seed=0)
    assert "ate" in out
    assert "ci_low" in out
    assert "ci_high" in out
    assert out["ci_low"] <= out["ate"] <= out["ci_high"]
