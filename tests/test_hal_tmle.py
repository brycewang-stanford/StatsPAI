"""Tests for HAL-TMLE (Qian-van der Laan 2025)."""

import warnings
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp


def _tmle_data(n=300, p=4, tau=1.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    ps = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    A = (rng.uniform(size=n) < ps).astype(int)
    Y = 2.0 + tau * A + 0.5 * X[:, 0] + X[:, 1] ** 2 + rng.normal(0, 1.0, n)
    return pd.DataFrame({"y": Y, "a": A, **{f"x{j}": X[:, j] for j in range(p)}})


def test_hal_tmle_ate_recovery():
    df = _tmle_data(n=400, seed=0)
    r = sp.hal_tmle(
        df, y="y", treat="a",
        covariates=[f"x{j}" for j in range(4)],
        variant="delta", max_anchors_per_col=15, n_folds=3,
    )
    assert abs(r.estimate - 1.5) < 0.5
    assert r.se > 0
    assert r.ci[0] < r.estimate < r.ci[1]


def test_hal_tmle_delta_variant_produces_result():
    df = _tmle_data(n=200, seed=1)
    cov = [f"x{j}" for j in range(4)]
    r = sp.hal_tmle(df, y="y", treat="a", covariates=cov,
                    variant="delta", max_anchors_per_col=10, n_folds=3)
    assert r.model_info.get("variant") == "delta"
    assert np.isfinite(r.estimate)


def test_hal_tmle_projection_variant_raises_notimplemented():
    # The projection variant shipped a no-op shrinkage in v1.11.x and
    # earlier (the formula post-multiplied ``model_info["eps"]`` after
    # ``result.estimate`` was already computed → variant flag did
    # nothing to the point estimate). v1.11.5 surfaces this honestly
    # by raising NotImplementedError until the proper Riesz-projection
    # step (Li-Qiu-Wang-vdL 2025 §3.2) is ported.
    df = _tmle_data(n=200, seed=1)
    cov = [f"x{j}" for j in range(4)]
    with pytest.raises(NotImplementedError, match="projection"):
        sp.hal_tmle(df, y="y", treat="a", covariates=cov,
                    variant="projection", max_anchors_per_col=10, n_folds=3)


def test_hal_tmle_rejects_invalid_variant():
    df = _tmle_data(n=100, seed=2)
    with pytest.raises(ValueError):
        sp.hal_tmle(df, y="y", treat="a",
                     covariates=[f"x{j}" for j in range(4)],
                     variant="bogus")


def test_hal_tmle_registry():
    fns = sp.list_functions()
    assert "hal_tmle" in fns
    assert "HALRegressor" in fns or "HALClassifier" in fns


def test_hal_regressor_standalone():
    rng = np.random.default_rng(7)
    n = 200
    X = rng.normal(size=(n, 2))
    y = X[:, 0] ** 2 + rng.normal(0, 0.3, n)
    reg = sp.HALRegressor(max_anchors_per_col=10)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape
    # Should outperform a constant predictor
    mse_hal = np.mean((y - y_pred) ** 2)
    mse_const = np.mean((y - y.mean()) ** 2)
    assert mse_hal < mse_const
