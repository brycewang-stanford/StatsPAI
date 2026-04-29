"""GRF-inspired extensions for CausalForest: variable_importance, BLP, ate, att."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.forest.causal_forest import CausalForest


@pytest.fixture(scope="module")
def fitted_cf():
    rng = np.random.default_rng(42)
    n = 600
    X = rng.standard_normal((n, 3))
    T = rng.binomial(1, 0.5, n)
    # CATE = X[:, 0]  (heterogeneous along dim 0 only)
    Y = X[:, 0] * T + X[:, 1] + rng.standard_normal(n)
    data = pd.DataFrame({
        "Y": Y, "T": T, "X0": X[:, 0], "X1": X[:, 1], "X2": X[:, 2],
    })
    cf = CausalForest(n_estimators=50, random_state=42)
    cf.fit("Y ~ T | X0 + X1 + X2", data=data)
    return cf


def test_variable_importance_shape_and_norm(fitted_cf):
    vi = fitted_cf.variable_importance()
    assert len(vi) == 3
    np.testing.assert_allclose(vi.sum(), 1.0, atol=1e-8)
    assert all(v >= 0 for v in vi.values)


def test_variable_importance_sums_to_one(fitted_cf):
    vi = fitted_cf.variable_importance()
    np.testing.assert_allclose(vi.sum(), 1.0, atol=1e-8)


def test_blp_detects_heterogeneity(fitted_cf):
    blp = fitted_cf.best_linear_projection()
    # X0 should have a significant t-stat (drives CATE)
    assert abs(blp.loc["X0", "t"]) > 2.0
    assert blp.loc["X0", "p"] < 0.05


def test_blp_returns_full_table(fitted_cf):
    blp = fitted_cf.best_linear_projection()
    assert set(blp.columns) == {"coef", "se", "t", "p"}
    assert "Intercept" in blp.index
    assert "X0" in blp.index


def test_ate_finite(fitted_cf):
    ate = fitted_cf.ate()
    assert np.isfinite(ate)
    # With n=600 and honest splitting, ATE should be in a reasonable range
    assert abs(ate) < 2.0


def test_att_runs(fitted_cf):
    att = fitted_cf.att()
    assert np.isfinite(att)
