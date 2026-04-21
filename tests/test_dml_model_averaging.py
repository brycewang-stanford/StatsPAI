"""Tests for Model Averaging DML (Ahrens et al. JAE 2025)."""

import numpy as np
import pandas as pd
import pytest
import warnings

warnings.filterwarnings("ignore")

import statspai as sp


def _synth_dml_data(n=400, p=8, theta=1.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    D = X[:, :3].sum(1) + rng.normal(0, 1.0, n)
    Y = theta * D + np.sin(X[:, 0]) + X[:, 1] ** 2 + rng.normal(0, 0.5, n)
    return pd.DataFrame({
        "y": Y, "d": D,
        **{f"x{j}": X[:, j] for j in range(p)},
    })


def test_averaging_recovers_theta():
    df = _synth_dml_data(seed=1)
    r = sp.dml_model_averaging(
        df, y="y", treat="d",
        covariates=[f"x{j}" for j in range(8)],
        n_folds=5, seed=1,
    )
    assert abs(r.estimate - 1.5) < 0.5
    assert r.se >= 0


def test_weight_rules():
    df = _synth_dml_data(seed=2, n=300)
    cov = [f"x{j}" for j in range(8)]
    for rule in ("inverse_risk", "equal", "single_best"):
        r = sp.dml_model_averaging(df, y="y", treat="d",
                                    covariates=cov, n_folds=5, seed=2,
                                    weight_rule=rule)
        assert r.estimate == r.estimate  # not NaN
        total_w = sum(r.model_info["weights"].values())
        assert abs(total_w - 1.0) < 1e-6

    # Single-best should put all mass on one candidate
    r_best = sp.dml_model_averaging(df, y="y", treat="d",
                                     covariates=cov, n_folds=5, seed=2,
                                     weight_rule="single_best")
    weights = list(r_best.model_info["weights"].values())
    assert max(weights) == pytest.approx(1.0)
    assert sorted(weights)[-2] == pytest.approx(0.0)


def test_custom_candidates():
    from sklearn.linear_model import LassoCV
    df = _synth_dml_data(seed=3, n=200)
    cand = [
        (LassoCV(cv=3), LassoCV(cv=3), "lasso_a"),
        (LassoCV(cv=3), LassoCV(cv=3), "lasso_b"),
    ]
    r = sp.dml_model_averaging(df, y="y", treat="d",
                                covariates=[f"x{j}" for j in range(8)],
                                candidates=cand, n_folds=3, seed=3)
    assert set(r.model_info["candidates"]) == {"lasso_a", "lasso_b"}


def test_invalid_weight_rule_raises():
    df = _synth_dml_data(seed=4, n=100)
    with pytest.raises(ValueError):
        sp.dml_model_averaging(df, y="y", treat="d",
                                covariates=["x0"], weight_rule="bogus")


def test_registered_in_public_api():
    fns = sp.list_functions()
    assert "dml_model_averaging" in fns
    assert "model_averaging_dml" in fns
