"""Option scan (Oct 2026) of ``statspai.conformal_causal``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _rct_po(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    g = rng.integers(0, 2, n)
    d = rng.integers(0, 2, n)
    y0 = x + rng.normal(size=n)
    y1 = x + 1.0 + rng.normal(size=n)
    df = pd.DataFrame({"y": np.where(d == 1, y1, y0), "d": d, "x": x, "g": g})
    return df, y1 - y0


def test_conformal_cate_se_is_sampling_error_not_cate_dispersion():
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=n)
    d = rng.integers(0, 2, n)
    y = 1.0 * d + rng.normal(size=n)  # x irrelevant: fitted CATE nearly flat
    df = pd.DataFrame({"y": y, "d": d, "x": x})
    res = sp.conformal_cate(
        df, y="y", treat="d", covariates=["x"], model=LinearRegression()
    )
    dim_se = np.sqrt(
        np.var(y[d == 1], ddof=1) / (d == 1).sum()
        + np.var(y[d == 0], ddof=1) / (d == 0).sum()
    )
    # The ATE's standard error is of the order of the difference-in-means SE.
    assert 0.5 * dim_se < res.se < 2.0 * dim_se


def test_conformal_fair_ite_covers_the_effect():
    train, _ = _rct_po(3000, 1)
    test, ite = _rct_po(3000, 2)
    res = sp.conformal_fair_ite(
        train,
        y="y",
        treat="d",
        covariates=["x"],
        protected="g",
        test_data=test,
        alpha=0.1,
    )
    lo, hi = res.intervals[:, 0], res.intervals[:, 1]
    coverage = np.mean((ite >= lo) & (ite <= hi))
    assert coverage >= 0.88, coverage


def test_conformal_cate_estimate_is_the_metalearner_aipw_average():
    """Same outcome model, same default propensity, same folds: the ATE and
    SE are the ones sp.metalearner reports (the AIPW convention)."""
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(3)
    n = 600
    x = rng.normal(size=n)
    d = rng.integers(0, 2, n)
    y = 0.5 * x + (1.0 + 0.3 * x) * d + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x": x})
    cc = sp.conformal_cate(
        df, y="y", treat="d", covariates=["x"], model=LinearRegression()
    )
    ml = sp.metalearner(
        df,
        y="y",
        treat="d",
        covariates=["x"],
        learner="t",
        outcome_model=LinearRegression(),
    )
    assert cc.estimate == pytest.approx(ml.estimate, rel=1e-12)
    assert cc.se == pytest.approx(ml.se, rel=1e-12)
    assert cc.model_info["cate_mean"] == pytest.approx(
        np.mean(cc.model_info["cate"]), rel=1e-12
    )
