"""``sp.aipw(ps_learner=, outcome_learner=)``: user learners for the
nuisance models."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _dgp(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    e = 1 / (1 + np.exp(-(np.sin(2 * x1) + 0.5 * x2**2 - 0.5)))
    d = rng.binomial(1, e)
    y = 2 * d + np.sin(2 * x1) * 2 + x2**2 + rng.normal(size=n)
    return pd.DataFrame(dict(y=y, d=d, x1=x1, x2=x2))


def test_linear_learners_reproduce_the_default_models():
    df = _dgp(n=800)
    a = sp.aipw(df, "y", "d", ["x1", "x2"])
    b = sp.aipw(
        df,
        "y",
        "d",
        ["x1", "x2"],
        ps_learner=LogisticRegression(penalty=None, tol=1e-12, max_iter=5000),
        outcome_learner=LinearRegression(),
    )
    assert abs(a.estimate - b.estimate) < 1e-6 and abs(a.se - b.se) < 1e-6


def test_flexible_learners_recover_the_effect_the_linear_models_miss():
    df = _dgp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.aipw(
            df,
            "y",
            "d",
            ["x1", "x2"],
            ps_learner=GradientBoostingClassifier(max_depth=2, random_state=0),
            outcome_learner=GradientBoostingRegressor(max_depth=3, random_state=0),
        )
    assert abs(r.estimate - 2.0) < 3 * r.se


def test_learners_refuse_the_parametric_sandwich():
    with pytest.raises(MethodIncompatibility):
        sp.aipw(
            _dgp(n=200),
            "y",
            "d",
            ["x1"],
            cross_fit=False,
            se_method="sandwich",
            outcome_learner=LinearRegression(),
        )
