"""``sp.ipw(ps_learner=, cross_fit=, n_folds=)``: a user learner for the
propensity score."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _logit():
    return LogisticRegression(penalty=None, tol=1e-12, max_iter=5000)


def _linear_dgp(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * x1 - 0.4 * x2))))
    y = 1.5 * d + x1 + x2 + rng.normal(size=n)
    s = rng.uniform(0.5, 2.0, size=n)
    return pd.DataFrame(dict(y=y, d=d, x1=x1, x2=x2, s=s))


def _nonlinear_dgp(n=3000, seed=1):
    # the score and the outcome both depend on x1 through sin(2 x1):
    # a logit that is linear in x1 leaves that confounding in place
    rng = np.random.default_rng(seed)
    x1, x2 = rng.uniform(-2, 2, size=n), rng.normal(size=n)
    e = 1 / (1 + np.exp(-(1.5 * np.sin(2 * x1) + 0.3 * x2)))
    d = rng.binomial(1, e)
    y = 2.0 * d + 2 * np.sin(2 * x1) + x2 + rng.normal(size=n)
    return pd.DataFrame(dict(y=y, d=d, x1=x1, x2=x2))


@pytest.mark.parametrize("estimand", ["ATE", "ATT", "ATO"])
def test_unpenalised_logit_learner_reproduces_the_default(estimand):
    # Without cross-fitting no folds are drawn, so the bootstrap draws the
    # same resamples; the two scores agree to the logit's tolerance.
    df = _linear_dgp()
    a = sp.ipw(df, "y", "d", ["x1", "x2"], estimand=estimand, n_bootstrap=40, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        b = sp.ipw(
            df,
            "y",
            "d",
            ["x1", "x2"],
            estimand=estimand,
            n_bootstrap=40,
            seed=3,
            ps_learner=_logit(),
            cross_fit=False,
        )
    assert abs(a.estimate - b.estimate) < 1e-6
    assert abs(a.se - b.se) < 1e-6
    assert b.model_info["ps_model"] == "LogisticRegression"


def test_learner_gets_the_sampling_weights():
    df = _linear_dgp()
    a = sp.ipw(df, "y", "d", ["x1", "x2"], weights="s", n_bootstrap=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        b = sp.ipw(
            df,
            "y",
            "d",
            ["x1", "x2"],
            weights="s",
            n_bootstrap=0,
            ps_learner=_logit(),
            cross_fit=False,
        )
    assert abs(a.estimate - b.estimate) < 1e-6


def test_cross_fitted_scores_are_out_of_fold():
    # Identity: the cross-fitted score of a row equals the prediction of a
    # learner trained on the rows of the other folds.
    from statspai.inference._ps_learner import learner_pscore

    df = _linear_dgp(n=300)
    X, T = df[["x1", "x2"]].to_numpy(), df["d"].to_numpy(float)
    folds = np.arange(300) % 3
    e = learner_pscore(_logit(), X, T, None, folds)
    for f in range(3):
        m = _logit().fit(X[folds != f], T[folds != f])
        np.testing.assert_allclose(
            e[folds == f], m.predict_proba(X[folds == f])[:, 1], atol=1e-10
        )


def test_flexible_learner_recovers_the_effect():
    df = _nonlinear_dgp()
    learner = make_pipeline(
        SplineTransformer(n_knots=6, degree=3), LogisticRegression(max_iter=2000)
    )
    r = sp.ipw(df, "y", "d", ["x1", "x2"], ps_learner=learner, n_bootstrap=60, seed=0)
    assert r.model_info["cross_fit"] is True and r.model_info["n_folds"] == 5
    assert abs(r.estimate - 2.0) < 3 * r.se


@pytest.mark.parametrize(
    "kwargs",
    [
        {"se_method": "sandwich"},
        {"propensity": 0.5},
        {"ps_model": "probit"},
        {"n_folds": 1},
    ],
)
def test_options_a_learner_cannot_serve_are_refused(kwargs):
    with pytest.raises(MethodIncompatibility):
        sp.ipw(_linear_dgp(n=100), "y", "d", ["x1"], ps_learner=_logit(), **kwargs)


def test_in_sample_learner_warns():
    with pytest.warns(RuntimeWarning, match="cross_fit=False"):
        sp.ipw(
            _linear_dgp(n=200),
            "y",
            "d",
            ["x1"],
            ps_learner=_logit(),
            cross_fit=False,
            n_bootstrap=0,
        )
