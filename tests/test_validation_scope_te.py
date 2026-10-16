"""Configuration-level evidence for the treatment-effect estimators.

Pins what ``sp.validation_scope`` reports for ``aipw`` / ``ipw`` / ``tmle`` /
``ebalance`` / ``g_computation`` fits (``statspai/_validation_scope_te.py``),
including the cells it must *not* report as covered: trimmed IPW, the
bootstrap SE, and TMLE with the built-in Super Learner library.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(7)
    n = 500
    x = rng.normal(size=n)
    d = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * x))).astype(int)
    return pd.DataFrame(
        {
            "y": 1 + d + x + rng.normal(size=n),
            "d": d,
            "x": x,
            "g": rng.integers(0, 40, n),
            "w": rng.uniform(0.5, 2, n),
        }
    )


def _scope(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.validation_scope(fn(*a, **k))


K = dict(y="y", treat="d", covariates=["x"])

CASES = [
    ("aipw", lambda df: _scope(sp.aipw, df, **K), "covered"),
    (
        "aipw",
        lambda df: _scope(
            sp.aipw,
            df,
            cross_fit=False,
            se_method="sandwich",
            weights="w",
            cluster="g",
            **K,
        ),
        "covered",
    ),
    (
        "ipw",
        lambda df: _scope(sp.ipw, df, se_method="sandwich", estimand="ATT", **K),
        "covered",
    ),
    (
        "ipw",
        lambda df: _scope(sp.ipw, df, se_method="sandwich", trim=0.05, **K),
        "not_covered",
    ),
    (
        "ipw",
        lambda df: _scope(sp.ipw, df, n_bootstrap=20, seed=1, **K),
        "estimate_only",
    ),
    ("tmle", lambda df: _scope(sp.tmle, df, **K), "not_covered"),
    ("ebalance", lambda df: _scope(sp.ebalance, df, **K), "covered"),
    (
        "g_computation",
        lambda df: _scope(sp.g_computation, df, n_boot=20, seed=1, **K),
        "covered",
    ),
    (
        "g_computation",
        lambda df: _scope(sp.g_computation, df, by_arm=True, se_method="analytic", **K),
        "covered",
    ),
]


@pytest.mark.parametrize("function,make,status", CASES)
def test_status(df, function, make, status):
    scope = make(df)
    assert scope["function"] == function
    assert scope["status"] == status


def test_tmle_records_whether_the_library_is_the_default(df):
    from sklearn.linear_model import LinearRegression, LogisticRegression

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = sp.tmle(df, **K)
        user = sp.tmle(
            df,
            outcome_library=[LinearRegression()],
            propensity_library=[LogisticRegression()],
            **K,
        )
    assert default.model_info["library"] == "default"
    assert user.model_info["library"] == "user"
    assert (
        sp.validation_scope(default)["configuration"]["nuisance"] == "default_library"
    )


def test_lm_lin_is_not_routed_to_g_computation(df):
    df = df.assign(z=(df["x"] > 0).astype(int))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.lm_lin(df, y="y", treat="z", covariates=["x"])
    from statspai.validation_scope import _function_of

    assert _function_of(fit) is None
