"""
Tests for sp.bayes_dml (Bayesian Double Machine Learning).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _make_plr_data(n: int = 600, true_tau: float = 0.7, seed: int = 73) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X1 = rng.normal(0, 1, size=n)
    X2 = rng.normal(0, 1, size=n)
    D = rng.binomial(1, 1.0 / (1 + np.exp(-(0.3 * X1 + 0.2 * X2))), size=n).astype(float)
    Y = 0.5 * X1 - 0.2 * X2 + true_tau * D + rng.normal(0, 0.5, size=n)
    return pd.DataFrame({"Y": Y, "D": D, "X1": X1, "X2": X2})


def test_bayes_dml_conjugate_recovers_true_tau():
    df = _make_plr_data(n=500, true_tau=0.7, seed=73)
    res = sp.bayes_dml(
        df, y="Y", treatment="D", covariates=["X1", "X2"],
        model="plr", mode="conjugate", random_state=73,
    )
    assert isinstance(res, sp.BayesianDMLResult)
    assert abs(res.posterior_mean - 0.7) < 3 * res.posterior_sd + 0.2
    # With a weakly-informative prior, posterior should ~= DML point
    assert abs(res.posterior_mean - res.dml_point) < 0.05
    assert res.mode == "conjugate"
    assert 0.95 < res.posterior_prob_positive <= 1.0
    lo, hi = res.ci
    assert lo < res.posterior_mean < hi


def test_bayes_dml_informative_prior_shrinks():
    """A tight prior at 0 should pull the posterior toward 0."""
    df = _make_plr_data(n=300, true_tau=0.7, seed=79)
    # Weak-prior run
    weak = sp.bayes_dml(
        df, y="Y", treatment="D", covariates=["X1", "X2"],
        mode="conjugate", prior_sd=10.0, random_state=79,
    )
    tight = sp.bayes_dml(
        df, y="Y", treatment="D", covariates=["X1", "X2"],
        mode="conjugate", prior_mean=0.0, prior_sd=0.05,
        random_state=79,
    )
    # Tight prior pulls posterior closer to 0
    assert abs(tight.posterior_mean) < abs(weak.posterior_mean)
    assert tight.posterior_sd < weak.posterior_sd


def test_bayes_dml_rejects_bad_mode():
    df = _make_plr_data(n=100)
    with pytest.raises(ValueError, match="mode must be"):
        sp.bayes_dml(
            df, y="Y", treatment="D", covariates=["X1", "X2"],
            mode="bogus",
        )


def test_bayes_dml_rejects_bad_prior_sd():
    df = _make_plr_data(n=100)
    with pytest.raises(ValueError, match="prior_sd"):
        sp.bayes_dml(
            df, y="Y", treatment="D", covariates=["X1", "X2"],
            prior_sd=-1.0,
        )


def test_bayes_dml_summary_and_registry():
    df = _make_plr_data(n=200, seed=83)
    res = sp.bayes_dml(
        df, y="Y", treatment="D", covariates=["X1", "X2"], random_state=83,
    )
    assert "Bayesian DML" in res.summary()
    assert res.draws is not None
    assert len(res.draws) == 2000
    assert "bayes_dml" in set(sp.list_functions())
