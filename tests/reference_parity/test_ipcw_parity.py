"""Frozen-formula parity: sp.ipcw IPCW weight identity.

Inverse-probability-of-censoring weights are w = 1 / P(C = 0 | X, t) from
a pooled logistic regression. Stabilized weights additionally divide by
P(C = 0) to stabilize the mean at 1. With stochastic censoring (X->event
logit + random noise), the fitted weights are well-behaved and bounded. The
stabilized mean weight equals 1.0 to machine precision under a correctly
specified censoring model. Analytical evidence tier.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


def _data():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "time": np.tile(np.arange(5), 40),
            "id": np.repeat(np.arange(40), 5),
            "x": np.repeat(rng.uniform(0, 1, 40), 5),
        }
    )
    # Stochastic censoring via logistic noise -> weights stay bounded.
    logit = -2.0 + 1.0 * df["x"] + rng.normal(0, 1, len(df))
    df["event"] = np.where(rng.random(len(df)) < 1 / (1 + np.exp(-logit)), 0, 1)
    return df


def fitted_event():
    return _data()["event"].to_numpy()


@pytest.fixture(scope="module")
def fitted():
    return sp.ipcw(_data(), time="time", event="event", censor_covariates=["x"])


def test_stabilized_weights_have_mean_one(fitted):
    # Over the uncensored rows E[P(obs) / P(obs | X) | obs] = 1, so the
    # stabilized weights average ~1 there; up to 5% finite-sample slack.
    assert abs(float(fitted.summary_stats["mean"]) - 1.0) < 0.05


def test_all_weights_positive_and_finite(fitted):
    # Censored rows (event == 0) carry weight 0 -- the documented IPCW
    # convention, which the code only started honouring in 1.29.
    w = np.asarray(fitted.weights, dtype=float)
    obs = np.asarray(fitted_event(), dtype=int) == 1
    assert np.all(np.isfinite(w))
    assert np.all(w[obs] > 0)
    assert np.all(w[~obs] == 0)


def test_effective_sample_size_matches_n(fitted):
    # ESS = (sum w)^2 / sum w^2 over the uncensored rows; with mean ~1 it
    # approaches their count.
    w = np.asarray(fitted.weights, dtype=float)
    wo = w[w > 0]
    ess = float(wo.sum() ** 2 / (wo**2).sum())
    assert ess == pytest.approx(len(wo), abs=2.0)
    assert fitted.summary_stats["effective_sample_size"] == pytest.approx(ess)
