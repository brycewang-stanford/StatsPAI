"""Analytical parity: conformal prediction / conformal causal intervals.

The defining guarantee of conformal prediction is *coverage*: a level ``1-alpha``
interval covers the truth at least ``1-alpha`` of the time on exchangeable data.

* ``sp.weighted_conformal_prediction`` (unit weights, exchangeable data) attains
  marginal coverage close to the nominal ``1-alpha``.
* ``sp.conformal_ite_interval`` covers a known *homogeneous* treatment effect at
  no less than the nominal rate (nested ITE intervals are conservative) and its
  point estimate recovers the ATE.
* ``sp.conformal_cate`` recovers the average effect under a homogeneous DGP.

Analytical evidence tier (coverage / known-truth recovery on deterministic
DGPs).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


# --------------------------------------------------------------------------
# split-conformal marginal coverage
# --------------------------------------------------------------------------
def test_weighted_conformal_marginal_coverage():
    rng = np.random.default_rng(0)
    beta = np.array([1.0, -0.5, 0.3])

    def gen(n):
        X = rng.normal(0, 1, (n, 3))
        y = X @ beta + rng.normal(0, 1.0, n)
        return X, y

    X_tr, y_tr = gen(1500)
    X_ca, y_ca = gen(1500)
    X_te, y_te = gen(5000)
    lower, upper, _ = sp.weighted_conformal_prediction(
        X_tr, y_tr, X_ca, y_ca, X_te, alpha=0.1
    )
    cov = ((y_te >= np.asarray(lower)) & (y_te <= np.asarray(upper))).mean()
    # Split conformal is finite-sample valid; allow Monte-Carlo slack.
    assert cov == pytest.approx(0.90, abs=0.03)


# --------------------------------------------------------------------------
# conformal ITE intervals (homogeneous truth)
# --------------------------------------------------------------------------
def _homogeneous_effect_data(seed=0, n=4000, tau=2.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 3))
    d = (rng.random(n) < 0.5).astype(int)
    y = X @ np.array([1.0, 0.5, -0.3]) + tau * d + rng.normal(0, 1, n)
    df = pd.DataFrame(X, columns=["x0", "x1", "x2"])
    df["d"] = d
    df["y"] = y
    return df


def test_conformal_ite_covers_true_homogeneous_effect():
    df = _homogeneous_effect_data(tau=2.0)
    res = sp.conformal_ite_interval(
        df, y="y", treat="d", covariates=["x0", "x1", "x2"],
        alpha=0.1, random_state=0,
    )
    lower = np.asarray(res.lower, dtype=float)
    upper = np.asarray(res.upper, dtype=float)
    assert np.all(lower <= upper)
    cov = ((2.0 >= lower) & (2.0 <= upper)).mean()
    # Nested ITE intervals are conservative -> at least the nominal 0.9.
    assert cov >= 0.9


def test_conformal_ite_point_recovers_ate():
    df = _homogeneous_effect_data(tau=2.0)
    res = sp.conformal_ite_interval(
        df, y="y", treat="d", covariates=["x0", "x1", "x2"],
        alpha=0.1, random_state=0,
    )
    assert float(np.mean(res.point)) == pytest.approx(2.0, abs=0.2)


def test_conformal_cate_recovers_average_effect():
    df = _homogeneous_effect_data(tau=2.0)
    res = sp.conformal_cate(
        df, y="y", treat="d", covariates=["x0", "x1", "x2"], alpha=0.1
    )
    assert float(res.estimate) == pytest.approx(2.0, abs=0.2)
