"""The influence functions behind ``sp.dml_sensitivity``'s bound standard errors.

The lower bias bound is ``theta - c * sqrt(sigma^2 * nu^2)``. Its standard
error comes from the influence function ``psi_theta - c * psi_S``, and the
upper bound's from ``psi_theta + c * psi_S``. Which sign goes with which
bound matters: ``doubleml`` 0.11.3 has them exchanged (see
``tests/external_parity/test_dml_sensitivity_parity.py``), and the textbook
code of Chernozhukov, Hansen, Kallus, Spindler and Syrgkanis agrees with the
assignment used here.

The check is exact and needs no reference package. An influence function is
the derivative of the statistic with respect to the weight of one
observation, so re-weighting observation ``i`` by ``1 + h`` and differencing
gives it numerically. The design has skewed, heteroskedastic errors so that
the two candidate standard errors differ by a factor of about 1.7 and cannot
be confused.

References
----------
[@chernozhukov2022long]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import statspai as sp

N = 400
STRENGTH = 0.3  # c = sqrt(cf_y * cf_d / (1 - cf_d))


@pytest.fixture(scope="module")
def residuals():
    rng = np.random.default_rng(0)
    v = rng.normal(size=N)
    eta = rng.exponential(size=N) - 1.0
    y = 1.0 * v + eta * (1.0 + v)
    return y, v


def _bounds(y, v, w):
    w = w / w.sum()
    theta = np.sum(w * v * y) / np.sum(w * v * v)
    sigma2 = np.sum(w * (y - theta * v) ** 2)
    nu2 = 1.0 / np.sum(w * v * v)
    half = STRENGTH * np.sqrt(sigma2 * nu2)
    return theta - half, theta + half


def _numerical_influence(y, v):
    base_low, base_high = _bounds(y, v, np.ones(N))
    h = 1e-6
    low, high = np.zeros(N), np.zeros(N)
    for i in range(N):
        w = np.ones(N)
        w[i] += h
        lo, hi = _bounds(y, v, w)
        low[i] = (lo - base_low) / h * N
        high[i] = (hi - base_high) / h * N
    return low, high


def test_bound_standard_errors_are_those_of_the_numerical_influence(residuals):
    """sp.dml_sensitivity's se_low / se_high against finite differences.

    Tolerance 1e-4 relative: the finite-difference step is 1e-6 and the
    derivative is exact to first order in it.
    """
    y, v = residuals
    # A PLR fit whose cross-fitted residuals are (y, v) themselves: one
    # irrelevant covariate, and a learner that predicts zero.
    df = pd.DataFrame({"y": y, "d": v, "x": np.zeros(N)})
    zero = LinearRegression(fit_intercept=False)
    fit = sp.dml(
        df,
        y="y",
        treat="d",
        covariates=["x"],
        model="plr",
        ml_g=zero,
        ml_m=zero,
        n_folds=2,
        fold_indices=np.arange(N) % 2,
    )
    cf_d = 0.2
    cf_y = STRENGTH**2 * (1 - cf_d) / cf_d
    sens = sp.dml_sensitivity(fit, cf_y=cf_y, cf_d=cf_d)

    low, high = _numerical_influence(y, v)
    se_low = np.sqrt(np.mean(low**2) / N)
    se_high = np.sqrt(np.mean(high**2) / N)
    assert se_high / se_low > 1.5, "design must separate the two candidates"
    assert sens.se_low == pytest.approx(se_low, rel=1e-4)
    assert sens.se_high == pytest.approx(se_high, rel=1e-4)
    lo, hi = _bounds(y, v, np.ones(N))
    assert sens.adjusted_estimate_low == pytest.approx(lo, rel=1e-12)
    assert sens.adjusted_estimate_high == pytest.approx(hi, rel=1e-12)


def test_zero_confounding_returns_the_estimate_and_its_standard_error(residuals):
    y, v = residuals
    df = pd.DataFrame({"y": y, "d": v, "x": np.zeros(N)})
    zero = LinearRegression(fit_intercept=False)
    fit = sp.dml(
        df,
        y="y",
        treat="d",
        covariates=["x"],
        model="plr",
        ml_g=zero,
        ml_m=zero,
        n_folds=2,
        fold_indices=np.arange(N) % 2,
    )
    sens = sp.dml_sensitivity(fit, cf_y=0.0, cf_d=0.0)
    assert sens.bias_bound == 0.0
    assert sens.se_low == pytest.approx(fit.se, rel=1e-12)
    assert sens.se_high == pytest.approx(fit.se, rel=1e-12)
