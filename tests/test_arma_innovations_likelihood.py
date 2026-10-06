"""The innovations-algorithm likelihood behind ``sp.arima``.

It must be the number a Kalman filter started from the stationary
distribution returns; the reference here is statsmodels' filter, which
shares no code with it. A closed-form AR(1) and MA(1) check guards against
both being wrong together.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statspai.timeseries import _arma_core as core


@pytest.mark.parametrize("seed", range(12))
def test_matches_the_kalman_filter_on_random_seasonal_models(seed):
    rng = np.random.default_rng(seed)
    p, q = (int(v) for v in rng.integers(0, 4, 2))
    P, Q = (int(v) for v in rng.integers(0, 3, 2))
    s = int(rng.choice([4, 12]))
    if p + q + P + Q == 0:
        p = 1
    n = int(rng.integers(60, 200))
    x = 0.1 * np.cumsum(rng.normal(size=n)) + rng.normal(size=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            x, order=(p, 0, q), seasonal_order=(P, 0, Q, s), concentrate_scale=True
        )
        par = np.asarray(
            model.transform_params(rng.normal(0, 0.8, model.k_params)), dtype=float
        )
        ref_ll = float(model.loglike(par))
        ref_scale = float(model.smooth(par).scale)
    phi, theta = core.expand(par, p, q, P, Q, s)
    ll, sigma2 = core.arma_loglike(x, phi, theta)
    # statsmodels solves a Lyapunov equation for the initial covariance to
    # its own tolerance; agreement is at 1e-8, not at machine precision
    assert ll == pytest.approx(ref_ll, rel=1e-7, abs=1e-7)
    assert sigma2 == pytest.approx(ref_scale, rel=1e-7)


def test_ar1_and_ma1_closed_forms():
    rng = np.random.default_rng(3)
    x = rng.normal(size=40)
    n = len(x)
    # AR(1): first observation has variance 1 / (1 - phi^2), then innovations
    phi = 0.6
    e = np.r_[x[0] * np.sqrt(1 - phi**2), x[1:] - phi * x[:-1]]
    sig = float(e @ e) / n
    expect = -0.5 * (n * (np.log(2 * np.pi * sig) + 1) + np.log(1 / (1 - phi**2)))
    ll, s2 = core.arma_loglike(x, np.array([phi]), np.empty(0))
    assert ll == pytest.approx(expect, rel=1e-12)
    assert s2 == pytest.approx(sig, rel=1e-12)
    # MA(1): the covariance matrix is tridiagonal; use it directly
    theta = -0.4
    cov = (1 + theta**2) * np.eye(n) + theta * np.eye(n, k=1) + theta * np.eye(n, k=-1)
    sig = float(x @ np.linalg.solve(cov, x)) / n
    expect = -0.5 * (n * (np.log(2 * np.pi * sig) + 1) + np.linalg.slogdet(cov)[1])
    ll, s2 = core.arma_loglike(x, np.empty(0), np.array([theta]))
    assert ll == pytest.approx(expect, rel=1e-12)
    assert s2 == pytest.approx(sig, rel=1e-12)


def test_white_noise_expansion_and_breakdown():
    x = np.array([1.0, -2.0, 0.5, 1.5])
    ll, s2 = core.arma_loglike(x, np.empty(0), np.empty(0))
    assert s2 == pytest.approx(np.mean(x**2))
    assert ll == pytest.approx(-0.5 * 4 * (np.log(2 * np.pi * s2) + 1))
    # (1 - 0.5 B)(1 - 0.3 B^4) and (1 + 0.2 B)(1 + 0.4 B^4)
    phi, theta = core.expand(np.array([0.5, 0.2, 0.3, 0.4]), 1, 1, 1, 1, 4)
    np.testing.assert_allclose(phi, [0.5, 0.0, 0.0, 0.3, -0.15])
    np.testing.assert_allclose(theta, [0.2, 0.0, 0.0, 0.4, 0.08])
    # a non-stationary AR coefficient has no stationary covariance
    ll, _ = core.arma_loglike(np.arange(10.0), np.array([1.0]), np.empty(0))
    assert not np.isfinite(ll)
