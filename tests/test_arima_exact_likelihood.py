"""``sp.arima`` maximises the exact Gaussian likelihood, of every observation.

Through 1.38.0 the default method started the state from a diffuse prior and
left the first ``max(p, q + 1)`` observations out of the likelihood. The
estimates were not the maximum likelihood estimates Stata, R and
statsmodels' ``ARIMA`` report, and ``auto=True`` ranked models scored on
different numbers of observations. The reference here is the AR(1)
likelihood written out in closed form and maximised by scipy, which shares
no code with the state-space filter.
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import minimize

import statspai as sp
from statspai.exceptions import MethodIncompatibility
from statspai.timeseries.arima import _kpss_ndiffs, _kpss_stat


def _ar1(n, phi, mu, seed):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n + 200)
    x = np.zeros(n + 200)
    for t in range(1, n + 200):
        x[t] = phi * x[t - 1] + e[t]
    return mu + x[200:]


def _exact_ar1_loglik(theta, y):
    mu, phi, log_s2 = theta
    s2 = np.exp(log_s2)
    if abs(phi) >= 1:
        return -np.inf
    e = y - mu
    first = -0.5 * np.log(2 * np.pi * s2 / (1 - phi**2)) - e[0] ** 2 * (
        1 - phi**2
    ) / (2 * s2)
    innov = e[1:] - phi * e[:-1]
    rest = -0.5 * (len(y) - 1) * np.log(2 * np.pi * s2) - (innov @ innov) / (2 * s2)
    return first + rest


@pytest.mark.parametrize("method", ["statespace", "innovations_mle"])
def test_ar1_is_the_maximiser_of_the_closed_form_likelihood(method):
    y = _ar1(90, 0.6, 2.0, seed=11)
    opt = minimize(
        lambda th: -_exact_ar1_loglik(th, y),
        x0=[y.mean(), 0.3, np.log(y.var())],
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000},
    )
    res = sp.arima(y, order=(1, 0, 0), method=method)
    got = np.asarray(res.params, dtype=float)
    # two optimisers on one likelihood: agreement to optimiser tolerance
    np.testing.assert_allclose(got[:2], opt.x[:2], atol=2e-4)
    assert got[2] == pytest.approx(np.exp(opt.x[2]), rel=2e-4)
    # the likelihood counts all 90 observations
    assert res.log_likelihood == pytest.approx(-opt.fun, abs=1e-5)
    assert res.log_likelihood == pytest.approx(
        _exact_ar1_loglik([got[0], got[1], np.log(got[2])], y), abs=1e-8
    )


def test_the_two_methods_agree_on_an_arma_model():
    y = _ar1(200, 0.5, 1.0, seed=3)
    a = sp.arima(y, order=(1, 0, 1))
    b = sp.arima(y, order=(1, 0, 1), method="innovations_mle")
    np.testing.assert_allclose(a.params, b.params, atol=5e-4)
    assert a.log_likelihood == pytest.approx(b.log_likelihood, abs=1e-5)


def test_auto_can_select_white_noise_and_a_random_walk():
    picks_noise, picks_walk = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(12):
            e = np.random.default_rng(900 + seed).normal(size=120)
            picks_noise.append(sp.arima(e, auto=True, max_p=1, max_q=1).order)
            picks_walk.append(sp.arima(np.cumsum(e), auto=True, max_p=1, max_q=1).order)
    # Neither order could be chosen before: (0, d, 0) was skipped, and d
    # was picked by comparing likelihoods of different series.
    assert picks_noise.count((0, 0, 0)) >= 6
    assert picks_walk.count((0, 1, 0)) >= 6
    assert all(order[1] == 0 for order in picks_noise)


def test_kpss_differencing_rule_matches_statsmodels_statistic():
    from statsmodels.tsa.stattools import kpss

    # seed 5 is one of the draws on which a 5% test rejects a true null
    rng = np.random.default_rng(1)
    noise = rng.normal(size=150)
    walk = np.cumsum(noise)
    for series in (noise, walk, np.cumsum(walk)):
        lags = int(3 * np.sqrt(len(series)) / 13)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = kpss(series, regression="c", nlags=lags)[0]
        # the same statistic, written out twice
        assert _kpss_stat(series) == pytest.approx(ref, rel=1e-10)
    assert _kpss_stat(noise) < 0.463 < _kpss_stat(walk)
    assert _kpss_ndiffs(noise, 2) == 0
    assert _kpss_ndiffs(walk, 2) == 1
    assert _kpss_ndiffs(np.cumsum(walk), 2) == 2
    assert _kpss_ndiffs(walk, 0) == 0


def test_conditional_sum_of_squares_is_refused_not_relabelled():
    y = _ar1(80, 0.4, 0.0, seed=1)
    for name in ("css", "conditional"):
        with pytest.raises(MethodIncompatibility, match="conditional sum of squares"):
            sp.arima(y, order=(1, 0, 0), method=name)
    # R's name for "start from CSS, finish with exact ML" stays an alias
    assert sp.arima(y, order=(1, 0, 0), method="css-ml").log_likelihood < 0
