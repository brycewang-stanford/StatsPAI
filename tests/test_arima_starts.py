"""``sp.arima`` on an over-parameterised mixed model: the likelihood found
is never below that of a plain quasi-Newton fit of the same model.

The likelihood of ARMA(5,2) fitted to an ARMA(1,3) has several maxima.
``sp.arima`` keeps the best of three starts (the default values, the
conditional-sum-of-squares estimates, zero ARMA coefficients) and a
simplex check. The case that motivated the third start is on data that
cannot be redistributed (``tests/external_parity/test_neusser_time_series.py``:
-110.13 before, R's -107.03 after); on these simulated series the search
was already at the maximum, and the test keeps it there. Series 8, 10 and
11 are ones where the reference fit itself stops one to three
log-likelihood points short.
"""

import warnings

import numpy as np
import pytest

import statspai as sp


def _series(seed, T=91):
    rng = np.random.default_rng(seed)
    e = rng.normal(scale=0.77, size=T + 53)
    x = np.zeros(T + 53)
    for t in range(3, T + 53):
        x[t] = 0.48 * x[t - 1] + e[t] + 0.58 * e[t - 1] + 0.62 * e[t - 2]
        x[t] += 0.52 * e[t - 3]
    return 1.27 + x[53:]


@pytest.mark.parametrize("seed", [0, 8, 10, 11])
def test_likelihood_not_below_a_plain_quasi_newton_fit(seed):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    x = _series(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = sp.arima(x, order=(5, 0, 2)).log_likelihood
        ref = SARIMAX(x, order=(5, 0, 2), trend="c").fit(
            method="lbfgs", maxiter=2000, disp=0
        )
        nested = sp.arima(x, order=(1, 0, 2)).log_likelihood
    # same exact Gaussian likelihood on both sides; 1e-4 is optimiser noise
    assert ours >= ref.llf - 1e-4
    # and never below a model it nests
    assert ours >= nested - 1e-4
