"""``sp.arima`` with its default method against Stata 18 ``arima``.

Track A module ``39_arima`` runs ``method="innovations_mle"``. This file
covers the default (``method="statespace"``), which through 1.38.0 was not
the exact maximum likelihood estimator and was not compared with anything.

The series is ``_fixtures/arima_default_stata.csv`` (160 observations of a
simulated ARMA(2,1) around 1.5, rounded to six decimals). The reference
numbers were printed by Stata 18 MP from::

    import delimited arima_default_stata.csv, clear asdouble
    gen t = _n
    tsset t
    arima y, <spec> nolog vce(oim) nrtolerance(1e-10) ///
        tolerance(1e-12) ltolerance(1e-13)

Stata's last parameter is sigma; ``sp.arima`` reports sigma squared.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIXTURE = Path(__file__).parent / "_fixtures" / "arima_default_stata.csv"

# order -> (e(ll), constant and ARMA coefficients, sigma)
STATA = {
    (1, 0, 0): (-234.2635511172, [1.435698218944, 0.539648648857], 1.045121416745),
    (2, 0, 0): (
        -218.3236274201,
        [1.431202029886, 0.770420129368, -0.423516345595],
        0.944844948489,
    ),
    (1, 0, 1): (
        -213.2892083271,
        [1.434944254733, 0.192823167758, 0.707010118433],
        0.914879119041,
    ),
    (0, 0, 2): (
        -212.5745157337,
        [1.435024175350, 0.938553057805, 0.208638512241],
        0.910726215522,
    ),
}


@pytest.fixture(scope="module")
def y():
    return pd.read_csv(FIXTURE)["y"].to_numpy()


@pytest.mark.parametrize("order", sorted(STATA))
@pytest.mark.parametrize("method", ["statespace", "innovations_mle"])
def test_arima_matches_stata(y, order, method):
    ll, coefs, sigma = STATA[order]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.arima(y, order=order, method=method)
    got = np.asarray(res.params, dtype=float)
    # Both sides maximise one likelihood with a quasi-Newton search that
    # stops on its own gradient tolerance: the log-likelihood agrees to
    # 1e-6 and the coefficients to the fourth decimal.
    assert res.log_likelihood == pytest.approx(ll, abs=1e-6)
    np.testing.assert_allclose(got[:-1], coefs, atol=2e-4)
    assert np.sqrt(got[-1]) == pytest.approx(sigma, abs=2e-4)


def test_information_criteria_use_every_observation(y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.arima(y, order=(1, 0, 1))
    k = 4
    assert res.aic == pytest.approx(-2 * STATA[(1, 0, 1)][0] + 2 * k, abs=1e-5)
    assert res.bic == pytest.approx(
        -2 * STATA[(1, 0, 1)][0] + k * np.log(len(y)), abs=1e-5
    )
