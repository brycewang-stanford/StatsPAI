"""Bayesian VAR tests."""
import numpy as np, pandas as pd, pytest
from statspai.timeseries.bvar import bvar


@pytest.fixture(scope="module")
def var_dgp():
    rng = np.random.default_rng(0)
    T, K = 200, 3
    A = np.array([[0.5, 0.1, 0.0], [0.0, 0.6, 0.1], [0.0, 0.0, 0.4]])
    Y = np.zeros((T, K))
    for t in range(1, T):
        Y[t] = A @ Y[t-1] + rng.standard_normal(K) * 0.3
    return pd.DataFrame(Y, columns=["gdp", "inf", "rate"])


def test_bvar_posterior_coef_near_ols(var_dgp):
    res = bvar(var_dgp, lags=1, lambda1=10.0)  # loose prior → near OLS
    # own-lag coefs should be close to true diagonal values
    assert abs(res.coef[0, 0] - 0.5) < 0.15


def test_bvar_shrinkage_tighter(var_dgp):
    r_loose = bvar(var_dgp, lags=1, lambda1=10.0)
    r_tight = bvar(var_dgp, lags=1, lambda1=0.01)
    # tight prior should have own-lag coefs closer to 1.0 (RW prior)
    assert abs(r_tight.coef[0, 0] - 1.0) < abs(r_loose.coef[0, 0] - 1.0)


def test_bvar_forecast_shape(var_dgp):
    res = bvar(var_dgp, lags=2)
    fc = res.forecast(10)
    assert fc.shape == (10, 3)


def test_bvar_irf_shape(var_dgp):
    res = bvar(var_dgp, lags=1)
    irfs = res.irf(shock_var=0, horizon=20)
    assert irfs.shape == (20, 3)


def test_exported():
    import statspai as sp
    assert callable(sp.bvar)
