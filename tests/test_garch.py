"""GARCH(p,q) tests."""
import numpy as np, pytest
from statspai.timeseries.garch import garch


@pytest.fixture(scope="module")
def garch_dgp():
    rng = np.random.default_rng(42)
    T = 2000
    eps = np.zeros(T); s2 = np.zeros(T)
    omega, alpha, beta = 0.01, 0.1, 0.85
    s2[0] = omega / (1 - alpha - beta)
    for t in range(1, T):
        s2[t] = omega + alpha * eps[t-1]**2 + beta * s2[t-1]
        eps[t] = np.sqrt(s2[t]) * rng.standard_normal()
    return eps, omega, alpha, beta


def test_garch_persistence_near_truth(garch_dgp):
    y, omega, alpha, beta = garch_dgp
    res = garch(y, p=1, q=1)
    assert abs(res.persistence - (alpha + beta)) < 0.1


def test_garch_alpha_positive(garch_dgp):
    y, *_ = garch_dgp
    res = garch(y, p=1, q=1)
    assert res.alpha[0] > 0


def test_garch_forecast_shape(garch_dgp):
    y, *_ = garch_dgp
    res = garch(y, p=1, q=1)
    fc = res.forecast(horizon=10)
    assert fc.shape == (10,)
    assert np.all(fc > 0)


def test_garch_std_residuals_near_unit_variance(garch_dgp):
    y, *_ = garch_dgp
    res = garch(y, p=1, q=1)
    assert abs(res.std_residuals.std() - 1.0) < 0.1


def test_garch_summary(garch_dgp):
    y, *_ = garch_dgp
    res = garch(y, p=1, q=1)
    assert "GARCH(1,1)" in res.summary()


def test_exported():
    import statspai as sp
    assert callable(sp.garch)
