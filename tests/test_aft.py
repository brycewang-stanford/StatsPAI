"""AFT (Accelerated Failure Time) model tests."""
import numpy as np, pandas as pd, pytest
from statspai.survival.aft import aft


@pytest.fixture(scope="module")
def weibull_dgp():
    rng = np.random.default_rng(42)
    n = 300; x = rng.standard_normal(n)
    logT = 2 + 0.5 * x + 0.8 * rng.gumbel(size=n)
    T = np.exp(logT); E = (T < 50).astype(int); T = np.minimum(T, 50)
    return pd.DataFrame({"T": T, "E": E, "x": x})


def test_aft_weibull_recovers_sign(weibull_dgp):
    res = aft("T + E ~ x", weibull_dgp, family="weibull")
    assert res.beta[1] > 0  # positive effect of x on log-time


def test_aft_lognormal_runs(weibull_dgp):
    res = aft("T + E ~ x", weibull_dgp, family="lognormal")
    assert np.isfinite(res.log_likelihood)


def test_aft_exponential_sigma_is_one(weibull_dgp):
    res = aft("T + E ~ x", weibull_dgp, family="exponential")
    assert res.sigma == 1.0


def test_aft_loglogistic_runs(weibull_dgp):
    res = aft("T + E ~ x", weibull_dgp, family="loglogistic")
    assert np.isfinite(res.aic)


def test_exported():
    import statspai as sp
    assert callable(sp.aft)
