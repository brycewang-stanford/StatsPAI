"""Cox frailty model tests."""
import numpy as np, pandas as pd, pytest
from statspai.survival.frailty import cox_frailty


@pytest.fixture(scope="module")
def frailty_dgp():
    rng = np.random.default_rng(42)
    n = 300; nc = 30
    cluster = np.repeat(np.arange(nc), n // nc)
    z = rng.gamma(5, 1/5, nc)
    x = rng.standard_normal(n)
    lp = 0.5 * x
    T = rng.exponential(1 / (z[cluster] * np.exp(lp)))
    E = (T < 5).astype(int); T = np.minimum(T, 5)
    return pd.DataFrame({"T": T, "E": E, "x": x, "cluster": cluster})


def test_frailty_recovers_beta(frailty_dgp):
    res = cox_frailty("T + E ~ x", frailty_dgp, cluster="cluster")
    assert abs(res.beta[0] - 0.5) < 0.15


def test_frailty_theta_positive(frailty_dgp):
    res = cox_frailty("T + E ~ x", frailty_dgp, cluster="cluster")
    assert res.theta > 0


def test_frailty_n_clusters_correct(frailty_dgp):
    res = cox_frailty("T + E ~ x", frailty_dgp, cluster="cluster")
    assert res.n_clusters == 30


def test_frailty_concordance_reasonable(frailty_dgp):
    res = cox_frailty("T + E ~ x", frailty_dgp, cluster="cluster")
    assert 0.5 < res.concordance < 0.9


def test_exported():
    import statspai as sp
    assert callable(sp.cox_frailty)
