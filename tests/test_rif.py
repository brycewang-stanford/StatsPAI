"""RIF regression and decomposition tests."""
import numpy as np, pandas as pd, pytest
from statspai.decomposition.rif import rifreg, rif_decomposition, rif_values


@pytest.fixture(scope="module")
def wage_dgp():
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.standard_normal(n)
    g = (rng.uniform(size=n) > 0.5).astype(int)
    y = 1 + 0.5 * x + 1.5 * g + rng.standard_normal(n)
    return pd.DataFrame({"y": y, "x": x, "g": g})


def test_rif_ols_median_slope(wage_dgp):
    r = rifreg("y ~ x", wage_dgp, statistic="quantile", tau=0.5)
    assert abs(r.params["x"] - 0.5) < 0.15


def test_rif_ols_variance_positive_effect(wage_dgp):
    r = rifreg("y ~ x", wage_dgp, statistic="variance")
    assert r.params["x"] > 0


def test_rif_decomposition_total_equals_sum(wage_dgp):
    r = rif_decomposition("y ~ x", wage_dgp, group="g",
                          statistic="quantile", tau=0.5)
    assert abs(r.total_diff - (r.explained + r.unexplained)) < 1e-8


def test_rif_decomposition_unexplained_captures_group_effect(wage_dgp):
    r = rif_decomposition("y ~ x", wage_dgp, group="g",
                          statistic="quantile", tau=0.5)
    assert r.unexplained > 1.0     # group coeff ≈ 1.5


def test_rif_values_quantile_shape():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rif = rif_values(y, "quantile", tau=0.5)
    assert rif.shape == (5,)


def test_exported():
    import statspai as sp
    assert callable(sp.rifreg)
    assert callable(sp.rif_decomposition)
