"""GARCH(p,q) tests."""

import numpy as np
import pytest

import statspai as sp
from statspai.timeseries.garch import garch


@pytest.fixture(scope="module")
def garch_dgp():
    rng = np.random.default_rng(42)
    T = 2000
    eps = np.zeros(T)
    s2 = np.zeros(T)
    omega, alpha, beta = 0.01, 0.1, 0.85
    s2[0] = omega / (1 - alpha - beta)
    for t in range(1, T):
        s2[t] = omega + alpha * eps[t - 1] ** 2 + beta * s2[t - 1]
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


# ---------------------------------------------------------------------------
# Higher orders, AR mean, Student t innovations
# ---------------------------------------------------------------------------


def _simulate_general(seed=3, n=2500, rho=0.3, nu=None):
    rng = np.random.default_rng(seed)
    omega, alpha, b1, b2 = 0.05, 0.12, 0.35, 0.45
    z = rng.standard_normal(n + 300)
    if nu is not None:
        z = rng.standard_t(nu, size=n + 300) * np.sqrt((nu - 2.0) / nu)
    s2 = np.full(n + 300, omega / (1 - alpha - b1 - b2))
    eps = np.zeros(n + 300)
    u = np.zeros(n + 300)
    for t in range(2, n + 300):
        s2[t] = omega + alpha * eps[t - 1] ** 2 + b1 * s2[t - 1] + b2 * s2[t - 2]
        eps[t] = np.sqrt(s2[t]) * z[t]
        u[t] = rho * u[t - 1] + eps[t]
    return 0.1 + u[300:]


def test_fast_variance_path_equals_the_loop():
    from statspai.timeseries.garch import _garch_filter, _variance_path

    rng = np.random.default_rng(0)
    y = rng.standard_normal(300)
    for p, q in [(1, 1), (2, 1), (1, 3), (0, 2), (3, 2)]:
        alpha = np.full(q, 0.3 / q)
        beta = np.full(p, 0.5 / max(p, 1))
        theta = np.concatenate([[0.1, 0.2], alpha, beta])
        for presample in ("stata", "rugarch"):
            loop = _garch_filter(theta, y, p, q, True, presample)[0]
            fast = _variance_path(0.2, alpha, beta, (y - 0.1) ** 2, presample)
            # same recursion; only the order of the floating-point sums differs
            np.testing.assert_allclose(fast, loop, rtol=1e-12)


def test_a_higher_order_model_never_fits_worse_than_the_one_it_nests():
    # Through 1.38.0 the simplex could stop with beta[2] = 0 and a
    # log-likelihood below GARCH(1,1).
    y = _simulate_general(rho=0.0)
    ll11 = sp.garch(y, p=1, q=1).log_likelihood
    fit21 = sp.garch(y, p=2, q=1)
    assert fit21.log_likelihood >= ll11 - 1e-6
    assert fit21.beta[1] > 0.1  # the second lag is in the DGP
    assert sp.garch(y, p=1, q=2).log_likelihood >= ll11 - 1e-6


def test_ar_mean_recovers_the_autoregressive_coefficient():
    fit = sp.garch(_simulate_general(rho=0.3), p=2, q=1, ar=1)
    assert fit.ar[0] == pytest.approx(0.3, abs=4 * fit.std_errors["ar[1]"])
    assert fit.mu == pytest.approx(0.1, abs=4 * fit.std_errors["mu"])
    # the innovations, not the disturbances, are what the variance filters
    assert not np.allclose(fit.residuals, fit.disturbances)
    assert fit.forecast_mean(50)[-1] == pytest.approx(fit.mu, abs=1e-6)


def test_student_t_recovers_the_degrees_of_freedom():
    y = _simulate_general(rho=0.0, nu=6.0)
    fit = sp.garch(y, p=2, q=1, dist="t")
    assert fit.nu == pytest.approx(6.0, abs=4 * fit.std_errors["nu"])
    assert fit.log_likelihood > sp.garch(y, p=2, q=1).log_likelihood
    # heavier tails: the t quantile is further out than the normal one
    normal = sp.garch(y, p=2, q=1)
    assert fit.value_at_risk(0.001) < normal.value_at_risk(0.001)


def test_boundary_estimate_warns():
    rng = np.random.default_rng(5)
    y = _simulate_general(rho=0.0)[:400] * 0 + rng.standard_normal(400)
    with pytest.warns(RuntimeWarning, match="boundary"):
        sp.garch(y, p=2, q=2)


def test_bad_distribution_and_ar_are_refused():
    y = np.random.default_rng(1).standard_normal(200)
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.garch(y, dist="ged")
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.garch(y, ar=-1)
