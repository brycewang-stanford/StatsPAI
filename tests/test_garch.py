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


# ---------------------------------------------------------------------------
# Asymmetric models
# ---------------------------------------------------------------------------


def _simulate_leverage(seed=7, n=3000):
    rng = np.random.default_rng(seed)
    omega, alpha, gamma, beta = 0.04, 0.03, 0.14, 0.86
    s2 = np.full(n + 300, omega / (1 - alpha - gamma / 2 - beta))
    eps = np.zeros(n + 300)
    for t in range(1, n + 300):
        bad = gamma * eps[t - 1] ** 2 * (eps[t - 1] < 0)
        s2[t] = omega + alpha * eps[t - 1] ** 2 + bad + beta * s2[t - 1]
        eps[t] = np.sqrt(s2[t]) * rng.standard_normal()
    return eps[300:]


def test_gjr_recovers_the_leverage_effect_and_nests_garch():
    y = _simulate_leverage()
    gjr = sp.garch(y, model="gjr")
    assert gjr.gamma[0] == pytest.approx(0.14, abs=4 * gjr.std_errors["gamma[1]"])
    assert gjr.pvalues["gamma[1]"] < 0.001
    # gamma = 0 is the symmetric model
    assert gjr.log_likelihood > sp.garch(y).log_likelihood + 5
    assert gjr.persistence == pytest.approx(
        gjr.alpha.sum() + 0.5 * gjr.gamma.sum() + gjr.beta.sum()
    )
    assert "GJR-GARCH(1,1)" in gjr.summary()


def test_gjr_without_asymmetry_matches_garch():
    y = _simulate_general(rho=0.0)
    sym = sp.garch(y)
    gjr = sp.garch(y, model="gjr")
    # the symmetric DGP: gamma is insignificant and the fits coincide
    assert abs(gjr.gamma[0]) < 3 * gjr.std_errors["gamma[1]"]
    assert gjr.log_likelihood >= sym.log_likelihood - 1e-6
    assert gjr.log_likelihood < sym.log_likelihood + 4


def test_egarch_sees_the_leverage_effect_and_forecasts():
    y = _simulate_leverage()
    eg = sp.garch(y, model="egarch")
    assert eg.theta[0] < -3 * eg.std_errors["theta[1]"]  # bad news raises variance
    assert eg.gamma[0] > 0
    assert eg.alpha.size == 0
    assert 0 < eg.persistence < 1
    path = eg.forecast(5)
    assert np.all(path > 0)
    # one step ahead is the recursion itself
    z, kappa = eg.std_residuals[-1], np.sqrt(2 / np.pi)
    one = eg.omega + eg.theta[0] * z + eg.gamma[0] * (abs(z) - kappa)
    one += eg.beta[0] * np.log(eg.sigma2[-1])
    assert path[0] == pytest.approx(np.exp(one), rel=1e-10)


def test_gjr_forecast_uses_half_the_threshold_term_for_future_shocks():
    gjr = sp.garch(_simulate_leverage(), model="gjr")
    f = gjr.forecast(3)
    step = gjr.omega + gjr.persistence * f[1]
    assert f[2] == pytest.approx(step, rel=1e-10)


def test_asymmetric_model_argument_errors():
    y = np.random.default_rng(1).standard_normal(300)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="model"):
        sp.garch(y, model="tgarch")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="presample"):
        sp.garch(y, model="egarch", presample="rugarch")


def test_stata_asymmetric_arch_commands_are_translated():
    gjr = sp.from_stata("arch r, arch(1) tarch(1) garch(1)")
    assert gjr["arguments"]["model"] == "gjr"
    assert any("gamma = -tarch" in note for note in gjr["semantics"])
    eg = sp.from_stata("arch r, earch(1/2) egarch(1) distribution(t)")
    assert eg["arguments"] == {
        "y": "r",
        "p": 1,
        "q": 2,
        "model": "egarch",
        "dist": "t",
        "vce": "opg",
    }
    fewer = sp.from_stata("arch r, arch(1/2) tarch(1) garch(1)")
    assert fewer["arguments"]["threshold"] == 1 and fewer["arguments"]["q"] == 2
    in_mean = sp.from_stata("arch r, arch(1) garch(1) archm")
    assert in_mean["arguments"]["in_mean"] is True
    # a threshold lag without its ARCH term, mixed families: no translation
    for command in (
        "arch r, arch(1) tarch(1/2) garch(1)",
        "arch r, earch(1) garch(1)",
    ):
        assert not sp.from_stata(command).get("python_code")


def test_in_mean_recovers_a_risk_premium():
    rng = np.random.default_rng(12)
    n, omega, alpha, beta, psi = 4000, 0.05, 0.15, 0.8, 0.4
    s2 = np.full(n + 300, omega / (1 - alpha - beta))
    eps = np.zeros(n + 300)
    y = np.zeros(n + 300)
    for t in range(1, n + 300):
        s2[t] = omega + alpha * eps[t - 1] ** 2 + beta * s2[t - 1]
        eps[t] = np.sqrt(s2[t]) * rng.standard_normal()
        y[t] = 0.1 + psi * s2[t] + eps[t]
    fit = sp.garch(y[300:], in_mean=True)
    assert list(fit.params.index) == ["mu", "archm", "omega", "alpha[1]", "beta[1]"]
    assert fit.archm == pytest.approx(psi, abs=4 * fit.std_errors["archm"])
    assert fit.pvalues["archm"] < 0.01
    assert "in mean" in fit.summary()
    # the mean forecast carries the premium on the forecast variance
    f = fit.forecast_mean(2)
    assert f[0] == pytest.approx(fit.mu + fit.archm * fit.forecast(1)[0], rel=1e-10)
    # without a premium in the data the coefficient is insignificant
    flat = sp.garch(_simulate_general(rho=0.0), in_mean=True)
    assert abs(flat.archm) < 3 * flat.std_errors["archm"]


def test_threshold_and_in_mean_argument_errors():
    y = np.random.default_rng(1).standard_normal(300)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="threshold"):
        sp.garch(y, threshold=1)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="threshold"):
        sp.garch(y, model="gjr", q=1, threshold=2)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="in_mean"):
        sp.garch(y, in_mean=True, presample="rugarch")


def test_in_mean_forms_and_lags():
    y = _simulate_general(rho=0.0)
    sd = sp.garch(y, in_mean="sd")
    assert sd.in_mean == "sd" and sd.in_mean_lags == (0,)
    lagged = sp.garch(y, in_mean=True, in_mean_lags=[0, 1])
    assert list(lagged.params.index[:3]) == ["mu", "archm", "archm[L1]"]
    assert np.shape(lagged.archm) == (2,)
    # one more free term cannot lower the likelihood
    assert lagged.log_likelihood >= sp.garch(y, in_mean=True).log_likelihood - 1e-6
    # the mean forecast uses g(variance) at each lag
    psi = lagged.archm
    one = lagged.mu + psi[0] * lagged.forecast(1)[0] + psi[1] * lagged.sigma2[-1]
    assert lagged.forecast_mean(1)[0] == pytest.approx(one, rel=1e-10)
    f_sd = sd.mu + sd.archm * np.sqrt(sd.forecast(1)[0])
    assert sd.forecast_mean(1)[0] == pytest.approx(f_sd, rel=1e-10)
    for bad in ({"in_mean": "square"}, {"in_mean": True, "in_mean_lags": [1, 0]},
                {"in_mean": True, "in_mean_lags": [-1]}):  # fmt: skip
        with pytest.raises(sp.exceptions.MethodIncompatibility):
            sp.garch(y, **bad)
    translated = sp.from_stata("arch r, arch(1) garch(1) archm archmexp(sqrt(X))")
    assert translated["arguments"]["in_mean"] == "sd"
    lags = sp.from_stata("arch r, arch(1) garch(1) archm archmlags(1/2)")
    assert lags["arguments"]["in_mean_lags"] == [0, 1, 2]
    assert not sp.from_stata("arch r, arch(1) garch(1) archm archmexp(X^2)").get(
        "python_code"
    )
