"""Option and degenerate-input tests for ``sp.garch`` (scan round 3).

Every numerical assertion compares with a computation written here (the
variance recursion and the Gaussian / Student t likelihood in plain loops,
central-difference scores and Hessians of that likelihood, closed-form
forecast recursions) or with an identity (equivariance to the unit of
measurement).
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility, StatsPAIError
from statspai.timeseries.garch import garch_loglik

KAPPA = float(np.sqrt(2.0 / np.pi))


def _simulate(T=600, seed=0, omega=0.05, a=0.1, b=0.85, lev=0.08, shift=0.3):
    rng = np.random.default_rng(seed)
    e = np.zeros(T)
    s = np.ones(T) * omega / (1 - a - b - lev / 2)
    for t in range(1, T):
        s[t] = omega + (a + lev * (e[t - 1] < 0)) * e[t - 1] ** 2 + b * s[t - 1]
        e[t] = np.sqrt(s[t]) * rng.standard_normal()
    return e + shift


@pytest.fixture(scope="module")
def y():
    return _simulate()


def _fit(y, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.garch(y, **kw)


def _reference_filter(y, mu, rho, omega, first, gamma, beta, nu, model):
    """Variances, innovations and log-likelihood by the textbook loops.

    Pre-sample squared shocks and variances equal the mean squared
    innovation, pre-sample disturbances are zero, and the asymmetric terms
    have no pre-sample value (the start-up documented for ``sp.garch``).
    """
    u = y - mu
    eps = u.copy()
    for k in range(len(rho)):
        eps[k + 1 :] -= rho[k] * u[: -k - 1]
    T, q, p = len(y), len(first), len(beta)
    m = float(np.mean(eps**2))
    s2 = np.empty(T)
    for t in range(T):
        v = omega
        if model == "egarch":
            for i in range(q):
                if t - 1 - i >= 0:
                    z = eps[t - 1 - i] / np.sqrt(s2[t - 1 - i])
                    v += first[i] * z + gamma[i] * (abs(z) - KAPPA)
            for j in range(p):
                v += beta[j] * np.log(s2[t - 1 - j] if t - 1 - j >= 0 else m)
            s2[t] = np.exp(v)
            continue
        for i in range(q):
            if t - 1 - i >= 0:
                e2 = eps[t - 1 - i] ** 2
                v += first[i] * e2
                if i < len(gamma) and eps[t - 1 - i] < 0:
                    v += gamma[i] * e2
            else:
                v += first[i] * m
        for j in range(p):
            v += beta[j] * (s2[t - 1 - j] if t - 1 - j >= 0 else m)
        s2[t] = v
    if nu is None:
        ll = float(np.sum(stats.norm.logpdf(eps, scale=np.sqrt(s2))))
    else:
        scale = np.sqrt(s2 * (nu - 2.0) / nu)
        ll = float(np.sum(stats.t.logpdf(eps / scale, nu) - np.log(scale)))
    return ll, s2, eps


CONFIGS = [
    {},
    {"p": 2, "q": 2},
    {"p": 0, "q": 2},
    {"ar": 2},
    {"dist": "t"},
    {"model": "gjr"},
    {"model": "gjr", "q": 2, "threshold": 1},
    {"model": "egarch"},
    {"mean": False, "ar": 1, "model": "gjr", "dist": "t"},
]


class TestReportedLikelihood:
    @pytest.mark.parametrize("kw", CONFIGS, ids=lambda kw: repr(kw))
    def test_likelihood_and_paths_follow_from_the_reported_parameters(self, y, kw):
        r = _fit(y, **kw)
        model = kw.get("model", "garch")
        first = r.theta if model == "egarch" else r.alpha
        gamma = [] if r.gamma is None else r.gamma
        ll, s2, eps = _reference_filter(
            y, r.mu, r.ar, r.omega, first, gamma, r.beta, r.nu, model
        )
        # Same recursion in float64; the two differ in summation order only.
        assert r.log_likelihood == pytest.approx(ll, abs=1e-8)
        np.testing.assert_allclose(r.sigma2, s2, rtol=1e-10)
        np.testing.assert_allclose(r.residuals, eps, atol=1e-10)
        k = len(r.params)
        assert r.aic == pytest.approx(-2 * ll + 2 * k, abs=1e-7)
        assert r.bic == pytest.approx(-2 * ll + k * np.log(len(y)), abs=1e-7)
        np.testing.assert_allclose(r.std_residuals, eps / np.sqrt(s2), rtol=1e-9)

    def test_variance_in_mean_residuals_use_the_lagged_transformed_variance(self, y):
        r = _fit(y, in_mean="sd", in_mean_lags=[0, 1])
        assert list(r.params.index[:3]) == ["mu", "archm", "archm[L1]"]
        m = float(np.mean(r.residuals**2))
        lagged = np.r_[m, r.sigma2[:-1]]
        u = y - r.mu - r.archm[0] * np.sqrt(r.sigma2) - r.archm[1] * np.sqrt(lagged)
        np.testing.assert_allclose(r.residuals, u, atol=1e-10)
        ll = np.sum(stats.norm.logpdf(r.residuals, scale=np.sqrt(r.sigma2)))
        assert r.log_likelihood == pytest.approx(float(ll), abs=1e-8)

    def test_rugarch_presample_fixes_the_first_max_p_q_variances(self, y):
        r = _fit(y, presample="rugarch", p=2, q=1)
        m = float(np.mean(r.residuals**2))
        np.testing.assert_allclose(r.sigma2[:2], m, rtol=1e-12)
        expect = (
            r.omega
            + r.alpha[0] * r.residuals[1] ** 2
            + r.beta[0] * r.sigma2[1]
            + r.beta[1] * r.sigma2[0]
        )
        assert r.sigma2[2] == pytest.approx(expect, rel=1e-12)
        assert garch_loglik(y, r.params, 2, 1, True, "rugarch") == pytest.approx(
            r.log_likelihood, abs=1e-9
        )


class TestCovarianceOptions:
    @pytest.fixture(scope="class")
    def pieces(self, y):
        """Scores and Hessian of the Gaussian GARCH(1,1) likelihood by
        central differences of a likelihood written here."""
        th = _fit(y).params.to_numpy()

        def terms(t):
            mu, om, a, b = t
            e = y - mu
            m = np.mean(e**2)
            s = np.empty(len(y))
            pe, ps = m, m
            for i in range(len(y)):
                s[i] = om + a * pe + b * ps
                pe, ps = e[i] ** 2, s[i]
            return -0.5 * (np.log(2 * np.pi) + np.log(s) + e**2 / s)

        def jac(f, t):
            cols = []
            for i in range(len(t)):
                h = 1e-5 * max(abs(t[i]), 1e-2)
                d = np.zeros(len(t))
                d[i] = h
                cols.append((f(t + d) - f(t - d)) / (2 * h))
            return np.array(cols).T

        S = jac(terms, th)
        H = -jac(lambda t: jac(terms, t).sum(axis=0), th)
        return S, (H + H.T) / 2

    @pytest.mark.parametrize("vce", ["oim", "opg", "robust"])
    def test_standard_errors_are_the_named_estimator(self, y, pieces, vce):
        S, H = pieces
        Hi = np.linalg.inv(H)
        V = {"oim": Hi, "opg": np.linalg.inv(S.T @ S), "robust": Hi @ S.T @ S @ Hi}[vce]
        got = _fit(y, vce=vce).std_errors.to_numpy()
        # The Hessian here is a second difference of the likelihood, the
        # package's a first difference of the analytic score: they agree to
        # about five digits (observed 2e-5).
        np.testing.assert_allclose(got, np.sqrt(np.diag(V)), rtol=5e-4)

    def test_vce_changes_only_the_standard_errors(self, y):
        a, b = _fit(y, vce="oim"), _fit(y, vce="robust")
        np.testing.assert_allclose(a.params, b.params, rtol=0, atol=0)
        assert a.log_likelihood == b.log_likelihood
        z = a.params / a.std_errors
        np.testing.assert_allclose(a.tvalues, z, rtol=1e-12)
        np.testing.assert_allclose(a.pvalues, 2 * stats.norm.sf(np.abs(z)), rtol=1e-12)

    @pytest.mark.parametrize("vce", ["opg", "robust"])
    def test_general_path_accepts_every_vce(self, y, vce):
        r = _fit(y, model="gjr", vce=vce)
        assert r.vce == vce
        assert np.all(np.isfinite(r.std_errors)) and np.all(r.std_errors > 1e-4)


class TestUnitOfMeasurement:
    @pytest.mark.parametrize("kw", [{}, {"model": "gjr"}, {"dist": "t", "ar": 1}])
    def test_rescaling_the_series_rescales_the_estimates(self, y, kw):
        c = 1000.0
        a, b = _fit(y, **kw), _fit(c * y, **kw)
        power = {"mu": 1, "omega": 2}
        factor = np.array([c ** power.get(n, 0) for n in a.params.index])
        # The search is not run on a standardised series, so the optimum is
        # reproduced to optimiser tolerance, not to rounding; nu is the
        # flattest direction (observed 5e-4).
        np.testing.assert_allclose(b.params, a.params * factor, rtol=3e-3)
        np.testing.assert_allclose(b.std_errors, a.std_errors * factor, rtol=1e-2)
        assert b.log_likelihood == pytest.approx(
            a.log_likelihood - len(y) * np.log(c), abs=1e-5
        )

    def test_egarch_scale_enters_omega_only(self, y):
        c = 1000.0
        a, b = _fit(y, model="egarch"), _fit(c * y, model="egarch")
        for name in ("theta[1]", "gamma[1]", "beta[1]"):
            assert b.params[name] == pytest.approx(a.params[name], rel=1e-3)
        # ln sigma2 shifts by 2 ln c, so omega shifts by (1 - beta) 2 ln c
        shift = (1 - a.params["beta[1]"]) * 2 * np.log(c)
        assert b.omega - a.omega == pytest.approx(shift, rel=1e-3)


class TestForecasts:
    def test_variance_forecast_recursion_uses_every_lag(self, y):
        r = _fit(y, ar=1, p=1, q=2)
        e = r.residuals
        f = r.forecast(3)
        one = (
            r.omega
            + r.alpha[0] * e[-1] ** 2
            + r.alpha[1] * e[-2] ** 2
            + r.beta[0] * r.sigma2[-1]
        )
        two = r.omega + r.alpha[0] * one + r.alpha[1] * e[-1] ** 2 + r.beta[0] * one
        three = r.omega + (r.alpha[0] + r.beta[0]) * two + r.alpha[1] * one
        np.testing.assert_allclose(f, [one, two, three], rtol=1e-12)

    def test_arch_only_model_forecasts(self, y):
        r = _fit(y, p=0, q=2)
        e = r.residuals
        one = r.omega + r.alpha[0] * e[-1] ** 2 + r.alpha[1] * e[-2] ** 2
        assert r.forecast(1)[0] == pytest.approx(one, rel=1e-12)
        assert r.beta.size == 0

    def test_mean_forecast_decays_at_the_ar_coefficient(self, y):
        r = _fit(y, ar=1)
        u = y[-1] - r.mu
        np.testing.assert_allclose(
            r.forecast_mean(3), r.mu + r.ar[0] ** np.arange(1, 4) * u, rtol=1e-10
        )

    def test_mean_forecast_carries_the_variance_in_mean_terms(self, y):
        r = _fit(y, in_mean="sd", in_mean_lags=[0, 1])
        f = r.forecast(2)
        expect = [
            r.mu + r.archm[0] * np.sqrt(f[0]) + r.archm[1] * np.sqrt(r.sigma2[-1]),
            r.mu + r.archm[0] * np.sqrt(f[1]) + r.archm[1] * np.sqrt(f[0]),
        ]
        np.testing.assert_allclose(r.forecast_mean(2), expect, rtol=1e-12)

    def test_no_mean_model_forecasts_zero(self, y):
        r = _fit(y, mean=False)
        assert r.mu == 0.0 and "mu" not in r.params.index
        np.testing.assert_array_equal(r.forecast_mean(2), [0.0, 0.0])

    def test_value_at_risk_uses_the_unit_variance_student_quantile(self, y):
        r = _fit(y, dist="t")
        nu = r.nu
        q = stats.t.ppf(0.01, nu) * np.sqrt((nu - 2) / nu)
        assert r.value_at_risk(0.01) == pytest.approx(
            r.mu + np.sqrt(r.forecast(1)[0]) * q, rel=1e-12
        )
        g = _fit(y)
        assert g.value_at_risk(0.05) == pytest.approx(
            g.mu + np.sqrt(g.forecast(1)[0]) * stats.norm.ppf(0.05), rel=1e-12
        )
        with pytest.raises(MethodIncompatibility, match="alpha must be in"):
            g.value_at_risk(1.0)

    def test_egarch_one_step_forecast_is_exact(self, y):
        r = _fit(y, model="egarch")
        z = r.std_residuals[-1]
        log_one = (
            r.omega
            + r.theta[0] * z
            + r.gamma[0] * (abs(z) - KAPPA)
            + r.beta[0] * np.log(r.sigma2[-1])
        )
        f = r.forecast(2)
        assert np.log(f[0]) == pytest.approx(log_one, rel=1e-12)
        # normal innovations: E|z| = sqrt(2 / pi), so the shock terms drop out
        assert np.log(f[1]) == pytest.approx(r.omega + r.beta[0] * log_one, rel=1e-12)

    def test_gjr_forecast_with_fewer_threshold_terms_than_arch_lags(self, y):
        r = _fit(y, model="gjr", q=2, threshold=1)
        e = r.residuals
        one = (
            r.omega
            + (r.alpha[0] + r.gamma[0] * (e[-1] < 0)) * e[-1] ** 2
            + r.alpha[1] * e[-2] ** 2
            + r.beta[0] * r.sigma2[-1]
        )
        assert r.forecast(1)[0] == pytest.approx(one, rel=1e-12)

    def test_egarch_student_multi_step_forecast_is_the_expected_log_variance(self):
        rng = np.random.default_rng(4)
        T, nu0 = 800, 5.0
        z = rng.standard_t(nu0, size=T) * np.sqrt((nu0 - 2) / nu0)
        ls, e = np.zeros(T), np.zeros(T)
        for t in range(1, T):
            ls[t] = -0.1 * z[t - 1] + 0.3 * (abs(z[t - 1]) - KAPPA) + 0.9 * ls[t - 1]
            e[t] = np.exp(ls[t] / 2) * z[t]
        r = _fit(e, model="egarch", dist="t")
        nu = r.nu
        e_abs = (
            np.sqrt((nu - 2) / np.pi)
            * special.gamma((nu - 1) / 2)
            / special.gamma(nu / 2)
        )
        f = r.forecast(4)
        expect = r.omega + r.gamma[0] * (e_abs - KAPPA) + r.beta[0] * np.log(f[0])
        # an identity in the reported parameters, exact up to rounding
        assert np.log(f[1]) == pytest.approx(expect, abs=1e-10)
        # and the mean of the log variance over paths of the fitted model,
        # simulated forward from the last observed state
        draws = 200_000
        sim = np.random.default_rng(11)
        ls_next = np.full(draws, np.log(f[0]))  # known at T
        for h in range(1, 4):
            zf = sim.standard_t(nu, size=draws) * np.sqrt((nu - 2) / nu)
            ls_next = (
                r.omega
                + r.theta[0] * zf
                + r.gamma[0] * (np.abs(zf) - KAPPA)
                + r.beta[0] * ls_next
            )
            mc_se = ls_next.std(ddof=1) / np.sqrt(draws)
            assert abs(ls_next.mean() - np.log(f[h])) < 4 * mc_se
            # the correction is not within simulation noise of zero
            if h == 3:
                old = np.log(f[h]) - r.gamma[0] * (e_abs - KAPPA) * (
                    1 + r.beta[0] + r.beta[0] ** 2
                )
                assert abs(ls_next.mean() - old) > 6 * mc_se


class TestInputs:
    def test_data_argument_trims_missing_ends_only(self, y):
        frame = pd.DataFrame({"r": np.r_[np.nan, y, np.nan]})
        a, b = _fit("r", data=frame), _fit(y)
        np.testing.assert_allclose(a.params, b.params, rtol=0, atol=0)
        assert a.n == len(y)
        hole = frame.copy()
        hole.loc[50, "r"] = np.nan
        with pytest.raises(MethodIncompatibility, match="NaN or inf"):
            sp.garch("r", data=hole)

    def test_column_name_without_data_and_array_with_data_are_refused(self, y):
        with pytest.raises(MethodIncompatibility, match="names a column"):
            sp.garch("r")
        with pytest.raises(MethodIncompatibility, match="name of one of its columns"):
            sp.garch(y, data=pd.DataFrame({"r": y}))
        with pytest.raises(MethodIncompatibility, match="no non-missing"):
            sp.garch("r", data=pd.DataFrame({"r": [np.nan] * 30}))

    @pytest.mark.parametrize(
        "kw, match",
        [
            ({"q": 0}, "not identified"),
            ({"p": 0, "q": 0}, "not both zero"),
            ({"p": -1}, "non-negative"),
            ({"presample": "r"}, "presample must be"),
            ({"vce": "hc1"}, "vce must be"),
            ({"dist": "ged"}, "dist="),
            ({"model": "tgarch"}, "model="),
            ({"ar": 1.5}, "non-negative lag count"),
            ({"in_mean": "var"}, "in_mean="),
            ({"in_mean": 2}, "in_mean="),
            ({"in_mean": True, "in_mean_lags": [1, 0]}, "increasing order"),
            ({"in_mean": True, "in_mean_lags": [-1]}, "increasing order"),
            ({"in_mean": True, "presample": "rugarch"}, "presample='stata' only"),
            ({"model": "gjr", "presample": "rugarch"}, "model='garch' only"),
            ({"threshold": 1}, "model='gjr'"),
            ({"model": "gjr", "threshold": 2}, "between 1 and q"),
        ],
    )
    def test_unknown_or_inconsistent_options_are_named(self, y, kw, match):
        with pytest.raises(MethodIncompatibility, match=match):
            sp.garch(y, **kw)

    def test_series_shorter_than_the_lags_is_refused(self, y):
        with pytest.raises(ValueError, match="too short"):
            sp.garch(y[:11], p=1, q=2)

    def test_summary_names_the_model(self, y):
        text = _fit(y, ar=1, dist="t", model="gjr").summary()
        assert text.splitlines()[0] == "AR(1)-GJR-GARCH(1,1), Student t innovations"
        assert "gamma[1]" in text and "nu" in text

    def test_constant_series_is_refused(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(StatsPAIError):
                sp.garch(np.ones(200))

    def test_two_dimensional_input_is_refused(self, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(StatsPAIError):
                sp.garch(np.column_stack([y, y[::-1]]))


class TestScreening:
    def test_single_column_array_is_the_series(self, y):
        a, b = _fit(y.reshape(-1, 1)), _fit(y)
        np.testing.assert_array_equal(a.params.to_numpy(), b.params.to_numpy())

    def test_gjr_padded_threshold_terms_leave_the_full_model_alone(self, y):
        # threshold = q: the forecast recursion written out for two lags
        r = _fit(y, model="gjr", q=2)
        e = r.residuals
        one = (
            r.omega
            + sum(
                (r.alpha[i] + r.gamma[i] * (e[-1 - i] < 0)) * e[-1 - i] ** 2
                for i in range(2)
            )
            + r.beta[0] * r.sigma2[-1]
        )
        assert r.forecast(1)[0] == pytest.approx(one, rel=1e-12)
        short = _fit(y, model="gjr", q=2, threshold=1)
        assert np.isfinite(short.value_at_risk(0.05))
