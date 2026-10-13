"""Option and degenerate-input tests for ``sp.arima`` (scan round 3).

Numerical assertions compare with closed forms written here (the exact
Gaussian AR(1) likelihood, forecast recursions and their variances, the
random walk with drift), with statsmodels' ``ARIMA`` where it maximises the
same exact likelihood, or with identities (equivariance to the unit of
measurement, a Box-Cox fit equals a fit on the transformed series).
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility, StatsPAIError

N = 200


@pytest.fixture(scope="module")
def series():
    """ARMA(1,1) disturbance ``u``, a regressor ``x`` and ``y = 5 + 2x + u``."""
    rng = np.random.default_rng(1)
    e = rng.normal(size=N)
    u = np.zeros(N)
    for t in range(1, N):
        u[t] = 0.6 * u[t - 1] + e[t] + 0.3 * e[t - 1]
    x = rng.normal(size=N)
    return {"u": u, "x": x, "y": 5 + 2 * x + u}


def _fit(*args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.arima(*args, **kw)


class TestExactLikelihood:
    def test_ar1_is_the_maximum_of_the_exact_gaussian_likelihood(self, series):
        y = series["u"] + 5

        def neg_ll(theta):
            c, phi, log_s2 = theta
            s2 = np.exp(log_s2)
            d = y - c
            ll = stats.norm.logpdf(d[0], scale=np.sqrt(s2 / (1 - phi**2)))
            ll += stats.norm.logpdf(d[1:] - phi * d[:-1], scale=np.sqrt(s2)).sum()
            return -ll

        opt = optimize.minimize(
            neg_ll,
            [y.mean(), 0.5, 0.0],
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000},
        )
        r = _fit(y, order=(1, 0, 0))
        # two optimisers on the same smooth objective
        assert r.log_likelihood == pytest.approx(-opt.fun, abs=1e-7)
        np.testing.assert_allclose(
            r.params.to_numpy(), [opt.x[0], opt.x[1], np.exp(opt.x[2])], rtol=2e-4
        )
        assert r.sigma2 == pytest.approx(r.params["sigma2"])

    def test_arma_matches_statsmodels_and_the_information_criteria(self, series):
        from statsmodels.tsa.arima.model import ARIMA

        y = series["u"] + 5
        r = _fit(y, order=(1, 0, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm = ARIMA(y, order=(1, 0, 1)).fit()
        # statsmodels' ARIMA maximises the same exact likelihood with a
        # single quasi-Newton run; agreement is to its tolerance.
        np.testing.assert_allclose(r.params.to_numpy(), sm.params, rtol=1e-3)
        np.testing.assert_allclose(r.se.to_numpy(), sm.bse, rtol=1e-3)
        assert r.log_likelihood == pytest.approx(sm.llf, abs=1e-6)
        k, ll = 4, r.log_likelihood
        assert r.aic == pytest.approx(-2 * ll + 2 * k)
        assert r.bic == pytest.approx(-2 * ll + k * np.log(N))
        assert r.aicc == pytest.approx(r.aic + 2 * k * (k + 1) / (N - k - 1))
        z = stats.norm.ppf(0.95)
        ci = r.conf_int(0.10)
        np.testing.assert_allclose(ci["lower"], r.params - z * r.se, rtol=1e-12)
        np.testing.assert_allclose(r.tvalues, r.params / r.se, rtol=1e-12)
        np.testing.assert_allclose(
            r.pvalues, 2 * stats.norm.sf(np.abs(r.params / r.se)), rtol=1e-10
        )

    def test_the_two_methods_agree_on_a_pure_ar_model(self, series):
        a = _fit(series["u"], order=(2, 0, 0), method="innovations_mle")
        b = _fit(series["u"], order=(2, 0, 0))
        np.testing.assert_allclose(a.params, b.params, atol=1e-5)
        np.testing.assert_allclose(a.se, b.se, atol=1e-5)
        assert a.log_likelihood == pytest.approx(b.log_likelihood, abs=1e-8)

    def test_information_criteria_count_observations_after_differencing(self, series):
        y = np.cumsum(series["u"])
        r = _fit(y, order=(0, 1, 1), seasonal_order=(0, 1, 1, 4))
        k, n_eff = 3, N - 1 - 4
        assert r.bic == pytest.approx(-2 * r.log_likelihood + k * np.log(n_eff))
        assert r.aicc == pytest.approx(r.aic + 2 * k * (k + 1) / (n_eff - k - 1))


class TestTrendDefault:
    def test_constant_without_differencing_none_with_it(self, series):
        y = np.cumsum(series["u"] + 0.5)
        assert "const" in _fit(series["u"], order=(1, 0, 0)).params.index
        assert list(_fit(y, order=(1, 1, 0)).params.index) == ["ar.L1", "sigma2"]
        assert list(_fit(series["u"], order=(1, 0, 0), trend="n").params.index) == [
            "ar.L1",
            "sigma2",
        ]

    def test_random_walk_with_drift_is_the_mean_and_variance_of_the_changes(
        self, series
    ):
        y = np.cumsum(series["u"] + 0.5)
        r = _fit(y, order=(0, 1, 0), trend="c")
        d = np.diff(y)
        # closed-form MLE; the package reaches it by a numerical search
        assert r.params["drift"] == pytest.approx(d.mean(), rel=1e-6)
        assert r.params["sigma2"] == pytest.approx(d.var(), rel=1e-6)
        f = r.forecast(3)
        np.testing.assert_allclose(
            f["forecast"], y[-1] + r.params["drift"] * np.arange(1, 4), rtol=1e-9
        )
        z = stats.norm.ppf(0.975)
        np.testing.assert_allclose(
            f["upper"] - f["forecast"],
            z * np.sqrt(r.params["sigma2"] * np.arange(1, 4)),
            rtol=1e-6,
        )

    def test_constant_with_two_differences_is_refused(self, series):
        with pytest.raises(MethodIncompatibility, match="polynomial trend"):
            sp.arima(series["u"], order=(0, 2, 1), trend="c")
        with pytest.raises(MethodIncompatibility, match="polynomial trend"):
            sp.arima(
                series["u"], order=(0, 1, 1), seasonal_order=(0, 1, 0, 4), trend="c"
            )


class TestForecast:
    def test_ar1_forecast_and_interval_closed_form(self, series):
        y = series["u"] + 5
        r = _fit(y, order=(1, 0, 0))
        c, phi, s2 = r.params
        h = np.arange(1, 5)
        f = r.forecast(4, alpha=0.10)
        np.testing.assert_allclose(f["forecast"], c + phi**h * (y[-1] - c), rtol=1e-9)
        sd = np.sqrt(s2 * np.cumsum(phi ** (2 * (h - 1))))
        z = stats.norm.ppf(0.95)
        np.testing.assert_allclose(f["upper"] - f["forecast"], z * sd, rtol=1e-8)
        np.testing.assert_allclose(f["forecast"] - f["lower"], z * sd, rtol=1e-8)

    def test_level_layout_holds_the_same_numbers_as_alpha(self, series):
        r = _fit(series["u"], order=(1, 0, 1))
        a80, a95 = r.forecast(3, alpha=0.20), r.forecast(3, alpha=0.05)
        lv = r.forecast(3, level=(80, 95))
        assert list(lv.columns) == [
            "forecast",
            "lower_80",
            "upper_80",
            "lower_95",
            "upper_95",
        ]
        np.testing.assert_allclose(lv["lower_80"], a80["lower"], rtol=1e-12)
        np.testing.assert_allclose(lv["upper_95"], a95["upper"], rtol=1e-12)

    def test_dof_adjust_scales_the_width_by_root_n_over_n_minus_k(self, series):
        y = np.cumsum(series["u"])
        r = _fit(y, order=(1, 1, 0), trend="c")
        plain, adj = r.forecast(3), r.forecast(3, dof_adjust=True)
        n_star, k = N - 1, 2  # drift and one AR coefficient
        ratio = (adj["upper"] - adj["lower"]) / (plain["upper"] - plain["lower"])
        np.testing.assert_allclose(ratio, np.sqrt(n_star / (n_star - k)), rtol=1e-12)
        np.testing.assert_allclose(adj["forecast"], plain["forecast"], rtol=0, atol=0)

    def test_dated_series_forecast_continues_the_index(self, series):
        idx = pd.date_range("2000-01-01", periods=N, freq="MS")
        f = _fit(pd.Series(series["u"], index=idx)).forecast(2)
        assert list(f.index) == list(pd.date_range("2016-09-01", periods=2, freq="MS"))

    def test_horizon_and_level_are_validated(self, series):
        r = _fit(series["u"])
        with pytest.raises(MethodIncompatibility, match="horizon must be at least 1"):
            r.forecast(0)
        with pytest.raises(MethodIncompatibility, match="coverage between 0 and 100"):
            r.forecast(2, level=195)

    def test_alpha_outside_the_unit_interval_is_refused(self, series):
        r = _fit(series["u"])
        with pytest.raises(StatsPAIError):
            r.forecast(2, alpha=1.5)


class TestRegressors:
    def test_forecast_is_conditional_on_the_future_regressor(self, series):
        frame = pd.DataFrame({"x": series["x"]})
        r = _fit(series["y"], order=(1, 0, 0), exog=frame)
        assert r.exog_names == ("x",)
        c, b, phi, _ = r.params
        u_last = series["y"][-1] - c - b * series["x"][-1]
        expect = [c + b * 1.0 + phi * u_last, c + b * 2.0 + phi**2 * u_last]
        by_name = r.forecast(2, exog=pd.DataFrame({"x": [1.0, 2.0], "junk": [9, 9]}))
        by_position = r.forecast(2, exog=[1.0, 2.0])
        np.testing.assert_allclose(by_name["forecast"], expect, rtol=1e-8)
        np.testing.assert_allclose(by_position["forecast"], expect, rtol=1e-8)

    def test_column_names_through_data(self, series):
        frame = pd.DataFrame(
            {
                "y": np.r_[np.nan, series["y"]],
                "x": np.r_[0.0, series["x"]],
            }
        )
        a = _fit("y", order=(1, 0, 0), exog=["x"], data=frame)
        b = _fit(series["y"], order=(1, 0, 0), exog=series["x"])
        assert a.n == N and "x" in a.params.index and "x1" in b.params.index
        np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=1e-10)

    def test_regressor_errors(self, series):
        y, x = series["y"], series["x"]
        r = _fit(y, order=(1, 0, 0), exog=x)
        with pytest.raises(MethodIncompatibility, match="their values over the"):
            r.forecast(2)
        with pytest.raises(MethodIncompatibility, match=r"\(2, 1\) is needed"):
            r.forecast(2, exog=np.ones((3, 1)))
        with pytest.raises(MethodIncompatibility, match="missing values"):
            r.forecast(2, exog=[1.0, np.nan])
        with pytest.raises(MethodIncompatibility, match="lacks the regressors"):
            _fit(y, order=(1, 0, 0), exog=pd.DataFrame({"x": x})).forecast(
                2, exog=pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]})
            )
        with pytest.raises(MethodIncompatibility, match="has no regressors"):
            _fit(y).forecast(2, exog=[1.0, 2.0])
        with pytest.raises(MethodIncompatibility, match="exog has 199 rows"):
            sp.arima(y, exog=x[:-1])
        bad = x.copy()
        bad[3] = np.nan
        with pytest.raises(MethodIncompatibility, match="exog has missing values"):
            sp.arima(y, exog=bad)
        with pytest.raises(MethodIncompatibility, match="are not in data"):
            sp.arima("y", exog=["nope"], data=pd.DataFrame({"y": y}))

    def test_collinear_regressors_are_refused_or_dropped(self, series):
        x = series["x"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r = sp.arima(
                    series["y"], order=(1, 0, 0), exog=np.column_stack([x, 2 * x])
                )
            except StatsPAIError:
                return
        # the later column is omitted and named; what is left is the fit
        # on the first column alone
        ref = _fit(series["y"], order=(1, 0, 0), exog=x)
        assert [o["variable"] for o in r.model_info["omitted"]] == ["x2"]
        assert list(r.params.index) == list(ref.params.index)
        np.testing.assert_array_equal(r.params.to_numpy(), ref.params.to_numpy())
        np.testing.assert_array_equal(r.se.to_numpy(), ref.se.to_numpy())
        # new regressor values are accepted in the layout of the call
        new = np.column_stack([np.ones(3), 2 * np.ones(3)])
        pd.testing.assert_frame_equal(
            r.forecast(3, exog=new), ref.forecast(3, exog=np.ones(3))
        )

    def test_regressor_collinear_with_the_constant_is_refused_or_dropped(self, series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r = sp.arima(series["u"] + 5, order=(1, 0, 0), exog=np.ones(N))
            except StatsPAIError:
                return
        ref = _fit(series["u"] + 5, order=(1, 0, 0))
        assert r.params["const"] == pytest.approx(ref.params["const"], rel=1e-4)


class TestUnitOfMeasurement:
    @pytest.mark.parametrize(
        "kind, kw",
        [
            ("level", {"order": (1, 0, 1)}),
            ("integrated", {"order": (1, 1, 1)}),
            ("integrated", {"order": (1, 1, 0), "trend": "c"}),
            ("seasonal", {"order": (0, 1, 1), "seasonal_order": (0, 1, 1, 4)}),
        ],
    )
    def test_estimates_forecasts_and_errors_are_equivariant(self, series, kind, kw):
        u = series["u"]
        if kind == "level":
            y = u + 5
        elif kind == "integrated":
            y = np.cumsum(u + 0.5)
        else:
            y = u.copy()
            for t in range(4, N):
                y[t] += 0.5 * y[t - 4]
            y = np.cumsum(y)
        a, b = 1e4, 7e5
        r0, r1 = _fit(y, **kw), _fit(a * y + b, **kw)
        power = {"sigma2": 2, "drift": 1}
        for name in r0.params.index:
            scale = a ** power.get(name, 0)
            if name == "const":
                assert r1.params[name] == pytest.approx(a * r0.params[name] + b)
                scale = a
            else:
                # optimiser tolerance on the same likelihood
                assert r1.params[name] == pytest.approx(
                    scale * r0.params[name], rel=1e-4, abs=1e-6 * scale
                )
            assert r1.se[name] == pytest.approx(scale * r0.se[name], rel=2e-3)
        f0, f1 = r0.forecast(3), r1.forecast(3)
        np.testing.assert_allclose(
            (f1["forecast"] - b) / a, f0["forecast"], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            (f1["upper"] - f1["lower"]) / a, f0["upper"] - f0["lower"], rtol=1e-4
        )
        n_eff = N - kw["order"][1] - (4 if "seasonal_order" in kw else 0)
        assert r1.log_likelihood == pytest.approx(
            r0.log_likelihood - n_eff * np.log(a), abs=1e-5
        )


class TestBoxCox:
    def test_log_transform_equals_a_fit_on_the_logs(self, series):
        level = np.exp(series["u"] / 5 + 2)
        a = _fit(level, order=(1, 0, 0), boxcox=0)
        b = _fit(np.log(level), order=(1, 0, 0))
        np.testing.assert_allclose(a.params, b.params, rtol=1e-10)
        np.testing.assert_allclose(a.residuals, b.residuals, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(a.fitted_values, np.exp(b.fitted_values), rtol=1e-8)
        fa, fb = a.forecast(2), b.forecast(2)
        np.testing.assert_allclose(fa.to_numpy(), np.exp(fb.to_numpy()), rtol=1e-10)

    def test_bias_adjustment_is_the_second_order_mean_correction(self, series):
        level = np.exp(series["u"] / 5 + 2)
        med = _fit(level, order=(1, 0, 0), boxcox=0).forecast(2, level=95)
        mean = _fit(level, order=(1, 0, 0), boxcox=0, biasadj=True).forecast(
            2, level=95
        )
        log_fit = _fit(np.log(level), order=(1, 0, 0))
        phi, s2 = log_fit.params["ar.L1"], log_fit.params["sigma2"]
        var_h = s2 * np.array([1.0, 1.0 + phi**2])
        # The adjustment is the second-order expansion used by R's forecast
        # package, median * (1 + variance / 2) for the logarithm, not the
        # exact lognormal mean median * exp(variance / 2).
        np.testing.assert_allclose(
            mean["forecast"], med["forecast"] * (1 + var_h / 2), rtol=1e-6
        )
        np.testing.assert_allclose(mean["lower_95"], med["lower_95"], rtol=1e-12)


class TestAutomaticSearch:
    def test_the_chosen_model_has_the_smallest_aicc_among_the_candidates(self, series):
        r = _fit(series["u"] + 5, auto=True, stepwise=False, max_p=2, max_q=2)
        table = r.candidates
        assert table["aicc"].min() == pytest.approx(r.aicc, abs=1e-8)
        refit = _fit(series["u"] + 5, order=r.order)
        assert refit.aicc == pytest.approx(r.aicc, abs=1e-6)

    def test_search_with_a_regressor_keeps_the_regressor(self, series):
        r = _fit(series["y"], auto=True, exog=series["x"])
        assert "x1" in r.params.index
        fixed = _fit(series["y"], order=r.order, exog=series["x"])
        assert r.params["x1"] == pytest.approx(fixed.params["x1"], rel=1e-5)


class TestOptionValidation:
    @pytest.mark.parametrize(
        "kw, exc, match",
        [
            ({"trend": "ct"}, MethodIncompatibility, "trend must be None"),
            ({"method": "zzz"}, ValueError, "method must be"),
            ({"method": "css"}, MethodIncompatibility, "conditional sum of squares"),
            ({"period": 0, "auto": True}, MethodIncompatibility, "positive integer"),
            ({"period": 4}, MethodIncompatibility, "needs auto=True"),
            (
                {"period": 12, "seasonal_order": (1, 0, 0, 4)},
                MethodIncompatibility,
                "contradicts seasonal_order",
            ),
        ],
    )
    def test_unknown_or_contradictory_options(self, series, kw, exc, match):
        with pytest.raises(exc, match=match):
            sp.arima(series["u"], **kw)

    def test_column_name_needs_data(self, series):
        with pytest.raises(MethodIncompatibility, match="names a column"):
            sp.arima("y")
        with pytest.raises(MethodIncompatibility, match="name of one of its columns"):
            sp.arima("nope", data=pd.DataFrame({"y": series["u"]}))
        with pytest.raises(MethodIncompatibility, match="no non-missing"):
            sp.arima("y", data=pd.DataFrame({"y": [np.nan] * 5}))


class TestDegenerateSeries:
    def test_constant_series_is_refused(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(StatsPAIError):
                sp.arima(np.ones(100), order=(1, 0, 0))

    def test_exactly_linear_series_differenced_is_refused(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(StatsPAIError):
                sp.arima(np.arange(100.0), order=(1, 1, 0))

    def test_fewer_observations_than_parameters_is_refused(self, series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(StatsPAIError):
                sp.arima(series["u"][:3], order=(2, 0, 1))

    def test_missing_value_inside_the_series_is_named(self, series):
        y = series["u"].copy()
        y[50] = np.nan
        with pytest.raises(StatsPAIError, match="(?i)missing|nan"):
            sp.arima(y, order=(1, 0, 0))

    def test_two_dimensional_input_is_refused(self, series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(StatsPAIError):
                sp.arima(np.column_stack([series["u"], series["x"]]))


class TestScreening:
    """Input checks added with the scan fixes."""

    def test_single_column_array_is_the_series(self, series):
        a = _fit(series["u"].reshape(-1, 1), order=(1, 0, 0))
        b = _fit(series["u"], order=(1, 0, 0))
        np.testing.assert_array_equal(a.params.to_numpy(), b.params.to_numpy())

    def test_trend_regressor_repeats_the_drift_of_a_differenced_model(self, series):
        level = np.cumsum(series["y"])
        xs = np.cumsum(series["x"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = sp.arima(
                level,
                order=(1, 1, 0),
                trend="c",
                exog=np.column_stack([xs, np.arange(N)]),
            )
        ref = _fit(level, order=(1, 1, 0), trend="c", exog=xs)
        assert any("x2 omitted" in str(w.message) for w in caught)
        assert r.model_info["omitted"][0]["variable"] == "x2"
        np.testing.assert_array_equal(r.params.to_numpy(), ref.params.to_numpy())

    def test_forecast_when_every_regressor_was_omitted(self, series):
        r = _fit(series["u"] + 5, order=(1, 0, 0), exog=np.ones(N))
        ref = _fit(series["u"] + 5, order=(1, 0, 0))
        assert r.exog_names == ()
        pd.testing.assert_frame_equal(r.forecast(3, exog=np.ones(3)), ref.forecast(3))
        pd.testing.assert_frame_equal(r.forecast(3), ref.forecast(3))

    def test_automatic_search_omits_a_redundant_regressor(self, series):
        both = np.column_stack([series["x"], np.ones(N)])
        r = _fit(series["y"], auto=True, exog=both)
        ref = _fit(series["y"], auto=True, exog=series["x"])
        assert [o["variable"] for o in r.model_info["omitted"]] == ["x2"]
        assert r.order == ref.order
        np.testing.assert_array_equal(r.params.to_numpy(), ref.params.to_numpy())

    def test_a_large_level_is_not_mistaken_for_a_constant(self, series):
        # unit innovations on a level of 1e9: variation 1e-9 of the level
        walk = 1e9 + np.cumsum(series["u"])
        r = _fit(walk, order=(0, 1, 0))
        assert r.params["sigma2"] == pytest.approx(np.mean(np.diff(walk) ** 2))


class TestOmittedRegressorRecord:
    def test_model_info_is_there_when_nothing_was_omitted(self, series):
        fit = _fit(series["y"], order=(1, 0, 0), exog=series["x"])
        assert fit.model_info == {"omitted": []}
        assert fit.to_dict()["model_info"] == {"omitted": []}
        assert _fit(series["u"], order=(1, 0, 0)).model_info == {"omitted": []}

    def test_linear_regressor_under_two_differences_is_omitted(self, series):
        """A linear trend differences twice to rounding noise, not to exact
        zeros; its coefficient is not identified and the fit is the one
        without it."""
        y = np.cumsum(np.cumsum(series["u"]))
        trend = 0.1 * np.arange(N)
        assert np.any(np.diff(trend, 2) != 0.0)  # the case the test is about
        with pytest.warns(UserWarning, match="omitted because of collinearity"):
            fit = sp.arima(y, order=(0, 2, 1), exog=trend)
        ref = _fit(y, order=(0, 2, 1))
        assert [o["variable"] for o in fit.model_info["omitted"]] == ["x1"]
        assert list(fit.params.index) == list(ref.params.index)
        np.testing.assert_array_equal(fit.params.to_numpy(), ref.params.to_numpy())
        np.testing.assert_array_equal(fit.se.to_numpy(), ref.se.to_numpy())
        assert fit.log_likelihood == ref.log_likelihood

    def test_linear_regressor_is_omitted_next_to_an_identified_one(self, series):
        y = np.cumsum(np.cumsum(series["u"])) + 2 * series["x"]
        both = np.column_stack([0.1 * np.arange(N), series["x"]])
        with pytest.warns(UserWarning, match="x1 omitted"):
            fit = sp.arima(y, order=(0, 2, 1), exog=both)
        ref = _fit(y, order=(0, 2, 1), exog=series["x"])
        assert [o["variable"] for o in fit.model_info["omitted"]] == ["x1"]
        np.testing.assert_allclose(
            fit.params.to_numpy(), ref.params.to_numpy(), rtol=1e-10
        )
        assert np.all(np.isfinite(fit.se.to_numpy())) and np.all(fit.se > 0)


class TestTooShortAfterDifferencing:
    @pytest.mark.parametrize("order", [(0, 1, 0), (0, 2, 0)])
    def test_fewer_than_two_differenced_observations_are_refused(self, order):
        from statspai.exceptions import DataInsufficient

        y = np.array([1.0, 2.5, 2.0])[: order[1] + 1]
        with pytest.raises(DataInsufficient, match="after differencing"):
            sp.arima(y, order=order)
