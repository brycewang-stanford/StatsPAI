"""Degenerate and mis-ordered inputs to ``statspai.timeseries`` (October
review).

Each test asserts one of: equality with the result on sorted, re-indexed or
cleaned data; an identity; or a specific exception or warning class. No
test asserts that a number merely differs from another.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2020-01-01", periods=400)
    mkt = rng.normal(0, 0.01, 400)
    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "id": j,
                    "date": days,
                    "ret": 1.1 * mkt + rng.normal(0, 0.01, 400),
                    "mkt": mkt,
                }
            )
            for j in range(10)
        ]
    )
    events = pd.concat(
        [
            pd.DataFrame({"id": range(10), "event_date": days[280]}),
            pd.DataFrame({"id": range(10), "event_date": days[350]}),
        ]
    )
    return data, events


@pytest.fixture(scope="module")
def coint():
    rng = np.random.default_rng(1)
    common = np.cumsum(rng.normal(size=400))
    return pd.DataFrame(
        {
            "y": common + rng.normal(scale=0.5, size=400),
            "x": 0.5 * common + rng.normal(scale=0.5, size=400),
        }
    )


@pytest.fixture(scope="module")
def dated():
    rng = np.random.default_rng(2)
    n = 120
    x = rng.normal(size=n)
    d = pd.DataFrame({"d": pd.date_range("2000-01", periods=n, freq="MS"), "x": x})
    d["y"] = 1 + 0.5 * x + np.where(np.arange(n) < 60, 0, 1.5) + rng.normal(size=n)
    return d


@pytest.fixture(scope="module")
def monthly():
    rng = np.random.default_rng(3)
    t = np.arange(96)
    return pd.Series(
        100 + t + 10 * np.sin(t * 2 * np.pi / 12) + rng.normal(size=96),
        index=pd.date_range("2010-01", periods=96, freq="MS"),
        name="y",
    )


# ------------------------------------------------------ 1 abnormal_returns
class TestAbnormalReturns:
    def test_repeated_event_labels_are_keyed_by_position(self, returns):
        data, events = returns
        assert not events.index.is_unique
        got = sp.abnormal_returns(data, events, market="mkt")
        ref = sp.abnormal_returns(data, events.reset_index(drop=True), market="mkt")
        assert got.n_events == 20
        pd.testing.assert_frame_equal(got.events, ref.events)
        pd.testing.assert_frame_equal(got.tests, ref.tests)

    def test_unique_labels_are_kept(self, returns):
        data, events = returns
        ev = events.reset_index(drop=True)
        ev.index = [f"e{i}" for i in range(len(ev))]
        got = sp.abnormal_returns(data, ev, market="mkt")
        assert list(got.events.index) == list(ev.index)

    @pytest.mark.parametrize(
        "change, kwargs",
        [
            (lambda d: d.assign(ret=d["mkt"]), {"market": "mkt"}),
            (
                lambda d: d.assign(ret=d["mkt"]),
                {"market": "mkt", "model": "market_adjusted"},
            ),
            (lambda d: d.assign(ret=0.0), {"model": "mean_adjusted"}),
        ],
    )
    def test_returns_the_model_reproduces_exactly_are_refused(
        self, returns, change, kwargs
    ):
        data, events = returns
        ev = events.reset_index(drop=True)
        with pytest.warns(UserWarning, match="no residual variance"):
            with pytest.raises(DataInsufficient):
                sp.abnormal_returns(change(data), ev, **kwargs)

    def test_one_exact_security_is_left_out_and_the_rest_unchanged(self, returns):
        data, events = returns
        ev = events.reset_index(drop=True)
        flat = data.copy()
        flat.loc[flat["id"] == 0, "ret"] = flat.loc[flat["id"] == 0, "mkt"]
        with pytest.warns(UserWarning, match="2 event"):
            got = sp.abnormal_returns(flat, ev, market="mkt")
        ref = sp.abnormal_returns(data, ev[ev["id"] != 0], market="mkt")
        assert got.n_events == 18
        pd.testing.assert_frame_equal(got.events, ref.events)
        assert {s["reason"] for s in got.skipped} == {
            "no residual variance in the estimation window"
        }

    @pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0, -0.1])
    def test_alpha_is_validated(self, returns, alpha):
        data, events = returns
        with pytest.raises(MethodIncompatibility, match="alpha"):
            sp.abnormal_returns(data, events, market="mkt", alpha=alpha)


# ------------------------------------------------------- 2 row order, time=
class TestTimeColumn:
    def test_zivot_andrews_sorts_by_time(self, dated):
        ref = sp.zivot_andrews(dated, "y", time="d")
        got = sp.zivot_andrews(dated.sample(frac=1, random_state=0), "y", time="d")
        assert got.statistic == ref.statistic
        assert got.break_label == ref.break_label
        assert got.lags == ref.lags

    @pytest.mark.parametrize(
        "call",
        [
            lambda d: sp.unitroot(d, "y", time="d"),
            lambda d: sp.zivot_andrews(d, "y", time="d"),
            lambda d: sp.dlm("y ~ x", d, time="d"),
            lambda d: sp.tvp_var(d, ["y", "x"], time="d"),
            lambda d: sp.tvp_var_sv(d, ["y", "x"], time="d", draws=5, burnin=2),
            lambda d: sp.ardl(d, "y", "x", time="d"),
        ],
        ids=["unitroot", "zivot_andrews", "dlm", "tvp_var", "tvp_var_sv", "ardl"],
    )
    def test_repeated_dates_are_refused(self, dated, call):
        twice = dated.copy()
        twice.loc[5, "d"] = twice.loc[4, "d"]
        with pytest.raises(MethodIncompatibility, match="repeated"):
            call(twice)


# ------------------------------------------------- 3, 6, 8 VAR-type systems
class TestSystems:
    def test_vec_refuses_a_gap_and_trims_the_ends(self, coint):
        gap = coint.copy()
        gap.iloc[100, 0] = np.nan
        with pytest.raises(MethodIncompatibility, match="inside the sample"):
            sp.vec(gap, ["y", "x"])
        ends = coint.copy()
        ends.iloc[0, 0] = np.nan
        ends.iloc[-1, 1] = np.nan
        got = sp.vec(ends, ["y", "x"])
        ref = sp.vec(coint.iloc[1:-1], ["y", "x"])
        assert got.n_obs == ref.n_obs
        np.testing.assert_array_equal(got.beta.to_numpy(), ref.beta.to_numpy())
        assert got.log_likelihood == ref.log_likelihood

    @pytest.mark.parametrize(
        "call",
        [
            lambda d: sp.varsoc(d, maxlag=2),
            lambda d: sp.vec(d),
            lambda d: sp.johansen(d),
            lambda d: sp.johansen_lrtest(d, rank=1, beta_known=[1, -2, 0]),
            lambda d: sp.var(d, lags=1),
            lambda d: sp.tvp_var(d),
            lambda d: sp.tvp_var(d, method="forgetting"),
            lambda d: sp.tvp_var_sv(d, draws=5, burnin=2, seed=0),
        ],
        ids=[
            "varsoc",
            "vec",
            "johansen",
            "johansen_lrtest",
            "var",
            "tvp_var_kalman",
            "tvp_var_forgetting",
            "tvp_var_sv",
        ],
    )
    @pytest.mark.parametrize("extra", ["duplicate", "year"])
    def test_dependent_series_are_refused_by_name(self, coint, call, extra):
        if extra == "duplicate":
            frame, name = coint.assign(z=coint["y"]), "z"
        else:
            frame, name = coint.assign(year=np.arange(1900, 2300)), "year"
        with pytest.raises(MethodIncompatibility) as err:
            call(frame)
        assert name in str(err.value)
        assert "collinear" in str(err.value) or "dependent" in str(err.value)

    def test_the_check_leaves_valid_systems_alone(self, coint):
        from statspai.timeseries.var import _require_independent_series

        _require_independent_series(coint.to_numpy(), list(coint.columns), "t")
        # two rows cannot tell a dependence from a short sample
        _require_independent_series(coint.to_numpy()[:2], ["y", "x"], "t")

    def test_johansen_lrtest_asks_for_the_sample_vec_asks_for(self):
        rng = np.random.default_rng(0)
        trend = np.cumsum(rng.normal(size=9))
        frame = pd.DataFrame(
            {
                "c": trend + rng.normal(size=9),
                "y": trend + rng.normal(size=9),
                "r": np.cumsum(rng.normal(size=9)),
            }
        )
        for n in (6, 8, 9):
            with pytest.raises(DataInsufficient):
                sp.vec(frame.iloc[:n], lags=1, rank=1)
            with pytest.raises(DataInsufficient):
                sp.johansen_lrtest(frame.iloc[:n], rank=1, beta_known=[1, -1, 0])

    @pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
    def test_alpha_is_validated(self, coint, dated, alpha):
        with pytest.raises(MethodIncompatibility, match="alpha"):
            sp.varsoc(coint, maxlag=2, alpha=alpha)
        with pytest.raises(MethodIncompatibility, match="alpha"):
            sp.vec(coint, alpha=alpha)
        with pytest.raises(MethodIncompatibility, match="alpha"):
            sp.chow_test(dated, "y", ["x"], break_point=60, alpha=alpha)


# ------------------------------------------------------ 4 forecast accuracy
class TestForecastAccuracy:
    def test_percentage_errors_are_undefined_at_a_zero_actual(self):
        with pytest.warns(UserWarning, match="1 of the 3 actual values are zero"):
            acc = sp.forecast_accuracy([0.0, 1.0, 2.0], [1.0, 1.0, 2.0])
        assert np.isnan(acc.loc["forecast", "MAPE"])
        assert np.isnan(acc.loc["forecast", "MPE"])
        assert acc.loc["forecast", "MAE"] == pytest.approx(1.0 / 3.0)

    def test_mape_without_zeros_is_the_textbook_mean(self):
        a, f = np.array([2.0, 4.0, 5.0]), np.array([1.0, 5.0, 5.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            acc = sp.forecast_accuracy(a, f)
        assert acc.loc["forecast", "MAPE"] == pytest.approx(
            100 * np.mean(np.abs((a - f) / a))
        )
        assert acc.loc["forecast", "MPE"] == pytest.approx(100 * np.mean((a - f) / a))

    def test_crps_needs_intervals(self):
        with pytest.raises(MethodIncompatibility, match="prediction intervals"):
            sp.forecast_accuracy([1.0, 2.0, 3.0], [1.0, 1.0, 2.0], crps=True)

    def test_tscv_mape_is_undefined_at_a_zero_actual(self):
        y = np.r_[np.arange(1.0, 13.0), 0.0, 3.0, 4.0]
        cv = sp.tscv(y, "naive", horizon=1, initial=10)
        with pytest.warns(UserWarning, match="zero"):
            acc = cv.accuracy()
        assert np.isnan(acc.loc["h=1", "MAPE"])
        e = cv.errors["h=1"].to_numpy()
        assert acc.loc["h=1", "MAE"] == pytest.approx(np.mean(np.abs(e)))

    def test_tscv_refuses_a_forecaster_that_returns_too_many_values(self):
        y = np.cumsum(np.random.default_rng(0).normal(size=40)) + 50
        with pytest.raises(DataInsufficient, match="returned 3 values, 2 needed"):
            sp.tscv(y, lambda tr, h: np.full(h + 1, tr.iloc[-1]), horizon=2)
        ok = sp.tscv(y, lambda tr, h: np.full(h, tr.iloc[-1]), horizon=2)
        ref = sp.tscv(y, "naive", horizon=2)
        pd.testing.assert_frame_equal(ok.errors, ref.errors)


# ------------------------------------------------------------ 5 exact fits
class TestExactFits:
    @pytest.mark.parametrize("seed", range(4))
    def test_chow_refuses_an_exact_fit(self, seed):
        x = np.random.default_rng(seed).normal(size=120)
        for y in (1 + 2 * x, np.full(120, 3.0)):
            with pytest.raises(DataInsufficient, match="exactly"):
                sp.chow_test(pd.DataFrame({"y": y, "x": x}), "y", ["x"], break_point=60)

    def test_chow_guard_is_relative_to_the_scale_of_the_outcome(self, dated):
        ref = sp.chow_test(dated, "y", ["x"], break_point=60)
        tiny = dated.assign(y=dated["y"] * 1e-12)
        got = sp.chow_test(tiny, "y", ["x"], break_point=60)
        assert got["statistic"] == pytest.approx(ref["statistic"], rel=1e-9)

    def test_ardl_refuses_a_constant_outcome(self):
        x = np.random.default_rng(0).normal(size=300)
        frame = pd.DataFrame({"y": np.full(300, 3.0), "x": x})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(DataInsufficient, match="constant"):
                sp.ardl(frame, "y", "x")
            with pytest.raises(DataInsufficient, match="constant"):
                sp.ardl(frame, "y", lags="bic")

    def test_ardl_refuses_an_outcome_its_regressors_reproduce(self):
        x = np.random.default_rng(0).normal(size=200)
        frame = pd.DataFrame({"y": 1 + 2 * x, "x": x})
        with pytest.raises(DataInsufficient, match="exactly"):
            _quiet(lambda: sp.ardl(frame, "y", "x", contemporaneous=True))


# ------------------------------------------------- 7 redundant regressors
class TestRedundantRegressors:
    @pytest.fixture(scope="class")
    def switching(self):
        rng = np.random.default_rng(1)
        s = np.zeros(300, dtype=int)
        for t in range(1, 300):
            stay = 0.95 if s[t - 1] == 0 else 0.9
            s[t] = s[t - 1] if rng.uniform() < stay else 1 - s[t - 1]
        return np.where(s == 0, -1.0, 2.0) + rng.normal(size=300)

    def test_mswitch_omits_a_constant_regressor(self, switching):
        ref = sp.mswitch(switching, seed=0)
        with pytest.warns(UserWarning, match="omitted because of collinearity"):
            got = sp.mswitch(switching, x=np.ones(300), seed=0)
        assert got.model_info["omitted"][0]["variable"] == "x1"
        assert ref.model_info["omitted"] == []
        pd.testing.assert_frame_equal(got.params, ref.params)
        assert got.loglik == ref.loglik
        assert got.n_params == ref.n_params

    def test_mswitch_omits_the_common_copy_of_a_switching_regressor(self, switching):
        z = np.random.default_rng(5).normal(size=300)
        frame = pd.DataFrame({"y": switching, "z": z, "w": z})
        ref = sp.mswitch("y", data=frame, switch="z", seed=0)
        with pytest.warns(UserWarning, match="note: w omitted"):
            got = sp.mswitch("y", x="w", data=frame, switch="z", seed=0)
        pd.testing.assert_frame_equal(got.params, ref.params)

    @pytest.mark.parametrize(
        "formula, clean, gone",
        [("y ~ x + x2", "y ~ x", "x2"), ("y ~ x + c", "y ~ x", "c")],
    )
    def test_dlm_omits_a_redundant_regressor(self, formula, clean, gone):
        rng = np.random.default_rng(4)
        frame = pd.DataFrame({"x": rng.normal(size=150)})
        frame["x2"] = 2 * frame["x"]
        frame["c"] = 1.0
        frame["y"] = frame["x"] + rng.normal(size=150)
        ref = sp.dlm(clean, frame)
        with pytest.warns(UserWarning, match=f"note: {gone} omitted"):
            got = sp.dlm(formula, frame)
        assert [o["variable"] for o in got.model_info["omitted"]] == [gone]
        assert ref.model_info["omitted"] == []
        pd.testing.assert_frame_equal(got.smoothed, ref.smoothed)
        pd.testing.assert_series_equal(got.params, ref.params)
        assert got.loglik == ref.loglik


# ------------------------------------------------------- 9 raw tracebacks
class TestClearErrors:
    def test_ardl_outcome_listed_as_regressor(self):
        frame = pd.DataFrame({"y": np.random.default_rng(0).normal(size=80)})
        with pytest.raises(MethodIncompatibility, match="outcome"):
            sp.ardl(frame, "y", "y")

    def test_tvp_var_sv_training_longer_than_the_sample(self):
        frame = pd.DataFrame(
            np.random.default_rng(0).normal(size=(160, 2)), columns=["x", "z"]
        )
        for training in (159, 160, 200):
            with pytest.raises(DataInsufficient, match="tvp_var_sv"):
                sp.tvp_var_sv(frame, training=training, draws=5, burnin=2, seed=0)

    def test_boxcox_lambda_bounds(self, monthly):
        with pytest.raises(MethodIncompatibility, match="lower"):
            sp.boxcox_lambda(monthly, 12, lower=2, upper=-1)
        with pytest.raises(MethodIncompatibility, match="lower"):
            sp.boxcox_lambda(monthly, 12, lower=1, upper=1)

    def test_svar_pattern_given_as_a_word(self, coint):
        fit = sp.var(coint.diff().dropna(), lags=1)
        with pytest.raises(MethodIncompatibility, match="not a numeric"):
            sp.svar(fit, B="chol")


# ------------------------------------------------- 10 order of a date index
class TestDateIndexOrder:
    CALLS = {
        "ets": lambda s: sp.ets(s, model="ANN"),
        "simple_forecast": lambda s: sp.simple_forecast(s, "naive"),
        "stl": lambda s: sp.stl(s, period=12),
        "classical_decompose": lambda s: sp.classical_decompose(s, period=12),
        "tscv": lambda s: sp.tscv(s, "naive", initial=80),
        "bagged_forecast": lambda s: sp.bagged_forecast(s, period=12, n_boot=2, seed=0),
        "ljungbox": lambda s: sp.ljungbox(s),
        "boxcox_lambda": lambda s: sp.boxcox_lambda(s, 12),
        "arima": lambda s: sp.arima(s, order=(1, 0, 0)),
        "garch": lambda s: sp.garch(s.diff().dropna()),
        "unitroot": lambda s: sp.unitroot(s),
        "zivot_andrews": lambda s: sp.zivot_andrews(s),
    }

    @pytest.mark.parametrize("name", sorted(CALLS))
    def test_shuffled_dates_are_refused(self, monthly, name):
        shuffled = monthly.sample(frac=1, random_state=0)
        with pytest.raises(MethodIncompatibility, match="increasing order"):
            _quiet(lambda: self.CALLS[name](shuffled))

    @pytest.mark.parametrize("name", ["ets", "simple_forecast", "arima", "unitroot"])
    def test_period_index_and_repeated_dates(self, monthly, name):
        periods = monthly.copy()
        periods.index = monthly.index.to_period("M")
        with pytest.raises(MethodIncompatibility, match="increasing order"):
            _quiet(lambda: self.CALLS[name](periods.iloc[::-1]))
        twice = monthly.copy()
        twice.index = monthly.index[:-1].insert(0, monthly.index[0])
        with pytest.raises(MethodIncompatibility, match="repeated"):
            _quiet(lambda: self.CALLS[name](twice))

    def test_a_column_of_a_frame_with_shuffled_dates(self, monthly):
        frame = monthly.to_frame().sample(frac=1, random_state=0)
        with pytest.raises(MethodIncompatibility, match="increasing order"):
            sp.ets("y", model="ANN", data=frame)
        with pytest.raises(MethodIncompatibility, match="increasing order"):
            sp.unitroot(frame, "y")

    def test_sorting_gives_the_result_on_the_ordered_series(self, monthly):
        shuffled = monthly.sample(frac=1, random_state=0)
        ref = sp.ets(monthly, model="AAA", period=12).forecast(3)
        got = sp.ets(shuffled.sort_index(), model="AAA", period=12).forecast(3)
        pd.testing.assert_frame_equal(got, ref)

    def test_other_indexes_are_read_in_row_order(self, monthly):
        values = monthly.to_numpy()
        ref = sp.simple_forecast(values, "drift").forecast(2)["forecast"].to_numpy()
        for index in (
            pd.RangeIndex(96)[::-1],
            pd.Index(np.random.default_rng(0).permutation(96)),
            pd.Index([f"r{i}" for i in range(96)]),
        ):
            s = pd.Series(values, index=index)
            got = sp.simple_forecast(s, "drift").forecast(2)["forecast"].to_numpy()
            np.testing.assert_array_equal(got, ref)


# ----------------------------------------------------------- 11 the rest
class TestSmaller:
    def test_ljungbox_needs_positive_degrees_of_freedom(self):
        x = np.random.default_rng(0).normal(size=200)
        # no reference distribution at lags <= model_df: missing, and said so
        with pytest.warns(UserWarning, match="no degrees of freedom"):
            low = sp.ljungbox(x, lags=[2, 3, 5], model_df=3)
        assert low["p_value"].isna().tolist() == [True, True, False]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = sp.ljungbox(x, lags=[4, 6], model_df=3)
        assert list(out["df"]) == [1, 3]
        assert out["p_value"].notna().all()

    def test_boxcox_lambda_constant_series(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(DataInsufficient, match="constant"):
                sp.boxcox_lambda(np.ones(60), 12)
            with pytest.raises(DataInsufficient, match="constant"):
                sp.boxcox_lambda(np.full(60, 1e9), 12)

    @pytest.mark.parametrize("method", ["naive", "mean", "drift"])
    def test_simple_forecast_warns_about_zero_width_intervals(self, method):
        with pytest.warns(UserWarning, match="zero width"):
            fit = sp.simple_forecast(np.full(20, 7.0), method)
        table = fit.forecast(2)
        np.testing.assert_allclose(table["lower_95"], table["upper_95"], atol=1e-9)

    def test_simple_forecast_is_silent_on_an_ordinary_series(self, monthly):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sp.simple_forecast(monthly, "naive")

    @pytest.mark.parametrize("bandwidth", ["andrews", "newey-west", "rule", "sw", 4.0])
    def test_lrvar_refuses_a_constant_under_every_bandwidth(self, bandwidth):
        for level in (1.0, 1e8):
            with pytest.raises(DataInsufficient, match="constant"):
                sp.lrvar(np.full(100, level), bandwidth=bandwidth)

    def test_periodogram_refuses_a_series_with_nothing_left(self):
        with pytest.raises(DataInsufficient, match="nothing is left"):
            sp.periodogram(np.arange(50.0))
        with pytest.raises(DataInsufficient, match="constant"):
            sp.periodogram(np.full(50, 4.0), detrend="mean")
        # the guard is relative: a rescaled series has the rescaled spectrum
        z = 1e6 + np.arange(50.0) + np.random.default_rng(0).normal(size=50)
        ref = sp.periodogram(z).table["spectrum"].to_numpy()
        got = sp.periodogram(1e-9 * z).table["spectrum"].to_numpy()
        np.testing.assert_allclose(got, 1e-18 * ref, rtol=1e-6)

    def test_bds_refuses_an_epsilon_that_separates_nothing(self):
        e = np.random.default_rng(0).normal(size=200)
        with pytest.raises(MethodIncompatibility, match="every pair"):
            sp.bds(e, epsilon=1e6)
        with pytest.raises(MethodIncompatibility, match="no pair"):
            sp.bds(e, epsilon=1e-12)

    def test_reconcile_intervals_take_one_level(self):
        S = pd.DataFrame(
            [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            index=["total", "a", "b"],
            columns=["a", "b"],
        )
        base = pd.DataFrame(
            [[10.0, 6.0, 5.0], [12.0, 7.0, 6.0]], columns=["total", "a", "b"]
        )
        fit = sp.reconcile(base, S, method="ols", sd=np.ones((2, 3)))
        with pytest.raises(MethodIncompatibility, match="one coverage"):
            fit.intervals((80, 95))
        lo, hi = fit.intervals(95)
        lo2, hi2 = fit.intervals(0.95)
        pd.testing.assert_frame_equal(lo, lo2)
        pd.testing.assert_frame_equal(hi, hi2)
        assert ((hi - lo).to_numpy() > 0).all()
