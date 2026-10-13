"""Options and degenerate inputs of the forecasting toolbox:
decompositions, accuracy measures, rolling-origin cross-validation,
portmanteau test, differencing orders, Box-Cox parameter, Fourier terms,
seasonal dummies and series features.

Independent computations are statsmodels (same algorithm, stated where the
conventions are aligned) or the defining formula written out here.
"""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy import integrate, stats  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.exceptions import DataInsufficient, MethodIncompatibility  # noqa: E402


def _monthly(n: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 50 + 0.3 * t + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(size=n)


def _quarterly(n: int = 48, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    season = np.tile([1.1, 0.8, 0.9, 1.2], n // 4 + 1)[:n]
    return (20.0 + 0.2 * t) * season + 0.3 * rng.normal(size=n)


# ----------------------------------------------------------------------
# decompositions
# ----------------------------------------------------------------------
class TestSTL:
    def test_components_add_up_and_strength_follows_its_formula(self):
        y = _monthly()
        dec = sp.stl(y, 12)
        np.testing.assert_allclose(dec.trend + dec.seasonal + dec.remainder, y)
        np.testing.assert_allclose(dec.seasonally_adjusted, y - dec.seasonal)
        vr = np.var(dec.remainder, ddof=1)
        assert dec.strength["trend"] == pytest.approx(
            1 - vr / np.var(dec.trend + dec.remainder, ddof=1)
        )
        assert dec.strength["seasonal"] == pytest.approx(
            1 - vr / np.var(dec.seasonal + dec.remainder, ddof=1)
        )

    def test_default_windows(self):
        dec = sp.stl(_monthly(), 12)
        # trend: smallest odd integer >= 1.5 m / (1 - 1.5 / seasonal);
        # jumps: ceil(window / 10).
        assert dec.settings["seasonal"] == 11
        assert dec.settings["trend"] == 21
        assert dec.settings["low_pass"] == 13
        assert dec.settings["jumps"] == (2, 3, 2)
        assert (dec.settings["inner_iter"], dec.settings["outer_iter"]) == (2, 0)
        rob = sp.stl(_monthly(), 12, robust=True)
        assert (rob.settings["inner_iter"], rob.settings["outer_iter"]) == (1, 15)

    def test_statsmodels_defaults_are_reproduced_with_the_documented_arguments(self):
        from statsmodels.tsa.seasonal import STL

        y = _monthly()
        ours = sp.stl(
            y,
            12,
            seasonal=7,
            seasonal_deg=1,
            inner_iter=5,
            seasonal_jump=1,
            trend_jump=1,
            low_pass_jump=1,
        )
        ref = STL(y, period=12).fit()
        # Same routine with the same arguments.
        np.testing.assert_allclose(ours.seasonal, ref.seasonal, atol=1e-12)
        np.testing.assert_allclose(ours.trend, ref.trend, atol=1e-12)

    def test_robust_fit_keeps_an_outlier_in_the_remainder(self):
        y = _monthly()
        spiked = y.copy()
        spiked[60] += 40.0
        plain = sp.stl(spiked, 12)
        robust = sp.stl(spiked, 12, robust=True)
        clean = sp.stl(y, 12, robust=True)
        # The robust components away from the spike are close to those of
        # the clean series; the non-robust ones are pulled by it.
        err_robust = np.abs(robust.seasonal - clean.seasonal).max()
        err_plain = np.abs(plain.seasonal - clean.seasonal).max()
        assert err_robust < 0.5 < err_plain
        assert robust.remainder[60] > 35.0

    def test_periodic_window_repeats_one_seasonal_figure(self):
        dec = sp.stl(_monthly(), 12, seasonal="periodic")
        cycles = dec.seasonal.reshape(10, 12)
        assert np.ptp(cycles, axis=0).max() == 0.0
        assert dec.settings["seasonal"] == "periodic"

    def test_several_periods_give_one_component_each(self):
        rng = np.random.default_rng(1)
        t = np.arange(240)
        y = (
            10
            + 3 * np.sin(2 * np.pi * t / 6)
            + 2 * np.cos(2 * np.pi * t / 30)
            + 0.3 * rng.normal(size=240)
        )
        dec = sp.stl(y, [30, 6])
        assert dec.method == "MSTL" and dec.periods == (6, 30)
        total = dec.seasonal_components[6] + dec.seasonal_components[30]
        np.testing.assert_allclose(dec.seasonal, total)
        np.testing.assert_allclose(dec.trend + total + dec.remainder, y)
        frame = dec.to_frame()
        assert list(frame.columns) == [
            "observed",
            "trend",
            "seasonal_6",
            "seasonal_30",
            "remainder",
            "seasonally_adjusted",
        ]
        assert set(dec.strength) == {"trend", "seasonal", "seasonal_6", "seasonal_30"}
        assert dec.settings["seasonal"] == [11, 15]
        assert "period 30" in dec.summary()
        # A repeated period is one period.
        assert sp.stl(y, [6, 6]).method == "STL"

    @pytest.mark.parametrize("method", ["naive", "drift", "ets"])
    def test_forecast_is_the_adjusted_series_forecast_plus_the_last_cycle(self, method):
        y = _monthly()
        dec = sp.stl(y, 12)
        fc = dec.forecast(15, level=90, method=method)
        sa = dec.seasonally_adjusted
        if method == "ets":
            inner = sp.ets(sa, "ZZN").forecast(15, 90)
        else:
            inner = sp.simple_forecast(sa, method).forecast(15, 90)
        seas = np.tile(dec.seasonal[-12:], 2)[:15]
        for col in ("forecast", "lower_90", "upper_90"):
            np.testing.assert_allclose(fc[col], inner[col].to_numpy() + seas)
        assert fc.index.tolist() == list(range(120, 135))

    def test_forecast_passes_arguments_to_ets(self):
        dec = sp.stl(_monthly(), 12)
        fc = dec.forecast(3, method="ets", model="ANN", alpha=0.2)
        inner = sp.ets(dec.seasonally_adjusted, "ANN", alpha=0.2).forecast(3)
        np.testing.assert_allclose(
            fc["forecast"], inner["forecast"].to_numpy() + dec.seasonal[-12:-9]
        )

    def test_forecast_by_arima_of_the_adjusted_series(self):
        dec = sp.stl(_monthly(), 12)
        fc = dec.forecast(3, level=90, method="arima", order=(0, 1, 0))
        inner = sp.arima(dec.seasonally_adjusted, order=(0, 1, 0)).forecast(3, level=90)
        np.testing.assert_allclose(
            fc["forecast"], inner["forecast"].to_numpy() + dec.seasonal[-12:-9]
        )

    def test_index_is_carried_and_continued(self):
        idx = pd.period_range("2010-01", periods=120, freq="M")
        y = pd.Series(_monthly(), index=idx, name="sales")
        dec = sp.stl(y, 12)
        assert dec.to_frame().index.equals(idx)
        fc = dec.forecast(2, method="naive")
        assert fc.index.equals(pd.period_range("2020-01", periods=2, freq="M"))
        by_name = sp.stl("sales", 12, data=y.to_frame())
        np.testing.assert_allclose(by_name.trend, dec.trend)

    def test_plot_shows_the_components(self):
        dec = sp.stl(_monthly(), 12)
        fig = dec.plot()
        try:
            labels = [ax.get_ylabel() for ax in fig.axes]
            assert labels == ["observed", "trend", "seasonal", "remainder"]
            np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), dec.trend)
        finally:
            plt.close(fig)

    @pytest.mark.parametrize(
        "args, kwargs, exc, match",
        [
            ((1,), {}, MethodIncompatibility, "at least 2"),
            ((70,), {}, DataInsufficient, "fewer than two cycles"),
            ((12,), dict(seasonal=8), MethodIncompatibility, "odd integer"),
            ((12,), dict(seasonal="smooth"), MethodIncompatibility, "not a window"),
            ((12,), dict(trend=10), MethodIncompatibility, "trend=10"),
            (([6, 12],), dict(seasonal=[11]), MethodIncompatibility, "1 seasonal"),
            ((2.5,), {}, MethodIncompatibility, "positive integer"),
        ],
    )
    def test_refusals(self, args, kwargs, exc, match):
        with pytest.raises(exc, match=match):
            sp.stl(_monthly(), *args, **kwargs)

    def test_forecast_refusals(self):
        dec = sp.stl(_monthly(), 12)
        with pytest.raises(MethodIncompatibility, match="method='mean'"):
            dec.forecast(2, method="mean")
        with pytest.raises(MethodIncompatibility, match="horizon must be at least 1"):
            dec.forecast(0, method="naive")

    def test_odd_period_differs_from_r_by_the_documented_low_pass_window(self):
        t = np.arange(84.0)
        y = (
            20
            + 0.05 * t
            + 3 * np.sin(2 * np.pi * t / 7)
            + np.cos(1.7 * t)
            + 0.5 * np.sin(0.3 * t * t)
        )
        dec = sp.stl(y, 7)
        # R takes the low-pass window equal to an odd period; the smoother
        # used here needs a window above the period and gets period + 2.
        assert dec.settings["low_pass"] == 9
        assert (dec.settings["seasonal"], dec.settings["trend"]) == (11, 13)
        # stats::stl(ts(y, frequency = 7), s.window = 11), R 4.x, rows
        # 1, 2, 3, 40 and 84; windows reported by R: 11, 13, 7.
        rows = [0, 1, 2, 39, 83]
        seasonal = [
            0.106853991185,
            2.666324683991,
            2.999344901402,
            -1.728904490026,
            -2.389262900884,
        ]
        trend = [
            20.339862121755,
            20.327344997247,
            20.314827872739,
            22.084096258946,
            24.072669905343,
        ]
        # The documented difference: close to R (1e-3 on values around 20;
        # the docstring quotes 2e-5 to 2.5e-4 for these rows) ...
        np.testing.assert_allclose(dec.seasonal[rows], seasonal, atol=1e-3)
        np.testing.assert_allclose(dec.trend[rows], trend, atol=1e-3)
        # ... and explicitly not equal to it, as an even period is (1e-9
        # in the parity suite). If this starts to hold, R's window has
        # become available: drop the paragraph in the docstring.
        gap = max(
            np.max(np.abs(dec.seasonal[rows] - seasonal)),
            np.max(np.abs(dec.trend[rows] - trend)),
        )
        assert gap > 1e-5

    def test_low_pass_window_the_smoother_cannot_honour_is_refused(self):
        y = _monthly()
        # Not replaced silently: an odd period's own value (R's default), an
        # even window, a window below the period.
        for period, bad, smallest in ((7, 7, 9), (12, 14, 13), (12, 11, 13)):
            with pytest.raises(MethodIncompatibility, match=f"available is {smallest}"):
                sp.stl(y, period, low_pass=bad)
        # The smallest accepted value is the default.
        for period, smallest in ((7, 9), (12, 13)):
            a, b = sp.stl(y, period), sp.stl(y, period, low_pass=smallest)
            np.testing.assert_array_equal(a.trend, b.trend)
        assert sp.stl(y, 12, low_pass=15).settings["low_pass"] == 15


class TestClassicalDecomposition:
    @pytest.mark.parametrize("model", ["additive", "multiplicative"])
    @pytest.mark.parametrize("period, n", [(4, 48), (4, 46), (5, 48)])
    def test_equals_statsmodels(self, model, period, n):
        from statsmodels.tsa.seasonal import seasonal_decompose

        y = _quarterly(n)
        ours = sp.classical_decompose(y, period, model=model)
        ref = seasonal_decompose(y, period=period, model=model)
        # Same definition: centred moving average, period means of the
        # detrended series, centred to sum to zero (average one).
        np.testing.assert_allclose(ours.trend, ref.trend, atol=1e-12)
        np.testing.assert_allclose(ours.seasonal, ref.seasonal, atol=1e-12)
        np.testing.assert_allclose(ours.remainder, ref.resid, atol=1e-12)

    def test_components_recombine(self):
        y = _quarterly()
        add = sp.classical_decompose(y, 4)
        ok = np.isfinite(add.trend)
        assert (~ok).sum() == 4  # two at each end for a 2 x 4 average
        np.testing.assert_allclose(
            (add.trend + add.seasonal + add.remainder)[ok], y[ok]
        )
        assert add.seasonal[:4].sum() == pytest.approx(0.0, abs=1e-12)
        mult = sp.classical_decompose(y, 4, model="multiplicative")
        np.testing.assert_allclose(
            (mult.trend * mult.seasonal * mult.remainder)[ok], y[ok]
        )
        assert mult.seasonal[:4].mean() == pytest.approx(1.0)
        np.testing.assert_allclose(mult.seasonally_adjusted, y / mult.seasonal)
        # Strength on the log scale for a multiplicative decomposition.
        r, s = np.log(mult.remainder[ok]), np.log(mult.seasonal[ok])
        assert mult.strength["seasonal"] == pytest.approx(
            1 - np.var(r, ddof=1) / np.var(s + r, ddof=1)
        )

    def test_forecast_of_an_additive_decomposition(self):
        y = _quarterly()
        dec = sp.classical_decompose(y, 4)
        fc = dec.forecast(6, method="naive")
        np.testing.assert_allclose(
            fc["forecast"], (y - dec.seasonal)[-1] + np.tile(dec.seasonal[-4:], 2)[:6]
        )

    def test_refusals(self):
        y = _quarterly()
        with pytest.raises(MethodIncompatibility, match="model='x'"):
            sp.classical_decompose(y, 4, model="x")
        with pytest.raises(DataInsufficient, match="fewer than two cycles"):
            sp.classical_decompose(y[:7], 4)
        with pytest.raises(DataInsufficient, match="period >= 2"):
            sp.classical_decompose(y, 1)
        with pytest.raises(MethodIncompatibility, match="strictly positive"):
            sp.classical_decompose(y - 30, 4, model="multiplicative")
        mult = sp.classical_decompose(y, 4, model="multiplicative")
        with pytest.raises(MethodIncompatibility, match="multiplicative"):
            mult.forecast(2)


# ----------------------------------------------------------------------
# accuracy measures
# ----------------------------------------------------------------------
class TestForecastAccuracy:
    def _case(self):
        y = _quarterly(60)
        train, test = y[:48], y[48:]
        fc = sp.simple_forecast(train, "drift").forecast(12, level=(80, 95))
        return train, test, fc

    def test_point_measures_equal_their_formulas(self):
        train, test, fc = self._case()
        out = sp.forecast_accuracy(test, fc, train=train, period=4).iloc[0]
        e = test - fc["forecast"].to_numpy()
        assert out["ME"] == pytest.approx(e.mean())
        assert out["RMSE"] == pytest.approx(np.sqrt(np.mean(e**2)))
        assert out["MAE"] == pytest.approx(np.abs(e).mean())
        assert out["MPE"] == pytest.approx(100 * np.mean(e / test))
        assert out["MAPE"] == pytest.approx(100 * np.mean(np.abs(e / test)))
        # Scaled by the in-sample seasonal naive errors.
        d = train[4:] - train[:-4]
        assert out["MASE"] == pytest.approx(np.abs(e).mean() / np.abs(d).mean())
        assert out["RMSSE"] == pytest.approx(np.sqrt(np.mean(e**2) / np.mean(d**2)))
        c = e - e.mean()
        assert out["ACF1"] == pytest.approx((c[1:] @ c[:-1]) / (c @ c))

    def test_period_changes_the_scaling_only(self):
        train, test, fc = self._case()
        a = sp.forecast_accuracy(test, fc, train=train, period=1).iloc[0]
        b = sp.forecast_accuracy(test, fc, train=train, period=4).iloc[0]
        e = test - fc["forecast"].to_numpy()
        assert a["MASE"] == pytest.approx(
            np.abs(e).mean() / np.abs(np.diff(train)).mean()
        )
        assert a["RMSE"] == b["RMSE"] and a["MASE"] != b["MASE"]

    def test_interval_scores(self):
        train, test, fc = self._case()
        shifted = fc.copy()
        # Move the table so that some actual values fall outside.
        for col in shifted.columns:
            shifted[col] = shifted[col] + 1.5
        out = sp.forecast_accuracy(test, shifted, crps=True).iloc[0]
        for pct in (80, 95):
            lo = shifted[f"lower_{pct}"].to_numpy()
            hi = shifted[f"upper_{pct}"].to_numpy()
            a = 1 - pct / 100
            winkler = np.where(
                test < lo,
                (hi - lo) + 2 / a * (lo - test),
                np.where(test > hi, (hi - lo) + 2 / a * (test - hi), hi - lo),
            )
            assert out[f"winkler_{pct}"] == pytest.approx(winkler.mean())
            assert out[f"coverage_{pct}"] == pytest.approx(
                np.mean((test >= lo) & (test <= hi))
            )
        assert 0.0 < out["coverage_80"] < 1.0
        # CRPS of the normal forecast implied by the widest interval, by
        # numerical integration of (F(x) - 1{x >= y})^2.
        mu = shifted["forecast"].to_numpy()
        sd = (shifted["upper_95"] - shifted["lower_95"]).to_numpy() / (
            2 * stats.norm.ppf(0.975)
        )
        vals = []
        for m, s, obs in zip(mu, sd, test):
            left = integrate.quad(
                lambda x: stats.norm.cdf(x, m, s) ** 2, m - 12 * s, obs
            )[0]
            right = integrate.quad(
                lambda x: stats.norm.sf(x, m, s) ** 2, obs, m + 12 * s
            )[0]
            vals.append(left + right)
        # 1e-6: quadrature error.
        assert out["CRPS"] == pytest.approx(np.mean(vals), rel=1e-6)

    def test_inputs(self):
        train, test, fc = self._case()
        point = fc["forecast"].to_numpy()
        both = sp.forecast_accuracy(
            pd.Series(test), {"table": fc, "array": point}, train=pd.Series(train)
        )
        assert both.index.tolist() == ["table", "array"]
        assert both.loc["table", "RMSE"] == both.loc["array", "RMSE"]
        assert np.isnan(both.loc["array", "winkler_80"])
        # No training sample: nothing to scale by.
        bare = sp.forecast_accuracy(test, point).iloc[0]
        assert np.isnan(bare["MASE"]) and np.isnan(bare["RMSSE"])
        # A training sample no longer than the period, or a constant one.
        short = sp.forecast_accuracy(test, point, train=train[:4], period=4).iloc[0]
        assert np.isnan(short["MASE"])
        flat = sp.forecast_accuracy(test, point, train=np.ones(20)).iloc[0]
        assert np.isnan(flat["MASE"])

    def test_missing_pairs_are_left_out(self):
        train, test, fc = self._case()
        point = fc["forecast"].to_numpy()
        holed = test.copy()
        holed[[2, 7]] = np.nan
        keep = np.isfinite(holed)
        a = sp.forecast_accuracy(holed, point).iloc[0]
        b = sp.forecast_accuracy(test[keep], point[keep]).iloc[0]
        for k in ("ME", "RMSE", "MAE", "MPE", "MAPE"):
            assert a[k] == pytest.approx(b[k])

    def test_refusals(self):
        train, test, fc = self._case()
        with pytest.raises(MethodIncompatibility, match="12 forecasts for 5"):
            sp.forecast_accuracy(test[:5], fc)
        with pytest.raises(MethodIncompatibility, match="'forecast' column"):
            sp.forecast_accuracy(test, fc.drop(columns="forecast"))
        with pytest.raises(DataInsufficient, match="no pair"):
            sp.forecast_accuracy(np.full(12, np.nan), fc)
        with pytest.raises(MethodIncompatibility, match="needs prediction intervals"):
            sp.forecast_accuracy(test, fc[["forecast"]], crps=True)
        with pytest.raises(MethodIncompatibility, match="positive integer"):
            sp.forecast_accuracy(test, fc, period=0)


# ----------------------------------------------------------------------
# rolling-origin cross-validation
# ----------------------------------------------------------------------
class TestTSCV:
    @pytest.mark.parametrize("method", ["naive", "drift", "mean", "snaive"])
    def test_errors_equal_forecasts_refitted_at_every_origin(self, method):
        y = _quarterly(40)
        cv = sp.tscv(y, method, horizon=3, initial=20, step=4, period=4)
        origins = list(range(20, 40, 4))
        assert cv.errors.index.tolist() == [o - 1 for o in origins]
        for row, end in enumerate(origins):
            fc = sp.simple_forecast(y[:end], method, period=4).forecast(3)
            k = min(3, 40 - end)
            np.testing.assert_allclose(
                cv.forecasts.iloc[row], fc["forecast"].to_numpy()
            )
            np.testing.assert_allclose(cv.actuals.iloc[row, :k], y[end : end + k])
            np.testing.assert_allclose(
                cv.errors.iloc[row, :k], y[end : end + k] - fc["forecast"].iloc[:k]
            )
            # Beyond the end of the series there is nothing to compare.
            assert cv.errors.iloc[row, k:].isna().all()

    def test_rolling_window_forgets_the_older_observations(self):
        y = _quarterly(40)
        cv = sp.tscv(y, "mean", initial=5, window=8)
        assert cv.initial == 8 and cv.window == 8
        for row, end in enumerate(range(8, 40)):
            assert cv.forecasts.iloc[row, 0] == pytest.approx(y[end - 8 : end].mean())
        assert "rolling window 8" in cv.summary()

    def test_ets_with_arguments(self):
        y = _quarterly(40)
        cv = sp.tscv(y, "ets", initial=36, model="ANN", alpha=0.3)
        for row, end in enumerate(range(36, 40)):
            fit = sp.ets(y[:end], "ANN", alpha=0.3)
            assert cv.forecasts.iloc[row, 0] == pytest.approx(
                fit.forecast(1)["forecast"].iloc[0]
            )

    def test_arima_with_a_given_order(self):
        rng = np.random.default_rng(2)
        x = np.zeros(60)
        for i in range(1, 60):
            x[i] = 0.6 * x[i - 1] + rng.normal()
        cv = sp.tscv(x, "arima", initial=58, order=(1, 0, 0))
        for row, end in enumerate((58, 59)):
            fc = sp.arima(x[:end], order=(1, 0, 0)).forecast(1)
            assert cv.forecasts.iloc[row, 0] == pytest.approx(fc["forecast"].iloc[0])

    def test_callable_forecasters(self):
        idx = pd.period_range("2010Q1", periods=40, freq="Q")
        y = pd.Series(_quarterly(40), index=idx, name="sales")
        seen = []

        def last_two(train, h):
            seen.append(train)
            return np.full(h, train.iloc[-2:].mean())

        cv = sp.tscv(y, last_two, horizon=2, initial=30, step=5)
        assert cv.forecaster == "last_two"
        assert cv.errors.index.tolist() == [idx[29], idx[34]]
        assert isinstance(seen[0], pd.Series) and seen[0].name == "sales"
        assert seen[0].index.equals(idx[:30])
        assert cv.forecasts.iloc[1, 0] == pytest.approx(y.iloc[33:35].mean())
        # A fitted model or a forecast table is as good as an array.
        by_model = sp.tscv(y, lambda tr, h: sp.simple_forecast(tr, "naive"), initial=30)
        by_table = sp.tscv(
            y, lambda tr, h: sp.simple_forecast(tr, "naive").forecast(h), initial=30
        )
        ref = sp.tscv(y, "naive", initial=30)
        pd.testing.assert_frame_equal(by_model.errors, ref.errors)
        pd.testing.assert_frame_equal(by_table.errors, ref.errors)

    def test_accuracy_summarises_each_horizon(self):
        y = _quarterly(40)
        cv = sp.tscv(y, "naive", horizon=2, initial=20)
        acc = cv.accuracy()
        e = cv.errors["h=2"].dropna().to_numpy()
        a = cv.actuals["h=2"].dropna().to_numpy()
        assert acc.loc["h=2", "n"] == 19
        assert acc.loc["h=2", "RMSE"] == pytest.approx(np.sqrt(np.mean(e**2)))
        assert acc.loc["h=2", "MAE"] == pytest.approx(np.abs(e).mean())
        assert acc.loc["h=2", "MAPE"] == pytest.approx(100 * np.mean(np.abs(e / a)))
        assert "expanding" in repr(cv)

    def test_default_first_window(self):
        y = _quarterly(40)
        assert sp.tscv(y, "naive").initial == 10
        assert sp.tscv(y, "snaive", period=6).initial == 13

    def test_failed_origins_are_reported(self):
        y = _quarterly(40)

        def picky(train, h):
            if len(train) % 2:
                raise ValueError("odd")
            return np.zeros(h)

        with pytest.warns(UserWarning, match="failed at 5 of 10 origins"):
            cv = sp.tscv(y, picky, initial=30)
        assert len(cv.failures) == 5
        assert cv.forecasts.iloc[1].isna().all()
        assert cv.accuracy().loc["h=1", "n"] == 5
        assert "(5 failed)" in cv.summary()
        with pytest.raises(DataInsufficient, match="failed at every origin"):
            sp.tscv(y, lambda tr, h: np.zeros(h - 1), initial=30, horizon=2)

    @pytest.mark.parametrize(
        "kwargs, exc, match",
        [
            (dict(horizon=0), MethodIncompatibility, "at least 1"),
            (dict(step=0), MethodIncompatibility, "at least 1"),
            (dict(window=1), MethodIncompatibility, "window must be at least 2"),
            (dict(initial=40), DataInsufficient, "leaves\\s+nothing"),
            (dict(forecaster="prophet"), MethodIncompatibility, "not a built-in"),
            (dict(forecaster=3), MethodIncompatibility, "method name or a function"),
        ],
    )
    def test_refusals(self, kwargs, exc, match):
        with pytest.raises(exc, match=match):
            sp.tscv(_quarterly(40), **kwargs)

    def test_callable_takes_no_extra_arguments(self):
        with pytest.raises(MethodIncompatibility, match="unexpected arguments"):
            sp.tscv(_quarterly(40), lambda tr, h: np.zeros(h), alpha=0.1)

    def test_arguments_reach_the_benchmark_methods(self):
        y = _quarterly(40)
        cv = sp.tscv(y, "drift", initial=30, boxcox=0)
        for row, end in enumerate(range(30, 40)):
            fc = sp.simple_forecast(y[:end], "drift", boxcox=0).forecast(1)
            assert cv.forecasts.iloc[row, 0] == pytest.approx(fc["forecast"].iloc[0])
        plain = sp.tscv(y, "drift", initial=30)
        assert not np.allclose(cv.forecasts.to_numpy(), plain.forecasts.to_numpy())
        with pytest.raises(MethodIncompatibility, match="unexpected arguments"):
            sp.tscv(y, "naive", initial=30, typo=3)

    def test_arguments_reach_the_benchmark_methods_of_an_stl_forecast(self):
        y = np.exp(_monthly() / 40.0)
        dec = sp.stl(y, 12)
        for method in ("naive", "drift"):
            got = dec.forecast(3, method=method, boxcox=0)
            ref = sp.simple_forecast(dec.seasonally_adjusted, method, boxcox=0)
            seas = dec.seasonal[-12:][:3]
            np.testing.assert_allclose(
                got["forecast"], ref.forecast(3)["forecast"].to_numpy() + seas
            )
            with pytest.raises(MethodIncompatibility, match="unexpected arguments"):
                dec.forecast(3, method=method, typo=3)


# ----------------------------------------------------------------------
# small tools
# ----------------------------------------------------------------------
class TestLjungBox:
    def test_equals_statsmodels(self):
        from statsmodels.stats.diagnostic import acorr_ljungbox

        x = np.random.default_rng(0).normal(size=100)
        ref = acorr_ljungbox(x, [5, 10], model_df=2, boxpierce=True)
        lb = sp.ljungbox(x, [5, 10], model_df=2)
        bp = sp.ljungbox(x, [5, 10], model_df=2, method="Box_Pierce")
        np.testing.assert_allclose(lb["statistic"], ref["lb_stat"])
        np.testing.assert_allclose(lb["p_value"], ref["lb_pvalue"])
        np.testing.assert_allclose(bp["statistic"], ref["bp_stat"])
        np.testing.assert_allclose(bp["p_value"], ref["bp_pvalue"])
        assert lb["df"].tolist() == [3, 8]
        assert lb.attrs == {"method": "Ljung-Box", "model_df": 2}
        assert bp.attrs["method"] == "Box-Pierce"

    def test_default_lags(self):
        x = np.random.default_rng(0).normal(size=100)
        assert sp.ljungbox(x).index.tolist() == [10]
        assert sp.ljungbox(x, period=12).index.tolist() == [20]  # min(24, n / 5)
        assert sp.ljungbox(x[:30]).index.tolist() == [6]
        assert sp.ljungbox(x[:30], model_df=8).index.tolist() == [9]

    def test_fitted_model_supplies_residuals_and_period(self):
        y = _quarterly()
        fit = sp.simple_forecast(y, "snaive", period=4)
        res = fit.residuals[4:]
        a = sp.ljungbox(fit)
        b = sp.ljungbox(res, lags=8)
        pd.testing.assert_frame_equal(a, b)

    def test_no_degrees_of_freedom_left(self):
        x = np.random.default_rng(0).normal(size=100)
        out = sp.ljungbox(x, 2, model_df=3)
        assert out.loc[2, "df"] == -1 and np.isnan(out.loc[2, "p_value"])

    def test_refusals(self):
        x = np.random.default_rng(0).normal(size=30)
        with pytest.raises(MethodIncompatibility, match="method='q'"):
            sp.ljungbox(x, method="q")
        with pytest.raises(MethodIncompatibility, match="lags must be positive"):
            sp.ljungbox(x, 0)
        with pytest.raises(DataInsufficient, match="30 lags need more"):
            sp.ljungbox(x, 30)
        with pytest.raises(DataInsufficient, match="constant"):
            sp.ljungbox(np.ones(30))
        with pytest.raises(MethodIncompatibility, match="cannot be negative"):
            sp.ljungbox(x, model_df=-1)
        with pytest.raises(MethodIncompatibility, match="neither a series"):
            sp.ljungbox(object())


class TestDifferencingOrders:
    @pytest.mark.parametrize("alpha", [0.10, 0.05, 0.025, 0.01])
    def test_ndiffs_is_repeated_kpss_testing(self, alpha):
        from statsmodels.tsa.stattools import kpss

        rng = np.random.default_rng(3)
        crit = {0.10: 0.347, 0.05: 0.463, 0.025: 0.574, 0.01: 0.739}[alpha]
        series = {
            "noise": rng.normal(size=150),
            "walk": np.cumsum(rng.normal(size=150)),
            "double": np.cumsum(np.cumsum(rng.normal(size=150))),
        }
        for x in series.values():
            d, z = 0, x
            while d < 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Level stationarity, Bartlett kernel, the short
                    # truncation lag trunc(3 sqrt(n) / 13).
                    stat = kpss(z, regression="c", nlags=int(3 * np.sqrt(len(z)) / 13))[
                        0
                    ]
                if not stat > crit:
                    break
                z, d = np.diff(z), d + 1
            assert sp.ndiffs(x, alpha=alpha) == d
        assert sp.ndiffs(series["walk"]) == 1
        assert sp.ndiffs(series["double"]) == 2
        assert sp.ndiffs(series["double"], max_d=1) == 1
        assert sp.ndiffs(series["double"], max_d=0) == 0

    def test_ndiffs_degenerate_inputs(self):
        assert sp.ndiffs(np.ones(30)) == 0
        assert sp.ndiffs(np.arange(5.0)) == 0  # too short to test
        with pytest.raises(MethodIncompatibility, match="no tabulated"):
            sp.ndiffs(np.arange(30.0), alpha=0.07)
        with pytest.raises(MethodIncompatibility, match="cannot be negative"):
            sp.ndiffs(np.arange(30.0), max_d=-1)
        with pytest.raises(MethodIncompatibility, match="missing value"):
            sp.ndiffs(np.r_[1.0, 2.0, np.nan, 4.0, 5.0])

    def test_nsdiffs_compares_seasonal_strength_with_the_threshold(self):
        rng = np.random.default_rng(4)
        strong = np.tile([1.0, 5.0, 2.0, 8.0], 20) + 0.1 * rng.normal(size=80)
        noise = rng.normal(size=80)
        s_strong = sp.stl(strong, 4).strength["seasonal"]
        s_noise = sp.stl(noise, 4).strength["seasonal"]
        assert s_strong > 0.64 > s_noise
        assert sp.nsdiffs(strong, 4) == 1
        assert sp.nsdiffs(noise, 4) == 0
        assert sp.nsdiffs(strong, 4, threshold=0.9999) == int(s_strong > 0.9999)
        assert sp.nsdiffs(noise, 4, threshold=s_noise - 1e-6) == 1
        assert sp.nsdiffs(strong, 4, max_D=0) == 0
        # Second seasonal difference only if the differenced series still
        # has a strong season.
        diffed = strong[4:] - strong[:-4]
        again = sp.stl(diffed, 4).strength["seasonal"] > 0.64
        assert sp.nsdiffs(strong, 4, max_D=2) == 1 + int(again)

    def test_nsdiffs_degenerate_inputs(self):
        assert sp.nsdiffs(np.arange(6.0), 4) == 0  # under two cycles and one
        assert sp.nsdiffs(np.ones(40), 4) == 0
        with pytest.raises(MethodIncompatibility, match="period=1"):
            sp.nsdiffs(np.arange(40.0), 1)


class TestBoxCoxLambda:
    def test_minimises_the_coefficient_of_variation_over_cycles(self):
        rng = np.random.default_rng(5)
        y = np.exp(2 + 0.02 * np.arange(62) + 0.1 * rng.normal(size=62))
        lam = sp.boxcox_lambda(y, 12)
        # Two leading observations do not fill a cycle and are left out.
        mat = y[2:].reshape(5, 12)
        mean, sd = mat.mean(axis=1), mat.std(axis=1, ddof=1)

        def cv(v):
            rat = sd / mean ** (1 - v)
            return np.std(rat, ddof=1) / np.mean(rat)

        grid = np.linspace(-1, 2, 30001)
        best = grid[np.argmin([cv(v) for v in grid])]
        # Grid step 1e-4.
        assert lam == pytest.approx(best, abs=2e-4)
        assert cv(lam) <= min(cv(lam - 1e-3), cv(lam + 1e-3))

    def test_bounds_and_non_seasonal_grouping(self):
        rng = np.random.default_rng(5)
        y = np.exp(2 + 0.02 * np.arange(60) + 0.1 * rng.normal(size=60))
        free = sp.boxcox_lambda(y, 12)
        lo = free + 0.5
        assert sp.boxcox_lambda(y, 12, lower=lo, upper=lo + 1) == pytest.approx(
            lo, abs=1e-6
        )
        # A non-seasonal series is grouped in pairs.
        assert sp.boxcox_lambda(y) == sp.boxcox_lambda(y, 2)

    def test_refusals(self):
        with pytest.raises(MethodIncompatibility, match="strictly positive"):
            sp.boxcox_lambda(np.arange(20.0))
        with pytest.raises(DataInsufficient, match="fewer than two cycles"):
            sp.boxcox_lambda(np.arange(1.0, 20.0), 12)


class TestFourierTermsAndDummies:
    def test_sines_and_cosines_at_the_stated_frequencies(self):
        out = sp.fourier_terms(30, 12, 3)
        t = np.arange(1, 31)
        assert list(out.columns) == [
            "sin1_12",
            "cos1_12",
            "sin2_12",
            "cos2_12",
            "sin3_12",
            "cos3_12",
        ]
        for k in (1, 2, 3):
            np.testing.assert_allclose(
                out[f"sin{k}_12"], np.sin(2 * np.pi * k * t / 12)
            )
            np.testing.assert_allclose(
                out[f"cos{k}_12"], np.cos(2 * np.pi * k * t / 12)
            )

    def test_last_sine_is_dropped_at_half_the_period(self):
        out = sp.fourier_terms(8, 4, 2)
        assert list(out.columns) == ["sin1_4", "cos1_4", "cos2_4"]
        np.testing.assert_allclose(out["cos2_4"], np.tile([-1.0, 1.0], 4))
        # The full set spans the same space as seasonal dummies.
        X = np.column_stack([np.ones(8), out.to_numpy()])
        D = sp.seasonal_dummies(8, 4, drop_first=False).to_numpy()
        assert np.linalg.matrix_rank(np.column_stack([X, D])) == 4

    def test_start_continues_the_sample(self):
        whole = sp.fourier_terms(40, 52.18, 2)
        future = sp.fourier_terms(10, 52.18, 2, start=31)
        np.testing.assert_allclose(future.to_numpy(), whole.iloc[30:].to_numpy())
        assert list(whole.columns)[0] == "sin1_52.18"

    def test_index_of_a_series_is_kept(self):
        idx = pd.period_range("2020-01", periods=6, freq="M")
        s = pd.Series(np.zeros(6), index=idx)
        assert sp.fourier_terms(s, 12, 1).index.equals(idx)
        assert sp.fourier_terms(idx, 12, 1).index.equals(idx)
        assert sp.seasonal_dummies(s, 12).index.equals(idx)
        assert sp.seasonal_dummies(idx, 12).index.equals(idx)
        assert sp.fourier_terms(s.to_frame(), 12, 1).shape == (6, 2)

    def test_fourier_refusals(self):
        with pytest.raises(MethodIncompatibility, match="K=3"):
            sp.fourier_terms(10, 4, 3)
        with pytest.raises(MethodIncompatibility, match="K=0"):
            sp.fourier_terms(10, 4, 0)
        with pytest.raises(MethodIncompatibility, match="must be above 1"):
            sp.fourier_terms(10, 1, 1)
        with pytest.raises(MethodIncompatibility, match="at least one row"):
            sp.fourier_terms(0, 4, 1)

    def test_dummies_partition_the_periods(self):
        full = sp.seasonal_dummies(10, 4, start=3, drop_first=False)
        assert (full.sum(axis=1) == 1).all()
        season = (np.arange(3, 13) - 1) % 4 + 1
        for k in range(1, 5):
            np.testing.assert_array_equal(
                full[f"season_{k}"], (season == k).astype(int)
            )
        dropped = sp.seasonal_dummies(10, 4, start=3)
        pd.testing.assert_frame_equal(dropped, full.drop(columns="season_1"))

    def test_dummies_refusals(self):
        with pytest.raises(MethodIncompatibility, match="at least 2"):
            sp.seasonal_dummies(10, 1)
        with pytest.raises(MethodIncompatibility, match="n at least 1"):
            sp.seasonal_dummies(0, 4)


class TestSeriesFeatures:
    def test_autocorrelation_features_equal_statsmodels(self):
        from statsmodels.tsa.stattools import acf, pacf

        y = _monthly()
        z = (y - y.mean()) / y.std(ddof=1)
        f = sp.ts_features(y, 12)
        r = acf(z, nlags=12)
        assert f["x_acf1"] == pytest.approx(r[1])
        assert f["x_acf10"] == pytest.approx(np.sum(r[1:11] ** 2))
        assert f["seas_acf1"] == pytest.approx(r[12])
        d1 = acf(np.diff(z), nlags=10)
        assert f["diff1_acf1"] == pytest.approx(d1[1])
        assert f["diff1_acf10"] == pytest.approx(np.sum(d1[1:] ** 2))
        d2 = acf(np.diff(z, 2), nlags=10)
        assert f["diff2_acf10"] == pytest.approx(np.sum(d2[1:] ** 2))
        # Durbin-Levinson recursion on the sample autocorrelations.
        p = pacf(z, nlags=12, method="ldb")
        assert f["x_pacf5"] == pytest.approx(np.sum(p[1:6] ** 2))
        assert f["seas_pacf"] == pytest.approx(p[12])
        assert f["diff1x_pacf5"] == pytest.approx(
            np.sum(pacf(np.diff(z), nlags=5, method="ldb")[1:] ** 2)
        )

    def test_block_and_shape_features(self):
        import statsmodels.api as sm

        y = _monthly()
        z = (y - y.mean()) / y.std(ddof=1)
        f = sp.ts_features(y, 12)
        blocks = z.reshape(10, 12)
        assert f["lumpiness"] == pytest.approx(
            np.var(blocks.var(axis=1, ddof=1), ddof=1)
        )
        assert f["stability"] == pytest.approx(np.var(blocks.mean(axis=1), ddof=1))
        above = z > np.median(z)
        assert f["crossing_points"] == np.sum(above[1:] != above[:-1])
        means = pd.Series(z).rolling(12).mean().dropna().to_numpy()
        shift = np.abs(means[12:] - means[:-12])
        assert f["max_level_shift"] == pytest.approx(shift.max())
        assert f["time_level_shift"] == np.argmax(shift) + 12
        var = pd.Series(z).rolling(12).var().dropna().to_numpy()
        assert f["max_var_shift"] == pytest.approx(np.abs(var[12:] - var[:-12]).max())
        sq = (z - z.mean()) ** 2
        lags = np.column_stack([sq[12 - j : len(sq) - j] for j in range(1, 13)])
        r2 = sm.OLS(sq[12:], sm.add_constant(lags)).fit().rsquared
        assert f["arch_lm"] == pytest.approx(r2)

    def test_decomposition_features_come_from_stl(self):
        y = _monthly()
        z = (y - y.mean()) / y.std(ddof=1)
        f = sp.ts_features(y, 12)
        dec = sp.stl(z, 12)
        assert f["trend"] == pytest.approx(dec.strength["trend"])
        assert f["seasonal_strength"] == pytest.approx(dec.strength["seasonal"])
        figure = dec.seasonal.reshape(10, 12).mean(axis=0)
        assert f["peak"] == np.argmax(figure) + 1
        assert f["trough"] == np.argmin(figure) + 1
        c = dec.remainder - dec.remainder.mean()
        assert f["e_acf1"] == pytest.approx((c[1:] @ c[:-1]) / (c @ c))
        # Orthogonal polynomial regression of the trend on time.
        t = np.arange(1.0, 121.0)
        q, _ = np.linalg.qr(np.column_stack([np.ones(120), t, t**2]))
        assert abs(f["linearity"]) == pytest.approx(abs(q[:, 1] @ dec.trend))
        assert f["linearity"] > 0  # the series trends upwards

    def test_scaling_makes_the_features_unit_free(self):
        y = _monthly()
        a = sp.ts_features(y, 12)
        b = sp.ts_features(1000.0 * y - 7.0, 12)
        pd.testing.assert_series_equal(a, b, rtol=1e-8, check_names=False)
        raw = sp.ts_features(y, 12, scale=False)
        assert raw["x_acf1"] == pytest.approx(a["x_acf1"])
        assert raw["stability"] == pytest.approx(a["stability"] * y.var(ddof=1))

    def test_table_of_series_is_one_row_each(self):
        wide = pd.DataFrame({"a": _monthly(seed=1), "b": _monthly(seed=2)})
        table = sp.ts_features(wide, 12)
        assert table.index.tolist() == ["a", "b"]
        pd.testing.assert_series_equal(
            table.loc["b"], sp.ts_features(wide["b"], 12), check_names=False
        )
        by_name = sp.ts_features("b", 12, data=wide)
        assert by_name.name == "b"
        pd.testing.assert_series_equal(by_name, sp.ts_features(wide["b"], 12))

    def test_non_seasonal_series_has_no_decomposition_features(self):
        f = sp.ts_features(_monthly())
        assert "trend" not in f.index and "seas_acf1" not in f.index
        z = (_monthly() - _monthly().mean()) / _monthly().std(ddof=1)
        blocks = z.reshape(12, 10)  # blocks of ten without a period
        assert f["stability"] == pytest.approx(np.var(blocks.mean(axis=1), ddof=1))

    def test_refusals(self):
        with pytest.raises(DataInsufficient, match="constant"):
            sp.ts_features(np.ones(40))
        with pytest.raises(DataInsufficient, match="5 observations are too few"):
            sp.ts_features(np.arange(5.0))
        with pytest.raises(MethodIncompatibility, match="scale must be"):
            sp.ts_features(_monthly(), scale="no")

    def test_short_columns_of_a_table_are_refused_too(self):
        rng = np.random.default_rng(0)
        wide = pd.DataFrame({"a": rng.normal(size=5), "b": rng.normal(size=5)})
        with pytest.raises(DataInsufficient, match="too few"):
            sp.ts_features(wide)
        with pytest.raises(MethodIncompatibility, match="scale must be"):
            sp.ts_features(pd.DataFrame({"a": _monthly(), "b": _monthly()}), scale="no")

    def test_constant_series_is_refused_with_and_without_scaling(self):
        flat = np.full(40, 3.0)
        for scale in (True, False):
            with pytest.raises(DataInsufficient, match="constant"):
                sp.ts_features(flat, scale=scale)
            with pytest.raises(DataInsufficient, match="column 'b' is constant"):
                sp.ts_features(
                    pd.DataFrame({"a": _monthly()[:40], "b": flat}), scale=scale
                )
