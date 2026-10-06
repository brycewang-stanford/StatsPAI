"""Behaviour of the forecasting toolkit: known truths, hand-computed
values, and the errors each function raises on input it cannot handle.

Cross-language parity lives in
``tests/reference_parity/test_forecasting_r_parity.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def _seasonal_series(n=96, m=4, seed=0, mult=True):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seas = np.tile([0.9, 1.1, 1.2, 0.8], n // m + 1)[:n]
    if mult:
        return (100 + 0.8 * t) * seas * (1 + rng.normal(0, 0.02, n))
    return 100 + 0.8 * t + 20 * (seas - 1) + rng.normal(0, 1.0, n)


# ----------------------------------------------------------------------
# sp.ets
# ----------------------------------------------------------------------
def test_ets_recovers_smoothing_parameter_of_a_local_level():
    rng = np.random.default_rng(3)
    n, alpha = 600, 0.35
    level, y = 20.0, np.empty(n)
    for t in range(n):
        e = rng.normal(0, 1.0)
        y[t] = level + e
        level += alpha * e
    fit = sp.ets(y, "ANN")
    assert fit.model == "ETS(A,N,N)"
    assert fit.params["alpha"] == pytest.approx(alpha, abs=0.06)
    assert fit.sigma2 == pytest.approx(1.0, abs=0.12)
    # SES: the forecast is flat, the variance grows by alpha^2 sigma^2 a step
    fc = fit.forecast(3, level=95)
    assert np.ptp(fc["forecast"]) == pytest.approx(0.0, abs=1e-12)
    sd = (fc["upper_95"] - fc["forecast"]) / 1.959963984540054
    a = fit.params["alpha"]
    np.testing.assert_allclose(
        sd**2, fit.sigma2 * (1 + a**2 * np.arange(3)), rtol=1e-10
    )


def test_ets_fitted_values_are_the_one_step_recursion():
    y = _seasonal_series(mult=False)
    fit = sp.ets(y, "AAA", period=4)
    st = fit.states
    for t in (0, 5, 40):
        pred = st["level"].iloc[t] + st["slope"].iloc[t] + st["season_lag3"].iloc[t]
        assert fit.fitted_values[t] == pytest.approx(pred, rel=1e-12)
    np.testing.assert_allclose(fit.residuals, y - fit.fitted_values, rtol=1e-12)
    assert fit.initial_state[["s0", "s-1", "s-2", "s-3"]].sum() == pytest.approx(
        0.0, abs=1e-9
    )


def test_ets_automatic_choice_finds_multiplicative_seasonality():
    fit = sp.ets(_seasonal_series(), period=4)
    assert fit.season == "M" and fit.error == "M"
    cand = fit.candidates
    assert cand["aicc"].is_monotonic_increasing
    assert cand["model"].iloc[0] == fit.model
    # restricted combinations are not tried
    assert not cand["model"].str.startswith("ETS(A,A,M").any()


def test_ets_damped_trend_and_fixed_parameter():
    y = _seasonal_series(mult=False)[:60]
    fit = sp.ets(y, "AAdN", phi=0.9)
    assert fit.damped and fit.params["phi"] == 0.9 and fit.fixed == {"phi": 0.9}
    free = sp.ets(y, "AAN", damped=True)
    # a fixed parameter is not counted
    assert fit.n_params == free.n_params - 1
    fc = fit.forecast(40, level=80)["forecast"].to_numpy()
    # damping: the increments shrink geometrically
    inc = np.diff(fc)
    np.testing.assert_allclose(inc[1:] / inc[:-1], 0.9, rtol=1e-9)


def test_ets_multiplicative_error_intervals_widen_with_the_level():
    y = _seasonal_series()
    fit = sp.ets(y, "MAM", period=4)
    fc = fit.forecast(8, level=(80, 95))
    width = fc["upper_95"] - fc["lower_95"]
    assert (fc["lower_95"] < fc["lower_80"]).all()
    assert (fc["upper_80"] < fc["upper_95"]).all()
    # the widest interval sits on the seasonal peak
    assert width.iloc[:4].idxmax() == fc["forecast"].iloc[:4].idxmax()


def test_ets_simulated_intervals_agree_with_analytic_ones():
    y = _seasonal_series(mult=False)
    fit = sp.ets(y, "AAA", period=4)
    exact = fit.forecast(6, level=90)
    sim = fit.forecast(6, level=90, simulate=True, n_paths=40_000, seed=1)
    np.testing.assert_allclose(sim["lower_90"], exact["lower_90"], rtol=0.01)
    np.testing.assert_allclose(sim["upper_90"], exact["upper_90"], rtol=0.01)
    boot = fit.forecast(6, level=90, bootstrap=True, n_paths=20_000, seed=1)
    assert (boot["lower_90"] < exact["forecast"]).all()
    # models without an analytic variance are simulated, reproducibly
    mm = sp.ets(_seasonal_series(), "MMN")
    a = mm.forecast(4, seed=7)
    b = mm.forecast(4, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert (a["lower_95"] < a["forecast"]).all()


def test_ets_keeps_a_date_index_and_reads_a_column():
    idx = pd.period_range("2015Q1", periods=96, freq="Q")
    df = pd.DataFrame({"sales": _seasonal_series()}, index=idx)
    fit = sp.ets("sales", "MAM", period=4, data=df)
    fc = fit.forecast(3)
    assert list(fc.index.astype(str)) == ["2039Q1", "2039Q2", "2039Q3"]
    comp = fit.components()
    assert list(comp.columns) == ["sales", "level", "slope", "season", "remainder"]
    assert comp.index.equals(idx)
    assert "ETS(M,A,M)" in fit.summary()
    assert fit.simulate(5, n_paths=3).shape == (5, 3)


@pytest.mark.parametrize(
    "kwargs, exc, text",
    [
        ({"model": "AAA"}, MethodIncompatibility, "period=1"),
        ({"model": "XYZ"}, MethodIncompatibility, "not an ETS specification"),
        ({"model": "ANN", "gamma": 0.1}, MethodIncompatibility, "has no gamma"),
        ({"model": "ANN", "ic": "hqic"}, MethodIncompatibility, "ic="),
        ({"model": "ANA", "period": 52}, MethodIncompatibility, "seasonal starting"),
    ],
)
def test_ets_refuses_inconsistent_requests(kwargs, exc, text):
    y = _seasonal_series(n=120)
    with pytest.raises(exc, match=text):
        sp.ets(y, **kwargs)


def test_ets_refuses_bad_data():
    y = _seasonal_series()
    with pytest.raises(MethodIncompatibility, match="strictly positive"):
        sp.ets(y - y.mean(), "MNN")
    with pytest.raises(DataInsufficient, match="constant"):
        sp.ets(np.ones(30), "ANN")
    with pytest.raises(DataInsufficient, match="too few"):
        sp.ets([1.0, 2.0, 3.0], "ANN")
    # a series with non-positive values: the search stays additive
    fit = sp.ets(y - y.mean(), period=4)
    assert "M" not in (fit.error, fit.trend, fit.season)


# ----------------------------------------------------------------------
# sp.simple_forecast
# ----------------------------------------------------------------------
def test_simple_forecast_formulas_by_hand():
    y = np.array([3.0, 5.0, 4.0, 6.0, 8.0, 7.0, 9.0, 10.0])
    T = len(y)
    z = 1.959963984540054
    mean = sp.simple_forecast(y, "mean")
    assert mean.sigma == pytest.approx(y.std(ddof=1))
    fc = mean.forecast(2, level=95)
    assert fc["forecast"].tolist() == [y.mean()] * 2
    assert fc["upper_95"].iloc[0] == pytest.approx(
        y.mean() + z * y.std(ddof=1) * np.sqrt(1 + 1 / T)
    )
    naive = sp.simple_forecast(y, "naive")
    s = np.sqrt(np.sum(np.diff(y) ** 2) / (T - 1))
    assert naive.sigma == pytest.approx(s)
    fc = naive.forecast(4, level=95)
    np.testing.assert_allclose(fc["upper_95"] - 10.0, z * s * np.sqrt([1, 2, 3, 4]))
    drift = sp.simple_forecast(y, "drift")
    b = (y[-1] - y[0]) / (T - 1)
    assert drift.drift == pytest.approx(b)
    sd = np.sqrt(np.sum((np.diff(y) - b) ** 2) / (T - 2))
    fc = drift.forecast(3, level=95)
    np.testing.assert_allclose(fc["forecast"], 10.0 + b * np.arange(1, 4))
    h = np.arange(1, 4)
    np.testing.assert_allclose(
        fc["upper_95"] - fc["forecast"], z * sd * np.sqrt(h * (1 + h / (T - 1)))
    )


def test_seasonal_naive_repeats_the_last_cycle():
    y = np.arange(1.0, 13.0) + np.tile([0.0, 5.0, -3.0, 1.0], 3)
    fit = sp.simple_forecast(y, "snaive", period=4)
    fc = fit.forecast(6, level=80)
    np.testing.assert_allclose(fc["forecast"], np.r_[y[-4:], y[-4:-2]])
    half = fc["upper_80"] - fc["forecast"]
    assert half.iloc[4] / half.iloc[0] == pytest.approx(np.sqrt(2.0))
    assert np.isnan(fit.residuals[:4]).all() and np.isfinite(fit.residuals[4:]).all()


def test_simple_forecast_bootstrap_and_errors():
    rng = np.random.default_rng(0)
    y = 50 + np.cumsum(rng.normal(0, 1, 300))
    fit = sp.simple_forecast(y, "naive")
    a = fit.forecast(5, level=95)
    b = fit.forecast(5, level=95, bootstrap=True, n_paths=20_000)
    np.testing.assert_allclose(b["lower_95"], a["lower_95"], rtol=0.01)
    assert fit.simulate(4, n_paths=6).shape == (4, 6)
    with pytest.raises(MethodIncompatibility, match="not a benchmark"):
        sp.simple_forecast(y, "holt")
    with pytest.raises(MethodIncompatibility, match="period >= 2"):
        sp.simple_forecast(y, "snaive")
    with pytest.raises(DataInsufficient):
        sp.simple_forecast(y[:3], "snaive", period=4)
    with pytest.raises(MethodIncompatibility, match="horizon"):
        fit.forecast(0)
    with pytest.raises(MethodIncompatibility, match="coverage"):
        fit.forecast(2, level=120)


# ----------------------------------------------------------------------
# sp.forecast_accuracy / sp.tscv
# ----------------------------------------------------------------------
def test_forecast_accuracy_by_hand():
    actual = np.array([10.0, 12.0, 8.0, 11.0])
    fc = np.array([9.0, 13.0, 10.0, 11.0])
    train = np.array([5.0, 7.0, 6.0, 9.0, 8.0])
    acc = sp.forecast_accuracy(actual, fc, train=train).iloc[0]
    e = actual - fc
    assert acc["ME"] == pytest.approx(e.mean())
    assert acc["RMSE"] == pytest.approx(np.sqrt(np.mean(e**2)))
    assert acc["MAE"] == pytest.approx(np.mean(np.abs(e)))
    assert acc["MAPE"] == pytest.approx(np.mean(np.abs(100 * e / actual)))
    scale = np.mean(np.abs(np.diff(train)))
    assert acc["MASE"] == pytest.approx(np.mean(np.abs(e)) / scale)
    assert acc["RMSSE"] == pytest.approx(
        np.sqrt(np.mean(e**2) / np.mean(np.diff(train) ** 2))
    )
    no_train = sp.forecast_accuracy(actual, fc).iloc[0]
    assert np.isnan(no_train["MASE"])


def test_forecast_accuracy_interval_scores():
    actual = np.array([10.0, 20.0, 30.0])
    frame = pd.DataFrame(
        {
            "forecast": [11.0, 20.0, 25.0],
            "lower_80": [9.0, 18.0, 22.0],
            "upper_80": [13.0, 22.0, 28.0],
        }
    )
    acc = sp.forecast_accuracy(actual, frame, crps=True).iloc[0]
    # width, plus (2 / 0.2) * miss for the third outcome (30 > 28)
    assert acc["winkler_80"] == pytest.approx((4 + 4 + (6 + 10 * 2)) / 3)
    assert acc["coverage_80"] == pytest.approx(2 / 3)
    # CRPS of N(mu, sd) at the outcome; sd = half-width / z_0.90
    from scipy import stats

    z80 = stats.norm.ppf(0.9)
    sd = np.array([2.0, 2.0, 3.0]) / z80
    zz = (actual - frame["forecast"].to_numpy()) / sd
    crps = sd * (
        zz * (2 * stats.norm.cdf(zz) - 1) + 2 * stats.norm.pdf(zz) - 1 / np.sqrt(np.pi)
    )
    assert acc["CRPS"] == pytest.approx(crps.mean())


def test_forecast_accuracy_compares_methods_and_validates():
    y = _seasonal_series()
    train, test = y[:80], y[80:]
    fcs = {
        "naive": sp.simple_forecast(train, "naive").forecast(16),
        "snaive": sp.simple_forecast(train, "snaive", period=4).forecast(16),
        "ets": sp.ets(train, "MAM", period=4).forecast(16),
    }
    acc = sp.forecast_accuracy(test, fcs, train=train, period=4)
    assert list(acc.index) == ["naive", "snaive", "ets"]
    assert acc.loc["ets", "RMSE"] < acc.loc["snaive", "RMSE"] < acc.loc["naive", "RMSE"]
    assert acc.loc["ets", "MASE"] < 1
    with pytest.raises(MethodIncompatibility, match="forecasts for"):
        sp.forecast_accuracy(test, np.ones(3))
    with pytest.raises(MethodIncompatibility, match="needs prediction intervals"):
        sp.forecast_accuracy(test, pd.DataFrame({"forecast": test}), crps=True)
    with pytest.raises(MethodIncompatibility, match="'forecast' column"):
        sp.forecast_accuracy(test, pd.DataFrame({"yhat": test}))


def test_tscv_naive_errors_are_the_differences():
    y = np.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
    cv = sp.tscv(y, "naive", horizon=2, initial=2)
    assert cv.errors.shape == (4, 2)
    np.testing.assert_allclose(cv.errors["h=1"], [5.0, 7.0, 9.0, 11.0])
    np.testing.assert_allclose(cv.errors["h=2"].iloc[:3], [12.0, 16.0, 20.0])
    assert np.isnan(cv.errors["h=2"].iloc[3])  # beyond the sample
    acc = cv.accuracy()
    assert acc.loc["h=1", "n"] == 4 and acc.loc["h=2", "n"] == 3
    assert acc.loc["h=1", "RMSE"] == pytest.approx(np.sqrt(np.mean([25, 49, 81, 121])))


def test_tscv_window_step_and_callables():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    y = pd.Series(10 + np.cumsum(rng.normal(0, 1, 60)), index=idx)
    seen = []

    def last_three(train, h):
        seen.append(len(train))
        assert isinstance(train, pd.Series) and train.index[-1] in idx
        return np.full(h, train.iloc[-3:].mean())

    cv = sp.tscv(y, last_three, horizon=3, initial=20, step=5, window=10)
    assert set(seen) == {10}
    assert len(cv.errors) == len(range(20, 60, 5))
    assert cv.errors.index[0] == idx[19]
    # a forecaster may return a fitted model or a forecast table
    a = sp.tscv(y, lambda tr, h: sp.simple_forecast(tr, "drift"), horizon=2, initial=30)
    b = sp.tscv(y, "drift", horizon=2, initial=30)
    np.testing.assert_allclose(a.errors.to_numpy(), b.errors.to_numpy(), equal_nan=True)
    c = sp.tscv(
        y,
        lambda tr, h: sp.simple_forecast(tr, "drift").forecast(h),
        horizon=2,
        initial=30,
    )
    np.testing.assert_allclose(c.errors.to_numpy(), b.errors.to_numpy(), equal_nan=True)
    assert "Time series cross-validation" in cv.summary()


def test_tscv_reports_failures_instead_of_dropping_them():
    y = np.arange(30.0) + np.sin(np.arange(30.0))

    def flaky(train, h):
        if len(train) % 7 == 0:
            raise RuntimeError("boom")
        return np.full(h, train.iloc[-1])

    with pytest.warns(UserWarning, match="failed at"):
        cv = sp.tscv(y, flaky, horizon=1, initial=10)
    assert [lab for lab, _ in cv.failures] == [13, 20, 27]
    assert cv.errors["h=1"].isna().sum() == 3
    with pytest.raises(DataInsufficient, match="every origin"):
        sp.tscv(y, lambda tr, h: 1 / 0, horizon=1, initial=10)
    with pytest.raises(MethodIncompatibility, match="not a built-in"):
        sp.tscv(y, "prophet")
    with pytest.raises(DataInsufficient, match="leaves nothing"):
        sp.tscv(y, "naive", initial=30)


def test_tscv_ets_beats_naive_on_a_trending_seasonal_series():
    y = _seasonal_series(n=72)
    ets_cv = sp.tscv(y, "ets", horizon=4, initial=48, step=4, period=4, model="MAM")
    naive_cv = sp.tscv(y, "naive", horizon=4, initial=48, step=4)
    assert ets_cv.accuracy()["RMSE"].mean() < 0.5 * naive_cv.accuracy()["RMSE"].mean()


# ----------------------------------------------------------------------
# sp.ljungbox, sp.ndiffs, sp.nsdiffs, sp.boxcox_lambda, sp.fourier_terms
# ----------------------------------------------------------------------
def test_ljungbox_against_statsmodels_and_degrees_of_freedom():
    from statsmodels.stats.diagnostic import acorr_ljungbox

    rng = np.random.default_rng(5)
    x = rng.normal(size=150)
    ref = acorr_ljungbox(x, lags=[5, 10], boxpierce=True, model_df=2)
    out = sp.ljungbox(x, lags=[5, 10], model_df=2)
    np.testing.assert_allclose(out["statistic"], ref["lb_stat"], rtol=1e-12)
    np.testing.assert_allclose(out["p_value"], ref["lb_pvalue"], rtol=1e-10)
    assert out["df"].tolist() == [3, 8]
    bp = sp.ljungbox(x, lags=[5, 10], method="box-pierce")
    np.testing.assert_allclose(bp["statistic"], ref["bp_stat"], rtol=1e-12)
    assert out.attrs["method"] == "Ljung-Box" and out.attrs["model_df"] == 2


def test_ljungbox_reads_fitted_models():
    rng = np.random.default_rng(2)
    n = 200
    e = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + e[t]
    fit = sp.arima(y, order=(1, 0, 0))
    out = sp.ljungbox(fit, lags=10)
    assert out["df"].iloc[0] == 9  # one AR coefficient
    assert out["p_value"].iloc[0] > 0.05
    # the raw series is strongly autocorrelated
    assert sp.ljungbox(y, lags=10)["p_value"].iloc[0] < 1e-10
    ets_fit = sp.ets(_seasonal_series(), "MAM", period=4)
    assert sp.ljungbox(ets_fit).index.tolist() == [8]  # 2 x period
    with pytest.raises(MethodIncompatibility, match="ljung-box"):
        sp.ljungbox(y, method="durbin")
    with pytest.raises(DataInsufficient, match="more than"):
        sp.ljungbox(y[:8], lags=10)
    with pytest.raises(DataInsufficient, match="constant"):
        sp.ljungbox(np.ones(50), lags=5)
    with pytest.raises(MethodIncompatibility, match="neither a series"):
        sp.ljungbox(object())


def test_ndiffs_and_nsdiffs_on_known_processes():
    rng = np.random.default_rng(1)
    e = rng.normal(size=400)
    assert sp.ndiffs(e) == 0
    assert sp.ndiffs(np.cumsum(e)) == 1
    assert sp.ndiffs(np.cumsum(np.cumsum(e))) == 2
    assert sp.ndiffs(np.cumsum(np.cumsum(e)), max_d=1) == 1
    # a stricter test differences less readily
    assert sp.ndiffs(np.cumsum(e), alpha=0.01) <= sp.ndiffs(np.cumsum(e), alpha=0.10)
    with pytest.raises(MethodIncompatibility, match="tabulated"):
        sp.ndiffs(e, alpha=0.07)
    t = np.arange(240)
    strong = 8 * np.sin(2 * np.pi * t / 12) + rng.normal(size=240)
    assert sp.nsdiffs(strong, 12) == 1
    assert sp.nsdiffs(rng.normal(size=240), 12) == 0
    assert sp.nsdiffs(strong, 12, threshold=0.9999) == 0
    with pytest.raises(MethodIncompatibility, match="period=1"):
        sp.nsdiffs(strong, 1)


def test_boxcox_lambda_recovers_known_transformations():
    rng = np.random.default_rng(0)
    t = np.arange(240)
    season = 1 + 0.3 * np.sin(2 * np.pi * t / 12)
    # multiplicative swings: the logarithm stabilises them
    y_log = np.exp(0.02 * t) * season * np.exp(rng.normal(0, 0.01, 240))
    assert sp.boxcox_lambda(y_log, 12) == pytest.approx(0.0, abs=0.08)
    # additive swings around a trend: no transformation
    y_lin = 100 + 2 * t + 20 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, 240)
    assert sp.boxcox_lambda(y_lin, 12) == pytest.approx(1.0, abs=0.15)
    lam = sp.boxcox_lambda(y_log, 12, lower=0.2, upper=0.9)
    assert 0.2 <= lam <= 0.9
    with pytest.raises(MethodIncompatibility, match="strictly positive"):
        sp.boxcox_lambda(y_lin - y_lin.mean(), 12)
    with pytest.raises(DataInsufficient, match="fewer than two cycles"):
        sp.boxcox_lambda(y_lin[:20], 12)


def test_fourier_terms_values_and_validation():
    X = sp.fourier_terms(24, period=12, K=2)
    t = np.arange(1, 25)
    assert list(X.columns) == ["sin1_12", "cos1_12", "sin2_12", "cos2_12"]
    np.testing.assert_allclose(X["sin1_12"], np.sin(2 * np.pi * t / 12))
    np.testing.assert_allclose(X["cos2_12"], np.cos(4 * np.pi * t / 12))
    # the K = period / 2 sine is identically zero and left out
    assert "sin6_12" not in sp.fourier_terms(24, 12, 6).columns
    # future rows continue the cycle
    fut = sp.fourier_terms(12, 12, 1, start=25)
    np.testing.assert_allclose(fut.to_numpy(), X.iloc[:12, :2].to_numpy(), atol=1e-12)
    # non-integer periods and an index are accepted
    idx = pd.date_range("2024-01-07", periods=10, freq="W")
    W = sp.fourier_terms(pd.Series(range(10), index=idx), 52.18, 1)
    assert W.index.equals(idx) and list(W.columns) == ["sin1_52.18", "cos1_52.18"]
    with pytest.raises(MethodIncompatibility, match="between 1 and period/2"):
        sp.fourier_terms(24, 12, 7)
    with pytest.raises(MethodIncompatibility, match="above 1"):
        sp.fourier_terms(24, 1, 1)


# ----------------------------------------------------------------------
# sp.stl / sp.classical_decompose
# ----------------------------------------------------------------------
def test_stl_components_add_up_and_recover_the_signal():
    rng = np.random.default_rng(0)
    idx = pd.period_range("2010-01", periods=180, freq="M")
    t = np.arange(180)
    trend = 50 + 0.2 * t
    seas = 6 * np.sin(2 * np.pi * t / 12)
    y = pd.Series(trend + seas + rng.normal(0, 0.5, 180), index=idx, name="y")
    dec = sp.stl(y, 12)
    np.testing.assert_allclose(dec.trend + dec.seasonal + dec.remainder, y, rtol=1e-12)
    assert np.sqrt(np.mean((dec.trend - trend) ** 2)) < 0.4
    assert np.sqrt(np.mean((dec.seasonal - seas) ** 2)) < 0.4
    st = dec.strength
    assert st["trend"] > 0.99 and st["seasonal"] > 0.97
    frame = dec.to_frame()
    assert frame.index.equals(idx)
    np.testing.assert_allclose(frame["seasonally_adjusted"], y - dec.seasonal)
    assert "strength of seasonality" in dec.summary()
    per = sp.stl(y, 12, seasonal="periodic")
    np.testing.assert_allclose(per.seasonal[:12], per.seasonal[12:24], rtol=1e-12)


def test_stl_robust_keeps_an_outlier_in_the_remainder():
    rng = np.random.default_rng(1)
    t = np.arange(144)
    y = 30 + 0.1 * t + 4 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.3, 144)
    y[70] += 25.0
    plain = sp.stl(y, 12)
    robust = sp.stl(y, 12, robust=True)
    assert abs(robust.remainder[70]) > abs(plain.remainder[70])
    assert abs(robust.remainder[70]) > 22.0
    assert robust.settings["outer_iter"] == 15 and plain.settings["outer_iter"] == 0


def test_stl_multiple_periods_and_forecast():
    rng = np.random.default_rng(2)
    t = np.arange(480)
    s1 = 3 * np.sin(2 * np.pi * t / 8)
    s2 = 5 * np.cos(2 * np.pi * t / 48)
    y = 20 + 0.01 * t + s1 + s2 + rng.normal(0, 0.4, 480)
    dec = sp.stl(y, [8, 48])
    assert dec.method == "MSTL" and dec.periods == (8, 48)
    assert np.corrcoef(dec.seasonal_components[8], s1)[0, 1] > 0.98
    assert np.corrcoef(dec.seasonal_components[48], s2)[0, 1] > 0.98
    assert {"seasonal_8", "seasonal_48"} <= set(dec.strength)
    assert {"seasonal_8", "seasonal_48"} <= set(dec.to_frame().columns)
    fc = dec.forecast(48, level=95, method="naive")
    truth = (
        20
        + 0.01 * np.arange(480, 528)
        + 3 * np.sin(2 * np.pi * np.arange(480, 528) / 8)
        + 5 * np.cos(2 * np.pi * np.arange(480, 528) / 48)
    )
    assert np.sqrt(np.mean((fc["forecast"] - truth) ** 2)) < 1.5
    for method in ("ets", "drift", "arima"):
        assert dec.forecast(4, method=method).shape == (4, 5)
    with pytest.raises(MethodIncompatibility, match="not available"):
        dec.forecast(4, method="prophet")


def test_stl_and_classical_refuse_bad_input():
    y = _seasonal_series()
    with pytest.raises(DataInsufficient, match="two cycles"):
        sp.stl(y[:7], 4)
    with pytest.raises(MethodIncompatibility, match="odd integer"):
        sp.stl(y, 4, seasonal=10)
    with pytest.raises(MethodIncompatibility, match="at least 2"):
        sp.stl(y, 1)
    with pytest.raises(MethodIncompatibility, match="seasonal windows"):
        sp.stl(np.tile(y, 3), [4, 12], seasonal=[11])
    with pytest.raises(MethodIncompatibility, match="strictly positive"):
        sp.classical_decompose(y - y.mean(), 4, model="multiplicative")
    with pytest.raises(MethodIncompatibility, match="additive"):
        sp.classical_decompose(y, 4, model="log")
    mult = sp.classical_decompose(y, 4, model="multiplicative")
    k = np.isfinite(mult.trend)
    np.testing.assert_allclose(
        (mult.trend * mult.seasonal * mult.remainder)[k], y[k], rtol=1e-12
    )
    assert np.isnan(mult.trend[:2]).all() and np.isnan(mult.trend[-2:]).all()
    with pytest.raises(MethodIncompatibility, match="multiplicative decomposition"):
        mult.forecast(4)


# ----------------------------------------------------------------------
# sp.arima additions
# ----------------------------------------------------------------------
def test_arima_regression_with_arma_errors_recovers_the_coefficients():
    rng = np.random.default_rng(4)
    n = 400
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    u = np.zeros(n)
    e = rng.normal(0, 0.5, n)
    for t in range(1, n):
        u[t] = 0.6 * u[t - 1] + e[t]
    df = pd.DataFrame({"y": 2.0 + 1.5 * x1 - 0.7 * x2 + u, "x1": x1, "x2": x2})
    fit = sp.arima("y", order=(1, 0, 0), exog=["x1", "x2"], data=df)
    assert list(fit.params.index) == ["const", "x1", "x2", "ar.L1", "sigma2"]
    assert fit.params["x1"] == pytest.approx(1.5, abs=0.08)
    assert fit.params["x2"] == pytest.approx(-0.7, abs=0.08)
    assert fit.params["ar.L1"] == pytest.approx(0.6, abs=0.1)
    assert fit.exog_names == ("x1", "x2") and fit.n_arma_params == 1
    # a scenario forecast: far ahead it is the regression line
    xf = pd.DataFrame({"x2": np.full(60, 1.0), "x1": np.full(60, 2.0)})
    fc = fit.forecast(60, level=95, exog=xf)
    expect = fit.params["const"] + 2.0 * fit.params["x1"] + 1.0 * fit.params["x2"]
    assert fc["forecast"].iloc[-1] == pytest.approx(expect, abs=1e-6)
    # array input, old-style columns
    old = fit.forecast(3, exog=np.array([[2.0, 1.0]] * 3))
    assert list(old.columns) == ["forecast", "lower", "upper"]
    np.testing.assert_allclose(old["forecast"], fc["forecast"].iloc[:3])


def test_arima_forecast_refuses_missing_or_malformed_regressors():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.normal(size=80), "x": rng.normal(size=80)})
    fit = sp.arima("y", order=(1, 0, 0), exog=["x"], data=df)
    with pytest.raises(MethodIncompatibility, match="their values over the"):
        fit.forecast(4)
    with pytest.raises(MethodIncompatibility, match="shape"):
        fit.forecast(4, exog=np.zeros((3, 1)))
    with pytest.raises(MethodIncompatibility, match="missing values"):
        fit.forecast(2, exog=np.array([[1.0], [np.nan]]))
    plain = sp.arima(df["y"], order=(1, 0, 0))
    with pytest.raises(MethodIncompatibility, match="no regressors"):
        plain.forecast(2, exog=np.zeros((2, 1)))
    with pytest.raises(MethodIncompatibility, match="not in data"):
        sp.arima("y", order=(1, 0, 0), exog=["z"], data=df)
    bad = df.copy()
    bad.loc[5, "x"] = np.nan
    with pytest.raises(MethodIncompatibility, match="exog has missing"):
        sp.arima("y", order=(1, 0, 0), exog=["x"], data=bad)


def test_arima_drift_forecast_and_levels():
    rng = np.random.default_rng(6)
    idx = pd.period_range("2000", periods=120, freq="Y")
    y = pd.Series(10 + np.cumsum(rng.normal(0.5, 1.0, 120)), index=idx)
    fit = sp.arima(y, order=(0, 1, 0), trend="c")
    assert fit.params["drift"] == pytest.approx(np.diff(y).mean(), rel=1e-6)
    fc = fit.forecast(5, level=(80, 95))
    np.testing.assert_allclose(
        fc["forecast"], y.iloc[-1] + fit.params["drift"] * np.arange(1, 6), rtol=1e-8
    )
    assert str(fc.index[0]) == "2120"
    # the adjusted variance widens the interval by sqrt(n / (n - k))
    adj = fit.forecast(5, level=95, dof_adjust=True)
    ratio = (adj["upper_95"] - adj["forecast"]) / (fc["upper_95"] - fc["forecast"])
    np.testing.assert_allclose(ratio, np.sqrt(119 / 118), rtol=1e-9)
    assert fit.sigma2 == pytest.approx(float(fit.params["sigma2"]))


@pytest.mark.parametrize("seed", [10, 13])
def test_arima_escapes_the_local_maximum_of_the_quasi_newton_search(seed):
    """ARMA(2, 2) on 70 observations with near-cancelling roots: the
    likelihood has a second, inferior maximum and the quasi-Newton search
    alone stops there (by 1.8 and 0.9 log-likelihood units on these two
    series; 4 of the first 31 seeds of this design are affected)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    rng = np.random.default_rng(seed)
    n = 70
    e = rng.normal(size=n + 60)
    x = np.zeros(n + 60)
    for t in range(2, n + 60):
        x[t] = 1.3 * x[t - 1] - 0.6 * x[t - 2] + e[t] - 0.8 * e[t - 1] + 0.3 * e[t - 2]
    y = 5 + x[60:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = SARIMAX(
            y, exog=np.ones(n), order=(2, 0, 2), concentrate_scale=True
        ).fit(disp=False, maxiter=500)
    fit = sp.arima(y, order=(2, 0, 2))
    assert fit.log_likelihood > plain.llf + 0.5


def test_auto_arima_finds_seasonal_orders_and_drift():
    rng = np.random.default_rng(8)
    n, m = 160, 4
    t = np.arange(n)
    level = 500 + np.cumsum(rng.normal(0.0, 1.0, n))
    y = level + np.tile([12.0, -8.0, 3.0, -7.0], n // m) + rng.normal(0, 0.5, n)
    fit = sp.arima(y, auto=True, period=m)
    assert fit.seasonal_order is not None
    assert fit.seasonal_order[1] == 1 and fit.seasonal_order[3] == m
    assert fit.candidates is not None and len(fit.candidates) >= 5
    assert fit.candidates["aicc"].min() == pytest.approx(fit.aicc, abs=0.5)
    # a fixed seasonal part: only (p, q) are searched
    fixed = sp.arima(y, auto=True, seasonal_order=(0, 1, 1, m))
    assert fixed.seasonal_order == (0, 1, 1, m)
    # a random walk with a strong drift keeps a drift term
    walk = 100 + np.cumsum(rng.normal(1.0, 0.5, 200))
    wfit = sp.arima(walk, auto=True)
    assert wfit.order[1] == 1 and "drift" in wfit.params.index
    assert wfit.params["drift"] == pytest.approx(1.0, abs=0.1)
    # trend='n' forbids it
    assert "drift" not in sp.arima(walk, auto=True, trend="n").params.index
    assert t[-1] == n - 1


def test_arima_period_argument_is_validated():
    y = _seasonal_series(mult=False)
    with pytest.raises(MethodIncompatibility, match="needs auto=True"):
        sp.arima(y, order=(1, 0, 0), period=4)
    with pytest.raises(MethodIncompatibility, match="contradicts"):
        sp.arima(y, auto=True, period=4, seasonal_order=(0, 1, 1, 12))
    with pytest.raises(MethodIncompatibility, match="positive integer"):
        sp.arima(y, auto=True, period=0)


# ----------------------------------------------------------------------
# sp.hierarchy / sp.reconcile
# ----------------------------------------------------------------------
def _long_table(seed=0, n_t=40):
    rng = np.random.default_rng(seed)
    rows = []
    for state, regions in {"A": ["a1", "a2"], "B": ["b1", "b2", "b3"]}.items():
        for region in regions:
            for kind in ("x", "y"):
                base = rng.uniform(20, 60)
                for t in range(n_t):
                    rows.append(
                        {
                            "all": "Total",
                            "state": state,
                            "region": region,
                            "kind": kind,
                            "t": t,
                            "v": base + 0.3 * t + rng.normal(0, 2.0),
                        }
                    )
    return pd.DataFrame(rows)


def test_hierarchy_builds_coherent_series_and_summing_matrix():
    df = _long_table()
    spec = [
        ["all"],
        ["all", "state"],
        ["all", "kind"],
        ["all", "state", "region"],
        ["all", "state", "region", "kind"],
    ]
    h = sp.hierarchy(df, spec, time="t", value="v")
    assert h.S.shape == (1 + 2 + 2 + 5 + 10, 10)
    assert list(h.tags) == [
        "all",
        "all/state",
        "all/kind",
        "all/state/region",
        "all/state/region/kind",
    ]
    np.testing.assert_allclose(
        h.Y.to_numpy(), h.Y[h.bottom].to_numpy() @ h.S.to_numpy().T, rtol=1e-12
    )
    assert h.Y["Total"].iloc[0] == pytest.approx(df.loc[df.t == 0, "v"].sum())
    assert h.Y["Total/A"].iloc[3] == pytest.approx(
        df.loc[(df.t == 3) & (df.state == "A"), "v"].sum()
    )
    assert "Hierarchy" in repr(h)


def test_hierarchy_refuses_unbalanced_duplicated_or_inconsistent_input():
    df = _long_table()
    spec = [["all"], ["all", "state", "region", "kind"]]
    with pytest.raises(MethodIncompatibility, match="unbalanced"):
        sp.hierarchy(df.drop(index=5), spec, time="t", value="v")
    with pytest.raises(MethodIncompatibility, match="duplicated"):
        sp.hierarchy(df, [["all"], ["all", "state"]], time="t", value="v")
    with pytest.raises(MethodIncompatibility, match="does not contain"):
        sp.hierarchy(df, [["kind"], ["all", "state", "region"]], time="t", value="v")
    with pytest.raises(MethodIncompatibility, match="not in data"):
        sp.hierarchy(df, [["country"]], time="t", value="v")
    bad = df.copy()
    bad.loc[0, "v"] = np.nan
    with pytest.raises(MethodIncompatibility, match="missing values"):
        sp.hierarchy(bad, spec, time="t", value="v")


def _base_forecasts(h, horizon=4, seed=1):
    rng = np.random.default_rng(seed)
    train = h.Y.iloc[:-horizon]
    fits = {c: sp.simple_forecast(train[c], "drift") for c in h.Y.columns}
    base = pd.DataFrame({c: f.forecast(horizon)["forecast"] for c, f in fits.items()})
    base = base * (1 + rng.normal(0, 0.02, base.shape))  # make them incoherent
    res = pd.DataFrame({c: f.residuals for c, f in fits.items()})
    return train, base, res


@pytest.mark.parametrize(
    "method",
    [
        "bottom_up",
        "top_down",
        "ols",
        "wls_struct",
        "wls_var",
        "mint_shrink",
        "mint_cov",
    ],
)
def test_reconciled_forecasts_are_coherent(method):
    h = sp.hierarchy(
        _long_table(n_t=60),
        [["all"], ["all", "state"], ["all", "state", "region", "kind"]],
        time="t",
        value="v",
    )
    train, base, res = _base_forecasts(h)
    rec = sp.reconcile(base, h, method=method, residuals=res, history=train)
    f = rec.forecasts
    S = h.S.to_numpy()
    np.testing.assert_allclose(f.to_numpy(), f[h.bottom].to_numpy() @ S.T, atol=1e-8)
    assert list(f.columns) == list(h.S.index)
    np.testing.assert_allclose(
        rec.adjustment.to_numpy(), (f - base[f.columns]).to_numpy(), atol=1e-12
    )
    # projections leave coherent forecasts unchanged (top-down does not)
    coherent = pd.DataFrame(
        base[h.bottom].to_numpy() @ S.T, columns=h.S.index, index=base.index
    )
    again = sp.reconcile(coherent, h, method=method, residuals=res, history=train)
    if method != "top_down":
        np.testing.assert_allclose(again.forecasts, coherent, rtol=1e-9)
    assert rec.method == method and "Forecast reconciliation" in rec.summary()


def test_reconcile_ols_by_hand_and_bottom_up_with_single_child():
    S = pd.DataFrame(
        [[1, 1], [1, 0], [0, 1]], index=["T", "A", "B"], columns=["A", "B"], dtype=float
    )
    base = pd.DataFrame([[10.0, 4.0, 5.0]], columns=["T", "A", "B"])
    rec = sp.reconcile(base, S, method="ols")
    # least squares: spread the discrepancy of 1 as +1/3, +1/3 and -1/3
    np.testing.assert_allclose(rec.forecasts.to_numpy(), [[29 / 3, 13 / 3, 16 / 3]])
    ws = sp.reconcile(base, S, method="wls_struct")
    np.testing.assert_allclose(ws.forecasts.to_numpy(), [[9.5, 4.25, 5.25]])
    # an aggregate with a single child has the same row of S as the child:
    # bottom-up must use the child's forecast
    S2 = pd.DataFrame(
        [[1, 1], [1, 0], [0, 1], [1, 0], [0, 1]],
        index=["T", "A", "B", "A/a", "B/b"],
        columns=["A/a", "B/b"],
        dtype=float,
    )
    base2 = pd.DataFrame([[10.0, 99.0, 99.0, 4.0, 5.0]], columns=S2.index)
    bu = sp.reconcile(base2, S2, method="bottom_up")
    np.testing.assert_allclose(bu.forecasts.to_numpy(), [[9.0, 4.0, 5.0, 4.0, 5.0]])


def test_mint_improves_on_base_forecasts_in_simulation():
    """Known truth: with unbiased base forecasts of known error variances
    the mean squared errors of the base, bottom-up and MinT forecasts are
    traces that can be written down."""
    rng = np.random.default_rng(0)
    S = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1], *np.eye(4)], dtype=float)
    ids = ["T", "A", "B", "a1", "a2", "b1", "b2"]
    Sd = pd.DataFrame(S, index=ids, columns=ids[3:])
    sd = np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0])  # bottom forecasts are poor
    truth = S @ np.array([10.0, 20.0, 30.0, 40.0])
    res = pd.DataFrame(rng.normal(0, 1, (200, 7)) * sd, columns=ids)
    mse = {"base": 0.0, "bottom_up": 0.0, "mint_shrink": 0.0, "ols": 0.0}
    reps = 300
    for _ in range(reps):
        base = pd.DataFrame([truth + rng.normal(0, 1, 7) * sd], columns=ids)
        mse["base"] += float(((base.to_numpy() - truth) ** 2).sum())
        for m in ("bottom_up", "mint_shrink", "ols"):
            f = sp.reconcile(base, Sd, method=m, residuals=res).forecasts.to_numpy()
            mse[m] += float(((f - truth) ** 2).sum())
    mse = {k: v / reps for k, v in mse.items()}
    # population values: tr(W) for the base forecasts, tr(S W_b S') for
    # bottom-up, tr(S G W G' S') with the true W for MinT
    W = np.diag(sd**2)
    G = np.linalg.solve(S.T @ np.linalg.solve(W, S), np.linalg.solve(W, S).T)
    best = float(np.trace(S @ G @ W @ G.T @ S.T))  # 20.57
    assert mse["base"] == pytest.approx(float(np.trace(W)), rel=0.10)  # 39
    assert mse["bottom_up"] == pytest.approx(
        float(np.trace(S @ W[3:, 3:] @ S.T)), rel=0.12
    )  # 108
    # MinT with an estimated covariance comes within 12% of the optimum
    assert best * 0.9 < mse["mint_shrink"] < best * 1.12
    assert mse["mint_shrink"] < mse["ols"] < mse["base"] < mse["bottom_up"]


def test_reconcile_validates_its_inputs():
    S = pd.DataFrame(
        [[1, 1], [1, 0], [0, 1]], index=["T", "A", "B"], columns=["A", "B"], dtype=float
    )
    base = pd.DataFrame([[10.0, 4.0, 5.0]], columns=["T", "A", "B"])
    res = pd.DataFrame(np.random.default_rng(0).normal(size=(3, 3)), columns=S.index)
    with pytest.raises(MethodIncompatibility, match="needs residuals"):
        sp.reconcile(base, S, method="mint_shrink")
    with pytest.raises(MethodIncompatibility, match="needs history"):
        sp.reconcile(base, S, method="top_down")
    with pytest.raises(MethodIncompatibility, match="not available"):
        sp.reconcile(base, S, method="erm")
    with pytest.raises(MethodIncompatibility, match="lacks the series"):
        sp.reconcile(base.drop(columns="B"), S, method="ols")
    with pytest.raises(DataInsufficient, match="singular"):
        sp.reconcile(base, S, method="mint_cov", residuals=res)
    with pytest.raises(MethodIncompatibility, match="more rows than columns"):
        sp.reconcile(base[["A", "B"]], S.loc[["A", "B"]], method="ols")
    with pytest.raises(MethodIncompatibility, match="missing values"):
        sp.reconcile(base * np.nan, S, method="ols")
    with pytest.raises(MethodIncompatibility, match="must be a DataFrame"):
        sp.reconcile(base, S.to_numpy(), method="ols")


# ----------------------------------------------------------------------
# registry and laziness
# ----------------------------------------------------------------------
def test_forecasting_functions_are_registered_and_ets_is_lazy():
    import subprocess
    import sys

    names = [
        "ets",
        "simple_forecast",
        "forecast_accuracy",
        "tscv",
        "stl",
        "classical_decompose",
        "ljungbox",
        "ndiffs",
        "nsdiffs",
        "boxcox_lambda",
        "fourier_terms",
        "hierarchy",
        "reconcile",
    ]
    listed = set(sp.list_functions())
    for nm in names:
        assert nm in listed
        card = sp.describe_function(nm)
        assert card["category"] == "timeseries"
        assert sp.function_schema(nm)
    code = (
        "import sys, statspai as sp; assert 'numba' not in sys.modules; "
        "sp.ets; assert 'numba' in sys.modules"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


# ----------------------------------------------------------------------
# reconciled prediction intervals
# ----------------------------------------------------------------------
def _three_level():
    S = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1], *np.eye(4)], dtype=float)
    ids = ["T", "A", "B", "a1", "a2", "b1", "b2"]
    return pd.DataFrame(S, index=ids, columns=ids[3:]), ids


def test_reconciled_sd_of_bottom_up_is_the_sum_of_independent_variances():
    Sd, ids = _three_level()
    base = pd.DataFrame([[100.0, 40.0, 60.0, 15.0, 25.0, 20.0, 40.0]], columns=ids)
    sd = pd.DataFrame([[9.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0]], columns=ids)
    rec = sp.reconcile(base, Sd, method="bottom_up", sd=sd)
    got = rec.sd.iloc[0]
    assert got[["a1", "a2", "b1", "b2"]].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert got["A"] == pytest.approx(np.sqrt(1 + 4))
    assert got["B"] == pytest.approx(np.sqrt(9 + 16))
    assert got["T"] == pytest.approx(np.sqrt(30))
    lo, hi = rec.intervals(95)
    z = 1.959963984540054
    np.testing.assert_allclose(hi - rec.forecasts, z * rec.sd, rtol=1e-12)
    np.testing.assert_allclose(rec.forecasts - lo, z * rec.sd, rtol=1e-12)
    lo80, _ = rec.intervals(0.8)
    assert (lo80 > lo).all().all()


@pytest.mark.parametrize("method", ["ols", "wls_struct", "wls_var", "mint_shrink"])
def test_reconciled_sd_is_the_sd_of_reconciled_normal_draws(method):
    """Independent evidence: draw base forecasts from the normal law the
    intervals assume, reconcile each draw, and compare standard deviations."""
    Sd, ids = _three_level()
    rng = np.random.default_rng(5)
    A = rng.normal(size=(7, 7))
    res = pd.DataFrame(rng.normal(size=(80, 7)) @ A * 0.3, columns=ids)
    base = pd.DataFrame(
        [Sd.to_numpy() @ np.array([15.0, 25.0, 20.0, 40.0])], columns=ids
    )
    sd = pd.DataFrame([[4.0, 3.0, 3.5, 1.5, 2.0, 2.5, 3.0]], columns=ids)
    rec = sp.reconcile(base, Sd, method=method, residuals=res, sd=sd)
    corr = np.eye(7)
    if method == "mint_shrink":
        d = np.sqrt(np.diag(rec.weights))
        corr = rec.weights / np.outer(d, d)
    cov = corr * np.outer(sd.to_numpy()[0], sd.to_numpy()[0])
    draws = rng.multivariate_normal(base.to_numpy()[0], cov, size=200_000)
    SG = Sd.to_numpy() @ rec.G.to_numpy()
    mc = (draws @ SG.T).std(axis=0)
    # Monte Carlo error of a standard deviation from 200k draws: 0.16%
    np.testing.assert_allclose(rec.sd.to_numpy()[0], mc, rtol=0.01)
    # coherent forecasts of the aggregates are less uncertain than the base ones
    assert rec.sd.iloc[0]["T"] < sd.iloc[0]["T"]


def test_reconcile_intervals_validate_their_inputs():
    Sd, ids = _three_level()
    base = pd.DataFrame([[100.0, 40.0, 60.0, 15.0, 25.0, 20.0, 40.0]], columns=ids)
    rec = sp.reconcile(base, Sd, method="ols")
    assert rec.sd is None
    with pytest.raises(MethodIncompatibility, match="no forecast standard deviations"):
        rec.intervals()
    with pytest.raises(MethodIncompatibility, match="sd has shape"):
        sp.reconcile(base, Sd, method="ols", sd=np.ones((2, 7)))
    with pytest.raises(MethodIncompatibility, match="missing or negative"):
        sp.reconcile(base, Sd, method="ols", sd=-np.ones((1, 7)))
    ok = sp.reconcile(base, Sd, method="ols", sd=np.ones((1, 7)))
    with pytest.raises(MethodIncompatibility, match="coverage"):
        ok.intervals(150)


def test_gam_smooth_term_default_is_valid_on_every_supported_python():
    """A dataclass default of ``slice(0, 0)`` made ``import statspai`` fail
    on Python 3.11, where ``slice`` is unhashable."""
    import dataclasses

    from statspai.regression import gam

    f = {x.name: x for x in dataclasses.fields(gam._Smooth)}["cols"]
    assert f.default is dataclasses.MISSING
    assert f.default_factory() == slice(0, 0)


# ----------------------------------------------------------------------
# Box-Cox inside the forecasters; ETS with gaps
# ----------------------------------------------------------------------
def test_boxcox_zero_is_the_model_of_the_logarithm():
    y = _seasonal_series()
    direct = sp.ets(np.log(y), "AAA", period=4)
    wrapped = sp.ets(y, "AAA", period=4, boxcox=0)
    assert wrapped.boxcox == 0.0
    assert wrapped.log_likelihood == pytest.approx(direct.log_likelihood, abs=1e-8)
    a, b = direct.forecast(6, level=90), wrapped.forecast(6, level=90)
    np.testing.assert_allclose(b["forecast"], np.exp(a["forecast"]), rtol=1e-10)
    np.testing.assert_allclose(b["lower_90"], np.exp(a["lower_90"]), rtol=1e-10)
    np.testing.assert_allclose(wrapped.fitted_values, np.exp(direct.fitted_values))
    # the bias-adjusted mean of a lognormal: exp(mu) (1 + sigma^2 / 2)
    adj = sp.ets(y, "AAA", period=4, boxcox=0, biasadj=True).forecast(6, level=90)
    sd = (a["upper_90"] - a["lower_90"]) / (2 * 1.6448536269514722)
    np.testing.assert_allclose(
        adj["forecast"], np.exp(a["forecast"]) * (1 + sd**2 / 2), rtol=1e-10
    )
    assert (adj["forecast"] > b["forecast"]).all()
    np.testing.assert_allclose(adj["upper_90"], b["upper_90"])


def test_boxcox_auto_and_validation():
    y = _seasonal_series()
    fit = sp.ets(y, period=4, boxcox="auto")
    assert fit.boxcox == pytest.approx(sp.boxcox_lambda(y, 4))
    assert "M" not in (fit.error, fit.trend, fit.season)  # additive only
    ar = sp.arima(y, order=(1, 1, 0), boxcox=0.5)
    assert ar.boxcox == 0.5
    assert ar.forecast(3).shape == (3, 3) and (ar.forecast(3)["lower"] > 0).all()
    # fitted values come back on the scale of the data
    assert np.isfinite(ar.fitted_values[5:]).all() and (ar.fitted_values[5:] > 0).all()
    assert 0.5 < np.median(ar.fitted_values[5:] / y[5:]) < 2.0
    with pytest.raises(MethodIncompatibility, match="strictly positive"):
        sp.simple_forecast(y - y.mean(), "naive", boxcox=0)
    with pytest.raises(MethodIncompatibility, match="not a number or 'auto'"):
        sp.ets(y, "ANN", boxcox="log")


def test_ets_with_missing_values_skips_them_in_the_likelihood():
    y = _seasonal_series(mult=False)
    full = sp.ets(y, "AAA", period=4)
    gap = y.copy()
    gap[[20, 21, 50]] = np.nan
    fit = sp.ets(gap, "AAA", period=4)
    assert fit.n == len(y) - 3 and fit.n_missing == 3
    assert np.isnan(fit.residuals[[20, 21, 50]]).all()
    assert np.isfinite(fit.fitted_values).all()
    # the fitted values at the gaps are forecasts of what was not seen
    assert np.abs(fit.fitted_values[[20, 21, 50]] - y[[20, 21, 50]]).max() < 6.0
    np.testing.assert_allclose(
        fit.forecast(4)["forecast"], full.forecast(4)["forecast"], rtol=0.02
    )
    # with a zero innovation at the gap the level does not move
    ses = sp.ets(np.r_[y[:30], np.nan, y[31:]], "ANN")
    lvl = ses.states["level"].to_numpy()
    assert lvl[31] == pytest.approx(lvl[30])


# ----------------------------------------------------------------------
# top-down by forecast proportions, middle-out
# ----------------------------------------------------------------------
def test_forecast_proportions_by_hand():
    Sd, ids = _three_level()
    base = pd.DataFrame([[100.0, 30.0, 60.0, 10.0, 30.0, 10.0, 40.0]], columns=ids)
    rec = sp.reconcile(base, Sd, method="top_down", proportions="forecast").forecasts
    # A gets 30 / 90 of the total, a1 gets 10 / 40 of A, and so on
    np.testing.assert_allclose(rec["A"], 100 * 30 / 90)
    np.testing.assert_allclose(rec["a1"], 100 * (30 / 90) * (10 / 40))
    np.testing.assert_allclose(rec["b2"], 100 * (60 / 90) * (40 / 50))
    np.testing.assert_allclose(rec["T"], 100.0)
    np.testing.assert_allclose(
        rec.to_numpy(), rec[ids[3:]].to_numpy() @ Sd.to_numpy().T
    )


def test_middle_out_keeps_the_middle_level_and_sums_above_it():
    Sd, ids = _three_level()
    base = pd.DataFrame([[100.0, 30.0, 60.0, 10.0, 30.0, 10.0, 40.0]], columns=ids)
    rec = sp.reconcile(
        base, Sd, method="middle_out", middle=["A", "B"], proportions="forecast"
    ).forecasts
    assert rec["A"].iloc[0] == pytest.approx(30.0)
    assert rec["B"].iloc[0] == pytest.approx(60.0)
    assert rec["T"].iloc[0] == pytest.approx(90.0)
    assert rec["a1"].iloc[0] == pytest.approx(30 * 10 / 40)
    hist = pd.DataFrame(
        np.array([[4.0, 6.0, 1.0, 9.0], [6.0, 4.0, 3.0, 7.0]]) @ Sd.to_numpy().T,
        columns=ids,
    )
    avg = sp.reconcile(base, Sd, method="middle_out", middle=["A", "B"], history=hist)
    assert avg.forecasts["a1"].iloc[0] == pytest.approx(30 * np.mean([0.4, 0.6]))
    assert np.isfinite(avg.G.to_numpy()).all()
    # a level name works when the structure comes from sp.hierarchy
    h = sp.hierarchy(
        _long_table(n_t=12)
        .groupby(["all", "state", "region", "t"], as_index=False)["v"]
        .sum(),
        [["all"], ["all", "state"], ["all", "state", "region"]],
        time="t",
        value="v",
    )
    b2 = h.Y.iloc[-2:].reset_index(drop=True) * 1.1
    mo = sp.reconcile(
        b2, h, method="middle_out", middle="all/state", proportions="forecast"
    ).forecasts
    np.testing.assert_allclose(mo[h.tags["all/state"]], b2[h.tags["all/state"]])


def test_top_down_variants_validate_their_inputs():
    Sd, ids = _three_level()
    base = pd.DataFrame([[100.0, 30.0, 60.0, 10.0, 30.0, 10.0, 40.0]], columns=ids)
    with pytest.raises(MethodIncompatibility, match="needs middle="):
        sp.reconcile(base, Sd, method="middle_out", proportions="forecast")
    with pytest.raises(MethodIncompatibility, match="does not split"):
        sp.reconcile(
            base, Sd, method="middle_out", middle=["A"], proportions="forecast"
        )
    with pytest.raises(MethodIncompatibility, match="needs history="):
        sp.reconcile(base, Sd, method="top_down", proportions="average")
    with pytest.raises(MethodIncompatibility, match="not a linear map"):
        sp.reconcile(
            base, Sd, method="top_down", proportions="forecast", sd=np.ones((1, 7))
        )
    grouped = pd.DataFrame(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            *np.eye(4),
        ],
        index=["T", "A", "B", "x", "y", "a1", "a2", "b1", "b2"],
        columns=["a1", "a2", "b1", "b2"],
        dtype=float,
    )
    with pytest.raises(MethodIncompatibility, match="strict hierarchy"):
        sp.reconcile(
            np.ones((1, 9)), grouped, method="top_down", proportions="forecast"
        )


# ----------------------------------------------------------------------
# features, bootstrapped series, bagging, seasonal dummies
# ----------------------------------------------------------------------
def test_ts_features_values_and_table_input():
    rng = np.random.default_rng(0)
    t = np.arange(144)
    wide = pd.DataFrame(
        {
            "seasonal": 5 * np.sin(2 * np.pi * t / 12) + rng.normal(size=144),
            "trend": 0.2 * t + rng.normal(size=144),
            "noise": rng.normal(size=144),
        }
    )
    f = sp.ts_features(wide, period=12)
    assert list(f.index) == ["seasonal", "trend", "noise"]
    assert (
        f.loc["seasonal", "seasonal_strength"]
        > 0.9
        > f.loc["noise", "seasonal_strength"]
    )
    assert f.loc["trend", "trend"] > 0.95 and f.loc["trend", "x_acf1"] > 0.9
    assert abs(f.loc["noise", "x_acf1"]) < 0.3  # 3.6 standard errors at n = 144
    assert f.loc["seasonal", "seas_acf1"] > 0.8
    one = sp.ts_features(wide["noise"], period=12)
    pd.testing.assert_series_equal(one, f.loc["noise"], check_names=False)
    # scaling: the features of a shape do not depend on its units
    pd.testing.assert_series_equal(
        sp.ts_features(1000 * wide["seasonal"] + 7, 12),
        f.loc["seasonal"],
        check_names=False,
        rtol=1e-9,
    )
    x = np.array([0.0, 1, 0, 1, 0, 1, 0, 1, 5, 5, 5, 5, 0, 1])
    g = sp.ts_features(x, scale=False)
    # the median is 1: only the run of fives lies above it, entered and left once
    assert g["flat_spots"] == 4 and g["crossing_points"] == 2
    assert "seasonal_strength" not in g.index
    with pytest.raises(DataInsufficient, match="too few"):
        sp.ts_features(x[:5])
    with pytest.raises(DataInsufficient, match="constant"):
        sp.ts_features(np.ones(40))


def test_bootstrap_series_keeps_the_structure_of_the_series():
    rng = np.random.default_rng(0)
    idx = pd.period_range("2000Q1", periods=96, freq="Q")
    t = np.arange(96)
    y = pd.Series(
        (50 + t)
        * (1 + 0.2 * np.sin(2 * np.pi * t / 4))
        * np.exp(rng.normal(0, 0.03, 96)),
        index=idx,
    )
    boot = sp.bootstrap_series(y, 40, period=4, seed=1)
    assert boot.shape == (96, 40) and boot.index.equals(idx)
    np.testing.assert_allclose(boot["boot_0"], y)
    assert (boot > 0).all().all()
    again = sp.bootstrap_series(y, 40, period=4, seed=1)
    pd.testing.assert_frame_equal(boot, again)
    assert not np.allclose(boot["boot_1"], boot["boot_2"])
    # every bootstrapped series is as seasonal and as trended as the original
    feats = sp.ts_features(boot, period=4)
    assert feats["seasonal_strength"].min() > 0.9 and feats["trend"].min() > 0.95
    # the ensemble is centred on the series
    assert np.abs(boot.iloc[:, 1:].mean(axis=1) / y - 1).max() < 0.08
    flat = sp.bootstrap_series(100 + np.cumsum(rng.normal(size=60)), 5, boxcox=None)
    assert flat.shape == (60, 5)
    with pytest.raises(DataInsufficient, match="too few"):
        sp.bootstrap_series(np.arange(6.0) + 1, 3)
    with pytest.raises(MethodIncompatibility, match="block_size"):
        sp.bootstrap_series(y, 3, period=4, block_size=500)


def test_bagged_forecast_averages_its_members():
    y = _seasonal_series(n=72)
    bag = sp.bagged_forecast(y, "ets", horizon=4, n_boot=8, period=4, model="MAM")
    assert bag.members.shape == (4, 8) and bag.n_boot == 8
    np.testing.assert_allclose(bag.forecast["forecast"], bag.members.mean(axis=1))
    np.testing.assert_allclose(bag.forecast["ensemble_max"], bag.members.max(axis=1))
    single = sp.ets(y, "MAM", period=4).forecast(4)["forecast"].to_numpy()
    np.testing.assert_allclose(bag.members["boot_0"], single, rtol=1e-9)
    np.testing.assert_allclose(bag.forecast["forecast"], single, rtol=0.05)
    fn = sp.bagged_forecast(
        y, lambda s, h: np.full(h, s.iloc[-1]), horizon=2, n_boot=5, period=4
    )
    assert fn.forecast.shape == (2, 3) and "Bagged forecast" in fn.summary()
    with pytest.raises(MethodIncompatibility, match="not a built-in"):
        sp.bagged_forecast(y, "prophet", n_boot=3, period=4)


def test_seasonal_dummies_and_regression_use():
    X = sp.seasonal_dummies(8, 4)
    assert list(X.columns) == ["season_2", "season_3", "season_4"]
    assert X.sum().tolist() == [2, 2, 2] and X.iloc[0].sum() == 0
    full = sp.seasonal_dummies(8, 4, drop_first=False)
    assert (full.sum(axis=1) == 1).all()
    fut = sp.seasonal_dummies(2, 4, start=9)
    assert fut.values.tolist() == [[0, 0, 0], [1, 0, 0]]
    y = _seasonal_series(mult=False)
    df = pd.concat(
        [pd.DataFrame({"y": y, "trend": np.arange(1, 97)}), sp.seasonal_dummies(96, 4)],
        axis=1,
    )
    fit = sp.regress("y ~ trend + season_2 + season_3 + season_4", df)
    assert fit.params["trend"] == pytest.approx(0.8, abs=0.02)
    assert fit.params["season_2"] == pytest.approx(20 * (1.1 - 0.9), abs=0.8)
    with pytest.raises(MethodIncompatibility, match="at least 2"):
        sp.seasonal_dummies(8, 1)
