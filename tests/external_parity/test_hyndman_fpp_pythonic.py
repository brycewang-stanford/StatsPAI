"""Numbers printed in Hyndman et al., *Forecasting: Principles and
Practice, the Pythonic Way* (OTexts), reproduced with ``sp``.

Opt-in: the book's data are not shipped. Download ``fpppy_data.zip`` from
the book's site, unpack it and point ``STATSPAI_FPPPY_DIR`` at the folder
that holds the ``.csv`` files::

    STATSPAI_FPPPY_DIR=/path/to/data pytest \
        tests/external_parity/test_hyndman_fpp_pythonic.py

The book runs statsforecast, statsmodels and hierarchicalforecast. Each
expected value below is the number printed in the chapter named in the
test, to the digits printed there. Where ``sp`` and R's ``forecast``
agree with each other and not with the book -- statsforecast's ARIMA and
ETS optimisers stop short of the maximum on several models -- the test
says so and pins the R value (``forecast`` 9.0.2).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_DIR = os.environ.get("STATSPAI_FPPPY_DIR")
pytestmark = pytest.mark.skipif(
    not _DIR or not (Path(_DIR) / "global_economy.csv").exists(),
    reason="set STATSPAI_FPPPY_DIR to the folder with the book's csv files",
)


def _csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(Path(_DIR) / name, **kw)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def goog():
    g = _csv("gafa_stock.csv", parse_dates=["ds"])
    g = g[(g["unique_id"] == "GOOG_Close")]
    return g[g["ds"].dt.year == 2015]["y"].to_numpy(), g[
        (g["ds"] >= "2016-01-01") & (g["ds"] <= "2016-01-31")
    ]["y"].to_numpy()


@pytest.fixture(scope="module")
def beer():
    p = _csv("aus_production_formatted.csv", parse_dates=["ds"])
    b = p[(p["unique_id"] == "Beer") & (p["ds"] >= "1992")]
    return b[b["ds"].dt.year <= 2007]["y"].to_numpy(), b[b["ds"].dt.year > 2007][
        "y"
    ].to_numpy()


def _exports(code: str) -> np.ndarray:
    ge = _csv("global_economy.csv")
    return ge.loc[ge["Code"] == code, "Exports"].to_numpy(dtype=float)


# ---- chapter 3 ----------------------------------------------------------
def test_ch3_guerrero_lambda_for_gas_production():
    gas = _csv("aus_production.csv")["Gas"].to_numpy(dtype=float)
    assert sp.boxcox_lambda(gas, period=4) == pytest.approx(0.11, abs=0.005)


def test_ch3_stl_of_retail_employment_reproduces_statsmodels():
    from statsmodels.tsa.seasonal import STL

    us = _csv("us_employment.csv", parse_dates=["ds"])
    y = us[(us["unique_id"] == "Retail Trade") & (us["ds"] >= "1990")]["y"].to_numpy()
    ref = STL(y, period=12, seasonal=13, trend=21, robust=True).fit()
    dec = sp.stl(
        y, 12, seasonal=13, trend=21, robust=True, seasonal_deg=1,
        inner_iter=2, outer_iter=15, seasonal_jump=1, trend_jump=1, low_pass_jump=1,
    )
    np.testing.assert_allclose(dec.trend, ref.trend, rtol=1e-10)
    np.testing.assert_allclose(dec.seasonal, ref.seasonal, rtol=1e-8, atol=1e-8)
    assert dec.strength["trend"] > 0.99 and dec.strength["seasonal"] > 0.95


# ---- chapter 5 ----------------------------------------------------------
def test_ch5_ljung_box_of_naive_residuals(goog):
    train, _ = goog
    out = sp.ljungbox(np.diff(train), lags=[1, 2, 3, 10])
    stat, pval = out["statistic"].iloc[:3], out["p_value"].iloc[:3]
    np.testing.assert_allclose(stat, [2.42, 3.76, 5.19], atol=0.005)
    np.testing.assert_allclose(pval, [0.120, 0.153, 0.158], atol=0.0006)
    bp = sp.ljungbox(np.diff(train), lags=[1, 2, 3], method="box-pierce")
    np.testing.assert_allclose(bp["statistic"], [2.39, 3.71, 5.11], atol=0.005)
    # section 9.1 prints the lag-10 test of the same series
    assert out.loc[10, "statistic"] == pytest.approx(7.914, abs=0.001)
    assert out.loc[10, "p_value"] == pytest.approx(0.637, abs=0.001)


def test_ch5_accuracy_table_of_the_benchmark_methods_on_beer(beer):
    train, test = beer
    fcs = {
        "drift": sp.simple_forecast(train, "drift").forecast(len(test)),
        "mean": sp.simple_forecast(train, "mean").forecast(len(test)),
        "naive": sp.simple_forecast(train, "naive").forecast(len(test)),
        "snaive": sp.simple_forecast(train, "snaive", period=4).forecast(len(test)),
    }
    acc = sp.forecast_accuracy(test, fcs, train=train, period=4)
    book = {  # RMSE, MAE, MAPE, MASE
        "drift": (64.90, 58.88, 14.58, 4.117),
        "mean": (38.45, 34.83, 8.28, 2.435),
        "naive": (62.69, 57.40, 14.18, 4.014),
        "snaive": (14.31, 13.40, 3.17, 0.937),
    }
    for name, (rmse, mae, mape, mase) in book.items():
        row = acc.loc[name]
        assert row["RMSE"] == pytest.approx(rmse, abs=0.006)
        assert row["MAE"] == pytest.approx(mae, abs=0.006)
        assert row["MAPE"] == pytest.approx(mape, abs=0.006)
        assert row["MASE"] == pytest.approx(mase, abs=0.0006)


def test_ch5_accuracy_table_on_google_stock(goog):
    train, test = goog
    fcs = {
        m: sp.simple_forecast(train, m).forecast(len(test))
        for m in ("drift", "mean", "naive")
    }
    acc = sp.forecast_accuracy(test, fcs, train=train)
    book = {"drift": (53.1, 49.8, 6.99), "mean": (118.0, 116.9, 16.24),
            "naive": (43.4, 40.4, 5.67)}
    for name, (rmse, mae, mape) in book.items():
        assert acc.loc[name, "RMSE"] == pytest.approx(rmse, abs=0.06)
        assert acc.loc[name, "MAE"] == pytest.approx(mae, abs=0.06)
        assert acc.loc[name, "MAPE"] == pytest.approx(mape, abs=0.006)


def test_ch5_cross_validated_rmse_grows_with_the_horizon(goog):
    train, _ = goog
    cv = sp.tscv(train, "drift", horizon=8, initial=3)
    rmse = cv.accuracy()["RMSE"].to_numpy()
    assert (np.diff(rmse) > 0).all()


# ---- chapter 8 ----------------------------------------------------------
def test_ch8_simple_exponential_smoothing_of_algerian_exports():
    y = _csv("algeria_exports.csv")["y"].to_numpy(dtype=float)
    fit = sp.ets(y, "ANN")
    assert fit.params["alpha"] == pytest.approx(0.8401, abs=2e-3)
    assert fit.initial_state["l"] == pytest.approx(39.53, abs=0.02)
    fc = fit.forecast(5, level=(80, 95))
    assert np.ptp(fc["forecast"]) < 1e-12
    assert fc["forecast"].iloc[0] == pytest.approx(22.44, abs=0.02)


def test_ch8_holt_and_damped_trend_reach_a_higher_likelihood_than_the_book():
    """The book's Holt fit of the Australian population stops at
    alpha = 0.9999, beta* = 0.2001; R's ets reports beta = 0.3267 for the
    same model, and sp.ets is at least as likely as R's fit."""
    y = _csv("aus_economy.csv")["y"].to_numpy(dtype=float) / 1e6
    fit = sp.ets(y, "AAN")
    assert fit.params["alpha"] == pytest.approx(0.9999, abs=1e-3)
    assert fit.log_likelihood >= 49.8730 - 1e-4  # forecast::ets, same model
    damped = sp.ets(y, "AAdN", phi=0.9)
    assert damped.params["phi"] == 0.9
    h = np.diff(damped.forecast(15)["forecast"])
    assert (h[1:] < h[:-1]).all()  # the damped forecast flattens


def test_ch8_ets_of_holiday_trips_selects_a_multiplicative_error_model():
    t = _csv("tourism.csv", parse_dates=["ds"])
    y = (t[t["Purpose"] == "Holiday"].groupby("ds")["y"].sum() / 1e3).to_numpy()
    fit = sp.ets(y, period=4)
    assert fit.error == "M" and fit.trend == "N" and fit.season in ("A", "M")
    # forecast::ets picks ETS(M,N,A) with AICc 227.784; ours is no worse
    assert fit.aicc <= 227.784 + 1e-3
    assert sp.ljungbox(fit, lags=8)["p_value"].iloc[0] > 0.05


# ---- chapter 9 ----------------------------------------------------------
def test_ch9_kpss_and_differencing_of_google_stock(goog):
    train, _ = goog
    assert sp.unitroot(train, test="kpss", lags=5).statistic == pytest.approx(
        3.561, abs=0.001
    )
    assert sp.unitroot(np.diff(train), test="kpss", lags=5).statistic == pytest.approx(
        0.099, abs=0.001
    )
    assert sp.ndiffs(train) == 1
    retail = _csv("aus_retail.csv", parse_dates=["Month"])
    total = np.log(retail.groupby("Month")["Turnover"].sum().to_numpy())
    assert sp.nsdiffs(total, 12) == 1
    assert sp.ndiffs(total[12:] - total[:-12]) == 1


def test_ch9_arima_for_egyptian_exports():
    y = _exports("EGY")
    fit = sp.arima(y, auto=True)
    assert fit.order == (2, 0, 1)
    assert fit.params["ar.L1"] == pytest.approx(1.68, abs=0.005)
    assert fit.params["ar.L2"] == pytest.approx(-0.80, abs=0.005)
    assert fit.params["ma.L1"] == pytest.approx(-0.69, abs=0.005)
    assert fit.params["const"] == pytest.approx(20.18, abs=0.005)
    assert fit.aicc == pytest.approx(294.29, abs=0.005)
    assert sp.arima(y, order=(4, 0, 0)).aicc == pytest.approx(294.70, abs=0.005)
    # sigma^2 = 8.05 in the book is the degrees-of-freedom adjusted variance
    assert fit.sigma2 * 58 / 54 == pytest.approx(8.046, abs=0.002)


def test_ch9_arima_for_central_african_republic_exports():
    y = _exports("CAF")
    table = {(2, 1, 0): 275, (0, 1, 3): 275, (3, 1, 0): 275, (2, 1, 2): 275}
    for order, aicc in table.items():
        assert round(sp.arima(y, order=order).aicc) == aicc
    assert sp.arima(y, auto=True).order == (2, 1, 2)  # stepwise, as in the book
    full = sp.arima(y, auto=True, stepwise=False)
    assert full.order == (3, 1, 0)
    out = sp.ljungbox(full, lags=10)
    assert out["df"].iloc[0] == 7 and out["p_value"].iloc[0] > 0.05


def test_ch9_seasonal_arima_agrees_with_r_not_with_the_book():
    """US leisure employment, ARIMA(2,1,0)(1,1,1)[12]. statsforecast stops
    at sar1 = -0.046, log likelihood 390.49; R's Arima reaches sar1 =
    0.3295 and 394.96, which is what sp.arima reports."""
    us = _csv("us_employment.csv", parse_dates=["ds"])
    y = (
        us[(us["unique_id"] == "Leisure and Hospitality") & (us["ds"] >= "2001")][
            "y"
        ].to_numpy()
        / 1e3
    )
    fit = sp.arima(y, order=(2, 1, 0), seasonal_order=(1, 1, 1, 12))
    assert fit.params["ar.S.L12"] == pytest.approx(0.3295, abs=2e-3)
    assert fit.params["ma.S.L12"] == pytest.approx(-0.7507, abs=2e-3)
    assert fit.log_likelihood == pytest.approx(394.96, abs=0.03)
    assert fit.log_likelihood > 390.49 + 4.0


# ---- chapter 10 ---------------------------------------------------------
def test_ch10_regression_with_arima_errors_for_us_consumption():
    us = _csv("US_change.csv", parse_dates=["ds"])
    fit = sp.arima("y", order=(1, 0, 2), exog=["Income"], data=us)
    assert fit.params["ar.L1"] == pytest.approx(0.707, abs=0.002)
    assert fit.params["ma.L1"] == pytest.approx(-0.617, abs=0.002)
    assert fit.params["const"] == pytest.approx(0.595, abs=0.002)
    assert fit.params["Income"] == pytest.approx(0.198, abs=0.002)
    assert fit.aicc == pytest.approx(338.51, abs=0.01)
    assert sp.arima("y", auto=True, exog=["Income"], data=us).order == (1, 0, 2)
    future = np.full((8, 1), us["Income"].mean())
    fc = fit.forecast(8, level=(80, 95), exog=future)
    assert fc.shape == (8, 5) and (fc["lower_95"] < fc["forecast"]).all()
    out = sp.ljungbox(fit, lags=8)
    assert out["df"].iloc[0] == 5


# ---- chapter 11 ---------------------------------------------------------
def test_ch11_tourism_hierarchy_and_reconciliation():
    t = _csv("aus_tourism.csv", parse_dates=["ds"])
    parts = t["unique_id"].str.split("-")
    t = t.assign(
        Country="Australia", Region=parts.str[0], State=parts.str[1],
        Purpose=parts.str[2],
    )
    spec = [
        ["Country"], ["Country", "State"], ["Country", "Purpose"],
        ["Country", "State", "Region"], ["Country", "State", "Purpose"],
        ["Country", "State", "Region", "Purpose"],
    ]
    h = sp.hierarchy(t, spec, time="ds", value="y")
    assert h.S.shape == (425, 304)
    assert len(h.tags["Country/State"]) == 8
    assert len(h.tags["Country/State/Region"]) == 76
    train = h.Y[h.Y.index < "2016"]
    test = h.Y[h.Y.index >= "2016"]
    fits = {c: sp.simple_forecast(train[c], "snaive", period=4) for c in h.Y.columns}
    base = pd.DataFrame(
        {c: f.forecast(8)["forecast"].to_numpy() for c, f in fits.items()},
        index=test.index,
    )
    res = pd.DataFrame({c: f.residuals for c, f in fits.items()})
    S = h.S.to_numpy()
    for method in ("bottom_up", "ols", "wls_struct", "mint_shrink"):
        rec = sp.reconcile(base, h, method=method, residuals=res).forecasts
        np.testing.assert_allclose(
            rec.to_numpy(), rec[h.bottom].to_numpy() @ S.T, rtol=1e-8, atol=1e-6
        )
    # seasonal naive forecasts of sums are sums of seasonal naive
    # forecasts: the base forecasts are already coherent and stay put
    np.testing.assert_allclose(
        sp.reconcile(base, h, method="ols").forecasts.to_numpy(),
        base[h.S.index].to_numpy(), rtol=1e-8, atol=1e-6,
    )


# ---- chapter 12 ---------------------------------------------------------
def test_ch12_var_lag_orders_for_us_consumption_and_income():
    us = _csv("US_change.csv", parse_dates=["ds"])
    cols = ["y", "Income", "Production", "Savings", "Unemployment"]
    soc = sp.varsoc(us, cols, maxlag=8)
    assert int(soc["AIC"].idxmin()) == 5  # "Lags selected by AIC: 5"
    assert int(soc["SBIC"].idxmin()) == 1  # "Lags selected by BIC: 1"


def test_ch12_mstl_of_half_hourly_demand_has_two_seasonal_patterns():
    v = _csv("vic_elec.csv", parse_dates=["ds"])
    y = v[v["unique_id"] == "Demand"]["y"].to_numpy(dtype=float)[-48 * 7 * 12 :]
    dec = sp.stl(y, [48, 48 * 7])
    st = dec.strength
    assert st["seasonal_48"] > 0.8 and st["seasonal_336"] > 0.3
    np.testing.assert_allclose(
        dec.trend + dec.seasonal + dec.remainder, y, rtol=1e-10
    )
