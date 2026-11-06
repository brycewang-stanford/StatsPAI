"""Time-series input hygiene (rOpenSci audit gaps #7, #12, #13).

* Dates that skip a period of a regular series, a shuffled date index and
  repeated dates are refused by every time-series entry that reads a
  series or a frame in row order (``_forecast_common.require_time_order``).
* A fixed-order ``sp.arima`` fit with a lag-polynomial root near the unit
  circle is flagged (warning + ``model_info`` + ``violations()``).
* ``.plot()`` of ARIMA / ETS / STL results draws against the stored time
  index and labels the axis with its frequency.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.exceptions import AssumptionWarning, MethodIncompatibility  # noqa: E402
from statspai.timeseries._forecast_common import (  # noqa: E402
    date_frequency,
    frequency_label,
    missing_periods,
    time_axis,
)


def _ar1(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


@pytest.fixture
def monthly() -> pd.Series:
    idx = pd.date_range("2000-01-01", periods=120, freq="MS")
    return pd.Series(_ar1(120, 0.5, 1) + 10.0, index=idx, name="y")


@pytest.fixture
def quarterly_frame() -> pd.DataFrame:
    idx = pd.period_range("1990Q1", periods=120, freq="Q")
    return pd.DataFrame(
        {"a": _ar1(120, 0.5, 2), "b": _ar1(120, 0.3, 3)},
        index=idx,
    )


# ----------------------------------------------------------------------
# frequency inference and the gap finder
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "index, alias, missing",
    [
        (pd.date_range("2000", periods=40, freq="QS"), "QS-JAN", []),
        (
            pd.date_range("2000", periods=40, freq="YE").delete([3, 4]),
            "Y",
            ["2003", "2004"],
        ),
        (
            pd.date_range("2000-01-05", periods=40, freq="W-WED").delete(5),
            "W-WED",
            ["2000-02-03/2000-02-09"],
        ),
        (
            pd.period_range("2000Q1", periods=20, freq="Q").delete(7),
            "Q-DEC",
            ["2001Q4"],
        ),
        # mid-month dates: the month each falls in is what counts
        (
            (
                pd.date_range("2000-01-01", periods=24, freq="MS")
                + pd.Timedelta(days=14)
            ).delete(3),
            "M",
            ["2000-04"],
        ),
    ],
)
def test_missing_periods_names_the_skipped_periods(index, alias, missing):
    got_alias, got = missing_periods(index)
    assert got_alias == alias
    assert [str(p) for p in got] == missing


def test_daily_and_irregular_dates_are_not_checked():
    # a trading-day calendar leaves out holidays on purpose
    bdays = pd.bdate_range("2020-01-01", periods=300).delete([10, 50])
    assert missing_periods(bdays)[0] is None
    assert len(missing_periods(bdays)[1]) == 0
    # irregular spacing has no frequency to compare against
    odd = pd.DatetimeIndex(["2000-01-01", "2000-01-20", "2000-02-03", "2000-03-01"])
    assert date_frequency(odd) is None
    assert len(missing_periods(odd)[1]) == 0
    # two dates in one month: not a monthly series
    two = pd.DatetimeIndex(["2000-01-01", "2000-01-31", "2000-03-01", "2000-04-01"])
    assert len(missing_periods(two)[1]) == 0


def test_frequency_label_words():
    assert frequency_label(pd.date_range("2000", periods=5, freq="MS")) == "monthly"
    assert (
        frequency_label(pd.period_range("2000Q1", periods=5, freq="Q")) == "quarterly"
    )
    assert frequency_label(pd.date_range("2000", periods=5, freq="D")) == "daily"
    assert frequency_label(pd.RangeIndex(5)) is None


# ----------------------------------------------------------------------
# univariate entries: a skipped month is refused, naming the months
# ----------------------------------------------------------------------
_UNIVARIATE = {
    "arima": lambda s: sp.arima(s, order=(1, 0, 0)),
    "ets": lambda s: sp.ets(s, period=12),
    "ndiffs": lambda s: sp.ndiffs(s),
    "garch": lambda s: sp.garch(s),
    "unitroot": lambda s: sp.unitroot(s),
    "stl": lambda s: sp.stl(s, period=12),
    "simple_forecast": lambda s: sp.simple_forecast(s, method="naive"),
    "tsfilter": lambda s: sp.tsfilter(s),
    "corrgram": lambda s: sp.corrgram(s),
    "periodogram": lambda s: sp.periodogram(s),
    "beveridge_nelson": lambda s: sp.beveridge_nelson(s),
    "mswitch": lambda s: sp.mswitch(s),
    "bagged_forecast": lambda s: sp.bagged_forecast(
        s, "ets", horizon=2, n_boot=2, period=12
    ),
}


@pytest.mark.parametrize("name", sorted(_UNIVARIATE))
def test_univariate_refuses_a_skipped_month(monthly, name):
    gapped = monthly.drop(monthly.index[[30, 31]])
    with pytest.raises(MethodIncompatibility, match="skips 2 period") as info:
        _UNIVARIATE[name](gapped)
    diag = info.value.diagnostics
    assert diag["n_missing_periods"] == 2
    assert diag["missing_periods"] == ["2002-07", "2002-08"]


@pytest.mark.parametrize("name", sorted(_UNIVARIATE))
def test_univariate_refuses_a_shuffled_index(monthly, name):
    shuffled = monthly.sample(frac=1.0, random_state=0)
    with pytest.raises(MethodIncompatibility, match="not in increasing order"):
        _UNIVARIATE[name](shuffled)


def test_dates_do_not_change_a_clean_fit(monthly):
    # the same values with and without their (regular) dates
    with_dates = sp.arima(monthly, order=(1, 0, 0))
    without = sp.arima(monthly.to_numpy(), order=(1, 0, 0))
    np.testing.assert_allclose(with_dates.params, without.params, rtol=1e-12)
    np.testing.assert_allclose(with_dates.se, without.se, rtol=1e-12)


def test_the_suggested_repair_is_accepted(monthly):
    gapped = monthly.drop(monthly.index[[30, 31]])
    full = gapped.asfreq("MS")  # the two months come back as missing values
    assert int(full.isna().sum()) == 2
    fit = sp.ets(full, period=12)  # ETS skips them in the likelihood
    assert (fit.n, fit.n_missing) == (118, 2)  # observed, skipped
    filled = sp.arima(full.interpolate(), order=(1, 0, 0))
    assert filled.n == 120
    # declaring the values consecutive is the other way out
    assert sp.arima(gapped.to_numpy(), order=(1, 0, 0)).n == 118


def test_trading_days_with_holidays_still_fit():
    idx = pd.bdate_range("2020-01-01", periods=400).delete([10, 50, 200])
    r = pd.Series(np.random.default_rng(4).normal(0, 1, len(idx)), index=idx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # boundary note on noise
        fit = sp.garch(r)
    assert fit.n == len(idx)


def test_time_column_with_a_skipped_quarter_is_refused():
    df = pd.DataFrame(
        {
            "q": pd.period_range("1990Q1", periods=80, freq="Q").to_timestamp(),
            "y": _ar1(80, 0.6, 5),
        }
    ).drop(index=[20])
    shuffled = df.sample(frac=1.0, random_state=1)
    with pytest.raises(MethodIncompatibility, match="quarterly date index skips 1"):
        sp.unitroot(shuffled, "y", time="q")


# ----------------------------------------------------------------------
# multivariate entries
# ----------------------------------------------------------------------
_MULTIVARIATE = {
    "var": lambda d: sp.var(d, lags=1),
    "granger_causality": lambda d: sp.granger_causality(
        data=d, caused="a", causing="b", lags=1
    ),
    "vec": lambda d: sp.vec(d, lags=1, rank=1),
    "johansen": lambda d: sp.johansen(d, lags=1),
    "engle_granger": lambda d: sp.engle_granger(d, variables=["a", "b"]),
    "varsoc": lambda d: sp.varsoc(d, maxlag=2),
    "bvar": lambda d: sp.bvar(d, lags=1),
    "ardl": lambda d: sp.ardl(d, "a", x="b", lags=1),
    "local_projections": lambda d: sp.local_projections(
        d, outcome="a", shock="b", horizons=2
    ),
    "johansen_lrtest": lambda d: sp.johansen_lrtest(d, rank=1, beta_known=[1.0, -1.0]),
    "tvp_var": lambda d: sp.tvp_var(d, lags=1),
    "tvp_var_sv": lambda d: sp.tvp_var_sv(d, lags=1),
    "dlm": lambda d: sp.dlm("a ~ b", d),
}


@pytest.mark.parametrize("name", sorted(_MULTIVARIATE))
def test_multivariate_refuses_a_skipped_quarter(quarterly_frame, name):
    gapped = quarterly_frame.drop(quarterly_frame.index[[40]])
    with pytest.raises(MethodIncompatibility, match="skips 1 period") as info:
        _MULTIVARIATE[name](gapped)
    assert info.value.diagnostics["missing_periods"] == ["2000Q1"]


@pytest.mark.parametrize("name", sorted(_MULTIVARIATE))
def test_multivariate_refuses_a_shuffled_frame(quarterly_frame, name):
    shuffled = quarterly_frame.sample(frac=1.0, random_state=0)
    with pytest.raises(MethodIncompatibility, match="not in increasing order"):
        _MULTIVARIATE[name](shuffled)


def test_var_dates_do_not_change_a_clean_fit(quarterly_frame):
    a = sp.var(quarterly_frame, lags=2)
    b = sp.var(quarterly_frame.reset_index(drop=True), lags=2)
    for name in ("a", "b"):
        np.testing.assert_allclose(
            a.coefs[name].to_numpy(dtype=float),
            b.coefs[name].to_numpy(dtype=float),
            rtol=1e-12,
        )


# ----------------------------------------------------------------------
# #12: near-unit-root flag on fixed-order ARIMA fits
# ----------------------------------------------------------------------
def test_random_walk_as_ar1_is_flagged():
    rw = np.cumsum(np.random.default_rng(0).normal(size=200))
    with pytest.warns(AssumptionWarning, match="need another difference") as rec:
        fit = sp.arima(rw, order=(1, 0, 0))
    assert "sp.ndiffs" in rec[0].message.recovery_hint
    phi = float(fit.params["ar.L1"])
    # the root of 1 - phi z is 1 / phi
    assert fit.model_info["root_moduli"]["ar"] == pytest.approx(1.0 / phi, rel=1e-9)
    assert fit.model_info["near_unit_root"] is True
    tests = [v["test"] for v in fit.violations()]
    assert "near_unit_root" in tests


def test_over_differenced_white_noise_is_flagged():
    wn = np.random.default_rng(1).normal(size=300)
    with pytest.warns(AssumptionWarning, match="differenced once too often"):
        fit = sp.arima(wn, order=(0, 1, 1))
    theta = float(fit.params["ma.L1"])
    assert fit.model_info["root_moduli"]["ma"] == pytest.approx(
        1.0 / abs(theta), rel=1e-9
    )


def test_stationary_and_seasonal_fits_are_not_flagged():
    y = _ar1(300, 0.5, 6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        fit = sp.arima(y, order=(1, 0, 0))
    assert fit.model_info["near_unit_root"] is False
    assert "near_unit_root" not in [v["test"] for v in fit.violations()]
    # a seasonal AR of 0.6 at lag 12 has its root at 1/0.6 in units of
    # the season, not at 0.6 ** (-1/12) = 1.04 of the product polynomial
    rng = np.random.default_rng(7)
    e = rng.normal(size=360)
    s = np.zeros(360)
    for t in range(12, 360):
        s[t] = 0.6 * s[t - 12] + e[t]
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        sfit = sp.arima(s, order=(0, 0, 0), seasonal_order=(1, 0, 0, 12))
    big_phi = float(sfit.params["ar.S.L12"])
    assert sfit.model_info["root_moduli"]["seasonal_ar"] == pytest.approx(
        1.0 / big_phi, rel=1e-9
    )
    assert sfit.model_info["near_unit_root"] is False


def test_automatic_search_is_not_flagged():
    rw = np.cumsum(np.random.default_rng(0).normal(size=200))
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        fit = sp.arima(rw, auto=True)
    assert "near_unit_root" not in fit.model_info


# ----------------------------------------------------------------------
# #13: plots against the stored index
# ----------------------------------------------------------------------
def _line(ax, label):
    return next(ln for ln in ax.get_lines() if ln.get_label() == label)


def test_arima_plot_uses_the_dates(monthly):
    fit = sp.arima(monthly, order=(1, 0, 0))
    ax = fit.plot(horizon=6)
    try:
        for label in ("fitted", "observed"):
            x = pd.DatetimeIndex(_line(ax, label).get_xdata(orig=True))
            assert x.equals(pd.DatetimeIndex(monthly.index, freq=None))
        xf = pd.DatetimeIndex(_line(ax, "forecast").get_xdata(orig=True))
        expected = pd.date_range("2010-01-01", periods=6, freq="MS")
        assert list(xf) == list(expected)
        np.testing.assert_allclose(
            _line(ax, "forecast").get_ydata(), fit.forecast(6)["forecast"].to_numpy()
        )
        assert ax.get_xlabel() == "date (monthly)"
    finally:
        plt.close(ax.figure)


def test_arima_plot_of_a_period_series_uses_period_starts(quarterly_frame):
    fit = sp.arima(quarterly_frame["a"], order=(1, 0, 0))
    ax = fit.plot(horizon=4)
    try:
        x = pd.DatetimeIndex(_line(ax, "fitted").get_xdata(orig=True))
        assert list(x) == list(quarterly_frame.index.to_timestamp())
        xf = pd.DatetimeIndex(_line(ax, "forecast").get_xdata(orig=True))
        assert list(xf) == list(
            pd.period_range("2020Q1", periods=4, freq="Q").to_timestamp()
        )
        assert ax.get_xlabel() == "date (quarterly)"
    finally:
        plt.close(ax.figure)


def test_arima_plot_without_an_index_uses_positions():
    fit = sp.arima(_ar1(50, 0.5, 8), order=(1, 0, 0))
    ax = fit.plot(horizon=5)
    try:
        np.testing.assert_array_equal(_line(ax, "fitted").get_xdata(), np.arange(50))
        np.testing.assert_array_equal(
            _line(ax, "forecast").get_xdata(), np.arange(50, 55)
        )
        assert ax.get_xlabel() == "t"
    finally:
        plt.close(ax.figure)


def test_arima_plot_with_an_integer_year_index():
    y = pd.Series(_ar1(40, 0.5, 9), index=pd.Index(range(1981, 2021), name="year"))
    ax = sp.arima(y, order=(1, 0, 0)).plot(horizon=3)
    try:
        np.testing.assert_array_equal(
            _line(ax, "observed").get_xdata(), np.arange(1981, 2021)
        )
        np.testing.assert_array_equal(
            _line(ax, "forecast").get_xdata(), [2021, 2022, 2023]
        )
        assert ax.get_xlabel() == "year"
    finally:
        plt.close(ax.figure)


def test_ets_and_stl_plots_label_the_frequency(monthly):
    ax = sp.ets(monthly, period=12).plot(horizon=4)
    try:
        x = pd.DatetimeIndex(_line(ax, "observed").get_xdata(orig=True))
        assert list(x) == list(monthly.index)
        assert ax.get_xlabel() == "date (monthly)"
    finally:
        plt.close(ax.figure)
    fig = sp.stl(monthly, period=12).plot()
    try:
        x = pd.DatetimeIndex(fig.axes[0].get_lines()[0].get_xdata(orig=True))
        assert list(x) == list(monthly.index)
        assert fig.axes[-1].get_xlabel() == "date (monthly)"
    finally:
        plt.close(fig)


def test_time_axis_falls_back_to_positions_when_forecast_dates_are_unknown():
    # dates with no frequency: the forecast periods have no dates
    idx = pd.DatetimeIndex(["2000-01-01", "2000-01-20", "2000-02-03", "2000-03-01"])
    x, xf, label = time_axis(idx, 4, pd.RangeIndex(4, 6))
    np.testing.assert_array_equal(x, np.arange(4))
    np.testing.assert_array_equal(xf, [4, 5])
    assert label == "t"


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_gap_detection_does_not_depend_on_the_index_resolution(unit):
    """pandas 3 parses dates at microsecond resolution; the spacing of a
    gapped monthly index must be read in days whatever the unit (the
    1.43.0 Linux / pandas 3 spot check caught a nanosecond assumption)."""
    idx = pd.date_range("2020-01-01", periods=24, freq="MS").delete(7).as_unit(unit)
    alias, missing = missing_periods(idx)
    assert alias == "M"
    assert [str(p) for p in missing] == ["2020-08"]
