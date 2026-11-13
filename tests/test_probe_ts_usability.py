"""Usability defects found by the 2026-11 time-series probe.

Each test states what a user reading the result or the error now gets.
The numerical defects of the same round are in
``tests/reference_parity/test_probe_ts_parity.py``.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(11)
    n = 160
    season = 0.3 * np.sin(2 * np.pi * np.arange(n) / 12)
    y = np.cumsum(0.02 + rng.normal(scale=0.05, size=n)) + season + 5.0
    return y


@pytest.fixture(scope="module")
def system():
    rng = np.random.default_rng(5)
    n = 150
    e = rng.normal(size=(n, 2))
    x = np.zeros((n, 2))
    for t in range(1, n):
        x[t, 0] = 0.5 * x[t - 1, 0] + 0.2 * x[t - 1, 1] + e[t, 0]
        x[t, 1] = 0.3 * x[t - 1, 1] + e[t, 1]
    return pd.DataFrame(x, columns=["a", "b"])


# ----------------------------------------------------------------------
# forecast tables and accuracy
# ----------------------------------------------------------------------
def test_forecast_accuracy_scores_the_default_arima_interval(series):
    """``ARIMAResult.forecast(h)`` names its limits ``lower`` / ``upper``;
    the interval scores were silently left out for it."""
    train, test = series[:136], series[136:]
    fit = _quiet(sp.arima, train, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
    plain = fit.forecast(24, alpha=0.2)
    assert list(plain.columns) == ["forecast", "lower", "upper"]
    assert plain.attrs["level"] == pytest.approx(80.0)
    named = fit.forecast(24, level=80)
    a = sp.forecast_accuracy(test, plain, train=train, period=12)
    b = sp.forecast_accuracy(test, named, train=train, period=12)
    assert {"winkler_80", "coverage_80"} <= set(a.columns)
    assert a["winkler_80"].iloc[0] == pytest.approx(b["winkler_80"].iloc[0], rel=1e-12)
    assert a["coverage_80"].iloc[0] == b["coverage_80"].iloc[0]
    both = sp.forecast_accuracy(test, {"plain": plain, "named": named})
    assert both["winkler_80"].notna().all()


def test_arima_to_dict_holds_no_backend_objects(series):
    fit = _quiet(sp.arima, series, order=(1, 1, 0))
    d = fit.to_dict()
    assert not [k for k in d if k.startswith("_")]
    # the same fit serialises to the same text (no memory addresses)
    assert json.dumps(d, sort_keys=True) == json.dumps(fit.to_dict(), sort_keys=True)
    assert d["model_info"]["vce"] == "opg"


def test_vcov_refits_an_arima_with_another_information_matrix(series):
    df = pd.DataFrame({"y": series})
    fit = _quiet(sp.arima, "y", order=(1, 1, 0), data=df)
    again = _quiet(sp.vcov, fit, "oim")
    direct = _quiet(sp.arima, "y", order=(1, 1, 0), data=df, vce="oim")
    np.testing.assert_allclose(again.se, direct.se, rtol=1e-10)
    np.testing.assert_allclose(again.params, fit.params, rtol=1e-12)
    assert again.model_info["vce"] == "oim"


def test_arima_refuses_an_unknown_vce(series):
    with pytest.raises(MethodIncompatibility, match="vce must be"):
        sp.arima(series, order=(1, 0, 0), vce="hc1")


# ----------------------------------------------------------------------
# one name for the log-likelihood
# ----------------------------------------------------------------------
def test_state_space_results_answer_to_log_likelihood(series):
    df = pd.DataFrame({"y": series})
    fit = sp.dlm("y ~ 1", df)
    assert fit.log_likelihood == fit.loglik
    kf = sp.kalman_filter(series, F=[[1.0]], G=[[1.0]], Q=[[0.01]], R=[[0.01]])
    assert kf.log_likelihood == kf.loglik
    growth = np.diff(series)
    ms = sp.mswitch(growth, seed=1, starts=2)
    assert ms.log_likelihood == ms.loglik


# ----------------------------------------------------------------------
# summaries say which model was fitted
# ----------------------------------------------------------------------
def test_johansen_and_vec_summaries_name_the_deterministic_case(system):
    levels = system.cumsum()
    for trend, text in (
        ("rc", "constant restricted to the cointegrating relations"),
        ("c", "unrestricted constant"),
    ):
        assert (
            f"Deterministic terms: {text}"
            in sp.johansen(levels, ["a", "b"], lags=1, trend=trend).summary()
        )
        assert (
            f"Deterministic terms: {text}"
            in sp.vec(levels, ["a", "b"], lags=1, rank=1, trend=trend).summary()
        )


def test_panel_unitroot_summary_states_the_lag_order():
    rng = np.random.default_rng(2)
    rows = []
    for unit in range(12):
        y = np.cumsum(rng.normal(size=40))
        rows += [{"id": unit, "time": t, "y": y[t]} for t in range(40)]
    df = pd.DataFrame(rows)
    out = sp.panel_unitroot(df, "y", id="id", time="time", test="ips", lags=2)
    assert "Lagged differences in each ADF regression: 2" in out.summary()


def test_var_summary_states_the_reference_distribution(system):
    large = sp.var(system, ["a", "b"], lags=1)
    small = sp.var(system, ["a", "b"], lags=1, se_df="r")
    assert "normal reference" in large.summary()
    assert "t(T - m) reference" in small.summary()
    # same coefficients, the small-sample p-value is the larger one
    assert (small.pvalues >= large.pvalues).all()
    np.testing.assert_allclose(small.params, large.params, rtol=1e-12)


# ----------------------------------------------------------------------
# errors that say what to do
# ----------------------------------------------------------------------
def test_var_forecast_options_are_validated(system):
    fit = sp.var(system, ["a", "b"], lags=1)
    with pytest.raises(MethodIncompatibility, match="sigma_df"):
        fit.forecast(2, sigma_df="dfk")
    trend = sp.var(system, ["a", "b"], lags=1, trend="ct")
    with pytest.raises(MethodIncompatibility, match="time trend"):
        trend.forecast(2, param_uncertainty=True)
    # without a constant the regressor vector has no leading one
    bare = sp.var(system, ["a", "b"], lags=1, trend="n")
    known = bare.forecast(1)
    est = bare.forecast(1, param_uncertainty=True)
    assert (est.loc[1, "a_se"] / known.loc[1, "a_se"]) ** 2 == pytest.approx(
        1.0 + 2.0 / bare.n_obs, rel=1e-10
    )


def test_vec_sigma_df_is_validated(system):
    fit = sp.vec(system.cumsum(), ["a", "b"], lags=1, rank=1)
    with pytest.raises(MethodIncompatibility, match="sigma_df"):
        fit.irf(4, sigma_df="stata")
    with pytest.raises(MethodIncompatibility, match="sigma_df"):
        fit.forecast(2, sigma_df="stata")
    # shares do not depend on the divisor
    shares = fit.fevd(4)
    total = shares[shares["period"] == 4].groupby("response")["fevd"].sum()
    np.testing.assert_allclose(total, 1.0, rtol=1e-12)


def test_dm_test_refuses_inputs_it_cannot_use():
    rng = np.random.default_rng(0)
    e1, e2 = rng.normal(size=50), rng.normal(size=50)
    with pytest.raises(MethodIncompatibility, match="same periods"):
        sp.dm_test(e1, e2[:40])
    gap = e1.copy()
    gap[20] = np.nan
    with pytest.raises(MethodIncompatibility, match="inside the sample"):
        sp.dm_test(gap, e2)
    with pytest.raises(NumericalInstability, match="not positive"):
        sp.dm_test(e1, e1)
    with pytest.raises(MethodIncompatibility, match="alternative"):
        sp.dm_test(e1, e2, alternative="smaller")
    with pytest.raises(MethodIncompatibility, match="column name"):
        sp.dm_test("a", "b")
    with pytest.raises(DataInsufficient, match="too few"):
        sp.dm_test(e1[:4], e2[:4], h=3)
    # leading and trailing gaps are the unfilled corners of a rolling
    # evaluation and are trimmed
    lead = np.r_[np.nan, e1, np.nan]
    trimmed = sp.dm_test(lead, np.r_[np.nan, e2, np.nan])
    assert trimmed["n_obs"] == 50
    assert trimmed["statistic"] == pytest.approx(
        sp.dm_test(e1, e2)["statistic"], rel=1e-12
    )


def test_dm_test_is_found_by_the_words_a_user_types():
    for query in ("diebold mariano", "compare forecast accuracy test"):
        names = [
            r["name"] if isinstance(r, dict) else str(r)
            for r in sp.search_functions(query)
        ]
        assert "dm_test" in names[:3]


def test_its_refuses_a_time_column_with_missing_labels():
    df = pd.DataFrame(
        {
            "y": np.arange(30.0),
            "when": [f"p{i:02d}" if i != 7 else None for i in range(30)],
        }
    )
    with pytest.raises(DataInsufficient, match="missing values"):
        sp.its(df, y="y", time="when", intervention=15)
