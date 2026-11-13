"""Reference parity for the defects found by the 2026-11 time-series probe.

Twenty macro / finance analyses were run blind in StatsPAI and repeated in
R and Stata 18 on the same CSV bytes, and every printed number was
compared. The numbers below are the ones that did not agree, or that
StatsPAI could not produce, before the fixes of that round:

* ``ARIMAResult.residuals`` / ``fitted_values`` of a differenced model held
  filter artefacts at the ``d + s D`` diffuse observations (a fitted value
  of zero, the level of the series as the residual). Stata's ``predict,
  resid`` reports them as missing.
* ``sp.arima`` reported outer-product standard errors without saying so
  and offered nothing else; ``vce='oim'`` is what R reports and what
  Stata's ``vce(oim)`` gives.
* ``sp.var`` solved the normal equations: on series in levels the
  coefficients agreed with R's QR fit to eight digits only.
* ``VARResult.summary()`` referred the default (Stata) standard errors to
  ``t(T - m)``; Stata's ``var`` uses the normal distribution.
* ``VARResult.forecast`` ignored ``se_df`` and had no estimation-uncertainty
  term, so its standard errors were neither ``vars::predict``'s nor Stata
  ``fcast compute``'s.
* ``VECResult.irf`` claimed Stata's scaling but Stata's ``irf create`` and
  ``fcast compute`` after ``vec`` use the residual covariance over
  ``T - d``; ``sigma_df='unbiased'`` reproduces them.
* ``sp.structural_break(method='bai-perron')`` and ``method='global'``
  reported two different numbers as the BIC of the same partition.
* ``sp.its`` fitted nanoseconds as the trend when ``time`` was a date
  column and built its Fourier terms on the raw values of ``time``.
* ``sp.dm_test`` and ``VARResult.normality_test`` are new.

Frozen references: ``_fixtures/probe_ts_R.json`` from
``_fixtures/_generate_probe_ts_R.R`` (R 4.5.2; vars 1.6.1, urca 1.3.4,
forecast 9.0.2, strucchange 1.5.4, sandwich 3.1.1), which also writes the
``probe_ts_*.csv`` data, and ``_fixtures/probe_ts_Stata.csv`` from
``_fixtures/_generate_probe_ts_Stata.do`` (Stata 18 MP).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((FX / "probe_ts_R.json").read_text(encoding="utf-8"))
STATA = {
    str(k): float(v)
    for k, v in pd.read_csv(FX / "probe_ts_Stata.csv").itertuples(index=False)
}
VARS = ["e", "prod", "rw", "U"]
DK = ["LRM", "LRY", "IBO", "IDE"]


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(FX / f"probe_ts_{name}.csv")


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def air():
    return np.log(_csv("airpassengers")["y"].to_numpy(dtype=float))


@pytest.fixture(scope="module")
def airline(air):
    return _quiet(sp.arima, air, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))


@pytest.fixture(scope="module")
def canada():
    return _csv("canada")


@pytest.fixture(scope="module")
def var2(canada):
    return sp.var(canada, VARS, lags=2)


@pytest.fixture(scope="module")
def var2_r(canada):
    return sp.var(canada, VARS, lags=2, se_df="r")


# ----------------------------------------------------------------------
# ARIMA: residuals of a differenced model
# ----------------------------------------------------------------------
def test_arima_residuals_are_missing_where_stata_reports_them_missing(airline, air):
    e = np.asarray(airline.residuals, dtype=float)
    n_missing = int(STATA["air.n_missing_resid"])
    assert n_missing == 13  # d + s D = 1 + 12
    assert int(np.isnan(e).sum()) == n_missing
    assert np.isnan(e[:n_missing]).all() and np.isfinite(e[n_missing:]).all()
    assert np.isnan(np.asarray(airline.fitted_values)[:n_missing]).all()
    assert airline.model_info["n_diffuse"] == n_missing
    # the innovations themselves; the two optimisers stop at coefficients
    # 1e-6 apart, which moves a residual of 0.04 by about 1e-6
    for row in (14, 15, 144):
        assert e[row - 1] == pytest.approx(STATA[f"air.resid{row}"], abs=5e-6)
    # fitted + residual is the series wherever both are defined
    np.testing.assert_allclose(
        (np.asarray(airline.fitted_values) + e)[n_missing:],
        air[n_missing:],
        rtol=1e-12,
    )


def test_ljung_box_of_arima_residuals_matches_stata_wntestq(airline):
    """Through 1.43.1 the statistic was 26.618: the 13 artefacts were in."""
    kept = np.asarray(airline.residuals)[13:]
    out = sp.ljungbox(kept, lags=24)
    assert out.loc[24, "statistic"] == pytest.approx(STATA["air.q24"], rel=1e-4)
    assert out.loc[24, "p_value"] == pytest.approx(STATA["air.q24_p"], rel=1e-4)
    # passing the fit drops the missing entries itself
    assert sp.ljungbox(airline, lags=24).loc[24, "statistic"] == pytest.approx(
        out.loc[24, "statistic"], rel=1e-12
    )


def test_arima_first_residual_of_a_differenced_series_is_missing():
    y = _csv("nile")["y"].to_numpy(dtype=float)
    fit = _quiet(sp.arima, y, order=(0, 1, 1))
    e = np.asarray(fit.residuals)
    assert np.isnan(e[0]) and np.isfinite(e[1:]).all()
    # an undifferenced model has an innovation at every date
    lynx = np.log10(_csv("lynx")["y"].to_numpy(dtype=float))
    full = _quiet(sp.arima, lynx, order=(2, 0, 0))
    assert np.isfinite(np.asarray(full.residuals)).all()
    assert full.model_info["n_diffuse"] == 0


# ----------------------------------------------------------------------
# ARIMA: which information matrix
# ----------------------------------------------------------------------
def _arima_cases():
    nile = _csv("nile")["y"].to_numpy(dtype=float)
    lynx = np.log10(_csv("lynx")["y"].to_numpy(dtype=float))
    air_ = np.log(_csv("airpassengers")["y"].to_numpy(dtype=float))
    return {
        "nile": (nile, dict(order=(0, 1, 1)), {"ma.L1": "ma1"}),
        "lynx": (
            lynx,
            dict(order=(2, 0, 0)),
            {"ar.L1": "ar1", "ar.L2": "ar2", "const": "const"},
        ),
        "air": (
            air_,
            dict(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)),
            {"ma.L1": "ma1", "ma.S.L12": "sma1"},
        ),
    }


@pytest.mark.parametrize("case", ["nile", "air"])
def test_arima_default_standard_errors_are_stata_opg(case):
    y, kw, names = _arima_cases()[case]
    fit = _quiet(sp.arima, y, **kw)
    assert fit.model_info["vce"] == "opg"
    for ours, theirs in names.items():
        assert fit.se[ours] == pytest.approx(
            STATA[f"{case}.opg.{theirs}"], rel=2e-4
        )  # OPG by numerical scores in both programs


@pytest.mark.parametrize("case", ["nile", "lynx", "air"])
def test_arima_oim_standard_errors_match_r_and_stata(case):
    """Observed information by finite differences in three programs:
    they agree to three or four digits (R against Stata: 2.7e-4 on the
    Nile coefficient), so the tolerance is 1e-3."""
    y, kw, names = _arima_cases()[case]
    fit = _quiet(sp.arima, y, vce="oim", **kw)
    r_key = {"nile": "nile_011", "lynx": "lynx_200", "air": "air"}[case]
    r_names = {"const": "intercept"}
    for ours, theirs in names.items():
        assert fit.se[ours] == pytest.approx(STATA[f"{case}.oim.{theirs}"], rel=1e-3)
        assert fit.se[ours] == pytest.approx(
            R["arima"][r_key][r_names.get(theirs, theirs)], rel=1e-3
        )


def test_arima_vce_changes_only_the_standard_errors():
    y, kw, _ = _arima_cases()["nile"]
    a = _quiet(sp.arima, y, **kw)
    b = _quiet(sp.arima, y, vce="oim", **kw)
    np.testing.assert_allclose(a.params, b.params, rtol=1e-12)
    assert a.log_likelihood == pytest.approx(b.log_likelihood, rel=1e-12)
    # the two estimates of the information differ by a third here, which
    # is why the choice has to be visible
    assert b.se["ma.L1"] / a.se["ma.L1"] == pytest.approx(1.534, abs=0.01)
    assert "OPG" in a.summary() and "OIM" in b.summary()


def test_arima_oim_is_invariant_to_the_unit_of_measurement():
    y, kw, _ = _arima_cases()["nile"]
    a = _quiet(sp.arima, y, vce="oim", **kw)
    b = _quiet(sp.arima, 1000.0 * y, vce="oim", **kw)
    assert b.se["ma.L1"] == pytest.approx(a.se["ma.L1"], rel=1e-5)
    assert b.se["sigma2"] == pytest.approx(1e6 * a.se["sigma2"], rel=1e-5)


# ----------------------------------------------------------------------
# VAR: coefficients, reference distribution, forecasts, normality
# ----------------------------------------------------------------------
def test_var_coefficients_match_r_qr_to_nine_digits(var2):
    """Levels near 1000 with a constant: the normal equations lost half
    the digits (largest relative gap 8.7e-8 through 1.43.1)."""
    for eq in VARS:
        ref = R["var"]["coef"][eq]
        ours = var2.coefs[eq]["coef"]
        np.testing.assert_allclose(
            [ours[t] for t in ref["term"]], ref["coef"], rtol=1e-9
        )


def test_var_default_p_values_use_the_normal_reference_of_stata(var2):
    p = var2.pvalues
    for eq in VARS:
        for term in var2.coefs[eq].index:
            assert p[f"{eq}:{term}"] == pytest.approx(
                STATA[f"var.p.{eq}.{term}"], rel=1e-6, abs=1e-300
            )
    text = var2.summary()
    assert "P>|z|" in text and "normal reference" in text
    # L1.prod in the e equation: Stata prints 0.004; t(73) gave 0.0049
    assert p["e:L1.prod"] == pytest.approx(0.0037348, abs=1e-6)


def test_var_small_sample_p_values_are_those_of_r_lm(var2_r):
    p = var2_r.pvalues
    for eq in VARS:
        ref = R["var"]["coef"][eq]
        np.testing.assert_allclose(
            [p[f"{eq}:{t}"] for t in ref["term"]], ref["p"], rtol=1e-7
        )
        np.testing.assert_allclose(
            [var2_r.coefs[eq]["se"][t] for t in ref["term"]], ref["se"], rtol=1e-8
        )
    assert "P>|t|" in var2_r.summary()


def test_var_tidy_lists_every_coefficient(var2):
    table = var2.tidy()
    assert len(table) == 4 * 9
    row = table.set_index("term").loc["U:L1.e"]
    assert row["estimate"] == pytest.approx(STATA["var.b.U.L1.e"], rel=1e-9)
    assert row["std_error"] == pytest.approx(STATA["var.se.U.L1.e"], rel=1e-8)


def test_var_forecast_with_estimation_uncertainty_matches_stata_fcast(var2):
    fc = var2.forecast(8, param_uncertainty=True)
    for h in range(1, 9):
        for v in VARS:
            for col in (v, f"{v}_se", f"{v}_lower", f"{v}_upper"):
                assert fc.loc[h, col] == pytest.approx(
                    STATA[f"fcast.{h}.{col}"], rel=1e-8
                )


def test_var_forecast_follows_se_df_as_vars_predict_does(var2, var2_r):
    fc = var2_r.forecast(8)
    for v in VARS:
        ref = R["var"]["predict"][v]
        np.testing.assert_allclose(fc[v], ref["fcst"], rtol=1e-9)
        np.testing.assert_allclose(fc[f"{v}_lower"], ref["lower"], rtol=1e-9)
        np.testing.assert_allclose(fc[f"{v}_upper"], ref["upper"], rtol=1e-9)
    # the divisor is the only difference from the default fit
    T, m = 82, 9
    ml = var2.forecast(8)
    np.testing.assert_allclose(
        fc["e_se"], ml["e_se"] * np.sqrt(T / (T - m)), rtol=1e-12
    )
    np.testing.assert_allclose(
        var2_r.forecast(8, sigma_df="ml")["e_se"], ml["e_se"], rtol=1e-12
    )


def test_var_one_step_estimation_uncertainty_is_one_plus_m_over_t(var2):
    """Omega(1) = tr(Gamma^-1 Gamma) Sigma = m Sigma (Luetkepohl 3.5)."""
    known = var2.forecast(1)
    est = var2.forecast(1, param_uncertainty=True)
    for v in VARS:
        ratio = (est.loc[1, f"{v}_se"] / known.loc[1, f"{v}_se"]) ** 2
        assert ratio == pytest.approx(1.0 + 9.0 / 82.0, rel=1e-10)


def test_var_normality_test_matches_stata_varnorm_and_vars(var2):
    table = var2.normality_test().set_index(["test", "equation"])
    for test, key in (
        ("jarque-bera", "jb"),
        ("skewness", "skewness"),
        ("kurtosis", "kurtosis"),
    ):
        for eq in VARS + ["ALL"]:
            row = table.loc[(test, eq)]
            assert row["chi2"] == pytest.approx(
                STATA[f"varnorm.{key}.{eq}.chi2"], rel=1e-8
            )
            assert row["p"] == pytest.approx(STATA[f"varnorm.{key}.{eq}.p"], rel=1e-8)
            if eq != "ALL" and key != "jb":
                assert row["value"] == pytest.approx(
                    STATA[f"varnorm.{key}.{eq}.value"], rel=1e-8
                )
    ref = R["var"]["normality"]
    for test, key in (
        ("jarque-bera", "jb"),
        ("skewness", "skew"),
        ("kurtosis", "kurt"),
    ):
        assert table.loc[(test, "ALL"), "chi2"] == pytest.approx(
            np.ravel(ref[key])[0], rel=1e-9
        )
        assert table.loc[(test, "ALL"), "p"] == pytest.approx(
            np.ravel(ref[f"{key}_p"])[0], rel=1e-9
        )


# ----------------------------------------------------------------------
# VEC: the covariance Stata's post-estimation commands use
# ----------------------------------------------------------------------
@pytest.mark.parametrize("trend, stata", [("rc", "rconstant"), ("c", "constant")])
def test_vec_irf_and_forecast_with_stata_covariance(trend, stata):
    fit = sp.vec(_csv("denmark"), DK, lags=1, rank=1, trend=trend)
    irf = fit.irf(6, sigma_df="unbiased")["irf"]
    for i in DK:
        for j in DK:
            for h in range(7):
                # Stata stores .irf files in single precision
                assert irf[f"{i} -> {j}"][h] == pytest.approx(
                    STATA[f"vec.{stata}.oirf.{i}.{j}.{h}"], abs=2e-9, rel=2e-7
                )
    fc = fit.forecast(4, sigma_df="unbiased")
    for h in range(1, 5):
        for v in DK:
            assert fc.loc[h, v] == pytest.approx(
                STATA[f"vec.{stata}.fcast.{h}.{v}"], rel=1e-9
            )
            assert fc.loc[h, f"{v}_se"] == pytest.approx(
                STATA[f"vec.{stata}.fcast.{h}.{v}_se"], rel=1e-8
            )
    # against the default: one factor, sqrt(T / (T - d)), d = Stata e(df_m)
    T, d = fit.n_obs, int(STATA[f"vec.{stata}.df_m"])
    ml = fit.irf(6)["irf"]
    np.testing.assert_allclose(
        irf["LRM -> LRY"], ml["LRM -> LRY"] * np.sqrt(T / (T - d)), rtol=1e-10
    )
    np.testing.assert_allclose(
        fc["IBO_se"], fit.forecast(4)["IBO_se"] * np.sqrt(T / (T - d)), rtol=1e-10
    )


# ----------------------------------------------------------------------
# Structural breaks: one BIC per partition
# ----------------------------------------------------------------------
def test_sequential_and_global_break_search_report_the_same_bic():
    nile = _csv("nile")
    seq = sp.structural_break(nile, "y", method="bai-perron")
    glob = sp.structural_break(nile, "y", method="global")
    assert seq.break_dates == glob.break_dates == [R["breaks"]["date_m1"]]
    # strucchange::breakpoints, one break (977.09 through 1.43.1)
    assert seq.bic == pytest.approx(R["breaks"]["bic"][1], rel=1e-10)
    assert glob.bic == pytest.approx(R["breaks"]["bic"][1], rel=1e-10)
    np.testing.assert_allclose(glob.bic_by_breaks, R["breaks"]["bic"], rtol=1e-10)
    assert seq.rss_segments == pytest.approx(R["breaks"]["rss"][1], rel=1e-10)


# ----------------------------------------------------------------------
# Interrupted time series: the clock
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def its_frame(air):
    n = air.shape[0]
    dates = pd.date_range("1949-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "ly": air,
            "idx": np.arange(n),
            "date": dates,
            "period": dates.to_period("M"),
            "label": [d.strftime("%Y-%m") for d in dates],
            "years": 1949.0 + np.arange(n) / 12.0,
        }
    )


@pytest.mark.parametrize("time", [None, "idx", "date", "period", "label"])
def test_its_gives_the_r_segmented_regression_whatever_labels_the_rows(its_frame, time):
    """A date column used to be fitted as nanoseconds: every coefficient
    came back around 1e-28 with a p-value attached."""
    fit = sp.its(
        its_frame, y="ly", time=time, intervention=96, seasonality_period=12, hac_lag=12
    )
    np.testing.assert_allclose(fit.coefficients["coef"], R["its"]["coef"], rtol=1e-8)
    np.testing.assert_allclose(fit.coefficients["se"], R["its"]["se"], rtol=1e-8)
    assert fit.level_change == pytest.approx(R["its"]["coef"][2], rel=1e-8)
    assert fit.pvalue_slope == pytest.approx(R["its"]["p"][3], rel=1e-7)


def test_its_with_time_in_years_keeps_monthly_seasonality(its_frame):
    """Through 1.43.1 the Fourier terms used the raw years (a 12-year
    cycle), the level change came out -0.0182 (SE 0.082), and the slope
    change was per row while the trend was per year."""
    fit = sp.its(
        its_frame,
        y="ly",
        time="years",
        intervention=96,
        seasonality_period=12,
        hac_lag=12,
    )
    coef = fit.coefficients.set_index("variable")
    ref = dict(zip(["const", "time", "level", "slope"], R["its"]["coef"][:4]))
    ref_se = dict(zip(["const", "time", "level", "slope"], R["its"]["se"][:4]))
    assert fit.level_change == pytest.approx(ref["level"], rel=1e-6)
    assert fit.se_level == pytest.approx(ref_se["level"], rel=1e-6)
    # both slopes per year: twelve times the monthly ones
    assert coef.loc["time", "coef"] == pytest.approx(12 * ref["time"], rel=1e-6)
    assert fit.slope_change == pytest.approx(12 * ref["slope"], rel=1e-6)
    assert fit.se_slope == pytest.approx(12 * ref_se["slope"], rel=1e-6)
    assert fit.model_info["period_step"] == pytest.approx(1 / 12, rel=1e-8)


def test_its_names_the_intervention_by_its_time_label(its_frame):
    fit = sp.its(its_frame, y="ly", time="date", intervention=96)
    assert fit.intervention_time == 96
    assert str(fit.intervention_label).startswith("1957-01-01")
    assert "1957-01-01" in fit.summary() and "row 96" in fit.summary()
    json.dumps(fit.to_dict(), default=str)


# ----------------------------------------------------------------------
# Diebold-Mariano
# ----------------------------------------------------------------------
@pytest.mark.parametrize("alternative", ["two.sided", "less", "greater"])
@pytest.mark.parametrize("power", [1, 2])
def test_dm_test_matches_forecast_dm_test_at_one_step(alternative, power):
    err = _csv("dm_errors")
    out = sp.dm_test("e1", "e2", data=err, alternative=alternative, power=power)
    ref = R["dm"][f"h1_{alternative}_p{power}"]
    assert out["statistic"] == pytest.approx(ref["stat"], rel=1e-10)
    assert out["pvalue"] == pytest.approx(ref["p"], rel=1e-10)
    assert out["df"] == 83 and out["n_obs"] == 84


@pytest.mark.parametrize("varestimator", ["acf", "bartlett"])
def test_dm_test_matches_forecast_dm_test_at_four_steps(varestimator):
    err = _csv("dm_errors")
    # the last three origins have no four-step outcome yet
    out = sp.dm_test(err["e1h4"], err["e2h4"], h=4, varestimator=varestimator)
    ref = R["dm"][f"h4_{varestimator}"]
    assert out["n_obs"] == 81
    assert out["statistic"] == pytest.approx(ref["stat"], rel=1e-10)
    assert out["pvalue"] == pytest.approx(ref["p"], rel=1e-10)


def test_dm_test_identities():
    err = _csv("dm_errors")
    a = sp.dm_test(err["e1"], err["e2"])
    b = sp.dm_test(err["e2"], err["e1"])
    # swapping the forecasts flips the sign and nothing else
    assert b["statistic"] == pytest.approx(-a["statistic"], rel=1e-12)
    assert b["pvalue"] == pytest.approx(a["pvalue"], rel=1e-12)
    assert a["mean_loss_diff"] == pytest.approx(a["loss1"] - a["loss2"], rel=1e-12)
    # one-sided p-values of the two directions sum to one
    less = sp.dm_test(err["e1"], err["e2"], alternative="less")["pvalue"]
    greater = sp.dm_test(err["e1"], err["e2"], alternative="greater")["pvalue"]
    assert less + greater == pytest.approx(1.0, abs=1e-12)
    # without the correction: the original statistic, normal reference
    n = 84
    raw = sp.dm_test(err["e1"], err["e2"], harvey=False)
    assert raw["df"] is None
    assert a["statistic"] == pytest.approx(
        raw["statistic"] * np.sqrt((n - 1) / n), rel=1e-12
    )
