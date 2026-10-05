"""Reference parity for the static and dynamic modelling tools added or
corrected while working through a Python econometrics text built on
statsmodels: ARIMA on differenced series, automatic ARIMA selection, the
Chow test at a known date, the OLS-CUSUM test, the BDS test, portmanteau
tests on ARMA residuals, rolling regression, ARDL lag search / forecasts /
long-run effects, the Engle-Granger p-value, and forecasts and impulse
responses of a vector error-correction model.

Two kinds of reference, both on the committed synthetic file
``_fixtures/dynamic_modelling.csv``:

* R 4.5.2 output frozen in ``_fixtures/dynamic_modelling_R.json``
  (``_generate_dynamic_modelling_R.R`` reads the same CSV bytes):
  ``forecast::Arima`` / ``auto.arima``, ``strucchange::sctest``,
  ``stats::Box.test``, ``urca::ca.jo`` with ``vars::vec2var``.
* statsmodels, a core dependency, run inside the test: ``bds``,
  ``breaks_cusumolsresid``, ``acorr_ljungbox``, ``RollingOLS``, ``ARDL`` /
  ``ardl_select_order`` / ``UECM``, ``coint``, ``VECM``.

Tolerances
----------
* 1e-9 relative for closed-form quantities (test statistics, rolling
  coefficients, forecasts given coefficients).
* ARIMA: 1e-6 on the log-likelihood and 2e-4 on coefficients and
  forecasts against R (plus 1e-5 absolute on forecasts, which pass through
  zero). Both sides maximise the same likelihood with their own optimiser;
  R was run with ``reltol = 1e-14``.
* Stata 18 (``arima``, ``estat sbknown``, ``estat sbcusum, ols``,
  ``vargranger``): 1e-7 on the log-likelihoods and the stability
  statistics, 2e-6 on the Granger statistics, which Stata prints to eight
  digits, and on the rescaled series, which ``generate`` stored in single
  precision; 3e-4 on ARIMA coefficients, where Stata's optimiser stops
  1e-4 from R's.
* VECM against ``vars::vec2var``: 1e-7. ``ca.jo`` solves the same
  eigenproblem by a different factorisation.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"

# Stata 18 MP on the same CSV (_fixtures/_generate_dynamic_modelling_Stata.do)
STATA = {
    "arima_110": {"ll": -1183.1112353426, "ar": 0.5585562513, "sigma2": 169664.763769},
    "arima_110_cons": {
        "ll": -1163.9875483676,
        "ar": 0.2314153498,
        "cons": 325.73496873,
    },
    "arima_021": {"ll": -1163.0620474594, "ma": -0.9193581032},
    "arima_110_thousandth": {"ll": -84.7781293185, "ar": 0.5585562053},
    "sbknown_chi2": 40.586100946611,
    "sbknown_p": 1.53758862858e-09,
    "sbcusum_ols": 0.730362054528,
    # vargranger after var c1 c2 c3, lags(1/2): chi2, and F with `small`
    "granger_c1_c2": {"chi2": 40.824533, "F": 20.412267, "df_r": 151},
    "granger_c1_all": {"chi2": 42.569951, "F": 10.642488},
    "granger_c3_c2": {"chi2": 0.98211608, "F": 0.49105804},
}


@pytest.fixture(scope="module")
def R():
    return json.loads((FIX / "dynamic_modelling_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def d():
    return pd.read_csv(FIX / "dynamic_modelling.csv")


def close(ours, ref, rtol=1e-9, atol=0.0):
    assert np.allclose(ours, ref, rtol=rtol, atol=atol), (ours, ref)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ------------------------------------------------------------------ ARIMA
ARIMA_CASES = [
    # key, column, order, trend, {our name: R name}
    ("gdp_110", "gdp", (1, 1, 0), None, {"ar.L1": "ar1"}),
    ("gdp_011", "gdp", (0, 1, 1), None, {"ma.L1": "ma1"}),
    ("gdp_021", "gdp", (0, 2, 1), None, {"ma.L1": "ma1"}),
    ("gdp_110_drift", "gdp", (1, 1, 0), "c", {"ar.L1": "ar1", "drift": "drift"}),
    ("gdp_010_drift", "gdp", (0, 1, 0), "c", {"drift": "drift"}),
    ("x_100", "x", (1, 0, 0), None, {"ar.L1": "ar1", "const": "intercept"}),
]


@pytest.mark.parametrize("method", ["statespace", "innovations_mle"])
@pytest.mark.parametrize("key,col,order,trend,names", ARIMA_CASES)
def test_arima_matches_r_exact_ml(R, d, key, col, order, trend, names, method):
    ref = R["arima"][key]
    fit = _quiet(sp.arima, d[col], order=order, trend=trend, method=method)
    close(fit.log_likelihood, ref["loglik"], rtol=1e-6)
    close(fit.aic, ref["aic"], rtol=1e-6)
    close(fit.bic, ref["bic"], rtol=1e-6)
    close(fit.aicc, ref["aicc"], rtol=1e-6)
    for ours, theirs in names.items():
        close(fit.params[ours], ref["coef"][theirs], rtol=2e-4)
    close(fit.params["sigma2"], ref["sigma2_ml"], rtol=2e-4)
    # forecasts inherit the coefficients' tolerance; some are near zero
    close(fit.forecast(4)["forecast"].to_numpy(), ref["forecast"], rtol=2e-4, atol=1e-5)


@pytest.mark.parametrize("method", ["statespace", "innovations_mle"])
def test_arima_matches_stata(d, method):
    """Stata's ``arima`` differences the series and keeps a constant unless
    told not to; its optimiser stops near 1e-4 of R's maximiser."""
    fit = _quiet(sp.arima, d["gdp"], order=(1, 1, 0), method=method)
    close(fit.log_likelihood, STATA["arima_110"]["ll"], rtol=1e-7)
    close(fit.params["ar.L1"], STATA["arima_110"]["ar"], rtol=3e-4)
    close(fit.params["sigma2"], STATA["arima_110"]["sigma2"], rtol=3e-4)
    drift = _quiet(sp.arima, d["gdp"], order=(1, 1, 0), trend="c", method=method)
    close(drift.log_likelihood, STATA["arima_110_cons"]["ll"], rtol=1e-7)
    close(drift.params["ar.L1"], STATA["arima_110_cons"]["ar"], rtol=3e-4)
    close(drift.params["drift"], STATA["arima_110_cons"]["cons"], rtol=3e-4)
    twice = _quiet(sp.arima, d["gdp"], order=(0, 2, 1), method=method)
    close(twice.log_likelihood, STATA["arima_021"]["ll"], rtol=1e-7)
    close(twice.params["ma.L1"], STATA["arima_021"]["ma"], rtol=3e-4)
    small = _quiet(sp.arima, d["gdp"] / 1000, order=(1, 1, 0), method=method)
    close(small.log_likelihood, STATA["arima_110_thousandth"]["ll"], rtol=2e-6)
    close(small.params["ar.L1"], STATA["arima_110_thousandth"]["ar"], rtol=3e-4)


def test_arima_likelihood_at_a_near_unit_root_and_with_seasonal_terms(R, d):
    """Flat or boundary likelihoods: the maximum is compared, not its
    location (R puts the seasonal MA root at -0.99984)."""
    fit = _quiet(sp.arima, d["gdp"], order=(1, 1, 1))
    close(fit.log_likelihood, R["arima"]["gdp_111"]["loglik"], rtol=1e-5)
    seas = _quiet(sp.arima, d["gdp"], order=(0, 1, 1), seasonal_order=(0, 1, 1, 4))
    close(seas.log_likelihood, R["arima"]["gdp_011_011_4"]["loglik"], rtol=5e-4)
    close(seas.forecast(4)["forecast"].to_numpy(),
          R["arima"]["gdp_011_011_4"]["forecast"], rtol=2e-3)  # fmt: skip


@pytest.mark.parametrize("method", ["statespace", "innovations_mle"])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_arima_does_not_depend_on_the_unit_of_measurement(R, d, scale, method):
    """With a normal prior of variance 1e6 on the level, as statsmodels
    starts it, the AR coefficient of this series is 0.51 in the units
    given, 0.56 in thousands and 0.14 in thousandths."""
    fit = _quiet(sp.arima, d["gdp"] * scale, order=(1, 1, 0), method=method)
    close(fit.params["ar.L1"], R["arima"]["gdp_110"]["coef"]["ar1"], rtol=2e-4)
    n_used = len(d) - 1
    close(
        fit.log_likelihood + n_used * np.log(scale),
        R["arima"]["gdp_110"]["loglik"],
        rtol=1e-6,
    )
    drift = _quiet(
        sp.arima, d["gdp"] * scale, order=(1, 1, 0), trend="c", method=method
    )
    ref = R["arima"]["gdp_110_drift"]["coef"]
    close(drift.params["ar.L1"], ref["ar1"], rtol=5e-4)
    close(drift.params["drift"] / scale, ref["drift"], rtol=5e-4)


def test_arima_r_itself_is_scale_free(R):
    """The reference has the property the test above demands."""
    a = R["arima"]["gdp_110"]["coef"]["ar1"]
    close(R["arima"]["gdp_110_thousandth"]["coef"]["ar1"], a, rtol=1e-6)
    close(R["arima"]["gdp_110_thousandfold"]["coef"]["ar1"], a, rtol=1e-6)


@pytest.mark.parametrize("col", ["gdp", "x", "y", "c2", "ret", "dgdp"])
def test_auto_arima_selects_what_auto_arima_selects(R, d, col):
    """Order, differencing, and whether a mean or a drift is kept."""
    ref = R["auto"][col]
    series = d["gdp"].diff().dropna() if col == "dgdp" else d[col]
    fit = _quiet(sp.arima, series, auto=True, max_p=3, max_q=3)
    assert list(fit.order) == ref["order"]
    terms = ref["terms"]
    terms = [terms] if isinstance(terms, str) else list(terms or [])
    constant = {"intercept": "const", "drift": "drift"}
    ours = {n for n in fit.params.index if n in ("const", "drift")}
    assert ours == {constant[t] for t in terms if t in constant}
    close(fit.aicc, ref["aicc"], rtol=1e-6)


def test_auto_arima_drops_fits_on_the_unit_circle(d):
    """Differenced once too often, ``ret`` with a drift is fitted with its
    MA root at -1 and a lower AICc than any admissible model."""
    boundary = _quiet(sp.arima, d["ret"], order=(0, 1, 1), trend="c")
    chosen = _quiet(sp.arima, d["ret"], auto=True, max_p=3, max_q=3)
    assert boundary.params["ma.L1"] < -0.999
    assert boundary.aicc < chosen.aicc
    assert chosen.params["ma.L1"] > -0.99 and "drift" not in chosen.params.index


def test_auto_arima_keeps_an_explicit_trend(d):
    fit = _quiet(sp.arima, d["gdp"], auto=True, max_p=2, max_q=2, trend="n")
    assert "drift" not in fit.params.index


# ------------------------------------------------------- stability tests
def test_chow_at_a_known_date_matches_strucchange(R, d):
    for point, key in ((100, "chow100"), (60, "chow60")):
        res = sp.chow_test(d, "y", ["x"], break_point=point)
        close(res["statistic"], R["stability"][key]["stat"])
        close(res["pvalue"], R["stability"][key]["p"], rtol=1e-8)
        assert (res["df1"], res["df2"]) == (2, len(d) - 4)


def test_chow_wald_form_matches_stata_sbknown(d):
    """``estat sbknown, break(101)``: the row dated 101 opens the regime."""
    res = sp.chow_test(d, "y", ["x"], break_point=101, time="t")
    close(res["chi2"], STATA["sbknown_chi2"], rtol=1e-7)
    close(res["chi2_pvalue"], STATA["sbknown_p"], rtol=1e-6)
    # Stata reads the CSV into floats where they fit; 4e-8 apart
    ols = sp.cusum_test(d, "y", ["x"], method="ols")
    close(ols["statistic"], STATA["sbcusum_ols"], rtol=1e-7)


def test_granger_f_is_stata_small_not_the_classical_f(d):
    """``F_stat`` is the Wald statistic over its degrees of freedom, as
    Stata's ``vargranger`` reports after ``var, small``. The classical F of
    statsmodels divides the residual sum of squares by ``T - m`` and is
    smaller by ``(T - m) / T``; ``se_df='r'`` gives it."""
    from statsmodels.tsa.api import VAR

    names = ["c1", "c2", "c3"]
    fit = sp.var(d, names, lags=2)
    for caused, causing, key in (
        ("c1", "c2", "granger_c1_c2"),
        ("c1", ["c2", "c3"], "granger_c1_all"),
        ("c3", "c2", "granger_c3_c2"),
    ):
        res = sp.granger_causality(fit, caused=caused, causing=causing)
        close(res["chi2"], STATA[key]["chi2"], rtol=2e-6)
        close(res["F_stat"], STATA[key]["F"], rtol=2e-6)
    assert res["df2"] == STATA["granger_c1_c2"]["df_r"]

    ref = VAR(d[names]).fit(2).test_causality("c1", ["c2"], kind="f")
    classical = sp.granger_causality(
        sp.var(d, names, lags=2, se_df="r"), caused="c1", causing="c2"
    )
    close(classical["F_stat"], ref.test_statistic, rtol=1e-9)
    T, m = fit.n_obs, 7
    close(classical["F_stat"], STATA["granger_c1_c2"]["F"] * (T - m) / T, rtol=2e-6)


def test_chow_is_the_split_sample_formula(d):
    def rss(frame):
        X = np.column_stack([np.ones(len(frame)), frame["x"]])
        e = frame["y"] - X @ np.linalg.lstsq(X, frame["y"], rcond=None)[0]
        return float(e @ e)

    r, r1, r2 = rss(d), rss(d.iloc[:100]), rss(d.iloc[100:])
    f = ((r - r1 - r2) / 2) / ((r1 + r2) / (len(d) - 4))
    res = sp.chow_test(d, "y", ["x"], break_point=100)
    close(res["statistic"], f)
    close(res["rss_unrestricted"], r1 + r2)
    close(res["rss_restricted"], r)
    close(res["chi2"], 2 * f)
    assert list(res["regimes"].loc["n_obs"]) == [100, 60]


def test_chow_break_point_by_label_and_by_time_column(d):
    by_count = sp.chow_test(d, "y", ["x"], break_point=100)
    by_time = sp.chow_test(d, "y", ["x"], break_point=101, time="t")
    dated = d.set_index(pd.date_range("2000-01-01", periods=len(d), freq="QS"))
    by_label = sp.chow_test(dated, "y", ["x"], break_point="2025-01-01")
    close(by_time["statistic"], by_count["statistic"])
    close(by_label["statistic"], by_count["statistic"])
    assert by_time["break_point"] == by_label["break_point"] == 100


def test_chow_subset_and_robust_agree_with_a_dummy_regression(d):
    frame = d.assign(late=(d["t"] > 100).astype(float))
    frame["late_x"] = frame["late"] * frame["x"]
    fit = sp.regress("y ~ x + late_x", data=frame, robust="hc1")
    ref = sp.test(fit, "late_x = 0")
    res = sp.chow_test(d, "y", ["x"], break_point=100, break_vars=["x"], vce="hc1")
    close(res["statistic"], ref["statistic"])
    assert res["df1"] == 1


def test_chow_several_breaks_jointly(d):
    res = sp.chow_test(d, "y", ["x"], break_point=[60, 100])
    assert res["df1"] == 4 and res["df2"] == len(d) - 6
    assert res["regimes"].shape == (3, 3)


def test_chow_refuses_what_it_cannot_test(d):
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.chow_test(d, "y", ["x"], break_point=0)
    with pytest.raises(sp.exceptions.DataInsufficient):
        sp.chow_test(d, "y", ["x"], break_point=1)
    holed = d.copy()
    holed.loc[5, "x"] = np.nan
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="missing"):
        sp.chow_test(holed, "y", ["x"], break_point=100)


def test_ols_cusum_matches_strucchange_and_statsmodels(R, d):
    from statsmodels.stats.diagnostic import breaks_cusumolsresid

    res = sp.cusum_test(d, "y", ["x"], method="ols")
    close(res["statistic"], R["stability"]["ols_cusum"]["stat"])
    close(res["p_value"], R["stability"]["ols_cusum"]["p"], rtol=1e-8)
    assert not res["reject"]
    X = np.column_stack([np.ones(len(d)), d["x"]])
    e = d["y"] - X @ np.linalg.lstsq(X, d["y"], rcond=None)[0]
    sm = breaks_cusumolsresid(np.asarray(e), ddof=2)
    close(res["statistic"], sm[0])
    close(res["p_value"], sm[1])

    trend = sp.cusum_test(d, "gdp", ["t"], method="ols")
    close(trend["statistic"], R["stability"]["ols_cusum_trend"]["stat"])
    close(trend["p_value"], R["stability"]["ols_cusum_trend"]["p"], rtol=1e-6)
    assert trend["reject"]
    close(trend["critical_value"], 1.3580986393225507)


def test_cusum_default_is_still_the_recursive_test(d):
    res = sp.cusum_test(d, "y", ["x"])
    assert res["method"] == "recursive"
    assert len(res["cusum"]) == len(d) - 2
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.cusum_test(d, "y", ["x"], method="mosum")


# -------------------------------------------------------------------- BDS
@pytest.mark.parametrize("kwargs", [{}, {"distance": 1.0}, {"epsilon": 0.6}])
def test_bds_matches_statsmodels(d, kwargs):
    from statsmodels.tsa.stattools import bds

    stat, pval = bds(d["ret"].to_numpy(), max_dim=5, **kwargs)
    ours = sp.bds(d, "ret", max_dim=5, **kwargs)
    close(ours["statistic"].to_numpy(), stat)
    close(ours["pvalue"].to_numpy(), pval)
    assert list(ours.index) == [2, 3, 4, 5]


def test_bds_tells_dependence_from_independence(d):
    # uncorrelated but dependent: the Q test sees nothing, BDS does
    rng = np.random.default_rng(7)
    e = rng.normal(size=800)
    arch = np.zeros(800)
    for t in range(1, 800):
        arch[t] = e[t] * np.sqrt(0.2 + 0.7 * arch[t - 1] ** 2)
    assert sp.corrgram(arch, lags=8)["Prob>Q"].iloc[-1] > 0.05
    assert sp.bds(arch, max_dim=3)["pvalue"].max() < 1e-4
    assert sp.bds(e, max_dim=3)["pvalue"].min() > 0.05


def test_bds_accepts_a_fitted_regression(d):
    fit = sp.regress("y ~ x", data=d)
    close(
        sp.bds(fit)["statistic"].to_numpy(),
        sp.bds(np.asarray(fit.data_info["residuals"]))["statistic"].to_numpy(),
    )
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.bds(d)  # which column?
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.bds(np.ones(50))


# ------------------------------------------------------------ portmanteau
def test_portmanteau_on_arma_residuals_matches_box_test(R, d):
    ref = R["portmanteau"]
    table = sp.corrgram(np.asarray(ref["resid"]), lags=10, model_df=2, boxpierce=True)
    close(table.loc[10, "Q"], ref["lb"]["stat"])
    close(table.loc[10, "Prob>Q"], ref["lb"]["p"])
    close(table.loc[10, "BP"], ref["bp"]["stat"])
    close(table.loc[10, "Prob>BP"], ref["bp"]["p"])
    # no reference distribution until the lag exceeds the fitted orders
    assert table["Prob>Q"].iloc[:2].isna().all()
    assert table["Prob>Q"].iloc[2:].notna().all()

    raw = sp.corrgram(d, "ret", lags=8, boxpierce=True)
    close(raw.loc[8, "Q"], ref["lb_raw"]["stat"])
    close(raw.loc[8, "Prob>Q"], ref["lb_raw"]["p"])
    close(raw.loc[8, "BP"], ref["bp_raw"]["stat"])
    close(raw.loc[8, "Prob>BP"], ref["bp_raw"]["p"])


def test_portmanteau_matches_statsmodels_at_every_lag(d):
    from statsmodels.stats.diagnostic import acorr_ljungbox

    sm = acorr_ljungbox(d["x"], lags=12, boxpierce=True, model_df=1, return_df=True)
    ours = sp.corrgram(d, "x", lags=12, model_df=1, boxpierce=True)
    close(ours["Q"].to_numpy(), sm["lb_stat"].to_numpy())
    close(ours["BP"].to_numpy(), sm["bp_stat"].to_numpy())
    close(ours["Prob>Q"].to_numpy()[1:], sm["lb_pvalue"].to_numpy()[1:])
    close(ours["Prob>BP"].to_numpy()[1:], sm["bp_pvalue"].to_numpy()[1:])
    assert list(sp.corrgram(d, "x", lags=3).columns) == ["AC", "PAC", "Q", "Prob>Q"]
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.corrgram(d, "x", model_df=-1)


# ------------------------------------------------------ rolling regression
@pytest.mark.parametrize("vce,cov_type", [("nonrobust", "nonrobust"), ("hc0", "HC0")])
def test_rolling_matches_rolling_ols(d, vce, cov_type):
    import statsmodels.api as sm
    from statsmodels.regression.rolling import RollingOLS

    ref = RollingOLS(d["stk"], sm.add_constant(d[["mkt", "x"]]), window=36).fit(
        cov_type=cov_type
    )
    ours = sp.rolling("stk ~ mkt + x", d, window=36, vce=vce)
    names = ours.attrs["params"]
    assert names == ["Intercept", "mkt", "x"] and len(ours) == len(d) - 35
    close(ours[names].to_numpy(), ref.params.dropna().to_numpy())
    close(ours[[f"se_{n}" for n in names]].to_numpy(), ref.bse.dropna().to_numpy())
    close(ours["r2"].to_numpy(), ref.rsquared.dropna().to_numpy())
    assert (ours["nobs"] == 36).all()
    assert ours["mkt"].iloc[-1] > ours["mkt"].iloc[0]  # the beta drifts up


@pytest.mark.parametrize("vce", ["hc1", "hc2", "hc3"])
def test_rolling_windows_are_ordinary_regressions(d, vce):
    ours = sp.rolling("stk ~ mkt", d, window=40, step=7, vce=vce)
    j = 5
    lo = j * 7
    fit = sp.regress("stk ~ mkt", data=d.iloc[lo : lo + 40], robust=vce)
    close(ours.iloc[j][["Intercept", "mkt"]].to_numpy(float), fit.params.to_numpy())
    close(
        ours.iloc[j][["se_Intercept", "se_mkt"]].to_numpy(float),
        fit.std_errors.to_numpy(),
    )
    assert (ours["start"].iloc[j], ours["end"].iloc[j]) == (lo, lo + 39)


def test_recursive_least_squares(d):
    ours = sp.rolling("stk ~ mkt", d, window=30, recursive=True)
    assert list(ours["nobs"]) == list(range(30, len(d) + 1))
    assert (ours["start"] == 0).all()
    full = sp.regress("stk ~ mkt", data=d)
    close(ours.iloc[-1][["Intercept", "mkt"]].to_numpy(float), full.params.to_numpy())
    close(ours["se_mkt"].iloc[-1], full.std_errors["mkt"])


def test_rolling_applies_formula_lags_to_the_whole_series(d):
    ours = sp.rolling("y ~ x + x.shift(1)", d, window=24)
    assert ours["start"].iloc[0] == 1 and len(ours) == len(d) - 1 - 23


def test_rolling_reports_windows_it_cannot_fit(d):
    frame = d.assign(spike=(d["t"] == 80).astype(float))
    with pytest.warns(UserWarning, match="collinear in"):
        ours = sp.rolling("y ~ x + spike", frame, window=20)
    inside = (ours["start"] <= 79) & (ours["end"] >= 79)
    assert ours.loc[inside, "spike"].notna().all()
    assert ours.loc[~inside, ["x", "spike", "se_x"]].isna().all().all()
    with pytest.raises(sp.exceptions.DataInsufficient):
        sp.rolling("y ~ x", d, window=2)
    with pytest.raises(sp.exceptions.DataInsufficient):
        sp.rolling("y ~ x", d, window=len(d) + 1)
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.rolling("y ~ x", d, window=30, vce="cluster")


# ------------------------------------------------------------------- ARDL
def _growth(d):
    g = pd.DataFrame(
        {"dg": d["gdp"].diff(), "x": d["x"], "r": d["ret"]}, index=d.index
    ).dropna()
    return g.reset_index(drop=True)


def test_ardl_dynamic_forecast_matches_statsmodels(d):
    from statsmodels.tsa.ardl import ARDL

    g = _growth(d)
    future = pd.DataFrame(
        {"x": [0.3, -0.2, 0.1, 0.4, 0.0], "r": [0.1, 0.0, -0.3, 0.2, 0.1]}
    )
    future.index = range(len(g), len(g) + 5)

    sm = ARDL(g["dg"], 2, g[["x", "r"]], {"x": 1, "r": 2}).fit()
    ours = sp.ardl(
        g, "dg", ["x", "r"], lags=2, x_lags={"x": 1, "r": 2},
        contemporaneous=True, vce="nonrobust",
    )  # fmt: skip
    ref = sm.predict(start=len(g), end=len(g) + 4, exog_oos=future).to_numpy()
    fc = ours.forecast(steps=5, exog=future)
    close(fc["forecast"].to_numpy(), ref)
    # statsmodels scales the same psi weights by sqrt(RSS / n)
    se = sm.get_prediction(
        start=len(g), end=len(g) + 4, exog_oos=future
    ).summary_frame()
    close(
        fc["rmsfe"].to_numpy() * np.sqrt((ours.nobs - len(ours.params)) / ours.nobs),
        se["mean_se"].to_numpy(),
    )

    lagged = ARDL(g["dg"], 2, g[["x", "r"]], {"x": [1, 2], "r": [1]}, causal=True).fit()
    ours = sp.ardl(
        g, "dg", ["x", "r"], lags=2, x_lags={"x": 2, "r": 1}, vce="nonrobust"
    )
    ref = lagged.predict(start=len(g), end=len(g) + 4, exog_oos=future).to_numpy()
    close(ours.forecast(steps=5, exog=future)["forecast"].to_numpy(), ref)
    close(ours.forecast()["forecast"].to_numpy(), ref[:1])  # one step needs no x
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="after the sample"):
        ours.forecast(steps=3)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="at least 2"):
        ours.forecast(steps=3, exog=future.iloc[:1])


def test_ar_forecast_matches_autoreg(d):
    from statsmodels.tsa.ar_model import AutoReg

    g = _growth(d)
    for trend in ("c", "ct"):
        sm = AutoReg(g["dg"].to_numpy(), lags=3, trend=trend).fit()
        ours = sp.ardl(g, "dg", lags=3, trend=trend, vce="nonrobust")
        fc = ours.forecast(steps=8)
        close(fc["forecast"].to_numpy(), sm.predict(start=len(g), end=len(g) + 7))
        assert (np.diff(fc["rmsfe"]) >= 0).all()
        close(fc["rmsfe"].iloc[0], ours.ser)


@pytest.mark.parametrize("criterion", ["aic", "bic"])
@pytest.mark.parametrize("contemporaneous", [True, False])
def test_ardl_lag_search_matches_ardl_select_order(d, criterion, contemporaneous):
    from statsmodels.tsa.ardl import ardl_select_order

    g = _growth(d)
    sel = ardl_select_order(
        g["dg"], 3, g[["x", "r"]], 3, ic=criterion, causal=not contemporaneous
    )
    ours = sp.ardl(
        g, "dg", ["x", "r"], lags=criterion, x_lags=criterion, max_lags=3,
        contemporaneous=contemporaneous, vce="nonrobust",
    )  # fmt: skip
    assert ours.lags == len(sel.model.ar_lags or [])
    floor = -1 if contemporaneous else 0
    for var in ("x", "r"):
        chosen = sel.model.dl_lags.get(var)
        assert ours.x_lags[var] == (max(chosen) if chosen else floor)
    grid = 4 * (3 - floor + 1) ** 2
    assert len(ours.ic_table) == grid
    assert ours.ic_table[criterion].min() == pytest.approx(getattr(ours, criterion))


def test_ardl_lag_search_for_a_given_ar_order(d):
    g = _growth(d)
    ours = sp.ardl(
        g, "dg", ["x"], lags=1, x_lags="bic", max_lags=2, contemporaneous=True
    )
    assert ours.lags == 1 and set(ours.ic_table.index.get_level_values("lags")) == {1}
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="different"):
        sp.ardl(g, "dg", ["x"], lags="aic", x_lags="bic")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="no x"):
        sp.ardl(g, "dg", lags=1, x_lags="bic")


def test_ardl_long_run_matches_uecm(d):
    from statsmodels.tsa.ardl import ARDL, UECM

    g = _growth(d)
    uecm = UECM.from_ardl(ARDL(g["dg"], 2, g[["x", "r"]], {"x": 1, "r": 2})).fit()
    ours = sp.ardl(
        g, "dg", ["x", "r"], lags=2, x_lags={"x": 1, "r": 2},
        contemporaneous=True, vce="nonrobust",
    )  # fmt: skip
    lr = ours.long_run()
    # statsmodels normalises the relation as dg - theta'x = 0
    for name, theirs in (("Intercept", "const"), ("x", "x"), ("r", "r")):
        close(lr.loc[name, "coef"], -uecm.ci_params[theirs])
        close(lr.loc[name, "std err"], uecm.ci_bse[theirs], rtol=1e-8)
    phi = ours.params[["dg_L1", "dg_L2"]].sum()
    close(lr.attrs["ar_sum"], phi)
    close(lr.loc["x", "coef"], ours.params[["x", "x_L1"]].sum() / (1 - phi))


def test_ardl_long_run_of_a_geometric_lag_and_of_a_unit_root(d):
    g = _growth(d)
    koyck = sp.ardl(g, "dg", ["x"], lags=1, x_lags=0, contemporaneous=True)
    lr = koyck.long_run()
    close(lr.loc["x", "coef"], koyck.params["x"] / (1 - koyck.params["dg_L1"]))
    growing = pd.DataFrame({"z": 1.03 ** np.arange(80.0) + 0.01 * d["x"].iloc[:80]})
    explosive = sp.ardl(growing, "z", lags=1)
    assert explosive.params["z_L1"] > 1
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="no long-run"):
        explosive.long_run()


# ---------------------------------------------------------------- unit root
def test_adf_pvalue_of_an_explosive_series_is_one():
    """Stata 18, ``dfuller y, trend`` with y = 1.03^t + 0.5 sin(t) for
    t = 1..120: Z(t) = 2.8538185, MacKinnon p-value 1.0000. The fitted
    polynomial, read outside its range, says 0.41."""
    from statsmodels.tsa.stattools import adfuller

    t = np.arange(1.0, 121.0)
    y = 1.03**t + 0.5 * np.sin(t)
    for trend in ("ct", "c"):
        res = sp.unitroot(y, trend=trend, lags=0)
        ref = adfuller(y, maxlag=0, regression=trend, autolag=None)
        close(res.statistic, ref[0])
        assert res.pvalue == ref[1] == 1.0
        assert not res.reject
    # Stata generated y in single precision
    close(sp.unitroot(y, trend="ct", lags=0).statistic, 2.8538185, rtol=1e-6)
    close(sp.unitroot(y, trend="c", lags=0).statistic, 8.3965679, rtol=1e-6)


# ------------------------------------------------------------ Engle-Granger
@pytest.mark.parametrize("trend", ["c", "ct"])
def test_engle_granger_pvalue_matches_statsmodels(d, trend):
    from statsmodels.tsa.stattools import coint

    stat, pvalue, _ = coint(d["c1"], d["c2"], trend=trend, maxlag=1, autolag=None)
    res = sp.engle_granger(d, ["c1", "c2"], lags=1, trend=trend)
    close(res.test_stats, stat)
    close(res.pvalue, pvalue)
    stat3, pvalue3, _ = coint(d["c1"], d[["c2", "c3"]], maxlag=2, autolag=None)
    res3 = sp.engle_granger(d, ["c1", "c2", "c3"], lags=2)
    close(res3.test_stats, stat3)
    close(res3.pvalue, pvalue3)
    assert res.pvalue < 0.01 < sp.engle_granger(d, ["c1", "c3"], lags=1).pvalue
    assert "pvalue=" in repr(res) and "MacKinnon p-value" in res.summary()


def test_johansen_summary_names_the_statistic_it_prints(d):
    trace = sp.johansen(d, ["c1", "c2", "c3"], lags=2).summary()
    maxeig = sp.johansen(d, ["c1", "c2", "c3"], lags=2, test="maxeig").summary()
    assert "Trace stat" in trace and "r <= 1" in trace
    assert "Max-eig stat" in maxeig and "r = 1" in maxeig and "Trace" not in maxeig


# ------------------------------------------------------------------- VECM
VARS = ["c1", "c2", "c3"]


@pytest.mark.parametrize("trend", ["c", "rc"])
def test_vec_forecast_and_irf_match_vars_vec2var(R, d, trend):
    ref = R["vec"][trend]
    fit = sp.vec(d, VARS, lags=2, rank=1, trend=trend)
    fc = fit.forecast(6)
    for name in VARS:
        close(fc[name].to_numpy(), ref["forecast"][name], rtol=1e-7)
        close(fc[f"{name}_lower"].to_numpy(), ref["forecast_lower"][name], rtol=1e-6)
    for orthogonal, key in ((True, "irf_orth"), (False, "irf_unit")):
        paths = fit.irf(8, orthogonal=orthogonal)["irf"]
        for shock in VARS:
            theirs = np.asarray(ref[key][shock])  # periods x responses
            for i, response in enumerate(VARS):
                close(
                    paths[f"{shock} -> {response}"], theirs[:, i], rtol=1e-6, atol=1e-9
                )


@pytest.mark.parametrize(
    "trend,deterministic",
    [("n", "n"), ("rc", "ci"), ("c", "co"), ("rt", "coli"), ("ct", "colo")],
)
@pytest.mark.parametrize("rank", [1, 2])
def test_vec_forecast_and_irf_match_statsmodels(d, trend, deterministic, rank):
    from statsmodels.tsa.vector_ar.vecm import VECM

    sm = _quiet(
        VECM(d[VARS], k_ar_diff=2, coint_rank=rank, deterministic=deterministic).fit
    )
    fit = sp.vec(d, VARS, lags=2, rank=rank, trend=trend)
    close(fit.forecast(8)[VARS].to_numpy(), sm.predict(steps=8), rtol=1e-8)
    point, lower, upper = sm.predict(steps=4, alpha=0.1)
    ours = fit.forecast(4, alpha=0.1)
    close(ours[[f"{v}_lower" for v in VARS]].to_numpy(), lower, rtol=1e-7)
    close(ours[[f"{v}_upper" for v in VARS]].to_numpy(), upper, rtol=1e-7)
    ref = sm.irf(10)
    for orthogonal, theirs in ((False, ref.irfs), (True, ref.orth_irfs)):
        paths = fit.irf(10, orthogonal=orthogonal)["irf"]
        for j, shock in enumerate(VARS):
            for i, response in enumerate(VARS):
                close(
                    paths[f"{shock} -> {response}"],
                    theirs[:, i, j],
                    rtol=1e-7,
                    atol=1e-10,
                )


def test_vec_irf_is_permanent_and_fevd_adds_up(d):
    fit = sp.vec(d, VARS, lags=2, rank=1)
    far = fit.irf(200, orthogonal=False)["irf"]
    # one cointegrating relation among three series leaves two unit roots:
    # responses converge, and not to zero
    assert abs(far["c2 -> c2"][-1] - far["c2 -> c2"][-2]) < 1e-8
    assert abs(far["c2 -> c2"][-1]) > 0.1
    one = fit.irf(5, impulse="c2", response="c1")["irf"]
    assert list(one) == ["c2 -> c1"] and len(one["c2 -> c1"]) == 6
    table = fit.fevd(12)
    sums = table.groupby(["response", "period"])["fevd"].sum()
    close(sums[sums.index.get_level_values("period") > 0].to_numpy(), 1.0)
    assert (table.loc[table["period"] == 0, "fevd"] == 0).all()
    first = table[(table["period"] == 1) & (table["response"] == "c1")]
    assert first.set_index("shock")["fevd"]["c1"] == pytest.approx(1.0)  # ordered first
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        fit.irf(5, impulse="gdp")
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        fit.forecast(0)


# ------------------------------------------------- arguments people bring
def test_one_series_can_be_passed_as_it_is(d):
    from scipy import stats

    fit = sp.regress("y ~ x", data=d)
    e = fit.residuals()
    w, p = stats.shapiro(np.asarray(e))
    out = sp.swilk(e)
    close(out["W"].iloc[0], w)
    close(out["pvalue"].iloc[0], p, rtol=1e-6)
    assert sp.sktest(np.asarray(e)).shape[0] == 1
    t = stats.ttest_1samp(d["x"], 0.2)
    res = sp.ttest(d["x"], mu=0.2)
    close(res.statistic, t.statistic)
    close(res.pvalue, t.pvalue)
    close(sp.ttest(d["x"].to_numpy(), mu=0.2).statistic, t.statistic)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="bare series"):
        sp.ttest(d["x"], by="t")


def test_hints_for_statsmodels_habits(d):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="sqreg"):
        sp.qreg(d, "y ~ x", quantile=[0.25, 0.75])
    fit = sp.regress("y ~ x", data=d)
    with pytest.raises(AttributeError, match="influence_measures"):
        fit.get_influence()
