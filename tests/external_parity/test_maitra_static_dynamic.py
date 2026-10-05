"""Maitra, *A Practical Guide to Static and Dynamic Econometric Modelling*
(Springer; notebooks at github.com/saritmaitra/PracticalGuideEconometrics),
a Python text whose examples are written for statsmodels and scipy on
series downloaded from FRED and Yahoo Finance.

Each test reruns one of the book's computations with StatsPAI and compares
it with the library the book used, run here on the same data. Chapters 1
and 2 use data the book simulates from a fixed seed, so those tests always
run. Chapters 5 to 7 use downloaded series, which are not redistributed and
are revised by their publishers: set ``STATSPAI_MAITRA_DIR`` to a folder
holding ``data/<SERIES>.csv`` as FRED's ``fredgraph.csv`` serves them and
``data/yahoo_monthly_adjclose.csv`` (monthly adjusted closes of AAPL, CL=F
and ^GSPC from 2010 through 2023). Skipped otherwise. Because the vintages
differ from the book's, the numbers are compared with statsmodels on the
data in hand and not with the book's printed output.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

ROOT = os.environ.get("STATSPAI_MAITRA_DIR")
needs_data = pytest.mark.skipif(
    not ROOT or not (Path(ROOT) / "data" / "GDP.csv").is_file(),
    reason="set STATSPAI_MAITRA_DIR to the folder holding data/GDP.csv etc.",
)


def close(ours, ref, rtol=1e-9, atol=0.0):
    assert np.allclose(ours, ref, rtol=rtol, atol=atol), (ours, ref)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def fred(name, start, end):
    table = pd.read_csv(Path(ROOT) / "data" / f"{name}.csv")
    table.columns = ["date", name]
    table["date"] = pd.to_datetime(table["date"])
    table[name] = pd.to_numeric(table[name], errors="coerce")
    return table.set_index("date").loc[start:end]


# ------------------------------------------------ chapters 1-2 (simulated)
@pytest.fixture(scope="module")
def sim():
    """Table 8: the book's data-generating code, seed included."""
    np.random.seed(0)
    X = np.random.rand(100, 1) * 10
    Y = 2.5 * X + np.random.randn(100, 1) * 2
    return pd.DataFrame({"X": X.flatten(), "Y": Y.flatten()})


@pytest.fixture(scope="module")
def sim_ols(sim):
    import statsmodels.api as sm

    return sm.OLS(sim["Y"], sm.add_constant(sim["X"])).fit()


def test_ch1_ols_reproduces_the_printed_table(sim):
    fit = sp.regress("Y ~ X", data=sim)
    # Table 9 as printed: const 0.4443 (0.387), x1 2.4874 (0.070), R2 0.928
    close(fit.params.to_numpy(), [0.4443, 2.4874], atol=5e-5, rtol=0)
    close(fit.std_errors.to_numpy(), [0.387, 0.070], atol=5e-4, rtol=0)
    assert round(fit.r2, 3) == 0.928


def test_ch2_diagnostics_match_statsmodels(sim, sim_ols):
    from scipy import stats
    from statsmodels.stats.diagnostic import (
        acorr_breusch_godfrey,
        het_arch,
        het_breuschpagan,
        het_white,
    )
    from statsmodels.stats.outliers_influence import OLSInfluence, reset_ramsey
    from statsmodels.stats.stattools import durbin_watson

    fit = sp.regress("Y ~ X", data=sim)

    def estat(name, **kw):
        return sp.estat(fit, name, print_results=False, **kw)

    # Table 18, Shapiro-Wilk on the residuals (printed 0.96726, 0.01369)
    w, p = stats.shapiro(sim_ols.resid)
    out = sp.swilk(fit.residuals())
    close(out["W"].iloc[0], w)
    close(out["pvalue"].iloc[0], p, rtol=1e-6)
    # Table 21, Breusch-Pagan (printed LM 0.03799)
    bp = het_breuschpagan(sim_ols.resid, sim_ols.model.exog)
    close(estat("hettest")["statistic"], bp[0])
    close(estat("white")["statistic"], het_white(sim_ols.resid, sim_ols.model.exog)[0])
    # Table 23, Durbin-Watson (printed 2.08323)
    close(estat("dwatson")["statistic"], durbin_watson(sim_ols.resid))
    bg = acorr_breusch_godfrey(sim_ols, nlags=6)
    close(estat("bgodfrey", lags=6)["statistic"], bg[0])
    # Table 24, RESET with powers 2..5 of the fitted values (printed F 0.5262)
    reset = reset_ramsey(sim_ols, degree=5)
    close(estat("reset", powers=5)["statistic"], float(reset.fvalue), rtol=1e-6)
    # Table 25, Cook's distance: the book flags rows 20, 64, 87
    cooks = OLSInfluence(sim_ols).cooks_distance[0]
    ours = sp.influence_measures(fit)["cooksd"].to_numpy()
    close(ours, cooks)
    assert list(np.where(ours > 4 / len(sim))[0]) == [20, 64, 87]
    # Table 30, ARCH LM with statsmodels' ten lags (printed 11.3258)
    arch = het_arch(sim_ols.resid)
    close(estat("archlm", lags=10)["statistic"], arch[0])


def test_ch2_ttest_and_bartlett(sim):
    from scipy import stats

    # Table 19
    np.random.seed(1235)
    x = stats.norm.rvs(size=10000)
    ref = stats.ttest_1samp(x, 0.5)
    ours = sp.ttest(x, mu=0.5)
    close(ours.statistic, ref.statistic)
    close(ours.statistic, -49.76347123142897)  # as printed
    # Table 20, Bartlett's test of equal variances (printed 7.05744)
    np.random.seed(0)
    g1 = np.random.normal(loc=0, scale=1, size=30)
    g2 = np.random.normal(loc=0, scale=2, size=30)
    frame = pd.DataFrame({"y": np.r_[g1, g2], "g": [1] * 30 + [2] * 30})
    ref = stats.bartlett(g1, g2)
    out = sp.oneway(frame, "y", "g").estimates
    close(out["bartlett_chi2"], ref.statistic)
    close(out["bartlett_chi2"], 7.057436107054651)
    close(out["bartlett_p"], ref.pvalue)


def test_ch2_chow_test_as_chow_defined_it(sim):
    """Table 27 prints F = 46.02 from ``(RSS1 + RSS2) / RSS * (n - 2) / 2``,
    which is not Chow's statistic: it is near ``(n - 2) / 2`` whenever there
    is no break. The test at observation 50 is F(2, 96) = 3.10."""
    from scipy import stats

    def rss(frame):
        X = np.column_stack([np.ones(len(frame)), frame["X"]])
        e = frame["Y"] - X @ np.linalg.lstsq(X, frame["Y"], rcond=None)[0]
        return float(e @ e)

    r, r1, r2 = rss(sim), rss(sim.iloc[:50]), rss(sim.iloc[50:])
    close((r1 + r2) / r * 98 / 2, 46.0246640706214)  # the book's number
    f = ((r - r1 - r2) / 2) / ((r1 + r2) / 96)
    res = sp.chow_test(sim, "Y", ["X"], break_point=50)
    close(res["statistic"], f)
    close(res["pvalue"], stats.f.sf(f, 2, 96))
    assert 3.0 < res["statistic"] < 3.2


def test_ch2_cusum_and_bds_on_ols_residuals(sim, sim_ols):
    from statsmodels.stats.diagnostic import breaks_cusumolsresid
    from statsmodels.tsa.stattools import bds

    # Table 28: statsmodels divides the residual sum of squares by n
    # (printed 1.15406); with n - k the statistic is 1.14246
    printed = breaks_cusumolsresid(sim_ols.resid.to_numpy())
    close(printed[0], 1.154060357687865)
    ours = sp.cusum_test(sim, "Y", ["X"], method="ols")
    close(ours["statistic"] * np.sqrt(100 / 98), printed[0])
    ref = breaks_cusumolsresid(sim_ols.resid.to_numpy(), ddof=2)
    close(ours["statistic"], ref[0])
    close(ours["p_value"], ref[1])
    # Table 29 (printed -0.67419, p 0.50019)
    stat, pval = bds(sim_ols.resid.to_numpy())
    out = sp.bds(sp.regress("Y ~ X", data=sim))
    close(out["statistic"].iloc[0], float(stat))
    close(out["pvalue"].iloc[0], float(pval))
    close(out["statistic"].iloc[0], -0.6741896383124631)


# ------------------------------------------------------- chapters 5 to 7
@pytest.fixture(scope="module")
def monthly():
    table = pd.read_csv(Path(ROOT) / "data" / "yahoo_monthly_adjclose.csv")
    table["Date"] = pd.to_datetime(table["Date"])
    return table.set_index("Date")


@pytest.fixture(scope="module")
def factors(monthly):
    """Tables 66-67: oil excess returns and macro factors, as the book
    builds them (positional concatenation included)."""
    s, e = "2010-01-01", "2024-01-01"
    data = monthly[["CL=F", "^GSPC"]].reset_index()
    month_end = lambda name: fred(name, s, e).resample("ME").last() / 100  # noqa: E731
    cols = {
        "CPIAUCSL": fred("CPIAUCSL", s, e),
        "DGS3MO": month_end("DGS3MO"),
        "DGS10": month_end("DGS10"),
        "INDPRO": fred("INDPRO", s, e),
        "TOTALSL": fred("TOTALSL", s, e),
    }
    for name, table in cols.items():
        data[name] = table[name].to_numpy()[: len(data)]
    data = data.set_index("Date").dropna()
    df = pd.DataFrame(index=data.index)
    rf = data["DGS3MO"] / 12
    df["oil"] = np.log(data["CL=F"]).diff() - rf
    df["mkt"] = np.log(data["^GSPC"]).diff() - rf
    df["indpro"] = data["INDPRO"].diff()
    df["inflation"] = data["CPIAUCSL"].pct_change() * 100
    df["credit"] = np.log(data["TOTALSL"]).diff()
    spread = data["DGS10"] - data["DGS3MO"]
    df["term"] = spread.diff()
    df["spread"] = spread
    return df.dropna()


FACTOR_MODEL = "oil ~ mkt + inflation + indpro + term + spread + credit"


@needs_data
def test_ch5_capm_beta_and_rolling_beta(monthly):
    from scipy import stats

    r = np.log(monthly[["AAPL", "^GSPC"]]).diff().dropna()
    r.columns = ["aapl", "sp"]
    slope, intercept = stats.linregress(r["sp"], r["aapl"])[:2]  # Table 38
    fit = sp.regress("aapl ~ sp", data=r)
    close(fit.params.to_numpy(), [intercept, slope])
    # Table 39: 36-month rolling beta
    path = sp.rolling("aapl ~ sp", r, window=36)
    ref = [
        stats.linregress(r["sp"].iloc[i : i + 36], r["aapl"].iloc[i : i + 36])[0]
        for i in range(len(r) - 35)
    ]
    close(path["sp"].to_numpy(), ref)
    assert path.index[0] == r.index[35]


@needs_data
def test_ch5_multifactor_model_and_its_tests(factors):
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    ref = smf.ols(FACTOR_MODEL, factors).fit()
    fit = sp.regress(FACTOR_MODEL, data=factors)
    close(fit.params.to_numpy(), ref.params.to_numpy())
    close(fit.std_errors.to_numpy(), ref.bse.to_numpy())
    # Tables 55 and 71: the book's chained hypothesis strings
    for hypothesis in ("credit = spread = 0", "mkt = 1", "mkt = Intercept = 1"):
        f = ref.f_test(hypothesis)
        ours = sp.test(fit, hypothesis)
        close(ours["statistic"], float(f.fvalue))
        close(ours["pvalue"], float(f.pvalue))
    # Tables 78 and 81: HC3 and Newey-West with six lags
    hc3 = smf.ols(FACTOR_MODEL, factors).fit(cov_type="HC3")
    close(sp.regress(FACTOR_MODEL, data=factors, robust="hc3").std_errors, hc3.bse)
    hac = smf.ols(FACTOR_MODEL, factors).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    ours = sp.regress(FACTOR_MODEL, data=factors, robust="hac", hac_lags=6)
    close(ours.std_errors.to_numpy(), hac.bse.to_numpy())
    # Table 85: a power written with numpy inside the formula
    squared = FACTOR_MODEL + " + np.power(mkt, 2)"
    close(
        sp.regress(squared, data=factors).params.to_numpy(),
        smf.ols(squared, factors).fit().params.to_numpy(),
        rtol=1e-8,
    )
    # Table 70: the book calls variance_inflation_factor on a design with no
    # constant, which gives uncentred factors; with the constant they agree
    names = ["mkt", "inflation", "indpro", "term", "spread", "credit"]
    design = ref.model.exog
    centred = [variance_inflation_factor(design, i) for i in range(1, 7)]
    ours = sp.vif(factors, names).set_index("variable")["VIF"]
    close(ours[names].to_numpy(), centred)
    uncentred = [variance_inflation_factor(factors[names].to_numpy(), i) for i in range(6)]
    assert not np.allclose(uncentred, centred, rtol=0.01)


@needs_data
def test_ch5_stability_of_the_multifactor_model(factors):
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import breaks_cusumolsresid
    from statsmodels.tsa.stattools import bds

    names = ["mkt", "inflation", "indpro", "term", "spread", "credit"]
    ref = smf.ols(FACTOR_MODEL, factors).fit()
    # Table 94: Chow test at the middle of the sample, computed correctly
    half = len(factors) // 2
    r1 = smf.ols(FACTOR_MODEL, factors.iloc[:half]).fit().ssr
    r2 = smf.ols(FACTOR_MODEL, factors.iloc[half:]).fit().ssr
    f = ((ref.ssr - r1 - r2) / 7) / ((r1 + r2) / (len(factors) - 14))
    res = sp.chow_test(factors, "oil", names, break_point=half)
    close(res["statistic"], f)
    by_date = sp.chow_test(factors, "oil", names, break_point=factors.index[half])
    close(by_date["statistic"], f)
    # Table 91
    cus = breaks_cusumolsresid(ref.resid.to_numpy(), ddof=7)
    ours = sp.cusum_test(factors, "oil", names, method="ols")
    close(ours["statistic"], cus[0])
    close(ours["p_value"], cus[1])
    # Table 93
    stat, pval = bds(ref.resid.to_numpy())
    out = sp.bds(sp.regress(FACTOR_MODEL, data=factors))
    close(out["statistic"].iloc[0], float(stat))
    close(out["pvalue"].iloc[0], float(pval))
    # Table 92: 12-month rolling coefficients
    path = sp.rolling(FACTOR_MODEL, factors, window=12)
    one = smf.ols(FACTOR_MODEL, factors.iloc[20:32]).fit()
    close(path.iloc[20][list(one.params.index)].to_numpy(float), one.params, rtol=1e-6)


@needs_data
def test_ch5_quantile_regression_is_at_the_minimum(factors):
    """Table 56. statsmodels' ``quantreg`` iterates reweighted least squares
    and stops a little above the minimum of the check loss; the linear
    programme has an exact solution, which is what ``sp.qreg`` returns."""
    import statsmodels.formula.api as smf
    from scipy.optimize import linprog

    y = factors["oil"].to_numpy()
    X = np.column_stack([np.ones(len(y)), factors[["mkt", "inflation"]]])
    n, k = X.shape

    def loss(b, q):
        u = y - X @ b
        return float(np.sum(u * (q - (u < 0))))

    for q in (0.1, 0.5, 0.9):
        lp = linprog(
            np.r_[np.zeros(k), q * np.ones(n), (1 - q) * np.ones(n)],
            A_eq=np.column_stack([X, np.eye(n), -np.eye(n)]),
            b_eq=y,
            bounds=[(None, None)] * k + [(0, None)] * (2 * n),
            method="highs",
        )
        ours = sp.qreg(factors, "oil ~ mkt + inflation", quantile=q).params.to_numpy()
        theirs = smf.quantreg("oil ~ mkt + inflation", factors).fit(q=q).params
        close(loss(ours, q), lp.fun, rtol=1e-10)
        assert loss(ours, q) <= loss(theirs.to_numpy(), q)
        close(ours, theirs.to_numpy(), rtol=2e-3, atol=2e-4)


@pytest.fixture(scope="module")
def quarterly():
    s, e = "2000-01-01", "2023-12-31"
    table = pd.concat(
        [fred("GDP", s, e), fred("CPIAUCSL", s, e), fred("W068RCQ027SBEA", s, e)], axis=1
    )
    table.columns = ["GDP", "CPI", "EXP"]
    return table.dropna()


@needs_data
def test_ch6_unit_roots_and_ar_order(quarterly):
    from statsmodels.tsa.ar_model import AutoReg, ar_select_order
    from statsmodels.tsa.stattools import adfuller

    # Table 115
    for col in ("GDP", "CPI", "EXP"):
        for series in (quarterly[col], quarterly[col].diff().dropna()):
            ref = adfuller(series)
            # statsmodels rounds the largest lag up, StatsPAI down
            most = int(np.ceil(12 * (len(series) / 100) ** 0.25))
            ours = sp.unitroot(series, max_lags=most)
            close(ours.statistic, ref[0])
            close(ours.pvalue, ref[1])
            assert ours.lags == ref[2]
    # Tables 108-109: AR order by BIC, then the fit
    growth = np.log(quarterly["GDP"]).diff().dropna().to_numpy()
    sel = ar_select_order(growth, maxlag=6)
    frame = pd.DataFrame({"g": growth})
    ours = sp.ardl(frame, "g", lags="bic", max_lags=6, vce="nonrobust")
    assert ours.lags == len(sel.ar_lags or [])
    ref = AutoReg(growth, lags=3).fit()
    fit = sp.ardl(frame, "g", lags=3, vce="nonrobust")
    close(fit.params.to_numpy(), ref.params)
    # AutoReg divides the residual sum of squares by n, OLS by n - k
    close(fit.std_errors.to_numpy() * np.sqrt((fit.nobs - 4) / fit.nobs), ref.bse)
    close(fit.forecast(steps=6)["forecast"].to_numpy(), ref.forecast(6))


@needs_data
def test_ch6_ardl_order_search_fit_and_forecast(quarterly):
    from statsmodels.tsa.ardl import ARDL, ardl_select_order

    y, X = quarterly["GDP"], quarterly[["CPI", "EXP"]]
    # Table 116
    sel = ardl_select_order(y, 4, X, 4)
    ours = sp.ardl(
        quarterly, "GDP", ["CPI", "EXP"], lags="bic", x_lags="bic", max_lags=4,
        contemporaneous=True, vce="nonrobust",
    )  # fmt: skip
    assert ours.lags == len(sel.model.ar_lags)
    for var in ("CPI", "EXP"):
        assert ours.x_lags[var] == max(sel.model.dl_lags[var])
    p, q = ours.lags, ours.x_lags
    ref = ARDL(y, p, X, q).fit()
    fit = sp.ardl(
        quarterly, "GDP", ["CPI", "EXP"], lags=p, x_lags=q,
        contemporaneous=True, vce="nonrobust",
    )  # fmt: skip
    close(fit.params.to_numpy(), ref.params.to_numpy(), rtol=1e-7)
    # Tables 117-118: forecasts given a path for the regressors
    future = X.iloc[-4:].copy()
    future.index = pd.date_range("2024-01-01", periods=4, freq="QS")
    theirs = ref.predict(start=len(y), end=len(y) + 3, exog_oos=future)
    close(fit.forecast(steps=4, exog=future)["forecast"].to_numpy(), theirs, rtol=1e-7)
    # Table 112, the geometric lag: its long-run effect needs |lambda| < 1
    growth = np.log(quarterly).diff().dropna()
    koyck = sp.ardl(growth, "GDP", ["EXP"], lags=1, x_lags=0, contemporaneous=True)
    lr = koyck.long_run()
    close(lr.loc["EXP", "coef"], koyck.params["EXP"] / (1 - koyck.params["GDP_L1"]))


@needs_data
def test_ch6_arima_and_its_residuals():
    """Tables 122-127. The book's ARIMA(1,1,1) on GDP in levels prints an AR
    coefficient of 1.0000 with a standard error of 0.002: statsmodels starts
    the level from a prior of variance 1e6, and GDP's innovation variance is
    1.5e5. The fit depends on the unit GDP is measured in; ours does not."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.arima.model import ARIMA

    gdp = fred("GDP", "2010-01-01", "2023-12-31")["GDP"]
    growth = np.log(gdp).diff().dropna()
    ref = _quiet(ARIMA(growth.to_numpy(), order=(0, 0, 0)).fit)  # Table 122
    fit = sp.arima(growth, order=(0, 0, 0))
    # the maximum is the sample mean and variance; statsmodels' search
    # stops 4e-4 short of it on a series this small in scale
    close(fit.params.to_numpy(), [growth.mean(), growth.var(ddof=0)], rtol=5e-5)
    close(fit.params.to_numpy(), ref.params, rtol=1e-3)
    assert fit.log_likelihood >= ref.llf
    close(fit.aic, ref.aic, rtol=1e-6)
    # Table 124
    lb = acorr_ljungbox(ref.resid, lags=10, boxpierce=True, return_df=True)
    ours = sp.corrgram(np.asarray(fit.residuals), lags=10, boxpierce=True)
    close(ours["Q"].to_numpy(), lb["lb_stat"].to_numpy(), rtol=1e-5)
    close(ours["BP"].to_numpy(), lb["bp_stat"].to_numpy(), rtol=1e-5)

    def ar_coefficient(scale, engine):
        if engine == "statsmodels":
            return float(_quiet(ARIMA((gdp * scale).to_numpy(), order=(1, 1, 0)).fit).params[0])
        return float(_quiet(sp.arima, gdp * scale, order=(1, 1, 0)).params["ar.L1"])

    theirs = [ar_coefficient(s, "statsmodels") for s in (1e-3, 1.0, 1e3)]
    ours = [ar_coefficient(s, "statspai") for s in (1e-3, 1.0, 1e3)]
    assert max(theirs) - min(theirs) > 0.1
    assert max(ours) - min(ours) < 1e-4
    # in units small enough for the prior to be diffuse the two agree
    close(ours[0], theirs[0], rtol=1e-3)
    # Table 121: a random walk needs its drift to be a candidate
    auto = _quiet(sp.arima, gdp, auto=True, max_d=1, max_p=2, max_q=2)
    assert "drift" in auto.params.index


@pytest.fixture(scope="module")
def macro():
    s, e = "1970-01-01", "2024-01-01"
    table = pd.concat([fred("GDPC1", s, e), fred("CPIAUCSL", s, e), fred("FGEXPND", s, e)], axis=1)
    table.columns = ["gdp", "cpi", "gov"]
    return np.log(table.dropna()).diff().dropna()


@needs_data
def test_ch7_var_order_fit_dynamics_and_forecast(macro):
    from statsmodels.tsa.api import VAR

    model = VAR(macro.to_numpy())
    order = model.select_order(maxlags=8)  # Table 133
    ours = sp.varsoc(macro, maxlag=8)
    for theirs, col in (("aic", "AIC"), ("bic", "SBIC"), ("hqic", "HQIC"), ("fpe", "FPE")):
        assert ours.attrs["selected"][col] == order.selected_orders[theirs]
    p = order.selected_orders["aic"]
    ref = model.fit(p)  # Table 135
    fit = sp.var(macro, lags=p, se_df="r")
    for i, name in enumerate(macro.columns):
        coefs = fit.coefs[name]
        const = coefs.loc[[n for n in coefs.index if "cons" in n.lower()], "coef"]
        close(const.iloc[0], ref.params[0, i], rtol=1e-7)
    close(fit.forecast(5)[list(macro.columns)].to_numpy(), ref.forecast(macro.to_numpy()[-p:], 5), rtol=1e-8)  # Table 143
    # Tables 139-140: impulse responses, unit and orthogonalised
    irf = ref.irf(10)
    for orthogonal, theirs in ((False, irf.irfs), (True, irf.orth_irfs)):
        paths = fit.irf(10, orthogonal=orthogonal)["irf"]
        for j, shock in enumerate(macro.columns):
            for i, response in enumerate(macro.columns):
                close(paths[f"{shock} -> {response}"], theirs[:, i, j], rtol=1e-6, atol=1e-10)
    # Table 141: variance decomposition (first row of the book's is step 1)
    fevd = ref.fevd(10).decomp
    table = fit.fevd(10)
    for i, response in enumerate(macro.columns):
        for j, shock in enumerate(macro.columns):
            mine = table[(table["response"] == response) & (table["shock"] == shock)]
            close(mine["fevd"].to_numpy()[1:], fevd[i, :, j], rtol=1e-7, atol=1e-12)


@needs_data
def test_ch7_cointegration_and_the_error_correction_model():
    from statsmodels.tsa.stattools import coint
    from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

    s, e = "1990-01-01", "2024-01-01"
    table = pd.concat(
        [fred(n, s, e).resample("QE").mean() for n in ("CPIAUCNS", "PCE", "DPI")], axis=1
    ).dropna()
    table.columns = ["CPI", "PCE", "DPI"]
    # Table 148: pairwise Engle-Granger tests, with p-values
    for a, b in (("CPI", "PCE"), ("CPI", "DPI"), ("PCE", "DPI")):
        stat, pvalue, _ = coint(table[a], table[b], maxlag=2, autolag=None)
        ours = sp.engle_granger(table, [a, b], lags=2)
        close(ours.test_stats, stat)
        close(ours.pvalue, pvalue)
    # Tables 148-151: Johansen statistics
    jo = coint_johansen(table, det_order=0, k_ar_diff=1)
    close(sp.johansen(table, lags=1).test_stats, jo.lr1, rtol=1e-8)
    close(sp.johansen(table, lags=1, test="maxeig").test_stats, jo.lr2, rtol=1e-8)
    # Table 152 and the impulse responses after it. The book fits rank 3 of
    # 3, which is a VAR in levels; rank 1 is fitted here
    ref = _quiet(VECM(table, k_ar_diff=2, coint_rank=1, deterministic="co").fit)
    fit = sp.vec(table, lags=2, rank=1)
    close(fit.alpha.to_numpy().ravel(), ref.alpha.ravel(), rtol=1e-6)
    close(fit.beta.loc[["CPI", "PCE", "DPI"]].to_numpy().ravel(), ref.beta.ravel(), rtol=1e-6)
    close(fit.forecast(6)[["CPI", "PCE", "DPI"]].to_numpy(), ref.predict(steps=6), rtol=1e-8)
    irf = ref.irf(6).irfs
    paths = fit.irf(6, orthogonal=False)["irf"]
    for j, shock in enumerate(table.columns):
        for i, response in enumerate(table.columns):
            close(paths[f"{shock} -> {response}"], irf[:, i, j], rtol=1e-6, atol=1e-9)
