"""Stata parity for the estimators and tests an introductory econometrics
course runs after a regression.

The reference numbers are real Stata 18 output on four committed synthetic
datasets (``_fixtures/_generate_textbook_methods_stata.do`` reads the same
CSV bytes). They cover what was added or corrected while replaying a
textbook's do-files: ``prais``, ``corrgram``, the ``estat`` specification
tests, the model F with a robust covariance, ``varsoc`` and the ``var`` /
``vec`` post-estimation suite, ``hausman``, ``xtreg, re | mle`` extras,
``xtsum``, ``xtserial``, ``xtoverid`` and the regression control method.

Tolerances
----------
* 1e-9 relative by default: the quantities are closed-form given the data.
* 1e-6 for the variance inflation factors, White's test and RESET: Stata
  returns the first as ``float`` and builds the auxiliary regressors of the
  other two as ``float`` variables.
* 1e-6 / 2e-5 for coefficients / standard errors of models Stata fits by
  iterating a likelihood (``logit``, ``xtreg, mle``): its Hessian is taken
  where its own iterations stopped.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def G():
    table = pd.read_csv(FIX / "textbook_methods_Stata.csv", skipinitialspace=True)
    values = {}
    for key, value in zip(table["key"], table["value"]):
        values[key] = float(value)
    return values


@pytest.fixture(scope="module")
def cs():
    return pd.read_csv(FIX / "textbook_cs.csv")


@pytest.fixture(scope="module")
def ts():
    return pd.read_csv(FIX / "textbook_ts.csv")


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(FIX / "textbook_panel.csv")


def close(ours, stata, rtol=1e-9, atol=0.0):
    assert np.isclose(ours, stata, rtol=rtol, atol=atol), (ours, stata)


# ------------------------------------------------------------- regression
def test_regression_without_a_constant_reports_uncentred_fit(G, cs):
    fit = sp.regress("y ~ x1 + x2 + d - 1", data=cs)
    close(fit.diagnostics["R-squared"], G["nocons.r2"])
    close(fit.diagnostics["Adj. R-squared"], G["nocons.r2_a"])
    close(fit.diagnostics["F-statistic"], G["nocons.F"])
    assert fit.data_info["df_model"] == G["nocons.df_m"]
    rmse = np.sqrt(fit.data_info["rss"] / fit.data_info["df_resid"])
    close(rmse, G["nocons.rmse"])


def test_model_f_uses_the_robust_covariance(G, cs):
    robust = sp.regress("y ~ x1 + x2 + d", data=cs, robust="hc1")
    close(robust.diagnostics["F-statistic"], G["robust.F"])
    close(robust.diagnostics["R-squared"], G["robust.r2"])
    clustered = sp.regress("y ~ x1 + x2 + d", data=cs, cluster="g")
    close(clustered.diagnostics["F-statistic"], G["cluster.F"])
    # and it is not the ratio of sums of squares
    classical = sp.regress("y ~ x1 + x2 + d", data=cs)
    assert abs(classical.diagnostics["F-statistic"] / G["robust.F"] - 1) > 0.01


def test_estat_specification_tests(G, cs):
    fit = sp.regress("y ~ x1 + x2 + d", data=cs)

    def run(test, **kw):
        return sp.estat(fit, test, print_results=False, **kw)

    out = run("hettest", variables="fitted", version="normal")
    close(out["statistic"], G["hettest.normal_fitted.chi2"])
    close(out["pvalue"], G["hettest.normal_fitted.p"], rtol=1e-8)
    close(run("hettest", variables="fitted")["statistic"], G["hettest.iid_fitted.chi2"])
    out = run("hettest")  # sp default: every regressor, Koenker's N R^2
    close(out["statistic"], G["hettest.iid_rhs.chi2"])
    assert out["df"] == G["hettest.iid_rhs.df"]
    close(
        run("hettest", variables="rhs", version="normal")["statistic"],
        G["hettest.normal_rhs.chi2"],
    )
    close(run("hettest", variables=["x1", "d"])["statistic"], G["hettest.iid_x1d.chi2"])
    out = run("hettest", version="fstat")
    close(out["statistic"], G["hettest.fstat_rhs.F"])
    close(out["pvalue"], G["hettest.fstat_rhs.p"], rtol=1e-8)

    out = run("white")
    # imtest builds the squares and cross-products as float variables
    close(out["statistic"], G["white.chi2"], rtol=1e-6)
    # the square of the dummy is the dummy: 8 restrictions, not 9
    assert out["df"] == G["white.df"] == 8

    im = run("imtest")["table"]
    close(im.loc["skewness", "chi2"], G["imtest.skew"], rtol=1e-6)
    close(im.loc["kurtosis", "chi2"], G["imtest.kurt"], rtol=1e-6)
    close(im.loc["total", "chi2"], G["imtest.total"], rtol=1e-6)
    assert im.loc["skewness", "df"] == G["imtest.df_skew"]
    assert im.loc["total", "df"] == G["imtest.df_total"]

    out = run("reset", powers=4)
    close(out["statistic"], G["ovtest.F"], rtol=5e-6)
    close(out["pvalue"], G["ovtest.p"], rtol=5e-6)
    out = run("reset", powers=4, rhs=True)
    close(out["statistic"], G["ovtest_rhs.F"], rtol=5e-6)
    assert out["df1"] == G["ovtest_rhs.df"] == 6  # the dummy adds no power

    vif = run("vif")["vif_table"].set_index("variable")["VIF"]
    assert "vif.order_x1_x2_d" in G  # Stata lists them by decreasing VIF
    close(vif["x1"], G["vif.v1"], rtol=1e-6)
    close(vif["x2"], G["vif.v2"], rtol=1e-6)
    close(vif["d"], G["vif.v3"], rtol=1e-6)

    ic = run("ic")
    close(ic["ll"], G["ic.ll"])
    assert ic["k"] == G["ic.df"]
    close(ic["AIC"], G["ic.aic"])
    close(ic["BIC"], G["ic.bic"])
    # the regression's own diagnostics use the same definition
    close(fit.diagnostics["AIC"], G["ic.aic"])
    close(fit.diagnostics["BIC"], G["ic.bic"])


def test_endogeneity_overidentification_and_hausman(G, cs):
    iv = sp.iv("y ~ x1 + x2 + d + (endog ~ z1 + z2)", data=cs, small=False)
    close(iv.params["endog"], G["iv.b_endog"])
    close(iv.std_errors["endog"], G["iv.se_endog"])
    endo = sp.estat(iv, "endogenous", print_results=False)
    close(endo["statistic"], G["endog.wu"], rtol=1e-8)
    close(endo["pvalue"], G["endog.p_wu"], rtol=1e-6)
    close(endo["durbin"], G["endog.durbin"], rtol=1e-8)
    close(endo["durbin_pvalue"], G["endog.p_durbin"], rtol=1e-6)
    over = sp.estat(iv, "overid", print_results=False)
    close(over["statistic"], G["overid.sargan"], rtol=1e-8)
    close(over["pvalue"], G["overid.p_sargan"], rtol=1e-8)

    # a robust or clustered fit gets the regression-based test on that
    # covariance, not the homoskedastic F
    for kw, key in (
        ({"robust": "hc0"}, "endog_robust"),
        ({"cluster": "g"}, "endog_cluster"),
    ):
        fit = sp.iv("y ~ x1 + x2 + d + (endog ~ z1 + z2)", data=cs, small=False, **kw)
        robust = sp.estat(fit, "endogenous", print_results=False)
        close(robust["statistic"], G[f"{key}.regF"], rtol=1e-8)
        close(robust["pvalue"], G[f"{key}.p_regF"], rtol=1e-6)
        close(robust["wu_hausman_F"], G["endog.wu"], rtol=1e-8)
        assert "durbin" not in robust

    ols = sp.regress("y ~ endog + x1 + x2 + d", data=cs)
    out = sp.hausman(iv, ols, sigmamore=True)
    close(out["statistic"], G["hausman_iv.chi2"], rtol=1e-7)
    assert out["df"] == G["hausman_iv.df"] == 1
    close(out["pvalue"], G["hausman_iv.p"], rtol=1e-6)
    out = sp.hausman(iv, ols, constant=True, sigmamore=True)
    close(out["statistic"], G["hausman_iv_cons.chi2"], rtol=1e-7)
    assert out["df"] == G["hausman_iv_cons.df"]


def test_classification_table_and_likelihood_criteria(G, cs):
    fit = sp.logit("b ~ x1 + x2 + d", data=cs)
    table = sp.estat(fit, "classification", print_results=False)
    for key, name in (("sens", "sensitivity"), ("spec", "specificity"),
                      ("ppv", "ppv"), ("npv", "npv"),
                      ("correct", "correctly_classified")):  # fmt: skip
        close(100 * table[name], G[f"clas.{key}"])
    other = sp.estat(fit, "classification", threshold=0.3, print_results=False)
    close(100 * other["correctly_classified"], G["clas30.correct"])
    ic = sp.estat(fit, "ic", print_results=False)
    close(ic["ll"], G["logit_ic.ll"], rtol=1e-8)
    close(ic["AIC"], G["logit_ic.aic"], rtol=1e-8)
    close(ic["BIC"], G["logit_ic.bic"], rtol=1e-8)


# ------------------------------------------------------------ time series
def test_serial_correlation_tests(G, ts):
    fit = sp.regress("y ~ x1 + x2 + d", data=ts)

    def bg(**kw):
        return sp.estat(fit, "bgodfrey", print_results=False, **kw)["statistic"]

    close(bg(), G["bg.l1_zero"])
    close(bg(lags=3), G["bg.l3_zero"])
    close(bg(lags=2, fill="drop"), G["bg.l2_drop"])
    close(sp.estat(fit, "dwatson", print_results=False)["statistic"], G["dw.d"])


@pytest.mark.parametrize("method", ["prais", "corc"])
@pytest.mark.parametrize(
    "rhotype", ["regress", "freg", "tscorr", "dw", "theil", "nagar"]
)
@pytest.mark.parametrize("step", ["iter", "two"])
def test_prais(G, ts, method, rhotype, step):
    fit = sp.prais(
        "y ~ x1 + x2 + d", data=ts, method=method, rhotype=rhotype,
        twostep=(step == "two"), tol=1e-12, maxiter=500,
    )  # fmt: skip
    key = f"prais.{method}.{rhotype}.{step}."
    # Stata stops iterating when rho moves by less than 1e-6
    rtol = 1e-9 if step == "two" else 2e-5
    close(fit.model_info["rho"], G[key + "rho"], rtol=rtol)
    close(fit.params["x1"], G[key + "b_x1"], rtol=rtol)
    close(fit.std_errors["x1"], G[key + "se_x1"], rtol=rtol)
    close(fit.params["Intercept"], G[key + "b_cons"], rtol=max(rtol, 1e-8))
    close(fit.std_errors["Intercept"], G[key + "se_cons"], rtol=rtol)
    close(fit.diagnostics["R-squared"], G[key + "r2"], rtol=rtol)
    close(fit.diagnostics["F-statistic"], G[key + "F"], rtol=rtol)
    close(fit.diagnostics["Root MSE"], G[key + "rmse"], rtol=rtol)
    close(fit.model_info["dw_transformed"], G[key + "dw"], rtol=rtol)
    close(fit.model_info["dw_original"], G[key + "dw0"])
    assert fit.data_info["nobs"] == G[key + "N"]


def test_prais_robust_covariance(G, ts):
    fit = sp.prais("y ~ x1 + x2 + d", data=ts, vce="robust", tol=1e-12)
    close(fit.std_errors["x1"], G["prais.robust.se_x1"], rtol=2e-5)
    close(fit.diagnostics["F-statistic"], G["prais.robust.F"], rtol=5e-5)
    fit = sp.prais("y ~ x1 + x2 + d", data=ts, method="corc", vce="hc3", tol=1e-12)
    close(fit.std_errors["x1"], G["prais.hc3.se_x1"], rtol=2e-5)


def test_corrgram(G, ts):
    table = sp.corrgram(ts, "y", lags=8)
    for k in range(1, 9):
        close(table.loc[k, "AC"], G[f"corrgram.ac{k}"])
        close(table.loc[k, "PAC"], G[f"corrgram.pac{k}"])
        close(table.loc[k, "Q"], G[f"corrgram.q{k}"])
    yw = sp.corrgram(ts, "y", lags=4, pac="yw")
    for k in range(1, 5):
        close(yw.loc[k, "PAC"], G[f"corrgram_yw.pac{k}"])
    last = sp.corrgram(ts, "y", lags=6).iloc[-1]
    close(last["Q"], G["wntestq.stat"])
    close(last["Prob>Q"], G["wntestq.p"], rtol=1e-8)


def test_var_lag_selection_and_postestimation(G, ts):
    soc = sp.varsoc(ts, ["z1", "z2"], maxlag=4)
    for lag in range(5):
        close(soc.loc[lag, "LL"], G[f"varsoc.ll{lag}"])
        close(soc.loc[lag, "FPE"], G[f"varsoc.fpe{lag}"])
        close(soc.loc[lag, "AIC"], G[f"varsoc.aic{lag}"])
        close(soc.loc[lag, "HQIC"], G[f"varsoc.hqic{lag}"])
        close(soc.loc[lag, "SBIC"], G[f"varsoc.sbic{lag}"])
    close(soc.loc[2, "LR"], G["varsoc.lr2"], rtol=1e-8)
    close(soc.loc[2, "p"], G["varsoc.p2"], rtol=1e-7)

    fit = sp.var(ts, ["z1", "z2"], lags=2)
    close(fit.log_likelihood, G["var.ll"])
    close(fit.aic, G["var.aic"])
    close(fit.fpe, G["var.fpe"])
    close(fit.det_sigma, G["var.detsig"])
    eq = fit.equation_table()
    close(eq.loc["z1", "rmse"], G["var.rmse_1"])
    close(eq.loc["z2", "r2"], G["var.r2_2"])
    close(eq.loc["z1", "chi2"], G["var.chi2_1"])
    close(fit.coefs["z1"].loc["L1.z2", "coef"], G["var.b_z1_L1z2"])
    close(fit.coefs["z1"].loc["L1.z2", "se"], G["var.se_z1_L1z2"])

    wle = fit.lag_exclusion().set_index(["equation", "lag"])["chi2"]
    close(wle[("z1", 1)], G["varwle.z1_l1"])
    close(wle[("z1", 2)], G["varwle.z1_l2"])
    close(wle[("All", 2)], G["varwle.all_l2"])
    lm = sp.estat(fit, "varlmar", lags=3, print_results=False)["table"]
    for s in (1, 2, 3):
        close(lm.set_index("lag").loc[s, "chi2"], G[f"varlmar.l{s}"], rtol=1e-8)
    modulus = fit.stability()["modulus"]
    close(modulus[1], G["varstable.m2"], rtol=1e-7)
    close(modulus[3], G["varstable.m4"], rtol=1e-7)
    assert fit.stability().attrs["stable"]
    granger = fit.granger_table().set_index(["equation", "excluded"])["chi2"]
    close(granger[("z1", "z2")], G["vargranger.z1_z2"])
    close(granger[("z2", "z1")], G["vargranger.z2_z1"])
    ahead = fit.forecast(4)
    close(ahead.loc[1, "z1"], G["fcast.z1_h1"])
    close(ahead.loc[1, "z2"], G["fcast.z2_h1"])
    close(ahead.loc[4, "z1"], G["fcast.z1_h4"])


def test_vecm(G, ts):
    rank = sp.johansen(ts, ["c1", "c2", "c3"], lags=1)
    close(rank.test_stats[0], G["vecrank.trace0"], rtol=1e-8)
    close(rank.test_stats[1], G["vecrank.trace1"], rtol=1e-8)
    close(rank.eigenvalues[0], G["vecrank.lambda1"], rtol=1e-8)

    fit = sp.vec(ts, ["c1", "c2", "c3"], lags=1, rank=1)
    close(fit.log_likelihood, G["vec.ll"])
    close(fit.aic, G["vec.aic"])
    close(fit.bic, G["vec.sbic"])
    close(fit.det_sigma, G["vec.detsig"])
    close(fit.coefs["D_c1"].loc["L._ce1", "coef"], G["vec.alpha_c1"], rtol=1e-8)
    close(fit.coefs["D_c1"].loc["L._ce1", "se"], G["vec.se_alpha_c1"], rtol=1e-8)
    close(fit.coefs["D_c2"].loc["LD.c3", "coef"], G["vec.gamma_c2_c3"], rtol=1e-8)
    close(fit.coefs["D_c2"].loc["LD.c3", "se"], G["vec.se_gamma_c2_c3"], rtol=1e-8)
    close(fit.coefs["D_c3"].loc["_cons", "coef"], G["vec.cons_c3"], rtol=1e-7)
    close(fit.coefs["D_c3"].loc["_cons", "se"], G["vec.se_cons_c3"], rtol=1e-8)
    close(fit.beta.loc["c2", "_ce1"], G["vec.beta_c2"], rtol=1e-8)
    close(fit.beta.loc["c3", "_ce1"], G["vec.beta_c3"], rtol=1e-8)
    close(fit.beta.loc["_cons", "_ce1"], G["vec.beta_cons"], rtol=1e-7)
    close(fit.beta_se.loc["c2", "_ce1"], G["vec.se_beta_c2"], rtol=1e-8)
    close(fit.beta_se.loc["c3", "_ce1"], G["vec.se_beta_c3"], rtol=1e-8)
    eq = fit.equation_table()
    close(eq.loc["D_c1", "rmse"], G["vec.rmse_1"], rtol=1e-8)
    close(eq.loc["D_c1", "r2"], G["vec.r2_1"], rtol=1e-8)
    close(eq.loc["D_c1", "chi2"], G["vec.chi2_1"], rtol=1e-8)
    lm = sp.estat(fit, "veclmar", print_results=False)["table"].set_index("lag")
    close(lm.loc[1, "chi2"], G["veclmar.l1"], rtol=1e-7)
    close(lm.loc[2, "chi2"], G["veclmar.l2"], rtol=1e-7)
    stable = fit.stability()
    assert stable.attrs["unit_moduli"] == 2
    close(stable["modulus"][2], G["vecstable.m3"], rtol=1e-7)

    # two relations, the constant restricted to them
    fit = sp.vec(ts, ["c1", "c2", "c3"], lags=2, rank=2, trend="rc")
    close(fit.log_likelihood, G["vec_rc.ll"])
    close(fit.beta.loc["c3", "_ce1"], G["vec_rc.beta1_c3"], rtol=1e-8)
    close(fit.beta.loc["_cons", "_ce1"], G["vec_rc.beta1_cons"], rtol=1e-8)
    close(fit.beta.loc["c3", "_ce2"], G["vec_rc.beta2_c3"], rtol=1e-8)
    close(fit.beta_se.loc["c3", "_ce1"], G["vec_rc.se_beta1_c3"], rtol=1e-8)
    close(fit.beta_se.loc["_cons", "_ce1"], G["vec_rc.se_beta1_cons"], rtol=1e-8)
    close(fit.coefs["D_c2"].loc["L._ce2", "coef"], G["vec_rc.alpha_c2_ce2"], rtol=1e-8)
    close(fit.coefs["D_c2"].loc["L._ce2", "se"], G["vec_rc.se_alpha_c2_ce2"], rtol=1e-8)
    close(fit.equation_table().loc["D_c1", "rmse"], G["vec_rc.rmse_1"], rtol=1e-8)
    close(fit.equation_table().loc["D_c1", "chi2"], G["vec_rc.chi2_1"], rtol=1e-8)
    assert fit.n_params == G["vec_rc.k"]
    none = sp.vec(ts, ["c1", "c2", "c3"], lags=1, rank=1, trend="n")
    close(none.log_likelihood, G["vec_n.ll"])
    close(none.beta.loc["c2", "_ce1"], G["vec_n.beta_c2"], rtol=1e-8)
    close(none.beta_se.loc["c2", "_ce1"], G["vec_n.se_beta_c2"], rtol=1e-8)
    close(none.coefs["D_c1"].loc["L._ce1", "se"], G["vec_n.se_alpha_c1"], rtol=1e-8)


def test_vecm_with_linear_trends(G, ts):
    rt = sp.vec(ts, ["c1", "c2", "c3"], lags=1, rank=1, trend="rtrend")
    close(rt.log_likelihood, G["vec_rt.ll"])
    assert rt.n_params == G["vec_rt.k"]
    close(rt.beta.loc["c2", "_ce1"], G["vec_rt.beta_c2"], rtol=1e-8)
    close(rt.beta.loc["_trend", "_ce1"], G["vec_rt.beta_trend"], rtol=1e-7)
    close(rt.beta.loc["_cons", "_ce1"], G["vec_rt.beta_cons"], rtol=1e-7)
    close(rt.beta_se.loc["_trend", "_ce1"], G["vec_rt.se_beta_trend"], rtol=1e-8)
    close(rt.coefs["D_c1"].loc["_cons", "coef"], G["vec_rt.cons_c1"], rtol=1e-7)
    close(rt.coefs["D_c1"].loc["_cons", "se"], G["vec_rt.se_cons_c1"], rtol=1e-8)
    close(rt.coefs["D_c2"].loc["L._ce1", "se"], G["vec_rt.se_alpha_c2"], rtol=1e-8)

    ct = sp.vec(ts, ["c1", "c2", "c3"], lags=1, rank=1, trend="trend")
    close(ct.log_likelihood, G["vec_ct.ll"])
    assert ct.n_params == G["vec_ct.k"]
    close(ct.beta.loc["c2", "_ce1"], G["vec_ct.beta_c2"], rtol=1e-8)
    close(ct.beta.loc["_trend", "_ce1"], G["vec_ct.beta_trend"], rtol=1e-7)
    close(ct.beta.loc["_cons", "_ce1"], G["vec_ct.beta_cons"], rtol=1e-7)
    close(ct.coefs["D_c1"].loc["_trend", "coef"], G["vec_ct.trend_c1"], rtol=1e-7)
    close(ct.coefs["D_c1"].loc["_trend", "se"], G["vec_ct.se_trend_c1"], rtol=1e-8)
    close(ct.coefs["D_c1"].loc["_cons", "coef"], G["vec_ct.cons_c1"], rtol=1e-7)
    close(ct.coefs["D_c1"].loc["_cons", "se"], G["vec_ct.se_cons_c1"], rtol=1e-8)
    close(ct.coefs["D_c2"].loc["L._ce1", "se"], G["vec_ct.se_alpha_c2"], rtol=1e-8)


# ------------------------------------------------------------------- panel
def test_xtsum(G, panel):
    table = sp.xtsum(panel, ["y", "x1"], id="id")
    close(table.loc[("x1", "overall"), "sd"], G["xtsum.x1_sd"])
    close(table.loc[("x1", "between"), "sd"], G["xtsum.x1_sd_b"])
    close(table.loc[("x1", "within"), "sd"], G["xtsum.x1_sd_w"])
    close(table.loc[("x1", "within"), "min"], G["xtsum.x1_min_w"])
    close(table.loc[("x1", "between"), "max"], G["xtsum.x1_max_b"])


def test_xtreg_through_the_session(G, panel):
    """`xtreg` prints more than the slopes; the session derives the rest."""
    from statspai.agent._translation._stata_run import StataSession

    s = StataSession(panel)
    s.run("xtset id year")
    s.run("xtreg y x1 x2 w, fe")
    s.run("estimates store FE")
    e = s.stored["e"]
    close(s.stored["_b"]["_cons"], G["fe.cons"])
    close(s.stored["_se"]["_cons"], G["fe.se_cons"])
    for key in ("sigma_u", "sigma_e", "rho", "r2_w", "r2_b", "r2_o", "corr"):
        close(e[key], G[f"fe.{key}"], rtol=1e-8)
    s.run("xtreg y x1 x2 w, fe vce(robust)")
    close(s.stored["_se"]["_cons"], G["fe_robust.se_cons"], rtol=1e-8)
    close(s.stored["_se"]["x1"], G["fe_robust.se_x1"], rtol=1e-8)

    s.run("xtreg y x1 x2 w, re")
    s.run("estimates store RE")
    e = s.stored["e"]
    close(s.stored["_b"]["x1"], G["re.b_x1"], rtol=1e-8)
    close(s.stored["_se"]["x1"], G["re.se_x1"], rtol=1e-8)
    for key in ("sigma_u", "sigma_e", "rho", "r2_w", "r2_b", "r2_o"):
        close(e[key], G[f"re.{key}"], rtol=1e-8)
    xt = s.last.model_info["xt"]
    close(xt["theta_min"], G["re.thta_min"], rtol=1e-8)
    close(xt["theta_max"], G["re.thta_max"], rtol=1e-8)
    s.run("xttest0")
    close(s.output["statistic"], G["xttest0.lm"], rtol=1e-8)

    s.run("hausman FE RE")
    close(s.output["statistic"], G["hausman_xt.chi2"], rtol=1e-7)
    assert s.output["df"] == G["hausman_xt.df"]
    s.run("hausman FE RE, sigmamore")
    close(s.output["statistic"], G["hausman_xt_more.chi2"], rtol=1e-7)
    s.run("hausman FE RE, constant sigmamore")
    close(s.output["statistic"], G["hausman_xt_cons.chi2"], rtol=1e-7)
    assert s.output["df"] == G["hausman_xt_cons.df"]

    s.run("xtreg y x1 x2 w, re vce(robust)")
    close(s.stored["_se"]["x1"], G["re_robust.se_x1"], rtol=1e-8)
    s.run("xtoverid")
    close(s.output["statistic"], G["xtoverid.j"], rtol=1e-8)
    assert s.output["df"] == G["xtoverid.df"]

    s.run("xtreg y x1 x2 w, be")
    close(s.stored["_b"]["x1"], G["be.b_x1"], rtol=1e-8)
    close(s.stored["_se"]["x1"], G["be.se_x1"], rtol=1e-8)


def test_random_effects_by_maximum_likelihood(G, panel):
    fit = sp.panel(panel, "y ~ x1 + x2 + w", entity="id", time="year", method="mle")
    close(fit.params["x1"], G["mle.b_x1"], rtol=1e-6)
    close(fit.std_errors["x1"], G["mle.se_x1"], rtol=2e-5)
    close(fit.std_errors["const"], G["mle.se_cons"], rtol=2e-5)
    close(fit.model_info["sigma_u"], G["mle.sigma_u"], rtol=1e-5)
    close(fit.model_info["sigma_e"], G["mle.sigma_e"], rtol=1e-5)
    close(fit.model_info["ll"], G["mle.ll"], rtol=1e-9)
    close(fit.model_info["lr_sigma_u"], G["mle.chi2_c"], rtol=1e-7)
    close(fit.model_info["lr_chi2"], G["mle.chi2"], rtol=1e-7)


def test_mle_after_the_mixed_submodule_was_imported(panel):
    # importing statspai.multilevel.mixed rebinds the package attribute
    # `mixed` to the module; the fit must not depend on import order
    import statspai.multilevel.mixed  # noqa: F401

    fit = sp.panel(panel, "y ~ x1 + x2 + w", entity="id", time="year", method="mle")
    assert np.isfinite(fit.params["x1"])


def test_xtserial_and_xtoverid(G, panel):
    out = sp.xtserial(panel, "y", ["x1", "x2", "w"], id="id", time="year")
    close(out["statistic"], G["xtserial.F"], rtol=1e-8)
    close(out["pvalue"], G["xtserial.p"], rtol=1e-7)
    over = sp.xtoverid(panel, "y", ["x1", "x2", "w"], id="id")
    close(over["statistic"], G["xtoverid.j"], rtol=1e-8)


# ------------------------------------------------------ regression control
def test_regression_control_method(G):
    data = pd.read_csv(FIX / "textbook_rcm.csv")

    def fit(**kw):
        return sp.synth(
            data, "y", "unit", "t", treated_unit=1, treatment_time=31,
            method="rcm", **kw,
        )  # fmt: skip

    best = fit(placebo=False)
    info = best.model_info
    assert info["n_selected"] == G["rcm.best.K"]
    row = info["selection_table"].loc[info["n_selected"]]
    # rcm reshapes the outcome into float variables: seven digits
    close(row["aicc"], G["rcm.best.aicc"], rtol=1e-6)
    close(row["aic"], G["rcm.best.aic"], rtol=1e-6)
    close(row["bic"], G["rcm.best.bic"], rtol=1e-6)
    close(row["mbic"], G["rcm.best.mbic"], rtol=1e-6)
    close(info["pre_r2"], G["rcm.best.r2"], rtol=1e-6)
    close(best.estimate, G["rcm.best.att"], rtol=1e-6)
    # the mean squared error of the fit is per residual degree of freedom
    close(info["pre_mspe"], G["rcm.best.mse"], rtol=1e-6)
    close(info["pre_rmse"], G["rcm.best.rmse"], rtol=1e-6)

    forward = fit(placebo=False, selection="forward", criterion="bic")
    assert forward.model_info["n_selected"] == G["rcm.forward_bic.K"]
    close(forward.estimate, G["rcm.forward_bic.att"], rtol=1e-6)
    backward = fit(placebo=False, selection="backward", criterion="mbic")
    assert backward.model_info["n_selected"] == G["rcm.backward_mbic.K"]
    close(backward.estimate, G["rcm.backward_mbic.att"], rtol=1e-6)

    placebo = fit(placebo=True, placebo_cutoff=2.0)
    post = placebo.detail[placebo.detail["post"]].set_index("time")
    close(post.loc[31, "p_two_sided"], G["rcm.placebo.p_two_31"], rtol=1e-6)
    close(post.loc[36, "p_two_sided"], G["rcm.placebo.p_two_36"], rtol=1e-6)
    close(post.loc[36, "p_right"], G["rcm.placebo.p_right_36"], rtol=1e-6)
    units = placebo.model_info["placebo_units"]
    close(units.loc[1, "pre_mspe"], G["rcm.placebo.pre_mspe_treated"], rtol=1e-6)
    close(units.loc[1, "ratio"], G["rcm.placebo.ratio_treated"], rtol=1e-6)
    # the second row of Stata's table is unit 10 (it sorts the unit names)
    close(units.loc[10, "ratio"], G["rcm.placebo.ratio_row2"], rtol=1e-6)
    close(units.loc[10, "pre_mspe_relative"], G["rcm.placebo.relative_row2"], rtol=1e-6)
