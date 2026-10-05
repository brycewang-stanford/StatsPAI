"""Stata parity for the methods of a first causal-inference econometrics
course that were added or corrected while walking through its syllabus.

The reference numbers are real Stata 18 output on two committed synthetic
datasets (``_fixtures/_generate_textbook_syllabus_stata.do`` reads the same
CSV bytes): tests on a variance and z tests, ``ivregress gmm``, ``heckman``,
``truncreg``, factor-variable names in ``test`` / ``lincom``, ``testparm``,
``nlcom``, Durbin's alternative and ARCH LM tests, ``pperron``, ``kpss``,
``arima`` with a constant, and ``svar`` with short- and long-run restrictions.

Tolerances
----------
* 1e-9 relative by default: the quantities are closed-form given the data.
* 1e-6 for p-values of the variance tests near zero (absolute 1e-12).
* 1e-6 / 1e-5 for coefficients / standard errors of likelihoods Stata
  iterates (``heckman``, ``truncreg``, ``svar``): both sides are at the
  optimum to their own tolerance, and Stata differentiates numerically.
* 2e-5 / 2e-3 for ``arima``: Stata's Kalman filter and statsmodels'
  innovations algorithm stop at slightly different points of a flat
  likelihood, and the OPG standard errors inherit the difference.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
NAN = np.nan


@pytest.fixture(scope="module")
def G():
    table = pd.read_csv(FIX / "textbook_syllabus_Stata.csv", skipinitialspace=True)
    return {k: float(v) for k, v in zip(table["key"], table["value"])}


@pytest.fixture(scope="module")
def cs():
    return pd.read_csv(FIX / "textbook_cs.csv")


@pytest.fixture(scope="module")
def ts():
    return pd.read_csv(FIX / "textbook_ts.csv")


def close(ours, stata, rtol=1e-9, atol=0.0):
    assert np.isclose(ours, stata, rtol=rtol, atol=atol), (ours, stata)


# ------------------------------------------------------- variance, z tests
def test_one_sample_variance_test(G, cs):
    res = sp.sdtest(cs, "y", sd0=2)
    close(res.statistic, G["sd1.chi2"])
    assert res.df == G["sd1.df"]
    close(res.pvalue_less, G["sd1.p_l"], rtol=1e-6, atol=1e-12)
    close(res.pvalue, G["sd1.p"], rtol=1e-6, atol=1e-12)
    close(res.pvalue_greater, G["sd1.p_u"], rtol=1e-6, atol=1e-12)


def test_variance_ratio_test(G, cs):
    res = sp.sdtest(cs, "y", by="d")
    close(res.statistic, G["sd2.F"])
    assert res.df == (G["sd2.df_1"], G["sd2.df_2"])
    close(res.pvalue_less, G["sd2.p_l"], rtol=1e-6, atol=1e-12)
    close(res.pvalue, G["sd2.p"], rtol=1e-6, atol=1e-12)
    close(res.pvalue_greater, G["sd2.p_u"], rtol=1e-6, atol=1e-12)


def test_z_tests(G, cs):
    one = sp.ztest(cs, "y", mu=1, sd=2)
    close(one.statistic, G["z1.z"])
    close(one.se, G["z1.se"])
    close(one.pvalue, G["z1.p"], rtol=1e-6, atol=1e-12)
    close(one.pvalue_less, G["z1.p_l"], rtol=1e-6, atol=1e-12)
    two = sp.ztest(cs, "y", by="d", sd=(2, 3))
    close(two.statistic, G["z2.z"])
    close(two.se, G["z2.se"])
    close(two.pvalue, G["z2.p"], rtol=1e-6, atol=1e-12)


def test_immediate_forms(G):
    # Stata stores what it is given as float in the immediate commands
    sd = sp.sdtest(n=10, sd=1.14, sd0=2)
    close(sd.statistic, G["sdi.chi2"], rtol=1e-7)
    close(sd.pvalue_less, G["sdi.p_l"], rtol=1e-6)
    z = sp.ztest(n=10, mean=88, sd=0.7071, mu=85)
    close(z.statistic, G["zi.z"], rtol=1e-7)


def test_variance_test_rejects_at_the_nominal_rate_under_normality():
    """Known truth: with normal data and a true null the chi-squared test
    rejects 5% of the time, and the interval covers the true sd 95%."""
    rng = np.random.default_rng(20261005)
    reject = cover = 0
    reps = 4000
    for _ in range(reps):
        x = rng.normal(0.0, 3.0, size=25)
        res = sp.sdtest(n=25, sd=float(x.std(ddof=1)), sd0=3.0)
        reject += res.pvalue < 0.05
        cover += res.ci[0] < 3.0 < res.ci[1]
    # binomial se at 4000 replications: 0.0034
    assert abs(reject / reps - 0.05) < 0.012
    assert abs(cover / reps - 0.95) < 0.012


# ----------------------------------------------------------------- IV GMM
FML = "y ~ x1 + x2 + (endog ~ z1 + z2)"


def test_iv_gmm_with_the_robust_weight_matrix(G, cs):
    """``ivregress gmm`` defaults to wmatrix(robust) and a large-sample VCE:
    ``robust='hc1', small=False`` here (the conventions are pinned on Card's
    data in ``test_iv_gmm_stata_parity.py``; this is a second dataset)."""
    fit = sp.iv(FML, data=cs, method="gmm", robust="hc1", small=False)
    close(fit.params["endog"], G["gmm.b_endog"])
    close(fit.std_errors["endog"], G["gmm.se_endog"])
    close(fit.params["x1"], G["gmm.b_x1"])
    close(fit.std_errors["x1"], G["gmm.se_x1"])
    close(fit.diagnostics["Hansen J statistic"], G["gmm.J"])
    close(fit.diagnostics["Hansen J p-value"], G["gmm.J_p"])


def test_ivregress_gmm_runs_through_the_translator(G, cs):
    fit = sp.stata("ivregress gmm y x1 x2 (endog = z1 z2)", data=cs)
    close(fit.params["endog"], G["gmm.b_endog"])
    close(fit.std_errors["endog"], G["gmm.se_endog"])
    fit_u = sp.stata(
        "ivregress gmm y x1 x2 (endog = z1 z2), wmatrix(unadjusted)", data=cs
    )
    close(fit_u.params["endog"], G["gmmu.b_endog"])
    close(fit_u.std_errors["endog"], G["gmmu.se_endog"])


# ---------------------------------------------------------------- heckman
def _selected(cs):
    return cs.assign(ysel=cs["y"].where(cs["b"] == 1))


def test_heckman_two_step(G, cs):
    res = sp.heckman(_selected(cs), y="ysel", x=["x1", "x2"], z=["x1", "x2", "z1"])
    table = res.detail.set_index("variable")
    close(table.loc["x1", "coefficient"], G["heck2.b_x1"], rtol=1e-8)
    close(table.loc["x1", "se"], G["heck2.se_x1"], rtol=1e-7)
    close(table.loc["const", "coefficient"], G["heck2.b_cons"], rtol=1e-8)
    close(table.loc["const", "se"], G["heck2.se_cons"], rtol=1e-7)
    close(table.loc["lambda (IMR)", "coefficient"], G["heck2.lambda"], rtol=1e-8)
    close(table.loc["lambda (IMR)", "se"], G["heck2.se_lambda"], rtol=1e-7)
    close(res.model_info["rho"], G["heck2.rho"], rtol=1e-8)
    close(res.model_info["sigma"], G["heck2.sigma"], rtol=1e-8)


def test_heckman_maximum_likelihood(G, cs):
    data = _selected(cs)
    res = sp.heckman(data, y="ysel", x=["x1", "x2"], z=["x1", "x2", "z1"], method="mle")
    table = res.detail.set_index("variable")
    close(table.loc["x1", "coefficient"], G["heckml.b_x1"], rtol=1e-6)
    close(table.loc["x1", "se"], G["heckml.se_x1"], rtol=1e-5)
    close(table.loc["const", "coefficient"], G["heckml.b_cons"], rtol=1e-6)
    close(table.loc["athrho", "coefficient"], G["heckml.athrho"], rtol=1e-5)
    close(table.loc["athrho", "se"], G["heckml.se_athrho"], rtol=1e-4)
    close(res.model_info["log_likelihood"], G["heckml.ll"], rtol=1e-9)
    robust = sp.heckman(
        data, y="ysel", x=["x1", "x2"], z=["x1", "x2", "z1"], method="ml", vce="robust"
    )
    close(
        robust.detail.set_index("variable").loc["x1", "se"],
        G["heckrb.se_x1"],
        rtol=1e-5,
    )


def test_heckman_translation_follows_stata_defaults(G, cs):
    data = _selected(cs)
    ml = sp.from_stata("heckman ysel x1 x2, select(x1 x2 z1)")
    assert ml["arguments"]["method"] == "ml"
    assert "select" not in ml["arguments"]
    two = sp.stata("heckman ysel x1 x2, select(x1 x2 z1) twostep", data=data)
    close(two.detail.set_index("variable").loc["x1", "se"], G["heck2.se_x1"], rtol=1e-7)
    named = sp.from_stata("heckman y x1 x2, select(b = x1 x2 z1) vce(robust)")
    assert named["arguments"]["select"] == "b"
    assert named["arguments"]["vce"] == "robust"


def test_truncreg(G, cs):
    fit = sp.truncreg(cs[cs["y"] > 0], y="y", x=["x1", "x2"], ll=0)
    close(fit.params["x1"], G["trunc.b_x1"], rtol=1e-6)
    close(fit.std_errors["x1"], G["trunc.se_x1"], rtol=1e-5)
    close(fit.diagnostics["sigma"], G["trunc.sigma"], rtol=1e-6)
    close(fit.diagnostics["log_likelihood"], G["trunc.ll"], rtol=1e-9)
    call = sp.from_stata("truncreg y x1 x2, ll(0) vce(robust)")
    assert call["tool"] == "truncreg"
    assert call["arguments"] == {
        "y": "y",
        "x": ["x1", "x2"],
        "ll": 0.0,
        "robust": "robust",
    }


# -------------------------------------------- test / lincom / testparm / nlcom
def test_factor_variable_names(G, cs):
    fit = sp.regress("y ~ x1 + x2 + C(g)", data=cs)
    joint = sp.test(fit, "i.g")
    close(joint["statistic"], G["fv.testparm_F"])
    assert joint["df"][0] == G["fv.testparm_df"]
    close(sp.test(fit, "2.g = 3.g")["statistic"], G["fv.eq_F"])

    inter = sp.regress("y ~ x1 * C(b) + x2", data=cs)
    close(sp.test(inter, "1.b 1.b#c.x1")["statistic"], G["fv.joint_F"])
    combo = sp.lincom(inter, "x1 + 1.b#c.x1")
    close(combo["estimate"], G["fv.lincom_b"])
    close(combo["se"], G["fv.lincom_se"])


def test_factor_name_that_names_several_coefficients_is_refused(cs):
    fit = sp.regress("y ~ x1 + x2 + C(g)", data=cs)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="names"):
        sp.lincom(fit, "i.g + x1")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="Unknown"):
        sp.test(fit, "99.g")


def test_testparm_and_nlcom_run_through_the_translator(G, cs):
    out = sp.stata("regress y x1 x2 i.g\ntestparm i.g", data=cs)
    close(out["statistic"], G["fv.testparm_F"])
    ratio = sp.stata("regress y x1 x2 d\nnlcom _b[x1] / _b[x2]", data=cs)
    close(ratio["estimate"], G["nl.ratio_b"])
    close(ratio["se"], G["nl.ratio_se"])


def test_nlcom(G, cs):
    fit = sp.regress("y ~ x1 + x2 + d", data=cs)
    ratio = sp.nlcom(fit, "x1 / x2")
    close(ratio["estimate"], G["nl.ratio_b"])
    close(ratio["se"], G["nl.ratio_se"])
    growth = sp.nlcom(fit, "exp(_b[x1]) - 1")
    close(growth["estimate"], G["nl.exp_b"])
    close(growth["se"], G["nl.exp_se"])
    mix = sp.nlcom(fit, "x1 / (1 - d) + _cons^2")
    close(mix["estimate"], G["nl.mix_b"])
    close(mix["se"], G["nl.mix_se"])
    # a linear expression is lincom's, with a normal reference
    linear = sp.nlcom(fit, "x1 + 2 * x2")
    exact = sp.lincom(fit, "x1 + 2 * x2")
    close(linear["estimate"], exact["estimate"])
    close(linear["se"], exact["se"])


def test_nlcom_refuses_what_it_cannot_evaluate(cs):
    fit = sp.regress("y ~ x1 + x2 + d", data=cs)
    bad = sp.exceptions.MethodIncompatibility
    with pytest.raises(bad, match="unknown coefficient"):
        sp.nlcom(fit, "x1 / nothere")
    with pytest.raises(bad, match="not an arithmetic expression"):
        sp.nlcom(fit, "__import__('os').getcwd()")
    with pytest.raises(bad, match="not available"):
        sp.nlcom(fit, "eval(x1)")
    with pytest.raises(bad, match="no coefficients"):
        sp.nlcom(fit, "1 + 2")
    with pytest.raises(bad, match="not a valid expression"):
        sp.nlcom(fit, "x1 /")
    with pytest.raises(bad):
        sp.nlcom(fit, "x1.real")


def test_nlcom_delta_method_matches_the_sampling_spread():
    """Known truth: the delta-method se of a ratio agrees with the spread of
    the ratio across independent samples when the denominator is far from
    zero."""
    rng = np.random.default_rng(7)
    est, ses = [], []
    for _ in range(400):
        n = 500
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        y = 1.0 + 1.0 * x1 + 2.0 * x2 + rng.normal(size=n)
        fit = sp.regress("y ~ x1 + x2", data=pd.DataFrame({"y": y, "x1": x1, "x2": x2}))
        out = sp.nlcom(fit, "x1 / x2")
        est.append(out["estimate"])
        ses.append(out["se"])
    assert abs(np.mean(est) - 0.5) < 0.005
    # the sd of an sd estimate over 400 draws is about 3.5% of it
    assert abs(np.mean(ses) / np.std(est, ddof=1) - 1.0) < 0.12


# ------------------------------------------------- serial correlation, ARCH
def test_durbin_alternative_and_arch_lm(G, ts):
    fit = sp.regress("y ~ x1 + x2", data=ts)

    def stat(test, **kw):
        return sp.estat(fit, test, print_results=False, **kw)["statistic"]

    close(stat("durbinalt"), G["dalt1.chi2"])
    close(stat("durbinalt", lags=3), G["dalt3.chi2"])
    close(stat("archlm"), G["arch1.chi2"])
    close(stat("archlm", lags=3), G["arch3.chi2"])
    via = sp.stata("regress y x1 x2\nestat durbinalt, lags(3)", data=ts)
    close(via["statistic"], G["dalt3.chi2"])


def test_arch_lm_detects_arch_and_holds_its_size():
    """Known truth: 5% rejections on white noise, near-certain rejection on
    an ARCH(1) series."""
    rng = np.random.default_rng(11)
    size = power = 0
    reps = 400
    for _ in range(reps):
        e = rng.normal(size=300)
        arch = np.zeros(300)
        for t in range(1, 300):
            arch[t] = np.sqrt(0.2 + 0.7 * arch[t - 1] ** 2) * e[t]
        for series, bucket in ((e, "size"), (arch, "power")):
            fit = sp.regress("y ~ 1", data=pd.DataFrame({"y": series}))
            p = sp.estat(fit, "archlm", print_results=False)["pvalue"]
            if bucket == "size":
                size += p < 0.05
            else:
                power += p < 0.05
    assert abs(size / reps - 0.05) < 0.035
    assert power / reps > 0.95


# ---------------------------------------------------------- unit-root tests
def test_phillips_perron(G, ts):
    one = sp.unitroot(ts, "c1", test="pp")
    close(one.statistic, G["pp1.Zt"])
    close(one.z_rho, G["pp1.Zrho"])
    close(one.pvalue, G["pp1.p"], rtol=1e-6, atol=1e-10)
    assert one.lags == G["pp1.lags"]
    two = sp.unitroot(ts, "c2", test="pp", trend="ct", lags=3)
    close(two.statistic, G["pp2.Zt"])
    close(two.z_rho, G["pp2.Zrho"])
    close(two.pvalue, G["pp2.p"], rtol=1e-6, atol=1e-10)
    three = sp.unitroot(ts, "c3", test="pp", trend="n", lags=2)
    close(three.statistic, G["pp3.Zt"])
    close(three.z_rho, G["pp3.Zrho"])


def test_kpss(G, ts):
    for lag in (0, 3):
        trend = sp.unitroot(ts, "c1", test="kpss", trend="ct", lags=lag)
        close(trend.statistic, G[f"kpss_ct.l{lag}"], rtol=1e-6)
        level = sp.unitroot(ts, "c1", test="kpss", trend="c", lags=lag)
        close(level.statistic, G[f"kpss_c.l{lag}"], rtol=1e-6)
    assert level.null == "stationarity"
    assert level.critical_values == {
        "10%": 0.347,
        "5%": 0.463,
        "2.5%": 0.574,
        "1%": 0.739,
    }
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.unitroot(ts, "c1", test="kpss", trend="n")


def test_kpss_and_pp_point_the_same_way_on_known_processes():
    """Known truth: a random walk fails KPSS and passes PP's null; a
    stationary AR(1) does the opposite."""
    rng = np.random.default_rng(3)
    e = rng.normal(size=400)
    walk = np.cumsum(e)
    ar = np.zeros(400)
    for t in range(1, 400):
        ar[t] = 0.4 * ar[t - 1] + e[t]
    assert sp.unitroot(walk, test="kpss").reject
    assert not sp.unitroot(walk, test="pp").reject
    assert not sp.unitroot(ar, test="kpss").reject
    assert sp.unitroot(ar, test="pp").reject


def test_pperron_and_kpss_translate(ts):
    pp = sp.from_stata("pperron c2, trend lags(3)")
    assert pp["arguments"] == {"y": "c2", "test": "pp", "lags": 3, "trend": "ct"}
    kp = sp.from_stata("kpss c1, maxlag(3) notrend")
    assert kp["arguments"] == {"y": "c1", "test": "kpss", "trend": "c", "lags": 3}
    assert sp.from_stata("kpss c1")["ok"] is False


# -------------------------------------------------------------------- arima
def test_arima_estimates_a_constant_as_stata_does(G, ts):
    fit = sp.arima("y", order=(1, 0, 0), method="innovations_mle", data=ts)
    close(fit.params["const"], G["ar1.b_cons"], rtol=2e-5)
    close(fit.params["ar.L1"], G["ar1.b_ar"], rtol=2e-5)
    close(np.sqrt(fit.params["sigma2"]), G["ar1.sigma"], rtol=2e-5)
    close(fit.log_likelihood, G["ar1.ll"], rtol=1e-8)
    close(fit.se["const"], G["ar1.se_cons"], rtol=2e-3)
    close(fit.se["ar.L1"], G["ar1.se_ar"], rtol=2e-3)


def test_arima_drift_in_a_differenced_series(G, ts):
    fit = sp.arima("c1", order=(1, 1, 0), trend="c", method="innovations_mle", data=ts)
    close(fit.params["drift"], G["ari.b_cons"], rtol=2e-5)
    close(fit.params["ar.L1"], G["ari.b_ar"], rtol=2e-5)
    close(fit.log_likelihood, G["ari.ll"], rtol=1e-8)
    # R's and statsmodels' convention: no constant once differenced
    assert "drift" not in sp.arima("c1", order=(1, 1, 0), data=ts).params.index
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.arima("c1", order=(1, 2, 0), trend="c", data=ts)


def test_arima_default_recovers_the_ar_coefficient_of_a_series_with_a_mean():
    """Known truth: an AR(1) around a mean of 10. Fitted without a constant
    the coefficient is pushed towards one."""
    rng = np.random.default_rng(5)
    y = np.zeros(600)
    for t in range(1, 600):
        y[t] = 0.5 * y[t - 1] + rng.normal()
    y += 10.0
    fit = sp.arima(y, order=(1, 0, 0))
    assert abs(fit.params["ar.L1"] - 0.5) < 0.1
    assert abs(fit.params["const"] - 10.0) < 0.3
    assert sp.arima(y, order=(1, 0, 0), trend="n").params["ar.L1"] > 0.95


def test_arima_and_arch_translate():
    call = sp.from_stata("arima y, arima(1,1,1)")
    assert call["arguments"] == {
        "y": "y", "order": (1, 1, 1), "trend": "c", "method": "innovations_mle",
    }  # fmt: skip
    assert sp.from_stata("arima y, ar(1/2) noconstant")["arguments"]["trend"] == "n"
    assert sp.from_stata("arima y, ma(1 4)")["ok"] is False
    arch = sp.from_stata("arch y, arch(1) garch(1)")
    assert arch["arguments"] == {"y": "y", "p": 1, "q": 1, "vce": "opg"}
    assert sp.from_stata("arch y, arch(1)")["arguments"]["p"] == 0


def test_garch_without_an_arch_term_is_refused():
    rng = np.random.default_rng(0)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="not identified"):
        sp.garch(rng.normal(size=200), p=1, q=0)


# --------------------------------------------------------------------- svar
@pytest.fixture(scope="module")
def var_fit(ts):
    return sp.var(ts, variables=["c1", "c2", "c3"], lags=2)


def test_svar_short_run(G, var_fit):
    res = sp.svar(
        var_fit,
        A=[[1, 0, 0], [NAN, 1, 0], [NAN, NAN, 1]],
        B=[[NAN, 0, 0], [0, NAN, 0], [0, 0, NAN]],
    )
    t = res.table
    close(t.loc["A[2,1]", "estimate"], G["svar.A21"], rtol=1e-6)
    close(t.loc["A[2,1]", "se"], G["svar.se_A21"], rtol=1e-5)
    close(t.loc["A[3,2]", "estimate"], G["svar.A32"], rtol=1e-6)
    close(t.loc["A[3,2]", "se"], G["svar.se_A32"], rtol=1e-5)
    close(t.loc["B[1,1]", "estimate"], G["svar.B11"], rtol=1e-6)
    close(t.loc["B[1,1]", "se"], G["svar.se_B11"], rtol=1e-5)
    close(t.loc["B[3,3]", "estimate"], G["svar.B33"], rtol=1e-6)
    close(res.log_likelihood, G["svar.ll"], rtol=1e-9)
    assert res.overid is None
    irf = res.irf(4).set_index(["shock", "response", "period"])
    close(irf.loc[("shock2", "c3", 2), "irf"], G["svar.sirf_c2_c3_2"], rtol=1e-6)
    fevd = res.fevd(4).set_index(["shock", "response", "period"])
    close(fevd.loc[("shock1", "c2", 4), "fevd"], G["svar.sfevd_c1_c2_4"], rtol=1e-6)


def test_var_fevd(G, var_fit):
    fevd = var_fit.fevd(4).set_index(["shock", "response", "period"])
    close(fevd.loc[("c2", "c3", 3), "fevd"], G["var.fevd_c2_c3_3"], rtol=1e-8)
    assert (fevd.xs(0, level="period")["fevd"] == 0).all()
    shares = fevd.reset_index().query("period == 4").groupby("response")["fevd"].sum()
    assert np.allclose(shares, 1.0)


def test_svar_over_identified(G, var_fit):
    res = sp.svar(var_fit, A=[[1, 0, 0], [0, 1, 0], [NAN, NAN, 1]])
    close(res.table.loc["A[3,2]", "estimate"], G["svaro.A32"], rtol=1e-6)
    close(res.table.loc["A[3,2]", "se"], G["svaro.se_A32"], rtol=1e-5)
    close(res.log_likelihood, G["svaro.ll"], rtol=1e-9)
    close(res.overid["statistic"], G["svaro.chi2"], rtol=1e-6)
    assert res.overid["df"] == 1


def test_svar_long_run(G, var_fit):
    res = sp.svar(var_fit, long_run=[[NAN, 0, 0], [NAN, NAN, 0], [NAN, NAN, NAN]])
    t = res.table
    for name in ("C11", "C21", "C32"):
        key = f"C[{name[1]},{name[2]}]"
        close(t.loc[key, "estimate"], G[f"svarl.{name}"], rtol=1e-6)
        close(t.loc[key, "se"], G[f"svarl.se_{name}"], rtol=1e-5)
    close(t.loc["C[3,3]", "estimate"], G["svarl.C33"], rtol=1e-6)
    # the cumulative response converges to C
    long_run = res.irf(400, cumulative=True)
    last = long_run[long_run["period"] == 400].set_index(["response", "shock"])["irf"]
    assert abs(last[("c1", "shock2")]) < 1e-6 * abs(last[("c1", "shock1")])


def test_svar_recursive_is_the_cholesky_factor(var_fit):
    res = sp.svar(var_fit, B=np.tril(np.full((3, 3), NAN)))
    chol = np.linalg.cholesky(var_fit.sigma_u.to_numpy())
    assert np.allclose(res.impact.to_numpy(), chol, rtol=1e-7, atol=1e-10)


def test_svar_refuses_unidentified_and_ambiguous_requests(var_fit):
    with pytest.raises(sp.exceptions.IdentificationFailure, match="order condition"):
        sp.svar(var_fit, B=np.full((3, 3), NAN))
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="exactly one"):
        sp.svar(var_fit)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="exactly one"):
        sp.svar(var_fit, B=np.eye(3) * NAN, sign={"s": {"c1": "+"}})
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="3 x 3"):
        sp.svar(var_fit, B=np.full((2, 2), NAN))


def test_svar_sign_restrictions_contain_the_true_impact():
    """Known truth: a supply shock (output up, prices down) and a demand
    shock (both up). Every accepted rotation has the stated signs and
    reproduces the reduced-form covariance; the true impact matrix lies
    inside the range of the accepted set."""
    rng = np.random.default_rng(1)
    T = 3000
    truth = np.array([[1.0, 0.5], [-0.6, 0.8]])
    u = rng.normal(size=(T, 2))
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = 0.4 * y[t - 1] + truth @ u[t]
    fit = sp.var(pd.DataFrame(y, columns=["output", "prices"]), lags=1)
    res = sp.svar(
        fit,
        sign={
            "supply": {"output": "+", "prices": "-"},
            "demand": {"output": "+", "prices": "+"},
        },
        n_draws=1500,
        seed=0,
    )
    draws = res._draws
    assert res.n_accepted == 1500 and res.shock_names == ["supply", "demand"]
    assert (draws[:, 0, 0] > 0).all() and (draws[:, 1, 0] < 0).all()
    assert (draws[:, 0, 1] > 0).all() and (draws[:, 1, 1] > 0).all()
    sigma = fit.sigma_u.to_numpy()
    for P in draws[:25]:
        assert np.allclose(P @ P.T, sigma, rtol=1e-9, atol=1e-12)
    lo, hi = draws.min(axis=0), draws.max(axis=0)
    assert ((lo < truth) & (truth < hi)).all()
    band = res.irf(3)
    assert {"irf", "lower", "upper"} <= set(band.columns)
    assert (band["lower"] <= band["irf"]).all() and (band["irf"] <= band["upper"]).all()


def test_svar_sign_restrictions_that_cannot_hold_are_reported():
    rng = np.random.default_rng(2)
    y = rng.normal(size=(300, 2))
    fit = sp.var(pd.DataFrame(y, columns=["a", "b"]), lags=1)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="not a variable"):
        sp.svar(fit, sign={"s": {"zzz": "+"}})
    # three shocks with the same signs cannot be three orthogonal columns
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="shocks for"):
        sp.svar(fit, sign={k: {"a": "+"} for k in ("s1", "s2", "s3")})


# ---------------------------------------------------------- panel, engine
def test_fixed_effects_statistics_with_more_regressors_than_panels():
    """Year dummies on a short list of firms: the between regression of the
    random-effects components does not exist, the fixed-effects statistics
    do."""
    rng = np.random.default_rng(4)
    firms, years = 6, 12
    frame = pd.DataFrame(
        {
            "firm": np.repeat(np.arange(firms), years),
            "year": np.tile(np.arange(years), firms),
        }
    )
    frame["x"] = rng.normal(size=len(frame))
    frame["y"] = (
        frame["x"]
        + frame["firm"] * 0.5
        + frame["year"] * 0.1
        + rng.normal(size=len(frame))
    )
    out = sp.stata("xtset firm year\nxtreg y x i.year, fe", data=frame)
    direct = sp.feols("y ~ x + C(year) | firm", data=frame)
    close(out.params["x"], direct.params["x"])


def test_random_effects_mle_equals_the_mixed_model_fit():
    rng = np.random.default_rng(8)
    n, t = 80, 5
    frame = pd.DataFrame({"id": np.repeat(np.arange(n), t)})
    frame["x"] = rng.normal(size=n * t)
    frame["y"] = (
        1.0
        + 0.5 * frame["x"]
        + np.repeat(rng.normal(size=n), t)
        + rng.normal(size=n * t)
    )
    frame["year"] = np.tile(np.arange(t), n)
    fit = sp.panel(frame, "y ~ x", entity="id", time="year", method="mle")
    mixed = sp.mixed(frame, "y", ["x"], group="id", method="ml")
    close(fit.params["x"], mixed.params["x"], rtol=1e-6)
    close(fit.model_info["ll"], mixed.log_likelihood, rtol=1e-9)
    close(
        fit.model_info["sigma_u"] ** 2,
        mixed.variance_components["var(_cons)"],
        rtol=1e-5,
    )


def test_random_effects_mle_on_the_boundary():
    """No unit effect in the data: the likelihood peaks at sigma_u = 0 and
    the fit is pooled OLS."""
    rng = np.random.default_rng(12)
    n, t = 40, 4
    frame = pd.DataFrame({"id": np.repeat(np.arange(n), t)})
    frame["year"] = np.tile(np.arange(t), n)
    frame["x"] = rng.normal(size=n * t)
    # negative within-panel correlation pushes the estimate to the boundary
    e = rng.normal(size=(n, t))
    frame["y"] = 0.5 * frame["x"] + (e - e.mean(axis=1, keepdims=True)).ravel()
    fit = sp.panel(frame, "y ~ x", entity="id", time="year", method="mle")
    ols = sp.regress("y ~ x", data=frame)
    assert fit.model_info["sigma_u"] == 0.0
    close(fit.params["x"], ols.params["x"], rtol=1e-9)


def test_collinearity_scan_on_a_wide_dummy_design():
    """The scan omits the same columns whether a design has a few dummy
    columns or more than one block of them."""
    rng = np.random.default_rng(6)
    n, groups = 3000, 600
    g = rng.integers(0, groups, size=n)
    frame = pd.DataFrame({"g": g, "x": rng.normal(size=n)})
    frame["dup"] = (frame["g"] == 5).astype(float)  # a level dummy again
    frame["y"] = frame["x"] + rng.normal(size=n)
    fit = sp.regress("y ~ x + C(g) + dup", data=frame)
    omitted = fit.model_info.get("omitted") or fit.data_info.get("omitted") or []
    assert (
        any("dup" in str(o) for o in omitted)
        or "dup" not in fit.params.index
        or (fit.params["dup"] == 0)
    )
    within = sp.feols("y ~ x | g", data=frame)
    close(fit.params["x"], within.params["x"], rtol=1e-8)


# ------------------------- proportions, normality, intervals, association
def test_prtest(G, cs):
    one = sp.prtest(cs, "d", p=0.4)
    close(one.statistic, G["pr1.z"])
    close(one.se, G["pr1.se"])
    close(one.ci[0], G["pr1.lb"])
    close(one.ci[1], G["pr1.ub"])
    close(one.pvalue, G["pr1.p"])
    two = sp.prtest(cs, "d", by="b")
    close(two.statistic, G["pr2.z"])
    close(two.se, G["pr2.se"])  # unpooled: the interval's
    close(two.se_null, G["pr2.se0"])  # pooled: the statistic's
    close(two.ci[0], G["pr2.lb"])
    close(two.ci[1], G["pr2.ub"])
    close(two.pvalue, G["pr2.p"])
    close(sp.prtest(n=50, proportion=0.52, p=0.4).statistic, G["pri.z"])
    via = sp.stata("prtest d, by(b)", data=cs)
    close(via.statistic, G["pr2.z"])


def test_sktest(G, cs):
    out = sp.sktest(cs, ["y", "x1"])
    close(out.loc["y", "p_skew"], G["sk.y.p_skew"], rtol=1e-8)
    close(out.loc["y", "p_kurt"], G["sk.y.p_kurt"], rtol=1e-8)
    close(out.loc["y", "chi2"], G["sk.y.chi2"], rtol=1e-8)
    close(out.loc["y", "p_chi2"], G["sk.y.p_chi2"], rtol=1e-8)
    close(out.loc["x1", "p_skew"], G["sk.x1.p_skew"], rtol=1e-8)
    close(out.loc["x1", "chi2"], G["sk.x1.chi2"], rtol=1e-8)
    close(out.loc["x1", "p_chi2"], G["sk.x1.p_chi2"], rtol=1e-8)
    plain = sp.sktest(cs, "y", adjust=False)
    close(plain.loc["y", "chi2"], G["skna.y.chi2"], rtol=1e-8)
    close(plain.loc["y", "p_chi2"], G["skna.y.p_chi2"], rtol=1e-8)
    via = sp.stata("sktest y, noadjust", data=cs)
    close(via.loc["y", "chi2"], G["skna.y.chi2"], rtol=1e-8)


def test_swilk(G, cs):
    out = sp.swilk(cs, "y")
    # W comes from scipy's AS R94 and agrees with Stata to 1e-8; z and p
    # magnify that difference
    close(out.loc["y", "W"], G["sw.W"], rtol=1e-8)
    close(out.loc["y", "V"], G["sw.V"], rtol=1e-6)
    close(out.loc["y", "z"], G["sw.z"], rtol=1e-6)
    close(out.loc["y", "pvalue"], G["sw.p"], rtol=1e-5)


def test_normality_tests_hold_their_size_and_find_skewness():
    """Known truth: about 5% rejections on normal samples of 60, near
    certain rejection on exponential ones."""
    rng = np.random.default_rng(2026)
    reps = 600
    normal = pd.DataFrame(rng.normal(size=(60, reps)))
    skewed = pd.DataFrame(rng.exponential(size=(60, reps)))
    for test, col in ((sp.sktest, "p_chi2"), (sp.swilk, "pvalue")):
        size = float((test(normal)[col] < 0.05).mean())
        power = float((test(skewed)[col] < 0.05).mean())
        assert abs(size - 0.05) < 0.03, (test.__name__, size)
        assert power > 0.95, (test.__name__, power)


def test_confidence_intervals(G, cs):
    mean = sp.ci(cs, "y")
    close(mean.loc["y", "mean"], G["cim.mean"])
    close(mean.loc["y", "se"], G["cim.se"])
    close(mean.loc["y", "ci_lower"], G["cim.lb"])
    close(mean.loc["y", "ci_upper"], G["cim.ub"])
    close(sp.ci(cs, "y", alpha=0.10).loc["y", "ci_lower"], G["cim90.lb"])
    var = sp.ci(cs, "y", stat="variances")
    close(var.loc["y", "variance"], G["civ.var"])
    close(var.loc["y", "ci_lower"], G["civ.lb"])
    close(var.loc["y", "ci_upper"], G["civ.ub"])
    sd = sp.ci(cs, "y", stat="sd")
    close(sd.loc["y", "ci_lower"], G["cis.lb"])
    close(sd.loc["y", "ci_upper"], G["cis.ub"])
    for method in ("exact", "wald", "wilson", "agresti", "jeffreys"):
        prop = sp.ci(cs, "d", stat="proportions", method=method)
        close(prop.loc["d", "ci_lower"], G[f"cip.{method}.lb"], rtol=1e-8)
        close(prop.loc["d", "ci_upper"], G[f"cip.{method}.ub"], rtol=1e-8)
    via = sp.stata("ci proportions d, wilson", data=cs)
    close(via.loc["d", "ci_lower"], G["cip.wilson.lb"], rtol=1e-8)
    # the sd interval is the one sp.sdtest reports
    close(sd.loc["y", "ci_lower"], sp.sdtest(cs, "y", sd0=2).ci[0])


def test_ci_and_prtest_refusals(cs):
    bad = sp.exceptions.MethodIncompatibility
    with pytest.raises(bad, match="0 / 1"):
        sp.ci(cs, "y", stat="proportions")
    with pytest.raises(bad, match="stat="):
        sp.ci(cs, "y", stat="median")
    with pytest.raises(bad, match="method="):
        sp.ci(cs, "d", stat="proportions", method="score")
    with pytest.raises(bad, match="0 / 1"):
        sp.prtest(cs, "y")
    with pytest.raises(bad, match="exactly two groups"):
        sp.prtest(cs, "d", by="g")
    with pytest.raises(bad, match="null proportion"):
        sp.prtest(cs, "d", p=1.0)
    with pytest.raises(bad, match="both n= and proportion="):
        sp.prtest(n=50)
    with pytest.raises(sp.exceptions.DataInsufficient):
        sp.sktest(cs.head(5), "y")
    with pytest.raises(sp.exceptions.DataInsufficient):
        sp.swilk(cs.head(3), "y")


def test_tabulate_tests_of_association(G, cs):
    two = sp.stata("tabulate b d, chi2 exact lrchi2 V", data=cs).attrs["test"]
    close(two["chi2"], G["tab2.chi2"])
    close(two["pvalue"], G["tab2.p"])
    close(two["chi2_lr"], G["tab2.chi2_lr"])
    close(two["cramers_v"], G["tab2.V"])
    close(two["fisher_pvalue"], G["tab2.p_exact"], rtol=1e-8)
    close(two["fisher_pvalue_1sided"], G["tab2.p1_exact"], rtol=1e-8)
    many = sp.stata("tab g d, chi2 lrchi2 V", data=cs).attrs["test"]
    close(many["chi2"], G["tabk.chi2"])
    close(many["pvalue"], G["tabk.p"])
    close(many["chi2_lr"], G["tabk.chi2_lr"])
    close(many["pvalue_lr"], G["tabk.p_lr"])
    close(many["cramers_v"], G["tabk.V"])


def test_tab_reports_pearson_without_a_continuity_correction(G, cs):
    """sp.tab labels its statistic Pearson chi2; on a 2 x 2 table that is
    not the Yates-corrected value scipy returns by default."""
    table = sp.tab(cs, "b", "d", output="dataframe")
    close(table.attrs["test"]["chi2"], G["tab2.chi2"])
    assert f"{G['tab2.chi2']:.4f}" in sp.tab(cs, "b", "d")
    # for a 2 x 2 table Pearson's chi2 is the square of the two-sample z
    close(table.attrs["test"]["chi2"], sp.prtest(cs, "d", by="b").statistic ** 2)


def test_svar_runs_through_the_translator(G, ts):
    short = sp.stata(
        """
        tsset t
        matrix A = (1,0,0 \\ .,1,0 \\ .,.,1)
        matrix B = (.,0,0 \\ 0,.,0 \\ 0,0,.)
        svar c1 c2 c3, lags(1/2) aeq(A) beq(B)
        """,
        data=ts,
    )
    close(short.table.loc["A[2,1]", "estimate"], G["svar.A21"], rtol=1e-6)
    close(short.table.loc["B[1,1]", "se"], G["svar.se_B11"], rtol=1e-5)
    close(short.log_likelihood, G["svar.ll"], rtol=1e-9)
    long = sp.stata(
        """
        tsset t
        matrix C = (.,0,0 \\ .,.,0 \\ .,.,.)
        svar c1 c2 c3, lags(1/2) lreq(C)
        """,
        data=ts,
    )
    close(long.table.loc["C[2,1]", "estimate"], G["svarl.C21"], rtol=1e-6)
    with pytest.raises(
        sp.exceptions.MethodIncompatibility, match="has not been defined"
    ):
        sp.stata("tsset t\nsvar c1 c2, lags(1/2) aeq(Q)", data=ts)
