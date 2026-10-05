"""What the October 2026 pass over Wooldridge's *Introductory Econometrics*
changed, on synthetic data.

Each test pins one finding of ``docs/dev/2026-10-05-wooldridge-review.md``
against statsmodels / linearmodels or an identity. The same findings are
checked against Stata 18 on the book's datasets in
``tests/test_wooldridge_examples.py``.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def cross():
    rng = np.random.default_rng(20261005)
    n = 400
    df = pd.DataFrame(
        {
            "x": rng.uniform(1.0, 4.0, n),
            "w": rng.normal(size=n),
            "z": rng.normal(size=n),
            "g": rng.choice(["a", "b", "c"], n),
        }
    )
    df["x"] = df["x"] + 0.4 * df["z"].clip(-1.5, 1.5)
    df["y"] = np.exp(
        0.3 + 0.5 * np.log(df["x"]) + 0.2 * df["w"] + 0.3 * rng.normal(size=n)
    )
    df["d"] = (df["w"] + rng.normal(size=n) > 0).astype(int)
    df["c"] = rng.poisson(np.exp(0.2 + 0.3 * df["w"]))
    df["pop"] = rng.integers(1, 6, n).astype(float)
    return df


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(7)
    units, periods = 60, 5
    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(units), periods),
            "year": np.tile(np.arange(2001, 2001 + periods), units),
        }
    )
    alpha = rng.normal(size=units)[df["id"]]
    df["educ"] = rng.integers(8, 17, units)[df["id"]].astype(float)
    df["x"] = rng.normal(size=len(df)) + 0.5 * alpha
    df["m"] = (rng.normal(size=len(df)) + alpha > 0).astype(float)
    df["y"] = np.exp(
        0.1 * df["educ"]
        + 0.4 * df["x"]
        + 0.2 * df["m"]
        + alpha
        + 0.3 * rng.normal(size=len(df))
    )
    return df


# ------------------------------------------------------------------ formulas
def test_bare_log_and_sqrt_read_as_in_r(cross):
    ref = smf.ols("np.log(y) ~ np.log(x) + np.sqrt(x) + w", cross).fit()
    fit = sp.regress("log(y) ~ log(x) + sqrt(x) + w", cross)
    np.testing.assert_allclose(fit.params.values, ref.params.values, rtol=1e-10)
    assert list(fit.params.index) == ["Intercept", "log(x)", "sqrt(x)", "w"]


def test_unknown_name_in_formula_says_so(cross):
    with pytest.raises(MethodIncompatibility, match="wagee"):
        sp.regress("log(wagee) ~ x", cross)


def test_ivreg_takes_transformed_outcome_and_endogenous_regressor(cross):
    from linearmodels.iv import IV2SLS

    ref = IV2SLS.from_formula(
        "np.log(y) ~ 1 + w + I(w**2) + [np.log(x) ~ z + I(z**2)]", cross
    ).fit(cov_type="unadjusted", debiased=True)
    fit = sp.ivreg("log(y) ~ w + I(w**2) + (log(x) ~ z + I(z**2))", cross)
    np.testing.assert_allclose(
        np.sort(fit.params.values), np.sort(ref.params.values), rtol=1e-8
    )
    np.testing.assert_allclose(
        np.sort(fit.std_errors.values), np.sort(ref.std_errors.values), rtol=1e-8
    )


def test_ivreg_refuses_an_endogenous_term_of_several_columns(cross):
    with pytest.raises(MethodIncompatibility, match="single variable"):
        sp.ivreg("y ~ w + (C(g) ~ z)", cross)


def test_qreg_formula_builds_transformed_terms(cross):
    built = cross.assign(lx=np.log(cross["x"]), w2=cross["w"] ** 2)
    ref = sp.qreg(built, formula="y ~ lx + w2")
    fit = sp.qreg(cross, formula="y ~ np.log(x) + I(w**2)")
    np.testing.assert_allclose(fit.params.values, ref.params.values, rtol=1e-10)
    np.testing.assert_allclose(fit.std_errors.values, ref.std_errors.values, rtol=1e-10)


# --------------------------------------------------------------------- panel
def test_panel_takes_a_transformed_outcome(panel):
    built = panel.assign(ly=np.log(panel["y"]))
    for method in ("fe", "re", "fd"):
        ref = sp.panel(built, "ly ~ x + m", entity="id", time="year", method=method)
        fit = sp.panel(
            panel, "np.log(y) ~ x + m", entity="id", time="year", method=method
        )
        np.testing.assert_allclose(
            fit.params[["x", "m"]].values, ref.params[["x", "m"]].values, rtol=1e-12
        )


def test_first_differences_with_the_time_variable_as_a_regressor(panel):
    from linearmodels.panel import FirstDifferenceOLS

    built = panel.assign(ly=np.log(panel["y"]))
    ref = FirstDifferenceOLS.from_formula(
        "ly ~ year + x + m", built.set_index(["id", "year"], drop=False)
    ).fit()
    fit = sp.panel(built, "ly ~ year + x + m", entity="id", time="year", method="fd")
    np.testing.assert_allclose(
        fit.params.reindex(ref.params.index).values, ref.params.values, rtol=1e-10
    )


def test_fixed_effects_omit_absorbed_regressors_and_say_so(panel):
    built = panel.assign(ly=np.log(panel["y"]))
    ref = sp.panel(built, "ly ~ x + m", entity="id", time="year", method="fe")
    with pytest.warns(UserWarning, match="educ.*omitted"):
        fit = sp.panel(built, "ly ~ x + educ + m", entity="id", time="year")
    assert fit.model_info["omitted"] == ["educ"]
    assert list(fit.params.index) == ["x", "m"]
    np.testing.assert_allclose(fit.params.values, ref.params.values, rtol=1e-12)
    np.testing.assert_allclose(fit.std_errors.values, ref.std_errors.values, rtol=1e-12)


def test_fixed_effects_without_absorbed_regressors_do_not_warn(panel):
    built = panel.assign(ly=np.log(panel["y"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fit = sp.panel(built, "ly ~ x + m", entity="id", time="year")
    assert "omitted" not in fit.model_info


def test_mundlak_keeps_only_the_means_that_add_rank(panel):
    built = panel.assign(ly=np.log(panel["y"]))
    fit = sp.panel(
        built, "ly ~ x + m + educ + C(year)", entity="id", time="year", method="mundlak"
    )
    # a time-invariant regressor is its own mean, and the means of period
    # dummies are constants in a balanced panel
    assert fit.model_info["cre_means"] == ["x", "m"]
    assert "educ" in fit.model_info["cre_means_omitted"]
    # Mundlak (1978): the coefficients on the time-varying regressors are
    # the within estimates
    fe = sp.panel(built, "ly ~ x + m + C(year)", entity="id", time="year")
    np.testing.assert_allclose(
        fit.params[["x", "m"]].values, fe.params[["x", "m"]].values, rtol=1e-8
    )
    assert fit.diagnostics["CRE Wald df"] == 2


# --------------------------------------------------------------- diagnostics
@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(3)
    n = 250
    x = rng.normal(size=n)
    e = np.zeros(n)
    shock = rng.normal(size=n)
    for t in range(1, n):
        e[t] = 0.4 * e[t - 1] + shock[t] * np.sqrt(0.5 + 0.4 * e[t - 1] ** 2)
    return pd.DataFrame({"x": x, "y": 1.0 + 0.5 * x + e})


def test_dfbetas_and_studentized_residuals(cross):
    ref = smf.ols("y ~ x + w", cross).fit().get_influence()
    out = sp.estat(sp.regress("y ~ x + w", cross), "leverage", print_results=False)
    np.testing.assert_allclose(out["dfbetas"], ref.dfbetas, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(out["rstudent"], ref.resid_studentized_external)
    np.testing.assert_allclose(out["rstandard"], ref.resid_studentized_internal)


def test_arch_lm_test(series):
    ref_fit = smf.ols("y ~ x", series).fit()
    fit = sp.regress("y ~ x", series)
    for lags in (1, 3):
        ref = sm.stats.diagnostic.het_arch(ref_fit.resid, nlags=lags)
        lm = sp.estat(fit, "archlm", lags=lags, print_results=False)
        f = sp.estat(fit, "archlm", lags=lags, version="fstat", print_results=False)
        np.testing.assert_allclose([lm["statistic"], lm["pvalue"]], ref[:2])
        np.testing.assert_allclose([f["statistic"], f["pvalue"]], ref[2:])
        assert lm["df"] == lags


def test_breusch_godfrey_f_form_and_durbin_alternative(series):
    # the auxiliary regression, written out: residuals on the regressors and
    # two lags of themselves, the missing lags set to zero
    ref_fit = smf.ols("y ~ x", series).fit()
    u = ref_fit.resid.to_numpy()
    n = len(u)
    lagged = np.column_stack([np.r_[0.0, u[:-1]], np.r_[0.0, 0.0, u[:-2]]])
    aux = sm.OLS(u, np.column_stack([np.ones(n), series["x"], lagged])).fit()
    f_ref = float(aux.f_test(np.eye(2, 4, 2)).statistic)
    fit = sp.regress("y ~ x", series)
    f = sp.estat(fit, "bgodfrey", lags=2, version="fstat", print_results=False)
    np.testing.assert_allclose(f["statistic"], f_ref, rtol=1e-10)
    assert (f["df1"], f["df2"]) == (2, n - 4)
    alt = sp.estat(fit, "durbinalt", lags=2, print_results=False)
    np.testing.assert_allclose(alt["statistic"], 2 * f_ref, rtol=1e-10)
    assert alt["df"] == 2
    alt_f = sp.estat(fit, "durbinalt", lags=2, version="fstat", print_results=False)
    np.testing.assert_allclose(alt_f["statistic"], f_ref, rtol=1e-10)
    # the default is unchanged: N R-squared
    lm = sp.estat(fit, "bgodfrey", lags=2, print_results=False)
    np.testing.assert_allclose(lm["statistic"], n * aux.rsquared, rtol=1e-10)


def test_white_special_form(cross):
    ref_fit = smf.ols("y ~ x + w + z", cross).fit()
    aux = pd.DataFrame(
        {"c": 1.0, "f": ref_fit.fittedvalues, "f2": ref_fit.fittedvalues**2}
    )
    ref = sm.stats.diagnostic.het_breuschpagan(ref_fit.resid, aux)
    out = sp.estat(
        sp.regress("y ~ x + w + z", cross),
        "white",
        variables="fitted",
        print_results=False,
    )
    np.testing.assert_allclose([out["statistic"], out["pvalue"]], ref[:2], rtol=1e-9)
    assert out["df"] == 2


def test_iv_tests_state_their_conclusion(cross):
    fit = sp.ivreg("y ~ w + (x ~ z + I(z**2))", cross)
    for name in ("endogenous", "overid"):
        text = sp.estat(fit, name, print_results=False)["interpretation"]
        assert "not available" not in text


# ------------------------------------------------------------ test / lincom
def test_test_takes_names_with_operators_and_lists(cross):
    ref = smf.ols("y ~ x + I(x**2) + np.log(x) + w", cross).fit()
    fit = sp.regress("y ~ x + I(x**2) + np.log(x) + w", cross)
    want = ref.f_test(["I(x ** 2) = 0", "np.log(x) = 0"]).statistic
    for hypothesis in (
        "I(x ** 2) np.log(x)",
        ["I(x ** 2) = 0", "np.log(x) = 0"],
        "(I(x ** 2) = 0) (np.log(x) = 0)",
        "I(x ** 2) = np.log(x) = 0",
    ):
        np.testing.assert_allclose(
            sp.test(fit, hypothesis)["statistic"], want, rtol=1e-9
        )
    combo = sp.lincom(fit, "x + 4*I(x ** 2)")
    t = ref.t_test("x + 4*I(x ** 2) = 0")
    np.testing.assert_allclose(
        [combo["estimate"], combo["se"]], [t.effect[0], t.sd[0][0]], rtol=1e-9
    )
    with pytest.raises(MethodIncompatibility, match=r"Available terms.*I\(x \*\* 2\)"):
        sp.test(fit, "I(x ** 2) = nope")


# -------------------------------------------------------------------- lrtest
def test_lrtest_of_maximum_likelihood_regressions(cross):
    for fitter, ref_fitter in ((sp.logit, smf.logit), (sp.probit, smf.probit)):
        full = fitter("d ~ w + x + z", cross)
        restricted = fitter("d ~ w", cross)
        ref = 2 * (
            ref_fitter("d ~ w + x + z", cross).fit(disp=0).llf
            - ref_fitter("d ~ w", cross).fit(disp=0).llf
        )
        out = sp.lrtest(restricted, full)
        np.testing.assert_allclose(out.chi2, ref, rtol=1e-8)
        assert out.df == 2 and not out.boundary_corrected
    with pytest.raises(MethodIncompatibility, match="restricted model first"):
        sp.lrtest(full, restricted)
    with pytest.raises(MethodIncompatibility, match="different samples"):
        sp.lrtest(sp.probit("d ~ w", cross.iloc[:300]), full)
    with pytest.raises(MethodIncompatibility, match="different models"):
        sp.lrtest(sp.logit("d ~ w", cross), full)


# ------------------------------------------------------------------- predict
def test_predict_rebuilds_the_design_and_gives_intervals(cross):
    formula = "log(y) ~ log(x) + I(w**2) + C(g) + w:z"
    ref = smf.ols(formula.replace("log(", "np.log("), cross).fit()
    fit = sp.regress(formula, cross)
    new = pd.DataFrame(
        {"x": [1.5, 3.0], "w": [0.2, -1.0], "z": [0.0, 1.0], "g": ["c", "a"]},
        index=["p", "q"],
    )
    frame = ref.get_prediction(new).summary_frame(alpha=0.1)
    np.testing.assert_allclose(fit.predict(new), ref.predict(new), rtol=1e-10)
    ci = fit.predict(new, what="confidence", alpha=0.1)
    assert list(ci.index) == ["p", "q"]
    np.testing.assert_allclose(
        ci[["yhat", "se", "lower", "upper"]].values,
        frame[["mean", "mean_se", "mean_ci_lower", "mean_ci_upper"]].values,
        rtol=1e-9,
    )
    pi = fit.predict(new, what="prediction", alpha=0.1)
    np.testing.assert_allclose(
        pi[["lower", "upper"]].values,
        frame[["obs_ci_lower", "obs_ci_upper"]].values,
        rtol=1e-9,
    )
    with pytest.raises(MethodIncompatibility, match="predict"):
        fit.predict(new.drop(columns="g"))


def test_prediction_interval_is_refused_after_a_weighted_fit(cross):
    fit = sp.regress("y ~ x", cross, weights="pop")
    new = pd.DataFrame({"x": [2.0]})
    assert fit.predict(new, what="confidence").shape == (1, 4)
    with pytest.raises(MethodIncompatibility, match="weight"):
        fit.predict(new, what="prediction")


def test_predict_of_a_model_with_a_link_is_on_the_scale_of_the_outcome(cross):
    new = cross.iloc[:6]
    ref = smf.glm("c ~ w + I(w**2)", cross, family=sm.families.Poisson()).fit()
    frame = ref.get_prediction(new).summary_frame()
    for fit in (
        sp.glm("c ~ w + I(w**2)", cross, family="poisson"),
        sp.poisson("c ~ w + I(w**2)", cross),
    ):
        mean = fit.predict(new)
        np.testing.assert_allclose(mean, ref.predict(new), rtol=1e-7)
        # out of sample and in sample are the same quantity
        np.testing.assert_allclose(mean, np.asarray(fit.predict())[:6], rtol=1e-10)
        np.testing.assert_allclose(
            fit.predict(new, what="link"), np.log(ref.predict(new)), rtol=1e-7
        )
        ci = fit.predict(new, what="confidence")
        np.testing.assert_allclose(
            ci[["yhat", "se", "lower", "upper"]].values,
            frame[["mean", "mean_se", "mean_ci_lower", "mean_ci_upper"]].values,
            rtol=1e-5,
        )
        with pytest.raises(MethodIncompatibility, match="linear models"):
            fit.predict(new, what="prediction")


def test_predict_applies_the_exposure_of_the_new_rows(cross):
    ref = sm.GLM.from_formula(
        "c ~ w", cross, family=sm.families.Poisson(), exposure=cross["pop"]
    ).fit()
    fit = sp.poisson("c ~ w", cross, exposure="pop")
    new = cross.iloc[:5]
    np.testing.assert_allclose(
        fit.predict(new), ref.predict(new, exposure=new["pop"]), rtol=1e-7
    )
    with pytest.raises(MethodIncompatibility, match="exposure"):
        fit.predict(new.drop(columns="pop"))


def test_logit_and_probit_predict_from_a_dataframe(cross):
    new = pd.DataFrame({"w": [-1.0, 0.0, 2.0], "x": [1.0, 2.0, 3.0]})
    for fitter, ref_fitter in ((sp.logit, smf.logit), (sp.probit, smf.probit)):
        ref = ref_fitter("d ~ w + I(w**2) + log(x)".replace("log(", "np.log("), cross)
        ref = ref.fit(disp=0)
        fit = fitter("d ~ w + I(w**2) + log(x)", cross)
        np.testing.assert_allclose(fit.predict(new), ref.predict(new), rtol=1e-7)
        np.testing.assert_allclose(fit.predict(data=new), ref.predict(new), rtol=1e-7)
        np.testing.assert_allclose(fit.predict(), ref.predict(), rtol=1e-7)
        assert set(fit.predict(new, pred_type="class")) <= {0.0, 1.0}
        clone = pickle.loads(pickle.dumps(fit))
        np.testing.assert_allclose(clone.predict(new), fit.predict(new))
        with pytest.raises(MethodIncompatibility, match="coefficients"):
            fit.predict(np.ones((3, 2)))


# ----------------------------------------------------------------- glm scale
def test_glm_scale_is_the_quasi_likelihood_covariance(cross):
    ref_model = smf.glm("c ~ w + x", cross, family=sm.families.Poisson())
    plain = sp.glm("c ~ w + x", cross, family="poisson")
    np.testing.assert_allclose(plain.std_errors, ref_model.fit().bse, rtol=1e-7)
    assert plain.model_info["vcov_scale"] == 1.0
    pearson = sp.glm("c ~ w + x", cross, family="poisson", scale="x2")
    ref = ref_model.fit(scale="X2")
    np.testing.assert_allclose(pearson.std_errors, ref.bse, rtol=1e-7)
    np.testing.assert_allclose(pearson.params, plain.params, rtol=1e-12)
    np.testing.assert_allclose(pearson.model_info["vcov_scale"], ref.scale, rtol=1e-8)
    deviance = sp.glm("c ~ w + x", cross, family="poisson", scale="dev")
    np.testing.assert_allclose(
        deviance.std_errors, ref_model.fit(scale="dev").bse, rtol=1e-7
    )
    fixed = sp.glm("c ~ w + x", cross, family="poisson", scale=4.0)
    np.testing.assert_allclose(fixed.std_errors, 2.0 * plain.std_errors, rtol=1e-12)
    with pytest.raises(MethodIncompatibility, match="robust or clustered"):
        sp.glm("c ~ w + x", cross, family="poisson", scale="x2", robust="hc1")
    with pytest.raises(MethodIncompatibility, match="scale"):
        sp.glm("c ~ w + x", cross, family="poisson", scale="pearsonn")


# -------------------------------------------------------------- translations
@pytest.mark.parametrize(
    "command, code",
    [
        (
            "estat archlm, lags(2)",
            "sp.estat(result, test='archlm', lags=2, print_results=False)",
        ),
        (
            "estat durbinalt, lags(3) small",
            "sp.estat(result, test='durbinalt', lags=3, version='fstat', "
            "print_results=False)",
        ),
        (
            "heckman y x, select(d = x z)",
            "sp.heckman(data=df, y='y', x=['x'], select='d', z=['x', 'z'], "
            "method='ml')",
        ),
        (
            "heckman y x, sel(d = x z) two",
            "sp.heckman(data=df, y='y', x=['x'], select='d', z=['x', 'z'], "
            "method='twostep')",
        ),
        (
            "truncreg y x w, ll(0)",
            "sp.truncreg(data=df, y='y', x=['x', 'w'], ll=0.0)",
        ),
        (
            "glm c w x, family(poisson) scale(x2)",
            "sp.glm('c ~ w + x', data=df, family='poisson', scale='x2')",
        ),
        (
            "glm d w, f(bin) l(probit) vce(robust)",
            "sp.glm('d ~ w', data=df, family='binomial', link='probit', "
            "robust='robust')",
        ),
    ],
)
def test_stata_translations(command, code):
    out = sp.from_stata(command)
    assert out["python_code"] == code
    assert not out.get("untranslated_options")


def test_glm_translation_refuses_what_it_cannot_carry():
    for command in (
        "glm y x, family(nbinomial 2)",
        "glm y x, family(poisson) link(power 2)",
        "glm y x, family(poisson) scale(x2) vce(robust)",
    ):
        assert sp.from_stata(command).get("python_code") is None


def test_translated_glm_runs(cross):
    ref = smf.glm("c ~ w + x", cross, family=sm.families.Poisson()).fit(scale="X2")
    fit = sp.stata("glm c w x, family(poisson) scale(x2)", data=cross)
    np.testing.assert_allclose(fit.std_errors, ref.bse, rtol=1e-7)


# ------------------------------------------------------- second round items
def test_tobit_and_truncreg_take_a_formula(cross):
    built = cross.assign(lx=np.log(cross["x"]), w2=cross["w"] ** 2)
    built = built.rename(columns={"lx": "log(x)", "w2": "I(w ** 2)"})
    censored = built.assign(yc=built["y"].clip(lower=1.5))
    ref = sp.tobit(censored, "yc", ["log(x)", "I(w ** 2)"], ll=1.5)
    fit = sp.tobit(
        cross.assign(yc=censored["yc"]), formula="yc ~ log(x) + I(w**2)", ll=1.5
    )
    np.testing.assert_allclose(fit.params.values, ref.params.values, rtol=1e-10)
    np.testing.assert_allclose(fit.std_errors.values, ref.std_errors.values, rtol=1e-10)
    first = sp.tobit("yc ~ log(x) + I(w**2)", cross.assign(yc=censored["yc"]), ll=1.5)
    np.testing.assert_allclose(first.params.values, ref.params.values, rtol=1e-10)

    kept = built[built["y"] > 1.2]
    ref_t = sp.truncreg(kept, y="y", x=["log(x)", "I(w ** 2)"], ll=1.2)
    fit_t = sp.truncreg(cross[cross["y"] > 1.2], formula="y ~ log(x) + I(w**2)", ll=1.2)
    np.testing.assert_allclose(fit_t.params.values, ref_t.params.values, rtol=1e-8)

    with pytest.raises(MethodIncompatibility, match="not both"):
        sp.tobit(cross, "y", ["x"], formula="y ~ x")
    with pytest.raises(MethodIncompatibility, match="needed"):
        sp.tobit(cross)
    with pytest.raises(ValueError, match="not both"):
        sp.truncreg(cross, y="y", x=["x"], formula="y ~ x")


def test_survreg_takes_the_data_first(cross):
    data = cross.assign(t=cross["y"], dead=cross["d"])
    ref = sp.survreg("t ~ w + z", data=data, event="dead", dist="lognormal")
    for fit in (
        sp.survreg(data, duration="t", event="dead", x=["w", "z"], dist="lognormal"),
        sp.survreg(data, "t ~ w + z", event="dead", dist="lognormal"),
    ):
        np.testing.assert_allclose(fit.params.values, ref.params.values, rtol=1e-10)


def test_standardized_coefficients(cross):
    # the regression of the standardized outcome on the standardized
    # regressors has the betas as its slopes
    cols = ["y", "x", "w", "z"]
    z = (cross[cols] - cross[cols].mean()) / cross[cols].std()
    ref = smf.ols("y ~ 0 + x + w + z", z).fit().params
    fit = sp.regress("y ~ x + w + z", cross)
    out = sp.estat(fit, "beta", print_results=False)
    assert list(out["beta_table"]["variable"]) == ["x", "w", "z"]
    np.testing.assert_allclose(
        [out["beta"][name] for name in ("x", "w", "z")], ref.values, rtol=1e-10
    )
    np.testing.assert_allclose(
        out["beta_table"]["coef"].values, fit.params[["x", "w", "z"]].values
    )
    sp.estat(fit, "beta")  # prints
    weighted = sp.regress("y ~ x", cross, weights="pop")
    with pytest.raises(MethodIncompatibility, match="weighted"):
        sp.estat(weighted, "beta", print_results=False)
