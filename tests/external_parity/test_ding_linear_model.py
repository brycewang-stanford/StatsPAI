"""Peng Ding, *Linear Model and Extensions* (2024), chapter by chapter.

The book's replication files are 24 R programs that run ``lm``, ``glm``
and their relatives on some twenty datasets. The answer key is
``data/ding_linear_model_R.json``, made by rerunning the deterministic,
real-data part of those programs (``ding_linear_model_reference.R`` next to
this file). Here every number is recomputed with the ``sp.*`` function a
user would reach for.

Neither the programs nor the data are redistributed. Point
``STATSPAI_DING_LM_DIR`` at the unzipped dataverse folder, run the R script
once (it adds a ``_statspai`` subfolder with the datasets that ship inside
R packages and the stored simulation draws), then

    STATSPAI_DING_LM_DIR=/path/to/dataverse_files \\
        pytest tests/external_parity/test_ding_linear_model.py

It is skipped otherwise. What the pass found is in
``docs/dev/2026-10-05-ding-linear-model-review.md``; the same functions are
tested on a committed synthetic file in
``tests/reference_parity/test_linear_model_extensions_parity.py``.

Tolerances. ``EXACT`` (1e-8 relative) for closed forms and for convex
problems both sides solve to machine precision. ``GLM`` (1e-6) for anything
R reaches by iteration at a tolerance of its own. ``OPTIM`` (2e-3) where the
R routine is a general-purpose optimiser stopped early (``nnet::multinom``,
``MASS::polr``, ``pscl::zeroinfl`` all call ``optim``): the test then also
checks that our log likelihood is at least as high as R's.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

ROOT = os.environ.get("STATSPAI_DING_LM_DIR")
pytestmark = pytest.mark.skipif(
    not ROOT or not (Path(ROOT) / "_statspai" / "BostonHousing.csv").is_file(),
    reason="set STATSPAI_DING_LM_DIR to the book's dataverse folder and run "
    "ding_linear_model_reference.R once",
)

EXACT = 1e-8
GLM = 1e-6
OPTIM = 2e-3


@pytest.fixture(scope="module")
def R():
    path = Path(__file__).parent / "data" / "ding_linear_model_R.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _p(name: str) -> Path:
    return Path(ROOT) / name


def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(_p(name), sep=r"\s+")


def _rhs(df: pd.DataFrame, y: str) -> str:
    return " + ".join(c for c in df.columns if c != y)


def close(ours, ref, rtol):
    np.testing.assert_allclose(
        np.asarray(ours, dtype=float).ravel(),
        np.asarray(ref, dtype=float).ravel(),
        rtol=rtol,
        atol=1e-10,
    )


@pytest.fixture(scope="module")
def lalonde():
    return _table("lalonde.txt")


@pytest.fixture(scope="module")
def boston():
    return pd.read_csv(_p("_statspai/BostonHousing.csv"))


@pytest.fixture(scope="module")
def flu():
    return _table("fludata.txt").drop(columns="receive")


@pytest.fixture(scope="module")
def gym():
    return pd.read_stata(_p("gym_treatment_exp_weekly.dta"))


GYM = "weekly_visit ~ incentive_commit + incentive + target + member_gym_pre"


# ------------------------------------------------------------ chapters 1-11


def test_ch01_ols_and_logit(R):
    prostate = _table("prostate.txt")
    fit = sp.regress(
        "lpsa ~ lcavol + lweight + age + lbph + svi + lcp + gleason + pgg45", prostate
    )
    close(fit.params, R["ch1_prostate"]["est"], EXACT)
    close(fit.std_errors, R["ch1_prostate"]["se"], EXACT)
    mroz = _table("mroz.txt")
    logit = sp.logit(
        "inlf ~ kidslt6 + kidsge6 + age + educ + hushrs + husage + huseduc "
        "+ huswage + unem + city",
        mroz,
        tol=1e-13,
    )
    close(logit.params, R["ch1_mroz"]["est"], GLM)
    close(logit.std_errors, R["ch1_mroz"]["se"], GLM)
    assert logit.diagnostics["AIC"] == pytest.approx(R["ch1_mroz"]["aic"], rel=1e-10)


def test_ch05_intervals_and_joint_test(R, lalonde):
    galton = _table("GaltonFamilies.txt")
    fit = sp.regress("childHeight ~ midparentHeight", galton)
    new = pd.DataFrame({"midparentHeight": np.arange(60, 80.01, 0.5)})
    cols = ["yhat", "lower", "upper"]
    close(fit.predict(new, what="confidence")[cols], R["ch5_galton"]["ci"], EXACT)
    close(fit.predict(new, what="prediction")[cols], R["ch5_galton"]["pi"], EXACT)

    key = R["ch5_lalonde"]
    full = sp.regress("re78 ~ " + _rhs(lalonde, "re78"), lalonde)
    nulls = [f"{v}=0" for v in lalonde.columns if v not in ("re78", "treat")]
    test = sp.test(full, nulls)
    assert test["statistic"] == pytest.approx(key["F"], rel=EXACT)
    assert test["pvalue"] == pytest.approx(key["F_p"], rel=EXACT)
    robust = sp.regress("re78 ~ " + _rhs(lalonde, "re78"), lalonde, robust="hc3")
    assert sp.test(robust, nulls)["statistic"] == pytest.approx(key["F_hc3"], rel=EXACT)
    assert np.mean(full.predict(lalonde.assign(treat=1))) == pytest.approx(key["mean1"])
    assert np.mean(full.predict(lalonde.assign(treat=0))) == pytest.approx(key["mean0"])


def test_ch06_hc0_to_hc4(R, lalonde, boston):
    isq = pd.read_csv(_p("_statspai/isq_complete.csv"))
    cases = {
        "ch6_lalonde": ("re78 ~ " + _rhs(lalonde, "re78"), lalonde),
        "ch6_isq": ("multish ~ " + _rhs(isq, "multish"), isq),
        "ch6_isq_log": ("log(multish + 1) ~ " + _rhs(isq, "multish"), isq),
        "ch6_boston": ("medv ~ " + _rhs(boston, "medv"), boston),
        "ch6_boston_log": ("log(medv) ~ " + _rhs(boston, "medv"), boston),
    }
    for key, (formula, data) in cases.items():
        close(sp.regress(formula, data).std_errors, R[key]["ols"], EXACT)
        for kind in ("hc0", "hc1", "hc2", "hc3", "hc4"):
            fit = sp.regress(formula, data, robust=kind)
            close(fit.std_errors, R[key][kind], EXACT)
    # chapter 24 prints the same three through sandwich::vcovHC
    fit = sp.regress("medv ~ " + _rhs(boston, "medv"), boston, robust="hc3")
    close(fit.std_errors, R["ch24_boston"]["hc3"], EXACT)


def test_ch08_nested_models_by_wald_and_lr(R, lalonde):
    # anova(small, full) is the F test that the extra coefficients are zero
    full = sp.regress("re78 ~ " + _rhs(lalonde, "re78"), lalonde)
    extra = [f"{v}=0" for v in lalonde.columns if v not in ("re78", "treat")]
    test = sp.test(full, extra)
    assert test["statistic"] == pytest.approx(R["ch8_anova"]["F2"], rel=EXACT)
    assert test["pvalue"] == pytest.approx(R["ch8_anova"]["p2"], rel=EXACT)


def test_ch11_leverage_and_influence(R, lalonde):
    fit = sp.regress("re78 ~ " + _rhs(lalonde, "re78"), lalonde)
    out = sp.estat(fit, "leverage", print_results=False)
    key = R["ch11_lalonde"]
    close(out["leverage"], key["hat"], EXACT)
    close(out["rstandard"], key["rstandard"], EXACT)
    close(out["rstudent"], key["rstudent"], EXACT)
    close(out["cooks_d"], key["cook"], EXACT)
    close(out["dffits"], key["dffits"], EXACT)


# ----------------------------------------------------------- chapters 12-19


def test_ch12_leave_one_out_prediction_intervals(R, boston):
    ordered = boston.sort_values("medv", kind="stable").reset_index(drop=True)
    fit = sp.regress("medv ~ " + _rhs(boston, "medv"), ordered)
    out = sp.estat(fit, "leverage", print_results=False)
    key = R["ch12_boston"]
    pred = ordered["medv"].to_numpy() - out["loo_residuals"]
    close(pred, key["loo_pred"], EXACT)
    close(pred - out["loo_interval_halfwidth"], key["lower"], EXACT)
    close(pred + out["loo_interval_halfwidth"], key["upper"], EXACT)
    assert out["press"] == pytest.approx(key["press"], rel=EXACT)
    # the book's program counts 13 coefficients where the model has 14, so
    # its printed limits sit within 0.1% of the exact ones, not on them
    gap = np.abs(pred - out["loo_interval_halfwidth"] - np.asarray(key["lower_book"]))
    assert 1e-6 < gap.max() < 0.02


@pytest.mark.parametrize("key,file,y", [("ch13_penn", "pennbonus.txt", "duration"),
                                        ("ch13_boston", None, "medv")])
def test_ch13_best_subsets(R, boston, key, file, y):
    data = boston if file is None else _table(file)
    cand = [c for c in data.columns if c != y]
    ref = R[key]
    for crit in ("aic", "bic"):
        fit = sp.best_subset(data, y, cand, criterion=crit)
        assert sorted(fit.selected) == sorted(ref[f"which_{crit}"])
        assert len(fit.selected) == ref[f"best_{crit}"]
    close(fit.history["rss"], ref["rss"], EXACT)
    close(fit.history["adj_r_squared"], ref["adjr2"], EXACT)
    # the book's AIC and BIC drop constants; differences across sizes agree
    assert np.ptp(fit.history["bic"].to_numpy() - np.asarray(ref["bic"])) < 1e-7


def test_ch13_stepwise_misses_the_best_subset_on_boston(R, boston):
    cand = [c for c in boston.columns if c != "medv"]
    best = sp.best_subset(boston, "medv", cand, criterion="bic")
    step = sp.stepwise(boston, "medv", cand, criterion="bic", verbose=False)
    assert best.final_model["bic"] < step.final_model["bic"] - 1.0


def test_ch14_ridge(R, boston):
    ref = R["ch14_ridge"]
    formula = "medv ~ " + _rhs(boston, "medv")
    fit = sp.ridge(formula, boston, lambda_=ref["lambda"])
    names = ["Intercept"] + ref["names"][1:]
    close(fit.path[names].to_numpy(), np.asarray(ref["coef"]), EXACT)
    close(fit.path["gcv"], ref["gcv"], EXACT)
    assert fit.lambda_hkb == pytest.approx(ref["kHKB"], rel=EXACT)
    assert fit.lambda_lw == pytest.approx(ref["kLW"], rel=EXACT)
    fine = sp.ridge(formula, boston, lambda_=np.arange(0, 5.001, 0.01))
    assert fine.lambda_ == pytest.approx(ref["gcv_min_lambda"])


def test_ch16_boxcox_and_polynomials(R):
    penn = _table("pennbonus.txt")
    ref = R["ch16_boxcox_penn"]
    fit = sp.boxcox("duration ~ " + _rhs(penn, "duration"), penn, lambdas=ref["lambda"])
    assert np.ptp(fit.profile["loglik"].to_numpy() - np.asarray(ref["loglik"])) < 1e-7
    # R's estimate is the best point of a 0.001 grid
    assert fit.lambda_ == pytest.approx(ref["lambda_hat"], abs=1e-3)
    assert fit.ci[0] == pytest.approx(ref["ci"][0], abs=2e-3)
    assert fit.ci[1] == pytest.approx(ref["ci"][1], abs=2e-3)

    census = pd.read_stata(_p("census00.dta"))
    squared = sp.regress("logwk ~ educ + exper + I(exper^2) + black", census)
    close(squared.params, R["ch16_census2"]["est"], 1e-7)
    close(squared.std_errors, R["ch16_census2"]["se"], 1e-7)


def test_ch17_interactions(R):
    hsb = _table("hsbdemo.txt")
    fit = sp.regress("read ~ math*socst", hsb)
    close(fit.params, R["ch17_inter"]["est"], EXACT)
    close(fit.std_errors, R["ch17_inter"]["se"], EXACT)


def test_ch19_weighted_least_squares(R, boston):
    lav = pd.read_csv(_p("lavoteall.csv"))
    wls = sp.regress("t ~ x", lav, weights="n")
    close(wls.params, R["ch19_lav_wls"]["est"], EXACT)
    close(wls.std_errors, R["ch19_lav_wls"]["se"], EXACT)
    # Goodman's regression: no intercept, x and 1 - x
    goodman = sp.regress("t ~ 0 + x + I(1 - x)", lav, weights="n")
    close(goodman.params, R["ch19_lav_wls0"]["est"], EXACT)
    close(goodman.std_errors, R["ch19_lav_wls0"]["se"], EXACT)

    formula = "medv ~ " + _rhs(boston, "medv")
    ols = sp.regress(formula, boston)
    aux = boston.assign(medv=np.log(np.asarray(ols.residuals()) ** 2))
    weights = np.exp(-np.asarray(sp.regress(formula, aux).predict(aux)))
    close(weights, R["ch19_fgls_w"], EXACT)
    fgls = sp.regress(formula, boston.assign(w=weights), weights="w")
    close(fgls.params, R["ch19_fgls"]["est"], EXACT)
    close(fgls.std_errors, R["ch19_fgls"]["se"], EXACT)

    census = pd.read_stata(_p("census00.dta"))
    fit = sp.regress("logwk ~ age + educ + exper + exper2 + black", census, weights="perwt")
    close(fit.params, R["ch19_census_wls"]["est"], 1e-7)
    close(fit.std_errors, R["ch19_census_wls"]["se"], 1e-7)


def test_ch19_local_linear_fit_is_exact_where_locpoly_bins(R):
    sim = pd.read_csv(_p("_statspai/sim_locpoly.csv"))
    ref = R["ch19_locpoly"]
    grid = np.asarray(ref["x"])
    fit = sp.lpoly(sim, "y", "x", bandwidth=ref["h"], degree=1, kernel="gaussian",
                   grid=grid, ci=False)
    x, y, h = sim["x"].to_numpy(), sim["y"].to_numpy(), ref["h"]
    exact = []
    for x0 in grid:
        w = np.exp(-0.5 * ((x - x0) / h) ** 2)
        X = np.column_stack([np.ones_like(x), x - x0])
        exact.append(np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))[0])
    close(fit.fitted, exact, EXACT)
    # KernSmooth::locpoly bins the data onto 401 points first; it tracks the
    # exact fit closely in the interior and loosely at the two ends
    inner = (grid > 0.1) & (grid < 0.9)
    assert np.max(np.abs(np.asarray(ref["y"])[inner] - np.asarray(exact)[inner])) < 0.02


# ----------------------------------------------------------- chapters 20-24


def test_ch20_logit_predictions_and_marginal_effects(R, flu):
    key = R["ch20_flu"]
    formula = "outcome ~ " + _rhs(flu, "outcome")
    fit = sp.logit(formula, flu, tol=1e-13)
    close(fit.params, key["est"], GLM)
    close(fit.std_errors, key["se"], GLM)
    assert fit.diagnostics["LR chi2"] == pytest.approx(key["lr"], rel=1e-9)
    assert fit.diagnostics["Prob > chi2"] == pytest.approx(key["lr_p"], rel=1e-7)

    at = pd.DataFrame([flu.mean(), flu.mean()])
    at.iloc[0, 0], at.iloc[1, 0] = 1, 0
    glm = sp.glm(formula, flu, family="binomial", tol=1e-13)
    pred = glm.predict(at, what="confidence")
    close(pred["yhat"], key["pred"], GLM)
    close(pred["se"], key["pred_se"], GLM)

    ame = sp.margins(fit, flu).set_index("variable")
    close(ame.loc[key["ame_names"], "dy/dx"], key["ame"], 1e-6)
    # R's margins differentiates numerically; its SEs carry that error
    close(ame.loc[key["ame_names"], "se"], key["ame_se"], 1e-4)

    # chapter 24: sandwich(glm) is HC0; Stata's vce(robust) adds N/(N-1)
    hc0 = sp.logit(formula, flu, tol=1e-13, robust="hc0")
    close(hc0.std_errors, R["ch24_flu_sandwich"], GLM)
    stata = sp.logit(formula, flu, tol=1e-13, robust="robust")
    n = len(flu)
    close(stata.std_errors, np.asarray(R["ch24_flu_sandwich"]) * np.sqrt(n / (n - 1)), GLM)


@pytest.mark.parametrize("link", ["probit", "logit", "cloglog", "cauchit"])
def test_ch20_four_binary_links(R, link):
    sim = pd.read_csv(_p("_statspai/sim_links.csv"))
    key = R[f"ch20_link_{link}"]
    # R's glm reports the expected information; ours defaults to observed
    fit = sp.glm("y ~ x", sim, family="binomial", link=link, tol=1e-13,
                 information="expected")
    close(fit.params, key["est"], GLM)
    close(fit.std_errors, key["se"], GLM)
    assert fit.diagnostics["Deviance"] == pytest.approx(key["deviance"], rel=1e-9)


def test_ch21_categorical_outcomes(R):
    kar = _table("karolinska.txt")
    formula = "survival ~ highdiag + age + rural + male"
    key = R["ch21_multinom"]
    mn = sp.mlogit(formula, kar, tol=1e-12)
    close(mn.params, np.asarray(key["est"]).ravel(), OPTIM)
    close(mn.std_errors, np.asarray(key["se"]).ravel(), OPTIM)
    assert -2 * mn.model_info["log_likelihood"] <= key["deviance"] + 1e-8
    np.testing.assert_allclose(mn.predict(kar.head(5)), key["probs"], atol=1e-5)

    key = R["ch21_polr"]
    po = sp.ologit(formula, kar, tol=1e-12)
    close(po.params, key["est"], OPTIM)
    close(po.std_errors, key["se"], OPTIM)
    assert -2 * po.model_info["log_likelihood"] <= key["deviance"] + 1e-8
    np.testing.assert_allclose(po.predict(kar.head(5)), key["probs"], atol=1e-4)
    assert list(po.predict(kar.head(2)).columns) == ["1", "2-4", "5+"]


def test_ch21_discrete_choice_on_the_fishing_data(R):
    fishing = pd.read_csv(_p("_statspai/Fishing.csv"))
    alts = ["beach", "pier", "boat", "charter"]
    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "id": np.arange(len(fishing)),
                    "alt": a,
                    "chosen": (fishing["mode"] == a).astype(int),
                    "price": fishing[f"price.{a}"],
                    "catch": fishing[f"catch.{a}"],
                }
            )
            for a in alts
        ]
    )
    cond = sp.clogit("chosen ~ price + catch", long, group="id", tol=1e-12)
    close(cond.params, R["ch21_fish_cond0"]["est"], 1e-5)
    close(cond.std_errors, R["ch21_fish_cond0"]["se"], 1e-5)
    for a in alts[1:]:
        long[a] = (long["alt"] == a).astype(int)
    asc = sp.clogit("chosen ~ price + catch + boat + charter + pier", long,
                    group="id", tol=1e-12)
    key = dict(zip(R["ch21_fish_cond"]["names"], R["ch21_fish_cond"]["est"]))
    ours = [asc.params[n] for n in ("boat", "charter", "pier", "price", "catch")]
    ref = [key[f"(Intercept):{a}"] for a in ("boat", "charter", "pier")]
    close(ours, ref + [key["price"], key["catch"]], 1e-5)
    # individual-specific regressors only: the multinomial logit
    ind = sp.mlogit("mode ~ income", fishing, tol=1e-12)
    key = dict(zip(R["ch21_fish_ind"]["names"], R["ch21_fish_ind"]["est"]))
    for a in ("boat", "charter", "pier"):
        assert ind.params[f"[{a}]_cons"] == pytest.approx(key[f"(Intercept):{a}"], rel=1e-5)
        assert ind.params[f"[{a}]income"] == pytest.approx(key[f"income:{a}"], rel=1e-5)


@pytest.mark.parametrize("week", [1, 5, 20, 52])
def test_ch22_count_models_by_week(R, gym, week):
    key = R["ch22"][f"week{week}"]
    data = gym[gym["incentive_week"] == week]
    ols = sp.regress(GYM, data)
    close(ols.std_errors, key["ols"]["se"], EXACT)
    # R's AIC counts the error variance as a parameter, Stata's does not
    assert ols.diagnostics["AIC"] + 2 == pytest.approx(key["ols"]["aic"], rel=1e-10)

    pois = sp.poisson(GYM, data, tol=1e-13)
    close(pois.params, key["poisson"]["est"], GLM)
    close(pois.std_errors, key["poisson"]["se"], GLM)
    assert pois.diagnostics["AIC"] == pytest.approx(key["poisson"]["aic"], rel=1e-9)

    nb = sp.nbreg(GYM, data, tol=1e-12)
    close(nb.params.to_numpy()[:5], key["nb"]["est"], 1e-5)
    assert nb.diagnostics["Dispersion (alpha)"] == pytest.approx(1 / key["nb"]["theta"], rel=1e-5)
    assert nb.diagnostics["AIC"] == pytest.approx(key["nb"]["aic"], rel=1e-8)
    # glm.nb's standard errors hold theta fixed (expected information of the
    # mean model); ours come from the joint observed information, as Stata's
    # nbreg. They agree to a few percent, not to the digit.
    close(nb.std_errors.to_numpy()[:5], key["nb"]["se"], 0.05)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zip_ = sp.zip_model(GYM, data, tol=1e-12)
        zinb = sp.zinb(GYM, data, tol=1e-12)
    close(zip_.params.to_numpy()[:5], key["zip"]["count"], 1e-4)
    close(zip_.params.to_numpy()[5:10], key["zip"]["zero"], 1e-4)
    close(zip_.std_errors.to_numpy()[:5], key["zip"]["count_se"], 1e-4)
    assert zip_.diagnostics["ll"] >= key["zip"]["loglik"] - 1e-6
    # these weeks show no overdispersion beyond the zeros: the negative
    # binomial runs to its Poisson limit (R's theta in the millions), where
    # the likelihood is flat and the two optimisers stop 1e-5 apart
    assert zinb.diagnostics["ll"] == pytest.approx(key["zinb"]["loglik"], abs=1e-3)
    assert zinb.diagnostics["ll"] <= zip_.diagnostics["ll"] + 1e-3
    close(zinb.params.to_numpy()[:5], key["zinb"]["count"], OPTIM)


def test_ch24_sandwich_for_count_models(R):
    sim = pd.read_csv(_p("_statspai/sim_counts.csv"))
    for y, key in [("y_pois", "ch24_pois_pois"), ("y_nb", "ch24_nb_pois"),
                   ("y_wr", "ch24_wr_pois")]:
        close(sp.poisson(f"{y} ~ x", sim, tol=1e-13).std_errors, R[key]["se"], GLM)
        hc0 = sp.poisson(f"{y} ~ x", sim, tol=1e-13, robust="hc0")
        close(hc0.std_errors, R[key]["robust"], GLM)
    # overdispersed counts: the model-based Poisson SE is far too small
    assert R["ch24_nb_pois"]["robust"][1] > 2 * R["ch24_nb_pois"]["se"][1]


# ----------------------------------------------------------- chapters 25-27


def _gee_check(fit, key, rtol=1e-7):
    close(fit.params, key["est"], rtol)
    close(fit.model_info["se_model"], key["naive"], rtol)
    close(fit.std_errors, key["robust"], rtol)
    close(fit.model_info["scale"], key["scale"], rtol)


def test_ch25_gee_split_plot_mice(R):
    pten = pd.read_csv(_p("PtenAnalysisData.csv")).iloc[:, :6]
    ind = sp.gee("somasize ~ C(fa)*pten", pten, id="mouseid")
    exch = sp.gee("somasize ~ C(fa)*pten", pten, id="mouseid", corstr="exchangeable")
    order = ["Intercept", "C(fa)[T.1]", "C(fa)[T.2]", "pten", "C(fa)[T.1]:pten",
             "C(fa)[T.2]:pten"]
    for fit, key in [(ind, R["ch25_pten_ind"]), (exch, R["ch25_pten_exch"])]:
        close(fit.params[order], key["est"], 1e-7)
        close(fit.model_info["se_model"][order], key["naive"], 1e-7)
        close(fit.std_errors[order], key["robust"], 1e-7)
    assert exch.model_info["corr_alpha"] == pytest.approx(R["ch25_pten_exch"]["alpha"], rel=1e-7)


def test_ch25_gee_cluster_randomised_hygiene_trial(R):
    hyg = pd.read_csv(_p("_statspai/hygaccess_analysis.csv"))
    ind = sp.gee("y ~ C(z)", hyg, id="vid", family="binomial")
    _gee_check(ind, R["ch25_hyg_ind"])
    exch = sp.gee("y ~ C(z)", hyg, id="vid", family="binomial", corstr="exchangeable")
    _gee_check(exch, R["ch25_hyg_exch"])
    assert exch.model_info["corr_alpha"] == pytest.approx(R["ch25_hyg_exch"]["alpha"], rel=1e-7)
    # ignoring the villages understates the standard error fourfold
    assert ind.std_errors.iloc[0] > 3.5 * ind.model_info["se_model"].iloc[0]


def test_ch25_gee_gym_panel(R, gym):
    _gee_check(sp.gee(GYM, gym, id="id"), R["ch25_gym_normal"])
    _gee_check(sp.gee(GYM, gym, id="id", family="poisson"), R["ch25_gym_poisson"])
    short = gym[(gym["incentive_week"] > 0) & (gym["incentive_week"] < 15)]
    _gee_check(sp.gee(GYM, short, id="id", family="poisson"), R["ch25_gym_poisson_short"])


def test_ch26_quantile_regression(R):
    galton = _table("GaltonFamilies.txt")
    key = R["ch26_galton"]
    est = np.column_stack(
        [sp.qreg(galton, "childHeight ~ midparentHeight", quantile=t).params
         for t in key["taus"]]
    )
    np.testing.assert_allclose(est, key["coef"], atol=1e-8)


@pytest.mark.parametrize("year", ["80", "00"])
def test_ch26_weighted_quantile_regression_on_the_census(R, year):
    census = pd.read_stata(_p(f"census{year}.dta"))
    key = R[f"ch26_census{year}"]
    formula = "logwk ~ educ + exper + exper2 + black"
    for col, tau in [(0, 0.1), (8, 0.9)]:
        fit = sp.qreg(census, formula, quantile=tau, weights="perwt", vce="ker")
        close(fit.params, np.asarray(key["coef_w"])[:, col], 1e-6)
        close(fit.std_errors, np.asarray(key["se_ker_w"])[:, col], 1e-5)
        plain = sp.qreg(census, formula, quantile=tau, vce="nid")
        close(plain.std_errors, np.asarray(key["se_nid"])[:, col], 1e-5)


def _cox_by_name(fit, key, what="est"):
    def rename(name: str) -> str:
        name = name.replace("C(GENDER)[T.male]", "GENDERmale")
        return name.replace("C(site)[T.", "site").replace("]", "")

    ours = {rename(c): (fit.params[c], fit.std_errors[c]) for c in fit.params.index}
    pos = 0 if what == "est" else 1
    return [ours[n][pos] for n in key["names"]], key["est" if what == "est" else "robust"]


def test_ch27_cox_with_tied_relapse_days(R):
    combine = pd.read_csv(_p("combine_data.txt"), sep="\t").drop(columns="ID")
    formula = "futime ~ NALTREXONE*THERAPY + AGE + C(GENDER) + T0_PDA + C(site)"
    fit = sp.cox(formula, combine, event="relapse", robust="hc0")
    close(*_cox_by_name(fit, R["ch27_combine"]), 1e-7)
    close(*_cox_by_name(fit, R["ch27_combine"], "se"), 1e-7)
    breslow = sp.cox(formula, combine, event="relapse", robust="hc0", ties="breslow")
    close(*_cox_by_name(breslow, R["ch27_combine_breslow"], "se"), 1e-7)
    strat = sp.cox(
        "futime ~ NALTREXONE*THERAPY + AGE + C(GENDER) + T0_PDA",
        combine, event="relapse", robust="hc0", strata="site",
    )
    close(*_cox_by_name(strat, R["ch27_combine_strata"], "se"), 1e-7)


def test_ch27_gehan_logrank_and_cox(R):
    gehan = pd.read_csv(_p("_statspai/gehan.csv"))
    gehan["control"] = (gehan["treat"] == "control").astype(float)
    key = R["ch27_gehan"]
    fit = sp.cox(data=gehan, duration="time", event="cens", x=["control"])
    close(fit.params, key["est"], 1e-6)
    close(fit.std_errors, key["se"], 1e-6)
    assert fit.concordance == pytest.approx(key["concordance"], rel=1e-9)
    assert fit.diagnostics["LR chi2"] == pytest.approx(key["lr"], rel=1e-8)
    lr = sp.logrank_test(gehan, "time", "cens", "treat")
    assert lr["test_statistic"] == pytest.approx(key["logrank_chisq"], rel=1e-10)
    assert lr["p_value"] == pytest.approx(key["logrank_p"], rel=1e-8)


def test_ch27_paired_eyes_need_the_cluster(R):
    path = _p("_statspai/diabetes_timereg.csv")
    if not path.is_file():
        pytest.skip("the R key was built without the timereg package")
    eyes = pd.read_csv(path)
    kw = dict(data=eyes, duration="time", event="status", x=["treat", "adult", "agedx"])
    close(sp.cox(robust="hc0", **kw).std_errors, R["ch27_diabetes_robust"]["robust"], 1e-7)
    G = eyes["id"].nunique()
    cluster = sp.cox(cluster="id", **kw)
    # ours carries Stata's G / (G - 1); survival::coxph(cluster =) does not
    close(cluster.std_errors * np.sqrt((G - 1) / G), R["ch27_diabetes_cluster"]["robust"], 1e-7)


def test_ch26_star_median_regression_clustered_by_class(R):
    star = pd.read_csv(_p("star.csv"))
    key = R["ch26_star"]
    formula = (
        "pscore ~ small + regaide + black + girl + poor + tblack + texp "
        "+ tmasters + C(fe)"
    )
    fit = sp.qreg(star, formula, quantile=0.5, vce="ker")
    used = star.dropna(subset=["pscore", "small", "regaide", "black", "girl", "poor",
                               "tblack", "texp", "tmasters", "fe"])
    import patsy

    X = patsy.dmatrix(formula.split("~")[1], used, return_type="dataframe")
    beta = np.array([fit.params["const" if c == "Intercept" else c] for c in X.columns])
    objective = np.sum(np.abs(used["pscore"].to_numpy() - X.to_numpy() @ beta)) / 2
    # the median fit is not unique here (R warns so): the two programs stop
    # at different vertices of one optimal face, with the same check loss
    assert objective == pytest.approx(key["objective"], rel=1e-9)
    np.testing.assert_allclose(fit.params[key["names"]], key["est"], rtol=0.05, atol=0.05)
    # students share classrooms: the clustered standard errors of the
    # class-level regressors are about 1.5 times the unclustered ones, by
    # the analytic formula here and by the clustered bootstrap in the book
    clustered = sp.qreg(star, formula, quantile=0.5, cluster="classid")
    ours = (clustered.std_errors[key["names"]] / fit.std_errors[key["names"]]).to_numpy()
    book = np.asarray(key["se_boot_cluster"]) / np.asarray(key["se_boot"])
    assert ours[0] > 1.4 and book[0] > 1.4
    np.testing.assert_allclose(clustered.std_errors[key["names"]], key["se_boot_cluster"], rtol=0.15)


def test_ch12_conformal_intervals_on_boston(boston):
    # the chapter's exercise: predict each house from the others and count
    # how often the 95% interval holds its price
    formula = "medv ~ " + _rhs(boston, "medv")
    full = sp.conformal_regression(formula, boston, method="full", alpha=0.05)
    covered = ((boston["medv"] >= full["lower"]) & (boston["medv"] <= full["upper"])).mean()
    assert 0.93 <= covered <= 0.97
    # the rank rule of the book's program, on one house, by brute force
    i = 100
    rest = boston.drop(index=i)
    X = np.column_stack([np.ones(len(rest)), rest.drop(columns="medv").to_numpy(float)])
    x0 = np.append(1.0, boston.drop(columns="medv").iloc[i].to_numpy(float))
    Xa = np.vstack([X, x0])
    maker = np.eye(len(Xa)) - Xa @ np.linalg.solve(Xa.T @ Xa, Xa.T)
    grid = np.arange(-20.0, 70.0, 0.02)
    y = rest["medv"].to_numpy()
    n = len(y)
    keep = []
    for g in grid:
        r = np.abs(maker @ np.append(y, g))
        keep.append((np.sum(r[:-1] >= r[-1]) + 1) > 0.05 * (n + 1))
    kept = grid[np.asarray(keep)]
    assert full["lower"].iloc[i] == pytest.approx(kept.min(), abs=0.02)
    assert full["upper"].iloc[i] == pytest.approx(kept.max(), abs=0.02)
    # prices are skewed: the distribution-free interval is not the normal one
    jack = sp.conformal_regression(formula, boston, method="jackknife+", alpha=0.05)
    assert abs((jack["upper"] - jack["lower"]).mean() / (full["upper"] - full["lower"]).mean() - 1) < 0.1


def test_ch16_additive_model_for_wages(R):
    key = R.get("ch16_gam")
    if key is None:
        pytest.skip("the R key predates the additive-model entry; rerun the R script")
    census = pd.read_stata(_p("census00.dta"))
    formula = "logwk ~ s(educ, k=10) + s(exper, k=10) + black"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.gam(formula, census)
        at_theirs = sp.gam(formula, census, lambda_=np.asarray(key["sp"]) / 16.0)
    # same basis and criterion as mgcv with bs = "ps", method = "REML": at
    # mgcv's smoothing parameters the fit is mgcv's, on 65,000 workers
    close(at_theirs.smooth_terms["edf"], key["edf"], EXACT)
    close(at_theirs.params, key["par"], EXACT)
    close(at_theirs.std_errors, key["par_se"], EXACT)
    close(at_theirs.fitted_values[:20], key["fitted_head"], EXACT)
    assert at_theirs.scale == pytest.approx(key["scale"], rel=EXACT)
    # the schooling curve uses nearly all of its basis, where the REML
    # surface is flat: the two optimisers stop 2% apart in lambda, ours at
    # the (slightly) lower criterion, and the fits agree to five digits
    assert fit.gcv <= at_theirs.gcv + 1e-6
    close(fit.smooth_terms["lambda"] * 16.0, key["sp"], 0.05)
    close(fit.smooth_terms["edf"], key["edf"], 1e-3)
    close(fit.params, key["par"], 1e-5)
    close(fit.fitted_values[:20], key["fitted_head"], 1e-4)
    # the book's own call uses a thin plate basis and GCV: a different
    # curve from the same data, close on the fitted values
    np.testing.assert_allclose(fit.fitted_values[:20], key["tp_fitted_head"], rtol=0.01)
    assert fit.params["black"] == pytest.approx(key["tp_black"], abs=0.005)
