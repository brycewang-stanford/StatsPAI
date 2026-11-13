"""Reference parity for the survival / epidemiology items of the 2026-11 gap list.

Each item is something an epidemiology or biostatistics analysis plan asks
for by name and that StatsPAI did not have:

1. ``sp.cox(ties='exact')``: the exact partial likelihood.
2. ``sp.aft(robust=, cluster=)``: sandwich variances of the AFT model.
3. ``sp.g_computation(outcome_model='logit' | 'poisson', contrast=)``:
   standardisation with a logistic / Poisson outcome model and the risk
   difference, risk ratio and odds ratio.
4. ``sp.ipw(truncate=)``: percentile truncation of the weights.
5. ``sp.gee``: QIC, QICu, the quasi-likelihood and CIC.
7. ``sp.mixed`` / ``sp.meglm``: standard errors of the variance components.
8. ``sp.overlap_weights(se_method='sandwich')``.
9. ``sp.mr('all', ...)`` with the summary statistics as arrays.

References on the same CSV bytes (``gaps_surv.csv``, ``gaps_obs.csv``,
``gaps_long.csv`` from ``_generate_gaps_epi_data.py``):

* ``_fixtures/gaps_epi_R.json`` from ``_generate_gaps_epi_R.R``: R 4.5.2,
  survival 3.8.3, marginaleffects 0.32.0, WeightIt 1.7.0, sandwich 3.1.1,
  geepack 1.3.13, PSweight 2.1.2.
* ``_fixtures/gaps_epi_Stata.csv`` from ``_generate_gaps_epi_Stata.do``:
  Stata 18 MP (``stcox, exactp``, ``streg``, ``teffects ra``, ``mixed``,
  ``melogit``, ``mepoisson``).

Tolerances are relative 1e-8 where both sides are closed form at a fully
converged fit (observed 1e-11 or better). The exceptions, each with its
mechanism:

* **AFT sandwich** (2e-6; observed 7e-7). ``sp.aft`` calls ``sp.survreg``,
  whose Hessian is a finite difference of exact scores; its
  observed-information standard errors agree with R to 1.2e-7 and the
  sandwich applies that matrix twice. R and Stata agree with each other to
  1e-12, so the remainder is ours. Coefficients agree to 1e-13.
* **Mixed-model variance components** (1e-6 for a random intercept, observed
  3e-7; 1e-3 with a random slope, observed 4e-4). Stata stops its Newton
  iterations earlier than the optimum; the test also asserts that our
  log-likelihood is at least Stata's. The random-slope likelihood is nearly
  flat in the slope variance (its standard error is 80% of the estimate),
  which magnifies the stopping error.
* **Adaptive quadrature GLMM** (``melogit`` / ``mepoisson`` with seven
  points; 1e-3). As recorded in ``test_panel_glmm_parity.py``, Stata's
  adaptive-quadrature fits stop short of the maximum of their objective;
  here too our log-likelihood is the higher one. The Laplace ``melogit``
  fit converges on both sides and pins the variance and its standard error
  at 1e-7 (observed 4e-9).
* **GEE QIC with a fitted working correlation** (1e-6). ``geepack``
  estimates the correlation parameter by another moment equation than
  ``sp.gee`` (which follows R ``gee`` and Stata ``xtgee``), so the R side is
  ``geeglm(corstr = "fixed")`` at the correlation ``sp.gee`` estimates
  (``gaps_gee_alpha.csv``); the coefficients then agree to 1.5e-8, the
  iteration tolerance of ``geepack``. Independence fits need no such step
  and agree to 1e-12.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

FIX = pathlib.Path(__file__).parent / "_fixtures"
RTOL = 1e-8
XS = ["age", "ecog", "arm"]
XO = ["x1", "x2", "x3"]
XL = ["trt", "visit", "age"]


@pytest.fixture(scope="module")
def surv() -> pd.DataFrame:
    return pd.read_csv(FIX / "gaps_surv.csv")


@pytest.fixture(scope="module")
def obs() -> pd.DataFrame:
    return pd.read_csv(FIX / "gaps_obs.csv")


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.read_csv(FIX / "gaps_long.csv")


@pytest.fixture(scope="module")
def ref() -> dict:
    return json.loads((FIX / "gaps_epi_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def stata() -> dict:
    df = pd.read_csv(FIX / "gaps_epi_Stata.csv")
    return {
        case: dict(zip(g["name"], g["value"].astype(float)))
        for case, g in df.groupby("case")
    }


# ---------------------------------------------------------------------------
# 1. Cox: exact partial likelihood
# ---------------------------------------------------------------------------

COX_KW = {
    "plain": {},
    "strata": {"strata": "site"},
    "entry": {"start": "entry"},
    "entry_strata": {"start": "entry", "strata": "site"},
}


def _cox_exact(surv, case):
    return sp.cox(
        "time ~ age + ecog + arm",
        data=surv,
        event="event",
        ties="exact",
        **COX_KW[case],
    )


@pytest.mark.parametrize("case", ["plain", "strata"])
def test_cox_exact_matches_coxph(surv, ref, case):
    fit = _cox_exact(surv, case)
    want = ref["cox_exact"][case]
    np.testing.assert_allclose(fit.params, want["coef"], rtol=RTOL)
    np.testing.assert_allclose(fit.std_errors, want["se"], rtol=RTOL)
    d = fit.diagnostics
    assert d["Log-likelihood (null)"] == pytest.approx(want["loglik"][0], rel=1e-12)
    assert d["Log-likelihood"] == pytest.approx(want["loglik"][1], rel=1e-12)
    assert d["Score chi2"] == pytest.approx(want["score"], rel=RTOL)
    assert d["Wald chi2"] == pytest.approx(want["wald"], rel=RTOL)
    assert d["Concordance (C)"] == pytest.approx(want["concordance"], rel=1e-10)
    assert fit.model_info["ties"] == "exact"


@pytest.mark.parametrize("case", list(COX_KW))
def test_cox_exact_matches_stcox_exactp(surv, stata, case):
    """Including delayed entry, where R's exact method does not finish."""
    fit = _cox_exact(surv, case)
    want = stata[f"cox_exact_{case}"]
    for name in XS:
        assert fit.params[name] == pytest.approx(want[f"b_{name}"], rel=RTOL)
        assert fit.std_errors[name] == pytest.approx(want[f"se_{name}"], rel=RTOL)
    assert fit.diagnostics["Log-likelihood"] == pytest.approx(want["ll"], rel=1e-12)
    if "ll_0" in want:
        assert fit.diagnostics["Log-likelihood (null)"] == pytest.approx(
            want["ll_0"], rel=1e-12
        )


def test_cox_exact_baseline_hazard_is_breslow_at_the_exact_estimate(surv, ref):
    """R ``basehaz`` after an exact fit; Stata's ``basechazard`` is another
    estimator and is not followed (docs/dev/modules/survival.md)."""
    fit = _cox_exact(surv, "plain")
    want = ref["cox_exact"]["plain"]
    table = fit.baseline_hazard()
    ours = table[table["time"] > 0].set_index("time")["baseline_cumhaz"]
    theirs = pd.Series(want["basehaz"], index=want["basehaz_time"])
    np.testing.assert_allclose(ours, theirs.loc[ours.index], rtol=RTOL)


def test_cox_exact_equals_the_other_rules_without_ties_and_its_definition():
    rng = np.random.default_rng(3)
    n = 60
    df = pd.DataFrame(
        {
            "t": rng.permutation(n) + 1.0,  # no ties
            "e": (rng.random(n) < 0.8).astype(int),
            "x": rng.normal(size=n),
            "z": rng.integers(0, 2, n),
        }
    )
    exact = sp.cox("t ~ x + z", data=df, event="e", ties="exact")
    for other in ("efron", "breslow"):
        fit = sp.cox("t ~ x + z", data=df, event="e", ties=other)
        np.testing.assert_allclose(exact.params, fit.params, rtol=1e-10)
        np.testing.assert_allclose(exact.std_errors, fit.std_errors, rtol=1e-10)
    # definition, by enumerating the subsets of each risk set
    from itertools import combinations

    from statspai.survival._cox_exact import ExactCoxKernel

    m = 14
    X = rng.normal(size=(m, 2))
    T = rng.integers(1, 5, m).astype(float)
    E = (rng.random(m) < 0.8).astype(float)
    b = np.array([0.3, -0.7])
    r = np.exp(X @ b)
    ll = 0.0
    for t in np.unique(T[E == 1]):
        risk = np.flatnonzero(T >= t)
        dead = np.flatnonzero((T == t) & (E == 1))
        denom = sum(np.prod(r[list(c)]) for c in combinations(risk, len(dead)))
        ll += float(np.sum(X[dead] @ b)) - np.log(denom)
    assert -ExactCoxKernel(X, T, E).neg_loglik(b) == pytest.approx(ll, rel=1e-12)


def test_cox_exact_refuses_what_the_likelihood_does_not_define(surv):
    base = dict(data=surv, event="event", ties="exact")
    for kw in (
        {"weights": "age"},
        {"robust": "hc0"},
        {"cluster": "clinic"},
        {"tvc": "age"},
    ):
        with pytest.raises(MethodIncompatibility, match="ties='exact'"):
            sp.cox("time ~ age + ecog + arm", **base, **kw)
    fit = sp.cox("time ~ age + ecog + arm", **base)
    with pytest.raises(MethodIncompatibility, match="not available after"):
        fit.ph_test()
    with pytest.raises(ValueError, match="'efron', 'breslow' or 'exact'"):
        sp.cox("time ~ age", data=surv, event="event", ties="exactm")


# ---------------------------------------------------------------------------
# 2. AFT: robust and clustered variance
# ---------------------------------------------------------------------------

AFT_FAMILIES = ["weibull", "exponential", "lognormal", "loglogistic"]
AFT_ANCILLARY = {"weibull": "ln_p", "lognormal": "lnsigma", "loglogistic": "lngamma"}
AFT_SE_RTOL = 2e-6  # see the module docstring


@pytest.mark.parametrize("family", AFT_FAMILIES)
@pytest.mark.parametrize("kind", ["robust", "cluster"])
def test_aft_sandwich_matches_streg(surv, stata, family, kind):
    kw = {"robust": "robust"} if kind == "robust" else {"cluster": "clinic"}
    fit = sp.aft("time + event ~ age + ecog + arm", surv, family=family, **kw)
    want = stata[f"aft_{family}_{kind}"]
    names = ["_cons"] + XS
    np.testing.assert_allclose(fit.beta, [want[f"b__t__{n}"] for n in names], rtol=1e-9)
    np.testing.assert_allclose(
        fit.se, [want[f"se__t__{n}"] for n in names], rtol=AFT_SE_RTOL
    )
    if family in AFT_ANCILLARY:
        # ln(p) = -ln(sigma) for the Weibull: the same standard error
        assert fit.se_log_sigma == pytest.approx(
            want[f"se_{AFT_ANCILLARY[family]}"], rel=AFT_SE_RTOL
        )
    if kind == "cluster":
        assert fit.n_clusters == want["N_clust"]
        assert fit.vcov_type == "cluster(clinic)"
    else:
        assert fit.vcov_type == "robust"


@pytest.mark.parametrize("family", AFT_FAMILIES)
def test_aft_sandwich_is_survreg_times_the_small_sample_factor(surv, ref, family):
    """R has no factor; ours carry Stata's N/(N-1) and G/(G-1)."""
    want = ref["aft"][family]
    n, g = want["n"], want["n_clusters"]
    rob = sp.aft(
        "time + event ~ age + ecog + arm", surv, family=family, robust="robust"
    )
    clu = sp.aft(
        "time + event ~ age + ecog + arm", surv, family=family, cluster="clinic"
    )
    np.testing.assert_allclose(
        rob.se * np.sqrt((n - 1) / n), want["se_robust"][:4], rtol=AFT_SE_RTOL
    )
    np.testing.assert_allclose(
        clu.se * np.sqrt((g - 1) / g), want["se_cluster"][:4], rtol=AFT_SE_RTOL
    )
    # the default is unchanged: observed information
    plain = sp.aft("time + event ~ age + ecog + arm", surv, family=family)
    assert plain.vcov_type == "nonrobust" and plain.n_clusters is None
    np.testing.assert_allclose(plain.se, want["se"][:4], rtol=1e-6)
    assert "Std. errors" not in plain.summary()
    assert "Std. errors: cluster(clinic) (30 clusters)" in clu.summary()


# ---------------------------------------------------------------------------
# 3. g-computation with logistic / Poisson outcome models
# ---------------------------------------------------------------------------

GC = [("logit", "ybin"), ("poisson", "ycnt")]


def _ra(obs, y, model, **kw):
    return sp.g_computation(
        obs, y, "a", XO, by_arm=True, se_method="analytic", outcome_model=model, **kw
    )


@pytest.mark.parametrize("model, y", GC)
def test_gcomp_by_arm_matches_teffects_ra(obs, stata, model, y):
    ate = _ra(obs, y, model)
    want = stata[f"ra_{model}_ate"]
    assert ate.estimate == pytest.approx(want["b"], rel=RTOL)
    assert ate.se == pytest.approx(want["se"], rel=RTOL)
    assert ate.model_info["pomean0"] == pytest.approx(want["pom0"], rel=RTOL)
    assert ate.model_info["pomean0_se"] == pytest.approx(want["pom0_se"], rel=RTOL)
    att = _ra(obs, y, model, estimand="ATT")
    want = stata[f"ra_{model}_att"]
    assert att.estimate == pytest.approx(want["b"], rel=RTOL)
    assert att.se == pytest.approx(want["se"], rel=RTOL)
    assert att.model_info["pomean0"] == pytest.approx(want["pom0"], rel=RTOL)
    pom = stata[f"ra_{model}_pom"]
    assert ate.model_info["pomean1"] == pytest.approx(pom["pom1"], rel=RTOL)
    assert ate.model_info["pomean1_se"] == pytest.approx(pom["pom1_se"], rel=RTOL)


@pytest.mark.parametrize("model, y", GC + [("linear", "y")])
def test_gcomp_ratio_contrasts_match_nlcom(obs, stata, model, y):
    """Risk / rate ratio and odds ratio of the potential-outcome means."""
    want = stata[f"ra_{model}_pom"]
    rr = _ra(obs, y, model, contrast="ratio")
    assert rr.estimand == "ATE (ratio)"
    assert rr.estimate == pytest.approx(want["ratio"], rel=RTOL)
    assert rr.se == pytest.approx(want["ratio_se"], rel=RTOL)
    assert rr.model_info["log_se"] == pytest.approx(want["lnratio_se"], rel=RTOL)
    # the interval is built on the log scale
    z = 1.959963984540054
    lo, hi = rr.ci
    assert lo == pytest.approx(rr.estimate * np.exp(-z * want["lnratio_se"]), rel=1e-8)
    assert hi == pytest.approx(rr.estimate * np.exp(z * want["lnratio_se"]), rel=1e-8)
    if model == "logit":
        odds = _ra(obs, y, model, contrast="odds_ratio")
        assert odds.model_info["log_estimate"] == pytest.approx(want["lnor"], rel=RTOL)
        assert odds.model_info["log_se"] == pytest.approx(want["lnor_se"], rel=RTOL)
        assert odds.estimate == pytest.approx(np.exp(want["lnor"]), rel=RTOL)


@pytest.mark.parametrize("model, y", GC)
@pytest.mark.parametrize("estimand", ["ATE", "ATT", "ATC"])
def test_gcomp_pooled_glm_matches_marginaleffects(obs, ref, model, y, estimand):
    """One GLM with the treatment as a regressor, standardised."""
    want = ref["gcomp_pooled"][model][estimand]
    kw = dict(estimand=estimand, outcome_model=model, n_boot=2, seed=0)
    diff = sp.g_computation(obs, y, "a", XO, **kw)
    assert diff.estimate == pytest.approx(want["difference"], rel=RTOL)
    ratio = sp.g_computation(obs, y, "a", XO, contrast="ratio", **kw)
    assert ratio.estimate == pytest.approx(want["ratio"], rel=RTOL)
    if model == "logit":
        odds = sp.g_computation(obs, y, "a", XO, contrast="odds_ratio", **kw)
        assert odds.estimate == pytest.approx(want["odds_ratio"], rel=RTOL)
    assert diff.model_info["outcome_model"] == model


def test_gcomp_bootstrap_agrees_with_the_analytic_error_and_defaults_hold(obs):
    analytic = _ra(obs, "ybin", "logit", contrast="ratio")
    boot = sp.g_computation(
        obs,
        "ybin",
        "a",
        XO,
        by_arm=True,
        outcome_model="logit",
        contrast="ratio",
        n_boot=400,
        seed=11,
    )
    assert boot.estimate == analytic.estimate
    # 400 resamples: the two standard errors agree within Monte Carlo error
    assert boot.se == pytest.approx(analytic.se, rel=0.15)
    # default call: no new keys, same label
    plain = sp.g_computation(obs, "y", "a", XO, n_boot=20, seed=1)
    assert plain.estimand == "ATE"
    assert "outcome_model" not in plain.model_info
    assert "contrast" not in plain.model_info
    with pytest.raises(MethodIncompatibility, match="0/1 outcome"):
        sp.g_computation(obs, "y", "a", XO, outcome_model="logit")
    with pytest.raises(MethodIncompatibility, match="outcome_model must be"):
        sp.g_computation(obs, "ybin", "a", XO, outcome_model="probit")
    with pytest.raises(MethodIncompatibility, match="contrast must be"):
        sp.g_computation(obs, "ybin", "a", XO, contrast="rr")
    with pytest.raises(MethodIncompatibility, match="dose-response"):
        sp.g_computation(
            obs,
            "ybin",
            "x1",
            ["x2"],
            estimand="dose_response",
            treat_values=[0, 1],
            contrast="ratio",
        )
    with pytest.raises(MethodIncompatibility, match="strictly between 0 and 1"):
        _ra(obs, "ycnt", "poisson", contrast="odds_ratio")


# ---------------------------------------------------------------------------
# 4. IPW with truncated weights
# ---------------------------------------------------------------------------

TRUNC = [
    (est, at, side)
    for est in ("ATE", "ATT", "ATC")
    for at, side in ((0.95, "both"), (0.99, "both"), (0.9, "upper"))
]


@pytest.mark.parametrize("estimand, at, side", TRUNC)
def test_ipw_truncate_matches_weightit_trim(obs, ref, estimand, at, side):
    """``WeightIt::trim`` and the HC0 error of the weighted regression."""
    want = ref["ipw_truncate"][f"{estimand}_{at}_{side}"]
    truncate = (1 - at, at) if side == "both" else (0.0, at)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.ipw(
            obs,
            "y",
            "a",
            XO,
            estimand=estimand,
            se_method="sandwich",
            truncate=truncate,
        )
    assert fit.estimate == pytest.approx(want["estimate"], rel=RTOL)
    assert fit.se == pytest.approx(want["se_hc0"], rel=RTOL)
    lower, upper = fit.model_info["truncation_bounds"]
    assert upper == pytest.approx(want["upper"], rel=1e-10)
    if side == "both":
        assert lower == pytest.approx(want["lower"], rel=1e-10)
        assert fit.model_info["n_truncated_low"] > 0
    else:
        assert fit.model_info["n_truncated_low"] == 0
    assert fit.model_info["n_truncated_high"] > 0
    assert fit.model_info["truncate"] == truncate


def test_ipw_truncate_none_is_the_untruncated_estimator_and_refusals(obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = sp.ipw(obs, "y", "a", XO, se_method="sandwich")
        wide = sp.ipw(obs, "y", "a", XO, se_method="sandwich", truncate=(0.0, 1.0))
        seeded = sp.ipw(
            obs, "y", "a", XO, truncate=(0.01, 0.99), n_bootstrap=50, seed=3
        )
        again = sp.ipw(obs, "y", "a", XO, truncate=(0.01, 0.99), n_bootstrap=50, seed=3)
    # (0, 1) truncates nothing: the same point estimate
    assert wide.estimate == pytest.approx(base.estimate, rel=1e-13)
    assert "truncate" not in base.model_info
    assert seeded.se == again.se > 0
    for bad in ((0.99, 0.01), (0.5, 0.5), (-0.1, 0.9), 0.99, (0.1, 1.2)):
        with pytest.raises(MethodIncompatibility, match="truncate must be"):
            sp.ipw(obs, "y", "a", XO, truncate=bad)
    with pytest.raises(MethodIncompatibility, match="bounded by construction"):
        sp.ipw(obs, "y", "a", XO, estimand="ATO", truncate=(0.01, 0.99))
    with pytest.raises(MethodIncompatibility, match="weights= or design="):
        sp.ipw(obs.assign(w=1.0), "y", "a", XO, weights="w", truncate=(0.01, 0.99))


# ---------------------------------------------------------------------------
# 5. GEE: QIC
# ---------------------------------------------------------------------------

GEE_Y = {"binomial": "ybin", "poisson": "ycnt", "gaussian": "y"}


@pytest.mark.parametrize("family", list(GEE_Y))
@pytest.mark.parametrize("corstr", ["independence", "exchangeable", "ar1"])
def test_gee_qic_matches_geepack(panel, ref, family, corstr):
    want = ref["gee_qic"][f"{family}_{corstr}"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.gee(
            f"{GEE_Y[family]} ~ trt + visit + age",
            panel,
            id="pid",
            time="visit",
            family=family,
            corstr=corstr,
            tol=1e-13,
        )
    tol = 1e-10 if corstr == "independence" else 1e-6
    np.testing.assert_allclose(fit.params, want["coef"], rtol=tol)
    d = fit.diagnostics
    assert d["QIC"] == pytest.approx(want["QIC"], rel=tol)
    assert d["QICu"] == pytest.approx(want["QICu"], rel=tol)
    assert d["Quasi-likelihood"] == pytest.approx(want["quasi"], rel=tol)
    assert d["CIC"] == pytest.approx(want["CIC"], rel=tol)
    # the definitions
    assert d["QIC"] == pytest.approx(
        -2 * d["Quasi-likelihood"] + 2 * d["CIC"], rel=1e-13
    )
    assert d["QICu"] == pytest.approx(-2 * d["Quasi-likelihood"] + 2 * 4, rel=1e-13)


def test_gee_quasi_likelihood_is_the_glm_log_likelihood_kernel(panel):
    """Bernoulli: the quasi-likelihood is the log-likelihood itself."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.gee("ybin ~ trt + visit + age", panel, id="pid", family="binomial")
        logit = sp.logit("ybin ~ trt + visit + age", data=panel)
        nb = sp.gee(
            "ycnt ~ trt + visit + age", panel, id="pid", family="negative_binomial"
        )
    ll = float(logit.diagnostics["Log-Likelihood"])
    assert fit.diagnostics["Quasi-likelihood"] == pytest.approx(ll, rel=1e-9)
    assert "QIC" not in nb.diagnostics  # no quasi-likelihood for this family


# ---------------------------------------------------------------------------
# 7. Standard errors of the variance components
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method, key", [("reml", "reml"), ("ml", "mle")])
def test_mixed_random_intercept_variance_ses_match_stata(panel, stata, method, key):
    fit = sp.mixed(panel, "y", XL, "pid", method=method)
    want = stata[f"mixed_ri_{key}"]
    vc, se = fit.variance_components, fit.variance_components_se
    assert list(se) == list(vc)
    assert vc["var(_cons)"] == pytest.approx(want["var_cons"], rel=1e-6)
    assert se["var(_cons)"] == pytest.approx(want["var_cons_se"], rel=1e-6)
    assert vc["var(Residual)"] == pytest.approx(want["var_resid"], rel=1e-6)
    assert se["var(Residual)"] == pytest.approx(want["var_resid_se"], rel=1e-6)
    assert fit.log_likelihood >= want["ll"] - 1e-9
    assert "Std.Err" in fit.summary()


def test_mixed_random_slope_variance_ses_match_stata(panel, stata):
    fit = sp.mixed(panel, "y", XL, "pid", x_random=["visit"], method="ml")
    want = stata["mixed_rs_mle"]
    pairs = {
        "var(_cons)": "var_cons",
        "var(visit)": "var_visit",
        "cov(_cons,visit)": "cov",
        "corr(_cons,visit)": "corr",
        "var(Residual)": "var_resid",
    }
    assert list(fit.variance_components_se) == list(fit.variance_components)
    for ours, theirs in pairs.items():
        assert fit.variance_components[ours] == pytest.approx(want[theirs], rel=1e-3)
        assert fit.variance_components_se[ours] == pytest.approx(
            want[f"{theirs}_se"], rel=1e-3
        )
    # our optimum is at least as high as Stata's stopping point
    assert fit.log_likelihood >= want["ll"] - 1e-9


def test_mixed_variance_ses_equal_the_balanced_one_way_closed_form():
    """Balanced one-way layout: REML is ANOVA, with known sampling variances.

    With ``a`` groups of size ``n``, ``MSE`` and ``MSA`` are independent
    scaled chi-squares, ``var(Residual) = MSE`` and ``var(_cons) = (MSA -
    MSE) / n``, so ``Var(MSE) = 2 MSE^2 / (a (n - 1))`` and ``Var(var(_cons))
    = (2 / n^2) [MSA^2 / (a - 1) + MSE^2 / (a (n - 1))]`` at the estimates.
    """
    rng = np.random.default_rng(5)
    a, n = 40, 6
    g = np.repeat(np.arange(a), n)
    y = 1.0 + rng.normal(0, 1.0, a)[g] + rng.normal(0, 0.7, a * n)
    df = pd.DataFrame({"y": y, "g": g})
    fit = sp.mixed(df, "y", [], "g", method="reml")
    means = df.groupby("g")["y"].mean()
    msa = n * float(((means - y.mean()) ** 2).sum()) / (a - 1)
    mse = float(((y - means.loc[g].to_numpy()) ** 2).sum()) / (a * (n - 1))
    assert fit.variance_components["var(Residual)"] == pytest.approx(mse, rel=1e-7)
    assert fit.variance_components["var(_cons)"] == pytest.approx(
        (msa - mse) / n, rel=1e-7
    )
    se = fit.variance_components_se
    assert se["var(Residual)"] == pytest.approx(
        np.sqrt(2 * mse**2 / (a * (n - 1))), rel=1e-5
    )
    assert se["var(_cons)"] == pytest.approx(
        np.sqrt(2 / n**2 * (msa**2 / (a - 1) + mse**2 / (a * (n - 1)))), rel=1e-5
    )


def test_meglm_variance_ses_match_stata(panel, stata):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lap = sp.meglm(panel, "ybin", XL, "pid", family="binomial")
        agq = sp.meglm(panel, "ybin", XL, "pid", family="binomial", nAGQ=7)
        pois = sp.meglm(panel, "ycnt", XL, "pid", family="poisson", nAGQ=7)
    want = stata["melogit_laplace"]
    assert lap.variance_components["var(_cons)"] == pytest.approx(
        want["var_cons"], rel=1e-7
    )
    assert lap.variance_components_se["var(_cons)"] == pytest.approx(
        want["var_cons_se"], rel=1e-7
    )
    assert lap.log_likelihood == pytest.approx(want["ll"], rel=1e-10)
    for fit, key in ((agq, "melogit"), (pois, "mepoisson")):
        want = stata[key]
        assert fit.variance_components["var(_cons)"] == pytest.approx(
            want["var_cons"], rel=1e-3
        )
        assert fit.variance_components_se["var(_cons)"] == pytest.approx(
            want["var_cons_se"], rel=1e-3
        )
        assert fit.log_likelihood >= want["ll"] - 1e-9
    assert "Std.Err" in lap.summary()


# ---------------------------------------------------------------------------
# 8. Overlap weights: sandwich standard error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weight, estimand",
    [("overlap", "ATO"), ("IPW", "ATE"), ("treated", "ATT"), ("entropy", "ENTROPY")],
)
def test_overlap_weights_sandwich_matches_psweight(obs, ref, weight, estimand):
    want = ref["psweight"][weight]
    fit = sp.overlap_weights(obs, "y", "a", XO, estimand=estimand, se_method="sandwich")
    assert fit.estimate == pytest.approx(want["estimate"], rel=RTOL)
    assert fit.se == pytest.approx(want["se"], rel=RTOL)
    assert fit.model_info["se_method"] == "sandwich"


def test_overlap_weights_sandwich_equals_teffects_ipw_and_refusals(obs):
    """For the ATE the same estimator is ``sp.ipw(se_method='sandwich')``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ipw = sp.ipw(obs, "y", "a", XO, se_method="sandwich")
    ate = sp.overlap_weights(obs, "y", "a", XO, estimand="ATE", se_method="sandwich")
    assert ate.estimate == pytest.approx(ipw.estimate, rel=1e-10)
    assert ate.se == pytest.approx(ipw.se, rel=1e-8)
    boot = sp.overlap_weights(obs, "y", "a", XO, n_bootstrap=300, seed=2)
    ato = sp.overlap_weights(obs, "y", "a", XO, se_method="sandwich")
    assert boot.estimate == ato.estimate
    assert boot.se == pytest.approx(ato.se, rel=0.15)  # Monte Carlo error
    assert "se_method" not in boot.model_info
    with pytest.raises(MethodIncompatibility, match="kink"):
        sp.overlap_weights(obs, "y", "a", XO, estimand="matching", se_method="sandwich")
    with pytest.raises(MethodIncompatibility, match="clipped"):
        sp.overlap_weights(obs, "y", "a", XO, trim=0.05, se_method="sandwich")
    with pytest.raises(MethodIncompatibility, match="se_method must be"):
        sp.overlap_weights(obs, "y", "a", XO, se_method="analytic")


# ---------------------------------------------------------------------------
# 9. sp.mr('all') with summary arrays
# ---------------------------------------------------------------------------


def test_mr_all_accepts_the_summary_arrays():
    df = pd.read_csv(FIX / "mr_ldl.csv")
    cols = [c for c in df.columns if df[c].dtype.kind == "f"][:4]
    bx, sx, by, sy = (df[c].to_numpy() for c in cols)
    from_frame = sp.mr(
        "all",
        data=df,
        beta_exposure=cols[0],
        se_exposure=cols[1],
        beta_outcome=cols[2],
        se_outcome=cols[3],
        seed=7,  # the weighted-median standard error is a bootstrap
    )
    from_arrays = sp.mr(
        "all",
        beta_exposure=bx,
        se_exposure=sx,
        beta_outcome=by,
        se_outcome=sy,
        seed=7,
    )
    pd.testing.assert_frame_equal(from_arrays.estimates, from_frame.estimates)
    assert from_arrays.n_snps == len(df)
    # each row is the single-method call on the same arrays
    ivw = sp.mr("ivw", beta_exposure=bx, se_exposure=sx, beta_outcome=by, se_outcome=sy)
    row = from_arrays.estimates.iloc[0]
    assert float(row["estimate"]) == pytest.approx(float(ivw["estimate"]), rel=1e-12)
    with pytest.raises(MethodIncompatibility, match="no data= was passed"):
        sp.mr("all", beta_exposure="bx", se_exposure=sx, beta_outcome=by, se_outcome=sy)
    with pytest.raises(MethodIncompatibility, match="equal length"):
        sp.mr(
            "all", beta_exposure=bx[:-1], se_exposure=sx, beta_outcome=by, se_outcome=sy
        )
