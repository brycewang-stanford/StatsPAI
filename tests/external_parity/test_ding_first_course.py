"""Peng Ding, *A First Course in Causal Inference* (2024), chapter by chapter.

The book's replication files (https://doi.org/10.7910/DVN/ZX3VEV) are 28 R
programs that write their estimators out by hand: randomization tests,
Neyman and Lin estimators, stratification, weighting, doubly robust
estimation, matching, E-values, Rosenbaum bounds, discontinuities,
instruments, principal strata, mediation. The answer key is
``data/ding_first_course_R.json``, made by rerunning the deterministic part
of those programs (``ding_first_course_reference.R`` next to this file).
Here every number is recomputed with the ``sp.*`` function a user would
reach for.

Neither the programs nor the data are redistributed. Point
``STATSPAI_DING_DIR`` at the unzipped dataverse folder, run the R script
once (it adds a ``_statspai`` subfolder with ``Matching::lalonde``, the
matched pairs of chapter 19 and the JOBS II model matrices), then

    STATSPAI_DING_DIR=/path/to/dataverse_ZX3VEV \\
        pytest tests/external_parity/test_ding_first_course.py

It is skipped otherwise. What the pass found is in
``docs/dev/2026-10-05-ding-first-course-review.md``.

Tolerances. Closed-form quantities (means, OLS, sandwich variances,
matching, E-values) are held to 1e-8 relative; anything that goes through a
logit fitted by R's ``glm`` at its default 1e-8 deviance tolerance to 1e-6;
``rdrobust`` to 1e-6.
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

ROOT = os.environ.get("STATSPAI_DING_DIR")
pytestmark = pytest.mark.skipif(
    not ROOT or not (Path(ROOT) / "_statspai" / "lalonde_matching.csv").is_file(),
    reason="set STATSPAI_DING_DIR to Ding's dataverse folder and run "
    "ding_first_course_reference.R once",
)

EXACT = 1e-8
GLM = 1e-6


@pytest.fixture(scope="module")
def R():
    path = Path(__file__).parent / "data" / "ding_first_course_R.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _path(name: str) -> Path:
    return Path(ROOT) / name


@pytest.fixture(scope="module")
def lalonde():
    return pd.read_csv(_path("_statspai/lalonde_matching.csv"))


@pytest.fixture(scope="module")
def cps():
    d = pd.read_csv(_path("cps1re74.csv"), sep=r"\s+")
    d["u74"] = (d.re74 == 0).astype(float)
    d["u75"] = (d.re75 == 0).astype(float)
    return d


@pytest.fixture(scope="module")
def nhanes():
    d = pd.read_csv(_path("nhanes_bmi.csv")).iloc[:, 1:]
    return d, [c for c in d.columns if c not in ("BMI", "School_meal")]


@pytest.fixture(scope="module")
def jobs():
    d = pd.read_csv(_path("_statspai/jobs_X.csv"))
    return d, [c for c in d.columns if c not in ("treat", "comply", "job_seek", "depress2")]


LALONDE_X = ["age", "educ", "black", "hisp", "married", "nodegr", "re74", "re75"]
CPS_X = [
    "age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75",
    "u74", "u75",
]  # fmt: skip


# -- Part I: randomized experiments ------------------------------------------


def test_ch01_regression_on_the_lalonde_observational_data(R, cps):
    fit = sp.regress("re78 ~ treat + " + " + ".join(CPS_X), cps)
    assert fit.params["treat"] == pytest.approx(R["ch1_lm_all"][0], rel=EXACT)
    assert fit.std_errors["treat"] == pytest.approx(R["ch1_lm_all"][1], rel=EXACT)
    raw = sp.regress("re78 ~ treat", cps)
    assert raw.params["treat"] == pytest.approx(R["ch1_lm_treat"][0], rel=EXACT)


def test_ch03_randomization_test_statistics(R, lalonde):
    t_eq, t_uneq, W, D, p_eq, p_uneq = R["ch3"][:6]
    pooled = sp.ttest(lalonde, "re78", by="treat")
    welch = sp.ttest(lalonde, "re78", by="treat", unequal=True)
    assert abs(pooled.statistic) == pytest.approx(t_eq, rel=EXACT)
    assert abs(welch.statistic) == pytest.approx(t_uneq, rel=EXACT)
    assert pooled.pvalue == pytest.approx(p_eq, rel=EXACT)
    assert welch.pvalue == pytest.approx(p_uneq, rel=EXACT)
    n, n1 = len(lalonde), int(lalonde.treat.sum())
    kw = dict(n_perms=2000, seed=1)
    t = sp.ri_test(lalonde, "re78", "treat", stat="t", **kw)
    assert t["observed"] == pytest.approx(t_uneq, rel=EXACT)
    rank = sp.ri_test(lalonde, "re78", "treat", stat="rank_sum", **kw)
    assert rank["observed"] + n1 * (n + 1) / 2 - n1 * (n1 + 1) / 2 == pytest.approx(W)
    ks = sp.ri_test(lalonde, "re78", "treat", stat="ks", **kw)
    assert ks["observed"] == pytest.approx(D, rel=EXACT)
    # The book's Monte Carlo p-values (10^4 draws) are 0.002, 0.002, 0.006
    # and 0.040; 2000 draws give them to about +-0.005.
    assert t["p_one_sided"] < 0.01 and rank["p_one_sided"] < 0.015
    assert 0.02 < ks["p_one_sided"] < 0.06


def test_ch04_neyman_and_the_hc_family(R, lalonde):
    tau, se, ols, hc3, hc0, hc2 = R["ch4"]
    dim = sp.difference_in_means(lalonde, "re78", "treat")
    assert dim.estimate == pytest.approx(tau, rel=EXACT)
    assert dim.se == pytest.approx(se, rel=EXACT)
    for robust, ref in [("nonrobust", ols), ("hc0", hc0), ("hc2", hc2), ("hc3", hc3)]:
        fit = sp.regress("re78 ~ treat", lalonde, robust=robust)
        assert fit.std_errors["treat"] == pytest.approx(ref, rel=EXACT)
    # Neyman's variance estimator is HC2 exactly
    assert hc2 == pytest.approx(se, rel=1e-12)


def test_ch05_stratified_experiments(R):
    penn = pd.read_csv(_path("Penn46_ascii.txt"), sep=r"\s+")
    penn["ly"] = np.log(penn.duration)
    res = sp.difference_in_means(penn, "ly", "treatment", blocks="quarter")
    assert res.estimate == pytest.approx(R["ch5_penn"][0], rel=EXACT)
    assert res.se == pytest.approx(R["ch5_penn"][1], rel=EXACT)

    chong = pd.read_csv(_path("_statspai/chong.csv"))
    phys = chong[chong.treatment != "Soccer Player"].copy()
    phys["z"] = (phys.treatment == "Physician").astype(int)
    strat = sp.difference_in_means(phys, "gradesq34", "z", blocks="class_level")
    assert strat.estimate == pytest.approx(R["ch5_chong_S"][0], rel=EXACT)
    assert strat.se == pytest.approx(R["ch5_chong_S"][1], rel=EXACT)
    phys["cell"] = phys.class_level.astype(str) + phys.anemic
    post = sp.difference_in_means(phys, "gradesq34", "z", blocks="cell")
    assert post.estimate == pytest.approx(R["ch5_chong_SPS"][0], rel=EXACT)
    assert post.se == pytest.approx(R["ch5_chong_SPS"][1], rel=EXACT)


def test_ch06_and_ch09_lin_estimator(R, lalonde):
    star = pd.read_stata(_path("star.dta"))
    star = star[(star.control == 1) | (star.sfsp == 1)].copy()
    star["y"] = star.GPA_year1.fillna(star.GPA_year1.mean())
    unadj, se_unadj, adj, se_adj = R["ch6"]
    raw = sp.regress("y ~ sfsp", star, robust="hc2")
    assert raw.params["sfsp"] == pytest.approx(unadj, rel=1e-7)  # float32 .dta
    assert raw.std_errors["sfsp"] == pytest.approx(se_unadj, rel=1e-7)
    lin = sp.lm_lin(star, "y", "sfsp", ["female", "gpa0"])
    assert lin.estimate == pytest.approx(adj, rel=1e-7)
    assert lin.se == pytest.approx(se_adj, rel=1e-7)

    est, se_ehw, se_super = R["ch9_lalonde"]
    fixed = sp.lm_lin(lalonde, "re78", "treat", LALONDE_X, vce="hc3")
    superpop = sp.lm_lin(
        lalonde, "re78", "treat", LALONDE_X, vce="hc3", superpopulation=True
    )
    assert fixed.estimate == pytest.approx(est, rel=EXACT)
    assert fixed.se == pytest.approx(se_ehw, rel=EXACT)
    assert superpop.se == pytest.approx(se_super, rel=EXACT)


def test_ch07_matched_pairs(R):
    # Darwin's Zea mays pairs (Fisher 1935), in inches
    d = np.array(
        [6.125, -8.375, 1, 2, 0.75, 2.875, 3.5, 5.125, 1.75, 3.625, 7, 3, 9.375,
         7.5, -6]
    )  # fmt: skip
    long = pd.DataFrame(
        {
            "pair": np.repeat(np.arange(15), 2),
            "treat": np.tile([1, 0], 15),
            "y": np.column_stack([d, np.zeros(15)]).ravel(),
        }
    )
    # 2^15 sign flips, all enumerated
    ri = sp.ri_test(long, "y", "treat", strata="pair", n_perms=40000)
    assert ri["exact"] and ri["n_perms"] == 2**15
    assert ri["p_one_sided"] == pytest.approx(R["ch7_darwin_p"], rel=1e-12)
    pair = sp.difference_in_means(long, "y", "treat", blocks="pair")
    assert pair.estimate == pytest.approx(d.mean(), rel=1e-12)
    assert pair.se == pytest.approx(d.std(ddof=1) / np.sqrt(15), rel=1e-12)


def test_ch08_covariate_adjusted_randomization_statistics(R):
    chong = pd.read_csv(_path("_statspai/chong.csv"))
    soccer = chong[chong.treatment != "Physician"].copy()
    soccer["z"] = (soccer.treatment == "Soccer Player").astype(int)
    soccer["x"] = (soccer.anemic == "Yes").astype(float)
    ref = np.asarray(R["ch8_soccer_k"], dtype=float)
    if ref.shape == (7, 5):
        ref = ref.T  # one row per class level
    for k in range(1, 6):
        sub = soccer[soccer.class_level == k]
        tau_n, se_n, t_n, tau_l, se_l, t_l, n = ref[k - 1]
        assert len(sub) == n
        dim = sp.difference_in_means(sub, "gradesq34", "z")
        lin = sp.lm_lin(sub, "gradesq34", "z", ["x"])
        assert (dim.estimate, dim.se) == pytest.approx((tau_n, se_n), rel=EXACT)
        assert (lin.estimate, lin.se) == pytest.approx((tau_l, se_l), rel=EXACT)
        kw = dict(n_perms=200, seed=k)
        assert sp.ri_test(sub, "gradesq34", "z", stat="t", **kw)[
            "observed"
        ] == pytest.approx(t_n, rel=EXACT)
        assert sp.ri_test(sub, "gradesq34", "z", stat="lin_t", covariates=["x"], **kw)[
            "observed"
        ] == pytest.approx(t_l, rel=EXACT)
    tau_s, se_s, lin_s, se_lin_s = R["ch8_soccer"]
    pooled = sp.difference_in_means(soccer, "gradesq34", "z", blocks="class_level")
    assert (pooled.estimate, pooled.se) == pytest.approx((tau_s, se_s), rel=EXACT)
    # Lin's estimator stratum by stratum, combined by stratum shares
    lin = sp.lm_lin(soccer, "gradesq34", "z", ["x"], blocks="class_level")
    assert (lin.estimate, lin.se) == pytest.approx((lin_s, se_lin_s), rel=EXACT)
    phys = chong[chong.treatment != "Soccer Player"].copy()
    phys["z"] = (phys.treatment == "Physician").astype(int)
    phys["x"] = (phys.anemic == "Yes").astype(float)
    lin = sp.lm_lin(phys, "gradesq34", "z", ["x"], blocks="class_level")
    assert (lin.estimate, lin.se) == pytest.approx(tuple(R["ch8_phys"][2:]), rel=EXACT)


# -- Part III: observational studies -----------------------------------------


def test_ch11_weighting_with_truncated_propensity_scores(R, nhanes):
    d, x = nhanes
    for (ht, hajek), trim in zip(np.asarray(R["ch11_ipw"]).T, [0, 0.01, 0.05, 0.1]):
        kw = dict(trim=trim, n_bootstrap=5, seed=0)
        a = sp.ipw(d, "BMI", "School_meal", x, normalize=False, **kw)
        b = sp.ipw(d, "BMI", "School_meal", x, normalize=True, **kw)
        assert a.estimate == pytest.approx(ht, rel=GLM)
        assert b.estimate == pytest.approx(hajek, rel=GLM)


def test_ch12_outcome_regression_and_doubly_robust(R, nhanes):
    d, x = nhanes
    reg, _, _, dr = R["ch12"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sp.g_computation(
            d, "BMI", "School_meal", x, by_arm=True, n_boot=5, seed=0
        )
        a = sp.aipw(d, "BMI", "School_meal", x, cross_fit=False)
    assert g.estimate == pytest.approx(reg, rel=EXACT)
    assert a.estimate == pytest.approx(dr, rel=GLM)


def test_ch13_effect_on_the_treated(R, nhanes):
    d, x = nhanes
    reg0, reg, ht, hajek, dr = R["ch13"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sp.g_computation(
            d, "BMI", "School_meal", x, estimand="ATT", by_arm=True, n_boot=5, seed=0
        )
        a = sp.aipw(d, "BMI", "School_meal", x, estimand="ATT", cross_fit=False)
    kw = dict(estimand="ATT", n_bootstrap=5, seed=0)
    assert g.estimate == pytest.approx(reg, rel=EXACT)
    assert sp.ipw(d, "BMI", "School_meal", x, normalize=False, **kw).estimate == (
        pytest.approx(ht, rel=GLM)
    )
    assert sp.ipw(d, "BMI", "School_meal", x, normalize=True, **kw).estimate == (
        pytest.approx(hajek, rel=GLM)
    )
    assert a.estimate == pytest.approx(dr, rel=GLM)


def test_ch15_matching(R, lalonde, cps):
    est, se, _ = R["ch15_exp"]
    kw = dict(method="nnmatch", estimand="ATT", metric="ivariance", vce="iid")
    m = sp.match(lalonde, "re78", "treat", LALONDE_X, bias_adjust=True, **kw)
    assert (m.estimate, m.se) == pytest.approx((est, se), rel=EXACT)
    est, se = R["ch15_obs_adj"][:2]
    m = sp.match(cps, "re78", "treat", CPS_X, bias_adjust=True, **kw)
    assert (m.estimate, m.se) == pytest.approx((est, se), rel=EXACT)
    est, se = R["ch19_obs"][:2]
    m = sp.match(cps, "re78", "treat", CPS_X, bias_adjust=False, **kw)
    assert (m.estimate, m.se) == pytest.approx((est, se), rel=EXACT)


def test_ch17_e_values(R):
    # Hammond and Horn's smoking table
    rr = (397 / (397 + 78557)) / (51 / (51 + 108778))
    ev = sp.evalue(rr)
    assert ev["evalue_estimate"] == pytest.approx(rr + np.sqrt(rr * (rr - 1)))
    assert ev["evalue_estimate"] == pytest.approx(20.95, abs=5e-3)
    est, lower = R["ch17"]
    out = sp.evalue(est, ci=(lower, 2 * est), measure="OR", rare=True)
    assert out["evalue_estimate"] == pytest.approx(est + np.sqrt(est * (est - 1)))
    assert out["evalue_ci"] == pytest.approx(lower + np.sqrt(lower * (lower - 1)))


def test_ch19_rosenbaum_bounds_for_the_mean_difference(R):
    pairs = pd.read_csv(_path("_statspai/lalonde_pairs.csv"))
    assert len(pairs) == R["ch19_npairs"]
    res = sp.rosenbaum_bounds(
        pairs.iloc[:, 0], pairs.iloc[:, 1], method="t", gamma_grid=[1, 1.1, 1.2, 1.3]
    )
    np.testing.assert_allclose(res.pvalue_upper, R["ch19_p"], rtol=1e-10)
    fine = sp.rosenbaum_bounds(
        pairs.iloc[:, 0],
        pairs.iloc[:, 1],
        method="t",
        gamma_grid=np.round(np.arange(1, 1.4005, 0.001), 3),
    )
    assert fine.gamma_critical == pytest.approx(R["ch19_gammastar"])


# -- Part IV and V: discontinuities and instruments --------------------------


def _rd_rows(res):
    conv, rob = res.model_info["conventional"], res.model_info["robust"]
    return conv["estimate"], conv["se"], rob["estimate"], rob["se"]


def test_ch20_and_ch24_regression_discontinuity(R):
    house = pd.read_csv(_path("house.csv")).iloc[:, 1:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sharp = sp.rdrobust(house, "y", "x")
        road = pd.read_csv(_path("indianroad.csv"))
        road["runv"] = road.left + road.right
        fz_road = sp.rdrobust(road, "occupation_index_andrsn", "runv", fuzzy="r2012")
        italy = pd.read_csv(_path("italy.csv"))
        fz_italy = sp.rdrobust(italy, "outcome", "rv0", fuzzy="D")
    ref = np.asarray(R["ch20_rd"])  # rows conventional / bias-corrected / robust
    assert _rd_rows(sharp) == pytest.approx(
        (ref[0, 0], ref[0, 1], ref[2, 0], ref[2, 1]), rel=GLM
    )
    assert sharp.model_info["bandwidth_h"] == pytest.approx(R["ch20_bw"][0][0], rel=GLM)
    for res, key in [(fz_road, "ch24_road_rd"), (fz_italy, "ch24_italy_rd")]:
        ref = np.asarray(R[key])
        assert _rd_rows(res) == pytest.approx(
            (ref[0, 0], ref[0, 1], ref[2, 0], ref[2, 1]), rel=GLM
        )
    # the book's fixed-bandwidth version: two-stage least squares with HC2
    for h, est, se, n in zip([10, 40, 80], *R["ch24_road_h"]):
        sub = road[road.runv.abs() <= h]
        fit = sp.ivreg(
            "occupation_index_andrsn ~ (r2012 ~ t) + left + right", sub, robust="hc2"
        )
        assert len(sub) == n
        assert fit.params["r2012"] == pytest.approx(est, rel=EXACT)
        assert fit.std_errors["r2012"] == pytest.approx(se, rel=EXACT)


def test_ch21_wald_estimator_and_the_far_interval(R, jobs):
    d, x = jobs
    tau_d, tau_y, cace, se = R["ch21_wald"]
    # The delta-method standard error of the Wald estimator is HC2.
    fit = sp.ivreg("job_seek ~ (comply ~ treat)", d, robust="hc2")
    assert fit.params["comply"] == pytest.approx(cace, rel=EXACT)
    assert fit.std_errors["comply"] == pytest.approx(se, rel=EXACT)
    # Fieller-Anderson-Rubin: the book inverts a z-test on a 0.001 grid.
    ar = sp.anderson_rubin_test(d, "job_seek", "comply", ["treat"], ar_vcov="HC2")
    assert ar["ar_ci"] == pytest.approx(tuple(R["ch21_far"]), abs=1.5e-3)
    # With covariates the Wald ratio of two Lin estimators
    num = sp.lm_lin(d, "job_seek", "treat", x).estimate
    den = sp.lm_lin(d, "comply", "treat", x).estimate
    assert (den, num) == pytest.approx(tuple(R["ch21_lin"]), rel=EXACT)


def test_ch23_card_two_stage_least_squares_and_far(R):
    card = pd.read_csv(_path("card1995.csv"))
    x = [
        "exper", "expersq", "black", "south", "smsa", "reg661", "reg662", "reg663",
        "reg664", "reg665", "reg666", "reg667", "reg668", "smsa66",
    ]  # fmt: skip
    est, se = R["ch23_tsls"]
    fit = sp.ivreg("lwage ~ (educ ~ nearc4) + " + " + ".join(x), card, robust="hc0")
    assert fit.params["educ"] == pytest.approx(est, rel=EXACT)
    assert fit.std_errors["educ"] == pytest.approx(se, rel=EXACT)
    ar = sp.anderson_rubin_test(
        card, "lwage", "educ", ["nearc4"], exog=x, ar_vcov="HC3"
    )
    _, lo, hi = R["ch23_far"]  # the book's interval, on a 0.001 grid
    assert ar["ar_ci"] == pytest.approx((lo, hi), abs=1e-3)
    assert ar["beta_2sls"] == pytest.approx(est, rel=EXACT)


def test_ch25_mendelian_randomization(R):
    snp = pd.read_csv(_path("mr_bmisbp.csv"))
    args = [
        snp[c].to_numpy()
        for c in ("beta.exposure", "beta.outcome", "se.exposure", "se.outcome")
    ]
    fixed = sp.mr_ivw(*args, model="fixed")
    random = sp.mr_ivw(*args, model="random")
    assert (fixed["estimate"], fixed["se"]) == pytest.approx(
        tuple(R["ch25_fw"][:2]), rel=EXACT
    )
    # the weighted regression through the origin, residual scale estimated
    assert random["se"] == pytest.approx(R["ch25_egger0"][0][1], rel=EXACT)
    # The book runs Egger regression on the SNPs as coded. StatsPAI first
    # orients every SNP to a positive exposure association, as Bowden et
    # al. define the method; 78 of the 160 are flipped, so the two differ.
    assert (snp["beta.exposure"] < 0).sum() == 78
    egger = sp.mr_egger(*args)
    assert egger["estimate"] != pytest.approx(R["ch25_egger"][1][0], rel=0.05)


def test_ch26_principal_strata(R, jobs):
    d, x = jobs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = sp.principal_strat(
            d, "job_seek", "treat", "comply", covariates=x,
            method="principal_score", n_boot=20, seed=0,
        )  # fmt: skip
    eff = res.effects.set_index("stratum")["estimate"]
    # The book divides the weighted control sum by the complier share of the
    # treated arm; StatsPAI normalises the weights (Hajek), so adding a
    # constant to the outcome changes nothing. Both decompose the same
    # intention-to-treat effect.
    share = res.strata_proportions
    itt = R["ch21_wald"][1]
    book = R["ch26_psw"]
    pi_c = d.comply[d.treat == 1].mean()
    assert pi_c * book[0] + (1 - pi_c) * book[1] == pytest.approx(itt, abs=2e-3)
    assert share["complier"] * eff["Complier PCE"] + share["never-taker"] * eff[
        "Never-taker PCE"
    ] == pytest.approx(itt, abs=2e-3)


def test_ch27_baron_kenny(R):
    d = pd.read_csv(_path("_statspai/jobs_X2.csv"))
    x = [c for c in d.columns if c not in ("treat", "job_seek", "depress2")]
    nde, nie, _, _ = R["ch27"]
    res = sp.mediate(d, "depress2", "treat", "job_seek", covariates=x, inference="delta")
    eff = res.detail.set_index("effect")["estimate"]
    assert eff["ACME (indirect)"] == pytest.approx(nie, rel=EXACT)
    assert eff["ADE (direct)"] == pytest.approx(nde, rel=EXACT)
