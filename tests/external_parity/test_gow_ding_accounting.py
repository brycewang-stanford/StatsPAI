"""Gow and Ding, *Empirical Research in Accounting: Tools and Methods*.

The book's code is R. Most chapters draw on CRSP and Compustat through
WRDS and cannot be rerun without a subscription; the answer key
``data/gow_ding_accounting_R.json`` covers what can: the chapters on
statistical inference, regression, panel data, instrumental variables,
extreme values and generalized linear models, and the simulation or
packaged-data parts of four others
(``gow_ding_accounting_reference.R`` next to this file, R 4.5.2 with
``farr`` 1.0.9). Here every number is recomputed with the ``sp.*`` call a
reader of the book would reach for.

The data come from the book's companion package and are not redistributed
here. Run the R script once with ``STATSPAI_GOW_DING_DIR`` pointing at an
empty folder (it writes the data sets there as CSV), then

    STATSPAI_GOW_DING_DIR=/that/folder \\
        pytest tests/external_parity/test_gow_ding_accounting.py

It is skipped otherwise. What the pass found is in
``docs/dev/2026-10-06-gow-ding-accounting-review.md``; the same functions
are tested on committed synthetic files in
``tests/reference_parity/test_accounting_research_parity.py``.

Tolerances. ``EXACT`` (1e-8 relative) for closed forms. ``GLM`` (1e-6) for
what R reaches by iteration at its own tolerance; the Poisson models of the
penalties chapter are compared at 1e-5 because ``glm`` stops them at a
deviance change of 1e-8.
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

DATA_DIR = os.environ.get("STATSPAI_GOW_DING_DIR", "")
pytestmark = pytest.mark.skipif(
    not DATA_DIR or not (Path(DATA_DIR) / "got.csv").exists(),
    reason="STATSPAI_GOW_DING_DIR is not set (see the module docstring)",
)
EXACT = 1e-8
GLM = 1e-6
KEEP = ["big_n", "cfo", "size", "lev", "mtb"]
COMP = (
    "ta ~ big_n + cfo + size + lev + mtb + "
    "factor(fyear) * (inv_at + I(d_sale - d_ar) + ppe)"
)
CMSW_CONTROLS = (
    "selfdealflag + blckownpct + initabret + lnvioperiod + bribeflag + "
    "mobflag + deter + lnempcleveln + lnuscodecnt + viofraudflag + "
    "misledflag + audit8flag + exectermflag + coopflag + impedeflag + "
    "pctinddir + recidivist + lnmktcap + mkt2bk + lev + lndistance + "
    "factor(ff12)"
)


def rel(got, ref) -> float:
    got, ref = np.asarray(got, dtype=float), np.asarray(ref, dtype=float)
    return float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)))


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(Path(DATA_DIR) / f"{name}.csv", skip_blank_lines=False)


@pytest.fixture(scope="module")
def R() -> dict:
    path = Path(__file__).parent / "data" / "gow_ding_accounting_R.json"
    return json.loads(path.read_text(encoding="utf-8"))


def pick(result, prefixes):
    return [[n for n in result.params.index if n.startswith(p)][0] for p in prefixes]


# ---------------------------------------------------------------------------
# Chapter 5, statistical inference: one regression, seven standard errors
# ---------------------------------------------------------------------------


def test_ch05_standard_errors_on_the_gow_ormazabal_taylor_panel(R):
    got = load("got")
    order = ["Intercept", "x"]
    assert rel(sp.regress("y ~ x", got).params[order], R["got_ols"]["coef"]) < EXACT
    assert rel(sp.regress("y ~ x", got).std_errors[order], R["got_ols"]["se"]) < EXACT
    white = sp.regress("y ~ x", got, robust="hc1")
    assert rel(white.std_errors[order], R["got_hc1"]["se"]) < EXACT
    # NW: plm::vcovNW, Newey-West within firms
    for lags, key in [(1, "got_nw"), (3, "got_nw_l3")]:
        nw = sp.regress(
            "y ~ x", got, robust="hac", hac_lags=lags, hac_panel=("firm", "year")
        )
        assert rel(nw.std_errors[order], R[key]["se"]) < EXACT
    # FM-t and FM-NW
    fm = sp.fama_macbeth("y ~ x", got, time="year")
    assert rel(fm.params[order], R["got_fm"]["coef"]) < EXACT
    assert rel(fm.std_errors[order], R["got_fm"]["se"]) < EXACT
    assert rel(fm.params[order], R["got_fm_t"]["coef"]) < EXACT
    years = fm.model_info["period_coefs"]
    assert rel(years["x"], R["got_fm_years"]["b1"]) < EXACT
    for lags, key in [(1, "got_fm_nw1"), (3, "got_fm_nw3")]:
        fm_nw = sp.fama_macbeth("y ~ x", got, time="year", lags=lags)
        assert rel(fm_nw.std_errors[order], R[key]["se"]) < EXACT
    # CL-i, CL-t, CL-2
    for key, cluster in [
        ("got_cl_i", "firm"),
        ("got_cl_t", "year"),
        ("got_cl_2", ["year", "firm"]),
    ]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cl = sp.regress("y ~ x", got, cluster=cluster)
        assert rel(cl.std_errors[order], R[key]["se"]) < EXACT


# ---------------------------------------------------------------------------
# Chapter 3, regression fundamentals
# ---------------------------------------------------------------------------


def test_ch03_difference_in_differences_on_the_test_scores(R):
    scores = load("camp_scores")
    sub = scores[scores["grade"].isin([6, 7])]
    for key, data, formula in [
        ("dd_67", sub, "score ~ treat * post"),
        ("dd_all", scores, "score ~ treat * post"),
        ("dd_trend", scores, "score ~ treat * post + grade"),
        ("dd_grade", scores, "score ~ factor(grade)"),
        ("dd_fe_grade", scores, "score ~ treat * post + factor(grade)"),
    ]:
        res = sp.regress(formula, data)
        # the coefficients come in a different order; compare as sets
        want = [v for v in R[key]["coef"] if v is not None]
        want_se = [v for v in R[key]["se"] if v is not None]
        if len(want) < len(R[key]["coef"]):
            # `post` is a sum of grade dummies: R reports NA for one
            # coefficient, StatsPAI omits one. The identified effect is
            # the interaction.
            name = [n for n in res.params.index if "treat" in n and "post" in n][0]
            i = R[key]["names"].index("treatTRUE:postTRUE")
            assert rel(res.params[name], R[key]["coef"][i]) < EXACT, key
            assert rel(res.std_errors[name], R[key]["se"][i]) < EXACT, key
            continue
        assert rel(np.sort(res.params), np.sort(want)) < EXACT, key
        assert rel(np.sort(res.std_errors), np.sort(want_se)) < EXACT, key
    # student and grade dummies written out: 1,004 identified coefficients
    full = sp.regress("score ~ treat * post + factor(grade) + factor(id)", scores)
    name = [n for n in full.params.index if "treat" in n and "post" in n][0]
    assert len(full.params) == R["dd_id"]["rank"]
    assert rel(full.params[name], R["dd_id"]["coef"]) < EXACT
    assert rel(full.std_errors[name], R["dd_id"]["se"]) < EXACT
    # the same model with the effects absorbed
    for key, kwargs in [
        ("dd_feols", {}),
        ("dd_feols_iid", {"vcov": "iid"}),
        ("dd_feols_cl2", {"vcov": {"CRV1": "grade+id"}}),
    ]:
        fe = sp.feols("score ~ I(post * treat) | grade + id", scores, **kwargs)
        assert rel(fe.params, R[key]["coef"]) < EXACT
        assert rel(fe.std_errors, R[key]["se"]) < EXACT


def test_ch03_accruals_regression_with_year_interactions(R):
    comp = load("comp_full")
    res = sp.regress(COMP, comp)
    ref = R["comp_raw_lm"]
    assert len(res.params) == ref["k"] and res.data_info["nobs"] == ref["n"]
    names = pick(res, KEEP)
    assert rel(res.params[names], ref["coef"]) < 1e-7
    assert rel(res.std_errors[names], ref["se"]) < 1e-7


# ---------------------------------------------------------------------------
# Chapter 24, extreme values
# ---------------------------------------------------------------------------


def test_ch24_winsorize_and_truncate(R):
    x = load("wins_x")
    ref = R["wins"]
    for key, kwargs in [
        ("w", {}),
        ("w05", {"cuts": (5, 95)}),
        ("w_asym", {"cuts": (0, 99)}),
        ("t", {"trim": True}),
    ]:
        out = sp.winsor(x, ["x"], **kwargs)
        got = out[[c for c in out.columns if c != "x"][0]]
        want = pd.Series(ref[key], dtype=float)
        assert (got.isna() == want.isna()).all()
        assert rel(got.dropna(), want.dropna()) < EXACT
    # the type-2 quantile, not numpy's default
    assert rel(np.nanpercentile(x["x"], [1, 99], method="averaged_inverted_cdf"), ref["q2"]) < EXACT
    assert rel(np.nanpercentile(x["x"], [1, 99]), ref["q2"]) > 1e-3


@pytest.mark.parametrize(
    "key, name",
    [("comp_raw", "comp_full"), ("comp_win", "comp_win"), ("comp_trunc", "comp_trunc")],
)
def test_ch24_two_way_clustered_regressions(R, key, name):
    """Raw, winsorized and truncated data, clustered by firm and year.

    With 21 years and 85 coefficients the two-way covariance has 46 or 47
    negative eigenvalues. fixest sets them to zero and so do we; the result
    depends on which year is the base category, so the exact comparison is
    with fixest run on the same base year (``*_base96``). The book's own
    run has an empty base level, 1995, and differs by a few percent."""
    data = load(name)
    with pytest.warns(RuntimeWarning, match="not positive semi-definite"):
        res = sp.regress(COMP, data, cluster=["gvkey", "fyear"])
    names = pick(res, KEEP)
    ref = R[key + "_base96"]
    assert res.data_info["nobs"] == ref["n"]
    assert rel(res.params[names], ref["coef"]) < 1e-7
    assert rel(res.std_errors[names], ref["se"]) < GLM
    assert res.diagnostics["Two-way VCOV negative eigenvalues"] == ref["n_negative"]
    # the matrix behind the standard errors is the one vcov() returns
    assert np.allclose(np.sqrt(np.diag(res.vcov())), res.std_errors)
    # sp.feols, which runs pyfixest, applies the same adjustment
    with pytest.warns(RuntimeWarning, match="not positive semi-definite"):
        fe = sp.feols(COMP, data, vcov={"CRV1": "gvkey + fyear"})
    assert rel(fe.std_errors[pick(fe, KEEP)], ref["se"]) < GLM
    # the book's parametrisation: same coefficients, nearby standard errors
    assert rel(res.params[names], R[key]["coef"]) < 1e-7
    assert rel(res.std_errors[names], R[key]["se"]) < 0.06
    # without the adjustment (what 1.38 reported) the gap was 4 to 19 percent
    assert rel(ref["se_unfixed"], ref["se"]) > 0.04


def test_ch24_cooks_distance(R):
    comp = load("comp_full")
    fit = sp.regress(COMP, comp)
    out = sp.estat(fit, "leverage", print_results=False)
    d = np.asarray(out["cooks_d"], dtype=float)
    assert out["n_influential"] == R["cooks"]["n_extreme"]
    ref_head = [v for v in R["cooks"]["head"] if v is not None]
    assert rel(d[: len(ref_head)], ref_head) < 1e-7
    assert rel(d.max(), R["cooks"]["max"]) < 1e-5  # leverage of 0.99999
    # dropping the flagged observations, as the chapter does
    used = comp.loc[fit.data_info["sample_index"]] if "sample_index" in fit.data_info else None
    if used is not None:
        kept = used[d <= 4 / len(d)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            again = sp.regress(COMP, kept, cluster=["gvkey", "fyear"])
        names = pick(again, KEEP)
        assert rel(again.params[names], R["comp_cooks"]["coef"]) < 1e-7


@pytest.mark.parametrize(
    "key, name, formula",
    [
        ("lmrob_sim1", "robust_sim", "cy ~ x1"),
        ("lmrob_sim3", "robust_sim", "cy ~ x1 + x2 + x3"),
        ("lmrob_comp", "comp_simple", "ta ~ big_n + cfo + size + lev + mtb"),
    ],
)
def test_ch24_robust_regression(R, key, name, formula):
    """lmrob(method = 'MM', tuning.psi = 3.4437). The last case is real
    Compustat data: 915 of 9,036 firm-years get weight zero."""
    ref = R[key]
    res = sp.robreg(formula, load(name), tuning=3.4437, tuning_s=1.54764, small=False)
    names = ["Intercept"] + pick(res, [n.replace("TRUE", "") for n in ref["names"][1:]])
    assert rel(res.params[names], ref["coef"]) < GLM
    assert rel(res.model_info["scale"], ref["scale"]) < GLM
    assert rel(res.model_info["weights"].sum(), ref["w_sum"]) < GLM
    assert res.model_info["n_zero_weight"] == ref["w_zero"]
    # not a parity claim: see the review document, open items
    assert rel(res.std_errors[names], ref["se"]) < 2e-3


@pytest.mark.parametrize("y", ["firmpenalty", "emppenalty", "empprisonmos"])
def test_ch24_penalties_poisson_and_log_ols(R, y):
    """Call, Martin, Sharp and Wilde's whistleblower regressions. With
    firmpenalty the `mobflag` cases are all zero: the model is
    quasi-separated and sp.poisson used to fail."""
    cmsw = load("cmsw")
    formula = f"{y} ~ wbflag + {CMSW_CONTROLS}"
    ref = R[f"pois_{y}"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pois = sp.poisson(formula, cmsw, robust="hc1")
        plain = sp.poisson(formula, cmsw)
    assert rel(pois.params["wbflag"], ref["coef"]) < GLM
    assert rel(pois.std_errors["wbflag"], ref["se"]) < 1e-5
    assert rel(plain.std_errors["wbflag"], ref["se_model"]) < 1e-5
    assert (pois.model_info.get("separated_terms") == ["mobflag"]) == (y == "firmpenalty")
    ols = sp.regress(f"ln_{formula}", cmsw, robust="hc1")
    ref = R[f"ols_{y}"]
    assert rel(ols.params["wbflag"], ref["coef"]) < EXACT
    assert rel(ols.std_errors["wbflag"], ref["se"]) < EXACT
    for alpha, want in zip((0.01, 0.05, 0.1), ref["itcv"]):
        out = sp.itcv(ols, "wbflag", alpha=alpha)
        if out["significant"]:
            assert rel(out["itcv"], want) < EXACT
        else:
            # the chapter's function divides by 1 - r# throughout; for an
            # estimate short of significance Frank's threshold (and
            # konfound) divide by 1 + r#
            r_crit = abs(out["r_crit"])
            assert rel(out["itcv"] * (1 + r_crit) / (1 - r_crit), want) < EXACT


def test_ch24_impacts_of_the_observed_controls(R):
    cmsw = load("cmsw")
    fit = sp.regress(f"ln_firmpenalty ~ wbflag + {CMSW_CONTROLS}", cmsw)
    imp = sp.itcv(fit, "wbflag")["impacts"]
    book = pd.DataFrame(R["impacts"]).set_index("var")
    common = [c for c in imp.index if c in book.index]
    assert len(common) == 21
    assert rel(imp.loc[common, "impact"], book.loc[common, "impact"]) < 1e-7
    assert imp.index[0] == "lnmktcap"


# ---------------------------------------------------------------------------
# Chapters 20, 23: instrumental variables, generalized linear models
# ---------------------------------------------------------------------------


def test_ch20_iv_with_overidentification_and_first_stage_tests(R):
    data = load("iv_sim")
    ref = R["iv"]
    res = sp.ivreg("y ~ (X ~ z_1 + z_2 + z_3)", data)
    assert rel(res.params["X"], ref["coef"][1]) < EXACT
    assert rel(res.std_errors["X"], ref["se"][1]) < EXACT
    fe = sp.feols("y ~ 1 | X ~ z_1 + z_2 + z_3", data)
    assert rel(fe.params["X"], ref["coef"][1]) < EXACT
    over = sp.estat(res, "overid", print_results=False)
    assert rel(over["statistic"], ref["sargan"]) < EXACT
    assert rel(over["pvalue"], ref["sargan_p"]) < EXACT
    first = sp.estat(res, "firststage", print_results=False)
    assert rel(first["statistic"], ref["f1"]) < EXACT
    endo = sp.estat(res, "endogenous", print_results=False)
    assert rel(endo["statistic"], ref["wh"]) < EXACT
    assert rel(endo["pvalue"], ref["wh_p"]) < EXACT


def test_ch23_marginal_effects_of_probit_logit_and_poisson(R):
    data = load("glm_sim")
    logit = sp.logit("yb ~ x", data)
    assert rel(logit.params, R["glm_logit"]["coef"]) < GLM
    assert rel(logit.std_errors, R["glm_logit"]["se"]) < GLM
    assert rel(sp.margins(logit, data)["dy/dx"].iloc[0], R["glm_logit"]["mfx"]) < GLM
    probit = sp.probit("yb ~ x", data)
    assert rel(probit.params, R["glm_probit"]["coef"]) < GLM
    assert rel(sp.margins(probit, data)["dy/dx"].iloc[0], R["glm_probit"]["mfx"]) < GLM
    # probit standard errors: observed information here and in Stata,
    # expected information in R's glm; 0.4% apart at n = 1,000
    assert 1e-4 < rel(probit.std_errors, R["glm_probit"]["se"]) < 1e-2
    pois = sp.poisson("yc ~ x", data)
    assert rel(pois.params, R["glm_poisson"]["coef"]) < GLM
    assert rel(pois.std_errors, R["glm_poisson"]["se"]) < GLM
    assert rel(sp.margins(pois, data)["dy/dx"].iloc[0], R["glm_poisson"]["mfx"]) < GLM


# ---------------------------------------------------------------------------
# Pieces of chapters 16, 15, 19 and 26
# ---------------------------------------------------------------------------


def test_ch16_binomial_test_of_rejection_rates(R):
    assert rel(sp.bitest(n=1000, successes=10, p=0.05).pvalue, R["binom"]["p10"]) < EXACT
    assert rel(sp.bitest(n=1000, successes=90, p=0.05).pvalue, R["binom"]["p90"]) < EXACT


def test_ch15_equality_of_extreme_decile_returns(R):
    data = load("decile_sim")
    fit = sp.regress("r ~ factor(dec) - 1", data)
    assert rel(fit.params, R["linhyp"]["coef"]) < EXACT
    out = sp.test(fit, "C(dec)[1] = C(dec)[10]")
    assert rel(out["statistic"], R["linhyp"]["F"]) < EXACT
    assert rel(out["pvalue"], R["linhyp"]["p"]) < EXACT


def test_ch19_four_estimators_of_one_treatment_effect(R):
    wide, long = load("ancova_wide"), load("ancova_long")
    did = sp.regress("y ~ treat * post", long)
    assert rel(np.sort(did.params), np.sort(R["nr_did"]["coef"])) < EXACT
    post = sp.regress("y_post ~ treat", wide)
    assert rel(post.std_errors, R["nr_post"]["se"]) < EXACT
    change = sp.regress("I(y_post - y_pre) ~ treat", wide)
    assert rel(change.params, R["nr_change"]["coef"]) < EXACT
    assert rel(change.std_errors, R["nr_change"]["se"]) < EXACT
    ancova = sp.regress("y_post ~ y_pre + treat", wide)
    assert rel(ancova.params, R["nr_ancova"]["coef"]) < EXACT
    assert rel(ancova.std_errors, R["nr_ancova"]["se"]) < EXACT


def test_ch26_auc_and_ndcg(R):
    data = load("auc_sim")
    ref = R["auc"]
    assert rel(sp.auc(data["response"], data["score"]), ref["auc"]) < EXACT
    for k, key in [(0.01, "ndcg01"), (0.02, "ndcg02"), (0.03, "ndcg03")]:
        assert rel(sp.ndcg(data["response"], data["score"], k), ref[key]) < EXACT
