"""``design=SurveyDesign`` on the causal and regression estimators vs R survey.

Reference: ``_fixtures/survey_causal_R.json`` written by
``_generate_survey_causal_R.R`` from ``_fixtures/survey_causal_data.csv``
(5 strata with 3-6 PSUs, 10-20 rows per PSU, informative weights, fpc in
count and fraction form, a lonely-PSU stratum variant) and
``_fixtures/survey_causal_surv.csv`` (a time / status pair per row).

What each block checks
----------------------
* The shared helper ``statspai.survey.linearization.design_vcov`` against
  ``survey::svyrecvar`` on a fixed contribution matrix, for ten designs and
  a raked design (R ``calibrate(calfun = "raking")``).
* ``sp.ipw`` (Hajek ATE / ATT): R builds the stacked estimating equations
  (survey-weighted logit score, two weighted means) by hand, with a
  closed-form Jacobian checked against ``numDeriv::jacobian``, and takes
  ``svyrecvar(-A^{-1} w_i psi_i)`` -- Binder's linearisation.
* ``sp.aipw(cross_fit=False, trim=0)``: the same for the five-block system
  (``se_method='sandwich'``) and with the nuisance fits held fixed
  (``se_method='influence'``).
* ``sp.regress`` / ``sp.logit``: ``svyglm`` (gaussian; quasibinomial refitted
  at the MLE).
* ``sp.tmle``: ``tmle::tmle(obsWeights =)`` influence curves, svyrecvar of
  ``IC / n``. Shared SL.glm initial fits converge to ~1e-9, hence 1e-6.
* ``sp.dml`` (PLR, IRM ATE) on three folds of whole PSUs with weighted lm /
  glm nuisances. IRM's logistic propensity converges to ~1e-9 (1e-6).
* ``sp.cox``: ``svycoxph`` (Efron ties). ``sp.kaplan_meier``: product-limit
  curve and ``svyrecvar(w_i r_i(t))`` with ``r_i`` from
  ``resid(survfit, type = "surv")`` (R's own infinitesimal jackknife).

Design df (#PSU - #strata) is R ``degf``. Tolerance 1e-10 relative unless
stated (observed ~1e-14).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import statspai as sp
from statspai.survey.linearization import design_df, design_vcov

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "survey_causal_R.json").read_text(encoding="utf-8"))
RTOL = 1e-10

DESIGNS = {
    "full": dict(strata="stratum", cluster="psu", fpc="fpc_N", nest=True),
    "frac": dict(strata="stratum", cluster="psu", fpc="fpc_f", nest=True),
    "nofpc": dict(strata="stratum", cluster="psu", nest=True),
    "cluster_only": dict(cluster="psu_global"),
    "element_fpc": dict(strata="stratum", fpc="fpc_el"),
    "weights_only": dict(),
}
for _lp in ("remove", "certainty", "adjust", "average"):
    DESIGNS[f"lonely_{_lp}"] = dict(
        strata="stratum_l", cluster="psu", nest=True, lonely_psu=_lp
    )
CAUSAL = ["full", "nofpc", "cluster_only", "weights_only", "lonely_adjust"]


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    d = pd.read_csv(_FIX / "survey_causal_data.csv")
    sv = pd.read_csv(_FIX / "survey_causal_surv.csv")
    d["time"], d["status"] = sv["time"], sv["status"]
    d["psu_global"] = d["stratum"] * 100 + d["psu"]
    d["fold"] = pd.factorize(d["psu_global"], sort=True)[0] % 3
    return d


def _design(df: pd.DataFrame, name: str) -> sp.SurveyDesign:
    return sp.svydesign(df, weights="w", **DESIGNS[name])


def _close(a, b, rtol=RTOL) -> None:
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=rtol)


@pytest.mark.parametrize("name", list(DESIGNS))
def test_helper_matches_svyrecvar(df, name):
    u = np.column_stack([df.w * df.y / df.w.sum(), df.w * (df.x1 - 0.3) * df.d]).astype(
        float
    )
    des = _design(df, name)
    _close(design_vcov(u, des), R["helper"][name]["vcov"])
    assert design_df(des) == R["helper"][name]["degf"]


def test_helper_raked_design_matches_svyrecvar(df):
    ref = R["helper"]["raked"]
    cal = _design(df, "nofpc").calibrate(
        margins={"x2": {0: ref["margins"]["x2_0"], 1: ref["margins"]["x2_1"]}},
        tol=1e-14,
    )
    _close(cal.weights, ref["weights"], rtol=1e-12)
    z = np.column_stack([df.y, (df.x1 - 0.3) * df.d]).astype(float)
    _close(design_vcov(cal.weights[:, None] * z / cal.weights.sum(), cal), ref["vcov"])


def test_helper_places_rows_as_a_domain(df):
    """Rows not listed contribute zero and keep their PSU: same as zeros."""
    des = _design(df, "full")
    u = (df.w * df.y / df.w.sum()).to_numpy()
    keep = np.flatnonzero(df.d.to_numpy() == 1)
    full = np.where(df.d.to_numpy() == 1, u, 0.0)
    _close(design_vcov(u[keep], des, keep), design_vcov(full, des), rtol=1e-14)


@pytest.mark.parametrize("estimand", ["ATE", "ATT"])
@pytest.mark.parametrize("name", CAUSAL)
def test_ipw_design_matches_R_binder(df, name, estimand):
    ref = R["causal"][name].get(f"ipw_{estimand.lower()}")
    if ref is None:
        pytest.skip("ATT not frozen for this design")
    r = sp.ipw(
        df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        estimand=estimand,
        design=_design(df, name),
    )
    _close(r.estimate, ref["estimate"])
    _close(r.se, ref["se"])
    assert r.model_info["survey_design"]["df"] == ref["degf"]
    assert r.model_info["se_method"] == "linearization"
    assert ref["jacobian_check"] < 1e-7  # R: closed form == numDeriv


@pytest.mark.parametrize("se_method", ["sandwich", "influence"])
@pytest.mark.parametrize("name", CAUSAL)
def test_aipw_design_matches_R(df, name, se_method):
    ref = R["causal"][name]["aipw"]
    r = sp.aipw(
        df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        cross_fit=False,
        trim=0,
        se_method=se_method,
        design=_design(df, name),
    )
    _close(r.estimate, ref["estimate"])
    _close(r.se, ref[f"se_{se_method}"])
    po = r.model_info["potential_outcome_means_se"]
    _close([po[1], po[0]], ref[f"se_mu_{se_method}"])


@pytest.mark.parametrize("name", CAUSAL)
def test_regress_and_logit_design_match_svyglm(df, name):
    ref = R["causal"][name]["glm"]
    des = _design(df, name)
    g = sp.regress("y ~ d + x1 + x2", df, design=des)
    _close(g.params.values, ref["gaussian"]["coef"])
    _close(g.std_errors.values, ref["gaussian"]["se"])
    _close(g.data_info["var_cov"], ref["gaussian"]["vcov"])
    b = sp.logit("yb ~ d + x1", df, design=des, tol=1e-14)
    _close(b.params.values, ref["logit"]["coef"])
    _close(b.std_errors.values, ref["logit"]["se"])
    # Stata svy: t with the design degrees of freedom
    assert g.data_info["df_inference"] == ref["degf"]
    assert b.data_info["df_inference"] == ref["degf"]


def _lr() -> LogisticRegression:
    return LogisticRegression(penalty=None, tol=1e-12, max_iter=100000)


@pytest.mark.parametrize("family", ["gaussian", "binomial"])
@pytest.mark.parametrize("name", ["full", "nofpc", "cluster_only", "weights_only"])
def test_tmle_design_matches_R_tmle_ic(df, name, family):
    ref = R["causal"][name]["tmle"][family]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.tmle(
            df,
            y="y" if family == "gaussian" else "yb",
            treat="d",
            covariates=["x1", "x2"],
            outcome_library=[LinearRegression() if family == "gaussian" else _lr()],
            propensity_library=[_lr()],
            fluctuation="per_arm",
            q_bound=5e-4,
            design=_design(df, name),
        )
    det = r.detail.set_index("parameter")
    # glm convergence of the shared initial fits leaves ~1e-9
    _close(r.estimate, ref["ate"], rtol=1e-6)
    _close([r.se, det.loc["EY1", "se"], det.loc["EY0", "se"]], ref["se"], rtol=1e-6)


@pytest.mark.parametrize("model", ["plr", "irm"])
@pytest.mark.parametrize("name", ["full", "nofpc", "cluster_only", "weights_only"])
def test_dml_design_matches_R_with_psu_folds(df, name, model):
    ref = R["causal"][name]["dml"][model]
    r = sp.dml(
        df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        model=model,
        ml_g=LinearRegression(),
        ml_m=_lr() if model == "irm" else LinearRegression(),
        n_folds=3,
        fold_indices="fold",
        design=_design(df, name),
    )
    tol = RTOL if model == "plr" else 1e-6  # logistic propensity ~1e-9
    _close(r.estimate, ref["estimate"], rtol=tol)
    _close(r.se, ref["se"], rtol=tol)


@pytest.mark.parametrize("name", ["full", "nofpc", "cluster_only", "weights_only"])
def test_cox_design_matches_svycoxph(df, name):
    ref = R["causal"][name]["surv"]["cox"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = sp.cox("time ~ d + x1", data=df, event="status", design=_design(df, name))
    _close(c.params.values, ref["coef"])
    _close(c.std_errors.values, ref["se"])


@pytest.mark.parametrize("name", ["full", "nofpc", "cluster_only", "weights_only"])
def test_kaplan_meier_design_matches_R_influence(df, name):
    ref = R["causal"][name]["surv"]
    des = _design(df, name)
    for key, group in (("km_all", None), ("km_d1", "1"), ("km_d0", "0")):
        km = sp.kaplan_meier(
            df,
            duration="time",
            event="status",
            group=None if group is None else "d",
            conf_type="log",
            design=des,
        )
        tab = km._tables["all" if group is None else group]
        tab = tab[tab["time"] > 0]
        r = ref[key]
        _close(tab["time"].values, r["time"], rtol=0)
        np.testing.assert_allclose(tab["survival"].values, r["surv"], atol=1e-14)
        se = np.asarray(r["se"], float)
        # the last point has S = 0: both standard errors are rounding zeros
        np.testing.assert_allclose(
            tab["std_err"].values, se, rtol=RTOL, atol=1e-12 * se.max()
        )
