"""Replicate-weight designs (``sp.svrepdesign`` / ``sp.as_svrepdesign``)
vs R ``survey`` 4.5 and Stata 18.

References
----------
* ``_fixtures/survey_replicate_R.json`` from
  ``_generate_survey_replicate_R.R``: a stratified two-PSU-per-stratum
  sample (``survey_replicate_data.csv``) with JKn / BRR / Fay(0.3) / JK1 /
  Rao-Wu bootstrap replicate weights made by R ``as.svrepdesign`` and
  exported, then declared with ``svrepdesign(type=)`` as a user of a
  public-use file would. ``svymean`` / ``svytotal`` / ``svyglm`` (gaussian,
  quasibinomial logit and probit) / ``svycoxph`` / ``withReplicates`` of a
  Hajek IPW ATE.
* ``_fixtures/survey_replicate_did_R.json`` from
  ``_generate_survey_replicate_did_R.R``: JKn weights of the unit-level
  design of ``survey_did_data.csv``; ``did::att_gt`` recomputed with every
  replicate (``withReplicates``) and ``svyglm(y ~ treat * post)``.
* ``_fixtures/survey_replicate_stata.json`` from
  ``_generate_survey_replicate_stata.do``: Stata 18 ``svy brr`` /
  ``svy jackknife`` / ``svy bootstrap`` regress, logit, mean on the same
  replicate weights.

R survey, did and Stata are used as black boxes (documentation and
outputs only). Tolerances: linear statistics agree to rounding (1e-10);
the R glm references are fitted with epsilon = 1e-13, so the logit agrees
to 1e-10 / 1e-9; probit and Cox refits stop at StatsPAI's / coxph's own
convergence criteria (1e-7 on estimates, 1e-6 on SEs); IPW and DiD
inherit the default glm criterion inside R's withReplicates functions.
Stata ``svy`` keeps replicate estimates in float unless
``saving(, double)`` is given (moving e(V) by ~1e-7); the do-file saves
in double.

Documented conventions (tests below pin each):

* df: R's (numerical rank of the replicate weights minus one). Stata
  reports #replicates - 1 for BRR / Fay (11 here, R and StatsPAI 10) and
  no df for the bootstrap; the SEs agree.
* Stata ``svy bootstrap`` divides by R, R ``svrepdesign(type='bootstrap')``
  by R - 1: ``scale=1/R`` reproduces Stata.
* Cox, Efron ties: a replicate here drops the rows of zero weight (as
  Stata ``[pw=]`` and ``sp.cox(weights=)`` do). R's ``svycoxph`` keeps them
  with a negligible weight, and Efron's tie fractions count deaths, not
  weights, so its replicate fits differ slightly (relative ~3e-4 on the
  SE here). Breslow ties agree to 1e-9, and refitting with a 1e-10 weight
  on the deleted rows reproduces R's Efron SEs -- the mechanism, shown.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "survey_replicate_R.json").read_text(encoding="utf-8"))
S = json.loads((_FIX / "survey_replicate_stata.json").read_text(encoding="utf-8"))
RD = json.loads((_FIX / "survey_replicate_did_R.json").read_text(encoding="utf-8"))

#: fixture name -> (column prefix, type, extra svrepdesign kwargs, mse)
SPECS = {
    "jkn": ("jkn", "JKn", "jkn", False),
    "jkn_mse": ("jkn", "JKn", "jkn", True),
    "brr": ("brr", "BRR", None, False),
    "fay": ("fay", "Fay", None, False),
    "jk1": ("jk1", "JK1", None, False),
    "boot": ("boot", "bootstrap", None, False),
    "boot_mse": ("boot", "bootstrap", None, True),
}


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "survey_replicate_data.csv")


def _design(data: pd.DataFrame, name: str, **extra):
    pre, type_, rs, mse = SPECS[name]
    kw = dict(extra)
    if rs is not None:
        kw.update(rscales=R["meta"][rs]["rscales"], scale=1.0)
    if type_ == "Fay":
        kw["rho"] = 0.3
    return sp.svrepdesign(
        data, repweights=f"^{pre}_", weights="w", type=type_, mse=mse, **kw
    )


def _close(a, b, rtol):
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=rtol)


@pytest.mark.parametrize("name", list(SPECS))
def test_design_constants_match_svrepdesign(data, name):
    D = _design(data, name)
    ref = R[name]
    _close(D.scale, ref["scale"], 1e-14)
    _close(D.rscales, ref["rscales"], 1e-14)
    assert D.mse == ref["mse"]
    assert D.df == ref["degf"]


@pytest.mark.parametrize("name", list(SPECS))
def test_mean_total_glm_match_R(data, name):
    D = _design(data, name)
    ref = R[name]
    m = sp.svymean("y", D)
    _close(m.estimate, ref["mean"], 1e-12)
    _close(m.std_error, ref["mean_se"], 1e-10)
    t = sp.svytotal("y", D)
    _close(t.std_error, ref["total_se"], 1e-10)
    g = sp.svyglm("y ~ x1 + x2", D)
    _close(g.estimate, ref["glm_coef"], 1e-10)
    _close(g.std_error, ref["glm_se"], 1e-10)
    assert g.dof == ref["degf"]
    b = sp.svyglm("yb ~ x1 + x2", D, family="binomial")
    _close(b.estimate, ref["logit_coef"], 1e-10)
    _close(b.std_error, ref["logit_se"], 1e-9)


@pytest.mark.parametrize("name", list(SPECS))
def test_regression_entry_points_match_R(data, name):
    D = _design(data, name)
    ref = R[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sp.regress("y ~ x1 + x2", data=data, design=D)
        lo = sp.logit("yb ~ x1 + x2", data=data, design=D)
        pr = sp.probit("yb ~ x1 + x2", data=data, design=D)
    _close(g.std_errors, ref["glm_se"], 1e-10)
    _close(lo.params, ref["logit_coef"], 1e-9)
    _close(lo.std_errors, ref["logit_se"], 1e-8)
    # sp.probit stops at its default tol (coefficients ~1e-8 apart)
    _close(pr.params, ref["probit_coef"], 1e-7)
    _close(pr.std_errors, ref["probit_se"], 1e-6)
    info = g.model_info["survey_design"]
    assert info["replicate_type"] == SPECS[name][1]
    assert g.data_info["df_inference"] == ref["degf"]


@pytest.mark.parametrize("name", list(SPECS))
def test_ipw_matches_withReplicates(data, name):
    D = _design(data, name)
    ref = R[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.ipw(data, y="y", treat="d", covariates=["x1", "x2"], design=D)
    _close(r.estimate, ref["ipw"], 1e-7)
    _close(r.se, ref["ipw_se"], 1e-6)
    from scipy import stats

    crit = stats.t.ppf(0.975, ref["degf"])
    _close(r.ci[1] - r.estimate, crit * r.se, 1e-12)


@pytest.mark.parametrize("name", list(SPECS))
def test_cox_breslow_matches_svycoxph(data, name):
    D = _design(data, name)
    ref = R[name]
    r = sp.cox(
        data=data,
        duration="time",
        event="status",
        x=["x1", "x2"],
        ties="breslow",
        design=D,
    )
    # coxph stops at its own convergence criterion on every replicate
    _close(r.params, ref["cox_breslow_coef"], 1e-8)
    _close(r.std_errors, ref["cox_breslow_se"], 1e-6)


def test_cox_efron_difference_is_rs_zero_weight_rows(data):
    """R's Efron replicate SEs come back when the deleted rows are kept with
    a negligible weight; ours drop them. The gap is small and bounded."""
    D = _design(data, "jk1")
    ref = R["jk1"]
    ours = sp.cox(data=data, duration="time", event="status", x=["x1", "x2"], design=D)
    # 5e-9, not 1e-10: svycoxph stops (coxph eps = 1e-9 on the log
    # likelihood) with a Newton step of 1.08e-9 still to take; ours takes it
    # and its remaining step is 2e-16 (until 2026-10-28 sp.cox stopped at the
    # same point as R through a line-search bug, see
    # tests/test_cox_newton_convergence.py)
    _close(ours.params, ref["cox_coef"], 5e-9)
    gap = np.abs(ours.std_errors.to_numpy() / np.asarray(ref["cox_se"]) - 1)
    assert gap.max() < 2e-3
    reps = []
    for r in range(D.n_replicates):
        w = D.repweights[:, r].copy()
        w[w == 0] = 1e-10
        reps.append(
            sp.cox(
                data=data.assign(ww=w),
                duration="time",
                event="status",
                x=["x1", "x2"],
                weights="ww",
                weights_as="sampling",
            ).params.to_numpy()
        )
    reps = np.asarray(reps)
    V = D.variance(ours.params.to_numpy(), reps)
    _close(np.sqrt(np.diag(V)), ref["cox_se"], 1e-7)


STATA = {"brr": "brr", "fay": "fay", "jk1": "jk1", "jk1mse": "jk1"}


@pytest.mark.parametrize("sname", list(STATA))
def test_regress_logit_mean_match_stata(data, sname):
    D = _design(data, STATA[sname])
    if sname == "jk1mse":
        D.mse = True
    ref = S[sname]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sp.regress("y ~ x1 + x2", data=data, design=D)
        lo = sp.logit("yb ~ x1 + x2", data=data, design=D)
    _close(g.std_errors, ref["glm_se"], 1e-9)
    # Stata's ml converges to its own tolerance
    _close(lo.std_errors, ref["logit_se"], 1e-6)
    _close(sp.svymean("y", D).std_error, ref["mean_se"], 1e-9)


def test_stata_bootstrap_is_scale_one_over_R(data):
    R_ = R["meta"]["boot"]["R"]
    D = sp.svrepdesign(
        data, repweights="^boot_", weights="w", type="bootstrap", scale=1.0 / R_
    )
    ref = S["boot"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sp.regress("y ~ x1 + x2", data=data, design=D)
    _close(g.std_errors, ref["glm_se"], 1e-9)
    _close(sp.svymean("y", D).std_error, ref["mean_se"], 1e-9)


# ---- as_svrepdesign ----------------------------------------------------- #


def test_as_svrepdesign_jackknife_weights_equal_R(data):
    lin = sp.svydesign(data, weights="w", strata="stratum", cluster="psu")
    J = sp.as_svrepdesign(lin)  # auto: JKn with strata
    assert J.type == "JKn"
    cols = [f"jkn_{i}" for i in range(1, R["meta"]["jkn"]["R"] + 1)]
    _close(J.repweights, data[cols].to_numpy(), 1e-14)
    _close(J.rscales, R["meta"]["jkn"]["rscales"], 1e-14)
    J1 = sp.as_svrepdesign(sp.svydesign(data, weights="w", cluster="psu"))
    assert J1.type == "JK1"
    cols = [f"jk1_{i}" for i in range(1, R["meta"]["jk1"]["R"] + 1)]
    np.testing.assert_allclose(J1.repweights, data[cols].to_numpy(), rtol=1e-13)
    _close(J1.scale, R["meta"]["jk1"]["scale"], 1e-14)


@pytest.mark.parametrize("type_", ["BRR", "Fay", "JKn", "JK1"])
def test_replicate_variance_of_a_total_is_the_linearisation(data, type_):
    """For a total, fully balanced BRR / Fay and the jackknife with
    mse=True reproduce the linearisation variance exactly (closed form:
    sum_h n_h / (n_h - 1) sum_j (t_hj - tbar_h)^2), with an fpc for the
    jackknife -- an identity, independent of R."""
    d = data.assign(fpc=60)
    if type_ == "JK1":
        lin = sp.svydesign(d, weights="w", cluster="psu", fpc="fpc")
    elif type_ == "JKn":
        lin = sp.svydesign(d, weights="w", strata="stratum", cluster="psu", fpc="fpc")
    else:
        lin = sp.svydesign(d, weights="w", strata="stratum", cluster="psu")
    rep = sp.as_svrepdesign(lin, type=type_, rho=0.3, mse=True)
    a = sp.svytotal(["y", "x1"], rep).std_error.to_numpy()
    b = sp.svytotal(["y", "x1"], lin).std_error.to_numpy()
    _close(a, b, 1e-12)


def test_as_svrepdesign_refusals(data):
    lin = sp.svydesign(data, weights="w", strata="stratum", cluster="psu")
    with pytest.raises(MethodIncompatibility, match="JK1"):
        sp.as_svrepdesign(lin, type="JK1")
    odd = sp.svydesign(data.assign(c=data.index % 3), weights="w", cluster="c")
    with pytest.raises(MethodIncompatibility, match="two PSUs"):
        sp.as_svrepdesign(
            sp.svydesign(
                data.assign(c=data.index % 3, s=0), weights="w", strata="s", cluster="c"
            ),
            type="BRR",
        )
    assert sp.as_svrepdesign(odd).type == "JK1"
    with pytest.raises(MethodIncompatibility, match="not built"):
        sp.as_svrepdesign(lin, type="bootstrap")


# ---- DiD -------------------------------------------------------------- #


@pytest.fixture(scope="module")
def did_long() -> pd.DataFrame:
    df = pd.read_csv(_FIX / "survey_did_data.csv")
    rw = pd.read_csv(_FIX / "survey_replicate_did_rw.csv")
    return df.merge(rw, on="id", how="left")


def test_callaway_santanna_replicates_match_did(did_long):
    D = sp.svrepdesign(
        did_long, "^rw_", weights="w", type="JKn", rscales=RD["rscales"], scale=1.0
    )
    assert D.df == RD["degf"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.callaway_santanna(
            did_long,
            y="y",
            g="g",
            t="t",
            i="id",
            x=["x"],
            estimator="dr",
            base_period="varying",
            design=D,
        )
    ref = RD["cs"]
    # did's propensity glm stops at its own deviance criterion
    _close(r.estimate, ref["simple"], 1e-9)
    _close(r.se, ref["se_simple"], 1e-7)
    _close(r.detail["att"], ref["att"], 1e-8)
    _close(r.detail["se"], ref["se_cells"], 1e-7)
    es = r.model_info["event_study"]
    _close(es["se"], ref["se_dynamic"], 1e-7)
    assert r.model_info["se_method"] == "replicate_weights"
    with pytest.raises(MethodIncompatibility):
        sp.aggte(r, type="dynamic")
    # the dispatcher reaches the same fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = sp.did(
            did_long,
            y="y",
            treat="g",
            time="t",
            id="id",
            covariates=["x"],
            base_period="varying",
            design=D,
        )
    _close(r2.se, r.se, 1e-12)


def test_did_2x2_replicates_match_svyglm(did_long):
    sub = did_long[did_long.t.isin([2, 3]) & did_long.g.isin([0, 3])].copy()
    sub["treat"] = (sub.g == 3).astype(int)
    sub["post"] = (sub.t == 3).astype(int)
    D = sp.svrepdesign(
        sub, "^rw_", weights="w", type="JKn", rscales=RD["rscales"], scale=1.0
    )
    r = sp.did_2x2(sub, y="y", treat="treat", time="post", design=D)
    _close(r.estimate, RD["did2x2"]["att"], 1e-12)
    _close(r.se, RD["did2x2"]["se"], 1e-10)
    assert D.df == RD["degf_2x2"]


# ---- the generic refit ------------------------------------------------ #


def _manual_se(fn, data, D, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = fn(data.assign(ww=D.weights), weights="ww", **kw)
        reps = []
        for r in range(D.n_replicates):
            w = D.repweights[:, r]
            reps.append(fn(data.assign(ww=w)[w > 0], weights="ww", **kw))
    est = lambda res: getattr(res, "ate", None) or res.estimate  # noqa: E731
    V = D.variance([est(full)], [[est(x)] for x in reps])
    return est(full), float(np.sqrt(V[0, 0]))


@pytest.mark.parametrize("name", ["brr", "jk1"])
def test_aipw_tmle_are_the_replicate_formula(data, name):
    """The refit is the replicate formula applied to the estimator's own
    weighted fits (plumbing identity)."""
    D = _design(data, name)
    kw = dict(y="y", treat="d", covariates=["x1", "x2"])
    for fn in (sp.aipw, sp.tmle):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = fn(data, design=D, **kw)
        est, se = _manual_se(fn, data, D, **kw)
        _close(r.estimate, est, 1e-12)
        _close(r.se, se, 1e-10)
        assert r.model_info["survey_design"]["n_replicates"] == D.n_replicates


def test_ltmle_is_the_replicate_formula():
    d = pd.read_csv(_FIX / "teffects_wide.csv").iloc[:400].reset_index(drop=True)
    i = np.arange(len(d))
    psu = i // 20
    lin = sp.svydesign(
        d.assign(psu=psu, w=1.0 + (i % 3) * 0.5), weights="w", cluster="psu"
    )
    D = sp.as_svrepdesign(lin)
    kw = dict(
        y="Y", treatments=["A0", "A1"], covariates_time=[["L0"], ["L1"]], baseline=["v"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.ltmle(D.data, design=D, **kw)
    est, se = _manual_se(lambda df, **k: sp.ltmle(df, **k), D.data, D, **kw)
    _close(r.ate, est, 1e-12)
    _close(r.se, se, 1e-10)
    assert r.detail["survey_design"]["replicate_type"] == "JK1"


def test_unsupported_estimators_refuse_replicate_designs(data):
    D = _design(data, "jk1")
    with pytest.raises(MethodIncompatibility, match="replicate"):
        sp.kaplan_meier(data, "time", "status", design=D)
    with pytest.raises(MethodIncompatibility, match="replicate"):
        sp.match(data, "y", "d", ["x1", "x2"], design=D)
    with pytest.raises(MethodIncompatibility, match="weights"):
        sp.ipw(data, y="y", treat="d", covariates=["x1"], weights="w", design=D)


def test_a_replicate_that_empties_an_arm_fails_loudly(data):
    d = data.copy()
    d["d"] = (d["psu"] == 1).astype(int)  # all treated units in PSU 1
    D = sp.as_svrepdesign(sp.svydesign(d, weights="w", cluster="psu"))
    with pytest.raises(MethodIncompatibility, match="replicates"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp.ipw(d, y="y", treat="d", covariates=["x1"], design=D)


def test_svrepdesign_input_errors(data):
    with pytest.raises(MethodIncompatibility, match="rscales"):
        sp.svrepdesign(data, "^jkn_", weights="w", type="JKn")
    with pytest.raises(MethodIncompatibility, match="rho"):
        sp.svrepdesign(data, "^fay_", weights="w", type="Fay")
    with pytest.raises(MethodIncompatibility, match="no column"):
        sp.svrepdesign(data, "^nothing_", weights="w")
    with pytest.raises(MethodIncompatibility, match="unknown type"):
        sp.svrepdesign(data, "^jk1_", weights="w", type="JK3")
