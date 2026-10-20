"""``design=SurveyDesign`` on the DiD and MSM estimators vs R did + survey.

Reference: ``_fixtures/survey_did_R.json`` written by
``_generate_survey_did_R.R``.

* ``sp.callaway_santanna(design=)`` (``estimator='dr'`` and ``'reg'``,
  ``base_period='varying'``): R ``did::att_gt(weightsname =)`` gives the
  unit-level influence functions (``inffunc``; ``aggte``'s
  ``inf.function`` for the simple and dynamic aggregates), and
  ``svyrecvar(IF / n)`` on the unit-level design (rows in att_gt's own unit
  order, read from its processed data) gives the design SEs. Point
  estimates match to ~1e-13 (the existing weighted-CS parity), SEs to the
  same order.
* ``sp.did_2x2(design=)``: ``svyglm(y ~ treat * post)`` on the long design.
* ``sp.msm(design=)``: stabilised weights by hand in R (pooled logits),
  ``svyglm`` with weights = survey weight x stabilised weight (the
  ``ipw::ipwtm`` + ``svyglm`` recipe), gaussian and quasibinomial.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "survey_did_R.json").read_text(encoding="utf-8"))
DESIGNS = {
    "full": dict(strata="stratum", cluster="psu", fpc="fpc_N", nest=True),
    "nofpc": dict(strata="stratum", cluster="psu", nest=True),
}


def _close(a, b, rtol=1e-10) -> None:
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=rtol)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.read_csv(_FIX / "survey_did_data.csv")


@pytest.mark.parametrize("name", list(DESIGNS))
@pytest.mark.parametrize("estimator", ["dr", "reg"])
def test_callaway_santanna_design_matches_R(panel, estimator, name):
    ref = R[f"cs_{estimator}"]
    des = sp.svydesign(panel, weights="w", **DESIGNS[name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.callaway_santanna(
            panel,
            y="y",
            g="g",
            t="t",
            i="id",
            x=["x"],
            estimator=estimator,
            base_period="varying",
            design=des,
        )
    _close(r.detail["att"].values, ref["att"], rtol=1e-9)
    _close(r.detail["se"].values, ref[name]["se_cells"], rtol=1e-9)
    _close(r.estimate, ref["simple"], rtol=1e-9)
    _close(r.se, ref[name]["se_simple"], rtol=1e-9)
    es = r.model_info["event_study"]
    _close(es["att"].values, ref["dynamic_att"], rtol=1e-9)
    _close(es["se"].values, ref[name]["se_dynamic"], rtol=1e-9)
    assert r.model_info["survey_design"]["df"] == ref[name]["degf"]
    # intervals on t(df), not the normal
    from scipy import stats

    crit = stats.t.ppf(0.975, ref[name]["degf"])
    _close(r.ci[1] - r.estimate, crit * r.se, rtol=1e-12)


@pytest.mark.parametrize("name", list(DESIGNS))
def test_did_2x2_design_matches_svyglm(panel, name):
    ref = R["did2x2"][name]
    sub = panel[panel.t.isin([2, 3]) & panel.g.isin([0, 3])].copy()
    sub["treat"] = (sub.g == 3).astype(int)
    sub["post"] = (sub.t == 3).astype(int)
    des = sp.svydesign(sub, weights="w", **DESIGNS[name])
    r = sp.did_2x2(sub, y="y", treat="treat", time="post", design=des)
    _close(r.estimate, ref["att"])
    _close(r.se, ref["se"])
    assert r.model_info["survey_design"]["df"] == ref["degf"]


@pytest.mark.parametrize("name", list(DESIGNS))
@pytest.mark.parametrize("family", ["gaussian", "binomial"])
def test_msm_design_matches_ipw_svyglm_recipe(name, family):
    m = pd.read_csv(_FIX / "survey_msm_data.csv")
    ref = R["msm"][name][family]
    des = sp.svydesign(m, weights="w", **DESIGNS[name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.msm(
            m,
            y="y" if family == "gaussian" else "yb",
            treat="A",
            id="id",
            time="t",
            time_varying=["L"],
            baseline=["V"],
            family=family,
            design=des,
        )
    # statsmodels / glm logits of the weight models agree to ~1e-12
    _close(r.detail["estimate"].values, ref["coef"], rtol=1e-8)
    _close(r.detail["se"].values, ref["se"], rtol=1e-8)


# ---- sp.did dispatcher forwards design= -------------------------------- #


@pytest.mark.parametrize("name", list(DESIGNS))
def test_did_dispatcher_cs_design_matches_R(panel, name):
    ref = R["cs_dr"]
    des = sp.svydesign(panel, weights="w", **DESIGNS[name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.did(
            panel,
            y="y",
            treat="g",
            time="t",
            id="id",
            covariates=["x"],
            base_period="varying",
            design=des,
        )
    _close(r.estimate, ref["simple"], rtol=1e-9)
    _close(r.se, ref[name]["se_simple"], rtol=1e-9)
    _close(r.detail["se"].values, ref[name]["se_cells"], rtol=1e-9)
    assert r.model_info["survey_design"]["df"] == ref[name]["degf"]


@pytest.mark.parametrize("name", list(DESIGNS))
def test_did_dispatcher_2x2_design_matches_svyglm(panel, name):
    ref = R["did2x2"][name]
    sub = panel[panel.t.isin([2, 3]) & panel.g.isin([0, 3])].copy()
    sub["treat"] = (sub.g == 3).astype(int)
    sub["post"] = (sub.t == 3).astype(int)
    des = sp.svydesign(sub, weights="w", **DESIGNS[name])
    r = sp.did(sub, y="y", treat="treat", time="post", design=des)
    _close(r.estimate, ref["att"])
    _close(r.se, ref["se"])
    assert r.model_info["survey_design"]["df"] == ref["degf"]


def test_did_dispatcher_design_refusals(panel):
    from statspai.exceptions import MethodIncompatibility

    des = sp.svydesign(panel, weights="w", **DESIGNS["full"])
    kw = dict(y="y", treat="g", time="t", id="id", design=des)
    with pytest.raises(MethodIncompatibility, match="method='2x2' and method='cs'"):
        sp.did(panel, method="sun_abraham", **kw)
    with pytest.raises(MethodIncompatibility, match="aggregation="):
        sp.did(panel, aggregation="dynamic", **kw)
    with pytest.raises(MethodIncompatibility, match="weights"):
        sp.did(panel, weights="w", **kw)
