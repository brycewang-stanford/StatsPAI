"""``sp.match(design=)`` vs R ``MatchIt`` 4.7.2 + ``survey`` 4.5.

Fixture: ``_fixtures/match_design_R.json`` from
``_fixtures/_generate_match_design_R.R`` on ``matchit_constraints.csv``
with the sampling weights of ``test_match_sweights_parity.py`` and a design
added on both sides (10 strata of 100 rows, PSUs of 20 rows, fpc = 40 PSUs
per stratum). The reference follows MatchIt's documentation for survey
data: the design weights as ``s.weights``, then ``svyglm(y ~ d)`` on the
matched rows as a domain of the full design (``subset()``), the matching
weights treated as fixed. MatchIt and survey are GPL: black boxes
(documentation and outputs only). The matched pairs themselves are already
checked against MatchIt by ``test_match_sweights_parity.py``.
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

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "match_design_R.json").read_text(encoding="utf-8"))
CASES = [k for k in REF if k != "versions"]
DESIGNS = {
    "full": dict(strata="stratum", cluster="psu", fpc="fpc"),
    "nofpc": dict(strata="stratum", cluster="psu"),
    "psu_only": dict(cluster="psu"),
}


@pytest.fixture(scope="module")
def data():
    df = pd.read_csv(_FIX / "matchit_constraints.csv")
    i = np.arange(1, len(df) + 1)
    df["sw"] = 0.4 + (i % 5) * 0.3
    df["stratum"] = (i - 1) // 100 + 1
    df["psu"] = (i - 1) // 20 + 1
    df["fpc"] = 40
    return df


def _fit(data, name, design):
    c = REF[name]
    kw = dict(replace=c["replace"], n_matches=c["ratio"], m_order="largest")
    if "caliper" in c:
        kw.update(caliper=c["caliper"], caliper_scale="sd")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.match(
            data, y="y", treat="d", covariates=["x1", "x2", "g"], design=design, **kw
        )


@pytest.mark.parametrize("dname", list(DESIGNS))
@pytest.mark.parametrize("name", CASES)
def test_match_design_matches_matchit_svyglm(data, name, dname):
    ref = REF[name][dname]
    r = _fit(data, name, sp.svydesign(data, weights="sw", **DESIGNS[dname]))
    assert abs(r.estimate - ref["est"]) < 1e-9 * max(1.0, abs(ref["est"]))
    # R's weighted logit stops at glm's 1e-8 deviance criterion; the pairs
    # are identical, so the SE inherits only that rounding
    assert abs(r.se - ref["se"]) < 1e-8 * ref["se"], (r.se, ref["se"])
    assert r.model_info["survey_design"]["df"] > 0


def test_point_estimate_is_the_sampling_weights_estimate(data):
    """design= changes only the variance: the estimate is that of
    ``sampling_weights=`` with the same weights (identity)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.match(
            data, "y", "d", ["x1", "x2", "g"], sampling_weights="sw", m_order="largest"
        )
    b = _fit(data, "nn1_replace", sp.svydesign(data, weights="sw"))
    assert abs(a.estimate - b.estimate) < 1e-12


def test_match_design_refusals(data):
    des = sp.svydesign(data, weights="sw", cluster="psu")
    with pytest.raises(MethodIncompatibility, match="influence function"):
        sp.match(data, "y", "d", ["x1", "x2"], method="cem", design=des)
    with pytest.raises(MethodIncompatibility):
        sp.match(data, "y", "d", ["x1"], estimand="ATE", design=des)
    with pytest.raises(MethodIncompatibility):
        sp.match(data, "y", "d", ["x1"], sampling_weights="sw", design=des)
    with pytest.raises(MethodIncompatibility, match="weights"):
        sp.match(data, "y", "d", ["x1"], weights="psu", design=des)
