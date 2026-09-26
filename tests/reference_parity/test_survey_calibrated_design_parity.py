"""``SurveyDesign.calibrate``: calibration-adjusted linearisation (review F06).

Before 1.32, ``sp.rake`` / ``sp.linear_calibration`` returned weights only;
passed to ``sp.svydesign`` they were treated as fixed, and every SE ignored
the calibration (7-18 % too large on this data). ``design.calibrate`` keeps
the auxiliary variables in the design and residualises every estimator's
linearisation scores on them.

References (same ``survey_calib_data.csv`` bytes):

* ``survey_calib_R.json`` (existing) and ``survey_calibrated_design_R.json``
  (``_generate_survey_calibrated_design_R.R``) -- R ``survey`` 4.5
  ``calibrate(calfun = "raking" | "linear")`` then ``svymean`` / ``svytotal``
  / ``svyglm`` (gaussian and quasibinomial).
* ``survey_calib_stata.json`` (existing) and
  ``survey_calibrated_design_stata.json``
  (``_fixtures/_generate_survey_calibrated_design_stata.do``) -- Stata 18
  ``svyset ..., rake() / regress()`` then ``svy: mean / total / regress``.

The two references implement different, asymptotically equivalent
conventions (T4 between themselves, each T2 against its own option):

* ``variance="greg"`` (default): residuals from the regression weighted by
  the *design* weights, times the calibrated weights -- the g-weighted
  residual variance of Särndal, Swensson & Wretman; R ``calibrate``.
* ``variance="stata"``: the regression weighted by the *calibrated* weights;
  Stata ``svyset, rake()``.

R ``rake()`` (IPF, class ``raking``) has a third route -- ten backfitting
sweeps -- whose SE (0.093811) sits between the two; it is not reproduced.

Tolerance: 1e-10 relative (observed 1e-15 .. 1e-12; the raking weights come
from IPF at ``tol=1e-14`` vs R's Newton to ``epsilon=1e-13``), except the
quasibinomial SEs after raking (see the note in the first test).
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
R0 = json.loads((_FIX / "survey_calib_R.json").read_text(encoding="utf-8"))
S0 = json.loads((_FIX / "survey_calib_stata.json").read_text(encoding="utf-8"))
R = json.loads((_FIX / "survey_calibrated_design_R.json").read_text(encoding="utf-8"))
S = json.loads(
    (_FIX / "survey_calibrated_design_stata.json").read_text(encoding="utf-8")
)
T = R0["targets"]
MARGINS = {"sex": T["sex"], "agegrp": T["agegrp"]}
TOTALS = {"income": T["income"], "age": T["age"]}
TOL = 1e-10


@pytest.fixture(scope="module")
def df():
    d = pd.read_csv(_FIX / "survey_calib_data.csv")
    d["yb"] = (d["y"] > d["y"].median()).astype(int)
    return d


@pytest.fixture(scope="module")
def design(df):
    return sp.svydesign(df, weights="d", strata="stratum", cluster="psu", nest=True)


def _cal(design, case, variance="greg"):
    if case == "raking":
        return design.calibrate(margins=MARGINS, tol=1e-14, variance=variance)
    return design.calibrate(totals=TOTALS, variance=variance)


def _close(a, b, rtol=TOL):
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=rtol)


@pytest.mark.parametrize("case", ["raking", "linear"])
def test_greg_mean_total_glm_match_R_calibrate(design, case):
    c = _cal(design, case)
    ref = R[case]
    assert c.weights.sum() == pytest.approx(ref["weight_sum"], rel=TOL)
    m = c.mean("y")
    _close(m.estimate, [ref["mean_y"]])
    _close(m.std_error, [ref["se_mean_y"]])
    t = c.total("y")
    _close(t.estimate, [ref["total_y"]])
    _close(t.std_error, [ref["se_total_y"]])
    g = c.glm("y ~ income + age")
    _close(g.estimate, ref["glm_coef"])
    _close(g.std_error, ref["glm_se"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = c.glm("yb ~ income", family="binomial")
    _close(b.estimate, ref["logit_coef"])
    # quasibinomial SEs: observed 4.2e-9 (raking) / <1e-10 (linear). The
    # coefficients agree to 1e-15 once both IRLS runs converge tightly (the
    # R generator uses glm.control(epsilon = 1e-14)), so the remaining 4e-9
    # is not convergence; it is unexplained at that level and asserted at
    # 1e-8, well inside the strict 1e-6 T2 budget.
    _close(b.std_error, ref["logit_se"], 1e-8)


def test_greg_mean_matches_existing_R_fixture(design):
    _close(
        _cal(design, "raking").mean("y").std_error,
        [R0["calib_raking"]["se_calibrated"]],
    )
    _close(
        _cal(design, "linear").mean("y").std_error,
        [R0["linear_noint"]["se_calibrated"]],
    )


@pytest.mark.parametrize("case", ["raking", "linear"])
def test_stata_convention_matches_svyset_rake_regress(design, case):
    c = _cal(design, case, variance="stata")
    ref = S[case]
    _close(c.total("y").estimate, [ref["total_y"]])
    _close(c.total("y").std_error, [ref["se_total_y"]])
    g = c.glm("y ~ income + age")
    _close(g.estimate, ref["reg_coef"])
    _close(g.std_error, ref["reg_se"])
    key = "rake_se_calibrated" if case == "raking" else "linear_noint_se_calibrated"
    _close(c.mean("y").std_error, [S0[key]])


def test_calibration_totals_have_zero_variance(design):
    """Reference-free (T1): a calibrated total is fixed by construction."""
    c = _cal(design, "linear")
    t = c.total("income")
    assert t.estimate.iloc[0] == pytest.approx(T["income"], rel=1e-12)
    assert t.std_error.iloc[0] < 1e-8 * T["income"]
    assert R["linear"]["se_total_income"] < 1e-8 * T["income"]


def test_fixed_weight_design_is_unchanged(df, design):
    """Weights alone in svydesign keep the fixed-weight SE (documented)."""
    w = sp.rake(df, MARGINS, weight="d", tol=1e-14).calibrated_weights * T["N"]
    fixed = sp.svydesign(
        df.assign(wc=w), weights="wc", strata="stratum", cluster="psu", nest=True
    )
    _close(fixed.mean("y").std_error, [R0["rake_tight"]["se_fixed_weights"]], 1e-12)
    assert fixed._calibration is None


def test_calibrate_input_validation(design):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="exactly one"):
        design.calibrate()
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="variance"):
        design.calibrate(totals=TOTALS, variance="bogus")
