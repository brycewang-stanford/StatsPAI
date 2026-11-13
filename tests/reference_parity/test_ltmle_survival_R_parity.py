"""``sp.ltmle_survival`` against R ``ltmle`` with ``survivalOutcome = TRUE``.

Reference: ltmle 1.3.0 (GPL-2, run as a black box) on
``_fixtures/ltmle_survival.csv`` (1,200 subjects, three intervals,
time-varying covariate and treatment, right censoring;
``_generate_ltmle_survival_data.py``). ``ltmle`` estimates the risk at one
horizon per call, so ``_generate_ltmle_survival_R.R`` calls it once per
horizon on the data cut after ``Y_t``, with the package's default
main-terms glm formulas, ``variance.method = "ic"`` and regimes
always-treat / never-treat. Two designs: with the censoring nodes, and the
889 never-censored subjects without them.

Compared, per horizon: the counterfactual risk ``1 - S(t)`` under each
regime, its influence-curve standard error, and at the last horizon the
risk difference and its standard error.

Tolerance: 1e-8 absolute on the risks (observed 1.6e-9), 1e-8 relative on
the standard errors (observed 2e-10). The residual is R's glm deviance
tolerance, as in the ``sp.ltmle`` and ``sp.ltmle_msm`` fixtures.

R bounds the cumulative product of the treatment and censoring
probabilities at 0.01; ``sp.ltmle_survival`` bounds each treatment
probability at ``propensity_bounds``. Neither binds on these data (the
test checks it), so the comparison does not depend on that difference.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "ltmle_survival_R.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(_FIX / "ltmle_survival.csv")
NEVER = ~(DATA[["C1", "C2", "C3"]] == 0).any(axis=1)


def _fit(design, horizon=3, **kw):
    df = DATA if design == "censored" else DATA[NEVER].reset_index(drop=True)
    ks = range(1, horizon + 1)
    return sp.ltmle_survival(
        df,
        event_indicators=[f"Y{k}" for k in ks],
        treatments=[f"A{k}" for k in ks],
        covariates_time=[[f"L{k}"] for k in ks],
        baseline=["W"],
        censoring=[f"C{k}" for k in ks] if design == "censored" else None,
        **kw,
    )


def test_fixture_is_ltmle_output():
    assert REF["versions"]["ltmle"] == "1.3.0"
    assert REF["n_uncensored"] == int(NEVER.sum()) == 889
    assert REF["censored"][2]["qform"] == [
        "Q.kplus1 ~ W + L1 + A1",
        "Q.kplus1 ~ W + L1 + A1 + L2 + A2",
        "Q.kplus1 ~ W + L1 + A1 + L2 + A2 + L3 + A3",
    ]
    # a real effect and real censoring, so the comparison is not vacuous
    assert REF["censored"][2]["rd"] < -0.1
    assert DATA["C1"].eq(0).sum() > 50


@pytest.mark.parametrize("design", ["censored", "uncensored"])
def test_survival_curves_are_one_ltmle_call_per_horizon(design):
    res = _fit(design)
    se_t = res.detail["survival_se_treated"]
    se_c = res.detail["survival_se_control"]
    for t in range(3):
        ref = REF[design][t]
        assert 1 - res.survival_treated[t] == pytest.approx(
            ref["risk_treated"], abs=1e-8
        )
        assert 1 - res.survival_control[t] == pytest.approx(
            ref["risk_control"], abs=1e-8
        )
        assert se_t[t] == pytest.approx(ref["risk_treated_se"], rel=1e-8)
        assert se_c[t] == pytest.approx(ref["risk_control_se"], rel=1e-8)
    last = REF[design][2]
    assert res.risk_difference_final == pytest.approx(last["rd"], abs=1e-8)
    assert res.risk_difference_final_se == pytest.approx(last["rd_se"], rel=1e-8)
    # RMST is the sum of the survival probabilities
    assert res.rmst_treated == pytest.approx(
        3 - sum(r["risk_treated"] for r in REF[design]), abs=3e-8
    )
    assert res.rmst_control == pytest.approx(
        3 - sum(r["risk_control"] for r in REF[design]), abs=3e-8
    )


@pytest.mark.parametrize("design", ["censored", "uncensored"])
@pytest.mark.parametrize("horizon", [1, 2])
def test_shorter_follow_up_gives_the_same_early_horizons(design, horizon):
    res = _fit(design, horizon=horizon)
    ref = REF[design][horizon - 1]
    assert res.risk_difference_final == pytest.approx(ref["rd"], abs=1e-8)
    assert res.risk_difference_final_se == pytest.approx(ref["rd_se"], rel=1e-8)


def test_probability_bounds_do_not_bind_on_the_fixture():
    default = _fit("censored")
    unbounded = _fit("censored", propensity_bounds=(0.0, 1.0))
    assert np.array_equal(default.survival_treated, unbounded.survival_treated)
    assert np.array_equal(default.survival_control, unbounded.survival_control)
    lo, hi = zip(*default.detail["propensity_summary"])
    assert min(lo) > 0.02 and max(hi) < 0.98
