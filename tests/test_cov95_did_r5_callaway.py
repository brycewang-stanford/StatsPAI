"""Coverage round-5 for statspai.did.callaway_santanna.

Targets the exact still-missing line ranges in callaway_santanna.py:
validation errors, control-group / estimator / base-period variants,
zero-variance covariate drop, propensity / outcome-regression fallbacks,
RCS (panel=False) branch, empty-cell / not-enough-control guards, and
event-study / pretrend aggregation edge paths.

All panels are real synthetic staggered DiD data; assertions check real
structural properties (signs, shapes, finiteness), never fabricated
numbers.
"""

import numpy as np
import pandas as pd
import pytest

from statspai.did.callaway_santanna import (
    CallawayNotImplemented,
    callaway_santanna,
    _get_gt_pairs,
    _estimate_pscore,
    _estimate_outcome_reg,
    _aggregate_simple,
    _aggregate_event_study,
    _pretrend_test,
    _estimate_single_att_rcs,
    _rcs_residualise_on_controls,
)
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def make_panel(seed=0, cohorts=(4, 6, 0), n_per=25, T=8, x=False, const_x=False):
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for g in cohorts:
        for _ in range(n_per):
            u_fe = rng.normal()
            xv = 1.0 if const_x else rng.normal()
            for t in range(1, T + 1):
                te = max(0, t - g + 1) if g > 0 else 0
                yv = u_fe + 0.3 * t + te + rng.normal() * 0.5 + (0.5 * xv if x else 0)
                row = {"i": uid, "t": t, "y": yv, "g": g}
                if x:
                    row["x1"] = xv
                rows.append(row)
            uid += 1
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Validation errors (lines 122, 123-127, 128-129, 160, 165, 317, 321)
# ----------------------------------------------------------------------


def test_bad_estimator_raises():
    df = make_panel()
    with pytest.raises(ValueError, match="estimator must be"):
        callaway_santanna(df, y="y", g="g", t="t", i="i", estimator="bogus")


def test_bad_control_group_raises():
    df = make_panel()
    with pytest.raises(ValueError, match="control_group must be"):
        callaway_santanna(df, y="y", g="g", t="t", i="i", control_group="weird")


def test_negative_anticipation_raises():
    df = make_panel()
    with pytest.raises(ValueError, match="anticipation must be"):
        callaway_santanna(df, y="y", g="g", t="t", i="i", anticipation=-1)


def test_missing_column_raises():
    df = make_panel()
    with pytest.raises(ValueError, match="not found in data"):
        callaway_santanna(df, y="nope", g="g", t="t", i="i")


def test_missing_covariate_raises():
    df = make_panel()
    with pytest.raises(ValueError, match="Covariate"):
        callaway_santanna(df, y="y", g="g", t="t", i="i", x=["missing_x"])


def test_validation_errors_expose_taxonomy_and_scalar_x():
    df = make_panel(x=True)
    r = callaway_santanna(df, y="y", g="g", t="t", i="i", x="x1")
    assert np.isfinite(r.estimate)
    with pytest.raises(MethodIncompatibility, match="estimator must be"):
        callaway_santanna(df, y="y", g="g", t="t", i="i", estimator="bogus")
    with pytest.raises(CallawayNotImplemented, match="estimator='reg'"):
        callaway_santanna(df, y="y", g="g", t="t", i="i", panel=False, estimator="dr")


def test_no_cohorts_raises():
    # everyone never-treated
    df = make_panel(cohorts=(0, 0))
    with pytest.raises(DataInsufficient, match="No treatment cohorts"):
        callaway_santanna(df, y="y", g="g", t="t", i="i")


def test_no_gt_pairs_raises():
    # All cohorts treated at the first period -> no pre-period -> no pairs
    df = make_panel(cohorts=(1, 0), T=4)
    with pytest.raises(ValueError, match="No valid"):
        callaway_santanna(df, y="y", g="g", t="t", i="i")


# ----------------------------------------------------------------------
# Estimator / control-group / base-period variants
# ----------------------------------------------------------------------


@pytest.mark.parametrize("est", ["dr", "ipw", "reg"])
@pytest.mark.parametrize("ctrl", ["nevertreated", "notyettreated"])
def test_estimator_control_grid(est, ctrl):
    df = make_panel()
    r = callaway_santanna(
        df, y="y", g="g", t="t", i="i", estimator=est, control_group=ctrl
    )
    assert np.isfinite(r.estimate)
    assert r.model_info["estimator"] == est.upper()
    assert r.model_info["control_group"] == ctrl


def test_varying_base_with_anticipation():
    df = make_panel()
    r = callaway_santanna(
        df, y="y", g="g", t="t", i="i", base_period="varying", anticipation=1
    )
    assert np.isfinite(r.estimate)
    assert r.model_info["base_period"] == "varying"
    assert r.model_info["anticipation"] == 1


def test_dr_with_covariates():
    df = make_panel(x=True)
    r = callaway_santanna(df, y="y", g="g", t="t", i="i", x=["x1"], estimator="dr")
    assert np.isfinite(r.estimate)


def test_ipw_with_covariates():
    df = make_panel(x=True)
    r = callaway_santanna(df, y="y", g="g", t="t", i="i", x=["x1"], estimator="ipw")
    assert np.isfinite(r.estimate)


def test_reg_with_covariates():
    df = make_panel(x=True)
    r = callaway_santanna(df, y="y", g="g", t="t", i="i", x=["x1"], estimator="reg")
    assert np.isfinite(r.estimate)


def test_zero_variance_covariate_dropped():
    # Constant covariate -> zero variance -> drop branch (lines 441-445).
    df = make_panel(x=True, const_x=True)
    r = callaway_santanna(df, y="y", g="g", t="t", i="i", x=["x1"], estimator="dr")
    assert np.isfinite(r.estimate)


def test_two_constant_covariates_all_dropped():
    df = make_panel(x=True, const_x=True)
    df["x2"] = 2.0  # also constant -> keep.sum()==0 path (443)
    r = callaway_santanna(
        df, y="y", g="g", t="t", i="i", x=["x1", "x2"], estimator="reg"
    )
    assert np.isfinite(r.estimate)


# ----------------------------------------------------------------------
# _get_gt_pairs edge: cohort with no available pre-period (line 364)
# ----------------------------------------------------------------------


def test_get_gt_pairs_skips_cohort_without_pre():
    # cohort treated at the very first available period -> no pre -> skipped
    pairs = _get_gt_pairs(
        cohorts=[1, 4],
        time_periods=[1, 2, 3, 4, 5],
        base_period="universal",
        anticipation=0,
    )
    groups = {g for g, _, _ in pairs}
    assert 1 not in groups  # cohort 1 dropped (no pre-period)
    assert 4 in groups


# ----------------------------------------------------------------------
# Nuisance estimators direct (pscore / outcome-reg fallbacks)
# ----------------------------------------------------------------------


def test_pscore_no_covariates_returns_constant():
    d = np.array([1.0, 0, 1, 0, 1, 0])
    ps = _estimate_pscore(d, None, n=6)
    assert np.allclose(ps, d.mean())


def test_pscore_empty_covariate_columns():
    d = np.array([1.0, 0, 1, 0])
    ps = _estimate_pscore(d, np.zeros((4, 0)), n=4)
    assert np.allclose(ps, d.mean())


def test_outcome_reg_no_covariates():
    dy = np.array([1.0, 2, 3, 4])
    c = np.array([1.0, 1, 0, 0])
    m = _estimate_outcome_reg(dy, c, None, n=4)
    assert np.allclose(m, dy[c.astype(bool)].mean())


def test_outcome_reg_too_few_controls_for_regression():
    # 2 controls, 2 covariates -> c_count <= k+1 -> mean fallback (line 653)
    dy = np.array([1.0, 2, 3, 4, 5])
    c = np.array([1.0, 1, 0, 0, 0])
    x = np.random.default_rng(0).normal(size=(5, 2))
    m = _estimate_outcome_reg(dy, c, x, n=5)
    assert np.allclose(m, dy[c.astype(bool)].mean())


# ----------------------------------------------------------------------
# Aggregation helpers edge paths
# ----------------------------------------------------------------------


def test_aggregate_simple_empty():
    empty = pd.DataFrame({"group": [], "att": [], "se": []})
    est, se, pval, ci = _aggregate_simple(empty, None, pd.Series(dtype=float), 10, 0.05)
    assert est == 0.0 and np.isinf(se) and pval == 1.0


def test_aggregate_simple_analytic_se_when_inf_none():
    detail = pd.DataFrame({"group": [4, 6], "att": [1.0, 2.0], "se": [0.1, 0.2]})
    cs = pd.Series({4: 25, 6: 25})
    est, se, pval, ci = _aggregate_simple(detail, None, cs, 50, 0.05)
    assert np.isfinite(est) and se > 0


def test_aggregate_event_study_analytic_se_when_inf_none():
    detail = pd.DataFrame(
        {
            "group": [4, 6],
            "relative_time": [0, 0],
            "att": [1.0, 2.0],
            "se": [0.1, 0.2],
        }
    )
    cs = pd.Series({4: 25, 6: 25})
    es = _aggregate_event_study(detail, None, cs, 50, 0.05)
    assert len(es) == 1
    assert np.isfinite(es["se"].iloc[0])


def test_pretrend_test_no_inf_matrix():
    detail = pd.DataFrame(
        {
            "relative_time": [-2, -1, 0],
            "att": [0.05, -0.03, 1.0],
            "se": [0.1, 0.1, 0.2],
        }
    )
    out = _pretrend_test(detail, None, 50)
    assert out["df"] == 2 and np.isfinite(out["statistic"])


def test_pretrend_test_no_pre_periods():
    detail = pd.DataFrame(
        {"relative_time": [0, 1], "att": [1.0, 2.0], "se": [0.1, 0.2]}
    )
    out = _pretrend_test(detail, None, 50)
    assert out["df"] == 0 and np.isnan(out["statistic"])


# ----------------------------------------------------------------------
# Repeated cross-sections branch (panel=False)
# ----------------------------------------------------------------------


def test_rcs_basic():
    df = make_panel()
    r = callaway_santanna(df, y="y", g="g", t="t", i="i", panel=False, estimator="reg")
    assert np.isfinite(r.estimate)
    assert "repeated cross-sections" in r.method
    assert r.model_info["panel"] is False


def test_rcs_with_covariates_residualises():
    df = make_panel(x=True)
    r = callaway_santanna(
        df, y="y", g="g", t="t", i="i", x=["x1"], panel=False, estimator="reg"
    )
    assert np.isfinite(r.estimate)
    assert "covariates" in r.model_info["estimator"]


def test_rcs_requires_reg():
    df = make_panel()
    with pytest.raises(NotImplementedError, match="estimator='reg'"):
        callaway_santanna(df, y="y", g="g", t="t", i="i", panel=False, estimator="dr")


def test_rcs_requires_nevertreated():
    df = make_panel()
    with pytest.raises(NotImplementedError, match="nevertreated"):
        callaway_santanna(
            df,
            y="y",
            g="g",
            t="t",
            i="i",
            panel=False,
            estimator="reg",
            control_group="notyettreated",
        )


def test_rcs_missing_column():
    df = make_panel()
    with pytest.raises(ValueError, match="not found in data"):
        callaway_santanna(
            df, y="nope", g="g", t="t", i="i", panel=False, estimator="reg"
        )


def test_rcs_missing_covariate():
    df = make_panel()
    with pytest.raises(ValueError, match="Covariate"):
        callaway_santanna(
            df, y="y", g="g", t="t", i="i", x=["zzz"], panel=False, estimator="reg"
        )


def test_rcs_all_nan_raises():
    df = make_panel()
    df["y"] = np.nan
    with pytest.raises(ValueError, match="No observations"):
        callaway_santanna(df, y="y", g="g", t="t", i="i", panel=False, estimator="reg")


def test_rcs_no_cohorts_raises():
    df = make_panel(cohorts=(0, 0))
    with pytest.raises(ValueError, match="No treatment cohorts"):
        callaway_santanna(df, y="y", g="g", t="t", i="i", panel=False, estimator="reg")


def test_rcs_no_gt_pairs_raises():
    df = make_panel(cohorts=(1, 0), T=4)
    with pytest.raises(ValueError, match="No valid"):
        callaway_santanna(df, y="y", g="g", t="t", i="i", panel=False, estimator="reg")


def test_rcs_single_att_empty_cell():
    # base_val absent from data -> empty cell -> inf SE (line 975)
    y = np.array([1.0, 2, 3, 4, 5, 6])
    g = np.array([4, 4, 0, 0, 4, 0])
    t = np.array([4, 4, 4, 4, 5, 5])
    att, se, inf = _estimate_single_att_rcs(
        y, g, t, g_val=4, t_val=5, base_val=99, n_obs=6
    )
    assert att == 0.0 and np.isinf(se)


def test_rcs_residualise_not_enough_controls():
    # < k+2 controls -> return untouched (line 1028)
    df = pd.DataFrame(
        {
            "g": [0, 4, 4, 4],
            "t": [1, 1, 2, 2],
            "x1": [1.0, 2, 3, 4],
        }
    )
    y = np.array([1.0, 2, 3, 4])
    out = _rcs_residualise_on_controls(y, df, "g", "t", ["x1"])
    assert np.allclose(out, y)
