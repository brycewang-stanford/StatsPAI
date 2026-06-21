"""Coverage round-5 helper-level edge paths.

Direct unit tests of internal (g, t) cell builders that the public-API
tests cannot route a real panel into without an unreasonable amount of
data shaping:

- timevarying_covariates._compute_att_gt: missing pre-period, too-few
  treated/control, rank-deficient design (n <= k), no estimable cells.
- ddd_heterogeneous._compute_ddd_gt: empty-subgroup DID None skip,
  zero affected-treated count skip.
- aggte._weights_simple: no post-treatment cells.
"""

import numpy as np
import pandas as pd

from statspai.did.timevarying_covariates import _compute_att_gt
from statspai.did.ddd_heterogeneous import _compute_ddd_gt
from statspai.did.aggte import _weights_simple

# ----------------------------------------------------------------------
# timevarying_covariates._compute_att_gt
# ----------------------------------------------------------------------


def test_tvc_missing_pre_period_no_cells():
    # cohort g=3 but period 2 (= g-1) absent -> pre_t not in times (259) -> no cells (303)
    rows = []
    for i, g in [(1, 3), (2, 3), (3, 0), (4, 0)]:
        for t in (1, 3):  # period 2 missing
            rows.append(
                {"i": i, "year": t, "earn": float(i + t), "g": g, "age_base": float(i)}
            )
    df = pd.DataFrame(rows)
    out = _compute_att_gt(
        df,
        y="earn",
        unit="i",
        time="year",
        cohort="g",
        covariates=["age"],
        treated_cohorts=[3],
        never_value=0,
    )
    assert np.isnan(out["att_overall"])
    assert out["cell_estimates"] == []


def test_tvc_too_few_treated_no_cells():
    # Only one treated unit -> n_treated < 2 -> cell skipped (276-277)
    rows = []
    for i, g in [(1, 4), (2, 0), (3, 0), (4, 0)]:
        for t in (3, 4):
            rows.append(
                {"i": i, "year": t, "earn": float(i + t), "g": g, "age_base": float(i)}
            )
    df = pd.DataFrame(rows)
    out = _compute_att_gt(
        df,
        y="earn",
        unit="i",
        time="year",
        cohort="g",
        covariates=["age"],
        treated_cohorts=[4],
        never_value=0,
    )
    assert np.isnan(out["att_overall"])


def test_tvc_rank_deficient_design_skipped():
    # 2 treated + 2 control but NaN covariate in the post period drops rows
    # so the valid design has n <= k (line 288-289).
    rows = []
    for i, g in [(1, 4), (2, 4), (3, 0), (4, 0)]:
        for t in (3, 4):
            age = np.nan if t == 4 else float(i)
            rows.append(
                {"i": i, "year": t, "earn": float(i + t), "g": g, "age_base": age}
            )
    df = pd.DataFrame(rows)
    out = _compute_att_gt(
        df,
        y="earn",
        unit="i",
        time="year",
        cohort="g",
        covariates=["age"],
        treated_cohorts=[4],
        never_value=0,
    )
    assert np.isnan(out["att_overall"])


# ----------------------------------------------------------------------
# ddd_heterogeneous._compute_ddd_gt
# ----------------------------------------------------------------------


def test_ddd_compute_gt_no_affected_cells():
    # Treated cohort has only unaffected (sub=0) units; never-treated has
    # both -> DID_b1 None for affected slice OR n_treated_affected == 0
    # -> every cell skipped (lines 273 / 277-278) -> no cells.
    rows = []
    for i, g, sub in [(1, 4, 0), (2, 4, 0), (3, 0, 0), (4, 0, 0), (5, 0, 1), (6, 0, 1)]:
        for t in (3, 4, 5):
            rows.append({"i": i, "year": t, "earn": float(i + t), "ft": g, "aff": sub})
    df = pd.DataFrame(rows)
    out = _compute_ddd_gt(
        df=df,
        y="earn",
        unit="i",
        time="year",
        cohort="ft",
        subgroup="aff",
        treated_cohorts=[4],
        never_value=0,
    )
    assert out["cell_estimates"] == []
    assert np.isnan(out["ddd_overall"])


# ----------------------------------------------------------------------
# aggte._weights_simple no post-treatment cells
# ----------------------------------------------------------------------


def test_weights_simple_no_post_cells():
    detail = pd.DataFrame({"group": [4, 4], "relative_time": [-2, -1]})
    labels, W = _weights_simple(detail, pd.Series({4: 10.0}))
    assert W.shape == (0, 2)
    assert list(labels) == ["overall"]
