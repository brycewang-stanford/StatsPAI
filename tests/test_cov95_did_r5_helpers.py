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

from statspai.did.aggte import _weights_simple
from statspai.did.ddd_heterogeneous import _compute_ddd_gt
from statspai.did.timevarying_covariates import _compute_att_gt

# ----------------------------------------------------------------------
# timevarying_covariates._compute_att_gt
# ----------------------------------------------------------------------


def _tvc_panel(spec, times, age=lambda i, t: float(i)):
    return pd.DataFrame(
        [
            {
                "i": i,
                "year": t,
                "earn": float(i + t) + 0.1 * i * t,
                "g": g,
                "age": age(i, t),
            }
            for i, g in spec
            for t in times
        ]
    )


def _tvc_cells(df, cohorts):
    return _compute_att_gt(
        df,
        y="earn",
        unit="i",
        time="year",
        cohort="g",
        covariates=["age"],
        treated_cohorts=cohorts,
        never_value=0,
    )


def _assert_no_cells(out):
    assert np.isnan(out["att_group"])
    assert np.isnan(out["att_simple"])
    assert out["cell_estimates"] == []


def test_tvc_missing_pre_period_no_cells():
    # cohort g=3 but period 2 (= g-1, the base and covariate period) is absent
    df = _tvc_panel([(1, 3), (2, 3), (3, 0), (4, 0)], times=(1, 3))
    _assert_no_cells(_tvc_cells(df, [3]))


def test_tvc_too_few_controls_no_cells():
    # 2 never-treated units against an intercept + 1 covariate: n_control <= k
    df = _tvc_panel([(1, 4), (2, 4), (3, 0), (4, 0)], times=(3, 4))
    _assert_no_cells(_tvc_cells(df, [4]))


def test_tvc_single_treated_unit_is_estimated():
    # One treated unit is enough for the outcome-regression ATT(g,t) (the
    # R reference estimates it); only the control side needs n > k.
    df = _tvc_panel([(1, 4), (2, 0), (3, 0), (4, 0), (5, 0)], times=(3, 4))
    out = _tvc_cells(df, [4])
    assert len(out["cell_estimates"]) == 1
    cell = out["cell_estimates"][0]
    assert (cell["n_treated"], cell["n_control"]) == (1, 4)
    assert np.isfinite(out["att_simple"])


def test_tvc_nan_covariate_drops_controls_below_rank():
    # Covariates are frozen at g-1 = 3; NaN there removes two of four controls,
    # leaving n_control = 2 <= k = 2, so the cell is skipped.
    df = _tvc_panel(
        [(1, 4), (2, 4), (3, 0), (4, 0), (5, 0), (6, 0)],
        times=(3, 4),
        age=lambda i, t: np.nan if (t == 3 and i in (5, 6)) else float(i),
    )
    _assert_no_cells(_tvc_cells(df, [4]))


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
