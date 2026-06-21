"""Coverage round-5 for statspai.did.ddd_heterogeneous.

Heterogeneity-robust triple-differences for staggered adoption.  Covers
the validation errors, the bootstrap-rep skip + degenerate-SE branch,
the scalar-covariance reshape, the placebo-joint None fallback, and the
per-(g, t) helper edge paths (missing pre-period, empty subgroup slice,
no estimable cells).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from statspai.did.ddd_heterogeneous import (
    ddd_heterogeneous,
    _compute_ddd_gt,
    _group_time_did,
)


def make_ddd(seed=0, cohorts=(4, 6, 0), n_per=20, T=8):
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for g in cohorts:
        for _ in range(n_per):
            ufe = rng.normal()
            sub = int(rng.integers(0, 2))
            for t in range(1, T + 1):
                te = (max(0, t - g + 1) if g > 0 else 0) * sub
                rows.append(
                    {
                        "i": uid,
                        "year": t,
                        "earn": ufe + 0.2 * t + te + rng.normal() * 0.4,
                        "ft": g,
                        "aff": sub,
                    }
                )
            uid += 1
    return pd.DataFrame(rows)


def test_basic():
    df = make_ddd()
    r = ddd_heterogeneous(
        df,
        y="earn",
        unit="i",
        time="year",
        cohort="ft",
        subgroup="aff",
        n_boot=40,
        seed=1,
    )
    assert np.isfinite(r.estimate)
    assert r.model_info["placebo_joint_test"] is not None
    assert r.model_info["n_cohorts"] == 2
    assert len(r.detail) > 0


def test_placebo_joint_none_and_degenerate_se_with_one_boot():
    # n_boot=1 -> not enough valid rows -> placebo_joint None (line 194);
    # single bootstrap draw -> nanstd ddof=1 is NaN -> p/ci NaN (179-180).
    df = make_ddd()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = ddd_heterogeneous(
            df,
            y="earn",
            unit="i",
            time="year",
            cohort="ft",
            subgroup="aff",
            n_boot=1,
            seed=1,
        )
    assert r.model_info["placebo_joint_test"] is None
    assert np.isnan(r.pvalue)


def test_missing_column_raises():
    df = make_ddd()
    with pytest.raises(ValueError, match="not in data"):
        ddd_heterogeneous(
            df, y="nope", unit="i", time="year", cohort="ft", subgroup="aff"
        )


def test_non_binary_subgroup_raises():
    df = make_ddd()
    df.loc[0, "aff"] = 7
    with pytest.raises(ValueError, match="must be binary"):
        ddd_heterogeneous(
            df, y="earn", unit="i", time="year", cohort="ft", subgroup="aff"
        )


def test_no_treated_cohorts_raises():
    df = make_ddd(cohorts=(0, 0))
    with pytest.raises(ValueError, match="No treated cohorts"):
        ddd_heterogeneous(
            df, y="earn", unit="i", time="year", cohort="ft", subgroup="aff"
        )


def test_no_never_treated_raises():
    df = make_ddd(cohorts=(4, 6))
    with pytest.raises(ValueError, match="No never-treated"):
        ddd_heterogeneous(
            df, y="earn", unit="i", time="year", cohort="ft", subgroup="aff"
        )


# ----------------------------------------------------------------------
# Helper-level edge paths
# ----------------------------------------------------------------------


def test_group_time_did_empty_slice_returns_none():
    cohort_df = pd.DataFrame({"year": [1, 2], "aff": [1, 1], "earn": [1.0, 2]})
    never_df = pd.DataFrame({"year": [1, 2], "aff": [1, 1], "earn": [3.0, 4]})
    # sub_val=0 has no rows in either frame -> None (line 328-329)
    out = _group_time_did(
        cohort_df,
        never_df,
        y="earn",
        time="year",
        subgroup="aff",
        sub_val=0,
        t_pre=1,
        t_post=2,
    )
    assert out is None


def test_compute_ddd_gt_missing_pre_period_skips_cell():
    # Cohort g=3 but period 2 (= g-1) absent -> pre_period not in times
    # (line 249-250) for that cell; with no estimable cells -> no_cells (292).
    df = pd.DataFrame(
        {
            "i": [1, 1, 2, 2, 3, 3, 4, 4],
            "year": [1, 3, 1, 3, 1, 3, 1, 3],  # period 2 (=g-1) missing
            "earn": [1.0, 2, 3, 4, 5, 6, 7, 8],
            "ft": [3, 3, 3, 3, 0, 0, 0, 0],
            "aff": [1, 1, 0, 0, 1, 1, 0, 0],
        }
    )
    out = _compute_ddd_gt(
        df=df,
        y="earn",
        unit="i",
        time="year",
        cohort="ft",
        subgroup="aff",
        treated_cohorts=[3],
        never_value=0,
    )
    assert np.isnan(out["ddd_overall"])
    assert out["cell_estimates"] == []
