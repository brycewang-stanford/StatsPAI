"""``sp.aggte`` vs R ``did::aggte`` on the bundled Cheng-Hoekstra panel.

Reference values come from R ``did`` 4.x on this machine, run against the
**same CSV this test loads** (``statspai/datasets/data/castle_2013.csv``)::

    library(did)
    df <- read.csv("castle_2013.csv")
    df$gvar <- ifelse(is.na(df$effyear), 0, df$effyear)
    out <- att_gt(yname="l_homicide", tname="year", idname="sid",
                  gname="gvar", data=df, control_group="nevertreated",
                  bstrap=FALSE, cband=FALSE, est_method="dr",
                  base_period="universal")
    for (ty in c("simple","dynamic","group","calendar"))
        aggte(out, type=ty, bstrap=FALSE, cband=FALSE)

``base_period="universal"`` matches StatsPAI's default; R's own default is
``"varying"``, which reports different *pre-treatment* cells (post-treatment
cells and every overall aggregate are identical either way).  Getting this
wrong makes StatsPAI look broken when it is not.

The castle panel is used rather than ``did::mpdta`` because it ships with
StatsPAI under the MIT-licensed mixtape redistribution — ``mpdta`` lives
inside the GPL-licensed ``did`` package and is not ours to vendor.

Why this file exists
--------------------
StatsPAI's aggregation standard errors were up to 8% too small: the
estimated cohort-share weights were treated as fixed, dropping R's
``did:::wif`` term.  Point estimates were always correct, so no
self-consistency test could catch it — only a cross-implementation check.
These assertions are that check.
"""

from __future__ import annotations

import numpy as np
import pytest

import statspai as sp

# R did 4.x, base_period="universal", bstrap=FALSE, est_method="dr" -------
R_OVERALL = {
    # type -> (overall att, overall se)
    "simple": (0.1103830284, 0.0387242393),
    "dynamic": (0.1102807445, 0.0366700461),
    "group": (0.1084474773, 0.0363328223),
    "calendar": (0.0741756563, 0.0314891268),
}

R_DYNAMIC_CELLS = {
    -9: (-0.40396745, 0.05714633),
    -5: (0.03162591, 0.06098663),
    -2: (0.05791600, 0.04377078),
    0: (0.09721535, 0.03964314),
    1: (0.11154911, 0.04932118),
    3: (0.13682539, 0.05724294),
    5: (0.11194189, 0.05085404),
}

R_GROUP_CELLS = {
    2005: (0.09306977, 0.03243296),
    2006: (0.10994502, 0.05268143),
    2007: (0.12840220, 0.05133150),
    2008: (0.12212066, 0.05672632),
    2009: (-0.00280808, 0.03850197),
}

R_CALENDAR_CELLS = {
    2005: (-0.12027707, 0.03584758),
    2006: (0.10735135, 0.04687581),
    2007: (0.15790057, 0.05544211),
    2008: (0.04012517, 0.06690213),
    2009: (0.16765242, 0.05479950),
    2010: (0.09230150, 0.04908496),
}

# The CSV round-trips Stata float storage through text, so agreement is
# ~1e-7 rather than machine epsilon.
ATOL = 1e-6


def _se(result) -> float:
    """Scalar SE from an aggte result (``std_errors`` may be a 1-element Series)."""
    se = result.std_errors
    return float(se.iloc[0]) if hasattr(se, "iloc") else float(se)


@pytest.fixture(scope="module")
def cs():
    df = sp.datasets.castle_doctrine(event_time=True)
    return sp.callaway_santanna(
        df,
        y="l_homicide",
        g="gvar",
        t="year",
        i="sid",
        control_group="nevertreated",
    )


@pytest.mark.parametrize("agg_type", sorted(R_OVERALL))
def test_overall_matches_r_did(cs, agg_type):
    """Overall ATT *and* SE must match R — the SE is the regression guard."""
    r_att, r_se = R_OVERALL[agg_type]
    out = sp.aggte(cs, type=agg_type, bstrap=False, cband=False)
    assert float(out.estimate) == pytest.approx(r_att, abs=ATOL)
    assert _se(out) == pytest.approx(r_se, abs=ATOL)


def test_headline_se_equals_simple_aggregation(cs):
    """``callaway_santanna``'s own SE is the simple aggregation's SE.

    Both must carry the weight-estimation term.  If only one does, the
    headline and ``aggte(type='simple')`` silently disagree — which is
    how the original defect first became visible.
    """
    simple = sp.aggte(cs, type="simple", bstrap=False)
    assert cs.se == pytest.approx(_se(simple), abs=1e-12)
    assert cs.se == pytest.approx(R_OVERALL["simple"][1], abs=ATOL)


@pytest.mark.parametrize(
    "agg_type,key,cells",
    [
        ("dynamic", "relative_time", R_DYNAMIC_CELLS),
        ("group", "group", R_GROUP_CELLS),
        ("calendar", "time", R_CALENDAR_CELLS),
    ],
)
def test_cells_match_r_did(cs, agg_type, key, cells):
    out = sp.aggte(cs, type=agg_type, bstrap=False, cband=False)
    got = out.detail.set_index(key)
    for label, (r_att, r_se) in cells.items():
        row = got.loc[label]
        assert float(row["att"]) == pytest.approx(
            r_att, abs=ATOL
        ), f"{agg_type} ATT at {key}={label}"
        assert float(row["se"]) == pytest.approx(
            r_se, abs=ATOL
        ), f"{agg_type} SE at {key}={label}"


def test_single_cohort_aggregate_has_zero_weight_influence(cs):
    """Within one cohort the estimated shares cancel out of the weights.

    This is the analytic signature of the fix, and it explains why the
    defect stayed hidden: every single-cohort aggregate was already
    correct, so internal consistency checks all passed.
    """
    from statspai.did._core import cohort_share_context, weight_influence

    unit_cohorts = cs.model_info["_unit_cohorts"]
    for g in R_GROUP_CELLS:
        pg, ind = cohort_share_context(np.full(4, float(g)), unit_cohorts)
        assert np.allclose(weight_influence(pg, ind), 0.0, atol=1e-12)


def test_mixed_cohort_aggregate_has_nonzero_weight_influence(cs):
    """...and conversely, cross-cohort aggregates genuinely need the term."""
    from statspai.did._core import cohort_share_context, weight_influence

    pg, ind = cohort_share_context(
        np.array([2005.0, 2006.0, 2007.0]), cs.model_info["_unit_cohorts"]
    )
    assert np.abs(weight_influence(pg, ind)).max() > 1e-6


def test_influence_export_roundtrip_keeps_the_correction(cs):
    """``aggte_from_influence`` must reproduce ``aggte`` exactly.

    The exported frame carries ``unit_cohort`` precisely so the weight
    term can be rebuilt; dropping it silently shrinks the SEs again.
    """
    rif = sp.influence_functions(cs)
    for agg_type in ("simple", "dynamic", "group", "calendar"):
        direct = sp.aggte(cs, type=agg_type, bstrap=False, cband=False)
        roundtrip = sp.aggte_from_influence(
            rif, type=agg_type, bstrap=False, cband=False
        )
        assert _se(roundtrip) == pytest.approx(
            _se(direct), abs=1e-12
        ), f"{agg_type} SE lost the weight-estimation term on roundtrip"
