"""Reference parity: not-yet-treated control sets of ``sp.callaway_santanna``.

R ``did`` (``get_did_cohort_index`` / ``compute.att_gt``) admits a unit as a
not-yet-treated control for cell ``(g, t)`` when its cohort exceeds
``time_periods[max(t, pret) + tfac] + anticipation`` -- untreated through
*both* periods of the 2x2 comparison.  Before 1.29 StatsPAI used ``G > t``,
which (i) under ``base_period="universal"`` let a cohort first treated
between ``t`` and the base period ``g - 1`` serve as a control for a
pre-treatment placebo although its base-period outcome was already treated,
and (ii) ignored ``anticipation`` in the control set.  On the fixture the
old rule moved cohort-6 placebo ATTs by up to 1.0.

The fixture (``_fixtures/_generate_cs_notyet_cutoff*.{py,R}``) runs
``did::att_gt(control_group="notyettreated", est_method="dr")`` for panel and
repeated cross-section data, both base periods, ``anticipation`` in {0, 1},
with and without covariates.  Tolerance ``rtol = 1e-8`` on ATT and SE
(observed worst 3.6e-12 / 8.9e-11): same data, same estimator.

Regenerate with::

    python tests/reference_parity/_fixtures/_generate_cs_notyet_cutoff_data.py
    Rscript tests/reference_parity/_fixtures/_generate_cs_notyet_cutoff.R
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_DIR = pathlib.Path(__file__).parent / "_fixtures"
_CSV = _DIR / "cs_notyet_cutoff_data.csv"
_JSON = _DIR / "cs_notyet_cutoff_R.json"

pytestmark = pytest.mark.skipif(
    not (_CSV.exists() and _JSON.exists()),
    reason="CS not-yet-treated cutoff fixture is not materialized",
)


def _cases():
    if not _JSON.exists():
        return []
    return json.loads(_JSON.read_text(encoding="utf-8"))["cases"]


def _case_id(case):
    return (
        f"{'panel' if case['panel'] else 'rcs'}-{case['base_period']}-"
        f"ant{case['anticipation']}-{'x' if case['covariates'] else 'nox'}"
    )


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_CSV)


@pytest.mark.parametrize("case", _cases(), ids=_case_id)
def test_notyettreated_att_gt_matches_r_did(data, case):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.callaway_santanna(
            data,
            y="y",
            g="g",
            t="t",
            i="id",
            x=["x1"] if case["covariates"] else None,
            control_group="notyettreated",
            base_period=case["base_period"],
            anticipation=int(case["anticipation"]),
            estimator="dr",
            panel=case["panel"],
            # R's sign for the pre-period comparison-regression term on
            # repeated cross-sections (documented T4; default is 'delta').
            variance_convention="reference",
        )
    ours = res.detail.set_index(["group", "time"])
    ref = pd.DataFrame(
        {
            "group": case["group"],
            "time": case["time"],
            "att": case["att"],
            "se": case["se"],
        }
    ).set_index(["group", "time"])
    joined = ref.join(ours[["att", "se"]], rsuffix="_sp", how="left")
    assert joined["att_sp"].notna().all(), "cells missing on the StatsPAI side"
    np.testing.assert_allclose(joined["att_sp"], joined["att"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(joined["se_sp"], joined["se"], rtol=1e-8)


def test_fixture_exercises_cells_where_the_cutoffs_differ(data):
    """Guard: the design must contain cells whose control set changes between
    ``G > t`` and ``G > max(t, base) + anticipation``; otherwise the parity
    above could not detect the bug it pins."""
    cohorts = sorted(c for c in data["g"].unique() if c > 0)
    periods = sorted(data["t"].unique())
    differing = 0
    for anticipation in (0, 1):
        for g in cohorts:
            base = g - 1 - anticipation
            for t in periods:
                if t == base:
                    continue
                old = {c for c in cohorts if c > t and c != g}
                new = {c for c in cohorts if c > max(t, base) + anticipation and c != g}
                differing += old != new
    assert differing >= 5
