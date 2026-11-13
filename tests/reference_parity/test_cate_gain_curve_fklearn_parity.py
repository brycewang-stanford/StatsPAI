"""``sp.cate_gain_curve`` against ``fklearn``'s effect curves.

Reference: fklearn 4.3.0 (Apache-2.0),
``fklearn.causal.validation.curves.effect_curves`` and
``effect_by_segment`` with the default ``linear_effect`` (the least-squares
slope of the outcome on the treatment), on
``_fixtures/cate_gain_curve.csv``: 3,000 rows of a randomized experiment
with a continuous and a binary treatment, ranked by an informative score
and by noise (``_generate_cate_gain_curve_fklearn.py`` ->
``cate_gain_curve_fklearn.json``).

With 3,000 rows and 100 steps both sides evaluate the curve on the top
30, 60, ..., 3,000 units, so the comparison is point by point:

* the cumulative effect at every step;
* the cumulative gain (``normalize=False``) and the relative cumulative
  gain (``normalize=True``);
* the area, which is the sum of the relative gain over the steps;
* the effect in each of ten quantile groups of the ranking;
* the average effect.

Tolerance 1e-10 relative with 1e-12 absolute (observed 4e-14). The
rankings have no ties: with ties the order within a tie is not defined on
either side.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "cate_gain_curve_fklearn.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(_FIX / "cate_gain_curve.csv")
CASES = {
    "dose_score": ("dose", "y_dose", "score"),
    "dose_noise": ("dose", "y_dose", "noise"),
    "d_score": ("d", "y_bin", "score"),
    "d_noise": ("d", "y_bin", "noise"),
}
TOL = dict(rtol=1e-10, atol=1e-12)


def _fit(case, **kw):
    treat, outcome, ranking = CASES[case]
    return sp.cate_gain_curve(DATA, cate=ranking, y=outcome, treat=treat, **kw)


def test_fixture_is_fklearn_output_with_a_ranking_that_matters():
    assert REF["versions"]["fklearn"] == "4.3.0"
    assert DATA["score"].is_unique and DATA["noise"].is_unique
    good = sum(REF["cases"]["dose_score"]["relative_gain"])
    noise = sum(REF["cases"]["dose_noise"]["relative_gain"])
    assert good > 10 * abs(noise)


@pytest.mark.parametrize("case", sorted(CASES))
def test_cumulative_effect_and_gain_match_fklearn(case):
    ref = REF["cases"][case]
    res = _fit(case, normalize=False)
    assert list(res.curve["n"]) == ref["n"]
    assert res.ate == pytest.approx(ref["ate"], rel=1e-12)
    np.testing.assert_allclose(
        res.curve["cumulative_effect"], ref["cumulative_effect"], **TOL
    )
    np.testing.assert_allclose(
        res.curve["cumulative_gain"], ref["cumulative_gain"], **TOL
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_normalized_gain_area_and_quantile_effects_match_fklearn(case):
    ref = REF["cases"][case]
    res = _fit(case)
    np.testing.assert_allclose(
        res.curve["cumulative_gain"], ref["relative_gain"], **TOL
    )
    assert res.auc == pytest.approx(sum(ref["relative_gain"]), rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(
        res.by_quantile["effect"], ref["effect_by_segment"], **TOL
    )
