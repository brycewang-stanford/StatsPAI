"""``sp.recommend`` on a panel passed without ``id=`` / ``time=``.

2026-10-19 audit: the panel was profiled as a cross-section, the design
came out "observational", and the unit, period and cohort columns were
offered as covariates (``sp.match(..., covariates=['unit', 'time',
'first_treat', 'group'])``).
"""

from __future__ import annotations

import ast
import warnings

import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility
from statspai.smart.recommend import _structural_columns

ROLE_COLUMNS = {"unit", "time", "first_treat", "group"}


@pytest.fixture(scope="module")
def panel():
    return sp.dgp_did(n_units=60, n_periods=8, staggered=True, seed=1)


def _covariate_lists(rec):
    for r in rec.recommendations:
        params = r.get("params") or {}
        for key in ("covariates", "x", "controls"):
            if isinstance(params.get(key), list):
                yield params[key]


def test_panel_without_ids_is_recommended_as_a_panel(panel):
    rec = sp.recommend(panel, y="y", treatment="treated")
    explicit = sp.recommend(panel, y="y", treatment="treated", id="unit", time="time")
    assert rec.design == explicit.design == "did"
    assert [r["function"] for r in rec.recommendations] == [
        r["function"] for r in explicit.recommendations
    ]
    assert any("sp.detect_design" in w for w in rec.warnings)


def test_role_columns_are_never_default_covariates(panel):
    for kwargs in ({}, {"id": "unit", "time": "time"}, {"design": "observational"}):
        rec = sp.recommend(panel, y="y", treatment="treated", **kwargs)
        for covs in _covariate_lists(rec):
            assert not (set(covs) & ROLE_COLUMNS), (kwargs, covs)


def test_cohort_columns_are_recognised_by_shape(panel):
    # first_treat is the adoption period, group a relabelling of it; both
    # are functions of when the unit is first treated.
    assert set(_structural_columns(panel, "treated", "unit", "time")) == {
        "first_treat",
        "group",
    }
    # A time-varying regressor is a covariate.
    noisy = panel.assign(x=range(len(panel)))
    assert "x" not in _structural_columns(noisy, "treated", "unit", "time")


def test_two_equally_good_layouts_are_refused(panel):
    twin = panel.assign(unit2=panel["unit"] + 1000)
    with pytest.raises(MethodIncompatibility, match="more than one"):
        sp.recommend(twin, y="y", treatment="treated")
    rec = sp.recommend(twin, y="y", treatment="treated", id="unit", time="time")
    assert rec.design == "did"


def test_cross_section_is_unchanged():
    df = sp.cps_wage()
    rec = sp.recommend(df, y="log_wage", treatment="union")
    assert rec.design == "observational"
    assert not any("detect_design" in w for w in rec.warnings)


def test_ols_code_is_a_valid_formula():
    df = sp.cps_wage()
    rec = sp.recommend(
        df, y="log_wage", treatment="union", covariates=["education", "experience"]
    )
    code = next(r["code"] for r in rec.recommendations if r["function"] == "regress")
    call = ast.parse(code).body[0].value
    formula = call.args[0].value
    assert formula == "log_wage ~ union + education + experience"
    fit = eval(code, {"sp": sp, "df": df})
    assert "education" in fit.params.index


def test_group_indicator_with_treat_time_gives_the_adoption_cohort():
    """A time-invariant ``group`` indicator plus ``treat_time``: the treated
    units adopt at treat_time. Read as a switch, the indicator's first
    "treated" period is the first observed one, 0, which is the
    never-treated code, so every unit was coded never treated and the fit
    raised (the recommend benchmark's repeated-cross-section case, 10-19 to
    10-25)."""
    df = sp.dgp_did(n_units=60, n_periods=4, staggered=False, seed=41)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = sp.recommend(
            df, y="y", treatment="group", time="time", treat_time=2, design="did"
        )
        card = rec.recommendations[0]
        assert card["function"] == "callaway_santanna"
        cohorts = card["params"]["data"]["_cohort_group"]
        assert set(cohorts.unique()) == {0, 2}
        # the derived cohort marks exactly the units with group == 1
        by_unit = card["params"]["data"].groupby("unit")
        assert (
            (by_unit["_cohort_group"].first() == 2) == (by_unit["group"].first() == 1)
        ).all()
        assert "= 2 for units with group == 1" in card["code"]
        res = rec.run(0)
    assert res.estimate == res.estimate  # finite
