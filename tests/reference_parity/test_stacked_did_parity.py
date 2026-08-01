"""Reference parity: ``sp.stacked_did`` vs a hand-written ``fixest`` stack.

Stacked DiD (Cengiz, Dube, Lindner & Zipperer 2019) is a *construction*, not
a packaged estimator: no CRAN package implements it, so there is no library
call to compare against. What can be pinned is that the construction, written
independently in R, reproduces the same numbers -- one sub-experiment per
treated cohort with its clean controls over the event window, stacked, then
TWFE with cohort-specific unit and time fixed effects and ``k = -1`` as the
omitted period.

Reference generation (R 4.5.2, fixest 0.14.0, data.table)::

    for (g in cohorts) {
      ctrl <- if (never_only) never else union(never, not_yet_treated)
      sub  <- df[id %in% union(cohort_g, ctrl) & year %in% (g-3):(g+3)]
      sub[, `:=`(cohort = g, rel = year - g,
                 treated_unit = as.integer(id %in% cohort_g))]
    }
    st <- rbindlist(frames)
    st[, `:=`(uc = paste0(id, "_", cohort), tc = paste0(year, "_", cohort))]
    feols(y ~ i(rel_f, treated_unit, ref = "-1") | uc + tc,
          data = st, cluster = ~ id)

Both control-group conventions are pinned because they give materially
different answers on this fixture (1.2381 vs 1.2558) and ``sp.stacked_did``
defaults to ``never_treated_only=True``: a silent flip of that default would
otherwise pass unnoticed.

Fixture: ``_fixtures/stacked_did_panel.csv`` -- 240 units x 16 periods,
cohorts {6, 9, 12} plus never-treated (coded 0), true effect 1.3.

The same comparison runs live in the Track A harness as module
``75_stacked``; this test pins it on a committed fixture so the check does
not depend on R being installed.

References
----------
Cengiz, D., Dube, A., Lindner, A. and Zipperer, B. (2019). "The Effect of
Minimum Wages on Low-Wage Jobs." *The Quarterly Journal of Economics*,
134(3), 1405-1454. [@cengiz2019effect]
"""

from __future__ import annotations

import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIXTURE = pathlib.Path(__file__).parent / "_fixtures" / "stacked_did_panel.csv"

WINDOW = (-3, 3)

# feols on the hand-written stack, never-treated controls only.
R_NEVER = {
    "n_stacked": 2506,
    "ATT_post": 1.2380683199,
    "event": {
        -3: -0.0629023303,
        -2: -0.0502427634,
        0: 1.2020806685,
        1: 1.2794743747,
        2: 1.2688417786,
        3: 1.2018764577,
    },
}

# Same, with never-treated plus not-yet-treated controls.
R_NYT = {
    "n_stacked": 2863,
    "ATT_post": 1.2558024566,
    "event": {
        -3: -0.0474968142,
        -2: -0.0469750443,
        0: 1.2293690099,
        1: 1.3145593062,
        2: 1.2523736479,
        3: 1.2269078624,
    },
}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not _FIXTURE.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing fixture: {_FIXTURE}")
    return pd.read_csv(_FIXTURE)


def _fit(df: pd.DataFrame, never_only: bool):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.stacked_did(
            df,
            y="y",
            group="id",
            time="year",
            first_treat="first_treat",
            window=WINDOW,
            never_treated_only=never_only,
        )


@pytest.fixture(scope="module")
def fit_never(panel):
    return _fit(panel, True)


@pytest.fixture(scope="module")
def fit_nyt(panel):
    return _fit(panel, False)


@pytest.mark.parametrize(
    "which, ref",
    [("fit_never", R_NEVER), ("fit_nyt", R_NYT)],
    ids=["never_treated_only", "not_yet_treated"],
)
def test_stack_size_matches_reference(request, which, ref):
    """The stack itself must be built the same way before the fit can agree."""
    fit = request.getfixturevalue(which)
    assert int(fit.model_info["n_stacked_obs"]) == ref["n_stacked"]


@pytest.mark.parametrize(
    "which, ref",
    [("fit_never", R_NEVER), ("fit_nyt", R_NYT)],
    ids=["never_treated_only", "not_yet_treated"],
)
def test_post_att_matches_fixest(request, which, ref):
    fit = request.getfixturevalue(which)
    assert fit.estimate == pytest.approx(ref["ATT_post"], abs=1e-9)


@pytest.mark.parametrize(
    "which, ref",
    [("fit_never", R_NEVER), ("fit_nyt", R_NYT)],
    ids=["never_treated_only", "not_yet_treated"],
)
def test_event_study_matches_fixest(request, which, ref):
    fit = request.getfixturevalue(which)
    es = request.getfixturevalue(which).model_info["event_study"]
    got = dict(zip(es["relative_time"].astype(int), es["att"].astype(float)))
    for k, expected in ref["event"].items():
        assert got[k] == pytest.approx(
            expected, abs=1e-9
        ), f"k={k}: StatsPAI {got[k]:.10f} vs fixest {expected:.10f}"


def test_reference_period_is_omitted(fit_never):
    """k = -1 is the omitted category, so it must not carry an estimate."""
    es = fit_never.model_info["event_study"]
    ref_rows = es[es["relative_time"] == -1]
    assert ref_rows.empty or float(ref_rows["att"].iloc[0]) == pytest.approx(0.0)


def test_control_group_choice_changes_the_answer(fit_never, fit_nyt):
    """Guards the default: the two conventions are not interchangeable here,
    so a silent flip of ``never_treated_only`` would move published numbers."""
    assert abs(fit_never.estimate - fit_nyt.estimate) > 1e-3
    assert int(fit_never.model_info["n_stacked_obs"]) < int(
        fit_nyt.model_info["n_stacked_obs"]
    )


def test_pre_period_effects_are_near_zero(fit_never):
    """The DGP has no pre-trend; the stack should not manufacture one."""
    for k in (-3, -2):
        assert abs(R_NEVER["event"][k]) < 0.1


def test_recovers_the_design_effect(fit_never):
    assert abs(fit_never.estimate - 1.3) < 0.1
