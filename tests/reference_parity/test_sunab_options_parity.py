"""``sp.sun_abraham``: the last cohort as control, and the reference window rule.

Two options had no reference row:

* ``control_group='lastcohort'`` makes the latest treated cohort the
  control, next to the never-treated units (which stay in the regression
  with no dummies, as they do in ``eventstudyinteract`` and ``fixest``).
  On mpdta the last cohort is 2007, so ``eventstudyinteract ...,
  control_cohort(c2007)`` (``option_parity/83_sunab_control_cohort.do``)
  is the same estimator, and so is ``control_cohort=2007``, whose
  per-period effects ``test_sunab_control_cohort_parity.py`` already pins
  and whose summary aggregates are added here.
* ``window_rule='reference'`` leaves the relative times outside
  ``event_window`` in the omitted category next to e = -1. That is
  ``eventstudyinteract`` run with relative-time dummies inside the window
  only (``option_parity/93_sunab_options.do``), and ``fixest::sunab`` with
  ``ref.p = c(-1, <outside>)``. The rule biases every coefficient by the
  effects of the treated observations it puts in the reference, which is
  why it stopped being the default in 1.39.0; it is pinned here because it
  is still offered.

References: Stata ``eventstudyinteract`` 0.1 (Sun's implementation, the
canonical one) for the per-relative-time effects and their SEs with the
cohort-share term (``share_variance=True``); ``fixest`` 0.14
(``_fixtures/_generate_sunab_options_R.R``) for the same effects with the
shares held fixed (``share_variance=False``) and for both summary
aggregates.

Tolerances: relative 1e-9. Measured 2026-10-15: 1.9e-10 against
``eventstudyinteract``, 1e-11 or better against ``fixest``.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MPDTA = _ROOT / "tests" / "orig_parity" / "data" / "02_mpdta_original.csv"
_OPT = _ROOT / "tests" / "stata_parity" / "option_parity" / "results"
STATA = json.loads((_OPT / "93_sunab_options_Stata.json").read_text(encoding="utf-8"))
STATA_83 = json.loads(
    (_OPT / "83_sunab_control_cohort_Stata.json").read_text(encoding="utf-8")
)
FIXEST = json.loads(
    (pathlib.Path(__file__).parent / "_fixtures" / "sunab_options_R.json").read_text(
        encoding="utf-8"
    )
)
RTOL = 1e-9

#: fixture block -> sp.sun_abraham options
_CASES = {
    "lastcohort": dict(control_group="lastcohort"),
    # the same control named explicitly, as eventstudyinteract spells it
    "control_cohort_2007": dict(control_cohort=2007),
    "reference_m3_p2": dict(event_window=(-3, 2), window_rule="reference"),
    "reference_m2_p1": dict(event_window=(-2, 1), window_rule="reference"),
    "reference_m3_p2_lastcohort": dict(
        event_window=(-3, 2), window_rule="reference", control_group="lastcohort"
    ),
}
_STATA_BLOCK = {
    "lastcohort": (STATA_83, "control_cohort_2007"),
    "control_cohort_2007": (STATA_83, "control_cohort_2007"),
    "reference_m3_p2": (STATA, "window_m3_p2_never"),
    "reference_m2_p1": (STATA, "window_m2_p1_never"),
    "reference_m3_p2_lastcohort": (STATA, "window_m3_p2_c2007"),
}


def _fit(case, **extra):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.sun_abraham(
            pd.read_csv(_MPDTA),
            y="lemp",
            g="first_treat",
            t="year",
            i="countyreal",
            **{**_CASES[case], **extra},
        )


def _by_time(res):
    return {
        int(r.relative_time): (float(r.att), float(r.se))
        for r in res.detail.itertuples()
    }


def _stata_rows(case):
    payload, block = _STATA_BLOCK[case]
    out = {}
    for nm, v in payload[block].items():
        e = int(nm[3:]) * (-1 if nm[2] == "m" else 1)
        # eventstudyinteract returns an exact 0 / 0 where no cohort is
        # estimable once 2007 is the control (e = -4); StatsPAI omits the row.
        if v["att"] == 0.0 and v["se"] == 0.0:
            continue
        out[e] = (v["att"], v["se"])
    return out


@pytest.mark.parametrize("case", sorted(_CASES))
def test_event_time_effects_match_eventstudyinteract(case):
    got = _by_time(_fit(case))
    want = _stata_rows(case)
    assert set(got) == set(want), case
    for e, (att, se) in want.items():
        assert got[e][0] == pytest.approx(att, rel=RTOL), (case, e, "att")
        assert got[e][1] == pytest.approx(se, rel=RTOL), (case, e, "se")


#: fixture block of the fixest fit for each case
_FIXEST_BLOCK = {**{c: c for c in _CASES}, "control_cohort_2007": "lastcohort"}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_fixed_share_effects_match_fixest(case):
    got = _by_time(_fit(case, share_variance=False))
    want = {
        int(k): (v["att"], v["se"])
        for k, v in FIXEST[_FIXEST_BLOCK[case]]["event_time"].items()
    }
    assert set(got) == set(want), case
    for e, (att, se) in want.items():
        assert got[e][0] == pytest.approx(att, rel=RTOL), (case, e, "att")
        assert got[e][1] == pytest.approx(se, rel=RTOL), (case, e, "se")


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("share_variance", [True, False])
@pytest.mark.parametrize("aggregation", ["event_time", "fixest_att"])
def test_summary_aggregate_matches_fixest(case, share_variance, aggregation):
    """Both summaries hold the cohort shares fixed on either setting."""
    res = _fit(case, aggregation=aggregation, share_variance=share_variance)
    ref = FIXEST[_FIXEST_BLOCK[case]]
    assert float(res.estimate) == pytest.approx(ref[f"att_{aggregation}"], rel=RTOL)
    assert float(res.se) == pytest.approx(ref[f"se_{aggregation}"], rel=RTOL)


def test_the_options_change_the_answer():
    """Otherwise the comparisons above would not test the options."""
    default = _by_time(_fit("reference_m3_p2", window_rule="report"))
    reference = _by_time(_fit("reference_m3_p2"))
    assert abs(default[0][0] - reference[0][0]) > 1e-3
    never = _by_time(_fit("lastcohort", control_group="nevertreated"))
    last = _by_time(_fit("lastcohort"))
    assert abs(never[0][0] - last[0][0]) > 1e-3
