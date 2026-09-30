"""``sp.stacked_did(events=)``: repeated (non-absorbing) events per unit.

A state raises its minimum wage several times (Minimum Wages, QJE 2019);
``first_treat`` cannot say that. With ``events=`` (a 0/1 column marking
event periods) every event becomes a sub-experiment over ``window``; its
controls are the units with no event inside that window (the clean-control
rule), and an event whose own unit has another event inside the window is
dropped by default.

Reference: the same stack built independently in Stata
(``_generate_stacked_events_Stata.do``: a loop over events, ``reghdfe`` with
unit x event and period x event effects, clustered by unit), unweighted and
``[aw=w]``. Stack size, dropped events, coefficients and SEs agree to 1e-12.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
STATA = json.loads((_FIX / "stacked_events_Stata.json").read_text(encoding="utf-8"))
K = {-3: "m3", -2: "m2", 0: "p0", 1: "p1", 2: "p2", 3: "p3", 4: "p4"}


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "stacked_events.csv")


@pytest.mark.parametrize("weights, prefix", [(None, ""), ("w", "w_")])
def test_matches_stata_event_stack(data, weights, prefix):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.stacked_did(
            data,
            y="y",
            group="unit",
            time="t",
            events="event",
            window=(-3, 4),
            weights=weights,
        )
    mi = r.model_info
    assert mi["n_stacked_obs"] == STATA["N"]
    assert mi["n_events_dropped_overlap"] == STATA["dropped"]
    assert mi["n_cohorts"] == STATA["kept"]
    es = mi["event_study"].set_index("relative_time")
    for k, nm in K.items():
        b, se = STATA[prefix + nm]
        assert es.loc[k, "att"] == pytest.approx(b, rel=1e-10, abs=1e-12)
        assert es.loc[k, "se"] == pytest.approx(se, rel=1e-10)


def test_keeping_overlapping_events_and_validation(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        keep = sp.stacked_did(
            data,
            y="y",
            group="unit",
            time="t",
            events="event",
            window=(-3, 4),
            own_overlap="keep",
        )
    assert keep.model_info["n_cohorts"] == STATA["kept"] + STATA["dropped"]
    with pytest.raises(sp.MethodIncompatibility, match="events"):
        sp.stacked_did(
            data,
            y="y",
            group="unit",
            time="t",
            events="event",
            first_treat="event",
            window=(-3, 4),
        )
    bad = data.assign(event=data.event * 2)
    with pytest.raises(sp.MethodIncompatibility, match="0/1"):
        sp.stacked_did(bad, y="y", group="unit", time="t", events="event")
