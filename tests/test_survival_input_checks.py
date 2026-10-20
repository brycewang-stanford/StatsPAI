"""Survival inputs that used to pass silently (2026-10-20 probe).

* negative durations were accepted by kaplan_meier / cox / rmst;
* an event column coded 0/1/2 had every non-zero value read as an event
  (Stata's stset rule, the opposite of R's 1/2 Surv coding), so the code is
  now refused rather than guessed;
* rows with a missing weight were dropped without a word.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


@pytest.fixture(scope="module")
def surv():
    rng = np.random.default_rng(0)
    n = 300
    a = rng.integers(0, 2, n)
    t = rng.exponential(np.where(a == 1, 12.0, 8.0))
    c = rng.exponential(30.0, n)
    return pd.DataFrame(
        {
            "a": a,
            "x": rng.normal(size=n),
            "time": np.minimum(t, c),
            "event": (t <= c).astype(int),
            "w": rng.uniform(0.5, 2, n),
        }
    )


CALLS = {
    "kaplan_meier": lambda d, **k: sp.kaplan_meier(
        d, duration="time", event="event", **k
    ),
    "cox": lambda d, **k: sp.cox(
        data=d, duration="time", event="event", x=["a", "x"], **k
    ),
    "rmst": lambda d, **k: sp.rmst(d, "time", "event", group="a", tau=10, **k),
}


@pytest.mark.parametrize("fn", sorted(CALLS))
def test_negative_duration_is_refused(surv, fn):
    bad = surv.copy()
    bad.loc[0, "time"] = -1.0
    with pytest.raises(DataInsufficient, match="negative"):
        CALLS[fn](bad)


@pytest.mark.parametrize("fn", sorted(CALLS))
def test_event_codes_other_than_0_1_are_refused(surv, fn):
    bad = surv.copy()
    bad.loc[bad.index[:5], "event"] = 2
    with pytest.raises(MethodIncompatibility, match="0/1"):
        CALLS[fn](bad)


@pytest.mark.parametrize("fn", sorted(CALLS))
def test_missing_weight_rows_are_dropped_with_a_warning(surv, fn):
    holed = surv.copy()
    holed.loc[holed.index[:4], "w"] = np.nan
    with pytest.warns(UserWarning, match="missing weight"):
        CALLS[fn](holed, weights="w")


def test_boolean_event_is_accepted(surv):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.kaplan_meier(surv, duration="time", event="event")
        b = sp.kaplan_meier(
            surv.assign(event=surv["event"].astype(bool)),
            duration="time",
            event="event",
        )
    pd.testing.assert_frame_equal(a.survival_table, b.survival_table)
