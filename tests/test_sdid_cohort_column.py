"""SDID entry points that infer the design from a cohort column must not
collapse several adoption periods onto the earliest one.

``sp.did(method="sdid")`` and ``sp.did_analysis(method="sdid")`` read each
unit's first treated period from ``treat``. They used to take the minimum
over treated units as the common adoption period, so a staggered panel was
estimated as a block design in which the later cohorts' pre-adoption
periods count as treated -- a different estimand, returned without warning
(1.54 against a true effect of 2 on the panel below).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

ADOPT = {0: 6, 1: 6, 2: 9, 3: 9}


def _panel(adopt):
    rng = np.random.default_rng(0)
    rows = []
    for u in range(30):
        a = rng.normal()
        for t in range(12):
            d = 1 if (u in adopt and t >= adopt[u]) else 0
            y = a + 0.1 * t + 2.0 * d + rng.normal(scale=0.3)
            rows.append((u, t, y, adopt.get(u, 0)))
    return pd.DataFrame(rows, columns=["unit", "time", "y", "g"])


@pytest.mark.parametrize("entry", ["did", "did_analysis"])
def test_several_adoption_periods_are_refused(entry):
    df = _panel(ADOPT)
    with pytest.raises(
        MethodIncompatibility, match=r"2 different periods \(6, 9\)"
    ) as exc:
        getattr(sp, entry)(df, y="y", treat="g", time="time", id="unit", method="sdid")
    assert exc.value.diagnostics["adoption_periods"] == [6, 9]
    assert "callaway_santanna" in exc.value.recovery_hint


def test_block_design_is_unchanged():
    # One cohort: same call, same number as before the guard existed.
    df = _panel({0: 6, 1: 6})
    res = sp.did(df, y="y", treat="g", time="time", id="unit", method="sdid", seed=0)
    direct = sp.sdid(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit=[0, 1],
        treatment_time=6,
        seed=0,
    )
    assert res.estimate == direct.estimate


def test_did_analysis_sdid_needs_id():
    df = _panel({0: 6})
    with pytest.raises(MethodIncompatibility, match="needs id="):
        sp.did_analysis(df, y="y", treat="g", time="time", method="sdid")
