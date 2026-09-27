"""SDID entry points that read the design from a cohort column.

``sp.did(method="sdid")`` and ``sp.did_analysis(method="sdid")`` take each
unit's first treated period from ``treat``. Through 1.31 they used the
earliest period for every treated unit, estimating a staggered panel as a
block design in which later cohorts' pre-adoption periods count as treated
(1.54 against a true effect of 2 on the panel below). Several adoption
periods now go through ``sp.sdid(treat=...)``: cohort-by-cohort fits
aggregated as Stata ``sdid`` does (pinned in
``tests/reference_parity/test_sdid_staggered_parity.py``).
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
            rows.append((u, t, y, adopt.get(u, 0), d))
    return pd.DataFrame(rows, columns=["unit", "time", "y", "g", "w"])


@pytest.mark.parametrize("entry", ["did", "did_analysis"])
def test_staggered_cohorts_are_estimated_by_cohort(entry):
    df = _panel(ADOPT)
    kw = dict(y="y", treat="g", time="time", id="unit", method="sdid")
    if entry == "did":
        res = sp.did(df, se_method="placebo", seed=0, **kw)
    else:
        res = sp.did_analysis(df, se_method="placebo", seed=0, **kw).main_result
    ref = sp.sdid(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treat="w",
        se_method="placebo",
        seed=0,
    )
    assert res.estimate == ref.estimate
    assert res.model_info["design"] == "staggered"
    assert list(res.model_info["tau_by_cohort"]["adoption"]) == [6, 9]
    # the collapsed-onto-earliest estimate was 1.54; truth is 2
    assert abs(res.estimate - 2.0) < 0.15


def test_block_design_is_unchanged():
    # One cohort: same call path, same number as before.
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
    assert res.se == direct.se


def test_did_analysis_sdid_needs_id():
    df = _panel({0: 6})
    with pytest.raises(MethodIncompatibility, match="needs id="):
        sp.did_analysis(df, y="y", treat="g", time="time", method="sdid")
