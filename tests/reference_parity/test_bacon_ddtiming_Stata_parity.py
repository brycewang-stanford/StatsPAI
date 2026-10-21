"""``sp.bacon_decomposition(balance='observed_rows')`` vs Stata ``ddtiming``.

Fixture: ``_fixtures/_generate_bacon_ddtiming_Stata.do`` (Stata 18,
``ddtiming`` 0.2, ``reghdfe`` 6.13.1) on the three unbalanced panels of
``_fixtures/bacon_ddtiming_data.csv``: an already-treated unit first observed
in a later period, a unit missing its first treated period, 2x2 comparisons
left without treatment variation by the missing rows, negative weights.

Tolerances: ddtiming's ``e()`` scalars and the TWFE coefficient to 1e-12
(observed 1.4e-15). The 2x2 rows come from ``savedata()``, which writes
8 significant digits: 1e-7 relative / 1e-8 absolute.

ddtiming reports an unidentified 2x2 as 0 and keeps its weight; ours is NaN
with ``identified=False``. Its type average is therefore
``sum(identified weight x estimate) / (total type weight)``, which the test
rebuilds from ours.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "bacon_ddtiming_Stata.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(_FIX / "bacon_ddtiming_data.csv")

_CODE = {
    "Earlier vs Later Treated": ("e", 1),
    "Later vs Earlier Treated": ("l", 2),
    "Treated vs Untreated": ("u", 3),
    "Later vs Always Treated": ("a", 4),
}


def _fit(panel: int):
    df = DATA[DATA["panel"] == panel].drop(columns="panel")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.bacon_decomposition(
            df, y="y", treat="d", time="year", id="unit", balance="observed_rows"
        )


@pytest.mark.parametrize("panel", [0, 1, 2])
def test_scalars_match_ddtiming(panel):
    r = _fit(panel)
    ref = REF[str(panel)]
    assert r["weighted_sum"] == pytest.approx(ref["scalars"]["dd"], abs=1e-12)
    assert r["beta_twfe"] == pytest.approx(ref["twfe"], abs=1e-12)
    dec = r["decomposition"]
    seen = set()
    for kind, (code, _) in _CODE.items():
        part = dec[dec["type"] == kind]
        if part.empty:
            continue
        seen.add(code)
        total = float(part["weight"].sum())
        known = part[part["identified"]]
        ddtiming_avg = float((known["weight"] * known["estimate"]).sum()) / total
        assert total == pytest.approx(ref["scalars"][f"wt_sum_{code}"], abs=1e-12)
        assert ddtiming_avg == pytest.approx(
            ref["scalars"][f"dd_avg_{code}"], abs=1e-12
        )
    assert seen == {k[len("wt_sum_") :] for k in ref["scalars"] if "wt_sum" in k}


@pytest.mark.parametrize("panel", [0, 1, 2])
def test_rows_match_ddtiming(panel):
    dec = _fit(panel)["decomposition"].copy()
    dec["code"] = dec["type"].map(lambda k: _CODE[k][1])
    rows = REF[str(panel)]["rows"]
    assert len(rows) == len(dec)
    used = set()
    for code, lo, hi, est, weight in rows:
        cand = dec[
            (dec["code"] == code)
            & (dec["time_lower"] == lo)
            & (dec["time_upper"] == hi)
            & ~dec.index.isin(used)
        ]
        if est == 0.0:  # ddtiming's unidentified 2x2
            cand = cand[~cand["identified"]]
            j = (cand["weight"] - weight).abs().idxmin()
        else:
            cand = cand[cand["identified"]]
            j = (cand["estimate"] - est).abs().idxmin()
            assert dec.at[j, "estimate"] == pytest.approx(est, rel=1e-7, abs=1e-8)
        assert dec.at[j, "weight"] == pytest.approx(weight, rel=1e-7, abs=1e-8)
        used.add(j)
    assert len(used) == len(dec)


def test_fixture_exercises_the_unbalanced_cases():
    # the panels are only a test of the rule if they contain what it is about
    fits = [_fit(p) for p in range(3)]
    assert all((~f["decomposition"]["identified"]).any() for f in fits)
    assert any((f["decomposition"]["weight"] < 0).any() for f in fits)
    gaps = [abs(f["weighted_sum"] - f["beta_twfe"]) for f in fits]
    assert min(gaps) > 1e-3
