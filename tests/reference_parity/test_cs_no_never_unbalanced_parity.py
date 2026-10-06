"""Reference parity: ``sp.callaway_santanna`` without never-treated units and
with ``control_group='notyettreated'`` on unbalanced panels.

Two silent errors found by the top-5 replications (QJE 2019 Princelings,
QJE 2023 AI-tocracy), both found against R ``did`` 2.3.0:

* **No never-treated units.** The late ATT(g, t) cells have no comparison
  units. StatsPAI returned them as ``att = 0, se = inf`` and averaged the
  zeros into every aggregate (simple ATT 0.55 against R's 0.92 on a design
  with true effect 1). R ``pre_process_did`` keeps the periods before the
  last cohort's treatment date minus anticipation and uses that cohort as the
  not-yet-treated comparison only; StatsPAI now does the same, and any cell
  still without comparison units is dropped with a warning.
* **Unbalanced panels.** ``allow_unbalanced_panel=True`` with
  ``estimator='reg'`` and no covariates went through a cell-mean path whose
  comparison arm was hard-wired to the never-treated, so
  ``control_group='notyettreated'`` returned the never-treated numbers.

Fixture: ``_fixtures/_generate_cs_no_never_unbalanced_{data.py,R.R}`` --
``att_gt`` cells and ``aggte(type='simple', na.rm=TRUE)`` (``sp.aggte``) for panel and
repeated cross-section data, both base periods, anticipation 0/1, ``reg`` and
``dr``; and the unbalanced panel under both control groups. Tolerance
``rtol = 1e-8`` on every cell's ATT and SE, on the aggregate ATT and on the
aggregate SE.

The fixture is generated with did 2.5.1. Against did 2.3.0 the aggregate SE
differed off the balanced panel (under 0.1% on the repeated cross-section,
0.9% / 2.2% on the unbalanced panel) while every cell agreed, and this test
held it to 1% / 3%. The difference was in the reference. did 2.5.0 records
that ``aggte()`` added the estimated-weight influence term in id-sorted order
to an influence function stored in first-appearance order on unbalanced
panels. With that fixed, did 2.5.1 and StatsPAI agree to 1e-15 on all twenty
cases.
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
_JSON = _DIR / "cs_no_never_unbalanced_R.json"
_DATA = {
    "no_never": _DIR / "cs_no_never_data.csv",
    "unbalanced": _DIR / "cs_unbalanced_notyet_data.csv",
}


def _cases():
    return json.loads(_JSON.read_text(encoding="utf-8"))["cases"]


def _id(c):
    return (
        f"{c['data']}-{'panel' if c['panel'] else 'rcs'}-{c['base_period']}"
        f"-ant{c['anticipation']}-{c['est_method']}-{c['control_group']}"
    )


def _fit(c):
    df = pd.read_csv(_DATA[c["data"]])
    kw = dict(
        y="y",
        g="g",
        t="t",
        i="id",
        estimator=c["est_method"],
        control_group=c["control_group"],
        base_period=c["base_period"],
        anticipation=c["anticipation"],
        panel=c["panel"],
    )
    if c["data"] == "unbalanced":
        kw["allow_unbalanced_panel"] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.callaway_santanna(df, **kw)


@pytest.mark.parametrize("case", _cases(), ids=_id)
def test_cells_and_simple_match_r(case):
    r = _fit(case)
    agg = sp.aggte(r, type="simple", bstrap=False)
    assert agg.estimate == pytest.approx(case["simple_att"], rel=1e-8)
    balanced_panel = case["panel"] and case["data"] == "no_never"
    assert agg.se == pytest.approx(case["simple_se"], rel=1e-8)
    if balanced_panel:
        # the balanced-panel headline is the same aggregate
        assert r.estimate == pytest.approx(case["simple_att"], rel=1e-8)
        assert r.se == pytest.approx(case["simple_se"], rel=1e-8)
    ref = pd.DataFrame({k: case[k] for k in ("group", "time", "att", "se")}).set_index(
        ["group", "time"]
    )
    got = r.detail.set_index(["group", "time"])[["att", "se"]]
    # R reports the base-period reference cells (se = NA) that StatsPAI
    # leaves out; everything StatsPAI estimates must be in R's table.
    joined = got.join(ref, rsuffix="_r", how="left")
    assert joined["att_r"].notna().all()
    np.testing.assert_allclose(joined["att"], joined["att_r"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(joined["se"], joined["se_r"], rtol=1e-8)


def test_no_never_warns_and_records_comparison_cohort():
    df = pd.read_csv(_DATA["no_never"])
    with pytest.warns(UserWarning, match="last treated cohort"):
        r = sp.callaway_santanna(
            df, y="y", g="g", t="t", i="id", control_group="notyettreated"
        )
    assert r.model_info["comparison_only_cohorts"] == [11.0]
    assert 11 not in set(r.detail["group"])
    assert r.detail["time"].max() == 10
    assert np.isfinite(r.detail["se"]).all()


def test_unbalanced_control_groups_differ():
    df = pd.read_csv(_DATA["unbalanced"])
    kw = dict(y="y", g="g", t="t", i="id", estimator="reg", allow_unbalanced_panel=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nev = sp.callaway_santanna(df, control_group="nevertreated", **kw).estimate
        nyt = sp.callaway_santanna(df, control_group="notyettreated", **kw).estimate
    assert abs(nev - nyt) > 1e-3


def test_cells_without_comparison_units_are_dropped():
    """Any cell that still has no comparison units is left out of the
    aggregates rather than counted as a zero effect."""
    from statspai.did.callaway_santanna import _drop_unidentified_cells

    detail = pd.DataFrame(
        {
            "group": [2, 2, 3],
            "time": [2, 3, 3],
            "att": [1.0, 0.0, 2.0],
            "se": [0.1, np.inf, 0.2],
        }
    )
    with pytest.warns(UserWarning, match=r"\(2, 3\)"):
        kept, infs, dropped = _drop_unidentified_cells(
            detail, [np.ones(3), np.zeros(3), np.ones(3)]
        )
    assert dropped == [(2, 3)]
    assert list(kept["att"]) == [1.0, 2.0]
    assert len(infs) == 2
