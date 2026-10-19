"""R ``did`` / ``DRDID`` argument names are accepted (2026-10-19 audit).

A call copied from R (``att_gt(yname=, tname=, idname=, gname=)``,
``DRDID::drdid(dname=, estMethod=)``) failed with an unknown-keyword
error. The R spellings are aliases: the fit is the same object as with the
StatsPAI names, so every assertion here is an identity.
"""

from __future__ import annotations

import numpy as np
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def castle():
    return sp.datasets.castle_doctrine()


def test_att_gt_names(castle):
    py = sp.callaway_santanna(
        castle, y="l_homicide", g="effyear", t="year", i="sid", estimator="reg"
    )
    r = sp.callaway_santanna(
        castle,
        yname="l_homicide",
        gname="effyear",
        tname="year",
        idname="sid",
        est_method="reg",
        alp=0.05,
    )
    assert r.estimate == py.estimate and r.se == py.se
    with pytest.raises(TypeError, match="both"):
        sp.callaway_santanna(
            castle, y="l_homicide", yname="l_homicide", g="effyear", t="year", i="sid"
        )


def test_aggte_names(castle):
    cs = sp.callaway_santanna(castle, y="l_homicide", g="effyear", t="year", i="sid")
    a = sp.aggte(cs, type="dynamic", alp=0.1)
    b = sp.aggte(cs, type="dynamic", alpha=0.1)
    assert a.estimate == b.estimate and tuple(a.ci) == tuple(b.ci)


def test_event_study_first_treat(castle):
    a = sp.event_study(
        castle, y="l_homicide", first_treat="effyear", time="year", unit="sid"
    )
    b = sp.event_study(
        castle, y="l_homicide", treat_time="effyear", time="year", unit="sid"
    )
    assert a.estimate == b.estimate and a.se == b.se


def test_drdid_names(castle):
    df = castle.query("year in (2005, 2006)")
    df = df[df.effyear.isna() | (df.effyear == 2006)].copy()
    df["post"] = (df.year == 2006).astype(int)
    df["d"] = df.effyear.notna().astype(int)
    py = sp.drdid(df, y="l_homicide", group="d", time="post", id="sid", method="trad")
    r = sp.drdid(
        df, yname="l_homicide", dname="d", tname="post", idname="sid", estMethod="trad"
    )
    assert r.estimate == py.estimate and r.se == py.se
    assert np.isfinite(r.estimate)
