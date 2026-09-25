"""The convenience event study on a raw CS fit carries the cohort-share term.

``sp.callaway_santanna(...).model_info['event_study']`` weights each event
time's ATT(g,t) by estimated cohort shares. Before 1.31 its standard errors
treated those shares as fixed (R ``did``'s ``wif`` term omitted), so they
were up to 47% too small on the castle-doctrine panel, while
``sp.aggte(type='dynamic')`` -- pinned to R ``did`` -- was right.
"""

import warnings

import numpy as np
import pytest

import statspai as sp


def _castle():
    df = sp.datasets.castle_doctrine()
    df["g0"] = df["effyear"].fillna(0)
    return df


CASES = [
    ("castle", dict(y="l_homicide", g="g0", t="year", i="sid")),
    ("castle_w", dict(y="l_homicide", g="g0", t="year", i="sid", weights="popwt")),
    (
        "castle_notyet",
        dict(y="l_homicide", g="g0", t="year", i="sid", control_group="notyettreated"),
    ),
    ("castle_x", dict(y="l_homicide", g="g0", t="year", i="sid", x=["northeast"])),
    ("mpdta", dict(y="lemp", g="first_treat", t="year", i="countyreal")),
    (
        "mpdta_rcs",
        dict(y="lemp", g="first_treat", t="year", i="countyreal", panel=False),
    ),
]


@pytest.mark.parametrize("name,kw", CASES, ids=[c[0] for c in CASES])
def test_convenience_event_study_equals_aggte_dynamic(name, kw):
    df = sp.datasets.mpdta() if name.startswith("mpdta") else _castle()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = sp.callaway_santanna(df, **kw)
        ag = sp.aggte(cs, type="dynamic", bstrap=False).detail
    es = cs.model_info["event_study"]
    mg = es.merge(ag, on="relative_time", suffixes=("", "_ag"))
    assert len(mg) == len(es)
    np.testing.assert_allclose(mg["att"], mg["att_ag"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(mg["se"], mg["se_ag"], rtol=1e-12)


def test_share_term_only_moves_mixed_cohort_event_times():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = sp.callaway_santanna(_castle(), y="l_homicide", g="g0", t="year", i="sid")
    es = cs.model_info["event_study"].set_index("relative_time")
    # k = -9 is identified by the 2005 cohort alone: no share term.
    # k = -8 mixes two cohorts; its SE was 0.0626 when shares were fixed.
    assert es.loc[-9, "se"] == pytest.approx(0.0571463, rel=1e-5)
    assert es.loc[-8, "se"] == pytest.approx(0.1188577, rel=1e-5)
