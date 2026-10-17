"""sp.honest_did warns on a varying-base event study.

With base_period='varying' each pre-treatment coefficient of
callaway_santanna is a one-period change, not the pre-trend path the
Rambachan-Roth restrictions bound. honest_did used to return intervals for
it without a word (on mpdta every M up to 0.0224 rejected zero).
"""

import warnings

import pytest

import statspai as sp


@pytest.fixture(scope="module")
def fits():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = sp.datasets.mpdta()
        out = {
            bp: sp.callaway_santanna(
                m, y="lemp", g="first_treat", t="year", i="countyreal", base_period=bp
            )
            for bp in ("universal", "varying")
        }
    return out


def _varying_warnings(result):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = sp.honest_did(result, e=0)
    return out, [w for w in caught if "base_period='varying'" in str(w.message)]


def test_aggte_carries_the_base_period(fits):
    for bp, cs in fits.items():
        assert sp.aggte(cs, type="dynamic").model_info["base_period"] == bp


@pytest.mark.parametrize("via_aggte", [False, True])
def test_varying_base_warns_and_is_recorded(fits, via_aggte):
    cs = fits["varying"]
    res = sp.aggte(cs, type="dynamic") if via_aggte else cs
    out, hits = _varying_warnings(res)
    assert len(hits) == 1
    assert out.attrs["base_period"] == "varying"


def test_universal_base_is_silent(fits):
    out, hits = _varying_warnings(sp.aggte(fits["universal"], type="dynamic"))
    assert hits == []
    assert "base_period" not in out.attrs
