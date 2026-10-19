"""``sp.from_stata('gsbounds, ...')`` -> ``sp.group_sequential``.

The executed translation reproduces Stata 18's own boundaries (the fixture of
``tests/reference_parity/test_group_sequential_stata_parity.py``) for lines
written with Stata's abbreviations.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "reference_parity" / "_fixtures"
CASES = {
    c["cmd"].replace(" intpointsscale(60) infotol(1e-12)", ""): c
    for c in json.loads((_FIX / "gsbounds_Stata.json").read_text(encoding="utf-8"))
}


@pytest.mark.parametrize(
    "line,stata",
    [
        (
            "gsbounds, eff(errobf) fut(errpoc) nlooks(3) upper",
            "gsbounds, efficacy(errobfleming) futility(errpocock) nlooks(3) upper",
        ),
        (
            "gsbounds, eff(kd(3)) fut(hsd(-2), bind) nlooks(3) onesided",
            "gsbounds, efficacy(kdemets(3)) futility(hsdecani(-2), binding) "
            "nlooks(3) upper",
        ),
        (
            "gsbounds, efficacy(obf) nlooks(4) a(0.025) b(0.1)",
            "gsbounds, efficacy(obfleming) nlooks(4) alpha(0.025) power(0.9)",
        ),
        (
            "gsbounds, efficacy(wt(0.25)) information(1 3 4 10)",
            "gsbounds, efficacy(wtsiatis(0.25)) information(1 3 4 10)",
        ),
    ],
)
def test_translation_reproduces_gsbounds(line, stata):
    out = sp.from_stata(line)
    assert out["ok"] and out["tool"] == "group_sequential", out
    res = sp.group_sequential(**out["arguments"])
    case = CASES[stata]
    assert res.info_ratio == pytest.approx(float(case["info_ratio"]), rel=1e-6)
    cols = case["cols"].split()
    j = cols.index("Efficacy:Upper")
    ref = np.array([float(r[j]) for r in case["rows"]])
    np.testing.assert_allclose(res.boundaries["efficacy_upper"], ref, rtol=1e-6)


def test_display_and_tuning_options_are_reported_not_dropped():
    out = sp.from_stata("gsbounds, nlooks(3) nopvalues intpointsscale(40)")
    assert out["ok"]
    notes = " ".join(out["notes"])
    assert "nopvalues" in notes and "intpointsscale" in notes


def test_refusals_carry_guidance():
    out = sp.from_stata("gsdesign twomeans 50 60, sd(35)")
    assert not out["ok"] and "n_fixed" in out["error"]
    assert not sp.from_stata("gsbounds, efficacy(kdemets)")["ok"]
    assert not sp.from_stata("gsbounds, efficacy(obf) upper lower")["ok"]
