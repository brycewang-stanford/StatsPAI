"""Cross-language parity: ``sp.evalue_rr`` vs R ``EValue::evalues.RR``.

Regenerate with ``Rscript tests/reference_parity/_fixtures/_generate_evalue_rr_R.R``.

Before 1.28.0 this file compared ``sp.evalue_rr`` against a closed form
typed into the test and stated that R's ``EValue`` "implements the
identical closed form". The parity index graded that as cross-language
evidence, but nothing here ever consulted R. The fixture below is
``evalues.RR``'s own output, including the branches that choose which
confidence limit is used and the rule that a CI crossing the null has an
E-value of exactly 1. The closed form is kept as a second, independent
check.
"""

from __future__ import annotations

import json
import math
import pathlib

import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
_CASES = json.loads((_FIX / "evalue_rr_R.json").read_text(encoding="utf-8"))["cases"]


def _evalue(rr: float) -> float:
    rr = 1.0 / rr if rr < 1.0 else rr
    return rr + math.sqrt(rr * (rr - 1.0))


@pytest.mark.parametrize(
    "case", _CASES, ids=lambda c: f"rr{c['rr']}_ci{c['lo']}-{c['hi']}"
)
def test_evalue_rr_matches_EValue(case):
    if case["lo"] is None:
        res = sp.evalue_rr(case["rr"])
    else:
        res = sp.evalue_rr(case["rr"], rr_lower=case["lo"], rr_upper=case["hi"])
        assert res["evalue_ci"] == pytest.approx(case["e_ci"], rel=1e-12)
    assert res["evalue_estimate"] == pytest.approx(case["e_point"], rel=1e-12)


@pytest.mark.parametrize("rr", [1.5, 2.0, 3.0, 5.0, 0.6])
def test_evalue_rr_point_matches_closed_form(rr):
    assert sp.evalue_rr(rr)["evalue_estimate"] == pytest.approx(_evalue(rr), abs=1e-12)


def test_evalue_rr_ci_crossing_the_null_is_one():
    assert sp.evalue_rr(1.5, rr_lower=0.8, rr_upper=2.4)["evalue_ci"] == 1.0
