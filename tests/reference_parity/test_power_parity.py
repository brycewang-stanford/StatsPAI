"""Reference parity: sp power calculations vs base-R z-approximation formulas.

``sp.power_rct`` (two-sample pooled-sigma), ``sp.power_two_proportions``
(``test="wald"``, unpooled; its Pearson default since 1.43 is checked
against Stata in ``tests/test_r6_open_items.py``) and ``sp.power_logrank`` (Schoenfeld) are deterministic
closed-form large-sample power functions. The reference values are the
canonical z-approximation formulas computed in base R (the same formulas
standard texts and Stata ``power`` use). Match is machine-precision.

Frozen reference: ``_fixtures/power_R.json`` (base R 4.5.2). Regenerate::

    Rscript tests/reference_parity/_generate_power_R.R
"""

from __future__ import annotations

import json
import pathlib

import pytest

import statspai as sp

_FIXTURES = pathlib.Path(__file__).parent / "_fixtures"
_TOL = 1e-12


@pytest.fixture(scope="module")
def r_reference():
    with open(_FIXTURES / "power_R.json", encoding="utf-8") as fh:
        return json.load(fh)


def _power(res):
    if isinstance(res, dict):
        return res["power"]
    return getattr(res, "power_val", getattr(res, "power"))


def test_power_rct_matches_z_approx(r_reference):
    for case in r_reference["rct"]:
        got = _power(
            sp.power_rct(n=case["n"], effect_size=case["d"], alpha=0.05, sigma=1.0)
        )
        assert got == pytest.approx(case["power"], abs=_TOL)


def test_power_two_proportions_matches_z_approx(r_reference):
    for case in r_reference["two_prop"]:
        got = _power(
            # The fixture is the unpooled Wald formula; pass it explicitly
            # since the default became test="chi2" in 1.43.
            sp.power_two_proportions(
                n=case["n"], p1=case["p1"], p2=case["p2"], test="wald"
            )
        )
        assert got == pytest.approx(case["power"], abs=_TOL)


def test_power_logrank_matches_schoenfeld(r_reference):
    for case in r_reference["logrank"]:
        got = _power(
            sp.power_logrank(
                n=case["n"], hazard_ratio=case["hr"], prob_event=case["pe"]
            )
        )
        assert got == pytest.approx(case["power"], abs=_TOL)
