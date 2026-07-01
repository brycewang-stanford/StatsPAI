"""Reference parity: sp.evalue_rr vs the VanderWeele-Ding E-value closed form.

The E-value for a risk ratio is ``E = RR + sqrt(RR*(RR-1))`` (RR >= 1), and the
E-value for a confidence limit is the same transform applied to the limit
nearest the null (or 1 if the interval crosses it). This is the identical
closed form the R ``EValue`` package implements (``sp.evalue`` is already
bit-exact against it via Track A module 23); this suite pins the RR-input
variant to the same formula to machine precision.

References
----------
- VanderWeele, T.J. & Ding, P. (2017). Sensitivity Analysis in Observational
  Research: Introducing the E-Value. *Annals of Internal Medicine* 167(4).
"""

from __future__ import annotations

import math

import pytest

import statspai as sp


def _evalue(rr: float) -> float:
    rr = 1.0 / rr if rr < 1.0 else rr
    return rr + math.sqrt(rr * (rr - 1.0))


@pytest.mark.parametrize("rr", [1.5, 2.0, 3.0, 5.0])
def test_evalue_rr_point_matches_closed_form(rr):
    res = sp.evalue_rr(rr)
    assert res["evalue_estimate"] == pytest.approx(_evalue(rr), abs=1e-12)


def test_evalue_rr_ci_uses_lower_limit():
    res = sp.evalue_rr(2.0, rr_lower=1.3, rr_upper=3.1)
    assert res["evalue_ci"] == pytest.approx(_evalue(1.3), abs=1e-12)


def test_evalue_rr_ci_crossing_null_is_one():
    res = sp.evalue_rr(1.5, rr_lower=0.9, rr_upper=2.5)
    assert res["evalue_ci"] == pytest.approx(1.0, abs=1e-12)
