"""Reference parity: sp.sensitivity_specificity / sp.power_case_control.

Both are deterministic closed-form functionals verified against their canonical
formulas (identical to epiR / standard epidemiology texts and Stata):

  * ``sensitivity_specificity`` : sensitivity = TP/(TP+FN), specificity =
    TN/(TN+FP), PPV, NPV and likelihood ratios are exact ratios of the 2x2
    diagnostic table.
  * ``power_case_control`` : case exposure prevalence from the odds ratio,
    ``p1 = OR*p0 / (1 + p0*(OR-1))``, then the unpooled-Wald two-proportion
    z-approximation power (1:1 allocation).

Closed-form identities (no external fixture needed); match is machine-precision.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import pytest

import statspai as sp

_TOL = 1e-12
_ZA = NormalDist().inv_cdf(0.975)


def _g(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def _power(res):
    if isinstance(res, dict):
        return res["power"]
    return getattr(res, "power_val", getattr(res, "power"))


def test_sensitivity_specificity_closed_form():
    tp, fn, fp, tn = 80, 15, 20, 85
    r = sp.sensitivity_specificity(tp=tp, fn=fn, fp=fp, tn=tn)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    assert _g(r, "sensitivity") == pytest.approx(sens, abs=_TOL)
    assert _g(r, "specificity") == pytest.approx(spec, abs=_TOL)
    assert _g(r, "ppv") == pytest.approx(tp / (tp + fp), abs=_TOL)
    assert _g(r, "npv") == pytest.approx(tn / (tn + fn), abs=_TOL)
    assert _g(r, "lr_pos") == pytest.approx(sens / (1 - spec), abs=_TOL)
    assert _g(r, "lr_neg") == pytest.approx((1 - sens) / spec, abs=_TOL)


@pytest.mark.parametrize(
    "nc,orr,p0",
    [(200, 2.0, 0.3), (150, 1.8, 0.25), (300, 2.5, 0.2)],
)
def test_power_case_control_closed_form(nc, orr, p0):
    p1 = orr * p0 / (1 + p0 * (orr - 1))
    s = math.sqrt(p1 * (1 - p1) + p0 * (1 - p0))
    shift = abs(p1 - p0) * math.sqrt(nc) / s
    # Two-sided: both rejection regions count (since 1.42).
    hand = NormalDist().cdf(shift - _ZA) + NormalDist().cdf(-shift - _ZA)
    got = _power(
        # The unpooled Wald closed form; the default test ("chi2") is pinned
        # to Stata power twoproportions in test_survival_epi_R_parity.py.
        sp.power_case_control(
            n_cases=nc, odds_ratio=orr, exposure_prevalence=p0, test="wald"
        )
    )
    assert got == pytest.approx(hand, abs=_TOL)
