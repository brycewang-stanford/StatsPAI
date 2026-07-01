"""Reference parity: epidemiological 2x2 measures vs base-R closed form.

``sp.odds_ratio`` (Woolf logit), ``sp.relative_risk`` (Katz-log),
``sp.risk_difference`` (Wald) and ``sp.mantel_haenszel`` (Robins-Breslow-
Greenland) are deterministic closed-form estimators on 2x2 cell counts. The
reference values are the canonical epidemiological formulas computed in base R
(identical to ``epiR::epi.2by2`` and Stata ``epitab``); the match is exact.

Frozen reference: ``_fixtures/epi_R.json`` (base R 4.5.2). Regenerate with::

    Rscript tests/reference_parity/_generate_epi_R.R

References
----------
- Woolf, B. (1955); Katz et al. (1978); Robins, Breslow & Greenland (1986).
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
    with open(_FIXTURES / "epi_R.json", encoding="utf-8") as fh:
        return json.load(fh)


def _g(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def test_odds_ratio_matches_R(r_reference):
    res = sp.odds_ratio(30, 70, 20, 80)
    ref = r_reference["OR"]
    assert _g(res, "estimate") == pytest.approx(ref["estimate"], abs=_TOL)
    assert _g(res, "se_log") == pytest.approx(ref["se_log"], abs=_TOL)
    lo, hi = _g(res, "ci")
    assert lo == pytest.approx(ref["ci_lo"], abs=_TOL)
    assert hi == pytest.approx(ref["ci_hi"], abs=_TOL)


def test_relative_risk_matches_R(r_reference):
    res = sp.relative_risk(30, 70, 20, 80)
    ref = r_reference["RR"]
    assert _g(res, "estimate") == pytest.approx(ref["estimate"], abs=_TOL)
    assert _g(res, "se_log") == pytest.approx(ref["se_log"], abs=_TOL)
    lo, hi = _g(res, "ci")
    assert lo == pytest.approx(ref["ci_lo"], abs=_TOL)
    assert hi == pytest.approx(ref["ci_hi"], abs=_TOL)


def test_risk_difference_matches_R(r_reference):
    res = sp.risk_difference(30, 70, 20, 80)
    ref = r_reference["RD"]
    assert _g(res, "estimate") == pytest.approx(ref["estimate"], abs=_TOL)
    assert _g(res, "se") == pytest.approx(ref["se"], abs=_TOL)
    lo, hi = _g(res, "ci")
    assert lo == pytest.approx(ref["ci_lo"], abs=_TOL)
    assert hi == pytest.approx(ref["ci_hi"], abs=_TOL)


def test_mantel_haenszel_matches_R(r_reference):
    res = sp.mantel_haenszel([[[30, 70], [20, 80]], [[40, 60], [25, 75]]], measure="OR")
    ref = r_reference["MH"]
    assert _g(res, "estimate") == pytest.approx(ref["estimate"], abs=_TOL)
    assert _g(res, "se_log") == pytest.approx(ref["se_log"], abs=_TOL)
    lo, hi = _g(res, "ci")
    assert lo == pytest.approx(ref["ci_lo"], abs=_TOL)
    assert hi == pytest.approx(ref["ci_hi"], abs=_TOL)


def test_prevalence_ratio_matches_R(r_reference):
    res = sp.prevalence_ratio(30, 70, 20, 80)
    ref = r_reference["PR"]
    assert _g(res, "estimate") == pytest.approx(ref["estimate"], abs=_TOL)
    assert _g(res, "se_log") == pytest.approx(ref["se_log"], abs=_TOL)
    lo, hi = _g(res, "ci")
    assert lo == pytest.approx(ref["ci_lo"], abs=_TOL)
    assert hi == pytest.approx(ref["ci_hi"], abs=_TOL)


def test_number_needed_to_treat_matches_R(r_reference):
    # Point estimate = 1/RD; CI convention differs when the RD CI crosses zero,
    # so only the estimate is pinned.
    res = sp.number_needed_to_treat(30, 70, 20, 80)
    ref = r_reference["NNT"]["estimate"]
    assert _g(res, "estimate") == pytest.approx(ref, abs=_TOL)


def test_incidence_rate_ratio_matches_R(r_reference):
    res = sp.incidence_rate_ratio(40, 100, 20, 100)
    ref = r_reference["IRR"]
    assert _g(res, "estimate") == pytest.approx(ref["estimate"], abs=_TOL)
    lo, hi = _g(res, "ci")
    assert lo == pytest.approx(ref["ci_lo"], abs=1e-10)
    assert hi == pytest.approx(ref["ci_hi"], abs=1e-10)
