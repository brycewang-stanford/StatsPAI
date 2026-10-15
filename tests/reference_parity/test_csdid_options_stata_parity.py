"""Stata ``csdid`` standard errors for the CS option values fixture 82 pins on
point estimates only, and ``csdid``'s pairwise rule on an unbalanced panel.

* ``notyet_cutoff='asinr'`` and ``'cohort'`` (``csdid ..., notyet long2
  [asinr] method(reg)``): every ATT(g, t) and its standard error. R ``did``
  has one not-yet-treated rule (``'period'``), so Stata is the only reference
  for these two values.
* ``estimator='ipw_abadie'`` (``method(ipw)``, Abadie 2005) with ``lpop``:
  cells and standard errors. (R ``DRDID::ipw_did_panel`` through ``att_gt``
  is the canonical reference, in ``test_cs_options_R_parity.py``; this is
  the second implementation.)
* ``balance='pair'`` -- StatsPAI's default on an unbalanced panel, and the
  rule of ``csdid`` 1.8x ("Will use observations with Pair balanced
  (observed at t0 and t1)"): each ATT(g, t) uses the units seen in both of
  its periods. R ``did`` instead drops incomplete units (``balance='full'``),
  so until now this default had no reference. Two fits on mpdta with every
  37th row removed: ``method(dripw)`` with ``lpop`` and ``notyet
  method(reg)``.

Fixture ``tests/stata_parity/option_parity/results/96_csdid_options_Stata.json``
from ``96_csdid_options.do`` (Stata 18 MP, csdid 1.81, double precision; data
``data_96_csdid_unbalanced.csv`` from ``_generate_data_96_csdid_options.py``).
Point estimates at ``rtol = 1e-8`` (measured 4e-14). Standard errors at 1e-8
for the regression fits (measured 3e-15) and 1e-7 where a propensity score is
estimated (measured 1.1e-8 for method(ipw), 2.0e-8 for method(dripw)): the
point estimates agree to 4e-14, so the logit coefficients agree, and the gap
is in the variance of the propensity-score step, which csdid / drdid build in
Mata from the logit's e(V).
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import pandas as pd
import pytest

import statspai as sp

_TESTS = Path(__file__).resolve().parents[1]
_OPTION = _TESTS / "stata_parity" / "option_parity"
_REF = json.loads(
    (_OPTION / "results" / "96_csdid_options_Stata.json").read_text(encoding="utf-8")
)
_MPDTA = _TESTS / "orig_parity" / "data" / "02_mpdta_original.csv"
_UNBALANCED = _OPTION / "data_96_csdid_unbalanced.csv"
_CELL = re.compile(r"g(\d+):t_(\d+)_(\d+)")
_RTOL = 1e-8
_RTOL_SE_PSCORE = 1e-7

#: fixture block -> (data file, sp.callaway_santanna options)
_CASES = {
    "notyet_asinr_reg": (
        _MPDTA,
        dict(estimator="reg", control_group="notyettreated", notyet_cutoff="asinr"),
    ),
    "notyet_cohort_reg": (
        _MPDTA,
        dict(estimator="reg", control_group="notyettreated", notyet_cutoff="cohort"),
    ),
    "ipw_abadie_lpop": (_MPDTA, dict(estimator="ipw_abadie", x=["lpop"])),
    "unbalanced_dripw_lpop": (_UNBALANCED, dict(estimator="dr", x=["lpop"])),
    "unbalanced_notyet_reg": (
        _UNBALANCED,
        dict(estimator="reg", control_group="notyettreated", notyet_cutoff="cohort"),
    ),
}


def _stata_cells(block: str) -> dict:
    """csdid's long2 cells keyed by (cohort, date): a pre-treatment cell
    ``t_<period>_<base>`` is dated by its first year, a post-treatment cell
    ``t_<base>_<period>`` by its second."""
    out = {}
    for name, (b, se) in _REF[block].items():
        m = _CELL.fullmatch(name)
        if m:
            g, first, second = (int(v) for v in m.groups())
            out[(g, first if second < g else second)] = (b, se)
    return out


def _fit(block: str):
    path, options = _CASES[block]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.callaway_santanna(
            pd.read_csv(path),
            y="lemp",
            g="first_treat",
            t="year",
            i="countyreal",
            base_period="universal",
            **options,
        )


def test_cases_cover_the_fixture():
    assert set(_CASES) == set(_REF) - {"_meta"}
    assert _REF["_meta"]["precision"] == "double"


@pytest.mark.parametrize("block", sorted(_CASES))
def test_cells_and_standard_errors_match_csdid(block):
    fit = _fit(block)
    want = _stata_cells(block)
    got = {
        (int(r.group), int(r.time)): (float(r.att), float(r.se))
        for r in fit.detail.itertuples()
    }
    assert set(got) == set(want), block
    rtol_se = _RTOL_SE_PSCORE if _CASES[block][1]["estimator"] != "reg" else _RTOL
    for key, (b, se) in want.items():
        assert got[key][0] == pytest.approx(b, rel=_RTOL, abs=1e-12), (block, key)
        assert got[key][1] == pytest.approx(se, rel=rtol_se), (block, key)


def test_the_unbalanced_fits_are_read_as_the_pairwise_sample():
    for block in ("unbalanced_dripw_lpop", "unbalanced_notyet_reg"):
        fit = _fit(block)
        assert fit.model_info["balance"] == "pair"
        scope = sp.validation_scope(fit)
        assert scope["configuration"]["sample"] == "unbalanced_pair"
        assert scope["outputs"]["se"]["status"] == "reference", block


def test_asinr_and_cohort_differ_on_pre_treatment_cells():
    """Not vacuous: the two cutoffs give different placebo cells."""
    a, c = _stata_cells("notyet_asinr_reg"), _stata_cells("notyet_cohort_reg")
    assert a[(2004, 2004)] == c[(2004, 2004)]
    assert abs(a[(2006, 2003)][0] - c[(2006, 2003)][0]) > 1e-4
