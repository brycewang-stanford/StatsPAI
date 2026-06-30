"""Reference parity: multiple-testing corrections vs base R ``stats::p.adjust``.

``sp.bonferroni`` / ``sp.holm`` / ``sp.benjamini_hochberg`` and the
``sp.adjust_pvalues`` dispatcher are deterministic functions of a p-value
vector; base R ``stats::p.adjust`` is the canonical reference. The match is
exact (the algorithms are identical step-down / step-up procedures).

Frozen reference: ``_fixtures/mht_R.json`` (base R 4.5.2). Regenerate with::

    Rscript tests/reference_parity/_generate_mht_R.R

References
----------
- Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure.
  *Scandinavian Journal of Statistics* 6(2), 65-70.
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the False Discovery Rate.
  *JRSS-B* 57(1), 289-300.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

import statspai as sp

_FIXTURES = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def r_reference():
    with open(_FIXTURES / "mht_R.json", encoding="utf-8") as fh:
        return json.load(fh)


def test_bonferroni_matches_R(r_reference):
    py = np.asarray(sp.bonferroni(r_reference["pvalues"]), dtype=float)
    assert np.allclose(py, r_reference["bonferroni"], atol=1e-15, rtol=0)


def test_holm_matches_R(r_reference):
    py = np.asarray(sp.holm(r_reference["pvalues"]), dtype=float)
    assert np.allclose(py, r_reference["holm"], atol=1e-15, rtol=0)


def test_benjamini_hochberg_matches_R(r_reference):
    py = np.asarray(sp.benjamini_hochberg(r_reference["pvalues"]), dtype=float)
    assert np.allclose(py, r_reference["BH"], atol=1e-15, rtol=0)


@pytest.mark.parametrize(
    "sp_method,r_key",
    [("bonferroni", "bonferroni"), ("holm", "holm"), ("bh", "BH")],
)
def test_adjust_pvalues_dispatcher_matches_R(r_reference, sp_method, r_key):
    py = np.asarray(
        sp.adjust_pvalues(r_reference["pvalues"], method=sp_method), dtype=float
    )
    assert np.allclose(py, r_reference[r_key], atol=1e-15, rtol=0)
