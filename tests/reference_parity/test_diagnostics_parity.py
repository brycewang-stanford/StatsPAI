"""Reference parity: ``sp.het_test`` / ``sp.reset_test`` vs R ``lmtest``.

``lmtest`` (Zeileis & Hothorn) is the canonical R reference for regression
specification diagnostics. StatsPAI reproduces both to machine precision on a
committed deterministic dataset:

  * ``sp.het_test``  → ``lmtest::bptest`` (studentized Breusch-Pagan / Koenker,
    the lmtest default).
  * ``sp.reset_test`` → ``lmtest::resettest(power = 2:3, type = "fitted")``.

Frozen reference: ``_fixtures/diagnostics_R.json`` (R 4.5.2 / lmtest 0.9-40).
Regenerate with::

    Rscript tests/reference_parity/_generate_diagnostics_R.R

References
----------
- Breusch, T.S. & Pagan, A.R. (1979); Koenker, R. (1981).
- Ramsey, J.B. (1969). Tests for Specification Errors. *JRSS-B* 31(2), 350-371.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

import statspai as sp

_FIXTURES = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def diag_data():
    return pd.read_csv(_FIXTURES / "diagnostics_data.csv")


@pytest.fixture(scope="module")
def r_reference():
    with open(_FIXTURES / "diagnostics_R.json", encoding="utf-8") as fh:
        return json.load(fh)


def test_het_test_matches_lmtest_bptest(diag_data, r_reference):
    res = sp.het_test(diag_data, y="y", x=["x1", "x2"])
    bp = r_reference["breusch_pagan"]
    assert res["statistic"] == pytest.approx(bp["statistic"], rel=1e-10)
    assert res["pvalue"] == pytest.approx(bp["pvalue"], rel=1e-10)
    assert int(res["df"]) == bp["df"]


def test_reset_test_matches_lmtest_resettest(diag_data, r_reference):
    res = sp.reset_test(diag_data, y="y", x=["x1", "x2"])
    rs = r_reference["reset"]
    assert res["statistic"] == pytest.approx(rs["statistic"], rel=1e-10)
    assert res["pvalue"] == pytest.approx(rs["pvalue"], rel=1e-10)
    assert int(res["df1"]) == rs["df1"]
    assert int(res["df2"]) == rs["df2"]
