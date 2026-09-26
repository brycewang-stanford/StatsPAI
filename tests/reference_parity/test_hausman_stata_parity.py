"""Hausman FE-vs-RE test against Stata 18 ``xtreg`` / ``hausman``.

Reference: ``_fixtures/_generate_hausman_stata.do`` (data generated and exported
by Stata; the JSON holds ``_b`` from ``xtreg, fe`` / ``xtreg, re`` and
``r(chi2)`` from ``hausman fe re`` with and without ``sigmamore``).

Panel ``a``: unit effect independent of x1. Panel ``b``: x1 loads on the unit
effect; Stata's classical statistic is negative there (it warns and gives no
usable p-value), while ``sigmamore`` rejects RE decisively.

Budget: the default 1e-6 relative.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import AssumptionWarning

FIX = Path(__file__).parent / "_fixtures"
REL = 1e-6


@pytest.fixture(scope="module")
def stata():
    return json.loads((FIX / "hausman_stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(FIX / "hausman_data.csv")


def _panel(data: pd.DataFrame, p: str) -> pd.DataFrame:
    return data.rename(columns={f"x1{p}": "x1", f"y{p}": "y"})[
        ["id", "year", "y", "x1", "x2"]
    ]


def _run(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AssumptionWarning)
        return sp.hausman_test(df, y="y", x=["x1", "x2"], id="id", time="year", **kw)


@pytest.mark.parametrize("p", ["a", "b"])
def test_coefficients_match_xtreg(stata, data, p):
    out = _run(_panel(data, p))
    ref = stata[p]
    for side in ("fe", "re"):
        for term in ("x1", "x2"):
            got = float(out[f"beta_{side}"][term])
            assert got == pytest.approx(ref[f"b_{side}_{term}"], rel=REL)


@pytest.mark.parametrize("p", ["a", "b"])
def test_classical_statistic_matches_hausman(stata, data, p):
    out = _run(_panel(data, p))
    assert out["df"] == stata[p]["df"]
    assert out["statistic"] == pytest.approx(stata[p]["chi2"], rel=REL)


@pytest.mark.parametrize("p", ["a", "b"])
def test_sigmamore_matches_hausman_sigmamore(stata, data, p):
    out = _run(_panel(data, p), sigmamore=True)
    assert out["sigmamore"] is True
    assert out["statistic"] == pytest.approx(stata[p]["chi2_sigmamore"], rel=REL)
    assert out["pvalue"] == pytest.approx(stata[p]["p_sigmamore"], rel=1e-5)


def test_negative_statistic_is_inconclusive_and_sigmamore_decides(stata, data):
    df = _panel(data, "b")
    assert stata["b"]["chi2"] < 0
    with pytest.warns(AssumptionWarning, match="sigmamore=True"):
        classical = sp.hausman_test(df, y="y", x=["x1", "x2"], id="id", time="year")
    assert classical["recommendation"] == "inconclusive"
    assert np.isnan(classical["pvalue"])
    more = _run(df, sigmamore=True)
    assert more["recommendation"] == "FE"


def test_function_and_panel_method_agree(data):
    df = _panel(data, "a")
    fe = sp.panel(df, "y ~ x1 + x2", entity="id", time="year", method="fe")
    for kw in ({}, {"sigmamore": True}):
        method = fe.hausman_test(**kw)
        free = _run(df, **kw)
        assert method["statistic"] == pytest.approx(free["statistic"], rel=1e-12)
        assert method["recommendation"] == free["recommendation"]
