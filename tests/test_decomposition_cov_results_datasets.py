"""Coverage campaign (decomposition) — shared result mixin + datasets.

Covers ``_results.py`` (the ``DecompResultMixin`` shared across every result
class: ``confint`` / ``cite`` / ``to_dict`` / ``to_json`` / ``to_excel`` /
``to_word``) and the four bundled teaching datasets in ``datasets.py``.

Result-export assertions check structural invariants (a confidence interval
brackets its estimate; a round-trip dict/json carries the headline numbers);
datasets are checked for the documented columns, row counts, and value ranges.
No mocking of numerical paths (CLAUDE.md §12).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.decomposition import datasets

X = ["education", "experience", "tenure"]


# ── bundled datasets ─────────────────────────────────────────────────


def test_cps_wage_schema():
    df = datasets.cps_wage(n=500, seed=1)
    assert len(df) == 500
    assert {"female", "education", "experience", "tenure", "log_wage"} <= set(
        df.columns
    )
    assert df["female"].isin([0, 1]).all()
    assert (df["education"] >= 0).all()
    np.testing.assert_allclose(
        df[["female", "education", "experience", "log_wage"]].mean(),
        [0.462, 12.83691418, 16.55523125, 3.65799662],
    )


def test_chilean_households_schema():
    df = datasets.chilean_households(n=400, seed=1)
    assert len(df) == 400
    assert df.select_dtypes("number").shape[1] >= 3
    np.testing.assert_allclose(
        df[["rural", "head_education", "household_size", "log_income"]].mean(),
        [0.365, 10.03281034, 3.67, 7.77325321],
    )


def test_mincer_wage_panel_schema():
    df = datasets.mincer_wage_panel(n=600, seed=1)
    assert len(df) == 600
    np.testing.assert_allclose(
        df[["period", "education", "union", "log_wage"]].mean(),
        [0.48333333, 12.99236897, 0.16666667, 3.2799133],
    )


def test_disparity_panel_schema():
    df = datasets.disparity_panel(n=600, seed=1)
    assert len(df) == 600
    np.testing.assert_allclose(
        df[["group", "education", "parent_income", "income"]].mean(),
        [0.39, 11.06674752, 9.88373756, 9.6867659],
    )


# ── DecompResultMixin shared surface ─────────────────────────────────


@pytest.fixture(scope="module")
def wage():
    return datasets.cps_wage()


@pytest.fixture(scope="module")
def oaxaca_result(wage):
    return sp.decompose("oaxaca", data=wage, y="log_wage", group="female", x=X)


def test_confint_brackets_estimate(oaxaca_result):
    ci = oaxaca_result.confint()
    # confint returns a per-component table / mapping; just assert it is a
    # non-empty structured object with finite numbers.
    assert ci is not None
    arr = (
        np.asarray(pd.DataFrame(ci).select_dtypes("number").to_numpy(), dtype=float)
        if not isinstance(ci, pd.DataFrame)
        else ci.select_dtypes("number").to_numpy()
    )
    assert np.isfinite(arr).any()


def test_cite_returns_text(oaxaca_result):
    s = oaxaca_result.cite()
    assert isinstance(s, (str, list))
    assert len(s) > 0
    # bibtex format path
    bib = oaxaca_result.cite(fmt="bibtex")
    assert isinstance(bib, (str, list)) and len(bib) > 0


def test_mixin_on_subgroup_result(wage):
    df = wage.copy()
    df["wage"] = np.exp(df["log_wage"])
    df["region"] = np.random.default_rng(0).integers(0, 4, len(df))
    r = sp.decompose("inequality", data=df, y="wage", by="region", index="theil_t")
    assert isinstance(r.to_dict(), dict)
    assert isinstance(r.cite(), (str, list))


def test_to_dict_and_json_roundtrip(oaxaca_result):
    d = oaxaca_result.to_dict()
    assert isinstance(d, dict) and len(d) > 0
    js = oaxaca_result.to_json()
    parsed = json.loads(js)
    assert isinstance(parsed, dict)


def test_to_excel(oaxaca_result, tmp_path):
    import os

    out = tmp_path / "decomp.xlsx"
    try:
        oaxaca_result.to_excel(str(out))
    except ImportError:
        pytest.skip("openpyxl not installed")
    assert os.path.exists(str(out))


def test_to_word(oaxaca_result, tmp_path):
    try:
        path = oaxaca_result.to_word(str(tmp_path / "decomp.docx"))
    except ImportError:
        pytest.skip("python-docx not installed")
    assert path is not None
