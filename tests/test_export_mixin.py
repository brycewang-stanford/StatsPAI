"""Tests for the generic, faithful ExportMixin on result objects.

The mixin gives result classes a uniform export quartet
(``to_markdown`` / ``to_latex`` / ``to_excel`` / ``to_word``) plus a
non-fabricating ``cite``. These tests pin:
  * faithfulness per result shape (tidy / coef-table / single-estimate / card),
  * that a 2-D coefficient matrix is NOT flattened into a misleading table,
  * renderers produce valid output that round-trips,
  * ``cite`` never invents a citation (CLAUDE.md §10),
  * a subclass's own export method always wins over the mixin,
  * real estimator results gain the methods without losing ``summary()``.
"""

import os
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from statspai.core.results import ExportMixin


# --- synthetic shapes -------------------------------------------------------

class _Coef(ExportMixin):
    def __init__(self):
        self.params = pd.Series({"x": 1.0, "z": 2.0})
        self.std_errors = pd.Series({"x": 0.1, "z": 0.2})


class _Estimate(ExportMixin):
    estimate = 0.5
    se = 0.1
    pvalue = 0.04
    ci = (0.3, 0.7)


@dataclass
class _Card(ExportMixin):
    a: float = 1.0
    b: str = "hello"
    arr: Any = None  # array-valued field must be omitted from the card


class _Matrix(ExportMixin):
    def __init__(self):
        self.coef = np.ones((3, 2))  # 2-D: must NOT become a coef table
        self.n = 10


class _Tidy(ExportMixin):
    def tidy(self):
        return pd.DataFrame({"term": ["a", "b"], "estimate": [1.0, 2.0]})


class _Cited(ExportMixin):
    _citation_key = "foo2020"
    _CITATIONS = {"foo2020": "@article{foo2020, title={Bar}, year={2020}}"}
    estimate = 1.0


class _Override(ExportMixin):
    estimate = 1.0

    def to_latex(self, *a, **k):
        return "BESPOKE-LATEX"


def test_coef_table_shape():
    f = _Coef()._export_frame()
    assert list(f["term"]) == ["x", "z"]
    assert list(f["estimate"]) == [1.0, 2.0]
    assert list(f["std_error"]) == [0.1, 0.2]


def test_single_estimate_shape():
    f = _Estimate()._export_frame()
    assert f.shape[0] == 1
    assert f.loc[0, "estimate"] == 0.5
    assert f.loc[0, "std_error"] == 0.1
    assert f.loc[0, "conf_low"] == 0.3 and f.loc[0, "conf_high"] == 0.7


def test_scalar_card_omits_arrays():
    f = _Card(arr=np.array([1.0, 2.0, 3.0]))._export_frame()
    assert list(f.columns) == ["field", "value"]
    fields = set(f["field"])
    assert {"a", "b"} <= fields
    assert "arr" not in fields  # array-valued field omitted faithfully


def test_matrix_params_not_flattened():
    """A 2-D coefficient matrix degrades to a scalar card, never a fake table."""
    f = _Matrix()._export_frame()
    assert list(f.columns) == ["field", "value"]
    assert "n" in set(f["field"])
    # No flattened 'param[i]' rows leaked in.
    assert not any(str(v).startswith("param[") for v in f["field"])


def test_tidy_is_preferred():
    f = _Tidy()._export_frame()
    assert list(f["term"]) == ["a", "b"]


def test_renderers_produce_valid_output():
    r = _Coef()
    md = r.to_markdown()
    assert "|" in md and "term" in md
    assert "tabular" in r.to_latex()
    with tempfile.TemporaryDirectory() as d:
        xlsx = r.to_excel(os.path.join(d, "a.xlsx"))
        docx = r.to_word(os.path.join(d, "a.docx"))
        assert os.path.exists(xlsx) and os.path.exists(docx)
        back = pd.read_excel(xlsx)
        assert back.shape == r._export_frame().shape


def test_to_markdown_writes_path():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.md")
        out = _Coef().to_markdown(p)
        assert out == p
        assert "term" in open(p, encoding="utf-8").read()


def test_cite_does_not_fabricate():
    msg = str(_Estimate().cite())
    assert "No verified citation" in msg
    assert "@" not in msg  # never invents a bibtex entry


def test_cite_returns_registered_citation():
    out = str(_Cited().cite())
    assert "@article{foo2020" in out


def test_subclass_export_method_wins():
    assert _Override().to_latex() == "BESPOKE-LATEX"


def test_real_estimators_gain_exports_keep_summary():
    import statspai as sp
    rng = np.random.default_rng(0)
    arima = sp.arima(np.cumsum(rng.normal(0, 1, 200)), order=(1, 1, 1))
    boot = sp.bootstrap(pd.DataFrame({"v": rng.normal(0, 1, 300)}),
                        statistic=lambda dd: dd["v"].mean(), n_boot=100)
    for r in (arima, boot):
        assert isinstance(r._export_frame(), pd.DataFrame)
        assert "|" in r.to_markdown()
        assert "tabular" in r.to_latex()
        assert r.summary()  # bespoke summary untouched
    # ARIMA exposes a real coefficient table; bootstrap a single estimate row.
    assert "term" in arima._export_frame().columns
    assert "estimate" in boot._export_frame().columns
