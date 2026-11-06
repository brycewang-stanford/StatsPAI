"""Exporters check the file suffix (rOpenSci audit probe P12).

``result.to_excel("t.docx")`` wrote xlsx bytes under a Word name and
``to_excel("tab")`` a file no spreadsheet program opens by name.
"""

from __future__ import annotations

import io
import pathlib
import warnings
import zipfile

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai._export_path import checks_export_path, export_path
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def results():
    rng = np.random.default_rng(4)
    n = 200
    d = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    d["d"] = (rng.uniform(size=n) < 0.5).astype(int)
    d["y"] = 1 + d.x + d.d + rng.normal(size=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = sp.regress("y ~ x", data=d)
        cr = sp.dml(d, y="y", treat="d", covariates=["x", "z"], model="plr")
    return {"regress": reg, "dml": cr, "regtable": sp.regtable(reg)}


@pytest.mark.parametrize("kind", ["regress", "dml", "regtable"])
@pytest.mark.parametrize(
    "method,wrong", [("to_excel", "t.docx"), ("to_word", "t.xlsx")]
)
def test_mismatched_suffix_is_refused(results, tmp_path, kind, method, wrong):
    with pytest.raises(MethodIncompatibility) as err:
        getattr(results[kind], method)(str(tmp_path / wrong))
    assert err.value.diagnostics["suffix"] == pathlib.Path(wrong).suffix
    assert list(tmp_path.iterdir()) == []  # nothing written


@pytest.mark.parametrize("kind", ["regress", "dml", "regtable"])
@pytest.mark.parametrize("method,suffix", [("to_excel", ".xlsx"), ("to_word", ".docx")])
def test_missing_suffix_is_appended(results, tmp_path, kind, method, suffix):
    getattr(results[kind], method)(str(tmp_path / "tab"))
    written = tmp_path / f"tab{suffix}"
    assert [p.name for p in tmp_path.iterdir()] == [written.name]
    assert zipfile.is_zipfile(written)  # both formats are zip containers


def test_returned_path_is_the_written_one(results, tmp_path):
    out = results["regress"].to_excel(tmp_path / "tab")
    assert pathlib.Path(out) == tmp_path / "tab.xlsx"
    assert isinstance(out, pathlib.Path)
    assert results["regress"].to_word(str(tmp_path / "t.docx")) == str(
        tmp_path / "t.docx"
    )


def test_latex_file_and_string(results, tmp_path):
    with pytest.raises(MethodIncompatibility):
        results["regress"].to_latex(str(tmp_path / "t.docx"))
    text = results["regress"].to_latex(str(tmp_path / "tab"))
    assert (tmp_path / "tab.tex").read_text(encoding="utf-8") == text
    assert isinstance(results["regress"].to_latex(), str)


def test_buffers_pass_through(results):
    buf = io.BytesIO()
    results["regress"].to_excel(buf)
    assert zipfile.is_zipfile(io.BytesIO(buf.getvalue()))


@pytest.mark.parametrize(
    "path,fmt,expected",
    [
        ("a.xlsx", "excel", "a.xlsx"),
        ("a.XLSM", "excel", "a.XLSM"),
        ("a", "excel", "a.xlsx"),
        ("a.v2", "word", "a.v2.docx"),
        ("dir.d/a", "latex", "dir.d/a.tex"),
        ("a.txt", "latex", "a.txt"),
        (None, "latex", None),
    ],
)
def test_export_path_rules(path, fmt, expected):
    assert export_path(path, fmt, method="m") == expected


def test_decorator_requires_a_path_parameter():
    with pytest.raises(TypeError):

        @checks_export_path("excel")
        def f(self, x):
            return x


def test_decorator_handles_keyword_and_positional():
    class Demo:
        @checks_export_path("word")
        def to_word(self, filename, title=None):
            return filename

    assert Demo().to_word("a") == "a.docx"
    assert Demo().to_word(filename="b") == "b.docx"
