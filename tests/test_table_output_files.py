"""``output=`` / ``format=`` of the table facades write what they say.

2026-10-19 audit: ``sp.modelsummary(output="x.tex")`` (or ``.html`` /
``.md``) fell through to the text renderer and wrote nothing; a misspelt
keyword returned text; ``sp.esttab(output="bogus")`` was accepted; and
``sp.outreg2(filename="t.html")`` wrote an Excel workbook under an ``.html``
name. Each file must now hold exactly what the matching renderer returns.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def model():
    r = np.random.default_rng(0)
    df = pd.DataFrame({"x": r.normal(size=200)})
    df["y"] = 1 + 2 * df.x + r.normal(size=200)
    return sp.regress("y ~ x", data=df)


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


@pytest.mark.parametrize(
    "ext, keyword",
    [(".tex", "latex"), (".html", "html"), (".md", "markdown"), (".txt", "text")],
)
def test_modelsummary_path_writes_the_rendering(model, tmp_path, ext, keyword):
    path = tmp_path / f"table{ext}"
    msg = _quiet(sp.modelsummary, model, output=str(path))
    assert path.exists() and str(path) in msg
    assert path.read_text(encoding="utf-8") == _quiet(
        sp.modelsummary, model, output=keyword
    )


def test_modelsummary_csv_is_the_dataframe(model, tmp_path):
    path = tmp_path / "table.csv"
    _quiet(sp.modelsummary, model, output=str(path))
    frame = _quiet(sp.modelsummary, model, output="dataframe")
    assert path.read_text(encoding="utf-8") == frame.to_csv()


@pytest.mark.parametrize("bad", ["latx", "table.pdf", "word"])
def test_modelsummary_unknown_output_raises(model, bad):
    with pytest.raises(sp.MethodIncompatibility):
        _quiet(sp.modelsummary, model, output=bad)


def test_esttab_unknown_output_raises(model):
    with pytest.raises(sp.MethodIncompatibility):
        _quiet(sp.esttab, model, output="bogus")


@pytest.mark.parametrize("ext", [".html", ".md", ".txt"])
def test_outreg2_text_formats_by_extension(model, tmp_path, ext):
    path = tmp_path / f"t{ext}"
    rendered = _quiet(sp.outreg2, model, filename=str(path))
    assert path.read_text(encoding="utf-8") == rendered
    # Not an Excel workbook (zip magic "PK") under a text name.
    assert not path.read_bytes().startswith(b"PK")


def test_outreg2_unknown_extension_raises(model, tmp_path):
    with pytest.raises(sp.MethodIncompatibility):
        _quiet(sp.outreg2, model, filename=str(tmp_path / "t.pdf"))
    with pytest.raises(sp.MethodIncompatibility):
        _quiet(sp.outreg2, model, filename=str(tmp_path / "t.xlsx"), format="pdf")


def test_outreg2_bare_name_still_writes_excel(model, tmp_path):
    pytest.importorskip("openpyxl")
    path = tmp_path / "results"
    _quiet(sp.outreg2, model, filename=str(path))
    assert path.read_bytes().startswith(b"PK")
