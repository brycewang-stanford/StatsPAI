"""Tests for the publication-table export surface on ``EconometricResults``.

Historically only :class:`CausalResult` carried ``.to_latex()`` /
``.to_html()`` / ``.to_markdown()`` / ``.to_excel()`` / ``.to_word()``;
``EconometricResults`` (returned by ``sp.regress`` / ``sp.ols`` / ``sp.iv``)
only had ``.to_docx()``.  These tests lock in the symmetric surface: every
format must exist on ``EconometricResults``, delegate to ``regtable`` (so the
output matches a one-column publication table), forward ``regtable`` kwargs
(``coef_labels`` / ``keep`` / ``drop`` / ``order`` / ``template`` / ``se_type``
…), and optionally write to a path.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import EconometricResults


@pytest.fixture
def ols_result():
    """A single OLS result with a known, well-identified DGP."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    df["y"] = 1.0 + 2.0 * df["x"] - 0.5 * df["z"] + rng.normal(size=n)
    r = sp.regress("y ~ x + z", data=df)
    assert isinstance(r, EconometricResults)
    return r


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ----------------------------------------------------------------------
# Surface: the methods must exist (the headline gap being closed)
# ----------------------------------------------------------------------


class TestExportSurfaceExists:

    @pytest.mark.parametrize(
        "method",
        ["to_latex", "to_html", "to_markdown", "to_excel", "to_word", "to_docx"],
    )
    def test_method_present(self, ols_result, method):
        assert hasattr(ols_result, method)
        assert callable(getattr(ols_result, method))

    def test_parity_with_causal_result(self):
        """EconometricResults must expose every string/file export that
        CausalResult does, so the two are interchangeable in user code.

        Introspects the class definitions directly (no fitting needed) so
        the contract is checked even if a particular estimator is slow.
        """
        from statspai.core.results import CausalResult

        export_methods = {"to_latex", "to_html", "to_markdown", "to_excel", "to_word"}
        causal_has = {
            m for m in export_methods if callable(getattr(CausalResult, m, None))
        }
        econ_has = {
            m for m in export_methods if callable(getattr(EconometricResults, m, None))
        }
        missing = causal_has - econ_has
        assert not missing, (
            "EconometricResults is missing export methods present on "
            f"CausalResult: {sorted(missing)}"
        )


# ----------------------------------------------------------------------
# LaTeX
# ----------------------------------------------------------------------


class TestToLatex:

    def test_returns_table_float(self, ols_result):
        tex = ols_result.to_latex()
        assert "\\begin{table}" in tex
        assert "\\begin{tabular}" in tex
        # SE in parentheses + stars are the regtable convention
        assert "(" in tex and ")" in tex

    def test_caption_and_label(self, ols_result):
        tex = ols_result.to_latex(caption="Main results", label="tab:main")
        assert "\\caption{Main results}" in tex
        assert "\\label{tab:main}" in tex
        # label must come right after the caption line
        lines = tex.split("\n")
        cap_i = next(
            i for i, x in enumerate(lines) if x.lstrip().startswith("\\caption")
        )
        assert lines[cap_i + 1].strip() == "\\label{tab:main}"

    def test_label_without_caption_anchors_on_centering(self, ols_result):
        tex = ols_result.to_latex(label="tab:nocap")
        assert "\\label{tab:nocap}" in tex

    def test_coef_labels_passthrough(self, ols_result):
        tex = ols_result.to_latex(coef_labels={"x": "Treatment effect"})
        assert "Treatment effect" in tex

    def test_keep_passthrough(self, ols_result):
        tex = ols_result.to_latex(keep=["x"])
        assert "x" in tex
        # z dropped: it should not appear as its own coefficient row label
        assert "z &" not in tex

    def test_template_passthrough(self, ols_result):
        # Should not raise and should still be a valid table float.
        tex = ols_result.to_latex(template="aer")
        assert "\\begin{table}" in tex

    def test_path_write(self, ols_result, tmp_dir):
        p = os.path.join(tmp_dir, "t.tex")
        out = ols_result.to_latex(path=p)
        assert os.path.exists(p)
        with open(p, encoding="utf-8") as fh:
            assert fh.read() == out


# ----------------------------------------------------------------------
# HTML / Markdown
# ----------------------------------------------------------------------


class TestToHtml:

    def test_returns_html_table(self, ols_result):
        html = ols_result.to_html()
        low = html.lower()
        assert "<table" in low or "<td" in low

    def test_se_type_passthrough(self, ols_result):
        # se_type="t" must not error and must still render.
        html = ols_result.to_html(se_type="t")
        assert "<" in html

    def test_path_write(self, ols_result, tmp_dir):
        p = os.path.join(tmp_dir, "t.html")
        out = ols_result.to_html(path=p)
        with open(p, encoding="utf-8") as fh:
            assert fh.read() == out


class TestToMarkdown:

    def test_returns_pipe_table(self, ols_result):
        md = ols_result.to_markdown()
        assert "|" in md

    def test_quarto_requires_label(self, ols_result):
        # Quarto cross-refs require an id; regtable enforces this.
        with pytest.raises(ValueError):
            ols_result.to_markdown(quarto=True)

    def test_quarto_with_label(self, ols_result):
        md = ols_result.to_markdown(
            quarto=True, quarto_label="main", quarto_caption="Main results"
        )
        assert "{#tbl-main}" in md

    def test_drop_passthrough(self, ols_result):
        md = ols_result.to_markdown(drop=["Intercept"])
        assert "Intercept" not in md

    def test_path_write(self, ols_result, tmp_dir):
        p = os.path.join(tmp_dir, "t.md")
        out = ols_result.to_markdown(path=p)
        with open(p, encoding="utf-8") as fh:
            assert fh.read() == out


# ----------------------------------------------------------------------
# Excel / Word (binary; require optional deps)
# ----------------------------------------------------------------------


class TestToExcel:

    def test_writes_file(self, ols_result, tmp_dir):
        openpyxl = pytest.importorskip("openpyxl")
        p = os.path.join(tmp_dir, "t.xlsx")
        out = ols_result.to_excel(p)
        assert out == p
        assert os.path.getsize(p) > 1000
        ws = openpyxl.load_workbook(p).active
        assert ws.sheet_view.showGridLines is False
        assert ws.freeze_panes == "B2"
        assert ws.page_setup.fitToWidth == 1

    def test_kwargs_passthrough(self, ols_result, tmp_dir):
        pytest.importorskip("openpyxl")
        p = os.path.join(tmp_dir, "t.xlsx")
        ols_result.to_excel(p, coef_labels={"x": "Treatment"})
        assert os.path.exists(p)


class TestToWord:

    def test_writes_file(self, ols_result, tmp_dir):
        pytest.importorskip("docx")
        from docx import Document

        p = os.path.join(tmp_dir, "t.docx")
        out = ols_result.to_word(p)
        assert out == p
        assert os.path.getsize(p) > 1000
        doc = Document(p)
        assert doc.sections[0].left_margin.inches == pytest.approx(1.0)

    def test_caption(self, ols_result, tmp_dir):
        pytest.importorskip("docx")
        p = os.path.join(tmp_dir, "t.docx")
        ols_result.to_word(p, caption="My caption")
        assert os.path.exists(p)


# ----------------------------------------------------------------------
# Cross-cutting: delegation fidelity
# ----------------------------------------------------------------------


class TestDelegationFidelity:

    def test_latex_matches_regtable(self, ols_result):
        """A single-model to_latex() must equal regtable(self).to_latex()."""
        assert ols_result.to_latex() == sp.regtable(ols_result).to_latex()

    def test_html_matches_regtable(self, ols_result):
        assert ols_result.to_html() == sp.regtable(ols_result).to_html()

    def test_markdown_matches_regtable(self, ols_result):
        assert ols_result.to_markdown() == sp.regtable(ols_result).to_markdown()
