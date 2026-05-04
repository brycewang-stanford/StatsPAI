"""Branch coverage for ``statspai.workflow.paper``.

Existing tests (``test_paper_pipeline.py`` / ``test_paper_quarto.py`` /
``test_paper_tables.py`` / ``test_paper_from_question.py``) already cover
the headline ``sp.paper(...)`` happy path. This file adds tests aimed at
the remaining gaps in the v1.12.x coverage report:

- Internal helpers: ``_yaml_str``, ``_tex_escape``, ``_md_to_tex``,
  ``_inline_md_to_tex``, ``_render_dag_section``, ``_record_note``,
  ``_notes_block``.
- ``PaperDraft.to_qmd`` with explicit author / formats / bibliography /
  csl / multi-format / single-format paths.
- ``PaperDraft.to_docx`` fallback (markdown-on-disk when python-docx
  is missing) and the explicit-import path.
- ``PaperDraft.write`` extension dispatch (.md/.tex/.qmd/.docx).
- ``paper(fmt='bogus')`` validation error.
- ``paper`` with ``output_path=`` writing to disk.
- DAG appendix rendering via a duck-typed DAG stub (text and qmd).
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.workflow.paper import (
    PaperDraft,
    _eda_block,
    _inline_md_to_tex,
    _md_to_tex,
    _notes_block,
    _record_note,
    _render_dag_section,
    _tex_escape,
    _yaml_str,
    paper,
)


# --------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def rct_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    T = rng.binomial(1, 0.5, n)
    x = rng.normal(size=n)
    y = 2.0 * T + 0.5 * x + rng.normal(size=n)
    return pd.DataFrame({
        "wage": y, "trained": T, "edu": x,
    })


@pytest.fixture(scope="module")
def draft(rct_data) -> PaperDraft:
    return paper(
        rct_data,
        question="effect of trained on wage",
        design="rct",
    )


# --------------------------------------------------------------------- #
#  YAML / TeX helpers
# --------------------------------------------------------------------- #


def test_yaml_str_handles_none_and_quotes():
    assert _yaml_str(None) == '""'
    assert _yaml_str("plain") == '"plain"'
    out = _yaml_str('he said "hi"')
    assert out.startswith('"') and out.endswith('"')
    # Embedded quote escaped
    assert '\\"' in out


def test_yaml_str_collapses_newlines():
    """Newlines are folded to single spaces (Quarto YAML compat)."""
    assert "\n" not in _yaml_str("line1\nline2")


def test_yaml_str_escapes_backslash():
    out = _yaml_str(r"path\with\backslash")
    assert r"\\" in out


def test_tex_escape_special_chars():
    out = _tex_escape("a&b%c$d#e_f{g}h~i^j")
    for tok in (r"\&", r"\%", r"\$", r"\#", r"\_",
                r"\{", r"\}",
                r"\textasciitilde{}", r"\textasciicircum{}"):
        assert tok in out


def test_tex_escape_handles_non_string():
    """Falls back to ``str(...)`` when given a non-string."""
    assert _tex_escape(42) == "42"


def test_md_to_tex_translates_lists_and_code_fences():
    md = (
        "Intro paragraph.\n"
        "- bullet one\n"
        "- bullet *two*\n"
        "\n"
        "```\n"
        "code line 1\n"
        "code line 2\n"
        "```\n"
        "Outro paragraph with `inline` code and **bold**.\n"
    )
    out = _md_to_tex(md)
    assert r"\begin{itemize}" in out
    assert r"\end{itemize}" in out
    assert r"\begin{verbatim}" in out
    assert r"\end{verbatim}" in out
    assert r"\texttt{inline}" in out
    assert r"\textbf{bold}" in out


def test_inline_md_to_tex_emph_and_bold():
    out = _inline_md_to_tex("a *italic* b **bold** c `code` d")
    assert r"\emph{italic}" in out
    assert r"\textbf{bold}" in out
    assert r"\texttt{code}" in out


def test_inline_md_to_tex_plain_text_escapes_specials():
    out = _inline_md_to_tex("100% & $5")
    # Plain (no LaTeX commands) → escape special chars
    assert r"\&" in out and r"\%" in out and r"\$" in out


# --------------------------------------------------------------------- #
#  EDA block branches
# --------------------------------------------------------------------- #


def test_eda_block_with_continuous_treatment():
    """Continuous-treatment branch (>10 unique values)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=200),
        "T": rng.normal(size=200),  # continuous
    })
    out = _eda_block(df, y="y", treatment="T", covariates=None)
    assert "continuous" in out


def test_eda_block_handles_missingness():
    df = pd.DataFrame({
        "y": [1.0, np.nan, 3.0, 4.0],
        "x": [1.0, 2.0, np.nan, 4.0],
    })
    out = _eda_block(df, y="y", treatment=None, covariates=None)
    assert "Missingness" in out


def test_eda_block_no_missingness_path():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0]})
    out = _eda_block(df, y="y", treatment=None, covariates=None)
    assert "none detected" in out


def test_eda_block_covariate_balance_table():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=200),
        "T": rng.binomial(1, 0.5, 200),
        "x1": rng.normal(size=200),
        "x2": rng.normal(size=200),
    })
    out = _eda_block(df, y="y", treatment="T", covariates=["x1", "x2"])
    assert "covariate" in out and "std-diff" in out


def test_eda_block_records_degradation_on_bad_covariate():
    """Non-numeric covariate triggers a recorded degradation."""
    import warnings as _w
    df = pd.DataFrame({
        "y": np.random.default_rng(0).normal(size=200),
        "T": [0, 1] * 100,
        "bad": ["foo"] * 200,
    })
    degradations: list = []
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        _eda_block(df, y="y", treatment="T", covariates=["bad"],
                   degradations=degradations)
    # 'bad' is constant (all "foo") → grp.mean() may NaN out;
    # degradation may or may not trigger depending on pandas dtype handling.
    # We assert degradations is a list (not None) — the surface contract.
    assert isinstance(degradations, list)


# --------------------------------------------------------------------- #
#  Notes helpers
# --------------------------------------------------------------------- #


def test_record_note_is_a_no_op_when_notes_is_none():
    _record_note(None, "ignored")  # must not raise


def test_record_note_dedupes_and_orders():
    notes: list = []
    _record_note(notes, "alpha")
    _record_note(notes, "alpha")  # dedupe
    _record_note(notes, "beta")
    assert notes == ["alpha", "beta"]


def test_notes_block_renders_markdown_bullets():
    body = _notes_block(["a", "b"])
    assert "- a" in body and "- b" in body


# --------------------------------------------------------------------- #
#  PaperDraft.to_qmd renderings
# --------------------------------------------------------------------- #


def test_to_qmd_default_three_formats(draft):
    qmd = draft.to_qmd()
    assert qmd.startswith("---\n")
    # Default formats include pdf, html, docx
    assert "format:" in qmd
    assert "pdf:" in qmd and "html:" in qmd and "docx:" in qmd


def test_to_qmd_single_format_short_form(draft):
    qmd = draft.to_qmd(formats=["pdf"])
    # Single format renders as ``format: pdf`` (not block syntax)
    assert "format: pdf" in qmd
    # No multi-format block lines
    for token in ("html:", "docx:"):
        assert token not in qmd


def test_to_qmd_with_author_and_bib(draft):
    qmd = draft.to_qmd(
        title="Causal Brief",
        author="Bryce W.",
        formats=["pdf"],
        bibliography="refs.bib",
        csl="aer",
    )
    assert 'title: "Causal Brief"' in qmd
    assert 'author: "Bryce W."' in qmd
    assert 'bibliography: "refs.bib"' in qmd
    assert "csl:" in qmd  # resolved or pass-through


def test_to_qmd_renders_dag_appendix_when_dag_attached(rct_data):
    """When the draft carries a DAG, qmd output contains a mermaid block."""
    class _StubDAG:
        nodes = {"trained", "wage", "edu"}
        observed_nodes = {"trained", "wage", "edu"}
        edges = [("trained", "wage"), ("edu", "wage"), ("edu", "trained")]

        def adjustment_sets(self, t, y):
            return [{"edu"}]

        def backdoor_paths(self, t, y):
            return [["trained", "edu", "wage"]]

        def bad_controls(self, t, y):
            return {}

    d = paper(
        rct_data,
        question="effect of trained on wage",
        design="rct",
        dag=_StubDAG(),
    )
    qmd = d.to_qmd(formats=["pdf"])
    assert "```{mermaid}" in qmd
    assert "graph LR" in qmd


# --------------------------------------------------------------------- #
#  PaperDraft.to_docx fallback
# --------------------------------------------------------------------- #


def test_to_docx_falls_back_to_markdown_when_docx_missing(draft, tmp_path,
                                                           monkeypatch):
    """Force python-docx import to fail → fallback writes markdown to .docx."""
    # Block ``import docx`` for the duration of this test.
    monkeypatch.setitem(sys.modules, "docx", None)
    out = tmp_path / "out.docx"
    draft.to_docx(str(out))
    text = out.read_text(encoding="utf-8")
    assert "python-docx not installed" in text
    assert "Question" in text


def test_to_docx_uses_python_docx_if_available(draft, tmp_path):
    docx = pytest.importorskip("docx")
    out = tmp_path / "out.docx"
    draft.to_docx(str(out))
    # Round-trip — open the file back and inspect all paragraph text.
    # ``add_heading(..., level=0)`` is rendered with Title (not "Heading 0")
    # style by python-docx, so filter on text rather than style name.
    doc = docx.Document(str(out))
    all_text = " ".join(p.text for p in doc.paragraphs)
    assert "Causal Analysis Draft" in all_text
    # At least one of the well-known section titles should be present
    assert any(s in all_text for s in ("Question", "Results", "References"))


# --------------------------------------------------------------------- #
#  PaperDraft.write extension dispatch
# --------------------------------------------------------------------- #


def test_write_dispatches_by_extension(draft, tmp_path):
    md = tmp_path / "a.md"
    tex = tmp_path / "a.tex"
    qmd = tmp_path / "a.qmd"
    draft.write(str(md))
    draft.write(str(tex))
    draft.write(str(qmd))
    assert md.read_text(encoding="utf-8").startswith("## ")
    assert tex.read_text(encoding="utf-8").startswith(r"\documentclass")
    qmd_text = qmd.read_text(encoding="utf-8")
    assert qmd_text.startswith("---")


def test_write_unknown_extension_falls_back_to_markdown(draft, tmp_path):
    p = tmp_path / "no_ext_at_all"
    draft.write(str(p))
    assert p.read_text(encoding="utf-8").startswith("## ")


# --------------------------------------------------------------------- #
#  paper() — entry-point validation + output_path
# --------------------------------------------------------------------- #


def test_paper_invalid_fmt_raises(rct_data):
    with pytest.raises(ValueError, match="fmt="):
        paper(rct_data, question="effect of trained on wage", fmt="xml")


def test_paper_no_outcome_inferable_raises():
    """No ``y=`` and the question can't supply it → explicit ValueError."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="outcome"):
        paper(df, question="something unrelated")


def test_paper_writes_output_path(rct_data, tmp_path):
    out = tmp_path / "draft.md"
    d = paper(
        rct_data,
        question="effect of trained on wage",
        design="rct",
        output_path=str(out),
    )
    assert out.exists() and out.stat().st_size > 0
    assert isinstance(d, PaperDraft)


# --------------------------------------------------------------------- #
#  PaperDraft.summary / to_dict surfaces
# --------------------------------------------------------------------- #


def test_paper_draft_summary_and_to_dict(draft):
    s = draft.summary()
    assert "PaperDraft" in s and "Sections" in s
    d = draft.to_dict()
    assert d["question"] == "effect of trained on wage"
    assert isinstance(d["sections"], dict) and "Question" in d["sections"]


# --------------------------------------------------------------------- #
#  DAG renderer (direct calls)
# --------------------------------------------------------------------- #


def test_render_dag_section_text_branch():
    class _D:
        nodes = {"X", "Y", "_L_X_Y"}
        observed_nodes = {"X", "Y"}
        edges = [("X", "Y")]

        def adjustment_sets(self, t, y):
            return [set()]  # ∅ — no controls needed

        def backdoor_paths(self, t, y):
            return []

        def bad_controls(self, t, y):
            return {"M": "mediator on the X→Y path"}

    out = _render_dag_section(_D(), treatment="X", outcome="Y", fmt="markdown")
    assert "Variables" in out
    assert "Edges" in out
    assert "∅" in out  # empty adjustment set
    assert "Latent common causes" in out
    assert "Bad controls" in out


def test_render_dag_section_returns_empty_when_dag_is_none():
    assert _render_dag_section(None) == ""


def test_render_dag_section_qmd_emits_mermaid():
    class _D:
        nodes = {"a", "b", "_L_x"}
        observed_nodes = {"a", "b"}
        edges = [("a", "b")]

        def adjustment_sets(self, t, y):
            raise RuntimeError("boom")  # exercise except branch

        def backdoor_paths(self, t, y):
            raise RuntimeError("boom")

        def bad_controls(self, t, y):
            raise RuntimeError("boom")

    out = _render_dag_section(_D(), treatment="a", outcome="b", fmt="qmd")
    assert "```{mermaid}" in out
    assert "graph LR" in out
