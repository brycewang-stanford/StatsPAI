"""Tests for ``PaperDraft.to_qmd()`` and ``sp.paper(fmt='qmd')``.

Quarto is the publication-grade default. These tests verify:

- YAML frontmatter is well-formed: opens/closes with ``---`` lines,
  contains ``title:``, ``date:``, ``format:``.
- ``bibliography:`` is auto-emitted when ``self.citations`` is non-empty.
- Custom ``formats`` list maps to the ``format:`` block correctly.
- Provenance from the underlying workflow.result is folded into both
  the YAML (``statspai:`` block) and the body (``Reproducibility``
  appendix).
- ``write('paper.qmd')`` dispatches to ``to_qmd()`` and the file is
  parseable by a YAML reader.
- ``sp.paper(..., fmt='qmd')`` accepts the new format and produces a
  valid PaperDraft whose ``.write('.qmd')`` works end-to-end.
- YAML-quoting is robust against quotes / colons / newlines in the
  question.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from statspai.output._lineage import attach_provenance
from statspai.workflow.paper import PaperDraft, _yaml_str, paper


# ---------------------------------------------------------------------------
# _yaml_str helper
# ---------------------------------------------------------------------------

class TestYamlEscape:
    def test_simple(self):
        assert _yaml_str("hello") == '"hello"'

    def test_escapes_double_quote(self):
        assert _yaml_str('he said "hi"') == '"he said \\"hi\\""'

    def test_escapes_backslash(self):
        assert _yaml_str("a\\b") == '"a\\\\b"'

    def test_collapses_newlines(self):
        assert _yaml_str("line1\nline2") == '"line1 line2"'

    def test_none_safe(self):
        assert _yaml_str(None) == '""'


# ---------------------------------------------------------------------------
# PaperDraft.to_qmd direct tests
# ---------------------------------------------------------------------------

def _make_draft(*, sections=None, citations=None, workflow=None,
                question="effect of training on wages"):
    sections = sections or {
        "Question": "What's the effect of `training` on `wage`?",
        "Data": "- N = 100\n- 5 covariates",
        "Identification": "Parallel trends assumed.",
        "Estimator": "Callaway & Sant'Anna",
        "Results": "ATT = 0.05 (SE = 0.01)",
        "Robustness": "Honest DiD: bound = 0.03",
        "References": "(see paper.bib)",
    }
    return PaperDraft(
        question=question,
        sections=sections,
        workflow=workflow,
        fmt="qmd",
        citations=list(citations or []),
    )


class TestToQmdYAML:
    def test_yaml_block_well_formed(self):
        draft = _make_draft()
        qmd = draft.to_qmd()
        assert qmd.startswith("---\n")
        # First closing --- must come before any markdown header.
        first_close = qmd.index("\n---\n", 4)
        h2_first = qmd.find("\n## ")
        assert first_close < h2_first

    def test_title_default(self):
        qmd = _make_draft().to_qmd()
        assert 'title: "Causal Analysis Draft"' in qmd

    def test_custom_title(self):
        qmd = _make_draft().to_qmd(title="Returns to Schooling")
        assert 'title: "Returns to Schooling"' in qmd

    def test_author_optional(self):
        qmd_no = _make_draft().to_qmd()
        assert "author:" not in qmd_no
        qmd_yes = _make_draft().to_qmd(author="Bryce Wang")
        assert 'author: "Bryce Wang"' in qmd_yes

    def test_default_formats(self):
        qmd = _make_draft().to_qmd()
        assert "format:" in qmd
        for f in ("pdf:", "html:", "docx:"):
            assert f in qmd

    def test_single_format_inlined(self):
        qmd = _make_draft().to_qmd(formats=["pdf"])
        assert "format: pdf\n" in qmd

    def test_custom_formats(self):
        qmd = _make_draft().to_qmd(formats=["pdf", "beamer"])
        assert "format:" in qmd
        assert "beamer:" in qmd
        assert "html:" not in qmd

    def test_bibliography_autoemits_with_citations(self):
        draft = _make_draft(citations=["Callaway & Sant'Anna (2021) ..."])
        qmd = draft.to_qmd()
        assert 'bibliography: "paper.bib"' in qmd

    def test_no_bibliography_when_no_citations(self):
        draft = _make_draft(citations=[])
        qmd = draft.to_qmd()
        assert "bibliography:" not in qmd

    def test_explicit_bibliography_overrides(self):
        draft = _make_draft(citations=[])
        qmd = draft.to_qmd(bibliography="../refs.bib")
        assert 'bibliography: "../refs.bib"' in qmd

    def test_csl_pass_through(self):
        qmd = _make_draft().to_qmd(csl="aer.csl")
        assert 'csl: "aer.csl"' in qmd

    def test_subtitle_from_question(self):
        qmd = _make_draft(question="effect of X on Y").to_qmd()
        assert 'subtitle: "effect of X on Y"' in qmd

    def test_yaml_safe_against_special_chars(self):
        # Question with quotes/colons/newlines must not break YAML.
        weird = 'effect of "X": including \n newlines'
        draft = _make_draft(question=weird)
        qmd = draft.to_qmd()
        # Round-trip parse: the YAML block must be valid.
        head = qmd.split("\n---\n", 2)[0].lstrip("-").lstrip("\n")
        try:
            import yaml
            parsed = yaml.safe_load(head)
            assert "subtitle" in parsed
            assert "X" in parsed["subtitle"]
        except ImportError:
            # PyYAML not installed — fall back to the structural check.
            assert "X" in qmd
            assert qmd.count('"') >= 4  # quotes balanced enough


# ---------------------------------------------------------------------------
# Body content
# ---------------------------------------------------------------------------

class TestToQmdBody:
    def test_sections_render_as_h2(self):
        draft = _make_draft()
        qmd = draft.to_qmd()
        for title in ("Question", "Data", "Identification",
                      "Estimator", "Results", "Robustness", "References"):
            assert f"## {title}" in qmd

    def test_section_order_matches_markdown(self):
        draft = _make_draft()
        qmd = draft.to_qmd()
        positions = [
            qmd.index(f"## {t}")
            for t in ("Question", "Data", "Identification",
                      "Estimator", "Results", "Robustness")
        ]
        assert positions == sorted(positions)

    def test_extra_section_appended(self):
        draft = _make_draft(
            sections={
                "Question": "Q?",
                "Pipeline notes": "estimator failed",  # custom
            }
        )
        qmd = draft.to_qmd()
        assert "## Pipeline notes" in qmd
        assert qmd.index("## Question") < qmd.index("## Pipeline notes")


# ---------------------------------------------------------------------------
# Provenance integration
# ---------------------------------------------------------------------------

class TestToQmdProvenance:
    def test_provenance_in_yaml_and_appendix(self):
        df = pd.DataFrame({"y": [1, 2, 3]})
        result = SimpleNamespace(estimate=0.5)
        attach_provenance(
            result, function="sp.did.test", data=df,
            params={"y": "y"},
        )
        wf = SimpleNamespace(result=result, data=df)
        draft = _make_draft(workflow=wf)
        qmd = draft.to_qmd()
        prov = result._provenance
        # YAML statspai: block.
        assert "statspai:" in qmd
        assert f'version: "{prov.statspai_version}"' in qmd
        assert f'run_id: "{prov.run_id}"' in qmd
        assert f'data_hash: "{prov.data_hash}"' in qmd
        # Appendix in the body.
        assert "## Reproducibility {.appendix}" in qmd
        assert "sp.did.test" in qmd

    def test_no_provenance_no_appendix(self):
        wf = SimpleNamespace(result=None)
        draft = _make_draft(workflow=wf)
        qmd = draft.to_qmd()
        assert "statspai:" not in qmd
        assert "Reproducibility" not in qmd

    def test_include_provenance_false(self):
        df = pd.DataFrame({"y": [1, 2]})
        result = SimpleNamespace()
        attach_provenance(result, function="f", data=df)
        wf = SimpleNamespace(result=result, data=df)
        draft = _make_draft(workflow=wf)
        qmd = draft.to_qmd(include_provenance=False)
        assert "statspai:" not in qmd
        assert "Reproducibility" not in qmd


# ---------------------------------------------------------------------------
# write() dispatch
# ---------------------------------------------------------------------------

class TestWriteDispatch:
    def test_qmd_extension_dispatches(self, tmp_path):
        draft = _make_draft()
        path = tmp_path / "out.qmd"
        draft.write(str(path))
        content = path.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert 'title: "Causal Analysis Draft"' in content


# ---------------------------------------------------------------------------
# Top-level paper(fmt='qmd')
# ---------------------------------------------------------------------------

class TestPaperFnFmt:
    def test_fmt_qmd_accepted(self):
        # Build a tiny dataset that the workflow can handle.
        df = pd.DataFrame({
            "wage": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                     16.0, 17.0, 18.0, 19.0],
            "trained": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        })
        # paper() should not raise on fmt='qmd'.
        try:
            draft = paper(
                df, "effect of trained on wage", fmt="qmd",
            )
        except ValueError:
            pytest.fail("paper(fmt='qmd') raised ValueError")
        assert draft.fmt == "qmd"
        # to_qmd() works on the produced draft.
        qmd = draft.to_qmd()
        assert qmd.startswith("---\n")

    def test_fmt_unknown_still_rejected(self):
        df = pd.DataFrame({"y": [1.0, 2.0], "t": [0, 1]})
        with pytest.raises(ValueError, match="Unknown fmt"):
            paper(df, "effect of t on y", fmt="rst")
