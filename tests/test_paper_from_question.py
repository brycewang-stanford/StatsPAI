"""Tests for the estimand-first paper path: ``CausalQuestion.paper()``
and the ``sp.paper(causal_question_obj)`` dispatch.

The estimand-first entry point declares Treatment / Outcome /
Population / Estimand / Design up front, then assembles a paper whose
sections match what was pre-registered. These tests cover:

- ``CausalQuestion.paper()`` produces a ``PaperDraft`` with the right
  sections and routes through ``identify()`` + ``estimate()``.
- ``sp.paper(question_obj)`` dispatches to the same path.
- Provenance is auto-attached to ``result.underlying`` so
  ``draft.to_qmd()`` emits the ``statspai:`` YAML block + Reproducibility
  appendix.
- Errors fire when ``data`` is missing.
- Non-default formats (``qmd`` / ``tex``) work end-to-end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import importlib

import statspai as sp
from statspai.workflow.paper import paper_from_question, PaperDraft


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def panel_df():
    """Two-period panel with a clean DiD signal."""
    rng = np.random.default_rng(0)
    n_units = 60
    rows = []
    for u in range(n_units):
        treated = u >= n_units // 2
        for t in (2018, 2019):
            post = int(t == 2019)
            base = rng.normal(loc=10.0, scale=1.0)
            te = 0.5 * post * treated  # treatment effect after 2019
            rows.append({
                "id": u,
                "year": t,
                "wage": base + te + rng.normal(scale=0.3),
                "trained": int(treated),
                "edu": rng.normal(),
                "post": post,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CausalQuestion.paper()
# ---------------------------------------------------------------------------

class TestQuestionPaperMethod:
    def test_returns_paper_draft(self, panel_df):
        q = sp.causal_question(
            "trained", "wage", data=panel_df,
            design="rct", estimand="ATE",
        )
        draft = q.paper()
        assert isinstance(draft, PaperDraft)
        # All canonical sections present.
        for s in ("Question", "Data", "Identification",
                  "Estimator", "Results", "Robustness", "References"):
            assert s in draft.sections, f"missing {s!r}"

    def test_question_section_carries_declaration(self, panel_df):
        q = sp.causal_question(
            "trained", "wage", data=panel_df,
            population="manufacturing workers, 2018-2019",
            design="rct", estimand="ATT",
            covariates=["edu"],
            notes="Pre-registered 2026-04-27.",
        )
        draft = q.paper()
        body = draft.sections["Question"]
        assert "trained" in body and "wage" in body
        assert "manufacturing workers" in body
        assert "ATT" in body
        assert "edu" in body
        assert "Pre-registered" in body

    def test_results_section_has_estimate_se_ci(self, panel_df):
        q = sp.causal_question(
            "trained", "wage", data=panel_df,
            design="rct", estimand="ATE",
        )
        draft = q.paper()
        body = draft.sections["Results"]
        # Format: "**ATE** (via `sp.X`): **+0.XXXX** (SE = 0.XXXX)"
        assert "SE = " in body
        assert "95% CI" in body
        assert "N obs" in body

    def test_identification_lists_assumptions(self, panel_df):
        q = sp.causal_question(
            "trained", "wage", data=panel_df,
            design="rct", estimand="ATE",
        )
        draft = q.paper()
        body = draft.sections["Identification"]
        # IdentificationPlan.assumptions are bulleted.
        assert "Required assumptions" in body
        assert "- " in body  # at least one bullet


# ---------------------------------------------------------------------------
# sp.paper(question_obj) dispatch
# ---------------------------------------------------------------------------

class TestSpPaperDispatch:
    def test_dispatch_on_causal_question(self, panel_df):
        q = sp.causal_question(
            "trained", "wage", data=panel_df,
            design="rct", estimand="ATE",
        )
        draft = sp.paper(q)
        assert isinstance(draft, PaperDraft)
        # The dispatch path generates a synthetic question string from
        # the declaration.
        assert "trained" in draft.question and "wage" in draft.question

    def test_dispatch_with_explicit_question_text(self, panel_df):
        q = sp.causal_question(
            "trained", "wage", data=panel_df, design="rct",
        )
        draft = sp.paper(q, question="my custom question")
        assert "my custom question" in draft.sections["Question"]

    def test_dispatch_qmd_format(self, panel_df):
        q = sp.causal_question(
            "trained", "wage", data=panel_df, design="rct",
        )
        draft = sp.paper(q, fmt="qmd")
        assert draft.fmt == "qmd"
        qmd = draft.to_qmd()
        assert qmd.startswith("---\n")
        # The synthetic question makes its way into the YAML subtitle.
        assert "trained" in qmd

    def test_dispatch_tex_format(self, panel_df):
        q = sp.causal_question(
            "trained", "wage", data=panel_df, design="rct",
        )
        draft = sp.paper(q, fmt="tex")
        tex = draft.to_tex()
        assert tex.startswith("\\documentclass")
        assert "Question" in tex


# ---------------------------------------------------------------------------
# Provenance integration
# ---------------------------------------------------------------------------

class TestProvenanceFromQuestion:
    def test_provenance_attached_to_underlying(self, panel_df):
        q = sp.causal_question(
            "trained", "wage", data=panel_df, design="rct",
        )
        draft = q.paper()
        # The lightweight workflow adapter exposes the underlying
        # estimator's result on .result; provenance lives there.
        result = draft.workflow.result
        prov = sp.get_provenance(result)
        assert prov is not None
        # When the underlying estimator is itself instrumented (Phase
        # 3.2: sp.regress / sp.callaway_santanna / sp.did_2x2 / sp.iv),
        # its more-specific provenance record wins over the
        # causal_question wrapper (overwrite=False semantics). Either is
        # acceptable here.
        assert (
            prov.function.startswith("sp.causal_question[")
            or prov.function in {"sp.regress", "sp.did.callaway_santanna",
                                  "sp.did.did_2x2", "sp.iv"}
        )
        assert prov.data_hash  # data was hashed

    def test_qmd_includes_provenance_appendix(self, panel_df):
        q = sp.causal_question(
            "trained", "wage", data=panel_df, design="rct",
        )
        draft = q.paper(fmt="qmd")
        qmd = draft.to_qmd()
        assert "statspai:" in qmd
        assert "## Reproducibility" in qmd

    def test_replication_pack_picks_up_lineage(self, panel_df, tmp_path):
        q = sp.causal_question(
            "trained", "wage", data=panel_df, design="rct",
        )
        draft = q.paper(fmt="qmd")
        rp = sp.replication_pack(
            draft, tmp_path / "estimand_first.zip",
            env=False, paper_format="qmd",
        )
        # The pack must include lineage.json (auto-collected from the
        # underlying estimator's _provenance via workflow.result).
        import zipfile
        import json
        with zipfile.ZipFile(rp.output_path) as zf:
            assert "lineage.json" in zf.namelist()
            lineage = json.loads(zf.read("lineage.json"))
            assert lineage["n_runs"] >= 1


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestErrors:
    def test_question_without_data_raises(self):
        q = sp.causal_question("trained", "wage", design="rct")
        with pytest.raises(ValueError, match="data must be set"):
            q.paper()

    def test_paper_from_question_function_validates(self, panel_df):
        q = sp.causal_question("trained", "wage", design="rct")
        # No data on the question — should fail.
        with pytest.raises(ValueError, match="data"):
            paper_from_question(q)

    def test_unknown_format_rejected(self, panel_df):
        q = sp.causal_question(
            "trained", "wage", data=panel_df, design="rct",
        )
        with pytest.raises(ValueError, match="Unknown fmt"):
            paper_from_question(q, fmt="rst")


class TestDegradations:
    def test_dag_render_failure_surfaces_in_draft(self, monkeypatch, panel_df):
        q = sp.causal_question(
            "trained", "wage", data=panel_df, design="rct",
        )
        paper_module = importlib.import_module("statspai.workflow.paper")

        def _boom(*args, **kwargs):
            raise RuntimeError("forced dag failure")

        monkeypatch.setattr(paper_module, "_render_dag_section", _boom)
        draft = q.paper(dag=object())
        assert draft.degradations
        assert "Pipeline notes" in draft.sections
        assert "forced dag failure" in draft.sections["Pipeline notes"]
