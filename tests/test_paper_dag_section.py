"""Tests for the Causal DAG appendix in PaperDraft.

When the user passes ``dag=`` to ``sp.paper(...)`` (or
``CausalQuestion.paper(dag=...)``), the draft gains a *Causal DAG*
section. The section renders fmt-aware:

- ``markdown`` / ``tex`` — text-art (variables list + edges + adjustment
  sets + back-door paths + bad controls).
- ``qmd`` — Quarto-native ``mermaid`` code block + the same text
  fallbacks below it.

Coverage
--------
- ``_render_dag_section()`` direct: edges, adjustment sets, latent
  detection, fmt='qmd' emits mermaid.
- ``sp.paper(..., dag=g)`` populates ``sections['Causal DAG']``.
- ``CausalQuestion.paper(dag=g)`` populates ``sections['Causal DAG']``.
- ``draft.to_qmd()`` regenerates the section with mermaid when
  ``draft.dag`` is set, even if the markdown body is text-art.
- DAG with no treatment/outcome match still renders variables + edges.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.dag.graph import DAG
from statspai.workflow._degradation import WorkflowDegradedWarning
from statspai.workflow.paper import _render_dag_section


# ---------------------------------------------------------------------------
# _render_dag_section direct
# ---------------------------------------------------------------------------

class TestRenderDagSection:
    def test_basic_markdown(self):
        g = DAG("Z -> X; Z -> Y; X -> Y")
        out = _render_dag_section(g, treatment="X", outcome="Y",
                                   fmt="markdown")
        # Variables + edges + adjustment sets present.
        assert "**Variables**" in out
        assert "`X`" in out and "`Y`" in out and "`Z`" in out
        assert "`Z` → `X`" in out
        assert "**Adjustment sets**" in out
        assert "`Z`" in out
        # No mermaid in markdown.
        assert "```{mermaid}" not in out

    def test_qmd_emits_mermaid(self):
        g = DAG("Z -> X; X -> Y")
        out = _render_dag_section(g, treatment="X", outcome="Y", fmt="qmd")
        assert "```{mermaid}" in out
        assert "graph LR" in out
        assert "Z --> X" in out
        assert "X --> Y" in out
        # Closing fence for mermaid block.
        assert out.count("```") >= 2

    def test_renders_latent_confounders(self):
        g = DAG("Z -> X; X -> Y; U <-> Y")
        out = _render_dag_section(g, treatment="X", outcome="Y",
                                   fmt="markdown")
        assert "**Latent common causes**" in out
        # Latent node syntax `_L_U_Y` is what statspai DAG generates.
        assert "_L_U_Y" in out

    def test_backdoor_paths_listed(self):
        g = DAG("Z -> X; Z -> Y; X -> Y")
        out = _render_dag_section(g, treatment="X", outcome="Y",
                                   fmt="markdown")
        assert "**Back-door paths**" in out
        # X — Z — Y is the canonical path.
        assert "`X` — `Z` — `Y`" in out or "`X` — `Z` — `Y`" in out.replace("·", "—")

    def test_no_treatment_outcome_still_renders_basics(self):
        g = DAG("X -> Y")
        out = _render_dag_section(g, fmt="markdown")
        assert "`X`" in out and "`Y`" in out
        assert "`X` → `Y`" in out
        # No adjustment-set section without treatment/outcome.
        assert "Adjustment sets" not in out

    def test_none_input_returns_empty(self):
        assert _render_dag_section(None) == ""

    def test_qmd_quotes_latent_node_ids(self):
        # Latent nodes start with ``_L_`` — mermaid is sensitive to
        # leading-underscore identifiers, so the renderer should quote
        # them.
        g = DAG("U <-> X; X -> Y")
        out = _render_dag_section(g, fmt="qmd")
        assert '"_L_U_X"' in out

    def test_subanalysis_failures_record_degradation_but_keep_section(self):
        class _ExplodingDag:
            nodes = {"X", "Y", "Z"}
            edges = [("Z", "X"), ("Z", "Y"), ("X", "Y")]

            def adjustment_sets(self, t, y):
                raise RuntimeError("adj failed")

            def backdoor_paths(self, t, y):
                raise RuntimeError("bd failed")

            def bad_controls(self, t, y):
                raise RuntimeError("bad failed")

        bag = []
        with pytest.warns(WorkflowDegradedWarning, match="adjustment_sets"):
            out = _render_dag_section(
                _ExplodingDag(),
                treatment="X",
                outcome="Y",
                fmt="markdown",
                degradations=bag,
            )

        assert "**Variables**" in out
        assert "**Edges**" in out
        assert "Adjustment sets" not in out
        assert "Back-door paths" not in out
        assert "Bad controls" not in out
        assert [d["section"] for d in bag] == [
            "DAG adjustment_sets sub-analysis",
            "DAG backdoor_paths sub-analysis",
            "DAG bad_controls sub-analysis",
        ]


# ---------------------------------------------------------------------------
# sp.paper(..., dag=g) integration
# ---------------------------------------------------------------------------

@pytest.fixture
def panel_df():
    rng = np.random.default_rng(0)
    n_units = 60
    rows = []
    for u in range(n_units):
        treated = u >= n_units // 2
        for t in (2018, 2019):
            post = int(t == 2019)
            rows.append({
                "id": u, "year": t,
                "wage": 10 + 0.5 * post * treated + rng.normal(scale=0.3),
                "trained": int(treated),
                "post": post,
            })
    return pd.DataFrame(rows)


class TestSpPaperDagIntegration:
    def test_dag_section_populated(self, panel_df):
        g = DAG("trained -> wage; edu -> wage; edu -> trained")
        draft = sp.paper(
            panel_df, "effect of trained on wage",
            treatment="trained", y="wage", dag=g,
        )
        assert "Causal DAG" in draft.sections
        body = draft.sections["Causal DAG"]
        assert "`trained`" in body
        assert "`wage`" in body
        # Identification section refers to the DAG appendix.
        # (We don't enforce this strictly — just check no regression.)

    def test_dag_persists_on_draft(self, panel_df):
        g = DAG("X -> Y")
        draft = sp.paper(
            panel_df, "effect of trained on wage",
            treatment="trained", y="wage", dag=g,
        )
        # The DAG is held on the draft for to_qmd() to re-render.
        assert draft.dag is g
        assert draft.dag_treatment == "trained"
        assert draft.dag_outcome == "wage"

    def test_to_qmd_emits_mermaid(self, panel_df):
        g = DAG("trained -> wage; edu -> trained; edu -> wage")
        draft = sp.paper(
            panel_df, "effect of trained on wage",
            treatment="trained", y="wage", dag=g, fmt="qmd",
        )
        qmd = draft.to_qmd()
        assert "## Causal DAG" in qmd
        assert "```{mermaid}" in qmd
        assert "graph LR" in qmd
        assert "trained --> wage" in qmd

    def test_no_dag_no_section(self, panel_df):
        draft = sp.paper(
            panel_df, "effect of trained on wage",
            treatment="trained", y="wage",
        )
        # No DAG passed → no Causal DAG section.
        assert "Causal DAG" not in draft.sections


# ---------------------------------------------------------------------------
# CausalQuestion.paper(dag=g) integration
# ---------------------------------------------------------------------------

class TestQuestionPaperDag:
    def test_dag_in_question_paper(self, panel_df):
        g = DAG("trained -> wage; edu -> wage")
        q = sp.causal_question(
            "trained", "wage", data=panel_df, design="rct",
        )
        draft = q.paper(dag=g)
        assert "Causal DAG" in draft.sections
        assert "`trained`" in draft.sections["Causal DAG"]
        # Identification section references the DAG appendix.
        assert (
            "Causal DAG" in draft.sections["Identification"]
            or "see appendix" in draft.sections["Identification"].lower()
        )

    def test_dag_persists_on_estimand_first_draft(self, panel_df):
        g = DAG("trained -> wage")
        q = sp.causal_question(
            "trained", "wage", data=panel_df, design="rct",
        )
        draft = q.paper(dag=g, fmt="qmd")
        qmd = draft.to_qmd()
        assert "```{mermaid}" in qmd
        assert "trained --> wage" in qmd
