"""Tests for P1-C: sp.paper(data, question) end-to-end pipeline.

Covers question parsing, the full data → markdown / TeX / file pipeline,
DiD/RD path detection, and graceful behaviour when the pipeline must
infer the design.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.workflow.causal_workflow import CausalWorkflow
from statspai.workflow.paper import (
    parse_question, _eda_block, _md_to_tex, _tex_escape,
)


# --------------------------------------------------------------------- #
#  Question parser
# --------------------------------------------------------------------- #


def test_parse_question_extracts_treatment_outcome():
    cols = ["wage", "trained", "edu", "experience"]
    out = parse_question("effect of trained on wage", cols)
    assert out["treatment"] == "trained"
    assert out["y"] == "wage"


def test_parse_question_handles_tilde_formula():
    cols = ["y", "x1", "x2"]
    out = parse_question("y ~ x1", cols)
    assert out["y"] == "y"
    assert out["treatment"] == "x1"


def test_parse_question_design_hints():
    cols = ["wage", "trained"]
    out = parse_question("difference-in-differences study", cols)
    assert out.get("design") == "did"

    out = parse_question("regression discontinuity at the cutoff", ["x"])
    assert out.get("design") == "rd"

    out = parse_question("randomised experiment", ["x"])
    assert out.get("design") == "rct"


def test_parse_question_picks_up_instrument():
    cols = ["wage", "edu", "distance"]
    out = parse_question(
        "estimate the return to edu using distance as an instrument",
        cols,
    )
    assert out["instrument"] == "distance"
    assert out["design"] == "iv"


def test_parse_question_picks_up_running_var_and_cutoff():
    cols = ["pass", "score"]
    out = parse_question(
        "regression discontinuity with running variable score and "
        "threshold at 50", cols,
    )
    assert out["running_var"] == "score"
    assert out["cutoff"] == 50.0
    assert out["design"] == "rd"


def test_parse_question_ignores_unknown_columns():
    out = parse_question("effect of treatment on income", [])
    # Neither column exists — only `raw_question` should be present.
    assert "treatment" not in out
    assert "y" not in out


def test_parse_question_empty_string():
    out = parse_question("", ["a", "b"])
    assert out == {"raw_question": ""}


# --------------------------------------------------------------------- #
#  paper() end-to-end
# --------------------------------------------------------------------- #


def _make_observational_df(seed: int = 42, n: int = 600):
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    treat = (0.5 * x1 + 0.3 * x2 + rng.standard_normal(n) > 0).astype(int)
    wage = (1.2 * treat + 0.4 * x1 + 0.3 * x2
            + rng.standard_normal(n))
    return pd.DataFrame({
        "wage": wage, "trained": treat, "edu": x1, "experience": x2,
    })


def test_paper_basic_observational():
    """End-to-end on simple observational data → all sections present."""
    df = _make_observational_df()
    draft = sp.paper(
        df, "effect of trained on wage",
        covariates=["edu", "experience"],
    )
    md = draft.to_markdown()
    for header in ("## Question", "## Data", "## Identification",
                   "## Estimator", "## Results", "## Robustness",
                   "## References"):
        assert header in md, f"Missing section: {header}"
    # Numerical sanity: estimate should be roughly in [0.8, 1.6] of true 1.2
    if hasattr(draft.workflow.result, "estimate"):
        est = float(draft.workflow.result.estimate)
        assert 0.5 < est < 1.7, f"Estimate {est!r} far from truth"
    elif hasattr(draft.workflow.result, "params"):
        est = float(draft.workflow.result.params["trained"])
        assert 0.5 < est < 1.7


def test_paper_did_pipeline():
    """DiD synthetic data → draft auto-picks DiD design."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(80):
        treated = i < 40
        for t in (0, 1):
            post = int(t > 0)
            y = (1.0 + 0.2 * t
                 + (0.5 if (treated and post) else 0.0)
                 + rng.standard_normal() * 0.3)
            rows.append({"id": i, "year": t, "wage": y,
                         "trained": int(treated and post)})
    df = pd.DataFrame(rows)
    draft = sp.paper(
        df, "difference-in-differences effect of trained on wage",
        y="wage", treatment="trained", time="year", id="id",
    )
    md = draft.to_markdown()
    assert draft.workflow.design == "did"
    # Estimator section should mention DID
    assert "DID" in md or "did" in md or "Difference" in md


def test_paper_question_parser_fills_y_and_treatment():
    """When y/treatment not passed, parser fills them from question."""
    df = _make_observational_df()
    draft = sp.paper(df, "effect of trained on wage")
    assert draft.parsed_hints.get("y") == "wage"
    assert draft.parsed_hints.get("treatment") == "trained"


def test_paper_explicit_args_override_parser():
    df = _make_observational_df()
    # Question has no useful info; explicit kwargs should be used.
    draft = sp.paper(df, "estimate the model", y="wage", treatment="trained")
    md = draft.to_markdown()
    assert "wage" in md
    assert "trained" in md


def test_paper_missing_outcome_raises():
    df = _make_observational_df()
    with pytest.raises(ValueError, match="outcome"):
        sp.paper(df, "(no question)")


def test_paper_tex_renders_with_sections_and_estimate():
    df = _make_observational_df()
    draft = sp.paper(df, "effect of trained on wage",
                     covariates=["edu", "experience"])
    tex = draft.to_tex()
    assert tex.startswith("\\documentclass")
    assert "\\section{Question}" in tex
    assert "\\section{Results}" in tex
    assert tex.rstrip().endswith("\\end{document}")


def test_paper_writes_to_disk_markdown(tmp_path):
    df = _make_observational_df()
    draft = sp.paper(df, "effect of trained on wage")
    out = tmp_path / "draft.md"
    draft.write(str(out))
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "## Question" in content


def test_paper_writes_to_disk_tex(tmp_path):
    df = _make_observational_df()
    draft = sp.paper(df, "effect of trained on wage")
    out = tmp_path / "draft.tex"
    draft.write(str(out))
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "\\documentclass" in content


def test_paper_output_path_via_kwarg(tmp_path):
    df = _make_observational_df()
    out = tmp_path / "auto.md"
    draft = sp.paper(df, "effect of trained on wage",
                     output_path=str(out))
    assert out.exists()
    assert isinstance(draft, sp.PaperDraft)


def test_paper_to_dict_round_trip():
    df = _make_observational_df()
    draft = sp.paper(df, "effect of trained on wage")
    d = draft.to_dict()
    assert d["question"] == "effect of trained on wage"
    assert "Question" in d["sections"]
    assert d["fmt"] in {"markdown", "tex", "docx"}


def test_paper_skip_eda_when_disabled():
    df = _make_observational_df()
    draft = sp.paper(df, "effect of trained on wage",
                     include_eda=False)
    assert "Data" not in draft.sections


def test_paper_include_robustness_false_skips_workflow_robustness(monkeypatch):
    df = _make_observational_df()

    def _should_not_run(self):
        raise AssertionError("workflow.robustness() should not run")

    monkeypatch.setattr(CausalWorkflow, "robustness", _should_not_run)
    draft = sp.paper(
        df, "effect of trained on wage",
        y="wage", treatment="trained",
        covariates=["edu", "experience"],
        design="observational",
        include_robustness=False,
    )
    assert "Robustness" not in draft.sections


def test_paper_stage_failure_surfaces_pipeline_notes(monkeypatch):
    df = _make_observational_df()

    def _boom(self):
        raise RuntimeError("forced recommend failure")

    monkeypatch.setattr(CausalWorkflow, "recommend", _boom)
    draft = sp.paper(
        df, "effect of trained on wage",
        y="wage", treatment="trained", design="observational",
    )
    assert "Pipeline notes" in draft.sections
    assert "recommend" in draft.sections["Pipeline notes"]
    assert "No fitted result available." in draft.sections["Results"]


def test_paper_handles_design_auto_detect_when_unspecified():
    """When design not passed and not in question, sp.causal should
    auto-detect (observational) without crashing."""
    df = _make_observational_df()
    draft = sp.paper(df, "what is the effect of trained on wage?")
    assert draft.workflow.design is not None


def test_paper_invalid_fmt_raises():
    df = _make_observational_df()
    with pytest.raises(ValueError, match="fmt"):
        sp.paper(df, "effect of trained on wage", fmt="pdf")


# --------------------------------------------------------------------- #
#  EDA block
# --------------------------------------------------------------------- #


def test_eda_block_reports_sample_size():
    df = _make_observational_df()
    block = _eda_block(df, "wage", "trained", ["edu", "experience"])
    assert "Sample size" in block
    assert "600" in block


def test_eda_block_handles_missingness():
    df = _make_observational_df()
    df.loc[:50, "edu"] = np.nan
    block = _eda_block(df, "wage", "trained", ["edu", "experience"])
    assert "Missingness" in block


# --------------------------------------------------------------------- #
#  Markdown → TeX inline conversion
# --------------------------------------------------------------------- #


def test_md_to_tex_handles_bold_and_lists():
    md = "Some **bold** text\n\n- item one\n- item two with `code`"
    tex = _md_to_tex(md)
    assert r"\textbf{bold}" in tex
    assert r"\begin{itemize}" in tex
    assert r"\end{itemize}" in tex
    assert r"\texttt{code}" in tex


def test_md_to_tex_handles_code_fence():
    md = "Plain line\n\n```\ncode_line()\n```\n\nAfter"
    tex = _md_to_tex(md)
    assert r"\begin{verbatim}" in tex
    assert "code_line()" in tex


def test_tex_escape_handles_specials():
    s = _tex_escape("100% & $50_value")
    assert r"\%" in s
    assert r"\&" in s
    assert r"\$" in s
    assert r"\_" in s


# --------------------------------------------------------------------- #
#  Registry / agent surface
# --------------------------------------------------------------------- #


def test_paper_registered():
    assert "paper" in sp.list_functions()
    spec = sp.describe_function("paper")
    assert spec["category"] == "workflow"
    # Spec should include agent-native fields.
    assert "assumptions" in spec
    assert "failure_modes" in spec


def test_paperdraft_exposed_at_top_level():
    assert hasattr(sp, "PaperDraft")
    assert hasattr(sp, "paper")
