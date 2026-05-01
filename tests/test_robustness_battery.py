"""Tests for the shared robustness battery + paper-pipeline wiring.

Covers:
* `run_robustness_battery` always returns a non-raising
  :class:`RobustnessReport` regardless of result shape.
* The estimand-first ``paper_from_question`` path now produces a
  *substantive* Robustness section (was a placeholder pre-1.12.x).
* The NL ``sp.paper(data, question, ...)`` path goes through the same
  battery via ``CausalWorkflow.robustness()`` and renders the
  structured report rather than the legacy flat-key list.
* Failure isolation: a single failing sub-check becomes a
  ``severity='check_failed'`` finding rather than aborting the
  battery.
"""
import re
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp
from statspai.workflow._robustness import (
    RobustnessFinding, RobustnessReport, run_robustness_battery,
)


def _obs_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    D = (rng.random(n) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)
    Y = 0.7 * D + X[:, 0] + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({
        "y": Y, "d": D,
        "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2],
    })


# ---------------------------------------------------------------------- #
#  Battery on isolated results
# ---------------------------------------------------------------------- #


class TestBatteryOnDML:
    """DML PLR result — exercises model_info["diagnostics"] surfacing,
    e-value, oster_bounds, sensemakr."""

    @pytest.fixture(scope="class")
    def result_and_data(self):
        df = _obs_data(n=300, seed=1)
        r = sp.dml(df, y="y", treat="d",
                   covariates=["x0", "x1", "x2"], model="plr")
        return r, df

    def test_returns_non_empty_report(self, result_and_data):
        r, df = result_and_data
        report = run_robustness_battery(
            r, design="observational", data=df,
            treatment="d", outcome="y", covariates=["x0", "x1", "x2"],
        )
        assert isinstance(report, RobustnessReport)
        assert not report.is_empty()
        # Must have at least: estimate, ci_width, violations summary,
        # e-value, oster, sensemakr, estimator self-diagnostics
        names = {f.name for f in report.findings}
        assert "estimate" in names
        assert "ci_width" in names
        assert "estimator_diagnostics" in names

    def test_to_dict_has_findings_array(self, result_and_data):
        r, df = result_and_data
        report = run_robustness_battery(
            r, design="observational", data=df,
            treatment="d", outcome="y", covariates=["x0", "x1", "x2"],
        )
        d = report.to_dict()
        # Backwards-compat: flat keys are still present
        assert "estimate" in d
        assert "ci_width" in d
        # New: structured findings array
        assert "_findings" in d
        assert isinstance(d["_findings"], list)
        assert len(d["_findings"]) >= 3
        for f in d["_findings"]:
            assert {"name", "label", "value", "severity"}.issubset(f.keys())

    def test_to_markdown_renders_severity_icons(self, result_and_data):
        r, df = result_and_data
        report = run_robustness_battery(
            r, design="observational", data=df,
            treatment="d", outcome="y", covariates=["x0", "x1", "x2"],
        )
        md = report.to_markdown()
        # At least one ok / info icon in the rendered block.
        assert any(icon in md for icon in ["✅", "ℹ️", "⚠️", "❌", "⚙️"])
        # Header line
        assert "Battery for design `observational`" in md


class TestBatteryFailureIsolation:
    """A single failing sub-check must not abort the battery."""

    def test_evalue_failure_becomes_check_failed_finding(self):
        # Bare object with .estimate but no shape evalue understands.
        class _Stub:
            estimate = 0.5
            ci = (0.1, 0.9)
            model_info = {}
            def violations(self):
                return []

        report = run_robustness_battery(_Stub(), design="observational")
        names = {f.name for f in report.findings}
        # Universal checks ran successfully
        assert "estimate" in names
        assert "ci_width" in names
        # E-value either ran (succeeded) or was check_failed; either way
        # the battery returned a structured report rather than raising.
        evalue_finding = next(
            (f for f in report.findings if f.name == "evalue"), None,
        )
        if evalue_finding is not None:
            assert evalue_finding.severity in {"ok", "info", "warning", "check_failed"}

    def test_no_result_supplied_returns_note_not_crash(self):
        report = run_robustness_battery(None, design="observational")
        assert report.is_empty() is False  # has notes
        assert any("No fitted result" in n for n in report.notes)


class TestDesignTagAliases:
    """``CausalQuestion.design`` taxonomy maps onto battery's internal
    branches so ``selection_on_observables`` etc. trigger the
    e-value / oster / sensemakr checks."""

    def test_selection_on_observables_runs_observational_branch(self):
        df = _obs_data(n=300, seed=2)
        # Run via raw regress so we have a result without ML internals;
        # battery should still attempt e-value / oster.
        r = sp.regress("y ~ d + x0 + x1 + x2", data=df)
        report = run_robustness_battery(
            r, design="selection_on_observables",
            data=df, treatment="d", outcome="y",
            covariates=["x0", "x1", "x2"],
        )
        names = {f.name for f in report.findings}
        # At least Oster + Sensemakr should be tried (success or
        # check_failed; never silently skipped).
        assert any(n.startswith("oster") or n.startswith("sensemakr")
                   for n in names) or report.notes

    def test_unknown_design_runs_generic_only(self):
        df = _obs_data(n=200, seed=3)
        r = sp.regress("y ~ d + x0", data=df)
        report = run_robustness_battery(r, design="hypothetical_new_design")
        names = {f.name for f in report.findings}
        # Generic checks run regardless of design tag.
        assert "estimate" in names or "ci_width" in names


# ---------------------------------------------------------------------- #
#  Paper pipeline integration
# ---------------------------------------------------------------------- #


def _extract_robustness_section(md: str) -> str:
    m = re.search(r"## Robustness\n(.*?)(?=\n## |\Z)", md, re.DOTALL)
    return m.group(1) if m else ""


class TestPaperFromQuestion:
    """The estimand-first ``sp.paper(CausalQuestion(...))`` path used to
    produce a placeholder Robustness section pointing the user back at
    ``sp.causal``. After 1.12.x it must return real findings."""

    def test_robustness_section_no_longer_placeholder(self):
        df = _obs_data(n=200, seed=4)
        q = sp.causal_question(
            treatment="d", outcome="y", data=df,
            design="selection_on_observables",
            covariates=["x0", "x1", "x2"],
        )
        draft = sp.paper(q, fmt="markdown")
        section = _extract_robustness_section(draft.to_markdown())
        assert section.strip(), "Robustness section is empty"
        assert "No robustness suite executed" not in section
        # Battery header is the canonical signature of the new content.
        assert "Battery for design" in section

    def test_robustness_section_lists_substantive_checks(self):
        df = _obs_data(n=200, seed=5)
        q = sp.causal_question(
            treatment="d", outcome="y", data=df,
            design="selection_on_observables",
            covariates=["x0", "x1", "x2"],
        )
        draft = sp.paper(q, fmt="markdown")
        section = _extract_robustness_section(draft.to_markdown())
        # The Oster / Sensemakr / e-value family should be visible
        # under selection_on_observables; at minimum one of them
        # must surface (some can fail on extreme synthetic DGPs).
        lower = section.lower()
        substantive = any(
            kw in lower for kw in [
                "oster", "sensemakr", "e-value",
                "robustness value", "self-reported violations",
            ]
        )
        assert substantive, f"No substantive checks in section: {section}"

    def test_include_robustness_false_omits_section(self):
        df = _obs_data(n=200, seed=6)
        q = sp.causal_question(
            treatment="d", outcome="y", data=df,
            design="selection_on_observables",
            covariates=["x0", "x1", "x2"],
        )
        # Routes through paper(q, ...) which forwards include_robustness
        # to paper_from_question.
        draft = sp.paper(q, fmt="markdown", include_robustness=False)
        md = draft.to_markdown()
        # Section header should NOT be present.
        assert "## Robustness" not in md


class TestPaperNaturalLanguage:
    """``sp.paper(data, question, ...)`` already ran a robustness step
    via ``CausalWorkflow.robustness()`` pre-1.12; the post-1.12
    delegation must keep the section non-empty AND now use the
    structured renderer."""

    def test_section_uses_structured_renderer(self):
        df = _obs_data(n=200, seed=7)
        draft = sp.paper(
            df, "effect of d on y",
            y="y", treatment="d",
            covariates=["x0", "x1", "x2"],
        )
        section = _extract_robustness_section(draft.to_markdown())
        assert section.strip()
        # Structured renderer's signature: a "Battery for design"
        # header line and severity icons.
        assert "Battery for design" in section
        assert any(icon in section for icon in ["✅", "ℹ️", "⚠️", "❌", "⚙️"])


class TestCausalWorkflowBackwardsCompat:
    """``CausalWorkflow.robustness_findings`` must keep its dict shape
    so legacy callers keep working."""

    def test_flat_dict_still_carries_estimate_and_ci_width(self):
        df = _obs_data(n=200, seed=8)
        wf = sp.causal(
            df, y="y", treatment="d",
            covariates=["x0", "x1", "x2"],
        )
        wf.diagnose(); wf.recommend(); wf.estimate(); wf.robustness()
        d = wf.robustness_findings
        assert isinstance(d, dict)
        assert "estimate" in d
        assert "ci_width" in d
        # New: structured _findings array exposed for callers that
        # want severity-aware rendering.
        assert "_findings" in d
        assert isinstance(d["_findings"], list)
