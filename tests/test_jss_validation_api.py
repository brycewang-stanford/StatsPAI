"""Tests for paper-facing validation meta APIs."""

from __future__ import annotations

import statspai as sp


def test_validation_report_summarizes_source_tree_evidence():
    report = sp.validation_report()

    assert report.registry["total_functions"] >= 900
    assert report.registry["total_categories"] > 10
    assert report.evidence["r_parity"]["matched_modules"] >= 30
    assert report.evidence["stata_parity"]["modules"] >= 20
    assert "jss_appendix_b" in report.artifacts
    assert "StatsPAI Validation Report" in report.to_markdown()


def test_validation_report_format_options():
    as_dict = sp.validation_report(fmt="dict")
    as_markdown = sp.validation_report(fmt="markdown")

    assert as_dict["registry"]["total_functions"] >= 900
    assert as_markdown.startswith("# StatsPAI Validation Report")


def test_coverage_matrix_category_and_parity_levels():
    category_rows = sp.coverage_matrix(fmt="records")
    parity_rows = sp.coverage_matrix(level="parity", fmt="records")

    assert any(row["category"] == "causal" for row in category_rows)
    assert any(row["r_parity_modules"] >= 1 for row in category_rows)
    assert len(parity_rows) >= 30
    assert parity_rows[0]["schema_registered"] is True
    assert parity_rows[0]["has_r_parity"] is True


def test_coverage_matrix_markdown_output():
    markdown = sp.coverage_matrix(level="parity", fmt="markdown")

    assert "module_id" in markdown
    assert "has_r_parity" in markdown


def test_reproduce_jss_tables_dry_run_core_plan():
    result = sp.reproduce_jss_tables(targets="core", dry_run=True)

    assert result.success is True
    assert result.dry_run is True
    assert result.targets == ["parity", "appendices", "inventory"]
    assert [step.name for step in result.steps] == [
        "r_parity_compare",
        "copy_appendix_b_parity",
        "gen_appendix_A",
        "gen_appendix_C",
        "generate_inventory",
    ]
    assert "StatsPAI JSS Table Reproduction" in result.to_markdown()
