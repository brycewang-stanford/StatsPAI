"""Tests for paper-facing validation meta APIs."""

from __future__ import annotations

import pytest

import statspai as sp


# JSS Section 5 (tab:internal-parity) headline test counts. If you add or
# remove a parity / coverage test, update BOTH this constant and the
# manuscript in the same commit — that lockstep is the whole point of the
# drift-guard test below.
JSS_HEADLINE_TEST_COUNTS = {
    "reference_parity": 124,
    "external_parity": 50,
    "coverage_monte_carlo": 12,
}
JSS_CERTIFIED_VALIDATED_SYMBOLS = 244


def test_validation_report_summarizes_source_tree_evidence():
    report = sp.validation_report()

    assert report.registry["total_functions"] >= 900
    assert report.registry["total_categories"] > 10
    assert report.registry["per_validation_status"]["certified"] >= 30
    assert report.evidence["r_parity"]["matched_modules"] >= 30
    assert report.evidence["stata_parity"]["modules"] >= 20
    assert report.evidence["parity_gaps"]["rows"] >= 1
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


def test_parity_gap_report_surfaces_open_gaps():
    rows = sp.parity_gap_report(fmt="records")
    assert rows
    assert any(row["kind"] == "documented_gap" for row in rows)
    assert any("next_action" in row for row in rows)
    md = sp.parity_gap_report(fmt="markdown")
    assert "next_action" in md


def test_certified_functions_surface_variant_level_gaps():
    """Certified must not read as blanket exact parity for every option."""
    expected_fragments = {
        "rdrobust": "bwselect='cct'",
        "rddensity": "bandwidth selector",
        "synth": "special_predictors",
        "causal_forest": "overlap",
        "did_imputation": "aggregation",
        "etwfe": "aggregation",
    }
    for name, fragment in expected_fragments.items():
        spec = sp.describe_function(name)
        text = " ".join(spec.get("limitations", []) + spec.get("validation_notes", []))
        assert fragment in text, f"{name} does not expose {fragment!r}: {text}"


def test_certified_validated_symbols_have_attached_evidence_notes():
    """The JSS validated-core count must not include naked status flags."""
    certified = sp.list_functions(validation_status="certified")
    validated = sp.list_functions(validation_status="validated")
    names = sorted(set(certified) | set(validated))

    assert len(names) == JSS_CERTIFIED_VALIDATED_SYMBOLS

    missing_notes = []
    certified_without_grade = []
    for name in names:
        spec = sp.describe_function(name)
        notes = spec.get("validation_notes", [])
        if not notes:
            missing_notes.append(name)
        if spec.get("validation_status") == "certified" and not any(
            "R parity module" in note
            or "Stata parity module" in note
            for note in notes
        ):
            certified_without_grade.append(name)

    assert not missing_notes
    assert not certified_without_grade


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


def test_validation_report_collected_counts_match_jss_headline():
    """``validation_report(collect_tests=True)`` must reproduce the exact
    pytest --collect-only counts that the JSS manuscript headlines, so the
    paper's "headline counts are not hand-copied" claim is script-verifiable.

    This fails if a parity/coverage test is added or removed without updating
    ``JSS_HEADLINE_TEST_COUNTS`` (and the manuscript's tab:internal-parity)
    in lockstep — i.e. it is the count-drift guard.
    """
    report = sp.validation_report(collect_tests=True, fmt="dict")
    collected = report["evidence"]["pytest_inventory"].get("collected")
    assert collected is not None
    for key, expected in JSS_HEADLINE_TEST_COUNTS.items():
        actual = collected.get(key)
        if actual is None:
            pytest.skip(f"pytest --collect-only unavailable for {key}")
        assert actual == expected, (
            f"{key}: collected {actual} tests but the JSS manuscript headlines "
            f"{expected}. Update JSS_HEADLINE_TEST_COUNTS and the paper's "
            f"tab:internal-parity in the same commit."
        )
