"""Tests for paper-facing validation meta APIs."""

from __future__ import annotations

import importlib.util

import pytest

import statspai as sp

# The four ``sp.dml`` vs doubleml-for-py pins in
# ``tests/external_parity/test_dml_python_parity.py`` (PLR, IRM, PLIV, IIVM —
# one per DoubleML model class) are gated by
# ``pytest.importorskip("doubleml")`` — doubleml is the opt-in ``parity``
# extra, not part of ``dev``. Without it those 4 tests do not collect, so
# the JSS external-parity headline (54) is only reachable when the extra is
# installed (CI installs it on the canonical ubuntu+3.10 job).
_HAS_DOUBLEML = importlib.util.find_spec("doubleml") is not None


# JSS Section 5 (tab:internal-parity) headline test counts, frozen at the
# 1.16.0 manuscript snapshot. These are treated as *published floors*, not
# exact lockstep targets: the live suite is only ever allowed to GROW beyond
# what the manuscript claims (the drift-guard below asserts ``collected >=
# headline``). Growing the parity/coverage suites is therefore additive and
# does not require editing the frozen manuscript — more validated tests than
# the paper claims never falsifies the paper's count. Only a count that drops
# BELOW a published floor is a regression worth failing on. Do not lower these
# numbers without retiring the corresponding manuscript claim.
JSS_HEADLINE_TEST_COUNTS = {
    "reference_parity": 124,
    "external_parity": 54,
    "coverage_monte_carlo": 12,
}
# Published floor (1.16.0 manuscript), not an exact target — validation
# *grade* is derived dynamically from test evidence (VALIDATED_GRADE_MARKERS
# below), so adding reference-parity tests promotes more symbols to
# certified/validated. That growth is the intended effect of new parity work,
# never a regression; the guard asserts ``>=`` for the same reason as the
# headline-count drift guard above.
JSS_CERTIFIED_VALIDATED_SYMBOLS = 73
VALIDATED_GRADE_MARKERS = (
    "tests/reference_parity/",
    "tests/external_parity/",
    "coverage",
    "Monte Carlo",
    "known-truth",
    "known truth",
    "Track A parity seed",
    "documented gap",
    "certification",
)


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


def test_validation_report_registry_counts_are_self_consistent():
    """Registry totals in the validation report must reconcile exactly."""
    report = sp.validation_report(fmt="dict")
    registry = report["registry"]

    total = registry["total_functions"]
    assert sum(registry["per_category"].values()) == total
    assert sum(registry["per_stability"].values()) == total
    assert sum(registry["per_validation_status"].values()) == total
    assert registry["handwritten_specs"] + registry["auto_specs"] == total
    assert registry["functions_with_limitations"] <= total


def test_coverage_matrix_category_and_parity_levels():
    category_rows = sp.coverage_matrix(fmt="records")
    parity_rows = sp.coverage_matrix(level="parity", fmt="records")

    assert any(row["category"] == "causal" for row in category_rows)
    assert any(row["r_parity_modules"] >= 1 for row in category_rows)
    assert len(parity_rows) >= 30
    assert parity_rows[0]["schema_registered"] is True
    assert parity_rows[0]["has_r_parity"] is True


def test_coverage_matrix_reconciles_category_totals():
    """The category coverage matrix must be a lossless registry aggregation."""
    report = sp.validation_report(fmt="dict")
    rows = sp.coverage_matrix(fmt="records")
    total_registered = sum(row["registered_functions"] for row in rows)

    assert total_registered == report["registry"]["total_functions"]
    assert total_registered / report["registry"]["total_functions"] == pytest.approx(
        1.0
    )
    for row in rows:
        status_total = (
            row["stable_functions"]
            + row["experimental_functions"]
            + row["deprecated_functions"]
        )
        assert status_total == row["registered_functions"]
        assert (
            row["handwritten_specs"] + row["auto_specs"] == row["registered_functions"]
        )
        assert row["agent_cards"] <= row["registered_functions"]


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


def test_validation_result_containers_have_failure_contracts():
    """Validation dataclasses should round-trip and expose failure state."""
    report = sp.ValidationReport(
        generated_at="2026-01-01T00:00:00+00:00",
        version="1.18.0",
        repo_root=".",
        registry={"total_functions": 3, "total_categories": 2},
        evidence={
            "r_parity": {"matched_modules": 1},
            "stata_parity": {"modules": 1},
            "monte_carlo": {"runs": 0},
            "agent_bench": {"trials": 0},
        },
    )
    assert report.to_dict()["registry"]["total_functions"] == pytest.approx(3)
    assert "3 registered functions across 2 categories" in report.summary()
    assert "| Registered functions | 3 |" in report.to_markdown()

    ok_step = sp.ReproductionStep(
        name="ok",
        action="command",
        command=["python", "-V"],
        cwd=".",
        returncode=0,
        elapsed_s=0.25,
    )
    failed_step = sp.ReproductionStep(
        name="bad",
        action="command",
        command=["python", "missing.py"],
        cwd=".",
        returncode=2,
        elapsed_s=1.5,
        stderr_tail="missing.py",
    )
    skipped_step = sp.ReproductionStep(
        name="skip",
        action="copy",
        command=["a", "b"],
        cwd=".",
        skipped=True,
    )

    assert ok_step.ok is True
    assert failed_step.ok is False
    assert skipped_step.ok is True
    assert failed_step.to_dict()["elapsed_s"] == pytest.approx(1.5)

    result = sp.ReproductionResult(
        generated_at="2026-01-01T00:00:00+00:00",
        repo_root=".",
        targets=["inventory"],
        dry_run=False,
        success=False,
        steps=[ok_step, failed_step, skipped_step],
        artifacts={"tables": []},
    )
    assert result.failed_steps() == ["bad"]
    assert result.to_dict()["steps"][1]["stderr_tail"] == "missing.py"
    markdown = result.to_markdown()
    assert "| bad | command | 2 | 1.50 |" in markdown
    assert "| skip | copy | skipped | 0.00 |" in markdown


def test_certified_functions_surface_variant_level_gaps():
    """Certified must not read as blanket exact parity for every option."""
    expected_fragments = {
        "rdrobust": "bwselect='cct'",
        "rddensity": "rdbwdensity combination",
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

    # Published floor, not exact lockstep: new reference-parity evidence only
    # ever grows this set (e.g. panel/poisson/nbreg/qreg/tobit/zip/zinb/sdid
    # were promoted to validated when their parity tests were added). A count
    # that dropped BELOW the floor would mean evidence was lost — that is the
    # regression worth failing on.
    assert len(names) >= JSS_CERTIFIED_VALIDATED_SYMBOLS

    missing_notes = []
    certified_without_grade = []
    validated_without_grade = []
    for name in names:
        spec = sp.describe_function(name)
        notes = spec.get("validation_notes", [])
        if not notes:
            missing_notes.append(name)
        if spec.get("validation_status") == "certified" and not any(
            "R parity module" in note or "Stata parity module" in note for note in notes
        ):
            certified_without_grade.append(name)
        if spec.get("validation_status") == "validated" and not any(
            marker in note for note in notes for marker in VALIDATED_GRADE_MARKERS
        ):
            validated_without_grade.append(name)

    assert not missing_notes
    assert not certified_without_grade
    assert not validated_without_grade


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
    """``validation_report(collect_tests=True)`` must reproduce *at least* the
    pytest --collect-only counts that the JSS manuscript headlines, so the
    paper's "headline counts are not hand-copied" claim stays script-verifiable.

    The published headline counts are treated as floors, not exact targets:
    the live parity/coverage suites are allowed to grow beyond the frozen
    1.16.0 manuscript snapshot (additive validation work should never be
    blocked by, nor silently rewrite, an in-print paper's numbers). This guard
    therefore fails only on a *regression* — a collected count that has dropped
    below a published floor.
    """
    report = sp.validation_report(collect_tests=True, fmt="dict")
    collected = report["evidence"]["pytest_inventory"].get("collected")
    assert collected is not None
    for key, expected in JSS_HEADLINE_TEST_COUNTS.items():
        actual = collected.get(key)
        if actual is None:
            pytest.skip(f"pytest --collect-only unavailable for {key}")
        if key == "external_parity" and not _HAS_DOUBLEML:
            # The manuscript's external-parity count includes the 4
            # doubleml-gated pins; without the optional `parity` extra they
            # don't collect, so verifying the floor here would be a spurious
            # failure. CI installs `.[parity,rd-cct]` on the canonical env to
            # check the full count and to run the CCT RD pins; skip it in
            # minimal environments instead.
            continue
        assert actual >= expected, (
            f"{key}: collected {actual} tests, which is BELOW the JSS "
            f"manuscript floor of {expected}. A parity/coverage test was "
            f"removed or failed to collect — restore it, or retire the "
            f"corresponding manuscript claim in tab:internal-parity."
        )
