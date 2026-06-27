"""Contract tests for the JSS source-snapshot release gate."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# ``Paper-JSS/`` is a git-ignored, local-only submission working directory, so
# these source-snapshot release-gate contract tests can only run where it is
# present (the maintainer's machine). On a clean checkout / CI they skip.
pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "Paper-JSS").exists(),
    reason="Paper-JSS/ submission working dir is git-ignored / local-only",
)

SOURCE_MANIFEST = (
    REPO_ROOT / "Paper-JSS" / "replication" / "scripts" / "source_snapshot_manifest.py"
)
RELEASE_AUDIT = (
    REPO_ROOT / "Paper-JSS" / "replication" / "scripts" / "release_boundary_audit.py"
)
CLAIM_LINT = REPO_ROOT / "Paper-JSS" / "replication" / "scripts" / "validate_claims.py"
GAP_LEDGER = (
    REPO_ROOT / "Paper-JSS" / "replication" / "scripts" / "methodological_gap_ledger.py"
)
VALIDATION_EVIDENCE_AUDIT = (
    REPO_ROOT / "Paper-JSS" / "replication" / "scripts" / "validation_evidence_audit.py"
)
VERIFY_PACKAGE = (
    REPO_ROOT / "Paper-JSS" / "replication" / "scripts" / "verify_submission_package.py"
)
PACKAGE_SCRIPT = (
    REPO_ROOT / "Paper-JSS" / "replication" / "scripts" / "jss_submission_package.py"
)
FULL_AUDIT = REPO_ROOT / "Paper-JSS" / "replication" / "scripts" / "jss_full_audit.py"
REGISTRY_STATS = REPO_ROOT / "scripts" / "registry_stats.py"
RESULTS = REPO_ROOT / "Paper-JSS" / "replication" / "results"
SUBMISSION_ARCHIVE = REPO_ROOT / "Paper-JSS" / "build" / "statspai-jss-submission.zip"
SUBMISSION_MANIFEST = (
    REPO_ROOT / "Paper-JSS" / "build" / "statspai-jss-submission-manifest.json"
)
SUBMISSION_ARCHIVE_INPUTS = (
    VERIFY_PACKAGE,
    PACKAGE_SCRIPT,
    SOURCE_MANIFEST,
    CLAIM_LINT,
    VALIDATION_EVIDENCE_AUDIT,
    REPO_ROOT / "Paper-JSS" / "README.md",
    REPO_ROOT / "Paper-JSS" / "cover-letter.md",
    REPO_ROOT / "Paper-JSS" / "REVIEWER-HARDENING-AUDIT.md",
    REPO_ROOT / "Paper-JSS" / "manuscript" / "README.md",
    REPO_ROOT / "Paper-JSS" / "manuscript" / "main.pdf",
    REPO_ROOT / "Paper-JSS" / "replication" / "results" / "jss_full_audit.md",
    REPO_ROOT / "Paper-JSS" / "replication" / "results" / "jss_full_audit.json",
    REPO_ROOT
    / "Paper-JSS"
    / "replication"
    / "results"
    / "jss_formal_compliance_audit.md",
    REPO_ROOT
    / "Paper-JSS"
    / "replication"
    / "results"
    / "jss_formal_compliance_audit.json",
    REPO_ROOT
    / "Paper-JSS"
    / "replication"
    / "scripts"
    / "jss_formal_compliance_audit.py",
    REPO_ROOT
    / "Paper-JSS"
    / "replication"
    / "results"
    / "manuscript_artifact_audit.md",
    REPO_ROOT
    / "Paper-JSS"
    / "replication"
    / "results"
    / "manuscript_artifact_audit.json",
    REPO_ROOT
    / "Paper-JSS"
    / "replication"
    / "scripts"
    / "manuscript_artifact_audit.py",
    REPO_ROOT / "CITATION.cff",
    REPO_ROOT / "docs" / "getting-started.md",
    REPO_ROOT / "docs" / "reference" / "index.md",
    REPO_ROOT / "tests" / "test_jss_release_manifest.py",
)
LEGACY_JOURNAL_REVIEW_PATHS = (
    REPO_ROOT / "docs" / ("jo" "ss_reviewer_guide.md"),
    REPO_ROOT / "docs" / ("jo" "ss_validation_dossier.md"),
    REPO_ROOT / "tests" / ("test_jo" "ss_reviewer_followups.py"),
)
ACTIVE_EXTERNAL_REVIEW_PATHS = {
    "paper.md",
    "docs/jo" "ss_reviewer_guide.md",
    "docs/jo" "ss_validation_dossier.md",
}
LEGACY_REVIEW_PATH_REDACTIONS = {
    (
        "tests/test_jo" "ss_reviewer_followups.py"
    ): "tests/retired-external-reviewer-followups.py",
}
ACTIVE_REVIEW_COMPAT_TEST_PATHS = {
    REPO_ROOT / "tests" / ("test_jo" "ss_reviewer_followups.py"),
}
COVERAGE_FINDINGS = REPO_ROOT / "tests" / "coverage_monte_carlo" / "FINDINGS.md"
COVERAGE_B1000 = (
    REPO_ROOT
    / "tests"
    / "coverage_monte_carlo"
    / "results_b1000"
    / "coverage_b1000.json"
)
COVERAGE_ROBUSTNESS_B1000 = (
    REPO_ROOT
    / "tests"
    / "coverage_monte_carlo"
    / "results_b1000"
    / "coverage_robustness_b1000.json"
)
TRACK_C_LOGLOG_FIGURE = REPO_ROOT / "tests" / "perf" / "figures" / "track_c_loglog.pdf"
MAIN_PDF = REPO_ROOT / "Paper-JSS" / "manuscript" / "main.pdf"
FIXED_PDF_DATE = b"D:20260531000000Z"
FIXED_ZIP_DATETIME = (2026, 5, 31, 0, 0, 0)
GENERATED_PDF_FIGURES = (
    TRACK_C_LOGLOG_FIGURE,
    *sorted((REPO_ROOT / "Paper-JSS" / "manuscript" / "figures").glob("*.pdf")),
)
AUDIT_SCRIPTS_WITH_FIXED_CLOCK = (
    FULL_AUDIT,
    REPO_ROOT / "Paper-JSS" / "replication" / "scripts" / "agent_interface_audit.py",
    REPO_ROOT
    / "Paper-JSS"
    / "replication"
    / "scripts"
    / "jss_formal_compliance_audit.py",
    REPO_ROOT / "Paper-JSS" / "replication" / "scripts" / "jss_submission_package.py",
    REPO_ROOT
    / "Paper-JSS"
    / "replication"
    / "scripts"
    / "manuscript_artifact_audit.py",
    REPO_ROOT
    / "Paper-JSS"
    / "replication"
    / "scripts"
    / "methodological_gap_ledger.py",
    REPO_ROOT / "Paper-JSS" / "replication" / "scripts" / "release_boundary_audit.py",
    REPO_ROOT
    / "Paper-JSS"
    / "replication"
    / "scripts"
    / "reproduction_environment_audit.py",
    REPO_ROOT / "Paper-JSS" / "replication" / "scripts" / "source_snapshot_manifest.py",
    REPO_ROOT / "Paper-JSS" / "replication" / "scripts" / "stata_bridge_audit.py",
    REPO_ROOT / "Paper-JSS" / "replication" / "scripts" / "validate_claims.py",
    REPO_ROOT
    / "Paper-JSS"
    / "replication"
    / "scripts"
    / "validation_evidence_audit.py",
)
PARITY_SECTION = REPO_ROOT / "Paper-JSS" / "manuscript" / "sections" / "05-parity.tex"
PARITY_COMPACT_SECTION = (
    REPO_ROOT / "Paper-JSS" / "manuscript" / "sections" / "05-parity-compact.tex"
)
COMPUTATIONAL_DETAILS_SECTION = (
    REPO_ROOT / "Paper-JSS" / "manuscript" / "sections" / "08-computational-details.tex"
)
ROOT_README = REPO_ROOT / "README.md"
ROOT_README_CN = REPO_ROOT / "README_CN.md"
MANUSCRIPT_MD_EXPORT = REPO_ROOT / "Paper-JSS" / "manuscript" / "main.md"
MANUSCRIPT_ZH_EXPORT = REPO_ROOT / "Paper-JSS" / "manuscript" / "main-zh.md"

REQUIRED_RELEASE_CHECKS = {
    "clean_combined_worktree",
    "package_tag_at_head",
    "versions_consistent",
    "unreleased_changelog_finalized",
    "source_paths_finalized",
    "paper_paths_finalized",
}

ROOT_RELEASE_PREFIXES = (
    "src/statspai/",
    "tests/",
    "scripts/schema_quality.py",
    "scripts/stability_audit.py",
    "scripts/registry_stats.py",
    "scripts/dump_schemas.py",
    "schemas/",
    "docs/",
    "CHANGELOG.md",
    "MIGRATION.md",
    "README.md",
    "README_CN.md",
    "pyproject.toml",
)


def _run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # ``Paper-JSS/`` is a git-ignored, local-only submission working directory
    # (see ``.gitignore``).  Its replication scripts are absent from fresh
    # clones and CI checkouts, so a release-gate test that shells out to one of
    # them is only meaningful when the local tree is present.  Tracked scripts
    # (e.g. ``scripts/registry_stats.py``) still run normally.
    if not Path(script).exists():
        pytest.skip(
            f"{Path(script).name} lives under the git-ignored Paper-JSS/ "
            "submission working dir; absent in this checkout"
        )
    env = os.environ.copy()
    env.setdefault("SOURCE_DATE_EPOCH", "1780185600")
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _require_paper_jss() -> None:
    """Skip when the git-ignored, local-only Paper-JSS/ tree is absent.

    Tests that read Paper-JSS manuscript/PDF/script artifacts directly (rather
    than via :func:`_run`) call this so they skip cleanly on fresh clones and
    CI checkouts instead of raising ``FileNotFoundError``.
    """
    if not (REPO_ROOT / "Paper-JSS").exists():
        pytest.skip("Paper-JSS/ submission working dir is git-ignored / local-only")


def _claim_lint_module():
    spec = importlib.util.spec_from_file_location("jss_validate_claims", CLAIM_LINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_status_paths() -> list[str]:
    res = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    paths: list[str] = []
    for line in res.stdout.splitlines():
        if not line:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    return paths


def _source_snapshot_display_path(path: str) -> str:
    return LEGACY_REVIEW_PATH_REDACTIONS.get(path, path)


def _inside_git_worktree() -> bool:
    res = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return res.returncode == 0 and res.stdout.strip() == "true"


def test_registry_stats_docs_are_live() -> None:
    """Live registry counts must match the public stats docs."""
    res = _run(REGISTRY_STATS, "--check")
    assert res.returncode == 0, res.stderr


def test_active_joss_review_artifacts_are_protected_from_jss_docs() -> None:
    """Active JOSS review docs stay in place but are not JSS guidance."""
    for path in LEGACY_JOURNAL_REVIEW_PATHS:
        if path.name.startswith("test_joss_"):
            if path in ACTIVE_REVIEW_COMPAT_TEST_PATHS:
                assert (
                    path.exists()
                ), f"active review compatibility test is missing: {path}"
                text = path.read_text(encoding="utf-8")
                assert "test_external_reviewer_followups.py" in text
                assert "reviewers copying that command" in text
            else:
                assert (
                    not path.exists()
                ), f"retired JOSS follow-up test should stay retired: {path}"
        else:
            assert path.exists(), f"active JOSS review artifact must remain: {path}"
    assert (REPO_ROOT / "docs" / "jss_source_audit_dossier.md").exists()

    for rel in ("paper.md",):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "JO" "SS" not in text
        assert "jo" "ss_" not in text


def test_source_snapshot_manifest_has_structured_release_gate() -> None:
    res = _run(SOURCE_MANIFEST)
    assert res.returncode == 0, res.stderr

    payload = json.loads((RESULTS / "source_snapshot_manifest.json").read_text())
    manifest_text = json.dumps(payload)
    assert "jo" "ss_" not in manifest_text
    assert "JO" "SS" not in manifest_text
    readiness = payload["release_readiness"]
    gate_checks = readiness["release_gate_checks"]
    names = {item["check"] for item in gate_checks}

    assert names == REQUIRED_RELEASE_CHECKS
    for item in gate_checks:
        assert isinstance(item["ok"], bool)
        assert isinstance(item["detail"], str)
        assert item["detail"]

    expected_ready = all(item["ok"] for item in gate_checks)
    assert readiness["ready_for_final_publication"] is expected_ready

    breakdown = readiness["release_blocker_breakdown"]
    assert set(breakdown) == {
        "generated_status_counts",
        "hand_edited_status_counts",
        "package_code_paths",
        "package_docs_paths",
        "paper_manuscript_paths",
        "paper_other_paths",
        "paper_replication_paths",
        "validation_test_paths",
    }

    md = (RESULTS / "source_snapshot_manifest.md").read_text()
    for snippet in (
        "Final publication checklist:",
        "Final-publication gate blocker breakdown:",
        "Display path redactions:",
        "Active external-review filenames are omitted from the JSS source-snapshot gate",
        "retired external-review filenames are displayed under retired-external aliases",
        "`clean_combined_worktree`",
        "`package_tag_at_head`",
        "`versions_consistent`",
        "`unreleased_changelog_finalized`",
        "`source_paths_finalized`",
        "`paper_paths_finalized`",
    ):
        assert snippet in md


def test_source_snapshot_manifest_watches_root_release_paths() -> None:
    """Dirty source, tests, docs, and release metadata must enter the gate."""
    if not _inside_git_worktree():
        pytest.skip("git-only release-path watcher; archive mode preserves manifest")
    assert _run(SOURCE_MANIFEST).returncode == 0
    payload = json.loads((RESULTS / "source_snapshot_manifest.json").read_text())
    watched = {
        item.split(" ", 1)[1]
        for item in payload["jss_source_snapshot"]["watched_dirty_paths"]
    }

    status_paths = _git_status_paths()
    expected = {
        _source_snapshot_display_path(path)
        for path in status_paths
        if path not in ACTIVE_EXTERNAL_REVIEW_PATHS
        if any(
            path == prefix.rstrip("/") or path.startswith(prefix)
            for prefix in ROOT_RELEASE_PREFIXES
        )
    }

    assert expected <= watched
    assert not (ACTIVE_EXTERNAL_REVIEW_PATHS & watched)
    if "tests/test_jo" "ss_reviewer_followups.py" in status_paths:
        assert "tests/retired-external-reviewer-followups.py" in watched
    assert not (
        "D tests/test_external_reviewer_followups.py"
        in payload["jss_source_snapshot"]["watched_dirty_paths"]
    )


def test_strict_release_exit_code_matches_manifest_readiness() -> None:
    _run(SOURCE_MANIFEST)
    readiness = json.loads((RESULTS / "source_snapshot_manifest.json").read_text())[
        "release_readiness"
    ]["ready_for_final_publication"]

    strict = _run(SOURCE_MANIFEST, "--strict-release")
    assert (strict.returncode == 0) is readiness


def test_release_boundary_audit_requires_structured_release_gate() -> None:
    assert _run(SOURCE_MANIFEST).returncode == 0
    res = _run(RELEASE_AUDIT)
    assert res.returncode == 0, res.stderr

    audit = json.loads((RESULTS / "release_boundary_audit.json").read_text())
    summary = audit["summary"]

    assert audit["status"] == "PASS"
    assert len(summary["release_gate_checks"]) == len(REQUIRED_RELEASE_CHECKS)
    assert set(summary["release_blocker_breakdown"])
    assert "docs/jss_source_audit_dossier.md" in audit["checked_files"]
    assert "paper.md" not in audit["checked_files"]

    source = RELEASE_AUDIT.read_text()
    assert "active JOSS review" in source
    assert "without requiring edits to `paper.md`" in source


def test_validation_claim_lint_covers_release_notes() -> None:
    """The release notes must not drift past the paper's scoped validation claim."""
    res = _run(CLAIM_LINT)
    assert res.returncode == 0, res.stderr

    payload = json.loads((RESULTS / "claim_lint.json").read_text())
    claim_lint = _claim_lint_module()
    stale_forest_t4 = "forest row remains a T4 stochastic calibration disclosure"
    stale_chinese_promotion = "".join(["模块 13 从 ", "GAP"])

    assert "CHANGELOG.md" in payload["checked_files"]
    assert "CONTRIBUTING.md" in payload["checked_files"]
    assert "CONTRIBUTORS.md" in payload["checked_files"]
    assert "papers/run_replication.py" in payload["checked_files"]
    assert "papers/run_experiments.py" in payload["checked_files"]
    assert "tools/audit_citations.py" in payload["checked_files"]
    assert "CITATION.cff" in payload["protected_joss_files"]
    assert "paper.md" in payload["protected_joss_files"]
    assert "README.md" in payload["protected_joss_files"]
    assert "docs/joss_reviewer_guide.md" in payload["protected_joss_files"]
    assert "docs/joss_validation_dossier.md" in payload["protected_joss_files"]
    assert (
        "StatsPAI_full_data_analysis_skill/SKILL.md" in payload["protected_joss_files"]
    )
    assert "docs/agent_cards_spec.md" in payload["checked_files"]
    assert "docs/guides/agent_native_workflow.md" in payload["checked_files"]
    assert stale_forest_t4 in claim_lint.FORBIDDEN_SNIPPETS
    assert "validated-or-better`.  Each may only go up" in claim_lint.FORBIDDEN_SNIPPETS
    assert "most complete " "across ecosystems" in claim_lint.FORBIDDEN_SNIPPETS
    assert (
        "Python's first "
        "feature-complete implementation" in claim_lint.FORBIDDEN_SNIPPETS
    )
    assert "first power-" "analysis tool" in claim_lint.FORBIDDEN_SNIPPETS
    assert "gold " "standard" in claim_lint.FORBIDDEN_SNIPPETS
    assert "empirical research " "workflow" in claim_lint.FORBIDDEN_SNIPPETS
    assert "manuscript-" "ready" in claim_lint.FORBIDDEN_SNIPPETS
    assert "full-" "stack" in claim_lint.FORBIDDEN_SNIPPETS
    assert "single, consistent " "Python API" in claim_lint.FORBIDDEN_SNIPPETS
    assert "all in one " "place" in claim_lint.FORBIDDEN_SNIPPETS
    assert "Stata has almost " "none" in claim_lint.FORBIDDEN_SNIPPETS
    assert "R has them " "scattered" in claim_lint.FORBIDDEN_SNIPPETS
    assert (
        "Every result object exposes "
        "the same interface" in claim_lint.FORBIDDEN_SNIPPETS
    )
    assert "shared result-" "object contract" in claim_lint.FORBIDDEN_SNIPPETS
    assert "same result-" "object contract" in claim_lint.FORBIDDEN_SNIPPETS
    assert "所有估计器共享一个公共结果对象" "契约" in claim_lint.FORBIDDEN_SNIPPETS
    assert "550+ top-" "level functions" in claim_lint.FORBIDDEN_SNIPPETS
    assert "7," "096" in claim_lint.FORBIDDEN_SNIPPETS
    assert "2," "360 (33.3%)" in claim_lint.FORBIDDEN_SNIPPETS
    assert "36 R-" "parity modules" in claim_lint.FORBIDDEN_SNIPPETS
    assert "21 Stata-" "parity modules" in claim_lint.FORBIDDEN_SNIPPETS
    assert (
        "50 receive a pass-type verdict, " "1 is a T4" in claim_lint.FORBIDDEN_SNIPPETS
    )
    assert (
        "StatsPAI: An Agent-" "Native Python Toolkit" in claim_lint.FORBIDDEN_SNIPPETS
    )
    assert "short reviewer " "guide" in claim_lint.FORBIDDEN_SNIPPETS
    assert "JSS validation " "dossier" in claim_lint.FORBIDDEN_SNIPPETS
    assert "docs/getting-started.md" in payload["checked_files"]
    assert "docs/reference/index.md" in payload["checked_files"]
    assert "docs/jss_source_audit_dossier.md" in payload["checked_files"]
    assert "Paper-JSS/manuscript/main-zh.md" in payload["historical_drift_files"]
    assert "Paper-JSS/JSS-research-plan.md" in payload["historical_drift_files"]
    assert "Paper-JSS/NEXT-STEPS.md" in payload["historical_drift_files"]
    assert len(payload["historical_drift_files"]) == 10
    assert stale_chinese_promotion in claim_lint.HISTORICAL_FORBIDDEN_SNIPPETS
    assert "paper2026jo" "ss" in claim_lint.HISTORICAL_FORBIDDEN_SNIPPETS
    assert payload["forbidden_snippet_count"] == len(claim_lint.FORBIDDEN_SNIPPETS)
    assert payload["historical_forbidden_snippet_count"] == len(
        claim_lint.HISTORICAL_FORBIDDEN_SNIPPETS
    )
    assert "CHANGELOG.md" in payload["required_snippet_files"]
    assert "docs/agent_cards_spec.md" in payload["required_snippet_files"]
    assert "docs/guides/agent_native_workflow.md" in payload["required_snippet_files"]
    assert payload["claim_counts"]["certified"] == 61
    assert payload["claim_counts"]["validated"] == 50
    assert payload["claim_counts"]["api_stable"] == 1016
    assert payload["claim_counts"]["experimental"] == 3


def test_validation_evidence_audit_separates_grade_from_supplemental_notes() -> None:
    """Validated/certified status must not be justified by weak add-on notes."""
    res = _run(VALIDATION_EVIDENCE_AUDIT)
    assert res.returncode == 0, res.stderr

    payload = json.loads((RESULTS / "validation_evidence_audit.json").read_text())
    report = (RESULTS / "validation_evidence_audit.md").read_text()
    summary = payload["summary"]

    assert payload["status"] == "PASS"
    assert summary["certified_validated_symbols"] == 111
    assert summary["certified_symbols"] == 61
    assert summary["validated_symbols"] == 50
    assert summary["certified_without_certified_grade_evidence"] == 0
    assert summary["validated_without_validated_grade_evidence"] == 0
    assert summary["supplemental_only_symbols"] == 0
    assert summary["symbols_with_limitations"] == 9
    for key in (
        "certified_validated_symbols",
        "certified_symbols",
        "validated_symbols",
        "missing_validation_notes",
        "certified_without_certified_grade_evidence",
        "validated_without_validated_grade_evidence",
        "supplemental_only_symbols",
        "symbols_with_limitations",
        "unique_evidence_paths",
    ):
        assert payload[key] == summary[key]
    assert payload["missing_evidence_path_count"] == summary["missing_evidence_paths"]
    assert "other" not in summary["qualifying_note_kind_symbols"]
    assert "api_unit_contract" not in summary["qualifying_note_kind_symbols"]
    assert "api_unit_contract" in summary["supplemental_note_kind_symbols"]
    assert "## Qualifying Evidence Note Kinds" in report
    assert "## Supplemental Evidence Note Kinds" in report
    assert "## Symbols With Scoped Limitations" in report
    assert "blanket parity or validation claims" in report
    assert ".;" not in report
    assert "## Validated-Tier Symbols" in report

    rows = {row["name"]: row for row in payload["rows"]}
    assert "reference_parity" in rows["rddensity"]["qualifying_note_kinds"]
    limited_rows = [row for row in rows.values() if row["limitations"]]
    assert len(limited_rows) == summary["symbols_with_limitations"]
    assert "rddensity" in {row["name"] for row in limited_rows}
    for row in rows.values():
        assert row["status_grade_kinds"], row["name"]


def test_coverage_findings_track_b1000_artifacts() -> None:
    """The coverage narrative must track the committed deep-audit JSON."""
    _require_paper_jss()
    findings = COVERAGE_FINDINGS.read_text()
    parity_long = PARITY_SECTION.read_text()
    parity_compact = PARITY_COMPACT_SECTION.read_text()
    computational_details = COMPUTATIONAL_DETAILS_SECTION.read_text()
    root_readme = ROOT_README.read_text()
    root_readme_cn = ROOT_README_CN.read_text()
    manuscript_md_export = MANUSCRIPT_MD_EXPORT.read_text()
    manuscript_zh_export = MANUSCRIPT_ZH_EXPORT.read_text()
    canonical = json.loads(COVERAGE_B1000.read_text())
    robustness = json.loads(COVERAGE_ROBUSTNESS_B1000.read_text())
    public_narratives = (
        findings,
        parity_long,
        parity_compact,
        computational_details,
        root_readme,
        root_readme_cn,
        manuscript_md_export,
        manuscript_zh_export,
    )
    rate_narratives = (
        findings,
        parity_long,
        parity_compact,
        root_readme,
        root_readme_cn,
        manuscript_md_export,
        manuscript_zh_export,
    )

    assert "results_b1000/coverage_b1000.json" in findings
    assert "results_b1000/coverage_robustness_b1000.json" in findings
    assert len(canonical) == 7
    assert "DML sits just above the upper edge" in findings
    assert "946/1000 = 0.946" in findings
    assert "seven known-truth DGPs" in findings
    assert "seven known-truth" in parity_long
    assert "all seven" in parity_compact
    assert "materialized seven-row" in computational_details
    assert "seven materialized nominal rows" in root_readme
    assert "7 个已物化 nominal 行" in root_readme_cn
    assert "all seven known-truth nominal" in manuscript_md_export
    assert "全部 7 个 known-truth nominal 行" in manuscript_zh_export

    for row in canonical:
        for narrative in rate_narratives:
            assert f"{row['rate']:.3f}" in narrative
        assert f"B={row['B']}" in findings
    for row in robustness:
        assert f"{row['rate']:.3f}" in findings
        assert f"B={row['B']}" in findings

    for stale in (
        "three " "cheap",
        "three" "-row",
        "B=200 " "cap",
        "B=300 unless noted",
        "0.940 (B=200",
        "0.91 (B=300)",
        "0.98 (B=50)",
        "cheap closed-" "form rows",
        "materialized " "three",
        "3 " "行廉价闭式",
        "廉价" "闭式行的 3 行",
    ):
        for narrative in public_narratives:
            assert stale not in narrative


def test_generated_pdf_figures_omit_creation_timestamp() -> None:
    """Generated PDFs must not drift just because time passed."""
    bad = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in GENERATED_PDF_FIGURES
        if b"/CreationDate" in path.read_bytes()
    ]
    assert not bad


def test_main_pdf_uses_fixed_source_date_epoch() -> None:
    """The active JSS PDF should not encode the local build clock."""
    _require_paper_jss()
    data = MAIN_PDF.read_bytes()
    assert b"/CreationDate (" + FIXED_PDF_DATE + b")" in data
    assert b"/ModDate (" + FIXED_PDF_DATE + b")" in data


def test_submission_archive_members_use_fixed_zip_timestamp() -> None:
    """The zip itself should not depend on checkout mtimes."""
    if not SUBMISSION_ARCHIVE.exists():
        pytest.skip("submission archive has not been built")
    with zipfile.ZipFile(SUBMISSION_ARCHIVE) as zf:
        bad = [
            item.filename
            for item in zf.infolist()
            if item.date_time != FIXED_ZIP_DATETIME
        ]
    assert not bad


def test_submission_manifest_discloses_fixed_zip_timestamp() -> None:
    """Reviewers should see the deterministic archive timestamp policy."""
    if not SUBMISSION_MANIFEST.exists():
        pytest.skip("submission manifest has not been built")
    manifest = json.loads(SUBMISSION_MANIFEST.read_text())
    assert manifest["generated_at_unix"] == 1780185600
    assert manifest["source_date_epoch"] == 1780185600
    assert manifest["zip_member_datetime"] == list(FIXED_ZIP_DATETIME)


def test_jss_audit_scripts_honor_source_date_epoch() -> None:
    """Reviewer-facing audit artifacts should be reproducible under Make."""
    _require_paper_jss()
    for script in AUDIT_SCRIPTS_WITH_FIXED_CLOCK:
        source = script.read_text()
        assert "SOURCE_DATE_EPOCH" in source, script
        assert "_generated_at_unix()" in source, script
    full_audit_source = FULL_AUDIT.read_text()
    assert "_elapsed_seconds" in full_audit_source
    assert "_normalize_command_output" in full_audit_source
    assert '"seconds": _elapsed_seconds(start)' in full_audit_source


def test_methodological_gap_ledger_pins_t4_metadata() -> None:
    """Methodological/T4 rows must expose native-vs-reference boundaries."""
    res = _run(GAP_LEDGER)
    assert res.returncode == 0, res.stderr

    payload = json.loads((RESULTS / "methodological_gap_ledger.json").read_text())
    rows = {row["module"]: row for row in payload["ledger"]}
    expected = {
        "07_scm": {
            "validation_tier": "identification_dependent_native",
            "reference_backend": "Synth",
            "solver_best_start": "regression",
            "solver_near_best_start_count": "4",
            "solver_near_best_weight_class_count": "2",
            "solver_near_best_weight_l1_max": "0.00513",
            "weight_solution_nonunique": "True",
        },
    }
    for module, required in expected.items():
        checks = {check["field"]: check for check in rows[module]["metadata_checks"]}
        for field, value in required.items():
            assert checks[field]["observed"] == value
            assert checks[field]["ok"] is True
    assert payload["summary"]["metadata_guard_failures"] == 0
    assert payload["summary"]["methodological_gap_count"] == 1
    assert payload["summary"]["classified_gap_count"] == 1
    assert payload["summary"]["category_counts"] == {
        "classical_scm_reference_disagreement": 1,
    }
    assert payload["summary"]["non_circular_guard_count"] == 1
    assert payload["summary"]["non_circular_guard_failures"] == 0
    assert payload["summary"]["reference_disagreement_guard_count"] == 1
    assert payload["summary"]["reference_disagreement_guard_failures"] == 0

    guards = {module: rows[module]["non_circular_guard"] for module in ("07_scm",)}
    assert guards["07_scm"]["path"] == "tests/r_parity/52_scm_unique.py"
    assert guards["07_scm"]["snippet"] == "unique convex-hull SCM"
    assert guards["07_scm"]["ok"] is True
    ref_guard = rows["07_scm"]["reference_disagreement_guard"]
    assert ref_guard["ok"] is True
    assert ref_guard["observed"]["statistic"] == "avg_post_gap"
    assert ref_guard["observed"]["py_stata_rel"] <= 1e-3
    assert ref_guard["observed"]["r_stata_rel"] >= 1e-2

    md = (RESULTS / "methodological_gap_ledger.md").read_text()
    assert "Non-circular native guard" in md
    assert "Reference disagreement guard" in md
    assert "py-Stata rel" in md
    assert "R-Stata rel" in md
    assert "multi-start diagnostics" in md
    assert "native tracks Stata synth" in md
    assert "uniquely identified convex-hull DGP" in md
    assert "18_augsynth" not in md
    assert "19_gsynth" not in md
    assert "gsynth_native_factor_convention_gap" not in md
    assert "09_rddensity" not in md
    assert "12_sdid" not in md

    parity_compact = " ".join(PARITY_COMPACT_SECTION.read_text().split())
    assert "two non-T2 Track A rows" in parity_compact
    assert (
        "classical SCM is a T4 identification/reference-disagreement disclosure"
        in parity_compact
    )
    assert "causal forest is a stochastic T3" in parity_compact
    assert "50 receive a pass-type verdict, " "1 is a T4" not in parity_compact


def test_r_parity_readme_does_not_hide_native_t4_rows() -> None:
    """The parity module list must say when Python-side rows are native."""
    readme = (REPO_ROOT / "tests" / "r_parity" / "README.md").read_text()
    for row in (
        "Historical verification worklog (not the current source-snapshot audit)",
        "modules 01--52",
        '| 07 | Classical SCM | `sp.synth(method="classic", backend="native")`',
        '| 09 | RD density (CJM) | `sp.rddensity(backend="native")`',
        '| 12 | Synthetic DID | `sp.sdid(backend="native")`',
        '| 18 | Augmented SCM | `sp.augsynth(backend="native")`',
        '| 19 | Generalized SCM | `sp.gsynth(backend="native")`',
        "| 50 | Arellano--Bond GMM | `sp.xtabond` | `plm::pgmm` |",
        '| 52 | Identified classical SCM DGP | `sp.synth(method="classic", backend="native")`',
    ):
        assert row in readme
    for stale_bridge_claim in (
        "Latest full verification record",
        '| 07 | Classical SCM | `sp.synth(method="classic", backend="synth")`',
        '| 09 | RD density (CJM) | `sp.rddensity(backend="r")`',
        '| 12 | Synthetic DID | `sp.sdid(backend="synthdid")`',
        '| 18 | Augmented SCM | `sp.augsynth(backend="augsynth")`',
        '| 19 | Generalized SCM | `sp.gsynth(backend="gsynth")`',
    ):
        assert stale_bridge_claim not in readme


def test_submission_package_verifier_pins_page_and_claim_guards() -> None:
    """The built archive verifier is part of the JSS release contract."""
    if not SUBMISSION_ARCHIVE.exists():
        pytest.skip("submission archive has not been built")
    if any(
        path.exists() and path.stat().st_mtime > SUBMISSION_ARCHIVE.stat().st_mtime
        for path in SUBMISSION_ARCHIVE_INPUTS
    ):
        pytest.skip("submission archive predates verifier inputs")

    res = _run(VERIFY_PACKAGE)
    assert res.returncode == 0, res.stderr

    verifier_source = VERIFY_PACKAGE.read_text()
    for snippet in (
        "JSS 30-page slow-review warning threshold",
        "does not report the live main.pdf page count",
        "claim_lint.json validation-status tier counts do not sum to registry",
        "certified + validated does not equal",
        "cover letter date does not match the source-snapshot audit date",
        "formal compliance audit did not inspect the final archive",
        "formal compliance install/import probe is not passing",
        "extracted archive pip install failed",
        "extracted archive import probe failed",
        "extracted archive import version mismatch",
        "manuscript artifact audit reports hash mismatches",
        "manifest source_date_epoch does not match verifier setting",
        "manifest zip_member_datetime does not match fixed timestamp",
        "fixed SOURCE_DATE_EPOCH",
        "zip timestamps",
        "JSS markup macros and labelled floats are used",
        "pip --no-deps --target install/import probe ok=True",
        "python -m pip install --upgrade pip setuptools wheel",
        "short reviewer replication path completes within one hour",
        "existing implementations and comparative scope are discussed",
        "ACTIVE_COMPARATIVE_SCOPE_SNIPPETS",
        "active manuscript lacks reviewer-facing comparative scope evidence",
        "related-software table",
        "StatsPAI advantages and disadvantages",
        "manifest JSS archive generated files do not match verifier allow-list",
        "manifest active external-review exclusions do not match verifier allow-list",
        "manifest active external-review exclusion reasons do not match verifier",
        "missing_related_tokens=[]",
        "Track B B=1000 nominal artifact count is not seven rows",
        "active manuscript does not state the seven-row Track B",
        "active manuscript does not report Track B nominal",
        "active manuscript contains stale Track B artifact",
        "archive root README lacks JSS boundary note",
        "Tier-1 reviewer transcript exceeds the one-hour reviewer path",
        "Tier-1 transcript states no R/Stata headline path",
        "Tier-1 live R/Stata dependency markers",
        "tier1_no_r_stata=True",
        "tier1_live_external=0",
        "cover letter does not disclose the no-R/no-Stata Tier-1",
        "manuscript README does not disclose the no-R/no-Stata",
        "brittle historical pytest",
        "jss_full_audit.json lacks top-level PASS status",
        "jss_full_audit.json top-level",
        "final_publication_gate_ready",
        "gate_blocker_paths",
        "jss_full_audit.json does not expose a no-R/no-Stata Tier-1",
        "does not prove the Tier-1",
        "R/Stata dependency markers",
        "checked_files=8",
        "release_boundary_audit.json",
        "release_boundary_audit.md does not report the JSS/JOSS",
        "release_boundary_audit.json has unexpected checked file count",
        "release_boundary_audit.json does not include the JSS source dossier",
        "CITATION.cff lacks",
        "root and package CITATION.cff metadata differ",
        "CITATION.cff version does not match pyproject.toml",
        "CITATION.cff includes active external-review preferred-citation metadata",
        "pyproject.toml still declares an Alpha development classifier",
        "pyproject.toml does not declare the reviewer-facing Beta classifier",
        "scoped validation-tiered description",
        "methodological_gap_ledger.md lacks required T4 metadata guard",
        "non-circular native guards",
        "non_circular_native_guards=1",
        "reference_disagreement_guards=1",
        "Reference disagreement guard",
        "py-Stata rel",
        "R-Stata rel",
        "native tracks Stata synth",
        "unique convex-hull SCM",
        "test_scm_family_recovers_known_att",
        "identification_dependent_native",
        "ARCHIVE_FORBIDDEN_CLAIM_SNIPPETS",
        "ARCHIVE_CLAIM_GUARD_ALLOWLIST",
        "archive contains forbidden package-facing claim snippets",
        '"Validation: " "validated."',
        "common result " "contract",
        "common result " "objects",
        '"[experimental] " "[experimental]"',
        "Validation: experimental" ".",
        "Validation: deprecated" ".",
        "historical_drift_files=10",
        "jss_full_audit.md does not expose historical claim-lint scope",
        "Historical drift files: 10",
        "claim_lint.md does not report historical drift scope",
        "claim_lint.json has unexpected historical drift scope",
        "claim_lint.json does not include excluded planning note",
        "docs/getting-started.md",
        "docs/reference/index.md",
        "docs/jss_source_audit_dossier.md",
        "Validated without validated-grade evidence: 0",
        "Supplemental-only certified/validated symbols: 0",
        "validation_evidence_audit.json lacks top-level",
        "validation_evidence_audit.json limitation rows",
        "rddensity scoped limitation row",
        "missing_evidence_path_count",
        "Qualifying Evidence Note Kinds",
        "Supplemental Evidence Note Kinds",
        "Symbols With Scoped Limitations",
        "Validated-Tier Symbols",
        "classifies 'other' as",
        'docs/jo" "ss_reviewer_guide.md',
        'docs/jo" "ss_validation_dossier.md',
        'For JO" "SS Reviewers',
        'JO" "SS Reviewer Guide',
        'JO" "SS Validation Dossier',
        'JO" "SS reviewer-facing documentation',
        'A JO" "SS paper for StatsPAI is currently under review',
        'submitted to JO" "SS remains open',
        "tests/r_parity/README.md",
        "R_PARITY_README_REQUIRED_NATIVE_ROWS",
        "reference-backend escape",
        "PACKAGE_SIZE_DISCLOSURE_MEMBERS",
        "does not report final archive size/count",
        "size_bytes",
        "size_mib",
        "size_mb_decimal",
        "MiB or decimal-MB accounting",
        "MB decimal",
        "unique to " "StatsPAI",
        "No other " "econometrics package",
        "most complete " "across ecosystems",
        "battle-" "tested",
        "drop-in " "replacement",
        "publication-" "grade",
        "publication-" "ready",
        "manuscript-" "ready",
        "journal-" "ready",
        "one-" "click",
        "one " "click",
        "state-of-" "the-art",
        "state of " "the art",
        "full research " "workflow",
        "full-" "stack",
        "single, consistent " "Python API",
        "single unified " "API",
        "all in one " "place",
        "Stata has almost " "none",
        "R has them " "scattered",
        "matches R's " "5-package " "spatial stack",
        "Neither Stata " "nor R",
        "R has no " "equivalent",
        "Stata requires paid " "add-ons",
        "one-call " "comprehensive",
        "comprehensive " "report",
        "full " "coverage",
        "Python's first " "feature-complete implementation",
        "Python's first " "unified CATE learner race",
        "Python's first " "unified spatial econometrics package",
        "first power-" "analysis tool",
        "gold " "standard",
        "empirical research " "workflow",
        '"Every result object exposes " "the same interface"',
        "across every " "registered estimator",
        "Every result object " "speaks the same export protocol",
        "Every result object " "follows the same contract",
        "A machine-readable schema of " "every estimator",
        "consumes " "every estimator",
        "work across " "all estimators",
        "call every " "StatsPAI function",
        "\\u81ea\\u7136\\u8bed\\u8a00\\u8c03\\u7528\\u6bcf\\u4e2a\\u51fd\\u6570",
        "1,018-" "function registry",
        "1,018 " "registered public functions",
        "1,018-" "function surface",
        "1018-" "function surface",
        "Tier-B 127 / " "1018",
        "550+ top-" "level functions",
        "Py-Stata-" "primary",
        "reported 50 rendered " "modules",
        "43 of 50 rendered " "modules",
        "**249," "457**",
        "86," "397",
        "266k LOC " "(core) + 93k LOC (tests)",
        "266k " "\\u884c\\u6838\\u5fc3\\u4ee3\\u7801 + 93k " "\\u884c\\u6d4b\\u8bd5",
        "7," "096",
        "7{,}" "096",
        "2," "360 (33.3%)",
        "206 " "(2.9%)",
        "\\u672c\\u8868\\u4e2d\\u8986\\u76d6\\u6700\\u5e7f",
        "\\u5b8c\\u6574\\u8986\\u76d6",
        "\\u76f8\\u540c\\u7684\\u7ed3\\u679c\\u5bf9\\u8c61\\u4e0e schema \\u5951\\u7ea6",
        "Full Python " "replication",
        "Data-driven bandwidth " "with formal optimality",
        "GPU placeholder " "notebook",
        "placeholder " "notebook",
        "\\u6bcf\\u4e2a\\u51fd\\u6570\\u90fd\\u652f\\u6301 Word",
        "StatsPAI: An Agent-",
        "Native Python Toolkit",
        "source_snapshot_manifest.md lacks display-path redaction note",
        "source_snapshot_manifest.md does not explain active",
        "source_snapshot_manifest.md does not explain retired-path",
        "source_snapshot_manifest.md lacks the retired external-review",
        "source_snapshot_manifest.json lacks active/retired external-review note",
        "source_snapshot_manifest.json marks the current external",
        "ACTIVE_EXTERNAL_REVIEW_ARTIFACTS_FORBIDDEN_IN_ARCHIVE",
        "archive includes active external-review artifacts",
        "_source_snapshot_archive_required",
        "blanket result-object claim",
        "Paper-JSS/colab_gpu_bench",
        "Paper-JSS/DRAFT-NOTES",
        "Paper-JSS/references/",
        "README_CN",
        ".egg-info",
        "\\u8bc4\\u5ba1",
        "\\u4e2d\\u6587",
    ):
        assert snippet in verifier_source

    with zipfile.ZipFile(SUBMISSION_ARCHIVE) as zf:
        names = set(zf.namelist())
    forbidden_members = (
        "paper.md",
        "docs/jo" "ss_reviewer_guide.md",
        "docs/jo" "ss_validation_dossier.md",
        "Paper-JSS/DRAFT-NOTES.md",
        "Paper-JSS/JSS-research-plan.md",
        "Paper-JSS/NEXT-STEPS.md",
        "Paper-JSS/REVIEW-IMPROVEMENTS.md",
        "Paper-JSS/colab_gpu_bench.ipynb",
        "Paper-JSS/manuscript/main.md",
        "Paper-JSS/manuscript/main-zh.md",
        "Paper-JSS/manuscript/\u8bc4\u5ba1\u610f\u89c1-\u4e2d\u6587.md",
        "README_CN.md",
    )
    assert not (set(forbidden_members) & names)
    for forbidden_prefix in (
        "Paper-JSS/100-emails/",
        "Paper-JSS/htmlcov/",
        "Paper-JSS/notes/",
        "Paper-JSS/parity/",
        "Paper-JSS/references/",
    ):
        assert not any(name.startswith(forbidden_prefix) for name in names)
