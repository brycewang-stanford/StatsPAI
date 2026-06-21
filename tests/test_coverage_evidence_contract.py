"""Coverage evidence must be generated live, not treated as stale fixtures."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_root_coverage_xml_is_not_a_committed_evidence_artifact():
    proc = subprocess.run(
        ["git", "ls-files", "coverage.xml"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == ""


def test_ci_coverage_ratchet_consumes_same_job_generated_xml():
    workflow = _read(".github/workflows/ci-cd.yml")
    coverage_step = (
        "pytest tests/ -v --cov=statspai --cov-report=xml "
        "--cov-report=term-missing --cov-fail-under=60"
    )
    ratchet_step = (
        "python scripts/coverage_campaign.py report --xml coverage.xml --check --min 95"
    )

    assert coverage_step in workflow
    assert ratchet_step in workflow
    assert workflow.index(coverage_step) < workflow.index(ratchet_step)
    assert "CI-generated coverage.xml just produced" in workflow


def test_coverage_campaign_and_docs_do_not_call_root_xml_committed_truth():
    script = _read("scripts/coverage_campaign.py")
    reviewer_qa = _read("docs/joss_reviewer_qa.md")

    assert "committed ``coverage.xml``" not in script
    assert "committed `coverage.xml`" not in reviewer_qa
    assert "working artifact, not a committed source of truth" in script
    assert "The current CI gate uses the" in reviewer_qa
    assert "`coverage.xml` generated inside the same job" in reviewer_qa
