"""What the R parity workflow re-derives is stated accurately.

The workflow header used to say the R closed-form parity "refreshes on
every push". It runs on pushes that touch ``tests/r_parity/**``, re-runs
17 of the R modules, against current CRAN rather than ``renv.lock``, and
never imports ``statspai`` (2026-10-02 review, R2). The reader-facing
statement is the "What CI re-derives" section of
``tests/r_parity/R_ENVIRONMENT.md``; this test keeps it, the workflow and
the artefacts on disk in step.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "r-parity.yml"
DOC = ROOT / "tests" / "r_parity" / "R_ENVIRONMENT.md"
R_PARITY = ROOT / "tests" / "r_parity"
STATA_PARITY = ROOT / "tests" / "stata_parity"

pytestmark = pytest.mark.skipif(
    not WORKFLOW.exists(), reason="source checkout only (.github/ not installed)"
)

_MODULE = re.compile(r"\b(\d{2}_[a-z0-9_]+)\b")


def _workflow_modules() -> list:
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index("verify_reproduce.py \\")
    end = text.index("--timeout", start)
    return _MODULE.findall(text[start:end])


def _doc_section() -> str:
    text = DOC.read_text(encoding="utf-8")
    start = text.index("## What CI re-derives")
    return text[start : text.index("\n## R\n", start)]


def test_ci_modules_exist_with_a_script_and_a_golden():
    modules = _workflow_modules()
    assert len(modules) == len(set(modules)) == 17
    for module in modules:
        assert (R_PARITY / f"{module}.R").exists(), module
        assert (R_PARITY / "results" / f"{module}_R.json").exists(), module


def test_doc_lists_exactly_the_modules_ci_runs():
    listed = re.findall(r"`(\d{2}_[a-z0-9_]+)`", _doc_section())
    assert sorted(listed) == sorted(_workflow_modules())


def test_doc_counts_match_the_artefacts_on_disk():
    section = _doc_section()
    n_r = len(list((R_PARITY / "results").glob("*_R.json")))
    n_stata = len(list((STATA_PARITY / "results").glob("*_Stata.json")))
    n_ci = len(_workflow_modules())
    assert f"The R reference of {n_ci} Track A modules" in section
    assert f"All {n_r} R modules" in section
    assert f"All {n_stata} Stata modules" in section
    assert f"The other {n_r - n_ci} R modules" in section


def test_triggers_are_what_the_doc_says():
    text = WORKFLOW.read_text(encoding="utf-8")
    on_block = text[text.index("\non:\n") : text.index("\nconcurrency:")]
    # Path-filtered, never every push; plus the weekly drift probe.
    assert on_block.count('- "tests/r_parity/**"') == 2
    assert "src/" not in on_block
    assert re.search(r'cron: "\d+ \d+ \* \* 1"', on_block), "weekly schedule missing"
    assert "workflow_dispatch" in on_block
    section = _doc_section()
    assert "every Monday" in section and "tests/r_parity/**" in section


def test_job_installs_current_cran_and_does_not_import_statspai():
    text = WORKFLOW.read_text(encoding="utf-8")
    steps = text[text.index("\njobs:\n") :]
    assert "any::fixest" in steps and "renv::restore" not in steps
    assert "pip install" not in steps, "the R job must not need statspai"
    verifier = (R_PARITY / "verify_reproduce.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from) statspai\b", verifier, flags=re.M)
    section = _doc_section()
    assert "not `renv.lock`" in section and "does not import" in section


def test_workflow_header_no_longer_claims_every_push():
    header = WORKFLOW.read_text(encoding="utf-8").split("\non:\n", 1)[0]
    assert "Not on every push" in header
    assert "literally true" not in header
