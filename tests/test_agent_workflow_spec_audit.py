"""Contract tests for ``scripts/agent_workflow_spec_audit.py``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "agent_workflow_spec_audit.py"
EXAMPLE = (
    REPO_ROOT
    / "plans"
    / "2026-06-21-agent-empirical-analysis-uplift"
    / "example_workflow_spec.json"
)
QUALITY_GATE = REPO_ROOT / "scripts" / "quality_gate.py"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_agent_workflow_spec_summary_renders() -> None:
    res = _run([str(EXAMPLE)])
    assert res.returncode == 0, res.stderr
    assert "StatsPAI agent empirical workflow spec audit" in res.stdout
    assert "Status : PASS" in res.stdout
    assert "Design : did" in res.stdout


def test_agent_workflow_spec_json_shape() -> None:
    res = _run([str(EXAMPLE), "--json"])
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert {"status", "score", "design", "issue_counts", "issues"} <= set(payload)
    assert payload["status"] == "pass"
    assert payload["score"] >= 85
    assert payload["issue_counts"] == {"error": 0, "warning": 0}


def test_agent_workflow_spec_check_mode_passes() -> None:
    res = _run([str(EXAMPLE), "--check"])
    assert res.returncode == 0, res.stdout + res.stderr
    assert "[agent_workflow_spec_audit] OK" in res.stdout


def test_agent_workflow_quality_gate_passes() -> None:
    res = subprocess.run(
        [sys.executable, str(QUALITY_GATE), "agent-workflow"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "agent-workflow: observed=0 baseline=0" in res.stdout


def test_agent_workflow_spec_check_mode_fails_bad_spec(tmp_path: Path) -> None:
    bad_spec = tmp_path / "bad_workflow.json"
    bad_spec.write_text(
        json.dumps({"research_question": {"treatment": "x"}}),
        encoding="utf-8",
    )
    res = _run([str(bad_spec), "--check"])
    assert res.returncode == 1
    assert "[agent_workflow_spec_audit] REGRESSION" in res.stderr
    assert "missing mapping section 'data'" in res.stdout


def test_did_workflow_requires_parallel_trends_diagnostics(tmp_path: Path) -> None:
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["diagnostics"] = [{"name": "summary_only", "type": "table"}]
    path = tmp_path / "did_without_pretrend.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    res = _run([str(path), "--json"])
    assert res.returncode == 0, res.stderr
    report = json.loads(res.stdout)
    issue_rules = {issue["rule"] for issue in report["issues"]}
    assert "did_diagnostics_any" in issue_rules
    assert report["status"] == "fail"


def test_agent_workflow_requires_full_execution_loop(tmp_path: Path) -> None:
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["agent_execution"] = {
        "workflow_steps": ["Fit the DID estimator and write a table."],
        "result_handle_policy": "Copy the fitted arrays into the next prompt.",
        "handoff_artifacts": ["table2.docx"],
        "stop_conditions": ["Continue unless the script crashes."],
    }
    path = tmp_path / "did_without_agent_loop.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    res = _run([str(path), "--json"])
    assert res.returncode == 0, res.stderr
    report = json.loads(res.stdout)
    issue_rules = {issue["rule"] for issue in report["issues"]}
    assert "agent_execution_steps" in issue_rules
    assert "result_handle_policy" in issue_rules
    assert report["status"] == "fail"


def test_estimator_functions_resolve_against_offline_schema(tmp_path: Path) -> None:
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    payload["estimators"][0]["statspai_function"] = "sp.not_a_real_estimator"
    path = tmp_path / "unknown_function.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    res = _run([str(path), "--json"])
    assert res.returncode == 0, res.stderr
    report = json.loads(res.stdout)
    assert {
        issue["rule"] for issue in report["issues"]
    } >= {"estimator_function_known"}
    assert "'sp.not_a_real_estimator' is not present" in res.stdout
    assert report["status"] == "fail"
