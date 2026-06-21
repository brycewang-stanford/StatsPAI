"""Contract tests for ``scripts/result_protocol_audit.py``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "result_protocol_audit.py"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_result_protocol_summary_renders() -> None:
    res = _run([])
    assert res.returncode == 0, res.stderr
    assert "StatsPAI result protocol audit" in res.stdout
    assert "Method coverage" in res.stdout
    assert "Protocol coverage" in res.stdout


def test_result_protocol_json_shape_and_floors() -> None:
    res = _run(["--json"])
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert {"total", "method_counts", "protocol_counts", "per_class", "floors"} <= set(
        payload
    )
    assert payload["total"] >= payload["floors"]["result_classes"]
    for method, count in payload["method_counts"].items():
        floor_key = f"method_{method}"
        if floor_key in payload["floors"]:
            assert count >= payload["floors"][floor_key]
    for protocol, count in payload["protocol_counts"].items():
        floor_key = f"protocol_{protocol}"
        assert count >= payload["floors"][floor_key]


def test_result_protocol_check_mode_passes() -> None:
    res = _run(["--check"])
    assert res.returncode == 0, res.stdout + res.stderr
    assert "[result_protocol_audit] OK" in res.stdout


def test_core_result_classes_expose_full_agent_protocol() -> None:
    res = _run(["--json"])
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    by_key = {row["key"]: row for row in payload["per_class"]}
    required = {
        "src/statspai/core/results.py:EconometricResults": {
            "summary",
            "tidy",
            "glance",
            "to_dict",
            "to_agent_summary",
            "brief",
        },
        "src/statspai/core/results.py:CausalResult": {
            "summary",
            "tidy",
            "glance",
            "to_dict",
            "to_agent_summary",
            "brief",
            "plot",
        },
        "src/statspai/bayes/_base.py:BayesianCausalResult": {
            "summary",
            "tidy",
            "glance",
            "to_dict",
            "to_agent_summary",
        },
        "src/statspai/output/regression_table.py:RegtableResult": {
            "summary",
            "to_dict",
            "to_agent_summary",
        },
        "src/statspai/fast/inference.py:BootTestResult": {
            "summary",
            "to_dict",
            "to_agent_summary",
        },
        "src/statspai/fast/inference.py:BootWaldResult": {
            "summary",
            "to_dict",
            "to_agent_summary",
        },
        "src/statspai/fast/inference.py:WaldTestResult": {
            "summary",
            "to_dict",
            "to_agent_summary",
        },
        "src/statspai/fast/feols.py:FeolsResult": {
            "summary",
            "to_dict",
            "to_agent_summary",
        },
        "src/statspai/fast/fepois.py:FePoisResult": {
            "summary",
            "to_dict",
            "to_agent_summary",
        },
        "src/statspai/fast/event_study.py:EventStudyResult": {
            "summary",
            "to_dict",
            "to_agent_summary",
        },
        "src/statspai/fast/bench.py:HDFEBenchResult": {
            "summary",
            "to_dict",
            "to_agent_summary",
        },
        "src/statspai/fast/jax_feols.py:FeolsBootstrapResult": {
            "summary",
            "to_dict",
            "to_agent_summary",
        },
        "src/statspai/_auto_estimators.py:AutoDIDResult": {
            "summary",
            "to_dict",
            "to_agent_summary",
        },
        "src/statspai/_auto_estimators.py:AutoIVResult": {
            "summary",
            "to_dict",
            "to_agent_summary",
        },
    }
    for key, methods in required.items():
        assert key in by_key
        assert methods <= set(by_key[key]["effective_methods"])
