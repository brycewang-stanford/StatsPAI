"""Contract tests for ``scripts/error_taxonomy_audit.py``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "error_taxonomy_audit.py"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_error_taxonomy_summary_renders() -> None:
    res = _run([])
    assert res.returncode == 0, res.stderr
    assert "StatsPAI error taxonomy audit" in res.stdout
    assert "taxonomy exceptions" in res.stdout
    assert "generic built-ins" in res.stdout


def test_error_taxonomy_json_is_summary_by_default() -> None:
    res = _run(["--json"])
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert {"totals", "by_exception", "thresholds"} <= set(payload)
    assert "raises" not in payload
    assert "broad_handlers" not in payload
    totals = payload["totals"]
    thresholds = payload["thresholds"]
    assert totals["taxonomy_raises"] >= thresholds["taxonomy_raise_min"]
    assert totals["generic_raises"] <= thresholds["generic_raise_max"]
    assert totals["broad_exception_handlers"] <= thresholds["broad_except_max"]


def test_error_taxonomy_details_can_list_sites() -> None:
    res = _run(["--json", "--details"])
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["raises"]
    assert payload["broad_handlers"]
    first = payload["raises"][0]
    assert {"path", "line", "exception", "category"} <= set(first)


def test_error_taxonomy_check_mode_passes() -> None:
    res = _run(["--check"])
    assert res.returncode == 0, res.stdout + res.stderr
    assert "[error_taxonomy_audit] OK" in res.stdout
