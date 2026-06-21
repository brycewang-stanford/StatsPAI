"""Contracts for the original-data parity harness.

These tests do not require R or Stata. They make the committed
``tests/orig_parity/results`` artifacts executable enough that a source
change cannot leave the Python-side JSONs silently stale.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "tests" / "orig_parity"
RESULTS = ORIG / "results"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_orig_harness(tmp_path: Path) -> Path:
    dest = tmp_path / "orig_parity"
    shutil.copytree(
        ORIG,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return dest


def _rows_by_stat(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["statistic"]: row for row in payload["rows"]}


def _assert_close(a: float | None, b: float | None, *, atol: float = 1e-9) -> None:
    if a is None or b is None:
        assert a is b
        return
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert abs(float(a) - float(b)) <= atol


def test_orig_parity_json_pairs_keep_joinable_schema():
    py_files = sorted(RESULTS.glob("*_py.json"))
    r_files = sorted(RESULTS.glob("*_R.json"))
    assert len(py_files) >= 11
    assert len(r_files) >= 11
    assert {p.stem[:-3] for p in py_files} == {p.stem[:-2] for p in r_files}

    for path in [*py_files, *r_files]:
        payload = _read_json(path)
        suffix = "_py" if path.name.endswith("_py.json") else "_R"
        side = "py" if suffix == "_py" else "R"
        module = path.stem[: -len(suffix)]

        assert payload["module"] == module
        assert payload["side"] == side
        assert payload["rows"], f"{path.name} has no rows"

        seen: set[str] = set()
        for row in payload["rows"]:
            assert row["module"] == module
            assert row["side"] == side
            assert isinstance(row["statistic"], str) and row["statistic"]
            assert row["statistic"] not in seen
            seen.add(row["statistic"])
            for key in ("estimate", "se", "published"):
                value = row.get(key)
                assert value is None or isinstance(value, (int, float))
                assert not isinstance(value, float) or math.isfinite(value)


def test_orig_parity_rollup_is_reproducible_from_committed_json(tmp_path):
    work = _copy_orig_harness(tmp_path)
    subprocess.run(
        [sys.executable, "compare_orig.py"],
        cwd=work,
        check=True,
        text=True,
        capture_output=True,
    )
    got = (work / "results" / "parity_table_orig.md").read_text(encoding="utf-8")
    expected = (RESULTS / "parity_table_orig.md").read_text(encoding="utf-8")
    assert got == expected


def test_nhefs_ch12_python_artifact_recomputes_from_live_code(tmp_path):
    """Regression guard for the IPW solver drift caught in June 2026."""
    work = _copy_orig_harness(tmp_path)
    subprocess.run(
        [sys.executable, "06_nhefs_ch12_ipw.py"],
        cwd=work,
        check=True,
        text=True,
        capture_output=True,
    )

    fresh = _read_json(work / "results" / "06_nhefs_ch12_ipw_py.json")
    committed = _read_json(RESULTS / "06_nhefs_ch12_ipw_py.json")
    r_gold = _read_json(RESULTS / "06_nhefs_ch12_ipw_R.json")

    fresh_rows = _rows_by_stat(fresh)
    committed_rows = _rows_by_stat(committed)
    r_rows = _rows_by_stat(r_gold)
    assert set(fresh_rows) == set(committed_rows)

    for stat, row in fresh_rows.items():
        expected = committed_rows[stat]
        _assert_close(row.get("estimate"), expected.get("estimate"))
        _assert_close(row.get("se"), expected.get("se"))
        assert row.get("n") == expected.get("n")

    _assert_close(
        fresh_rows["ipw_att"]["estimate"],
        r_rows["ipw_att"]["estimate"],
        atol=1e-6,
    )
    _assert_close(fresh["extra"]["ci_ipw"][0], committed["extra"]["ci_ipw"][0])
    _assert_close(fresh["extra"]["ci_ipw"][1], committed["extra"]["ci_ipw"][1])
