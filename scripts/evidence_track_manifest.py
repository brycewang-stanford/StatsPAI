#!/usr/bin/env python3
"""One manifest over every evidence track: what is stored, who reads it,
what locks it.

StatsPAI keeps numerical evidence in six places, each with its own
conventions. This script reads them all and reports, per track, the data
files it holds, which hash lock covers each file, and which files no
test, generator or harness names. It adds no new lock: it reads the ones
that exist, so it cannot be made stale by an unrelated commit.

==========================  ============================================
track                       what it holds
==========================  ============================================
``tests/r_parity``          Track A: CSV inputs, R and Python results
``tests/stata_parity``      Track A Stata results; option-level fixtures
``tests/reference_parity``  ``_fixtures/``: references read by pytest
``tests/orig_parity``       original-data ledger: inputs and results
``tests/external_parity``   published-number checks
``tests/coverage_monte_carlo``  Track B coverage and mechanism runs
==========================  ============================================

Locks read: ``tests/r_parity/TIER_A_FIXTURE_LOCK.json`` (Track A),
``tests/jss_review_freeze.json`` (what the JSS manuscript tabulates) and
``docs/evidence_inventory.json`` (option-level Stata fixtures).

Usage
-----
    python scripts/evidence_track_manifest.py           # table
    python scripts/evidence_track_manifest.py --json    # machine-readable
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = REPO_ROOT / "tests"

TRACKS = (
    "r_parity",
    "stata_parity",
    "reference_parity",
    "orig_parity",
    "external_parity",
    "coverage_monte_carlo",
)

DATA_SUFFIXES = {".json", ".csv", ".dta", ".rds", ".txt", ".parquet"}
#: Not evidence: scratch output, the locks themselves, package manifests.
SKIP_PARTS = {"__pycache__", "_repro_check", "_ado_fect", "renv"}
SKIP_NAMES = {"TIER_A_FIXTURE_LOCK.json", "renv.lock"}
CODE_SUFFIXES = {".py", ".R", ".r", ".do"}


def _data_files(track: str) -> List[pathlib.Path]:
    out = []
    for path in sorted((TESTS / track).rglob("*")):
        if not path.is_file() or path.suffix not in DATA_SUFFIXES:
            continue
        if SKIP_PARTS & set(path.parts) or path.name in SKIP_NAMES:
            continue
        out.append(path)
    return out


def _code_text() -> str:
    chunks = []
    for base in (TESTS, REPO_ROOT / "scripts"):
        for path in base.rglob("*"):
            if path.suffix in CODE_SUFFIXES and "__pycache__" not in path.parts:
                chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def _named(path: pathlib.Path, code: str) -> bool:
    """Is the file named by a test, a harness or a generator?

    Exactly, or through the stem a harness builds its name from: Track A
    files are ``<module>_<side>.json`` / ``<module>.csv`` and are opened by
    module name; several fixtures are written as ``<prefix>_<variant>``.
    """
    if path.name in code:
        return True
    stem = path.stem
    for suffix in ("_py", "_R", "_Stata"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)] in code
    return stem in code or stem.rsplit("_", 1)[0] in code


def build() -> Dict[str, Any]:
    tier_a = ""
    lock_path = TESTS / "r_parity" / "TIER_A_FIXTURE_LOCK.json"
    if lock_path.exists():
        tier_a = lock_path.read_text(encoding="utf-8")
    freeze: Dict[str, Any] = {}
    freeze_path = TESTS / "jss_review_freeze.json"
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8")).get("files", {})
    inventory = ""
    inv_path = REPO_ROOT / "docs" / "evidence_inventory.json"
    if inv_path.exists():
        inventory = inv_path.read_text(encoding="utf-8")
    code = _code_text()

    tracks = {}
    for track in TRACKS:
        files = _data_files(track)
        rows = []
        for path in files:
            rel = path.relative_to(REPO_ROOT).as_posix()
            locks = []
            if rel in tier_a or f'"{path.name}"' in tier_a:
                locks.append("tier_a_lock")
            if rel in freeze:
                locks.append("jss_freeze")
            if rel in inventory:
                locks.append("evidence_inventory")
            rows.append({"path": rel, "locks": locks, "named": _named(path, code)})
        tracks[track] = {
            "files": len(rows),
            "locked": sum(1 for r in rows if r["locks"]),
            "unlocked": sorted(r["path"] for r in rows if not r["locks"]),
            "unnamed": sorted(r["path"] for r in rows if not r["named"]),
            "by_lock": {
                lock: sum(1 for r in rows if lock in r["locks"])
                for lock in ("tier_a_lock", "jss_freeze", "evidence_inventory")
            },
        }
    return {"schema": 1, "tracks": tracks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--unlocked", action="store_true", help="also list every unlocked file"
    )
    args = parser.parse_args()
    report = build()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    header = f"{'track':<22}{'files':>6}{'locked':>8}{'tier A':>8}{'JSS':>6}{'inv.':>6}{'unnamed':>9}"
    print(header)
    for name, track in report["tracks"].items():
        by = track["by_lock"]
        print(
            f"{name:<22}{track['files']:>6}{track['locked']:>8}"
            f"{by['tier_a_lock']:>8}{by['jss_freeze']:>6}"
            f"{by['evidence_inventory']:>6}{len(track['unnamed']):>9}"
        )
    for name, track in report["tracks"].items():
        for path in track["unnamed"]:
            print(f"unnamed: {path}")
        if args.unlocked:
            for path in track["unlocked"]:
                print(f"unlocked: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
