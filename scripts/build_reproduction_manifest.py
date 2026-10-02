#!/usr/bin/env python3
"""Machine-readable record of when and how each reference was re-derived.

The three reproducibility reports (R, Stata, StatsPAI) are Markdown
tables written by the ``verify_reproduce*`` scripts. This turns them, the
provenance block of every golden file and the CI workflow into one JSON
record per module and side:

* ``status`` and the worst relative gaps of the last re-derivation;
* ``reference`` -- R / Stata version, platform, package count, read from
  the golden file itself;
* ``input_sha256`` / ``golden_sha256`` -- the bytes that were compared;
* ``rederived_in_ci`` -- whether ``.github/workflows/r-parity.yml`` re-runs
  the module automatically (17 R modules) or it is a frozen artefact
  re-derived locally;
* ``skip_reason`` -- for a module with no Stata side, the measured reason
  registered in ``tests/r_parity/compare.py``;
* per side, ``last_reproduced`` -- the commit and date that last changed
  the report, which is when the full local re-derivation was last recorded.

Nothing is re-run here. A long gap since ``last_reproduced`` is not a
defect; it is a fact a reader is entitled to see.

Output: ``docs/reproduction_manifest.json``.

Usage
-----
    python scripts/build_reproduction_manifest.py           # regenerate
    python scripts/build_reproduction_manifest.py --check   # drift gate
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
R_DIR = REPO_ROOT / "tests" / "r_parity"
STATA_DIR = REPO_ROOT / "tests" / "stata_parity"
OUT = REPO_ROOT / "docs" / "reproduction_manifest.json"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "r-parity.yml"

SIDES = {
    "R": {
        "report": R_DIR / "results" / "REPRODUCIBILITY_REPORT.md",
        "golden": lambda m: R_DIR / "results" / f"{m}_R.json",
        "command": "python tests/r_parity/verify_reproduce.py",
    },
    "Stata": {
        "report": STATA_DIR / "results" / "REPRODUCIBILITY_REPORT_STATA.md",
        "golden": lambda m: STATA_DIR / "results" / f"{m}_Stata.json",
        "command": "python tests/stata_parity/verify_reproduce_stata.py",
    },
    "py": {
        "report": R_DIR / "results" / "REPRODUCIBILITY_REPORT_PY.md",
        "golden": lambda m: R_DIR / "results" / f"{m}_py.json",
        "command": "python tests/r_parity/verify_reproduce_py.py",
    },
}

_ROW = re.compile(r"^\| `(\d{2}_[A-Za-z0-9_]+)` \| (.+)$")


def _sha(path: pathlib.Path) -> Optional[str]:
    if not path.exists():
        return None
    # LF-normalised so a Windows checkout hashes the same bytes.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _float(cell: str) -> Optional[float]:
    try:
        return float(cell)
    except ValueError:
        return None


def _report_rows(path: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ROW.match(line)
        if not match:
            continue
        cells = [c.strip() for c in match.group(2).rstrip("|").split("|")]
        status = "reproduces" if "reproduces" in cells[0] else cells[0]
        numbers = [c for c in cells[1:] if re.fullmatch(r"[0-9.eE+-]+", c)]
        shared = next((c for c in cells[1:] if re.fullmatch(r"\d+/\d+", c)), None)
        rows[match.group(1)] = {
            "status": status,
            "statistics_shared": shared,
            "worst_rel_estimate": _float(numbers[0]) if numbers else None,
            "worst_rel_se": _float(numbers[1]) if len(numbers) > 1 else None,
        }
    return rows


def _reference(golden: pathlib.Path) -> Dict[str, Any]:
    if not golden.exists():
        return {}
    prov = json.loads(golden.read_text(encoding="utf-8")).get("provenance") or {}
    out: Dict[str, Any] = {}
    for key in ("r_version", "platform", "running", "stata_version", "edition", "os"):
        if key in prov:
            out[key] = prov[key]
    if isinstance(prov.get("packages"), dict):
        out["n_packages"] = len(prov["packages"])
    return out


def _ci_modules() -> List[str]:
    if not WORKFLOW.exists():
        return []
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index("verify_reproduce.py \\")
    return re.findall(
        r"\b(\d{2}_[a-z0-9_]+)\b", text[start : text.index("--timeout", start)]
    )


def _skip_reasons() -> Dict[str, str]:
    """``STATA_SKIP_REASON`` from compare.py, read without importing it."""
    tree = ast.parse((R_DIR / "compare.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign) and node.targets:
            target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "STATA_SKIP_REASON":
            try:
                return dict(ast.literal_eval(node.value))
            except (ValueError, SyntaxError):
                return {}
    return {}


def _last_commit(path: pathlib.Path) -> Optional[Dict[str, str]]:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%h %cI", "--", str(path)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            close_fds=False,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    sha, date = out.split(" ", 1)
    return {"commit": sha, "date": date}


def build(with_git: bool = True) -> Dict[str, Any]:
    ci = set(_ci_modules())
    skip = _skip_reasons()
    modules = sorted(
        p.name[: -len("_R.json")] for p in (R_DIR / "results").glob("*_R.json")
    )
    sides: Dict[str, Any] = {}
    for side, spec in SIDES.items():
        report = spec["report"]
        rows = _report_rows(report)
        entries: Dict[str, Any] = {}
        for module in modules:
            golden = spec["golden"](module)
            if not golden.exists():
                if side == "Stata":
                    entries[module] = {
                        "status": "no_reference",
                        "skip_reason": skip.get(module),
                    }
                continue
            entry: Dict[str, Any] = dict(rows.get(module, {"status": "not_in_report"}))
            if side != "py":
                # The StatsPAI-side result is rewritten at every release
                # (it carries the version string); hashing it here would
                # make this manifest stale for a reason that is not a
                # change of reference.
                entry["golden_sha256"] = _sha(golden)
                entry["input_sha256"] = _sha(R_DIR / "data" / f"{module}.csv")
                entry["reference"] = _reference(golden)
            if side == "R":
                entry["rederived_in_ci"] = module in ci
            entries[module] = entry
        sides[side] = {
            "report": report.relative_to(REPO_ROOT).as_posix(),
            "report_sha256": _sha(report),
            "rederive_with": spec["command"],
            "last_reproduced": _last_commit(report) if with_git else None,
            "n_modules": sum(
                1 for e in entries.values() if e["status"] != "no_reference"
            ),
            "n_reproduce": sum(
                1 for e in entries.values() if e["status"] == "reproduces"
            ),
            "modules": entries,
        }
    return {
        "schema": 1,
        "generated_by": "scripts/build_reproduction_manifest.py",
        "reproducibility_tolerance": 1e-9,
        "ci_workflow": ".github/workflows/r-parity.yml",
        "sides": sides,
    }


def _strip_dates(manifest: Dict[str, Any]) -> Dict[str, Any]:
    clean = json.loads(json.dumps(manifest))
    for side in clean["sides"].values():
        side["last_reproduced"] = None
    return clean


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = build()
    if args.check:
        if not OUT.exists():
            print("[reproduction_manifest] missing; run the script", file=sys.stderr)
            return 1
        committed = json.loads(OUT.read_text(encoding="utf-8"))
        # The date comes from git history, which a shallow clone lacks:
        # everything read from files is compared, the date is not.
        if _strip_dates(committed) != _strip_dates(manifest):
            print(
                "[reproduction_manifest] STALE -- run "
                "python scripts/build_reproduction_manifest.py",
                file=sys.stderr,
            )
            return 1
        print("[reproduction_manifest] OK")
        return 0
    OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for side, block in manifest["sides"].items():
        last = block["last_reproduced"] or {}
        print(
            f"[reproduction_manifest] {side}: {block['n_reproduce']}/"
            f"{block['n_modules']} reproduce, report last changed "
            f"{last.get('date', 'unknown')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
