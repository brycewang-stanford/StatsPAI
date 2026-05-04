"""Print live registry / module statistics for StatsPAI.

This script is the **single source of truth** for the function-count,
submodule-count, and per-module breakdown numbers that appear in
``README.md``, ``README_CN.md``, ``CLAUDE.md``, ``docs/stats.md``, and
``docs/index.md``. Run it before a release (or when those numbers feel
stale) and copy the relevant figures into the markdown files.

Usage
-----
    python scripts/registry_stats.py                 # human-readable summary
    python scripts/registry_stats.py --table         # docs/stats.md per-module table
    python scripts/registry_stats.py --json          # machine-readable
    python scripts/registry_stats.py --check         # exit non-zero if README.md is stale

The ``--check`` mode lets CI flag drift between the live registry and
the loose floors quoted in ``README.md`` / ``README_CN.md``. It does not
require the floors to be exact; it only flags when the live count drops
below the documented floor (a regression) or exceeds it by enough that
the documented number is misleading.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "statspai"
TESTS_ROOT = REPO_ROOT / "tests"


def _module_loc_and_files() -> Dict[str, Tuple[int, int]]:
    """Walk src/statspai/<submodule>/ and count .py LOC + file count."""
    out: Dict[str, Tuple[int, int]] = {}
    for entry in sorted(SRC_ROOT.iterdir()):
        if not entry.is_dir() or entry.name.startswith("__"):
            continue
        loc = files = 0
        for path in entry.rglob("*.py"):
            files += 1
            try:
                loc += sum(1 for _ in path.open("r", encoding="utf-8"))
            except OSError:
                pass
        out[entry.name] = (loc, files)
    return out


def _tree_loc(root: Path) -> Tuple[int, int]:
    loc = files = 0
    for path in root.rglob("*.py"):
        files += 1
        try:
            loc += sum(1 for _ in path.open("r", encoding="utf-8"))
        except OSError:
            pass
    return loc, files


def _registered_per_module() -> Tuple[Counter, int]:
    """Map top-level submodule → number of registered ``sp.*`` symbols."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import statspai as sp  # noqa: WPS433  (side-effect import is the point)
    sp.list_functions()  # force full registry build
    from statspai.registry import _REGISTRY  # noqa: WPS433

    counts: Counter = Counter()
    for name in _REGISTRY:
        obj = getattr(sp, name, None)
        mod = getattr(obj, "__module__", "") or ""
        if mod.startswith("statspai."):
            counts[mod.split(".")[1]] += 1
    return counts, len(_REGISTRY)


def collect() -> dict:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import statspai as sp  # noqa: WPS433

    per_mod = _module_loc_and_files()
    fn_counts, total_fns = _registered_per_module()
    src_loc, src_files = _tree_loc(SRC_ROOT)
    tests_loc, tests_files = _tree_loc(TESTS_ROOT) if TESTS_ROOT.exists() else (0, 0)

    cat_counts: Counter = Counter()
    from statspai.registry import _REGISTRY  # noqa: WPS433
    for spec in _REGISTRY.values():
        cat_counts[spec.category] += 1

    return {
        "version": sp.__version__,
        "total_functions": total_fns,
        "total_submodules": len(per_mod),
        "src_loc": src_loc,
        "src_files": src_files,
        "tests_loc": tests_loc,
        "tests_files": tests_files,
        "per_module": {
            name: {
                "loc": loc,
                "files": files,
                "registered": fn_counts.get(name, 0),
            }
            for name, (loc, files) in per_mod.items()
        },
        "per_category": dict(cat_counts),
    }


# ----------------------------------------------------------------------- #
#  Renderers
# ----------------------------------------------------------------------- #

def render_summary(stats: dict) -> str:
    lines = [
        f"StatsPAI {stats['version']}",
        "=" * 40,
        f"Registered functions : {stats['total_functions']}",
        f"Submodules           : {stats['total_submodules']}",
        f"Source LOC           : {stats['src_loc']:,}  ({stats['src_files']} files)",
        f"Test LOC             : {stats['tests_loc']:,}  ({stats['tests_files']} files)",
        "",
        "Top categories",
        "--------------",
    ]
    for cat, n in sorted(stats["per_category"].items(), key=lambda x: -x[1]):
        lines.append(f"  {cat:<22} {n}")
    return "\n".join(lines)


def render_table(stats: dict) -> str:
    """Render the per-module markdown table for docs/stats.md."""
    rows = sorted(
        stats["per_module"].items(),
        key=lambda x: -x[1]["loc"],
    )
    lines = [
        "| Module              | LOC    | Files | Registered functions (`sp.*`) |",
        "| ------------------- | -----: | ----: | ----------------------------: |",
    ]
    for name, info in rows:
        lines.append(
            f"| `{name}` | {info['loc']:,} | {info['files']} | {info['registered']} |"
        )
    lines.append(
        f"| **Total** | **{stats['src_loc']:,}** | **{stats['src_files']}** | "
        f"**{stats['total_functions']}** |"
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------- #
#  Drift check
# ----------------------------------------------------------------------- #

# Loose floors we expect README.md to quote. The check passes as long as
# the live count is at or above the floor and within ``DRIFT_TOLERANCE``
# of it; otherwise the docs need refreshing.
README_FLOOR = 950          # README.md says "950+ functions"
SUBMODULE_FLOOR = 80        # README.md says "80+ submodules"
DRIFT_TOLERANCE = 100       # bump the floor once we're > floor + tolerance


def check_drift(stats: dict) -> int:
    issues = []
    fns = stats["total_functions"]
    mods = stats["total_submodules"]
    if fns < README_FLOOR:
        issues.append(
            f"Registered functions ({fns}) dropped below the README floor "
            f"({README_FLOOR}). Investigate or lower the floor."
        )
    if fns > README_FLOOR + DRIFT_TOLERANCE:
        issues.append(
            f"Registered functions ({fns}) exceed README floor "
            f"({README_FLOOR}) by > {DRIFT_TOLERANCE}. Bump the floor."
        )
    if mods < SUBMODULE_FLOOR:
        issues.append(
            f"Submodules ({mods}) dropped below README floor "
            f"({SUBMODULE_FLOOR}). Investigate or lower the floor."
        )
    if issues:
        for msg in issues:
            print(f"DRIFT: {msg}", file=sys.stderr)
        return 1
    print(
        f"OK: {fns} functions across {mods} submodules "
        f"(README floor: {README_FLOOR}+ / {SUBMODULE_FLOOR}+).",
    )
    return 0


# ----------------------------------------------------------------------- #
#  CLI
# ----------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table", action="store_true",
                        help="print the docs/stats.md per-module table")
    parser.add_argument("--json", action="store_true",
                        help="print machine-readable JSON")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if README quotes drift from reality")
    args = parser.parse_args(argv)

    stats = collect()
    if args.check:
        return check_drift(stats)
    if args.json:
        print(json.dumps(stats, indent=2))
        return 0
    if args.table:
        print(render_table(stats))
        return 0
    print(render_summary(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
