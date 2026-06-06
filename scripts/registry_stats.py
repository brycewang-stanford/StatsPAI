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
    python scripts/registry_stats.py --check         # exit non-zero if docs are stale

The ``--check`` mode lets CI flag drift between the live registry and
the user-facing docs. It gates the **registry/API surface** (per-module
registered-function counts + the README floors) exactly, but treats
**size metrics** (LOC) as a vanity metric (see ``docs/stats.md`` §5):
LOC is only flagged once it drifts past ``LOC_DRIFT_TOLERANCE``, so
routine source churn (a refactor, a dead-code removal) never reds CI.
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
DOCS_STATS = REPO_ROOT / "docs" / "stats.md"
DOCS_INDEX = REPO_ROOT / "docs" / "index.md"
DOCS_REFERENCE_INDEX = REPO_ROOT / "docs" / "reference" / "index.md"
README = REPO_ROOT / "README.md"
README_CN = REPO_ROOT / "README_CN.md"


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
README_FLOOR = 1000         # README.md says "1,000+ functions"
SUBMODULE_FLOOR = 80        # README.md says "80 submodules"
DRIFT_TOLERANCE = 100       # bump the floor once we're > floor + tolerance
# LOC is a vanity metric (docs/stats.md §5) that moves on every commit, so
# we only flag it once the docs lag the live tree by more than this many
# lines — a new module / large deletion trips it; a refactor does not.
#
# TEMPORARILY RAISED 3000 -> 60000 for the core-module ≥95% coverage
# campaign (see ``.coverage_campaign/CAMPAIGN.md``): adding tens of
# thousands of test lines across did/iv/rd/synth/dml/panel inflates the
# *test* LOC monotonically and far faster than the normal 3k band, which
# would otherwise red the ``parity-guards`` CI on every coverage commit.
# This relaxes ONLY the vanity LOC band — the registered-function and
# submodule floors/exact-counts stay enforced. Restore to 3000 and run a
# single ``registry_stats.py --table`` + README/README_CN refresh once the
# campaign completes and the test-LOC total stabilises.
LOC_DRIFT_TOLERANCE = 60000


def _module_fn_counts(table_text: str) -> Dict[str, int]:
    """Extract ``{module: registered_fn_count}`` from a per-module md table.

    Only the module name and the trailing registered-function count are
    read; the LOC and file columns are intentionally ignored so that
    source churn (which moves LOC every commit) never trips the gate —
    only a real change in the ``sp.*`` surface does. The ``**Total**`` row
    has no backtick-wrapped module name and is skipped by design (the
    aggregate count is already guarded by ``README_FLOOR``).
    """
    out: Dict[str, int] = {}
    for match in re.finditer(
        r"\|\s*`(\w+)`\s*\|\s*[\d,]+\s*\|\s*\d+\s*\|\s*(\d+)\s*\|",
        table_text,
    ):
        out[match.group(1)] = int(match.group(2))
    return out


def _loc_within_tolerance(written: str, live: int) -> bool:
    return abs(int(written.replace(",", "")) - live) <= LOC_DRIFT_TOLERANCE


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
    try:
        stats_doc = DOCS_STATS.read_text(encoding="utf-8")
    except OSError as exc:
        issues.append(f"Could not read {DOCS_STATS.relative_to(REPO_ROOT)}: {exc}")
    else:
        # API surface: every module's registered-function count must match
        # the live registry exactly (LOC/file columns are ignored).
        expected_fns = {
            name: info["registered"]
            for name, info in stats["per_module"].items()
        }
        actual_fns = _module_fn_counts(stats_doc)
        if actual_fns != expected_fns:
            drifted = sorted(
                name for name in set(expected_fns) | set(actual_fns)
                if expected_fns.get(name) != actual_fns.get(name)
            )
            issues.append(
                "docs/stats.md per-module registered-function counts are "
                f"stale (modules: {', '.join(drifted) or '—'}); regenerate "
                "with `python scripts/registry_stats.py --table`."
            )
        # Size metrics: tolerant — only flag gross LOC drift.
        src_match = re.search(
            r"\| \*\*StatsPAI\*\* `src/statspai/`\s*\| measured\s*\|\s*"
            r"[\d,]+\s*\|\s*\*\*([\d,]+)\*\*\s*\|",
            stats_doc,
        )
        if src_match is None:
            issues.append("docs/stats.md source LOC row is missing or malformed.")
        elif not _loc_within_tolerance(src_match.group(1), stats["src_loc"]):
            issues.append(
                f"docs/stats.md source LOC ({src_match.group(1)}) drifts from "
                f"live ({stats['src_loc']:,}) by > {LOC_DRIFT_TOLERANCE}; "
                "regenerate with `python scripts/registry_stats.py --table`."
            )
        test_match = re.search(
            r"\| StatsPAI tests \(`tests/`\)\s*\| measured\s*\|\s*"
            r"[\d,]+\s*\|\s*([\d,]+)\s*\|",
            stats_doc,
        )
        if test_match is None:
            issues.append("docs/stats.md test LOC row is missing or malformed.")
        elif not _loc_within_tolerance(test_match.group(1), stats["tests_loc"]):
            issues.append(
                f"docs/stats.md test LOC ({test_match.group(1)}) drifts from "
                f"live ({stats['tests_loc']:,}) by > {LOC_DRIFT_TOLERANCE}; "
                "regenerate with `python scripts/registry_stats.py --table`."
            )

    fns_text = f"{fns:,}"
    src_k = (stats["src_loc"] + 500) // 1000
    test_k = (stats["tests_loc"] + 500) // 1000
    readme_expectations = {
        README: (
            f"{fns_text} registered functions",
            f"{mods} submodules",
        ),
        README_CN: (
            f"{fns_text} 个注册函数",
            f"{mods} 个子模块",
        ),
        DOCS_INDEX: (
            "1,000+ registered functions",
            f"across {mods} submodules",
        ),
        DOCS_REFERENCE_INDEX: (
            f"{fns_text} registered public functions",
        ),
    }
    for path, snippets in readme_expectations.items():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            issues.append(f"Could not read {path.relative_to(REPO_ROOT)}: {exc}")
            continue
        text_norm = re.sub(r"\s+", " ", text)
        missing = [
            snippet for snippet in snippets
            if snippet not in text
            and re.sub(r"\s+", " ", snippet) not in text_norm
        ]
        if missing:
            issues.append(
                f"{path.relative_to(REPO_ROOT)} at-a-glance line is stale; "
                "missing " + ", ".join(repr(item) for item in missing)
            )

    # README LOC figures: vanity metric — checked numerically against the
    # same LOC_DRIFT_TOLERANCE band as docs/stats.md above, NOT by exact
    # "Nk" string presence. This stops the monotonic test-LOC growth of the
    # coverage campaign from reding CI on every commit; the quote only trips
    # the gate on gross drift. (The function/submodule snippets checked
    # above stay exact — those are the surface guards that matter.)
    loc_patterns = {
        README: r"(\d+)k LOC \(core\) \+ (\d+)k LOC \(tests\)",
        README_CN: r"(\d+)k 行核心代码 \+ (\d+)k 行测试",
    }
    for path, pat in loc_patterns.items():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue  # unreadable file already reported in the snippet loop
        m = re.search(pat, text)
        if m is None:
            issues.append(
                f"{path.relative_to(REPO_ROOT)} at-a-glance LOC figure is "
                "missing or malformed (want 'Nk LOC (core) + Mk LOC (tests)')."
            )
            continue
        written_core = int(m.group(1)) * 1000
        written_test = int(m.group(2)) * 1000
        if (abs(written_core - stats["src_loc"]) > LOC_DRIFT_TOLERANCE
                or abs(written_test - stats["tests_loc"]) > LOC_DRIFT_TOLERANCE):
            issues.append(
                f"{path.relative_to(REPO_ROOT)} at-a-glance LOC "
                f"({m.group(1)}k core / {m.group(2)}k tests) drifts from live "
                f"({src_k}k / {test_k}k) by > {LOC_DRIFT_TOLERANCE} lines."
            )
    if issues:
        for msg in issues:
            print(f"DRIFT: {msg}", file=sys.stderr)
        return 1
    print(
        f"OK: {fns} functions across {mods} submodules "
        f"(README floor: {README_FLOOR}+ / {SUBMODULE_FLOOR}).",
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
