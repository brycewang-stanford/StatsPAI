#!/usr/bin/env python3
"""Coverage-campaign helper for the core-module ≥95% initiative.

This tool exists because StatsPAI's coverage can only be measured reliably with
the *whole-package* source (``--cov=statspai``). Scoping ``--cov`` to a
sub-package (e.g. ``--cov=statspai.iv``) reorders imports relative to the
``scipy.optimize`` PyO3/pybind11 stabiliser in ``tests/conftest.py`` and
crashes with ``generic_type: type "ObjSense" is already registered``. So we
always measure with ``--cov=statspai`` and slice per-module numbers out of the
Cobertura XML here.

Usage
-----
    # Per-module % for the six core modules, from any coverage XML:
    python scripts/coverage_campaign.py report [--xml PATH]

    # Exact uncovered line ranges for one module (what still needs a test):
    python scripts/coverage_campaign.py gaps MODULE [--xml PATH] [--per-file]

The default XML is ``.coverage_campaign/baseline_fullsuite.xml`` if present,
else the repo-root committed ``coverage.xml``.

This script is read-only over coverage data; it never runs pytest itself so it
stays fast and side-effect free. Generate a fresh XML with, e.g.::

    pytest tests/ -q --cov-report=xml:.coverage_campaign/latest.xml --cov-report=

Note the authoritative metric is the *full-suite* number: a module is exercised
by many cross-module tests, so running only ``tests/test_<mod>*.py`` understates
its true coverage substantially (iv: 38% module-only vs 87% full-suite).
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

CORE_MODULES = ("did", "iv", "rd", "synth", "dml", "panel")
TARGET = 95.0


def _default_xml() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand = os.path.join(here, ".coverage_campaign", "baseline_fullsuite.xml")
    if os.path.exists(cand):
        return cand
    return os.path.join(here, "coverage.xml")


def _module_of(filename: str) -> str | None:
    parts = filename.replace("\\", "/").split("/")
    if "statspai" not in parts:
        return None
    i = parts.index("statspai")
    if i + 1 >= len(parts):
        return None
    name = parts[i + 1]
    return name[:-3] if name.endswith(".py") else name


def _iter_classes(xml_path: str):
    root = ET.parse(xml_path).getroot()
    yield from root.iter("class")


def cmd_report(args) -> int:
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # mod -> [covered, total]
    for cls in _iter_classes(args.xml):
        mod = _module_of(cls.get("filename", ""))
        if mod is None:
            continue
        lines = cls.find("lines")
        if lines is None:
            continue
        for ln in lines.findall("line"):
            agg[mod][1] += 1
            if int(ln.get("hits", "0")) > 0:
                agg[mod][0] += 1

    print(f"# coverage source: {args.xml}\n")
    print(f"{'module':10}{'cov%':>8}{'covered':>10}{'total':>8}{'+to 95%':>10}{'':>4}")
    print("-" * 50)
    threshold = getattr(args, "min", None) or TARGET
    below = []
    for m in CORE_MODULES:
        c, tot = agg[m]
        pct = 100 * c / tot if tot else 0.0
        need = max(0, int(__import__("math").ceil(threshold / 100 * tot)) - c)
        ok = pct >= threshold
        if not ok:
            below.append((m, pct))
        print(f"{m:10}{pct:8.1f}{c:10}{tot:8}{need:10}{'OK' if ok else 'LOW':>4}")

    if getattr(args, "check", False):
        if below:
            print(
                "\nFAIL: core-module coverage ratchet — below "
                f"{threshold:.1f}%: "
                + ", ".join(f"{m} ({p:.1f}%)" for m, p in below)
            )
            return 1
        print(f"\nOK: all {len(CORE_MODULES)} core modules ≥ {threshold:.1f}%")
    return 0


def cmd_gaps(args) -> int:
    mod = args.module
    per_file: dict[str, list[int]] = defaultdict(list)
    for cls in _iter_classes(args.xml):
        fn = cls.get("filename", "")
        if _module_of(fn) != mod:
            continue
        lines = cls.find("lines")
        if lines is None:
            continue
        miss = [
            int(ln.get("number"))
            for ln in lines.findall("line")
            if int(ln.get("hits", "0")) == 0
        ]
        if miss:
            per_file[fn] = sorted(miss)

    def _ranges(nums: list[int]) -> str:
        out, start, prev = [], None, None
        for n in nums:
            if start is None:
                start = prev = n
            elif n == prev + 1:
                prev = n
            else:
                out.append(f"{start}" if start == prev else f"{start}-{prev}")
                start = prev = n
        if start is not None:
            out.append(f"{start}" if start == prev else f"{start}-{prev}")
        return ", ".join(out)

    print(f"# uncovered lines for module '{mod}'  (source: {args.xml})\n")
    total_missing = 0
    for fn in sorted(per_file, key=lambda f: -len(per_file[f])):
        miss = per_file[fn]
        total_missing += len(miss)
        short = fn.replace("\\", "/").split("statspai/", 1)[-1]
        if args.per_file:
            print(f"{len(miss):5}  statspai/{short}\n        {_ranges(miss)}")
        else:
            print(f"{len(miss):5}  statspai/{short}")
    print(f"\n  total uncovered: {total_missing} lines across {len(per_file)} files")
    return 0


def _covered_sets(xml_path: str, mod: str) -> dict:
    """Map ``filename -> set(covered line numbers)`` for one module's files."""
    out: dict = defaultdict(set)
    for cls in _iter_classes(xml_path):
        fn = cls.get("filename", "")
        if _module_of(fn) != mod:
            continue
        short = fn.replace("\\", "/").split("statspai/", 1)[-1]
        lines = cls.find("lines")
        if lines is None:
            continue
        for ln in lines.findall("line"):
            if int(ln.get("hits", "0")) > 0:
                out[short].add(int(ln.get("number")))
    return out


def _line_totals(xml_path: str, mod: str) -> dict:
    """Map ``filename -> set(all measured line numbers)`` for one module."""
    out: dict = defaultdict(set)
    for cls in _iter_classes(xml_path):
        fn = cls.get("filename", "")
        if _module_of(fn) != mod:
            continue
        short = fn.replace("\\", "/").split("statspai/", 1)[-1]
        lines = cls.find("lines")
        if lines is None:
            continue
        for ln in lines.findall("line"):
            out[short].add(int(ln.get("number")))
    return out


def cmd_union(args) -> int:
    """Post-work full-suite coverage for one module, computed *fast* as

        baseline_covered ∪ (module-only-tests covered)

    The baseline full-suite XML already includes every cross-module test's
    contribution to this module, so any line the new module-scoped tests cover
    is purely additive — the union equals the true post-work full-suite number
    without re-running the whole ~6.5k-test suite (which takes >1h).
    """
    mod = args.module
    base_cov = _covered_sets(args.baseline, mod)
    new_cov = _covered_sets(args.tests_xml, mod)
    totals = _line_totals(args.baseline, mod)

    union_covered = 0
    total = 0
    gained = 0
    for fn, all_lines in totals.items():
        total += len(all_lines)
        u = base_cov.get(fn, set()) | new_cov.get(fn, set())
        u &= all_lines  # stay within measured lines
        union_covered += len(u)
        gained += len(u - base_cov.get(fn, set()))

    pct = 100 * union_covered / total if total else 0.0
    base_pct = 100 * sum(len(s) for s in base_cov.values()) / total if total else 0.0
    need = max(0, int(__import__("math").ceil(TARGET / 100 * total)) - union_covered)
    print(f"# module '{mod}'  (baseline={args.baseline}, tests={args.tests_xml})")
    print(f"  baseline full-suite : {base_pct:.1f}%")
    print(f"  + new module tests  : +{gained} lines newly covered")
    print(f"  => post-work est.   : {pct:.1f}%  ({union_covered}/{total})")
    print(f"  {'OK ✓ ≥95%' if pct >= TARGET else f'still need +{need} lines to 95%'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("report", help="per-module coverage for the 6 core modules")
    pr.add_argument("--xml", default=_default_xml())
    pr.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any core module is below the threshold "
        "(CI ratchet against silent coverage regressions)",
    )
    pr.add_argument(
        "--min",
        type=float,
        default=None,
        help=f"coverage threshold for --check (default {TARGET})",
    )
    pr.set_defaults(func=cmd_report)

    pu = sub.add_parser(
        "union", help="fast post-work module coverage = baseline ∪ module-test XML"
    )
    pu.add_argument("module", choices=CORE_MODULES)
    pu.add_argument(
        "--baseline",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "coverage.xml"
        ),
    )
    pu.add_argument(
        "--tests-xml",
        required=True,
        help="coverage XML from running just this module's test files",
    )
    pu.set_defaults(func=cmd_union)

    pg = sub.add_parser("gaps", help="uncovered line ranges for one module")
    pg.add_argument("module", choices=CORE_MODULES)
    pg.add_argument("--xml", default=_default_xml())
    pg.add_argument(
        "--per-file",
        action="store_true",
        help="also print the exact line ranges per file",
    )
    pg.set_defaults(func=cmd_gaps)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
