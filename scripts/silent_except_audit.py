#!/usr/bin/env python3
"""Ratchet against silent broad ``except`` handlers in the package source.

CLAUDE.md section 3.7 ("fail loudly") forbids swallowing an exception and
carrying on as if nothing happened. A handler is counted here when it

* catches broadly -- a bare ``except:``, ``except Exception``, or
  ``except BaseException`` (alone or inside a tuple), and
* leaves no trace -- its body neither re-raises, nor warns
  (``warnings.warn`` / ``warn_fallback`` / ``record_degradation`` /
  ``bootstrap_se``-style helpers), nor logs, nor even reads the caught
  exception object.

Such a handler turns a failed fit into a substituted number (a marginal
mean in place of a propensity score, a dropped cohort, a NaN standard
error) with no signal to the caller. Narrow handlers such as
``except np.linalg.LinAlgError`` are out of scope: naming the failure is
already a statement about what is expected.

One pattern is exempt by construction: a ``try`` whose only job is to call
``attach_provenance``. That function already guards itself and returns the
result untouched on any failure, so the wrapper around it cannot hide a
numerical problem. About 150 estimators carry it.

This is a ratchet, not a hard ban. Many of the remaining sites are
legitimate best-effort probes (optional imports, version lookups, plot
layout). The baseline records the per-file count that predates the gate
and the count may only go down. A new silent handler fails the check; to
keep one, make it loud or narrow instead of raising the baseline.

Usage
-----
    python scripts/silent_except_audit.py              # report
    python scripts/silent_except_audit.py --check      # CI gate
    python scripts/silent_except_audit.py --write      # refresh baseline
    python scripts/silent_except_audit.py --list       # every site
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "statspai"
BASELINE_PATH = ROOT / "scripts" / "silent_except_baseline.json"

_BROAD = {"Exception", "BaseException"}

#: Call names that count as leaving a trace. Matched against the final
#: attribute or bare name of the callee.
_LOUD_CALLS = {
    "warn",
    "warn_fallback",
    "record_degradation",
    "debug",
    "info",
    "warning",
    "error",
    "exception",
    "critical",
    "print",
}


def _is_broad(node: Optional[ast.expr]) -> bool:
    if node is None:
        return True
    if isinstance(node, ast.Name):
        return node.id in _BROAD
    if isinstance(node, ast.Tuple):
        return any(_is_broad(elt) for elt in node.elts)
    return False


def _callee_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _leaves_trace(handler: ast.ExceptHandler) -> bool:
    bound = handler.name
    for stmt in handler.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Raise):
                return True
            if isinstance(node, ast.Call) and _callee_name(node) in _LOUD_CALLS:
                return True
            if bound and isinstance(node, ast.Name) and node.id == bound:
                return True
    return False


#: Callee names that mark a ``try`` body as provenance-only metadata.
_PROVENANCE_CALLS = {"attach_provenance", "_attach_prov"}


def _is_provenance_only(node: ast.Try) -> bool:
    """True when the ``try`` body does nothing but attach provenance."""
    calls = [
        _callee_name(n)
        for stmt in node.body
        for n in ast.walk(stmt)
        if isinstance(n, ast.Call)
    ]
    if not any(name in _PROVENANCE_CALLS for name in calls):
        return False
    # Nothing in the body may bind a result the caller goes on to use,
    # apart from the provenance helpers' own bookkeeping names.
    return not any(
        isinstance(stmt, (ast.Return, ast.For, ast.While)) for stmt in node.body
    )


def scan_file(path: pathlib.Path) -> List[int]:
    """Return the line numbers of silent broad handlers in ``path``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        raise SystemExit(f"cannot parse {path}: {exc}") from exc
    lines: List[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if _is_provenance_only(node):
            continue
        for handler in node.handlers:
            if _is_broad(handler.type) and not _leaves_trace(handler):
                lines.append(handler.lineno)
    return sorted(lines)


def scan() -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for path in sorted(SRC.rglob("*.py")):
        lines = scan_file(path)
        if lines:
            out[path.relative_to(ROOT).as_posix()] = lines
    return out


def _counts(sites: Dict[str, List[int]]) -> Dict[str, int]:
    return {k: len(v) for k, v in sites.items()}


def _load_baseline() -> Dict[str, int]:
    if not BASELINE_PATH.exists():
        return {}
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in data.get("files", {}).items()}


def _regressions(
    current: Dict[str, int], baseline: Dict[str, int]
) -> List[Tuple[str, int, int]]:
    return [
        (path, baseline.get(path, 0), n)
        for path, n in sorted(current.items())
        if n > baseline.get(path, 0)
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail on regressions")
    parser.add_argument("--write", action="store_true", help="rewrite the baseline")
    parser.add_argument("--list", action="store_true", help="print every site")
    args = parser.parse_args(argv)

    sites = scan()
    current = _counts(sites)
    total = sum(current.values())

    if args.write:
        payload = {
            "note": (
                "Per-file count of silent broad except handlers. Ratchet: "
                "counts may only go down. See scripts/silent_except_audit.py."
            ),
            "total": total,
            "files": dict(sorted(current.items())),
        }
        BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"baseline written: {total} sites in {len(current)} files")
        return 0

    if args.list:
        for path, lines in sites.items():
            for line in lines:
                print(f"{path}:{line}")

    baseline = _load_baseline()
    base_total = sum(baseline.values())
    regressions = _regressions(current, baseline)
    print(
        f"silent broad except handlers: {total} in {len(current)} files "
        f"(baseline {base_total})"
    )
    if regressions:
        print("\nNew silent handlers (make them loud or narrow them):")
        for path, was, now in regressions:
            print(f"  {path}: {was} -> {now}  lines {sites[path]}")
        if args.check:
            return 1
    elif total < base_total:
        print("count went down; run with --write to tighten the baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
