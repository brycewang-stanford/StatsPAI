#!/usr/bin/env python3
"""Ratchet against smoke-only assertions in orchestration tests.

An orchestration tool — an MCP tool, a ``pipeline_*`` composite, an
agent transcript — reports success through a status field. Asserting only
that the field says "not an error" proves the call did not raise. It does
not prove the tool computed anything, and several tools shipped inert or
wrong behind exactly that assertion:

* ``sensitivity`` returned ``{"narrative": "**sensitivity**"}`` and nothing
  else, past ``assert isError is False``.
* ``sensitivity_from_result`` returned only ``source_result_id``, past
  ``assert payload["source_result_id"] == rid``.
* ``preflight`` returned FAIL on every MCP call, past
  ``assert "verdict" in payload``.
* ``pipeline_iv`` failed three of its four stages, past
  ``assert out["pipeline"] == "pipeline_iv"``.
* ``unified_sensitivity`` analysed the intercept, past
  ``assert sens["isError"] is False``.

The rule this enforces: a test function that asserts on a status/shape
signal must also assert on a *value* — a numeric comparison, a
``pytest.approx``, or an explicit check that no sub-step failed. It is a
ratchet, not a hard ban: the baseline records the functions that predate
it, and the count may only go down.

Usage
-----
    python scripts/orchestration_assertion_audit.py            # report
    python scripts/orchestration_assertion_audit.py --check    # CI gate
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
from typing import List, Set, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "scripts" / "orchestration_assertion_baseline.json"

#: Test files that drive multi-step orchestration surfaces.
WATCHED = (
    "tests/agent_eval/*.py",
    "tests/test_mcp_pipelines.py",
    "tests/test_mcp_*.py",
    "tests/test_paper_from_question.py",
)

#: Names whose presence in an assertion means "I only checked the shape".
_SHAPE_MARKERS = {"isError", "status", "verdict", "pipeline", "error"}

#: Call names that constitute a real value assertion.
_VALUE_CALLS = {"approx", "isclose", "allclose", "isfinite", "median", "mean"}


def _is_shape_assert(node: ast.expr) -> bool:
    """Does this assertion test a status/shape signal?"""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if sub.value in _SHAPE_MARKERS:
                return True
        if isinstance(sub, ast.Attribute) and sub.attr in _SHAPE_MARKERS:
            return True
        if isinstance(sub, ast.Name) and sub.id in _SHAPE_MARKERS:
            return True
    return False


def _is_value_assert(node: ast.expr) -> bool:
    """Does this assertion pin an actual value?"""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            name = getattr(sub.func, "attr", getattr(sub.func, "id", ""))
            if name in _VALUE_CALLS:
                return True
        # A numeric literal compared against something counts, as does a
        # comparison between two computed quantities.
        if isinstance(sub, ast.Compare):
            for side in [sub.left, *sub.comparators]:
                if (
                    isinstance(side, ast.Constant)
                    and isinstance(side.value, (int, float))
                    and not isinstance(side.value, bool)
                ):
                    return True
        # "no stage failed" is a value assertion about the whole run.
        if isinstance(sub, ast.Constant) and sub.value == "failed":
            return True
    # Pinning the *content of an error message* counts: an error-path
    # test whose job is "this returns a clean, explanatory envelope" does
    # real work when it asserts which explanation came back. The
    # exemption is deliberately narrow — comparing a status field to a
    # literal (``out["pipeline"] == "pipeline_did"``) is the very pattern
    # this audit exists to flag, so it must not qualify.
    if _touches_message_field(node):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Compare) and any(
                isinstance(op, (ast.In, ast.Eq)) for op in sub.ops
            ):
                for side in [sub.left, *sub.comparators]:
                    if (
                        isinstance(side, ast.Constant)
                        and isinstance(side.value, str)
                        and len(side.value) > 3
                    ):
                        return True
    return False


#: Fields that carry a human-readable explanation rather than a status.
_MESSAGE_FIELDS = {"error", "message", "hint", "detail", "reason", "msg"}


def _touches_message_field(node: ast.expr) -> bool:
    """Does the assertion read an explanatory message field?"""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr in _MESSAGE_FIELDS:
            return True
        if isinstance(sub, ast.Subscript):
            key = sub.slice
            if isinstance(key, ast.Constant) and key.value in _MESSAGE_FIELDS:
                return True
        if isinstance(sub, ast.Call):
            name = getattr(sub.func, "attr", getattr(sub.func, "id", ""))
            if name in {"get"} and sub.args:
                a0 = sub.args[0]
                if isinstance(a0, ast.Constant) and a0.value in _MESSAGE_FIELDS:
                    return True
    return False


def scan() -> List[Tuple[str, str]]:
    """Return ``(file, test_name)`` for every smoke-only test function."""
    offenders: List[Tuple[str, str]] = []
    seen: Set[pathlib.Path] = set()
    for pattern in WATCHED:
        for path in sorted(ROOT.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                if not node.name.startswith("test_"):
                    continue
                asserts = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]
                if not asserts:
                    continue
                shape = any(_is_shape_assert(a.test) for a in asserts)
                value = any(_is_value_assert(a.test) for a in asserts)
                if shape and not value:
                    offenders.append((str(path.relative_to(ROOT)), node.name))
    return offenders


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the count grew")
    parser.add_argument(
        "--write-baseline", action="store_true", help="record the current set"
    )
    args = parser.parse_args()

    offenders = scan()

    if args.write_baseline:
        BASELINE_PATH.write_text(
            json.dumps(
                {"allowed": [f"{f}::{t}" for f, t in offenders]},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[orchestration_assertion_audit] baseline: {len(offenders)}")
        return 0

    print("Orchestration assertion audit")
    print("=" * 60)
    print(f"  smoke-only test functions: {len(offenders)}")
    for f, t in offenders:
        print(f"    {f}::{t}")

    if not args.check:
        return 0

    baseline = set()
    if BASELINE_PATH.exists():
        baseline = set(json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["allowed"])
    current = {f"{f}::{t}" for f, t in offenders}
    new = sorted(current - baseline)
    if new:
        print("\n[orchestration_assertion_audit] REGRESSION", file=sys.stderr)
        for item in new:
            print(f"  {item}", file=sys.stderr)
        print(
            "\n  These tests assert a status/shape signal but never a value. "
            "A tool that returns nothing, or the wrong number, passes them. "
            "Add an assertion pinning a computed quantity — against a direct "
            "call, or that no sub-step reported 'failed'.",
            file=sys.stderr,
        )
        return 1
    fixed = sorted(baseline - current)
    if fixed:
        print(
            f"\n  {len(fixed)} baseline entries now assert values — "
            "re-run with --write-baseline to tighten the ratchet."
        )
    print("\n[orchestration_assertion_audit] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
