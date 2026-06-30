"""Audit generic exceptions versus the StatsPAI exception taxonomy.

The public package now exposes agent-native exceptions such as
``DataInsufficient`` and ``IdentificationFailure``.  This static audit makes
the migration measurable by counting:

* raises of taxonomy exceptions,
* raises of generic built-ins such as ``ValueError`` / ``RuntimeError``, and
* broad ``except Exception`` handlers.

The script parses source with ``ast`` and does not import ``statspai``.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "statspai"

TAXONOMY_EXCEPTIONS = {
    "StatsPAIError",
    "AssumptionViolation",
    "IdentificationFailure",
    "DataInsufficient",
    "ConvergenceFailure",
    "NumericalInstability",
    "MethodIncompatibility",
}

GENERIC_EXCEPTIONS = {
    "ValueError",
    "RuntimeError",
    "TypeError",
    "KeyError",
    "IndexError",
    "NotImplementedError",
    "Exception",
}

# Ratchet thresholds.  Generic/broad counts should trend down as domain
# failures migrate to the package exception hierarchy; taxonomy raises should
# increase.  The generic ceiling tracks *idiomatic* input/argument validation
# (``ValueError``/``TypeError`` for bad args), which is correctly generic and
# not migratable to the domain taxonomy.  The native sdid/augsynth/gsynth/
# rddensity ports added such validation, so the ceiling is bumped to the
# current count to keep catching unjustified future growth (taxonomy raises
# rose in tandem, so the migration trend is intact).  The recent correctness
# fixes (diagnostics/did/bartik/results clean-error guards) added a further 5
# idiomatic arg-validation / defensive raises, so the ceiling moves 1842 -> 1847
# while taxonomy raises held at 62 (>> the 42 floor) — migration trend intact.
# v1.17.0: the drdid traditional-method correctness fix added one idiomatic
# arg-validation raise (`ValueError` on a `method` other than 'imp'/'trad'),
# so the ceiling moves 1847 -> 1848 while taxonomy raises stayed >> the floor.
# v1.17.0 (#21): the public-health validation modules (evalue rewrite,
# survival/competing_risks, inference/meta_analysis, power/study_designs) added
# 26 idiomatic arg-validation / defensive raises, so the ceiling moves
# 1848 -> 1874 while taxonomy raises rose 62 -> 65 (still >> the 42 floor) —
# the migration trend is intact.
# v1.17.x parity hardening adds generic argument/shape/convention guards across
# RIF dineq matching, DML explicit folds, fixest SSC, SCM shape checks, ARIMA
# method selection, and adjacent parity-harness validation. Domain failures in
# the same batch were migrated to StatsPAI taxonomy exceptions, raising taxonomy
# sites to 76. Keep this as a ratchet baseline for the hardened parity surface.
# v1.17.x doc/API-surface hardening adds one more idiomatic defensive raise.
# Keep the ratchet at the current audited count while taxonomy raises remain 76.
GENERIC_RAISE_MAX = 1902
# v1.19.x pandas-3.0 hardening: the agent error-envelope builder
# (agent/tools/_dispatch.py) widened its ``to_dict`` fallback from three
# specific exception types to ``except Exception`` so a user-overridden
# ``to_dict`` raising any error degrades gracefully instead of crashing the
# dispatcher (the fallback only reads ``e.code`` / ``str(e)``, which cannot
# fail). That is a justified best-effort error path, so the broad-handler
# ceiling moves 586 -> 587.
# v1.20.x paper Methods-and-Formulas appendix: paper() and
# paper_from_question() each gained a record_degradation-guarded best-effort
# ``except Exception`` around the optional ``result.to_appendix()`` section
# (§7: orchestration paths catch broadly and record the degradation rather
# than crash the draft), alongside one further best-effort orchestration
# catch in the same release. Ceiling moves 587 -> 590.
BROAD_EXCEPT_MAX = 590
TAXONOMY_RAISE_MIN = 42


def _name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _name(node.func)
    if isinstance(node, ast.Subscript):
        return _name(node.value)
    if isinstance(node, ast.Tuple):
        names = [_name(elt) for elt in node.elts]
        return ",".join(n for n in names if n)
    return None


class ExceptionVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.raises: list[dict[str, Any]] = []
        self.broad_handlers: list[dict[str, Any]] = []

    def visit_Raise(self, node: ast.Raise) -> None:  # noqa: N802
        exc_name = _name(node.exc)
        category = "other"
        if exc_name in TAXONOMY_EXCEPTIONS:
            category = "taxonomy"
        elif exc_name in GENERIC_EXCEPTIONS:
            category = "generic"
        self.raises.append(
            {
                "path": self.path.relative_to(REPO_ROOT).as_posix(),
                "line": node.lineno,
                "exception": exc_name or "<bare>",
                "category": category,
            }
        )
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # noqa: N802
        exc_name = _name(node.type)
        names = set((exc_name or "").split(","))
        if node.type is None or names.intersection({"Exception", "BaseException"}):
            self.broad_handlers.append(
                {
                    "path": self.path.relative_to(REPO_ROOT).as_posix(),
                    "line": node.lineno,
                    "exception": exc_name or "<bare>",
                }
            )
        self.generic_visit(node)


def _scan_file(path: Path) -> ExceptionVisitor:
    visitor = ExceptionVisitor(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return visitor
    visitor.visit(tree)
    return visitor


def collect(src_root: Path = SRC_ROOT) -> dict[str, Any]:
    raises: list[dict[str, Any]] = []
    broad_handlers: list[dict[str, Any]] = []
    for path in sorted(src_root.rglob("*.py")):
        visitor = _scan_file(path)
        raises.extend(visitor.raises)
        broad_handlers.extend(visitor.broad_handlers)

    by_exception = Counter(row["exception"] for row in raises)
    by_category = Counter(row["category"] for row in raises)

    return {
        "totals": {
            "raise_statements": len(raises),
            "taxonomy_raises": by_category.get("taxonomy", 0),
            "generic_raises": by_category.get("generic", 0),
            "other_raises": by_category.get("other", 0),
            "broad_exception_handlers": len(broad_handlers),
        },
        "by_exception": dict(sorted(by_exception.items())),
        "raises": raises,
        "broad_handlers": broad_handlers,
        "thresholds": {
            "generic_raise_max": GENERIC_RAISE_MAX,
            "broad_except_max": BROAD_EXCEPT_MAX,
            "taxonomy_raise_min": TAXONOMY_RAISE_MIN,
        },
    }


def check(report: dict[str, Any]) -> int:
    totals = report["totals"]
    failures: list[str] = []
    if totals["generic_raises"] > GENERIC_RAISE_MAX:
        failures.append(
            "generic_raises: "
            f"observed={totals['generic_raises']} max={GENERIC_RAISE_MAX}"
        )
    if totals["broad_exception_handlers"] > BROAD_EXCEPT_MAX:
        failures.append(
            "broad_exception_handlers: "
            f"observed={totals['broad_exception_handlers']} max={BROAD_EXCEPT_MAX}"
        )
    if totals["taxonomy_raises"] < TAXONOMY_RAISE_MIN:
        failures.append(
            "taxonomy_raises: "
            f"observed={totals['taxonomy_raises']} min={TAXONOMY_RAISE_MIN}"
        )
    if failures:
        print("[error_taxonomy_audit] REGRESSION", file=sys.stderr)
        for item in failures:
            print(f"  {item}", file=sys.stderr)
        return 1
    print(
        "[error_taxonomy_audit] OK - "
        f"{totals['taxonomy_raises']} taxonomy raises, "
        f"{totals['generic_raises']} generic raises."
    )
    return 0


def render(report: dict[str, Any]) -> str:
    totals = report["totals"]
    lines: list[str] = []
    lines.append("StatsPAI error taxonomy audit")
    lines.append("=" * 50)
    lines.append(f"Raise statements          : {totals['raise_statements']}")
    lines.append(f"  taxonomy exceptions     : {totals['taxonomy_raises']}")
    lines.append(f"  generic built-ins       : {totals['generic_raises']}")
    lines.append(f"  other/custom exceptions : {totals['other_raises']}")
    lines.append(f"Broad except handlers     : {totals['broad_exception_handlers']}")
    lines.append("")
    lines.append("Most common raised exceptions")
    lines.append("-" * 50)
    ranked = sorted(
        report["by_exception"].items(),
        key=lambda item: (-item[1], item[0]),
    )
    for name, count in ranked[:20]:
        lines.append(f"  {name:24s}: {count:4d}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument("--check", action="store_true", help="Run CI ratchet.")
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include every raise/except site in --json output.",
    )
    args = parser.parse_args(argv)

    report = collect()
    if args.check:
        return check(report)
    if args.json:
        payload = dict(report)
        if not args.details:
            payload.pop("raises", None)
            payload.pop("broad_handlers", None)
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
