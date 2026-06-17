"""Audit result-object protocol coverage across StatsPAI.

StatsPAI has many lightweight ``*Result`` containers.  Agents, notebooks,
and exporters work best when those objects expose a predictable surface:
``summary()`` for humans, ``tidy()`` / ``glance()`` for tabular workflows,
``to_dict()`` for serialization, and ``to_agent_summary()`` / ``brief()`` for
LLM-facing summaries.

This audit is intentionally static: it parses source files with ``ast`` and
does not import ``statspai``.  That keeps it safe to run in CI before optional
dependencies are installed and makes it useful when debugging import-time
regressions.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "statspai"

METHODS: tuple[str, ...] = (
    "summary",
    "tidy",
    "glance",
    "to_dict",
    "to_agent_summary",
    "brief",
    "plot",
)

PROTOCOLS: dict[str, tuple[str, ...]] = {
    "printable": ("summary",),
    "serializable": ("summary", "to_dict"),
    "tidy_model": ("summary", "tidy", "glance"),
    "agent_ready": ("summary", "to_agent_summary"),
}

# Ratchet floors.  These are deliberately lower-bound counters, not aspirational
# targets.  Raise them when result classes gain a protocol method; do not lower
# them to hide a regression.
FLOORS: dict[str, int] = {
    "result_classes": 262,
    "method_summary": 256,
    "method_tidy": 21,
    "method_glance": 17,
    "method_to_dict": 35,
    "method_to_agent_summary": 27,
    "method_brief": 11,
    "protocol_printable": 256,
    "protocol_serializable": 34,
    "protocol_tidy_model": 17,
    "protocol_agent_ready": 27,
}

CANONICAL_CLASSES = {
    "src/statspai/core/results.py:EconometricResults": (
        "summary",
        "tidy",
        "glance",
        "to_dict",
        "to_agent_summary",
        "brief",
    ),
    "src/statspai/core/results.py:CausalResult": (
        "summary",
        "tidy",
        "glance",
        "to_dict",
        "to_agent_summary",
        "brief",
        "plot",
    ),
}


@dataclass(frozen=True)
class ClassInfo:
    path: Path
    name: str
    line: int
    bases: tuple[str, ...]
    methods: frozenset[str]

    @property
    def key(self) -> str:
        rel = self.path.relative_to(REPO_ROOT).as_posix()
        return f"{rel}:{self.name}"

    @property
    def identity(self) -> str:
        rel = self.path.relative_to(REPO_ROOT).as_posix()
        return f"{rel}:{self.line}:{self.name}"


def _base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    if isinstance(node, ast.Call):
        return _base_name(node.func)
    return None


def _is_result_class(name: str) -> bool:
    return name.endswith("Result") or name.endswith("Results")


def _scan_file(path: Path) -> list[ClassInfo]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return []

    out: list[ClassInfo] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or not _is_result_class(node.name):
            continue
        methods = frozenset(
            child.name
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        bases = tuple(
            b for b in (_base_name(base) for base in node.bases) if b is not None
        )
        out.append(
            ClassInfo(
                path=path,
                name=node.name,
                line=node.lineno,
                bases=bases,
                methods=methods,
            )
        )
    return out


def collect_classes(src_root: Path = SRC_ROOT) -> list[ClassInfo]:
    classes: list[ClassInfo] = []
    for path in sorted(src_root.rglob("*.py")):
        classes.extend(_scan_file(path))
    return classes


def _effective_methods(
    info: ClassInfo,
    by_name: dict[str, list[ClassInfo]],
    seen: Set[str] | None = None,
) -> frozenset[str]:
    seen = set() if seen is None else seen
    if info.identity in seen:
        return info.methods
    seen.add(info.identity)

    methods = set(info.methods)
    for base in info.bases:
        for parent in by_name.get(base, []):
            methods.update(_effective_methods(parent, by_name, seen))
    return frozenset(methods)


def collect() -> dict[str, Any]:
    classes = collect_classes()
    by_name: dict[str, list[ClassInfo]] = {}
    for info in classes:
        by_name.setdefault(info.name, []).append(info)

    per_class: list[dict[str, Any]] = []
    method_counts: dict[str, int] = {m: 0 for m in METHODS}
    protocol_counts: dict[str, int] = {p: 0 for p in PROTOCOLS}

    for info in classes:
        effective = _effective_methods(info, by_name)
        for method in METHODS:
            if method in effective:
                method_counts[method] += 1
        for protocol, required in PROTOCOLS.items():
            if all(method in effective for method in required):
                protocol_counts[protocol] += 1
        per_class.append(
            {
                "key": info.key,
                "path": info.path.relative_to(REPO_ROOT).as_posix(),
                "name": info.name,
                "line": info.line,
                "bases": list(info.bases),
                "direct_methods": sorted(info.methods & set(METHODS)),
                "effective_methods": sorted(effective & set(METHODS)),
                "missing": {
                    protocol: [
                        method for method in required if method not in effective
                    ]
                    for protocol, required in PROTOCOLS.items()
                    if any(method not in effective for method in required)
                },
            }
        )

    return {
        "total": len(classes),
        "method_counts": method_counts,
        "protocol_counts": protocol_counts,
        "per_class": sorted(per_class, key=lambda row: (row["path"], row["line"])),
        "floors": FLOORS,
    }


def _floor_snapshot(report: dict[str, Any]) -> dict[str, int]:
    snap = {"result_classes": report["total"]}
    for method, count in report["method_counts"].items():
        snap[f"method_{method}"] = count
    for protocol, count in report["protocol_counts"].items():
        snap[f"protocol_{protocol}"] = count
    return snap


def _canonical_failures(report: dict[str, Any]) -> list[str]:
    by_key = {row["key"]: row for row in report["per_class"]}
    failures: list[str] = []
    for key, required in CANONICAL_CLASSES.items():
        row = by_key.get(key)
        if row is None:
            failures.append(f"{key}: class not found")
            continue
        effective = set(row["effective_methods"])
        missing = [method for method in required if method not in effective]
        if missing:
            failures.append(f"{key}: missing {', '.join(missing)}")
    return failures


def check(report: dict[str, Any]) -> int:
    current = _floor_snapshot(report)
    failures: list[str] = []
    for key, floor in FLOORS.items():
        observed = current.get(key, 0)
        if observed < floor:
            failures.append(f"{key}: observed={observed} floor={floor}")
    failures.extend(_canonical_failures(report))
    if failures:
        print("[result_protocol_audit] REGRESSION", file=sys.stderr)
        for item in failures:
            print(f"  {item}", file=sys.stderr)
        return 1
    print(
        "[result_protocol_audit] OK - "
        f"{report['total']} result classes inspected."
    )
    return 0


def _pct(num: int, denom: int) -> str:
    return f"{100 * num / denom:5.1f}%" if denom else "  --"


def render(report: dict[str, Any], *, missing: str | None = None) -> str:
    total = report["total"]
    lines: list[str] = []
    lines.append("StatsPAI result protocol audit")
    lines.append("=" * 50)
    lines.append(f"Result classes inspected : {total}")
    lines.append("")
    lines.append("Method coverage")
    lines.append("-" * 50)
    for method in METHODS:
        count = report["method_counts"].get(method, 0)
        lines.append(f"  {method:18s}: {count:4d}  ({_pct(count, total)})")
    lines.append("")
    lines.append("Protocol coverage")
    lines.append("-" * 50)
    for protocol, required in PROTOCOLS.items():
        count = report["protocol_counts"].get(protocol, 0)
        req = ", ".join(required)
        lines.append(f"  {protocol:18s}: {count:4d}  ({_pct(count, total)})  [{req}]")
    if missing:
        rows = [
            row for row in report["per_class"]
            if missing in row.get("missing", {})
        ]
        lines.append("")
        lines.append(f"Classes missing protocol: {missing}")
        lines.append("-" * 50)
        for row in rows:
            gaps = ", ".join(row["missing"][missing])
            lines.append(f"  {row['path']}:{row['line']} {row['name']} missing={gaps}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument("--check", action="store_true", help="Run CI ratchet.")
    parser.add_argument(
        "--missing",
        choices=tuple(PROTOCOLS),
        help="List classes missing a named protocol in the human report.",
    )
    args = parser.parse_args(argv)

    report = collect()
    if args.check:
        return check(report)
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    print(render(report, missing=args.missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
