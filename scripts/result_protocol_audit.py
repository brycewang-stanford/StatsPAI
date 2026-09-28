"""Audit result-object protocol coverage across StatsPAI.

StatsPAI has many lightweight ``*Result`` containers.  Agents, notebooks,
and exporters work best when those objects expose a predictable surface:
``summary()`` for humans, ``tidy()`` / ``glance()`` for tabular workflows,
``to_dict()`` for serialization, and ``to_agent_summary()`` / ``brief()`` for
LLM-facing summaries.

The documented *agent contract* (AGENTS.md, ``schemas/result.schema.json``)
is tracked separately as the ``agent_contract`` protocol:
``to_dict(detail=...)`` (pseudo-method ``to_dict_detail``), ``violations()``,
``next_steps()``, ``result_card()`` and ``cite()``. ``to_dict`` counts as
detail-aware when the nearest definition takes ``detail`` or lives on a
``ResultProtocolMixin`` subclass / ``@attach_result_protocol`` class (the
mixin wraps a legacy ``to_dict`` so it accepts ``detail``).

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
    "to_dict_detail",
    "violations",
    "next_steps",
    "result_card",
    "cite",
)

#: The mixin that supplies (and, for legacy ``to_dict`` overrides, wraps)
#: the agent contract, and the decorator that grafts it onto NamedTuples.
MIXIN_NAME = "ResultProtocolMixin"
ATTACH_DECORATOR = "attach_result_protocol"

PROTOCOLS: dict[str, tuple[str, ...]] = {
    "printable": ("summary",),
    "serializable": ("summary", "to_dict"),
    "tidy_model": ("summary", "tidy", "glance"),
    "agent_ready": ("summary", "to_agent_summary"),
    "agent_contract": (
        "to_dict_detail",
        "violations",
        "next_steps",
        "result_card",
        "cite",
    ),
}

# Ratchet floors.  These are deliberately lower-bound counters, not aspirational
# targets.  Raise them when result classes gain a protocol method; do not lower
# them to hide a regression.
FLOORS: dict[str, int] = {
    "result_classes": 262,
    "method_brief": 14,
    "method_cite": 294,
    "method_glance": 20,
    "method_next_steps": 294,
    "method_plot": 69,
    "method_result_card": 294,
    "method_summary": 285,
    "method_tidy": 26,
    "method_to_agent_summary": 30,
    "method_to_dict": 298,
    "method_to_dict_detail": 294,
    "method_violations": 294,
    "protocol_agent_contract": 294,
    "protocol_agent_ready": 30,
    "protocol_printable": 285,
    "protocol_serializable": 285,
    "protocol_tidy_model": 20,
}

#: Result classes known to miss the agent contract, with the reason. The
#: ratchet fails on any *other* class missing it, and on an entry here that
#: no longer misses it (delete the entry when the class is fixed).
AGENT_CONTRACT_GAPS: dict[str, str] = {
    "src/statspai/fast/_jax_fallback.py:FeolsBootstrapResult": (
        "JAX-absent fallback stub of fast/jax_feols.py:FeolsBootstrapResult "
        "(which has the contract)"
    ),
    "src/statspai/help.py:HelpResult": (
        "documentation payload of sp.help, not a fitted result; .data is "
        "already the structured dict"
    ),
}

CANONICAL_CLASSES = {
    "src/statspai/core/results.py:EconometricResults": (
        "summary",
        "tidy",
        "glance",
        "to_dict",
        "to_agent_summary",
        "brief",
        "to_dict_detail",
        "violations",
        "next_steps",
        "result_card",
        "cite",
    ),
    "src/statspai/core/results.py:CausalResult": (
        "summary",
        "tidy",
        "glance",
        "to_dict",
        "to_agent_summary",
        "brief",
        "plot",
        "to_dict_detail",
        "violations",
        "next_steps",
        "result_card",
        "cite",
    ),
}


@dataclass(frozen=True)
class ClassInfo:
    path: Path
    name: str
    line: int
    bases: tuple[str, ...]
    methods: frozenset[str]
    #: ``True`` / ``False`` when the class itself defines ``to_dict`` (does
    #: it take ``detail``?); ``None`` when it inherits it.
    to_dict_detail: bool | None = None

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


def _takes_detail(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    args = fn.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    return "detail" in names


def _class_functions(
    body: list[ast.stmt],
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Methods of a class body, including those under ``if`` blocks."""
    out: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for child in body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(child)
        elif isinstance(child, ast.If):
            out.extend(_class_functions(child.body))
            out.extend(_class_functions(child.orelse))
    return out


def _scan_file(path: Path, *, all_classes: bool = False) -> list[ClassInfo]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return []

    out: list[ClassInfo] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not all_classes and not _is_result_class(node.name):
            continue
        funcs = _class_functions(node.body)
        methods = frozenset(f.name for f in funcs)
        to_dict_defs = [f for f in funcs if f.name == "to_dict"]
        # ``if TYPE_CHECKING: <stub> else: <real>`` defines to_dict twice;
        # the runtime one is detail-aware when any branch takes ``detail``.
        to_dict_detail = (
            any(_takes_detail(f) for f in to_dict_defs) if to_dict_defs else None
        )
        bases = tuple(
            b for b in (_base_name(base) for base in node.bases) if b is not None
        )
        decorators = {_base_name(d) for d in node.decorator_list}
        if ATTACH_DECORATOR in decorators:
            # The decorator grafts the mixin's members onto the class.
            bases = bases + (MIXIN_NAME,)
        out.append(
            ClassInfo(
                path=path,
                name=node.name,
                line=node.lineno,
                bases=bases,
                methods=methods,
                to_dict_detail=to_dict_detail,
            )
        )
    return out


def collect_classes(
    src_root: Path = SRC_ROOT, *, all_classes: bool = False
) -> list[ClassInfo]:
    classes: list[ClassInfo] = []
    for path in sorted(src_root.rglob("*.py")):
        classes.extend(_scan_file(path, all_classes=all_classes))
    return classes


def _ancestors(
    info: ClassInfo,
    by_name: dict[str, list[ClassInfo]],
    seen: Set[str] | None = None,
) -> set[str]:
    seen = set() if seen is None else seen
    if info.identity in seen:
        return set()
    seen.add(info.identity)
    names: set[str] = set(info.bases)
    for base in info.bases:
        for parent in by_name.get(base, []):
            names |= _ancestors(parent, by_name, seen)
    return names


def _detail_aware(
    info: ClassInfo,
    by_name: dict[str, list[ClassInfo]],
    seen: Set[str] | None = None,
) -> bool:
    """Does the ``to_dict`` this class resolves to accept ``detail``?"""
    seen = set() if seen is None else seen
    if info.identity in seen:
        return False
    seen.add(info.identity)
    if info.to_dict_detail is not None:
        if info.to_dict_detail:
            return True
        # A legacy override on a mixin subclass is wrapped at class
        # creation (ResultProtocolMixin.__init_subclass__).
        return info.name == MIXIN_NAME or MIXIN_NAME in _ancestors(info, by_name)
    for base in info.bases:  # left-to-right, like the MRO for these trees
        for parent in by_name.get(base, []):
            if "to_dict" in _effective_methods(parent, by_name):
                return _detail_aware(parent, by_name, seen)
    return False


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
    # Resolve inheritance against *every* class (mixins such as
    # ResultProtocolMixin / DecompResultMixin are not named ``*Result``),
    # but report only result classes.
    every = collect_classes(all_classes=True)
    classes = [info for info in every if _is_result_class(info.name)]
    by_name: dict[str, list[ClassInfo]] = {}
    for info in every:
        by_name.setdefault(info.name, []).append(info)

    per_class: list[dict[str, Any]] = []
    method_counts: dict[str, int] = {m: 0 for m in METHODS}
    protocol_counts: dict[str, int] = {p: 0 for p in PROTOCOLS}

    for info in classes:
        effective = set(_effective_methods(info, by_name))
        effective.discard("to_dict_detail")
        if "to_dict" in effective and _detail_aware(info, by_name):
            effective.add("to_dict_detail")
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
                    protocol: [method for method in required if method not in effective]
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
        "agent_contract_gaps": AGENT_CONTRACT_GAPS,
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


def _agent_contract_failures(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    missing = {
        row["key"]
        for row in report["per_class"]
        if "agent_contract" in row.get("missing", {})
    }
    for key in sorted(missing - set(AGENT_CONTRACT_GAPS)):
        failures.append(
            f"{key}: missing the agent contract (to_dict(detail=), "
            "violations, next_steps, result_card, cite) -- inherit "
            "ResultProtocolMixin or use @attach_result_protocol"
        )
    for key in sorted(set(AGENT_CONTRACT_GAPS) - missing):
        failures.append(
            f"{key}: listed in AGENT_CONTRACT_GAPS but now has the contract "
            "(or no longer exists) -- delete the entry"
        )
    return failures


def check(report: dict[str, Any]) -> int:
    current = _floor_snapshot(report)
    failures: list[str] = []
    for key, floor in FLOORS.items():
        observed = current.get(key, 0)
        if observed < floor:
            failures.append(f"{key}: observed={observed} floor={floor}")
    failures.extend(_canonical_failures(report))
    failures.extend(_agent_contract_failures(report))
    if failures:
        print("[result_protocol_audit] REGRESSION", file=sys.stderr)
        for item in failures:
            print(f"  {item}", file=sys.stderr)
        return 1
    print(
        "[result_protocol_audit] OK - " f"{report['total']} result classes inspected."
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
        rows = [row for row in report["per_class"] if missing in row.get("missing", {})]
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
