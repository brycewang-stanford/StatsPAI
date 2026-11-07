"""Measure agent-native / help-system documentation coverage for StatsPAI.

Companion to ``scripts/registry_stats.py``. Where ``registry_stats`` answers
*"how many functions are there"*, this script answers *"how well are they
documented for humans and agents"* — the share of registered public functions
that carry a docstring, a NumPy-style ``Parameters`` / ``Returns`` / ``Examples``
/ ``References`` section, and the agent-native metadata (assumptions,
pre-conditions, failure modes, alternatives, typical sample size) that
``sp.agent_card`` / ``sp.function_schema`` surface to LLM callers.

It is the **single source of truth** for the coverage floors enforced by
``tests/test_help_completeness.py``: that contract test imports
:func:`compute_coverage` and asserts the live numbers never drop below the
frozen baseline, so documentation quality can only ratchet up.

Usage
-----
    python scripts/help_coverage.py            # human-readable summary
    python scripts/help_coverage.py --json      # machine-readable
    python scripts/help_coverage.py --by-category   # per-category breakdown
    python scripts/help_coverage.py --check     # exit non-zero if below floors
                                                # or above a ceiling
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List

# NumPy-style section detectors. We accept either an explicit underlined
# section header (``Examples\n--------``) or, for examples, doctest prompts.
_SECTION = {
    "parameters": re.compile(r"\n\s*Parameters\s*\n\s*-{3,}"),
    "returns": re.compile(r"\n\s*Returns\s*\n\s*-{3,}"),
    "references": re.compile(r"\n\s*References?\s*\n\s*-{3,}"),
}
_EXAMPLES = re.compile(r"\n\s*Examples?\s*\n\s*-{3,}")

# Agent-native fields surfaced by FunctionSpec.agent_card().
_AGENT_FIELDS = (
    "assumptions",
    "pre_conditions",
    "failure_modes",
    "alternatives",
    "typical_n_min",
)

# Frozen baseline floors (percentages). These are ratchets: raise them as
# coverage improves; never lower them without a recorded reason. Measured on
# 2026-05-29 after the spatial-docstring backfill.
FLOORS: Dict[str, float] = {
    "docstring": 100.0,  # hard: every public function must have a docstring
    "parameters": 65.0,
    "returns": 56.0,
    "examples": 34.0,
    "references": 7.0,  # raised after verifying flagship + spatial citations
    "agent_native_any": 26.0,
}

# Frozen ceilings (counts). The opposite ratchet: these may only go down.
# ``validated_without_reference`` counts registered functions whose
# ``validation_status`` is ``certified`` or ``validated`` but whose registry
# ``reference`` is empty, so a reviewer cannot see the method source of a
# function the package calls validated (rOpenSci G1.0). 404 on 2026-11-05;
# 173 after ``statspai/_reference_seeds.py`` mapped existing paper.bib keys
# (2026-11-06); 130 after 42 verified entries were added to paper.bib for
# the seeds (2026-11-07). Lower it when you add references; never raise it
# without a recorded reason.
CEILINGS: Dict[str, int] = {
    "validated_without_reference": 130,
}


def _import_statspai() -> Any:
    import statspai as sp  # local import keeps --help fast and import errors scoped

    return sp


def compute_coverage() -> Dict[str, Any]:
    """Compute help / agent-native coverage over all registered functions.

    Returns
    -------
    dict
        ``{"total", "docstring", "parameters", "returns", "examples",
        "references", "agent_native_any", <agent fields...>, "missing_docstring",
        "validated_without_reference", "by_category"}``. The metric keys
        map to ``{"count", "pct"}`` dicts.
    """
    sp = _import_statspai()
    names = sp.list_functions()

    metrics = ["docstring", "parameters", "returns", "examples", "references"]
    counts = {m: 0 for m in metrics}
    agent_counts = {f: 0 for f in _AGENT_FIELDS}
    agent_any = 0
    missing_doc: List[str] = []
    total = 0

    by_cat_total: Dict[str, int] = defaultdict(int)
    by_cat_examples: Dict[str, int] = defaultdict(int)
    by_cat_refs: Dict[str, int] = defaultdict(int)
    by_cat_agent: Dict[str, int] = defaultdict(int)

    for name in names:
        obj = getattr(sp, name, None)
        if obj is None or not callable(obj):
            continue
        total += 1

        # Category from the registry spec (falls back to module-derived).
        try:
            spec = sp.describe_function(name)
        except Exception:
            spec = {}
        category = spec.get("category", "uncategorized")
        by_cat_total[category] += 1

        doc = inspect.getdoc(obj) or ""
        if doc.strip():
            counts["docstring"] += 1
        else:
            missing_doc.append(name)

        if _SECTION["parameters"].search(doc):
            counts["parameters"] += 1
        if _SECTION["returns"].search(doc):
            counts["returns"] += 1
        if _SECTION["references"].search(doc):
            counts["references"] += 1
            by_cat_refs[category] += 1
        if _EXAMPLES.search(doc) or ">>>" in doc:
            counts["examples"] += 1
            by_cat_examples[category] += 1

        has_native = False
        for f in _AGENT_FIELDS:
            if spec.get(f):
                agent_counts[f] += 1
                has_native = True
        if has_native:
            agent_any += 1
            by_cat_agent[category] += 1

    def pct(n: int) -> float:
        return round(100.0 * n / total, 1) if total else 0.0

    out: Dict[str, Any] = {"total": total}
    for m in metrics:
        out[m] = {"count": counts[m], "pct": pct(counts[m])}
    for f in _AGENT_FIELDS:
        out[f] = {"count": agent_counts[f], "pct": pct(agent_counts[f])}
    out["agent_native_any"] = {"count": agent_any, "pct": pct(agent_any)}
    out["missing_docstring"] = sorted(missing_doc)
    out["validated_without_reference"] = _validated_without_reference()

    by_category = {}
    for cat, n in sorted(by_cat_total.items(), key=lambda kv: -kv[1]):
        by_category[cat] = {
            "total": n,
            "examples_pct": round(100.0 * by_cat_examples[cat] / n, 1),
            "references_pct": round(100.0 * by_cat_refs[cat] / n, 1),
            "agent_native_pct": round(100.0 * by_cat_agent[cat] / n, 1),
        }
    out["by_category"] = by_category
    return out


def _validated_without_reference() -> Dict[str, Any]:
    """Certified / validated registered functions with an empty ``reference``."""
    from statspai import registry as _reg

    _reg._ensure_full_registry()
    names = sorted(
        name
        for name, spec in _reg._REGISTRY.items()
        if spec.validation_status in ("certified", "validated")
        and not (spec.reference or "").strip()
    )
    return {"count": len(names), "names": names}


def _print_human(cov: Dict[str, Any]) -> None:
    total = cov["total"]
    print(f"StatsPAI help / agent-native coverage  ({total} public functions)")
    print("=" * 64)
    rows = [
        ("docstring", "Docstring present"),
        ("parameters", "NumPy `Parameters` section"),
        ("returns", "NumPy `Returns` section"),
        ("examples", "`Examples` / doctest"),
        ("references", "`References` section"),
        ("agent_native_any", ">=1 agent-native field"),
    ]
    for key, label in rows:
        c = cov[key]
        floor = FLOORS.get(key)
        flag = ""
        if floor is not None:
            flag = "  OK" if c["pct"] >= floor else f"  BELOW FLOOR {floor}%"
        print(f"  {label:<32} {c['count']:>5} / {total}  ({c['pct']:>5.1f}%){flag}")
    vwr = cov["validated_without_reference"]["count"]
    ceiling = CEILINGS["validated_without_reference"]
    flag = "  OK" if vwr <= ceiling else f"  ABOVE CEILING {ceiling}"
    print(f"  {'certified/validated, no reference':<32} {vwr:>5}{flag}")
    print("-" * 64)
    print("  Agent-native fields:")
    for f in _AGENT_FIELDS:
        c = cov[f]
        print(f"    {f:<20} {c['count']:>5}  ({c['pct']:>5.1f}%)")
    if cov["missing_docstring"]:
        print("-" * 64)
        print(f"  MISSING docstrings ({len(cov['missing_docstring'])}):")
        for n in cov["missing_docstring"]:
            print(f"    - {n}")


def _print_by_category(cov: Dict[str, Any]) -> None:
    print(f"{'category':<28}{'n':>5}{'examples':>11}{'refs':>9}{'agent':>9}")
    print("-" * 62)
    for cat, d in cov["by_category"].items():
        print(
            f"{cat:<28}{d['total']:>5}{d['examples_pct']:>10.1f}%"
            f"{d['references_pct']:>8.1f}%{d['agent_native_pct']:>8.1f}%"
        )


def check_floors(cov: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable floor violations (empty == all OK)."""
    violations: List[str] = []
    for key, floor in FLOORS.items():
        pct = cov[key]["pct"]
        if pct < floor:
            violations.append(
                f"{key}: {pct:.1f}% is below floor {floor:.1f}% "
                f"({cov[key]['count']}/{cov['total']})"
            )
    for key, ceiling in CEILINGS.items():
        count = cov[key]["count"]
        if count > ceiling:
            violations.append(
                f"{key}: {count} is above ceiling {ceiling} "
                "(give each new one a paper.bib key, see "
                "src/statspai/_reference_seeds.py)"
            )
    if cov["missing_docstring"]:
        violations.append("missing docstrings: " + ", ".join(cov["missing_docstring"]))
    return violations


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument(
        "--by-category", action="store_true", help="per-category coverage breakdown"
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any metric is below its frozen floor",
    )
    args = ap.parse_args(argv)

    cov = compute_coverage()

    if args.check:
        violations = check_floors(cov)
        if violations:
            print("help-coverage CHECK FAILED:", file=sys.stderr)
            for v in violations:
                print(f"  - {v}", file=sys.stderr)
            return 1
        print("help-coverage CHECK PASSED — all metrics at or above floor.")
        return 0

    if args.json:
        print(json.dumps(cov, indent=2))
    elif args.by_category:
        _print_by_category(cov)
    else:
        _print_human(cov)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
