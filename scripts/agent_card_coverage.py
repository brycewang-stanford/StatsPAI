"""Audit curated agent-card field coverage on every registered function.

Complementary to :mod:`scripts.schema_quality`, which measures the
agent-facing JSON schema *after* the synthetic-fallback layer in
:func:`statspai.registry._param_description` fills empty fields with
heuristic text.  That view (used in JSS Table 9) tells you what an LLM
agent sees; this script tells you **what is curated or explicitly
inherited** — i.e. where metadata work is still missing without
counting generic fallback text.

Three tiers (matching ``docs/agent_cards_spec.md``):

* **Tier-B (baseline)**   description >30 chars · tags · example · reference · param.description
* **Tier-A (agent-native)** Tier-B + assumptions + pre_conditions + failure_modes + alternatives + typical_n_min
* **Tier-S (certified)**    Tier-A + validation_status in {certified, validated}

Usage
-----
::

    python scripts/agent_card_coverage.py                # summary table
    python scripts/agent_card_coverage.py --by-category  # per-category drill-down
    python scripts/agent_card_coverage.py --by-function  # one row per function (long)
    python scripts/agent_card_coverage.py --json         # machine-readable
    python scripts/agent_card_coverage.py --check        # CI ratchet — exit 1 on regression

The ``--check`` mode reads ``scripts/agent_card_coverage_floor.json``
and fails if any tracked counter has dropped.  Bump the floor only when
you have intentionally raised the bar; never lower it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
FLOOR_PATH = REPO_ROOT / "scripts" / "agent_card_coverage_floor.json"


# Tier-B fields — what a "baseline" card needs.
TIER_B_FIELDS: Tuple[str, ...] = (
    "description_30",
    "tags",
    "example",
    "reference",
    "any_param_description",
)

# Tier-A fields — what an "agent-native" card needs on top of Tier-B.
TIER_A_FIELDS: Tuple[str, ...] = (
    "assumptions",
    "pre_conditions",
    "failure_modes",
    "alternatives",
    "typical_n_min",
)


@dataclass
class FieldStatus:
    """Whether one field on one FunctionSpec is curated (not auto-filled)."""

    description_30: bool
    tags: bool
    example: bool
    reference: bool
    any_param_description: bool
    assumptions: bool
    pre_conditions: bool
    failure_modes: bool
    alternatives: bool
    typical_n_min: bool
    validation_status: str

    def tier_b_complete(self) -> bool:
        return all(getattr(self, f) for f in TIER_B_FIELDS)

    def tier_a_complete(self) -> bool:
        return self.tier_b_complete() and all(getattr(self, f) for f in TIER_A_FIELDS)

    def tier_s_complete(self) -> bool:
        return self.tier_a_complete() and self.validation_status in {
            "certified",
            "validated",
        }


def _status_for_spec(spec: Any) -> FieldStatus:
    """Inspect a FunctionSpec and report which fields are populated.

    Tier-A fields are measured against the *effective* (merged) view —
    a child with ``inherits_from`` that resolves to a Tier-A parent
    counts as Tier-A.  This matches what an agent actually sees from
    :meth:`FunctionSpec.agent_card`.  Tier-B fields stay raw because
    ``description`` / ``example`` / ``reference`` / params never
    inherit (those are method-specific).
    """
    desc = spec.description or ""
    # Single agent_card() call gives us the merged Tier-A view; cheaper
    # than calling _merge_inherited_view directly here.
    card = spec.agent_card(merge_inherited=True)
    return FieldStatus(
        description_30=isinstance(desc, str) and len(desc.strip()) > 30,
        tags=bool(spec.tags),
        example=bool(spec.example),
        reference=bool(spec.reference),
        any_param_description=any(
            isinstance(p.description, str) and p.description.strip()
            for p in spec.params
        ),
        assumptions=bool(card["assumptions"]),
        pre_conditions=bool(card["pre_conditions"]),
        failure_modes=bool(card["failure_modes"]),
        alternatives=bool(card["alternatives"]),
        typical_n_min=card["typical_n_min"] is not None,
        validation_status=spec.validation_status,
    )


def collect() -> Dict[str, Any]:
    """Walk the registry and return a structured coverage report."""
    # Local import to keep ``--help`` snappy and avoid importing statspai
    # if the user only wants the CLI surface.
    import statspai  # noqa: F401
    from statspai.registry import _REGISTRY, _ensure_full_registry

    _ensure_full_registry()

    per_function: Dict[str, Dict[str, Any]] = {}
    per_category: Dict[str, Dict[str, int]] = defaultdict(
        lambda: dict(total=0, tier_b=0, tier_a=0, tier_s=0)
    )
    field_counts: Dict[str, int] = defaultdict(int)
    validation_counts: Dict[str, int] = defaultdict(int)

    for name, spec in _REGISTRY.items():
        status = _status_for_spec(spec)
        per_function[name] = {
            "category": spec.category,
            **{f: getattr(status, f) for f in TIER_B_FIELDS + TIER_A_FIELDS},
            "validation_status": status.validation_status,
            "tier_b": status.tier_b_complete(),
            "tier_a": status.tier_a_complete(),
            "tier_s": status.tier_s_complete(),
        }
        per_category[spec.category]["total"] += 1
        if status.tier_b_complete():
            per_category[spec.category]["tier_b"] += 1
        if status.tier_a_complete():
            per_category[spec.category]["tier_a"] += 1
        if status.tier_s_complete():
            per_category[spec.category]["tier_s"] += 1

        for f in TIER_B_FIELDS + TIER_A_FIELDS:
            if getattr(status, f):
                field_counts[f] += 1
        validation_counts[status.validation_status] += 1

    total = len(per_function)
    return {
        "total": total,
        "field_counts": dict(field_counts),
        "validation_counts": dict(validation_counts),
        "per_category": {k: dict(v) for k, v in per_category.items()},
        "per_function": per_function,
        "tier_totals": {
            "tier_b": sum(1 for v in per_function.values() if v["tier_b"]),
            "tier_a": sum(1 for v in per_function.values() if v["tier_a"]),
            "tier_s": sum(1 for v in per_function.values() if v["tier_s"]),
        },
    }


def _pct(num: int, denom: int) -> str:
    return f"{100 * num / denom:5.1f}%" if denom else "  --"


def render_summary(report: Dict[str, Any]) -> str:
    total = report["total"]
    field_counts = report["field_counts"]
    validation_counts = report["validation_counts"]
    tier = report["tier_totals"]

    lines: List[str] = []
    lines.append(f"Registered functions       : {total}")
    lines.append("")
    lines.append("Tier completion (cumulative):")
    lines.append(
        f"  Tier-B (baseline)         : {tier['tier_b']:4d} / {total}  "
        f"({_pct(tier['tier_b'], total)})"
    )
    lines.append(
        f"  Tier-A (agent-native)     : {tier['tier_a']:4d} / {total}  "
        f"({_pct(tier['tier_a'], total)})"
    )
    lines.append(
        f"  Tier-S (certified)        : {tier['tier_s']:4d} / {total}  "
        f"({_pct(tier['tier_s'], total)})"
    )
    lines.append("")
    lines.append("Per-field coverage:")
    for f in TIER_B_FIELDS:
        c = field_counts.get(f, 0)
        lines.append(f"  [B] {f:25s}: {c:4d}  ({_pct(c, total)})")
    for f in TIER_A_FIELDS:
        c = field_counts.get(f, 0)
        lines.append(f"  [A] {f:25s}: {c:4d}  ({_pct(c, total)})")
    lines.append("")
    lines.append("Validation tier:")
    for status in (
        "certified",
        "validated",
        "api_stable",
        "experimental",
        "deprecated",
    ):
        c = validation_counts.get(status, 0)
        if c:
            lines.append(f"  {status:15s}: {c:4d}  ({_pct(c, total)})")
    return "\n".join(lines)


def render_by_category(report: Dict[str, Any]) -> str:
    per_cat = report["per_category"]
    lines: List[str] = []
    lines.append(
        f"{'category':<22} {'total':>6} {'tier-B':>8} {'tier-A':>8} {'tier-S':>8}"
    )
    lines.append("-" * 56)
    for cat, c in sorted(per_cat.items(), key=lambda kv: -kv[1]["total"]):
        t = c["total"]
        lines.append(
            f"{cat:<22} {t:>6} "
            f"{c['tier_b']:>4} ({_pct(c['tier_b'], t):>6}) "
            f"{c['tier_a']:>4} ({_pct(c['tier_a'], t):>6}) "
            f"{c['tier_s']:>4} ({_pct(c['tier_s'], t):>6})"
        )
    return "\n".join(lines)


def render_by_function(report: Dict[str, Any], tier: str = "tier_a") -> str:
    """List every function that has *not* reached the given tier."""
    missing = [
        (name, info)
        for name, info in report["per_function"].items()
        if not info.get(tier, False)
    ]
    missing.sort(key=lambda kv: (kv[1]["category"], kv[0]))
    lines = [f"# Functions below {tier} ({len(missing)} / {report['total']})"]
    for name, info in missing:
        gaps = [f for f in TIER_B_FIELDS + TIER_A_FIELDS if not info.get(f)]
        lines.append(f"{info['category']:<20} {name:<40} missing={','.join(gaps)}")
    return "\n".join(lines)


def _load_floor() -> Dict[str, int]:
    if not FLOOR_PATH.exists():
        return {}
    return json.loads(FLOOR_PATH.read_text(encoding="utf-8"))


def _current_floor_snapshot(report: Dict[str, Any]) -> Dict[str, int]:
    """The counters we ratchet — only these may not regress."""
    validation_counts = report["validation_counts"]
    snap = {
        "tier_b": report["tier_totals"]["tier_b"],
        "tier_a": report["tier_totals"]["tier_a"],
        "tier_s": report["tier_totals"]["tier_s"],
    }
    for f in TIER_B_FIELDS + TIER_A_FIELDS:
        snap[f"field_{f}"] = report["field_counts"].get(f, 0)
    snap["validation_certified"] = validation_counts.get("certified", 0)
    # `certified` is a stricter validation tier than `validated`. Treat
    # validation_validated as a cumulative "validated or better" floor so
    # promoting evidence from validated -> certified cannot look like a
    # regression in the lower tier.
    snap["validation_validated"] = validation_counts.get(
        "certified", 0
    ) + validation_counts.get("validated", 0)
    return snap


def check_against_floor(report: Dict[str, Any]) -> int:
    """Return 0 if every tracked counter is >= floor, else 1."""
    floor = _load_floor()
    if not floor:
        print(
            f"[agent_card_coverage] No floor file at {FLOOR_PATH}.\n"
            f"Run with --write-floor to seed it.",
            file=sys.stderr,
        )
        return 1
    current = _current_floor_snapshot(report)
    regressions: List[Tuple[str, int, int]] = []
    for key, baseline in floor.items():
        cur = current.get(key, 0)
        if cur < baseline:
            regressions.append((key, baseline, cur))
    if regressions:
        print(
            "[agent_card_coverage] REGRESSION — counters dropped below floor:",
            file=sys.stderr,
        )
        for key, baseline, cur in regressions:
            print(f"  {key:30s} floor={baseline:4d}  current={cur:4d}", file=sys.stderr)
        print(
            "\nDo not lower the floor to make CI pass.  Fix the missing cards.",
            file=sys.stderr,
        )
        return 1
    print(
        f"[agent_card_coverage] OK — all {len(floor)} tracked counters meet floor.",
        file=sys.stderr,
    )
    return 0


def write_floor(report: Dict[str, Any]) -> None:
    snap = _current_floor_snapshot(report)
    FLOOR_PATH.write_text(
        json.dumps(snap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[agent_card_coverage] Wrote floor to {FLOOR_PATH}", file=sys.stderr)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--summary", action="store_true", help="Print summary table (default)."
    )
    g.add_argument(
        "--by-category", action="store_true", help="Per-category drill-down."
    )
    g.add_argument(
        "--by-function",
        action="store_true",
        help="List every function below --tier (default tier-A).",
    )
    g.add_argument("--json", action="store_true", help="Emit JSON to stdout.")
    g.add_argument(
        "--check", action="store_true", help="CI ratchet — fail if counters regress."
    )
    g.add_argument(
        "--write-floor",
        action="store_true",
        help="Snapshot the current counters as the new CI floor.",
    )
    parser.add_argument(
        "--tier",
        choices=("tier_b", "tier_a", "tier_s"),
        default="tier_a",
        help="Which tier to use for --by-function gap listing.",
    )
    args = parser.parse_args(argv)

    report = collect()

    if args.json:
        json.dump(
            {
                "total": report["total"],
                "field_counts": report["field_counts"],
                "validation_counts": report["validation_counts"],
                "per_category": report["per_category"],
                "tier_totals": report["tier_totals"],
            },
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return 0
    if args.by_category:
        print(render_by_category(report))
        return 0
    if args.by_function:
        print(render_by_function(report, tier=args.tier))
        return 0
    if args.check:
        return check_against_floor(report)
    if args.write_floor:
        write_floor(report)
        return 0
    # Default
    print(render_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
