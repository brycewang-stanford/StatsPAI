"""§3 result-object contract audit.

StatsPAI's design principle §3 promises every result class exposes a uniform
six-method surface: ``summary()`` / ``plot()`` / ``to_latex()`` / ``to_word()`` /
``to_excel()`` / ``cite()``.  The other audit (``result_protocol_audit.py``)
tracks a different protocol (tidy_model / agent_ready / serializable) and
must not be conflated with §3.

This audit is a ratchet (additive) over live introspection: it counts
``*Result`` classes that satisfy all six §3 methods (via the
``ResultProtocolMixin`` aliases or per-class implementations).  The
``--check`` gate prevents the §3 surface from regressing.

Usage
-----
::

    python scripts/section3_contract_audit.py             # human report
    python scripts/section3_contract_audit.py --json      # machine-readable
    python scripts/section3_contract_audit.py --check     # CI ratchet
"""

from __future__ import annotations

import argparse
import inspect
import json
import pkgutil
import re
import sys
from pathlib import Path
from typing import Any, List, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

#: The §3 six-method surface.  Kept in this order so the column count is stable.
SECTION3_METHODS: Tuple[str, ...] = (
    "summary",
    "plot",
    "to_latex",
    "to_word",
    "to_excel",
    "cite",
)

#: Ratchet floor — the count of result classes satisfying all six methods.
#: Frozen at the current observed value (37) so the gate blocks *regressions*
#: only; raise it as new methods land on previously-bare classes.
FLOOR_FULL_SECTION3: int = 37

#: Class names we deliberately skip — internal dataclasses that are not part of
#: the public result surface (no, e.g., ``summary`` or ``to_latex``).
_SKIP_RE = re.compile(
    r"_?(?:Result|Results)$",  # only Result* classes are considered anyway
)


def _imported_classes() -> List[Tuple[str, str, type]]:
    """Walk ``statspai`` and collect candidate result classes."""
    import importlib

    import statspai

    found: List[Tuple[str, str, type]] = []
    for mod_info in pkgutil.walk_packages(statspai.__path__, prefix="statspai."):
        try:
            mod = importlib.import_module(mod_info.name)
        except Exception:
            continue
        for name in dir(mod):
            obj = getattr(mod, name, None)
            if not inspect.isclass(obj):
                continue
            if not name.endswith("Result"):
                continue
            # Skip abstract / test mixins
            if inspect.isabstract(obj):
                continue
            # Skip typing / base classes
            if name in ("BaseModel", "ResultProtocolMixin"):
                continue
            found.append((mod_info.name.split(".")[-1], name, obj))
    # Deduplicate (same class may be re-exported from multiple modules)
    seen: Set[type] = set()
    out: List[Tuple[str, str, type]] = []
    for entry in found:
        if entry[2] in seen:
            continue
        seen.add(entry[2])
        out.append(entry)
    return out


def _has(obj: type, method: str) -> bool:
    attr = getattr(obj, method, None)
    if attr is None:
        return False
    if getattr(attr, "__isabstractmethod__", False):
        return False
    return callable(attr)


def collect() -> dict[str, Any]:
    classes = _imported_classes()
    counts = {m: 0 for m in SECTION3_METHODS}
    full: List[Tuple[str, str]] = []
    near: List[Tuple[str, str, List[str]]] = []  # missing 1-2
    bare: List[Tuple[str, str, List[str]]] = []  # missing 3+
    for mod, name, cls in classes:
        presence = {m: _has(cls, m) for m in SECTION3_METHODS}
        for m, ok in presence.items():
            if ok:
                counts[m] += 1
        missing = [m for m, ok in presence.items() if not ok]
        if not missing:
            full.append((mod, name))
        elif len(missing) <= 2:
            near.append((mod, name, missing))
        else:
            bare.append((mod, name, missing))
    return {
        "totals": {
            "result_classes": len(classes),
            "full_section3": len(full),
        },
        "method_counts": counts,
        "near_full_examples": near[:8],
        "far_examples": bare[:5],
        "floor": FLOOR_FULL_SECTION3,
    }


def check(report: dict[str, Any]) -> int:
    observed = report["totals"]["full_section3"]
    floor = report["floor"]
    if observed < floor:
        print(
            f"[section3_contract_audit] REGRESSION: full-§3 count {observed} "
            f"< floor {floor}",
            file=sys.stderr,
        )
        return 1
    print(
        f"[section3_contract_audit] OK - {observed} result classes satisfy all "
        f"six §3 methods (>= floor {floor})."
    )
    return 0


def render(report: dict[str, Any]) -> str:
    t = report["totals"]
    lines: List[str] = []
    lines.append("StatsPAI §3 result-object contract audit")
    lines.append("=" * 50)
    lines.append(f"Result classes inspected   : {t['result_classes']}")
    lines.append(f"With all six §3 methods    : {t['full_section3']}")
    lines.append(
        f"Ratchet floor (full §3)     : {t['full_section3']} >= {report['floor']}"
    )
    lines.append("")
    lines.append("Per-method coverage")
    lines.append("-" * 50)
    for m in SECTION3_METHODS:
        lines.append(f"  {m:14s}: {report['method_counts'][m]:4d}")
    if report["near_full_examples"]:
        lines.append("")
        lines.append("Near-full (1-2 methods missing) — first 8:")
        for mod, name, missing in report["near_full_examples"]:
            lines.append(f"  {mod}.{name}: missing {missing}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    report = collect()
    if args.check:
        return check(report)
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
