"""Reverse-audit stable API entries against parity-test evidence.

StatsPAI now separates API lifecycle from numerical validation evidence:
``stability='stable'`` means the public signature is locked, while
``validation_status='certified'`` / ``'validated'`` carries the
validation-evidence signal. This script keeps the old risk visible by
counting stable API entries that still lack either registry-attached
validation evidence or a parity-test reference in
``tests/reference_parity/`` + ``tests/external_parity/``.

The catch: until v1.13 every newly-registered function was *implicitly*
``stable`` (the field's default), so the catalogue's ~970 stable
entries currently mix two populations:

* **Validation-backed** — the registry marks the function
  ``certified`` / ``validated`` or at least one test in
  ``tests/reference_parity/`` or ``tests/external_parity/`` exercises
  the function with R / Stata / paper-replication numbers.
* **API-stable but unbacked** — the public API is stable, but no
  machine-readable parity-test evidence has been found by this audit.

This script makes the split visible so a maintainer can either (a) add
a parity/reference test, (b) attach validation evidence through the
registry, or (c) flip genuinely immature APIs to
``stability='experimental'``.

It does **not** auto-downgrade. The decision belongs to a human:
something can be analytically correct without a published reference,
and we don't want a one-shot CI run to demote 700 functions overnight.

Usage
-----
::

    python scripts/stability_audit.py                  # human-readable report
    python scripts/stability_audit.py --json           # machine-readable
    python scripts/stability_audit.py --unbacked       # list unbacked names only
    python scripts/stability_audit.py --hand-written   # restrict to hand-written specs
    python scripts/stability_audit.py --check          # exit 1 if regression vs floor

The ``--check`` mode is meant for CI: it succeeds as long as the count
of *unbacked, hand-written* stable API entries has not increased beyond
a loose floor. Auto-registered specs are excluded from the floor because
classifying hundreds of them is a separate validation project.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PARITY_DIRS: Tuple[Path, ...] = (
    REPO_ROOT / "tests" / "reference_parity",
    REPO_ROOT / "tests" / "external_parity",
)

#: No hand-written stable API should lack registry-attached evidence by the
#: JSS submission snapshot. Auto-registered specs remain a separate cleanup
#: project, but human-authored stable entries need at least API/unit evidence.
UNBACKED_HANDWRITTEN_FLOOR = 0

#: Regex matching ``sp.<name>(`` references in test source.  Used to
#: attribute parity coverage to public ``sp.*`` symbols.
SP_CALL_RE = re.compile(r"\bsp\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
EVIDENCE_PATH_RE = re.compile(r"(?P<path>(?:tests|scripts|Paper-JSS)/[^\s,;:)`]+\.py)")

#: Some test files exercise multiple estimators (cross_estimator_parity,
#: published_replications, …) — we credit every ``sp.X(`` reference
#: found in such files even if the file's name doesn't tag a single
#: estimator family.


def _scan_parity_file(path: Path) -> Set[str]:
    """Return every ``sp.<name>`` symbol referenced in a parity test file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    return set(SP_CALL_RE.findall(text))


def _note_paths(note: str) -> List[str]:
    """Extract repository-relative Python evidence paths from a registry note."""
    return [match.group("path") for match in EVIDENCE_PATH_RE.finditer(note)]


def _backed_functions() -> Tuple[Set[str], Dict[str, List[str]]]:
    """Walk every parity test file once.

    Returns
    -------
    backed : Set[str]
        Every ``sp.<name>`` symbol referenced in any parity file.
    sources : Dict[str, List[str]]
        For each backed name, the list of parity test files that reference it.
    """
    backed: Set[str] = set()
    sources: Dict[str, List[str]] = {}
    for parity_dir in PARITY_DIRS:
        if not parity_dir.exists():
            continue
        # POSIX-keyed sort + as_posix() so the emitted evidence-source notes are
        # byte-identical on Windows and POSIX (matches the registry's
        # _scan_reference_tests fix and the other audit scripts which already
        # use .as_posix()).
        for path in sorted(
            parity_dir.rglob("test_*.py"),
            key=lambda p: p.relative_to(REPO_ROOT).as_posix(),
        ):
            for name in _scan_parity_file(path):
                backed.add(name)
                sources.setdefault(name, []).append(
                    path.relative_to(REPO_ROOT).as_posix()
                )
    return backed, sources


def _registry_specs():
    """Return (registry, hand_written_set).  Lazy-imports statspai."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import statspai as sp  # noqa: WPS433

    sp.list_functions()  # force full registry
    from statspai.registry import _REGISTRY  # noqa: WPS433

    hand_written: Set[str] = set()
    for name, spec in _REGISTRY.items():
        # Auto-registered specs are flagged ``_auto = True`` by
        # ``_auto_spec_from_callable``; hand-written ones aren't.
        if not getattr(spec, "_auto", False):
            hand_written.add(name)
    return _REGISTRY, hand_written


def collect() -> dict:
    registry, hand_written = _registry_specs()
    backed, sources = _backed_functions()
    evidence_sources: Dict[str, List[str]] = {k: list(v) for k, v in sources.items()}
    evidence_path_refs = 0
    evidence_paths: Set[str] = set()
    missing_evidence_paths: List[str] = []

    stable_handwritten: List[str] = []
    stable_auto: List[str] = []
    backed_handwritten: List[str] = []
    backed_auto: List[str] = []
    unbacked_handwritten: List[str] = []
    unbacked_auto: List[str] = []
    unbacked_auto_categories: Counter[str] = Counter()
    unbacked_auto_examples: Dict[str, List[str]] = {}
    unbacked_auto_classlike = 0
    experimental: List[str] = []
    deprecated: List[str] = []

    for name, spec in sorted(registry.items()):
        if spec.stability == "experimental":
            experimental.append(name)
            continue
        if spec.stability == "deprecated":
            deprecated.append(name)
            continue
        # spec.stability == "stable"
        is_hand = name in hand_written
        notes = list(getattr(spec, "validation_notes", []) or [])
        registry_backed = spec.validation_status in {"certified", "validated"} or bool(
            notes
        )
        if registry_backed:
            if not notes:
                notes = [f"registry validation_status={spec.validation_status}"]
            evidence_sources.setdefault(name, [])
            for note in notes:
                if note not in evidence_sources[name]:
                    evidence_sources[name].append(note)
                for rel_path in _note_paths(note):
                    evidence_path_refs += 1
                    evidence_paths.add(rel_path)
                    if not (REPO_ROOT / rel_path).exists():
                        missing_evidence_paths.append(
                            f"{name}: missing evidence file {rel_path}"
                        )
        is_backed = name in backed or registry_backed
        if is_hand:
            stable_handwritten.append(name)
            (backed_handwritten if is_backed else unbacked_handwritten).append(name)
        else:
            stable_auto.append(name)
            if is_backed:
                backed_auto.append(name)
            else:
                unbacked_auto.append(name)
                category = getattr(spec, "category", "") or "uncategorized"
                unbacked_auto_categories[category] += 1
                if len(unbacked_auto_examples.setdefault(category, [])) < 8:
                    unbacked_auto_examples[category].append(name)
                if name[:1].isupper():
                    unbacked_auto_classlike += 1

    return {
        "totals": {
            "registry": len(registry),
            "stable": len(stable_handwritten) + len(stable_auto),
            "stable_handwritten": len(stable_handwritten),
            "stable_auto": len(stable_auto),
            "experimental": len(experimental),
            "deprecated": len(deprecated),
        },
        "parity_coverage": {
            "backed_handwritten": len(backed_handwritten),
            "backed_auto": len(backed_auto),
            "unbacked_handwritten": len(unbacked_handwritten),
            "unbacked_auto": len(unbacked_auto),
            "parity_test_files": sum(
                1 for p in PARITY_DIRS if p.exists() for _ in p.rglob("test_*.py")
            ),
            "symbols_referenced_in_parity_tests": len(backed),
            "registry_validated_symbols": sum(
                1
                for spec in registry.values()
                if spec.stability == "stable"
                and spec.validation_status in {"certified", "validated"}
            ),
        },
        "lists": {
            "unbacked_handwritten": sorted(unbacked_handwritten),
            "unbacked_auto": sorted(unbacked_auto),
            "experimental": sorted(experimental),
            "deprecated": sorted(deprecated),
        },
        "auto_unbacked_breakdown": {
            "category_counts": dict(
                sorted(
                    unbacked_auto_categories.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "category_examples": {
                category: sorted(values)
                for category, values in sorted(
                    unbacked_auto_examples.items(),
                    key=lambda item: (-len(item[1]), item[0]),
                )
            },
            "classlike_symbol_count": unbacked_auto_classlike,
            "functionlike_symbol_count": len(unbacked_auto) - unbacked_auto_classlike,
        },
        "sources": {
            name: srcs
            for name, srcs in evidence_sources.items()
            # Only carry backed-handwritten sources in the JSON payload —
            # auto-registered specs aren't the focus of this audit.
            if name in set(backed_handwritten)
        },
        "evidence_paths": {
            "refs": evidence_path_refs,
            "unique": len(evidence_paths),
            "missing": sorted(missing_evidence_paths),
        },
        "floor": {
            "unbacked_handwritten": UNBACKED_HANDWRITTEN_FLOOR,
        },
    }


def render_report(stats: dict, *, show_unbacked: bool = False) -> str:
    t = stats["totals"]
    p = stats["parity_coverage"]
    lines: List[str] = []
    lines.append("StatsPAI stability/validation reverse-audit")
    lines.append("=" * 50)
    lines.append(f"Registry         : {t['registry']} functions")
    lines.append(
        f"  stable         : {t['stable']}  "
        f"({t['stable_handwritten']} hand-written, "
        f"{t['stable_auto']} auto-registered)"
    )
    lines.append(f"  experimental   : {t['experimental']}")
    lines.append(f"  deprecated     : {t['deprecated']}")
    lines.append("")
    lines.append("Validation coverage")
    lines.append("-" * 50)
    lines.append(f"  parity test files                : " f"{p['parity_test_files']}")
    lines.append(
        f"  distinct sp.* symbols referenced : "
        f"{p['symbols_referenced_in_parity_tests']}"
    )
    lines.append(
        f"  registry certified/validated     : " f"{p['registry_validated_symbols']}"
    )
    lines.append(f"  stable hand-written, BACKED      : " f"{p['backed_handwritten']}")
    lines.append(
        f"  stable hand-written, UNBACKED    : "
        f"{p['unbacked_handwritten']}  "
        f"(floor: {stats['floor']['unbacked_handwritten']})"
    )
    lines.append(f"  stable auto-registered, BACKED   : " f"{p['backed_auto']}")
    lines.append(f"  stable auto-registered, UNBACKED : " f"{p['unbacked_auto']}")
    auto_breakdown = stats.get("auto_unbacked_breakdown", {})
    category_counts = auto_breakdown.get("category_counts", {})
    if category_counts:
        lines.append(
            "    class/function split           : "
            f"{auto_breakdown.get('classlike_symbol_count', 0)} class-like / "
            f"{auto_breakdown.get('functionlike_symbol_count', 0)} function-like"
        )
        top_categories = list(category_counts.items())[:8]
        lines.append(
            "    top auto-unbacked categories   : "
            + ", ".join(f"{name}={count}" for name, count in top_categories)
        )
    e = stats["evidence_paths"]
    lines.append(
        f"  registry evidence path refs      : "
        f"{e['refs']} refs / {e['unique']} unique "
        f"(missing: {len(e['missing'])})"
    )
    lines.append("")
    lines.append("Interpretation")
    lines.append("-" * 50)
    lines.append(
        "* UNBACKED hand-written: a maintainer wrote a stable public "
        "API, but this audit found no registry validation evidence and "
        "no parity-test reference. Add evidence, add a test, or mark "
        "immature APIs experimental."
    )
    lines.append(
        "* UNBACKED auto-registered: classified as stable by default. "
        "Most are API-compatible wrappers or result/helper classes, but "
        "numerical evidence is not yet machine-readable. The category "
        "breakdown above keeps that denominator auditable instead of "
        "letting it masquerade as validated evidence."
    )
    lines.append("")
    if show_unbacked:
        lines.append("Unbacked hand-written stable functions")
        lines.append("-" * 50)
        for name in stats["lists"]["unbacked_handwritten"]:
            lines.append(f"  {name}")
        lines.append("")
    return "\n".join(lines)


def check_drift(stats: dict) -> int:
    n = stats["parity_coverage"]["unbacked_handwritten"]
    floor = stats["floor"]["unbacked_handwritten"]
    missing_paths = stats["evidence_paths"]["missing"]
    if missing_paths:
        print(
            "FAIL: registry evidence notes reference missing files: "
            + "; ".join(missing_paths[:20]),
            file=sys.stderr,
        )
        return 1
    if n > floor:
        print(
            f"FAIL: {n} hand-written stable API entries lack parity tests "
            f"or registry validation evidence (floor: {floor}). Either "
            f"add evidence, add tests, or downgrade immature APIs to "
            f"experimental.",
            file=sys.stderr,
        )
        return 1
    print(
        f"OK: {n} hand-written stable API entries lack parity tests "
        f"or registry validation evidence (floor: {floor})."
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    parser.add_argument(
        "--unbacked",
        action="store_true",
        help="list unbacked hand-written stable names",
    )
    parser.add_argument(
        "--hand-written",
        action="store_true",
        help="restrict report to hand-written specs (default)",
    )
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if unbacked count exceeds floor"
    )
    args = parser.parse_args(argv)

    stats = collect()
    if args.check:
        return check_drift(stats)
    if args.json:
        print(json.dumps(stats, indent=2))
        return 0
    print(render_report(stats, show_unbacked=args.unbacked))
    return 0


if __name__ == "__main__":
    sys.exit(main())
