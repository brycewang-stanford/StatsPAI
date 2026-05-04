"""Cross-check ``FunctionSpec.stability == 'stable'`` claims against
parity-test coverage in ``tests/reference_parity/`` + ``tests/external_parity/``.

A function classified as ``stability='stable'`` claims to be
*parity-grade* — numerically aligned with R / Stata / a published
reference, with a locked signature. By design that is the SLA an agent
relies on when filtering ``sp.list_functions(stability='stable')`` for
production use.

The catch: until v1.13 every newly-registered function was *implicitly*
``stable`` (the field's default), so the catalogue's ~970 stable
entries currently mix two populations:

* **Truly parity-backed** — at least one test in
  ``tests/reference_parity/`` or ``tests/external_parity/`` exercises
  the function with R / Stata / paper-replication numbers.
* **Stability-by-default** — never explicitly classified, never
  parity-tested; the ``stable`` label is a placeholder waiting for
  someone to decide.

This script makes the split visible so a maintainer can either (a) add
a parity test to back the claim or (b) flip the entry to
``stability='experimental'`` until one exists.

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
of *unbacked, hand-written* stable claims has not increased beyond a
loose floor.  Auto-registered specs are excluded from the floor because
classifying ~700 of them is a separate (large) project — see ``Step H``
in the v1.13 stability roadmap.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PARITY_DIRS: Tuple[Path, ...] = (
    REPO_ROOT / "tests" / "reference_parity",
    REPO_ROOT / "tests" / "external_parity",
)

#: Loose ceiling on how many *hand-written* stable specs may go without
#: a parity test before --check fails.  Bumped when we deliberately add
#: hand-written entries faster than parity tests.  Decrease over time as
#: the audit gets cleaned up.
UNBACKED_HANDWRITTEN_FLOOR = 220

#: Regex matching ``sp.<name>(`` references in test source.  Used to
#: attribute parity coverage to public ``sp.*`` symbols.
SP_CALL_RE = re.compile(r"\bsp\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")

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
        for path in sorted(parity_dir.rglob("test_*.py")):
            for name in _scan_parity_file(path):
                backed.add(name)
                sources.setdefault(name, []).append(
                    str(path.relative_to(REPO_ROOT))
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

    stable_handwritten: List[str] = []
    stable_auto: List[str] = []
    backed_handwritten: List[str] = []
    backed_auto: List[str] = []
    unbacked_handwritten: List[str] = []
    unbacked_auto: List[str] = []
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
        if is_hand:
            stable_handwritten.append(name)
            (backed_handwritten if name in backed else unbacked_handwritten).append(name)
        else:
            stable_auto.append(name)
            (backed_auto if name in backed else unbacked_auto).append(name)

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
                1 for p in PARITY_DIRS if p.exists()
                for _ in p.rglob("test_*.py")
            ),
            "symbols_referenced_in_parity_tests": len(backed),
        },
        "lists": {
            "unbacked_handwritten": sorted(unbacked_handwritten),
            "unbacked_auto": sorted(unbacked_auto),
            "experimental": sorted(experimental),
            "deprecated": sorted(deprecated),
        },
        "sources": {
            name: srcs for name, srcs in sources.items()
            # Only carry backed-handwritten sources in the JSON payload —
            # auto-registered specs aren't the focus of this audit.
            if name in set(backed_handwritten)
        },
        "floor": {
            "unbacked_handwritten": UNBACKED_HANDWRITTEN_FLOOR,
        },
    }


def render_report(stats: dict, *, show_unbacked: bool = False) -> str:
    t = stats["totals"]
    p = stats["parity_coverage"]
    lines: List[str] = []
    lines.append("StatsPAI stability reverse-audit")
    lines.append("=" * 50)
    lines.append(
        f"Registry         : {t['registry']} functions"
    )
    lines.append(
        f"  stable         : {t['stable']}  "
        f"({t['stable_handwritten']} hand-written, "
        f"{t['stable_auto']} auto-registered)"
    )
    lines.append(f"  experimental   : {t['experimental']}")
    lines.append(f"  deprecated     : {t['deprecated']}")
    lines.append("")
    lines.append("Parity coverage  (sp.<name> referenced in parity tests)")
    lines.append("-" * 50)
    lines.append(
        f"  parity test files                : "
        f"{p['parity_test_files']}"
    )
    lines.append(
        f"  distinct sp.* symbols referenced : "
        f"{p['symbols_referenced_in_parity_tests']}"
    )
    lines.append(
        f"  stable hand-written, BACKED      : "
        f"{p['backed_handwritten']}"
    )
    lines.append(
        f"  stable hand-written, UNBACKED    : "
        f"{p['unbacked_handwritten']}  "
        f"(floor: {stats['floor']['unbacked_handwritten']})"
    )
    lines.append(
        f"  stable auto-registered, BACKED   : "
        f"{p['backed_auto']}"
    )
    lines.append(
        f"  stable auto-registered, UNBACKED : "
        f"{p['unbacked_auto']}"
    )
    lines.append("")
    lines.append("Interpretation")
    lines.append("-" * 50)
    lines.append(
        "* UNBACKED hand-written: a maintainer wrote a parity-grade "
        "spec but no parity test exists. Either add a test or flip "
        "the entry to stability='experimental'."
    )
    lines.append(
        "* UNBACKED auto-registered: silently classified as stable "
        "by default. Most are probably correct (analytic) but the "
        "claim is unenforced. See Step H roadmap for the bulk audit."
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
    if n > floor:
        print(
            f"FAIL: {n} hand-written stable functions lack parity tests "
            f"(floor: {floor}). Either add tests or downgrade to "
            f"stability='experimental'.",
            file=sys.stderr,
        )
        return 1
    print(
        f"OK: {n} hand-written stable functions lack parity tests "
        f"(floor: {floor})."
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON")
    parser.add_argument("--unbacked", action="store_true",
                        help="list unbacked hand-written stable names")
    parser.add_argument("--hand-written", action="store_true",
                        help="restrict report to hand-written specs (default)")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if unbacked count exceeds floor")
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
