#!/usr/bin/env python3
"""Build the entry x configuration x output evidence inventory.

``sp.validation_scope`` answers, for one fit, which artifacts cover the
configuration that was run. This script asks the same question of every
configuration at once: for each estimator with a scope map it walks the
full grid of its dimensions, calls ``sp.validation_scope`` on each cell,
and counts, per output (estimate / se / coverage / diagnostic / vcov /
joint_test), how many cells have which evidence status. The result is the
generated replacement for the hand-maintained point-in-time table in
``docs/parity_object_coverage.md``.

Nothing here is asserted: every number is a count over
``statspai.validation_scope.SCOPES``, whose rows are themselves checked
against the artifacts they name by ``tests/test_validation_scope.py``. A
function without a scope map is not in the inventory, and absence is not
a statement that it lacks evidence -- it has function-level evidence only
(``sp.parity_status``).

The second section inventories the option-level Stata fixtures under
``tests/stata_parity/option_parity/``, which sit outside the Track A
enumeration on purpose: for each fixture, the tests that read it and
its SHA-256. ``TIER_A_FIXTURE_LOCK.json`` does not cover these files, so
the hash recorded here is their lock: editing a fixture makes ``--check``
fail until the inventory is regenerated, which shows up in the diff.

Outputs
-------
* ``docs/evidence_inventory.json`` -- machine-readable.
* ``docs/evidence_inventory.md`` -- the same, as tables.

Usage
-----
    python scripts/build_evidence_inventory.py            # regenerate
    python scripts/build_evidence_inventory.py --check    # drift gate
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pathlib
import sys
from typing import Any, Dict, List

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
JSON_OUT = REPO_ROOT / "docs" / "evidence_inventory.json"
MD_OUT = REPO_ROOT / "docs" / "evidence_inventory.md"
OPTION_DIR = REPO_ROOT / "tests" / "stata_parity" / "option_parity"
FIXTURE_LOCK = REPO_ROOT / "tests" / "r_parity" / "TIER_A_FIXTURE_LOCK.json"

#: Per-output statuses, strongest first (mirrors validation_scope).
STATUSES = (
    "reference",
    "seed_equivalence",
    "stochastic_screen",
    "coverage_simulation",
    "disclosure",
    "not_covered",
)
OVERALL = (
    "covered",
    "estimate_only",
    "stochastic_only",
    "disclosure_only",
    "not_covered",
)


def _scope_inventory(name: str) -> Dict[str, Any]:
    from statspai.validation_scope import OUTPUTS, SCOPES, validation_scope

    scope = SCOPES[name]
    dims = list(scope.dimensions)
    by_output: Dict[str, Dict[str, int]] = {}
    overall: Dict[str, int] = {}
    cells = 0
    for values in itertools.product(*(scope.domains[d] for d in dims)):
        cells += 1
        res = validation_scope(function=name, **dict(zip(dims, values)))
        overall[res["status"]] = overall.get(res["status"], 0) + 1
        for out in OUTPUTS:
            if out not in res["outputs"]:
                continue
            status = res["outputs"][out]["status"]
            slot = by_output.setdefault(out, {})
            slot[status] = slot.get(status, 0) + 1
    return {
        "dimensions": {d: list(scope.domains[d]) for d in dims},
        "primary_outputs": list(scope.primary),
        "invariant": {
            d: {"outputs": list(outs), "reason": reason}
            for d, (outs, reason) in scope.invariant.items()
        },
        "cells": cells,
        "overall": {k: overall[k] for k in OVERALL if k in overall},
        "by_output": {
            out: {s: counts[s] for s in STATUSES if s in counts}
            for out, counts in by_output.items()
        },
        "rows": [
            {
                "kind": row.kind,
                "artifact": row.artifact,
                "entry_point": row.entry_point,
                "outputs": list(row.outputs),
                "configuration": {k: sorted(v) for k, v in row.config.items()},
                "compares": row.compares,
            }
            for row in scope.rows
        ],
    }


def _option_fixtures() -> List[Dict[str, Any]]:
    """Each option-level Stata fixture, the tests reading it, its lock state."""
    locked = ""
    if FIXTURE_LOCK.exists():
        locked = FIXTURE_LOCK.read_text(encoding="utf-8")
    tests = sorted((REPO_ROOT / "tests").rglob("test_*.py"))
    sources = {p: p.read_text(encoding="utf-8") for p in tests}
    out = []
    for fixture in sorted(OPTION_DIR.rglob("*_Stata.json")):
        rel = fixture.relative_to(REPO_ROOT).as_posix()
        consumers = sorted(
            p.relative_to(REPO_ROOT).as_posix()
            for p, text in sources.items()
            if fixture.name in text
        )
        out.append(
            {
                "fixture": rel,
                "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                "consumers": consumers,
                "in_tier_a_lock": fixture.name in locked,
            }
        )
    return out


def build() -> Dict[str, Any]:
    from statspai.validation_scope import OUTPUTS, SCOPE_FUNCTIONS

    functions = {name: _scope_inventory(name) for name in SCOPE_FUNCTIONS}
    totals: Dict[str, Dict[str, int]] = {}
    for inv in functions.values():
        for out, counts in inv["by_output"].items():
            slot = totals.setdefault(out, {})
            for status, n in counts.items():
                slot[status] = slot.get(status, 0) + n
    return {
        "schema": 1,
        "generated_by": "scripts/build_evidence_inventory.py",
        "source": "statspai.validation_scope.SCOPES",
        "note": (
            "Counts are cells of each estimator's configuration grid. A cell "
            "is one value per dimension; 'reference' means a T1/T2 row ran "
            "that configuration and compared that output. Functions without "
            "a scope map are absent, which is not a statement about them."
        ),
        "functions": functions,
        "totals_by_output": {
            out: {s: totals[out][s] for s in STATUSES if s in totals[out]}
            for out in OUTPUTS
            if out in totals
        },
        "option_fixtures": _option_fixtures(),
    }


def _pct(n: int, d: int) -> str:
    return f"{n} / {d}" if d else "--"


def render(inv: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Evidence inventory: entry x configuration x output",
        "",
        "Generated by `python scripts/build_evidence_inventory.py` from",
        "`statspai.validation_scope.SCOPES`. Do not edit by hand; "
        "`--check` is the drift gate.",
        "",
        "Each estimator below has a configuration grid: one value per "
        "dimension. The table counts the cells in which an output has "
        "**reference** evidence, meaning a known-truth (T1) or same-byte "
        "cross-language (T2) artifact ran that configuration and compared "
        "that output. A function-level tier says nothing about which cells "
        "those are. For one fitted result, call `sp.validation_scope(result)`.",
        "",
        "Functions without a scope map are not listed. That is not a claim "
        "that they lack evidence: they carry function-level evidence only "
        "(`sp.parity_status`).",
        "",
        "## Cells with reference evidence, by output",
        "",
        "| Function | Cells | estimate | se | vcov | joint_test | diagnostic | "
        "Fully covered |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, f in inv["functions"].items():
        cells = f["cells"]

        def ref(out: str) -> str:
            counts = f["by_output"].get(out)
            if counts is None:
                return "--"
            return _pct(counts.get("reference", 0), cells)

        lines.append(
            f"| `{name}` | {cells} | {ref('estimate')} | {ref('se')} | "
            f"{ref('vcov')} | {ref('joint_test')} | {ref('diagnostic')} | "
            f"{_pct(f['overall'].get('covered', 0), cells)} |"
        )
    lines += [
        "",
        "`--` means no artifact compares that output for the function at "
        "all. `Fully covered` counts cells whose primary outputs all have "
        "reference evidence.",
        "",
        "## Other evidence statuses",
        "",
        "Cells whose strongest evidence for an output is not a reference "
        "row. These are reported at their own grade and are not parity.",
        "",
        "| Function | Output | seed_equivalence (T3) | stochastic_screen (S) | "
        "coverage_simulation (B) | disclosure (T4) | not_covered |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, f in inv["functions"].items():
        for out, counts in f["by_output"].items():
            rest = [
                counts.get(s, 0)
                for s in (
                    "seed_equivalence",
                    "stochastic_screen",
                    "coverage_simulation",
                    "disclosure",
                    "not_covered",
                )
            ]
            lines.append(
                f"| `{name}` | {out} | " + " | ".join(str(n) for n in rest) + " |"
            )
    lines += ["", "## Artifacts per function", ""]
    for name, f in inv["functions"].items():
        dims = "; ".join(
            f"`{d}` in {{{', '.join(v)}}}" for d, v in f["dimensions"].items()
        )
        lines += [
            f"### `{name}`",
            "",
            f"Dimensions: {dims}.",
            "",
            "| Kind | Outputs | Configuration run | Artifact | Entry point |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in f["rows"]:
            cfg = "; ".join(
                f"{k}={'/'.join(v)}" for k, v in row["configuration"].items()
            )
            lines.append(
                f"| {row['kind']} | {', '.join(row['outputs'])} | {cfg} | "
                f"`{row['artifact']}` | `{row['entry_point']}` |"
            )
        for d, spec in f["invariant"].items():
            lines += [
                "",
                f"`{d}` is ignored for {', '.join(spec['outputs'])}: "
                f"{spec['reason']}",
            ]
        lines.append("")
    lines += [
        "## Option-level Stata fixtures",
        "",
        "Fixtures under `tests/stata_parity/option_parity/` pin option "
        "switches within an estimator and sit outside the Track A "
        "enumeration on purpose, so `TIER_A_FIXTURE_LOCK.json` does not hash "
        "them. The SHA-256 below is their lock: an edited fixture fails "
        "`--check` until this file is regenerated. A fixture read by no test "
        "fails the build.",
        "",
        "| Fixture | Read by | SHA-256 (first 16) |",
        "| --- | --- | --- |",
    ]
    for fx in inv["option_fixtures"]:
        consumers = "<br>".join(f"`{c}`" for c in fx["consumers"])
        lines.append(f"| `{fx['fixture']}` | {consumers} | `{fx['sha256'][:16]}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed inventory is stale",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    inv = build()
    serialized = json.dumps(inv, indent=2, sort_keys=False) + "\n"
    doc = render(inv)

    orphans = [fx["fixture"] for fx in inv["option_fixtures"] if not fx["consumers"]]
    if orphans:
        print(
            "[build_evidence_inventory] option fixtures read by no test: "
            + ", ".join(orphans),
            file=sys.stderr,
        )
        return 1

    if args.check:
        stale = [
            path.relative_to(REPO_ROOT).as_posix()
            for path, want in ((JSON_OUT, serialized), (MD_OUT, doc))
            if not path.exists() or path.read_text(encoding="utf-8") != want
        ]
        if stale:
            print(
                "[build_evidence_inventory] STALE: "
                + ", ".join(stale)
                + " -- run python scripts/build_evidence_inventory.py",
                file=sys.stderr,
            )
            return 1
        print("[build_evidence_inventory] OK")
        return 0

    JSON_OUT.write_text(serialized, encoding="utf-8")
    MD_OUT.write_text(doc, encoding="utf-8")
    print(
        f"[build_evidence_inventory] wrote {JSON_OUT.relative_to(REPO_ROOT)} "
        f"and {MD_OUT.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
