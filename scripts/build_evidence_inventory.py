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


def _option_inputs() -> List[Dict[str, Any]]:
    """The CSV inputs the option-level do-files read, with their SHA-256.

    A fixture is only as fixed as the bytes it was computed from; these
    were hashed by nothing.
    """
    return [
        {
            "input": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(OPTION_DIR.glob("data_*.csv"))
    ]


HOLDOUT_DIR = REPO_ROOT / "tests" / "stata_translation_holdout"


def _translation_holdout() -> List[Dict[str, Any]]:
    """The frozen Stata-translation holdout, hashed so it stays frozen."""
    names = (
        "corpus.json",
        "holdout_Stata.json",
        "holdout_cross.csv",
        "holdout_panel.csv",
    )
    return [
        {
            "file": (HOLDOUT_DIR / name).relative_to(REPO_ROOT).as_posix(),
            "sha256": hashlib.sha256((HOLDOUT_DIR / name).read_bytes()).hexdigest(),
        }
        for name in names
        if (HOLDOUT_DIR / name).exists()
    ]


MC_DIR = REPO_ROOT / "tests" / "coverage_monte_carlo" / "results_b1000"


def _reliability() -> Dict[str, Any]:
    """Coverage, stress-design and size/power rows, read from Track B.

    Parity says an estimator reproduces a reference; it does not say the
    interval covers. These are the committed Monte Carlo results that speak
    to that, tabulated as they stand: every replication, failed ones
    included, is in the denominator, and the Monte Carlo standard error of
    each rate is shown so a rate is not read more finely than ``B`` allows.
    """

    def _load(name: str) -> List[Dict[str, Any]]:
        path = MC_DIR / name
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def _mc_se(rate: float, b: int) -> float:
        return round((rate * (1.0 - rate) / b) ** 0.5, 4)

    coverage = [
        {
            "design": r["name"],
            "B": r["B"],
            "covered": r["covered"],
            "failures": r.get("failures", 0),
            "rate": r["rate"],
            "mc_se": _mc_se(r["rate"], r["B"]),
            "se_to_sd_ratio": round(r["se_sd_ratio"], 3),
            "within_2_mc_se_of_nominal": abs(r["rate"] - 0.95)
            <= 2 * _mc_se(0.95, r["B"]),
        }
        for r in _load("coverage_b1000.json")
    ]
    stress = [
        {
            "design": r["name"],
            "B": r["B"],
            "rate": r["rate"],
            "mc_se": _mc_se(r["rate"], r["B"]),
            "documented_band": r.get("documented_band"),
            "note": r.get("note"),
        }
        for r in _load("coverage_robustness_b1000.json")
    ]
    size_power = [
        {
            "design": r["name"],
            "B": r["B"],
            "size": r["size"],
            "mc_se": _mc_se(r["size"], r["B"]),
            "power": dict(zip((str(d) for d in r["deltas"]), r["power"])),
        }
        for r in _load("size_power_b1000.json")
    ]
    return {
        "nominal_level": 0.95,
        "source": "tests/coverage_monte_carlo/results_b1000/",
        "coverage": coverage,
        "stress_designs": stress,
        "size_and_power": size_power,
        "studies": _reliability_studies(),
    }


STUDY_DIR = REPO_ROOT / "tests" / "reliability"

#: Keys of a study cell that describe the design (the rest are results).
_DESIGN_KEYS = (
    "model",
    "shape",
    "pattern",
    "errors",
    "treated",
    "sizes",
    "n",
    "N",
    "G",
    "size",
    "sigma",
    "support_per_side",
)
_METHOD_KEYS = ("method", "learner")


def _study_rows(study: str, block: str, cell: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per method of one design cell, refusals in the denominator."""
    design = ", ".join(f"{k}={cell[k]}" for k in _DESIGN_KEYS if k in cell)
    b = int(cell["B"])
    if "coverage" in cell or "rejection_rate" in cell:
        label = next((str(cell[k]) for k in _METHOD_KEYS if k in cell), "default")
        methods = {label: cell}
    else:
        methods = {
            k: v
            for k, v in cell.items()
            if isinstance(v, dict) and ("coverage" in v or "rejection_rate" in v)
        }
    rows = []
    for method, res in methods.items():
        quantity = "coverage" if "coverage" in res else "rejection"
        fitted_rate = float(
            res["coverage" if quantity == "coverage" else "rejection_rate"]
        )
        refused = int(sum((res.get("refused") or {}).values()))
        fitted = int(res.get("n_fitted", b - refused))
        # A refused fit did not produce an interval that covers, nor a
        # rejection: it counts in the denominator and not in the numerator.
        rate = fitted_rate * fitted / b if fitted else 0.0
        nominal = 0.95 if quantity == "coverage" else 0.05
        mc_se = (rate * (1.0 - rate) / b) ** 0.5
        nominal_se = (nominal * (1.0 - nominal) / b) ** 0.5
        rows.append(
            {
                "study": study,
                "block": block,
                "design": design,
                "method": method,
                "quantity": quantity,
                "B": b,
                "refused": refused,
                "rate": round(rate, 4),
                "mc_se": round(mc_se, 4),
                "nominal": nominal,
                "within_2_mc_se_of_nominal": abs(rate - nominal) <= 2 * nominal_se,
            }
        )
    return rows


def _reliability_studies() -> List[Dict[str, Any]]:
    """The simulation studies under ``tests/reliability/``, one row a method.

    Each study fixes its design in the script's docstring before the first
    run and stores every cell; this reads the stored files as they stand.
    """
    studies = []
    for path in sorted(STUDY_DIR.glob("*_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows: List[Dict[str, Any]] = []
        for block, cells in payload.items():
            if isinstance(cells, list):
                for cell in cells:
                    rows.extend(_study_rows(payload["study"], block, cell))
        studies.append(
            {
                "study": payload["study"],
                "source": str(path.relative_to(REPO_ROOT)),
                "script": str(
                    path.with_name(
                        path.name.replace("_results.json", ".py")
                    ).relative_to(REPO_ROOT)
                ),
                "rows": rows,
            }
        )
    return studies


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
        "option_inputs": _option_inputs(),
        "translation_holdout": _translation_holdout(),
        "inference_reliability": _reliability(),
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
    if inv.get("translation_holdout"):
        lines += [
            "",
            "## Stata translation holdout",
            "",
            "A frozen corpus of 39 Stata commands and the numbers Stata 18 MP "
            "gives for the 33 that run, scored on five layers by "
            "`tests/test_stata_translation_holdout.py`. Hashed here so the "
            "corpus cannot drift towards what the translator already handles.",
            "",
            "| File | SHA-256 (first 16) |",
            "| --- | --- |",
        ]
        for item in inv["translation_holdout"]:
            lines.append(f"| `{item['file']}` | `{item['sha256'][:16]}` |")
    rel = inv.get("inference_reliability") or {}
    if rel.get("coverage"):
        lines += [
            "",
            "## Inference reliability (Track B)",
            "",
            "Reference evidence says an estimator reproduces another "
            "implementation. Whether its interval covers is a separate "
            "question, answered here from the committed Monte Carlo runs "
            f"under `{rel['source']}`. Every replication is in the "
            "denominator, failed ones included. `MC SE` is the Monte Carlo "
            "standard error of the rate; a rate within two of them of 0.95 "
            "is not distinguishable from nominal at this `B`.",
            "",
            "| Design | B | Failed | Coverage | MC SE | Mean SE / MC SD | "
            "Within 2 MC SE of 0.95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for r in rel["coverage"]:
            lines.append(
                f"| {r['design']} | {r['B']} | {r['failures']} | {r['rate']:.3f} | "
                f"{r['mc_se']:.4f} | {r['se_to_sd_ratio']:.3f} | "
                f"{'yes' if r['within_2_mc_se_of_nominal'] else '**no**'} |"
            )
        lines += [
            "",
            "Designs built to break an assumption. The band is the range "
            "documented in advance for that design, not a pass mark for "
            "nominal coverage.",
            "",
            "| Stress design | B | Coverage | MC SE | Documented band | Note |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
        for r in rel["stress_designs"]:
            band = r["documented_band"]
            lines.append(
                f"| {r['design']} | {r['B']} | {r['rate']:.3f} | {r['mc_se']:.4f} | "
                f"{band[0]:.2f} to {band[1]:.2f} | {r['note'] or ''} |"
            )
        lines += [
            "",
            "| Design | B | Size at 5% | MC SE | Power by effect size |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for r in rel["size_and_power"]:
            power = ", ".join(f"{d}: {p:.3f}" for d, p in r["power"].items())
            lines.append(
                f"| {r['design']} | {r['B']} | {r['size']:.3f} | {r['mc_se']:.4f} | "
                f"{power} |"
            )
    if rel.get("studies"):
        lines += [
            "",
            "## Reliability studies",
            "",
            "Simulation studies under `tests/reliability/`, each with its "
            "design fixed in the script before the first run. A row of a "
            "study is one method on one design; a fit that was refused "
            "counts in the denominator. The table gives, per study and "
            "method, how many designs land within two Monte Carlo standard "
            "errors of the nominal rate, and the design furthest from it. "
            "Every row is in `docs/evidence_inventory.json`; the reading of "
            "each study is in `tests/reliability/README.md`.",
            "",
            "| Study | Method | Quantity | B | Designs | At nominal | "
            "Furthest from nominal |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
        for study in rel["studies"]:
            by_method: Dict[str, List[Dict[str, Any]]] = {}
            for row in study["rows"]:
                by_method.setdefault(row["method"], []).append(row)
            for method, rows in by_method.items():
                worst = max(rows, key=lambda r: abs(r["rate"] - r["nominal"]))
                ok = sum(r["within_2_mc_se_of_nominal"] for r in rows)
                refused = f", {worst['refused']} refused" if worst["refused"] else ""
                lines.append(
                    f"| `{study['study']}` | {method} | {rows[0]['quantity']} | "
                    f"{rows[0]['B']} | {len(rows)} | {ok} | "
                    f"{worst['rate']:.3f} ({worst['design']}{refused}) |"
                )
    lines += [
        "",
        "Inputs those do-files read (the other fixtures use "
        "`tests/orig_parity/data/02_mpdta_original.csv`):",
        "",
        "| Input | SHA-256 (first 16) |",
        "| --- | --- |",
    ]
    for item in inv.get("option_inputs", []):
        lines.append(f"| `{item['input']}` | `{item['sha256'][:16]}` |")
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
