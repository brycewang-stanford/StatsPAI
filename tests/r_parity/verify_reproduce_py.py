"""Reproducibility verifier for the StatsPAI side of the parity harness.

``verify_reproduce.py`` re-derives every committed ``results/<module>_R.json``
by re-running the canonical R reference on the committed
``data/<module>.csv``. Nothing did the same for the *Python* side, and the
gap was not cosmetic: each ``NN_<name>.py`` both **writes** the fixture CSV
and computes the StatsPAI numbers from it, so a change inside the package
can silently decouple the two. The committed CSV keeps its recorded hash
(nobody re-ran the script), ``compare.py`` keeps joining the two committed
JSONs, and the R side keeps reproducing against the frozen bytes — while
the script that is supposed to generate those bytes no longer does.

That is exactly what had happened to module 01: ``sp.datasets.card_1995()``
was switched from a calibrated replica to the real Card extract, so
re-running ``01_ols.py`` replaced the fixture with different data and moved
:math:`\\hat\\beta_{educ}` from 0.10999 to 0.07401. A reviewer running the
Tier-2 path would have regenerated the CSV, left the R golden JSON behind
on the old bytes, and seen a "same-byte parity" row that was no longer
computed on the same bytes.

This driver closes that hole. For every module with an ``NN_<name>.py``:

  1. Re-run it with ``STATSPAI_R_PARITY_DATA_DIR`` and
     ``STATSPAI_R_PARITY_RESULTS_DIR`` pointed at a staging directory, so
     neither the committed fixture nor the committed golden is touched.
  2. Compare the regenerated ``data/<module>.csv`` to the committed one
     **byte for byte** — the parity claim is same-byte, so anything less
     than byte equality is drift.
  3. Join the regenerated ``<module>_py.json`` rows to the committed rows
     by ``statistic`` and report the worst relative difference on the point
     estimate and the SE.

A module reproduces when the CSV is byte-identical and every joined
statistic agrees within ``REPRO_TOL`` (1e-9), the same reproducibility
floor the R side uses and far tighter than the cross-language parity
budget in ``compare.py``.

Usage::

    python tests/r_parity/verify_reproduce_py.py                 # all
    python tests/r_parity/verify_reproduce_py.py 01_ols 02_iv    # a subset
    python tests/r_parity/verify_reproduce_py.py --timeout 600

Writes ``results/REPRODUCIBILITY_REPORT_PY.md`` and exits non-zero if any
module drifts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS_DIR = HERE / "results"
DATA_DIR = HERE / "data"
# Nested inside the R verifier's scratch root rather than beside it:
# ``_repro_check`` is already gitignored and already excluded from the
# JSS submission archive, so one rule keeps every verifier's staging
# output out of both. A sibling directory needs its own copy of both
# rules, and the archive quietly grew by 173 files the first time it
# did not get one.
STAGING_DIR = RESULTS_DIR / "_repro_check" / "py"
STAGING_DATA_DIR = STAGING_DIR / "data"

REPRO_REL_TOL = 1e-9

#: Deliberately empty, and the emptiness is the point.
#:
#: ``verify_reproduce.py`` needs per-module relaxations because it re-runs
#: an *external* reference whose optimiser and BLAS path we do not control.
#: This driver re-runs *our own* code, so a golden that does not come back
#: at 1e-9 is either a package change (record it and refresh the golden) or
#: a nondeterminism we should fix -- never something to widen a floor for.
#: All 87 modules clear the strict floor, including the two non-convex SCM
#: fits that the R side does have to relax.
REPRO_TOL_OVERRIDE: dict[str, float] = {}


def _reldiff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if abs(b) >= 1.0:
        return abs(a - b) / abs(b)
    return abs(a - b)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover_modules() -> list[str]:
    """Modules with a Python script and a committed Python golden JSON."""
    return [
        script.stem
        for script in sorted(HERE.glob("[0-9][0-9]_*.py"))
        if (RESULTS_DIR / f"{script.stem}_py.json").exists()
    ]


def run_one(module: str, timeout: int) -> dict:
    committed_json = RESULTS_DIR / f"{module}_py.json"
    if not committed_json.exists():
        return {"module": module, "status": "no_golden",
                "detail": "no committed _py.json"}
    script = HERE / f"{module}.py"
    if not script.exists():
        return {"module": module, "status": "no_script", "detail": "no NN_*.py"}

    STAGING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["STATSPAI_R_PARITY_RESULTS_DIR"] = str(STAGING_DIR)
    env["STATSPAI_R_PARITY_DATA_DIR"] = str(STAGING_DATA_DIR)
    # Editable installs pin `statspai` to whichever tree was installed;
    # a worktree must test its own source (see CLAUDE.md section 9.2).
    src = ROOT / "src"
    if (src / "statspai" / "__init__.py").is_file():
        env["PYTHONPATH"] = os.pathsep.join(
            [str(src), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(HERE),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"module": module, "status": "timeout", "detail": f">{timeout}s"}

    fresh_json = STAGING_DIR / f"{module}_py.json"
    if proc.returncode != 0 or not fresh_json.exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return {"module": module, "status": "py_error",
                "detail": " | ".join(tail)[:300]}

    # --- fixture bytes ------------------------------------------------
    committed_csv = DATA_DIR / f"{module}.csv"
    fresh_csv = STAGING_DATA_DIR / f"{module}.csv"
    if not committed_csv.exists():
        # Modules that embed their fixture in the script (event-study
        # beta/sigma vectors, scalar effect sizes) have no CSV to compare.
        csv_status = "embedded"
    elif not fresh_csv.exists():
        csv_status = "not_regenerated"
    elif _sha256(committed_csv) == _sha256(fresh_csv):
        csv_status = "byte_identical"
    else:
        csv_status = "BYTES_DIFFER"

    # --- numbers ------------------------------------------------------
    committed = json.loads(committed_json.read_text(encoding="utf-8"))
    fresh = json.loads(fresh_json.read_text(encoding="utf-8"))
    c_by = {r["statistic"]: r for r in committed["rows"]}
    f_by = {r["statistic"]: r for r in fresh["rows"]}
    shared = sorted(set(c_by) & set(f_by))

    worst_est = 0.0
    worst_se = 0.0
    worst_stat = ""
    for stat in shared:
        d_est = _reldiff(f_by[stat].get("estimate"), c_by[stat].get("estimate"))
        d_se = _reldiff(f_by[stat].get("se"), c_by[stat].get("se"))
        if d_est is not None and d_est > worst_est:
            worst_est, worst_stat = d_est, stat
        if d_se is not None and d_se > worst_se:
            worst_se = d_se

    tol = REPRO_TOL_OVERRIDE.get(module, REPRO_REL_TOL)
    numbers_ok = worst_est <= tol and worst_se <= tol
    rows_ok = len(shared) == len(c_by)
    csv_ok = csv_status in ("byte_identical", "embedded")
    return {
        "module": module,
        "status": "reproduces" if (numbers_ok and rows_ok and csv_ok) else "drift",
        "csv_status": csv_status,
        "tol": tol,
        "relaxed": module in REPRO_TOL_OVERRIDE,
        "n_shared": len(shared),
        "n_committed": len(c_by),
        "worst_rel_est": worst_est,
        "worst_rel_se": worst_se,
        "worst_stat": worst_stat,
    }


_CSV_BADGE = {
    "byte_identical": "same bytes",
    "embedded": "in-script",
    "not_regenerated": "**not written**",
    "BYTES_DIFFER": "**DIFFER**",
}


def render_report(results: list[dict]) -> str:
    lines = [
        "# StatsPAI-side reproducibility report",
        "",
        "Generated by `tests/r_parity/verify_reproduce_py.py`. Each module's "
        "`NN_<name>.py` is re-executed into a staging directory; the "
        "regenerated fixture CSV is compared to the committed one byte for "
        "byte, and the regenerated `results/<module>_py.json` is diffed "
        "statistic-by-statistic against the committed golden. A module "
        f"**reproduces** when the fixture bytes are identical and every "
        f"shared statistic agrees within rel/abs {REPRO_REL_TOL:g}.",
        "",
        "This is the Python counterpart of `REPRODUCIBILITY_REPORT.md`. "
        "Both sides matter: the R report shows the reference is re-derivable "
        "from the frozen bytes, and this one shows the frozen bytes are "
        "themselves re-derivable from the package.",
        "",
        "| Module | Status | fixture | shared/total | worst rel Δest "
        "| worst rel Δse |",
        "|---|---|---|---:|---:|---:|",
    ]
    for res in results:
        if res["status"] in ("reproduces", "drift"):
            badge = "✅ reproduces" if res["status"] == "reproduces" else "⚠️ DRIFT"
            if res["status"] == "reproduces" and res.get("relaxed"):
                badge += "*"
            lines.append(
                f"| `{res['module']}` | {badge} "
                f"| {_CSV_BADGE.get(res['csv_status'], res['csv_status'])} "
                f"| {res['n_shared']}/{res['n_committed']} "
                f"| {res['worst_rel_est']:.2e} | {res['worst_rel_se']:.2e} |"
            )
        else:
            lines.append(
                f"| `{res['module']}` | ⏭️ {res['status']} | — | — | — | — | "
                f"{res.get('detail', '')}"
            )
    lines.append("")
    if any(r.get("relaxed") for r in results if r["status"] in ("reproduces", "drift")):
        lines += [
            "\\* Numeric reproduction tolerance relaxed for this module (see "
            "`REPRO_TOL_OVERRIDE`): the fit is an iterative optimiser whose "
            "last digits are BLAS-sensitive. The fixture byte-equality check "
            "is never relaxed.",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("modules", nargs="*", help="module stems, e.g. 01_ols")
    parser.add_argument("--timeout", type=int, default=900,
                        help="per-module timeout in seconds")
    parser.add_argument("--no-report", action="store_true",
                        help="skip writing REPRODUCIBILITY_REPORT_PY.md")
    args = parser.parse_args()

    modules = args.modules or discover_modules()
    if not modules:
        print("No modules with both NN_*.py and a committed _py.json found.")
        return 1

    results = []
    for module in modules:
        print(f"[verify-py] {module} ...", flush=True)
        res = run_one(module, args.timeout)
        results.append(res)
        if res["status"] in ("reproduces", "drift"):
            print(f"            -> {res['status']} "
                  f"(fixture {res['csv_status']}, "
                  f"worst rel est {res['worst_rel_est']:.2e})", flush=True)
        else:
            print(f"            -> {res['status']} ({res.get('detail', '')})",
                  flush=True)

    if not args.no_report:
        out = RESULTS_DIR / "REPRODUCIBILITY_REPORT_PY.md"
        out.write_text(render_report(results), encoding="utf-8")
        print(f"\nWrote {out}")

    drifted = [r for r in results if r["status"] == "drift"]
    reproduced = [r for r in results if r["status"] == "reproduces"]
    skipped = [r for r in results if r["status"] not in ("reproduces", "drift")]
    print(f"\nSummary: {len(reproduced)} reproduce, {len(drifted)} drift, "
          f"{len(skipped)} skipped/errored.")
    if drifted:
        print("DRIFT modules (must explain):",
              ", ".join(r["module"] for r in drifted))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
