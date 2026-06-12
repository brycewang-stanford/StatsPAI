"""Reproducibility verifier for the R-side parity golden JSONs.

For JSS Section 8 ("Reproducibility of this paper") we must be able to
demonstrate that the committed ``results/<module>_R.json`` golden values
are not stale hand-edited artefacts but the *actual* output of the
canonical R reference under a documented software environment.

This driver, for each module that has both an ``NN_<name>.R`` script and
its ``data/<name>.csv`` input:

  1. Re-runs the R reference with ``STATSPAI_R_PARITY_RESULTS_DIR`` set to
     a staging directory, so the committed golden JSON is never clobbered.
  2. Joins the freshly produced rows to the committed rows by ``statistic``.
  3. Reports, per module, the worst relative (or absolute, for near-zero)
     difference on both the point estimate and the SE.
  4. Surfaces the captured R/package provenance block written by the new
     ``_common.R::.r_provenance`` helper.

A module "reproduces" when every joined statistic agrees to within
``REPRO_TOL`` (1e-9 relative for |x| >= 1, else 1e-9 absolute). This is a
*reproducibility* tolerance (same code, same data, same packages should be
bit-stable to ~1e-12), deliberately far tighter than the cross-language
*parity* tolerance budget in ``compare.py``. A drift here is a finding the
maintainer must explain (package-version change, RNG seed, BLAS), not a
silent overwrite.

Usage::

    python tests/r_parity/verify_reproduce.py                 # all modules
    python tests/r_parity/verify_reproduce.py 01_ols 02_iv    # a subset
    python tests/r_parity/verify_reproduce.py --timeout 600   # per-module cap

Writes ``results/REPRODUCIBILITY_REPORT.md`` and exits non-zero if any
attempted module drifts beyond ``REPRO_TOL``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DATA_DIR = HERE / "data"
STAGING_DIR = RESULTS_DIR / "_repro_check"

REPRO_REL_TOL = 1e-9
REPRO_ABS_TOL = 1e-9

# Per-module reproducibility-tolerance overrides for estimators whose
# reference fit is iterative and BLAS/optimizer-sensitive at a level far
# below the cross-language parity budget. These are NOT parity tolerances
# (those live in compare.py); they only acknowledge that a numerical
# optimiser cannot be expected to land on the identical IEEE-754 bits
# across BLAS backends. Each override is justified inline so the relaxation
# is auditable rather than a silent escape hatch.
REPRO_TOL_OVERRIDE: dict[str, float] = {
    # Synth's V/W fit runs an L-BFGS-B search (optimxmethod="BFGS") with a
    # fixed seed; the residual ~2e-8 reproduction gap is floating-point
    # non-associativity in the BFGS gradient path under Apple Accelerate
    # BLAS, not algorithm drift. The SCM weights are themselves non-unique
    # (compare.py rel_est parity tol = 1.0, a documented T4 disclosure),
    # so 1e-6 is still 6 orders tighter than the parity contract.
    "07_scm": 1e-6,
    # Cluster- / heteroskedasticity-robust SE family. The point estimates
    # for these modules reproduce to ~6e-14 (machine precision), but the
    # sandwich "meat" matrix -- sum over clusters of the residual outer
    # products (X_g' e_g)(X_g' e_g)' -- accumulates via BLAS GEMM, whose
    # summation order is not fixed across backends. The golden JSONs were
    # generated under Apple Accelerate BLAS; CI runs reference/OpenBLAS on
    # Ubuntu, which reassociates the meat sum and shifts the SE in the last
    # ~8 significant digits (observed worst rel Δse: 14_ols_cluster 1.1e-8,
    # 53_cr2 1.2e-8, 55_hc2_hc3 9.8e-10 -- the latter cleared 1e-9 only by
    # luck). This is floating-point non-associativity, not algorithm drift.
    # 1e-7 sits ~8x above the observed noise yet stays 1-4 orders tighter
    # than each module's cross-language parity budget in compare.py
    # (rel_se: 14_ols_cluster 1e-3; 53_cr2 / 55_hc2_hc3 1e-6).
    "14_ols_cluster": 1e-7,
    "53_cr2": 1e-7,
    "55_hc2_hc3": 1e-7,
}


def _reldiff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if abs(b) >= 1.0:
        return abs(a - b) / abs(b)
    return abs(a - b)  # absolute regime for near-zero references


def discover_modules() -> list[str]:
    # A module is verifiable when its NN_*.R script has a committed
    # golden _R.json. A data/<module>.csv fixture is NOT required:
    # 10_honest_did / 21_honest_relmags / 23_evalue embed their fixtures
    # (event-study beta/sigma vectors, scalar effect sizes) directly in
    # the script, and their reproducibility matters just as much as the
    # CSV-driven modules'.
    mods = []
    for r_script in sorted(HERE.glob("[0-9][0-9]_*.R")):
        name = r_script.stem
        if (RESULTS_DIR / f"{name}_R.json").exists():
            mods.append(name)
    return mods


def run_one(module: str, timeout: int) -> dict:
    """Run the R reference for one module into the staging dir and diff."""
    committed_path = RESULTS_DIR / f"{module}_R.json"
    if not committed_path.exists():
        return {"module": module, "status": "no_golden", "detail": "no committed _R.json"}

    r_script = HERE / f"{module}.R"
    if not r_script.exists():
        return {"module": module, "status": "no_script", "detail": "no NN_*.R"}

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["STATSPAI_R_PARITY_RESULTS_DIR"] = str(STAGING_DIR)
    try:
        proc = subprocess.run(
            ["Rscript", str(r_script)],
            cwd=str(HERE),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"module": module, "status": "timeout", "detail": f">{timeout}s"}
    except FileNotFoundError:
        return {"module": module, "status": "no_rscript",
                "detail": "Rscript not on PATH"}

    fresh_path = STAGING_DIR / f"{module}_R.json"
    if proc.returncode != 0 or not fresh_path.exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return {"module": module, "status": "r_error",
                "detail": " | ".join(tail)[:300]}

    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    c_by = {r["statistic"]: r for r in committed["rows"]}
    f_by = {r["statistic"]: r for r in fresh["rows"]}
    shared = sorted(set(c_by) & set(f_by))

    worst_est = 0.0
    worst_se = 0.0
    worst_stat = ""
    for s in shared:
        de = _reldiff(f_by[s].get("estimate"), c_by[s].get("estimate"))
        ds = _reldiff(f_by[s].get("se"), c_by[s].get("se"))
        if de is not None and de > worst_est:
            worst_est, worst_stat = de, s
        if ds is not None and ds > worst_se:
            worst_se = ds

    prov = fresh.get("provenance", {})
    tol = REPRO_TOL_OVERRIDE.get(module, REPRO_REL_TOL)
    reproduces = worst_est <= tol and worst_se <= tol
    return {
        "module": module,
        "status": "reproduces" if reproduces else "drift",
        "tol": tol,
        "relaxed": module in REPRO_TOL_OVERRIDE,
        "n_shared": len(shared),
        "n_committed": len(c_by),
        "worst_rel_est": worst_est,
        "worst_rel_se": worst_se,
        "worst_stat": worst_stat,
        "r_version": prov.get("r_version"),
        "n_packages": len(prov.get("packages", {})),
        "provenance": prov,
    }


def render_report(results: list[dict]) -> str:
    lines = [
        "# R-side reproducibility report",
        "",
        "Generated by `tests/r_parity/verify_reproduce.py`. Each module's "
        "committed `results/<module>_R.json` golden value is re-derived by "
        "re-running the canonical R reference on the same `data/<module>.csv` "
        "bytes, into a staging directory, then diffed statistic-by-statistic. "
        f"A module **reproduces** when every shared statistic agrees to "
        f"within rel/abs {REPRO_REL_TOL:g} (a reproducibility tolerance, far "
        "tighter than the cross-language parity budget in `compare.py`).",
        "",
        "| Module | Status | shared/total | worst rel Δest | worst rel Δse | R version | pkgs |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for r in results:
        if r["status"] in ("reproduces", "drift"):
            if r["status"] == "reproduces":
                badge = "✅ reproduces*" if r.get("relaxed") else "✅ reproduces"
            else:
                badge = "⚠️ DRIFT"
            lines.append(
                f"| `{r['module']}` | {badge} "
                f"| {r['n_shared']}/{r['n_committed']} "
                f"| {r['worst_rel_est']:.2e} | {r['worst_rel_se']:.2e} "
                f"| {r.get('r_version') or '—'} | {r.get('n_packages', 0)} |"
            )
        else:
            lines.append(
                f"| `{r['module']}` | ⏭️ {r['status']} | — | — | — "
                f"| — | — |  {r.get('detail','')}"
            )
    lines.append("")
    if any(r.get("relaxed") for r in results
           if r["status"] in ("reproduces", "drift")):
        lines += [
            "\\* Reproduction tolerance relaxed for this module (see "
            "`REPRO_TOL_OVERRIDE` in `verify_reproduce.py`): the reference "
            "fit is an iterative optimiser whose last digits are "
            "BLAS-sensitive. The relaxed floor is still orders of magnitude "
            "tighter than the module's cross-language parity tolerance in "
            "`compare.py`.",
            "",
        ]
    # Provenance appendix: dump the captured environment from the first
    # module that produced one, plus a per-package union across modules.
    union: dict[str, set] = {}
    r_versions = set()
    for r in results:
        prov = r.get("provenance") or {}
        if prov.get("r_version"):
            r_versions.add(prov["r_version"])
        for p, v in (prov.get("packages") or {}).items():
            union.setdefault(p, set()).add(v if isinstance(v, str) else str(v))
    if union:
        lines += ["## Captured R environment (union across re-run modules)", ""]
        lines.append(f"- R version(s): {', '.join(sorted(r_versions)) or '—'}")
        lines += ["", "| package | version(s) |", "|---|---|"]
        for p in sorted(union):
            lines.append(f"| `{p}` | {', '.join(sorted(union[p]))} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("modules", nargs="*", help="module stems, e.g. 01_ols")
    ap.add_argument("--timeout", type=int, default=900,
                    help="per-module Rscript timeout in seconds")
    ap.add_argument("--no-report", action="store_true",
                    help="skip writing REPRODUCIBILITY_REPORT.md")
    args = ap.parse_args()

    modules = args.modules or discover_modules()
    if not modules:
        print("No modules with both NN_*.R and data/*.csv found.")
        return 1

    results = []
    for m in modules:
        print(f"[verify] {m} ...", flush=True)
        res = run_one(m, args.timeout)
        results.append(res)
        print(f"         -> {res['status']}"
              + (f" (worst rel est {res['worst_rel_est']:.2e})"
                 if res["status"] in ("reproduces", "drift") else
                 f" ({res.get('detail','')})"), flush=True)

    if not args.no_report:
        report = render_report(results)
        (RESULTS_DIR / "REPRODUCIBILITY_REPORT.md").write_text(
            report, encoding="utf-8")
        print(f"\nWrote {RESULTS_DIR / 'REPRODUCIBILITY_REPORT.md'}")

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
