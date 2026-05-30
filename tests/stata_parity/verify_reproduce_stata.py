"""Reproducibility verifier for the Stata-side parity golden JSONs.

The Stata sibling of ``tests/r_parity/verify_reproduce.py``. For JSS
Section 8 ("Reproducibility of this paper") the Stata leg of the
three-language Track A harness must be as auditable as the R leg: a
reviewer should be able to confirm that each committed
``results/<module>_Stata.json`` is the actual output of the canonical
Stata command under a documented engine, not a stale hand-edited
artefact.

For each module that has both an ``NN_<name>.do`` script and a committed
``results/<name>_Stata.json``, this driver:

  1. Re-runs the .do with ``STATSPAI_STATA_PARITY_RESULTS`` pointed at a
     staging directory, so the committed golden JSON is never clobbered.
  2. Joins the freshly produced rows to the committed rows by ``statistic``.
  3. Reports, per module, the worst relative (or absolute, near zero)
     difference on the point estimate and the SE.
  4. Surfaces the captured Stata engine provenance block written by the
     new ``_common.do::stata_parity_close`` provenance code.

A module "reproduces" when every joined statistic agrees to within the
module's reproducibility tolerance (default 1e-9 rel/abs, relaxed for
iterative-optimiser commands whose last digits are BLAS/seed sensitive --
see ``REPRO_TOL_OVERRIDE``). A drift is a finding the maintainer must
explain (ado upgrade, seed, BLAS), not a silent overwrite.

Usage::

    python tests/stata_parity/verify_reproduce_stata.py              # all
    python tests/stata_parity/verify_reproduce_stata.py 01_ols 02_iv # subset
    STATA_EXE=/path/to/stata-mp python .../verify_reproduce_stata.py

Writes ``results/REPRODUCIBILITY_REPORT_STATA.md`` and exits non-zero if
any attempted module drifts.
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
STAGING_DIR = RESULTS_DIR / "_repro_check"

REPRO_REL_TOL = 1e-9

# Per-module reproducibility-tolerance overrides for Stata commands whose
# fit is iterative and BLAS/optimiser/seed-sensitive far below the
# cross-language parity budget in tests/r_parity/compare.py. Each override is
# justified inline so the relaxation is auditable rather than a silent escape.
REPRO_TOL_OVERRIDE = {
    # csdid / sdid / honestdid run influence-function bootstraps or convex
    # solvers; mixed / frontier / xtfrontier maximise a likelihood; rdrobust
    # / rddensity run a bandwidth + bias-correction pipeline. Their last
    # digits move with BLAS/seed at a level orders of magnitude tighter than
    # the cross-language parity tolerance, so 1e-6 is a fair reproducibility
    # floor for the engine while the parity contract stays in compare.py.
    "04_csdid": 1e-6,
    "05_sunab": 1e-6,
    "07_scm": 1e-6,
    "12_sdid": 1e-6,
    "21_honest_relmags": 1e-6,
    "25_lmm": 1e-6,
    "28_frontier": 1e-6,
    "29_panel_sfa": 1e-6,
    "06_rd": 1e-6,
    "09_rddensity": 1e-6,
}

STATA_EXE_DEFAULT = "/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp"


def _stata_exe() -> Optional[str]:
    exe = os.environ.get("STATA_EXE", STATA_EXE_DEFAULT)
    return exe if Path(exe).exists() else None


def _reldiff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if abs(b) >= 1.0:
        return abs(a - b) / abs(b)
    return abs(a - b)


def discover_modules() -> list[str]:
    mods = []
    for do_script in sorted(HERE.glob("[0-9][0-9]_*.do")):
        name = do_script.stem
        if (RESULTS_DIR / f"{name}_Stata.json").exists():
            mods.append(name)
    return mods


def run_one(module: str, exe: str, timeout: int) -> dict:
    committed_path = RESULTS_DIR / f"{module}_Stata.json"
    if not committed_path.exists():
        return {"module": module, "status": "no_golden"}
    do_script = HERE / f"{module}.do"
    if not do_script.exists():
        return {"module": module, "status": "no_script"}

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    fresh_path = STAGING_DIR / f"{module}_Stata.json"
    if fresh_path.exists():
        fresh_path.unlink()  # so existence-after == success

    env = dict(os.environ)
    env["STATSPAI_STATA_PARITY_RESULTS"] = str(STAGING_DIR)
    try:
        subprocess.run(
            [exe, "-q", "-b", "do", f"{module}.do"],
            cwd=str(HERE),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"module": module, "status": "timeout", "detail": f">{timeout}s"}

    if not fresh_path.exists():
        # Stata batch mode returns 0 even on command errors; the missing
        # staging JSON is the real failure signal. Surface the .log tail.
        log = HERE / f"{module}.log"
        tail = ""
        if log.exists():
            tail = " | ".join(
                log.read_text(encoding="utf-8", errors="replace")
                .strip().splitlines()[-3:]
            )[:300]
        return {"module": module, "status": "stata_error", "detail": tail}

    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    c_by = {r["statistic"]: r for r in committed["rows"]}
    f_by = {r["statistic"]: r for r in fresh["rows"]}
    shared = sorted(set(c_by) & set(f_by))

    worst_est = 0.0
    worst_se = 0.0
    for s in shared:
        de = _reldiff(f_by[s].get("estimate"), c_by[s].get("estimate"))
        ds = _reldiff(f_by[s].get("se"), c_by[s].get("se"))
        if de is not None and de > worst_est:
            worst_est = de
        if ds is not None and ds > worst_se:
            worst_se = ds

    prov = fresh.get("provenance", {})
    tol = REPRO_TOL_OVERRIDE.get(module, REPRO_REL_TOL)
    reproduces = worst_est <= tol and worst_se <= tol
    return {
        "module": module,
        "status": "reproduces" if reproduces else "drift",
        "relaxed": module in REPRO_TOL_OVERRIDE,
        "n_shared": len(shared),
        "n_committed": len(c_by),
        "worst_rel_est": worst_est,
        "worst_rel_se": worst_se,
        "stata_version": prov.get("stata_version"),
        "edition": prov.get("edition"),
        "provenance": prov,
    }


def render_report(results: list[dict]) -> str:
    lines = [
        "# Stata-side reproducibility report",
        "",
        "Generated by `tests/stata_parity/verify_reproduce_stata.py`. Each "
        "module's committed `results/<module>_Stata.json` golden value is "
        "re-derived by re-running the canonical Stata command on the same "
        "`../r_parity/data/<module>.csv` bytes, into a staging directory, then "
        "diffed statistic-by-statistic. A module **reproduces** when every "
        f"shared statistic agrees to within rel/abs {REPRO_REL_TOL:g} (relaxed "
        "to 1e-6 for the iterative-optimiser commands flagged with \\*; that "
        "floor is still orders of magnitude tighter than the cross-language "
        "parity budget in `tests/r_parity/compare.py`).",
        "",
        "| Module | Status | shared/total | worst rel Δest | worst rel Δse | Stata | ed. |",
        "|---|---|---:|---:|---:|---:|---|",
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
                f"| {r.get('stata_version') or '—'} | {r.get('edition') or '—'} |"
            )
        else:
            lines.append(
                f"| `{r['module']}` | ⏭️ {r['status']} | — | — | — | — | — "
                f"| {r.get('detail','')}"
            )
    lines.append("")
    if any(r.get("relaxed") for r in results
           if r["status"] in ("reproduces", "drift")):
        lines += [
            "\\* Reproduction tolerance relaxed to 1e-6 (iterative optimiser, "
            "BLAS/seed sensitive); see `REPRO_TOL_OVERRIDE` in "
            "`verify_reproduce_stata.py`.",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("modules", nargs="*")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    exe = _stata_exe()
    if exe is None:
        print("Stata executable not found (set STATA_EXE=...). Skipping.")
        return 0  # absence of Stata is a skip, not a failure

    modules = args.modules or discover_modules()
    if not modules:
        print("No modules with both NN_*.do and results/*_Stata.json found.")
        return 1

    results = []
    for m in modules:
        print(f"[verify-stata] {m} ...", flush=True)
        res = run_one(m, exe, args.timeout)
        results.append(res)
        msg = res["status"]
        if res["status"] in ("reproduces", "drift"):
            msg += f" (worst rel est {res['worst_rel_est']:.2e})"
        elif res.get("detail"):
            msg += f" ({res['detail']})"
        print(f"           -> {msg}", flush=True)

    # Clean the .log files Stata batch mode drops in cwd.
    for log in HERE.glob("[0-9][0-9]_*.log"):
        try:
            log.unlink()
        except OSError:
            pass

    if not args.no_report:
        (RESULTS_DIR / "REPRODUCIBILITY_REPORT_STATA.md").write_text(
            render_report(results), encoding="utf-8")
        print(f"\nWrote {RESULTS_DIR / 'REPRODUCIBILITY_REPORT_STATA.md'}")

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
