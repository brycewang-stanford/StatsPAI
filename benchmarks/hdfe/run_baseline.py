"""HDFE Poisson baseline driver.

Orchestrates the Phase 0 baseline:

1. Materialise the requested datasets (``datasets.py``).
2. For each (dataset, backend) pair, run a subprocess (``run_python.py``
   or ``Rscript run_r.R``) and capture its JSON output.
3. Aggregate everything into ``results/baseline.json``.
4. Render a human-readable ``BASELINE.md`` with timing and coefficient
   cross-backend diffs.

Stata results, if present in ``results/<dataset>_stata.json``, are
ingested but never produced by this driver (Stata is not assumed to be
installed; see ``run_stata.do`` for the manual workflow).

Examples
--------
::

    # Default: small + medium, all available backends.
    python3 run_baseline.py

    # Just regenerate the report from cached results/*.json.
    python3 run_baseline.py --report-only

    # Add the heavy 1e7-row config (slow!).
    python3 run_baseline.py --datasets small medium large
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DATA_DIR = HERE / "data"


# ---------------------------------------------------------------------------
# Subprocess runners
# ---------------------------------------------------------------------------

PY_BACKENDS = ("pyfixest", "statspai")


def _python_repeats(dataset: str) -> Dict[str, int]:
    """Per-dataset benchmarking depth."""
    return {
        "small":  dict(warmup=1, repeats=3),
        "medium": dict(warmup=1, repeats=2),
        "large":  dict(warmup=0, repeats=1),
    }[dataset]


def run_python_backend(dataset: str, backend: str) -> Dict[str, Any]:
    """Invoke ``run_python.py`` in a fresh subprocess; return parsed JSON."""
    cfg = _python_repeats(dataset)
    cmd = [
        sys.executable, str(HERE / "run_python.py"),
        "--dataset", dataset,
        "--backend", backend,
        "--warmup", str(cfg["warmup"]),
        "--repeats", str(cfg["repeats"]),
    ]
    print(f"[driver] {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stderr.strip():
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0 and not proc.stdout.strip():
        return {
            "dataset": dataset,
            "backend": backend,
            "error": {
                "type": "SubprocessError",
                "message": f"exit={proc.returncode}",
            },
        }
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        return {
            "dataset": dataset,
            "backend": backend,
            "error": {"type": "JSONDecodeError", "message": str(exc), "stdout": proc.stdout},
        }


def run_r_backend(dataset: str) -> Optional[Dict[str, Any]]:
    """Invoke ``Rscript run_r.R``; return parsed JSON or None if R missing."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        return None
    cfg = _python_repeats(dataset)
    cmd = [
        rscript, str(HERE / "run_r.R"),
        "--dataset", dataset,
        "--warmup", str(cfg["warmup"]),
        "--repeats", str(cfg["repeats"]),
    ]
    print(f"[driver] {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stderr.strip():
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0 and not proc.stdout.strip():
        return {
            "dataset": dataset,
            "backend": "fixest",
            "error": {
                "type": "SubprocessError",
                "message": f"exit={proc.returncode}",
            },
        }
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        return {
            "dataset": dataset,
            "backend": "fixest",
            "error": {"type": "JSONDecodeError", "message": str(exc), "stdout": proc.stdout},
        }


def load_stata_result(dataset: str) -> Optional[Dict[str, Any]]:
    """Pick up a manually produced ``results/<dataset>_ppmlhdfe.json`` if it exists."""
    p = RESULTS_DIR / f"{dataset}_ppmlhdfe.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        return {"dataset": dataset, "backend": "ppmlhdfe", "error": {"type": "JSONDecodeError", "message": str(exc)}}
    # Stata leaves wall_min/mean/max blank — fill in.
    wall = data.get("wall") or {}
    runs = wall.get("wall_runs") or []
    runs = [r for r in runs if isinstance(r, (int, float))]
    if runs and (wall.get("wall_min") in (None, "", float("nan")) or
                 not isinstance(wall.get("wall_min"), (int, float))):
        wall["wall_min"] = float(min(runs))
        wall["wall_mean"] = float(sum(runs) / len(runs))
        wall["wall_max"] = float(max(runs))
        data["wall"] = wall
    return data


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------

def aggregate(datasets: List[str], skip_python: bool, skip_r: bool) -> Dict[str, Any]:
    RESULTS_DIR.mkdir(exist_ok=True)
    all_runs: List[Dict[str, Any]] = []

    for ds in datasets:
        if not skip_python:
            for backend in PY_BACKENDS:
                res = run_python_backend(ds, backend)
                _persist(res, ds, backend)
                all_runs.append(res)

        if not skip_r:
            res_r = run_r_backend(ds)
            if res_r is not None:
                _persist(res_r, ds, "fixest")
                all_runs.append(res_r)

        res_stata = load_stata_result(ds)
        if res_stata is not None:
            all_runs.append(res_stata)

    return {"runs": all_runs}


def _persist(result: Dict[str, Any], dataset: str, backend: str) -> None:
    path = RESULTS_DIR / f"{dataset}_{backend}.json"
    path.write_text(json.dumps(result, indent=2, default=str))


def load_cached_runs(datasets: List[str]) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    if not RESULTS_DIR.exists():
        return {"runs": runs}
    for ds in datasets:
        for p in sorted(RESULTS_DIR.glob(f"{ds}_*.json")):
            try:
                runs.append(json.loads(p.read_text()))
            except json.JSONDecodeError:
                continue
    return {"runs": runs}


def load_dataset_meta(name: str) -> Optional[Dict[str, Any]]:
    p = DATA_DIR / f"{name}.meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# BASELINE.md renderer
# ---------------------------------------------------------------------------

def _fmt_seconds(s: Optional[float]) -> str:
    if s is None:
        return "—"
    if s < 1.0:
        return f"{s * 1000:.0f}ms"
    if s < 60.0:
        return f"{s:.2f}s"
    return f"{s / 60:.1f}min"


def _fmt_mb(b: Optional[float]) -> str:
    if b is None:
        return "—"
    return f"{b / 1e6:.0f}MB"


def render_report(agg: Dict[str, Any], datasets: List[str]) -> str:
    lines: List[str] = []
    lines.append("# HDFE Poisson Baseline (Phase 0)")
    lines.append("")
    lines.append(
        "Cross-backend baseline for `fepois` on synthetic two-way HDFE Poisson "
        "data. Generated by `benchmarks/hdfe/run_baseline.py`. Numbers here "
        "freeze the **starting line** that StatsPAI's Rust HDFE backend "
        "(Phase 1+) must beat."
    )
    lines.append("")
    lines.append(f"_Generated_: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append("")

    # --- dataset metadata ---
    lines.append("## Datasets")
    lines.append("")
    lines.append("| name | n | fe1 cardinality | fe2 cardinality | y mean | y zero share | seed |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in datasets:
        meta = load_dataset_meta(name)
        if meta is None:
            lines.append(f"| {name} | (not generated) | — | — | — | — | — |")
            continue
        lines.append(
            f"| {name} | {meta['n']:,} | {meta['fe1_cardinality']:,} | "
            f"{meta['fe2_cardinality']:,} | {meta['y_mean']:.3f} | "
            f"{meta['y_zero_share']:.3f} | {meta['seed']} |"
        )
    lines.append("")

    # Index runs by (dataset, backend)
    runs_by: Dict[tuple, Dict[str, Any]] = {}
    for r in agg["runs"]:
        key = (r.get("dataset"), r.get("backend"))
        runs_by[key] = r

    # --- timing table ---
    lines.append("## Wall-clock (mean of timed runs)")
    lines.append("")
    backends_order = ["statspai", "pyfixest", "fixest", "ppmlhdfe"]
    header = "| dataset | " + " | ".join(backends_order) + " |"
    sep = "|---|" + "|".join(["---:"] * len(backends_order)) + "|"
    lines.append(header)
    lines.append(sep)
    for ds in datasets:
        row = [ds]
        for b in backends_order:
            r = runs_by.get((ds, b))
            if r is None or r.get("error") is not None:
                row.append("—")
                continue
            wall = r.get("wall") or {}
            row.append(_fmt_seconds(wall.get("wall_mean")))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # --- peak RSS table (Python only — R/Stata report differently) ---
    lines.append("## Peak RSS (Python subprocesses; sampled)")
    lines.append("")
    py_backends = ["statspai", "pyfixest"]
    lines.append("| dataset | " + " | ".join(py_backends) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(py_backends)) + "|")
    for ds in datasets:
        row = [ds]
        for b in py_backends:
            r = runs_by.get((ds, b))
            if r is None or r.get("error") is not None:
                row.append("—")
                continue
            row.append(_fmt_mb(r.get("peak_rss_bytes")))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # --- coefficient cross-backend diff ---
    lines.append("## Coefficient cross-backend diff (vs `fixest` if present, else `pyfixest`)")
    lines.append("")
    lines.append("Reports `max_abs_diff` of (β_x1, β_x2) against the reference backend.")
    lines.append("Acceptance threshold (Phase 0 sign-off): **≤ 1e-6** wherever fixest is the reference.")
    lines.append("")
    lines.append("| dataset | reference | backend | β_x1 | β_x2 | max_abs_diff |")
    lines.append("|---|---|---|---:|---:|---:|")

    for ds in datasets:
        ref_backend = None
        ref_coefs: Optional[Dict[str, float]] = None
        for cand in ("fixest", "pyfixest", "statspai"):
            r = runs_by.get((ds, cand))
            if r and not r.get("error") and r.get("coefs"):
                ref_backend = cand
                ref_coefs = r["coefs"]
                break
        if ref_coefs is None:
            lines.append(f"| {ds} | — | — | — | — | — |")
            continue
        # First, the reference itself
        lines.append(
            f"| {ds} | {ref_backend} | {ref_backend} | "
            f"{ref_coefs.get('x1', float('nan')):.8f} | "
            f"{ref_coefs.get('x2', float('nan')):.8f} | 0 |"
        )
        for b in [x for x in backends_order if x != ref_backend]:
            r = runs_by.get((ds, b))
            if r is None or r.get("error") or not r.get("coefs"):
                continue
            cf = r["coefs"]
            diffs = [
                abs(cf.get(k, float("nan")) - ref_coefs.get(k, float("nan")))
                for k in ("x1", "x2")
                if k in cf and k in ref_coefs
            ]
            mad = max(diffs) if diffs else float("nan")
            lines.append(
                f"| {ds} | {ref_backend} | {b} | "
                f"{cf.get('x1', float('nan')):.8f} | "
                f"{cf.get('x2', float('nan')):.8f} | {mad:.2e} |"
            )
    lines.append("")

    # --- errors ---
    errored = [r for r in agg["runs"] if r.get("error")]
    if errored:
        lines.append("## Errors")
        lines.append("")
        for r in errored:
            lines.append(
                f"- **{r.get('dataset')} / {r.get('backend')}**: "
                f"`{r['error'].get('type')}` — {r['error'].get('message')}"
            )
        lines.append("")

    # --- environment ---
    lines.append("## Environment")
    lines.append("")
    lines.append("Captured at run time so future regressions know what they're "
                 "compared against:")
    lines.append("")
    env = _capture_env()
    for k, v in env.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    return "\n".join(lines) + "\n"


def _capture_env() -> Dict[str, str]:
    out: Dict[str, str] = {}
    out["python"] = sys.version.split()[0]
    out["platform"] = sys.platform
    try:
        import numpy
        out["numpy"] = numpy.__version__
    except ImportError:
        pass
    try:
        import pandas
        out["pandas"] = pandas.__version__
    except ImportError:
        pass
    try:
        import pyfixest
        out["pyfixest"] = pyfixest.__version__
    except ImportError:
        pass
    try:
        import statspai
        out["statspai"] = statspai.__version__
    except ImportError:
        pass
    rscript = shutil.which("Rscript")
    if rscript:
        try:
            r_ver = subprocess.run(
                [rscript, "-e", "cat(paste(R.version$major, R.version$minor, sep='.'))"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            fixest_ver = subprocess.run(
                [rscript, "-e", "cat(as.character(packageVersion('fixest')))"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            out["R"] = r_ver
            out["R fixest"] = fixest_ver
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--datasets", nargs="+", default=["small", "medium"],
        help="Which datasets to run (default: small medium).",
    )
    ap.add_argument("--report-only", action="store_true",
                    help="Skip subprocess runs; rebuild BASELINE.md from results/*.json.")
    ap.add_argument("--skip-python", action="store_true")
    ap.add_argument("--skip-r", action="store_true")
    ap.add_argument("--force-data", action="store_true",
                    help="Regenerate dataset CSVs even if cached.")
    args = ap.parse_args()

    if not args.report_only:
        # Materialise data first (cached unless --force-data).
        from datasets import write_csv
        for name in args.datasets:
            write_csv(name, force=args.force_data)

        agg = aggregate(args.datasets, args.skip_python, args.skip_r)
    else:
        agg = load_cached_runs(args.datasets)

    # Persist machine-readable + human-readable outputs.
    (HERE / "baseline.json").write_text(json.dumps(agg, indent=2, default=str))
    report = render_report(agg, args.datasets)
    (HERE / "BASELINE.md").write_text(report)
    print(f"[driver] wrote {HERE / 'baseline.json'}", file=sys.stderr)
    print(f"[driver] wrote {HERE / 'BASELINE.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
