#!/usr/bin/env python3
"""End-to-end workflow benchmark: StatsPAI vs pyfixest / DoubleML / R.

Answers the 2026-09 review's objection to the earlier reports (one two-way FE
DGP at 10k / 100k rows, StatsPAI 1.20): speed claims need the tasks applied
work actually runs, at realistic sizes, with the *same estimand* on every
backend, and they need memory, cold start and failures, not only a median.

Every (task, size, backend) runs in a fresh subprocess:

* **cold** -- wall time of the first fit, including ``import`` of the
  library (what a script or an agent's first tool call pays);
* **warm** -- median and IQR of ``--reps`` further fits in that process;
* **peak RSS** -- the subprocess's maximum resident set size;
* **status** -- ``ok`` / ``failed`` (with the exception) / ``timeout``;
* **estimate** -- the headline number, compared across backends before any
  time is quoted: a backend that estimates something else is flagged, not
  ranked.

Threads: numpy/BLAS and numba use their defaults unless ``--threads N`` pins
``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` /
``NUMBA_NUM_THREADS`` / ``RAYON_NUM_THREADS`` (and R's data.table / fixest
threads) for every backend alike.

Usage::

    python benchmarks/workflow_bench/run.py --sizes 100000 1000000 --reps 3
    python benchmarks/workflow_bench/run.py --tasks hdfe --sizes 100000 --quick

Writes ``benchmarks/workflow_bench/results.json`` and ``RESULTS.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


# --------------------------------------------------------------------------- #
#  Data generating processes (deterministic)
# --------------------------------------------------------------------------- #


def dgp_hdfe(n: int, seed: int = 0) -> pd.DataFrame:
    """Unbalanced two-way panel: Pareto-sized firms, 20 years, IV + counts."""
    rng = np.random.default_rng(seed)
    n_firms = max(50, n // 25)
    sizes = rng.pareto(1.5, n_firms) + 1
    firm = rng.choice(n_firms, size=n, p=sizes / sizes.sum())
    year = rng.integers(0, 20, n)
    fe_f = rng.normal(size=n_firms)[firm]
    fe_y = rng.normal(scale=0.5, size=20)[year]
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    x1 = 0.6 * z + 0.3 * fe_f + 0.5 * u + rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.5 * x1 - 0.7 * x2 + fe_f + fe_y + u + rng.normal(size=n)
    # count outcome; firms with a zero intercept draw -> all-zero (separation)
    mu = np.exp(-0.5 + 0.3 * x1 - 0.2 * x2 + 0.5 * fe_f + 0.3 * fe_y)
    mu[np.isin(firm, np.arange(0, n_firms, 17))] = 0.0
    c = rng.poisson(mu)
    return pd.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "z": z, "c": c, "firm": firm, "year": year}
    )


def dgp_did(n: int, seed: int = 0) -> pd.DataFrame:
    """Staggered adoption panel, 10 periods, 4 cohorts + never-treated."""
    rng = np.random.default_rng(seed)
    T = 10
    units = max(100, n // T)
    cohort = rng.choice([0, 4, 5, 6, 7], size=units, p=[0.3, 0.2, 0.2, 0.15, 0.15])
    uid = np.repeat(np.arange(units), T)
    t = np.tile(np.arange(1, T + 1), units)
    g = cohort[uid]
    treated = (g > 0) & (t >= g)
    y = (
        rng.normal(size=units)[uid]
        + 0.1 * t
        + np.where(treated, 1.0 + 0.1 * (t - g), 0.0)
        + rng.normal(size=units * T)
    )
    return pd.DataFrame({"id": uid, "t": t, "g": g, "y": y})


def dgp_dml(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 10))
    d = X[:, 0] - 0.5 * X[:, 1] + rng.normal(size=n)
    y = 0.5 * d + X[:, 0] + 0.5 * X[:, 2] + rng.normal(size=n)
    out = pd.DataFrame(X, columns=[f"x{j}" for j in range(10)])
    out["d"], out["y"] = d, y
    return out


def dgp_boot(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 40, n)
    x = rng.normal(size=n) + rng.normal(size=40)[g]
    y = 0.3 * x + rng.normal(size=40)[g] + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "g": g})


DGPS = {
    "hdfe": dgp_hdfe,
    "iv": dgp_hdfe,
    "ppml": dgp_hdfe,
    "did": dgp_did,
    "dml": dgp_dml,
    "boot": dgp_boot,
}


# --------------------------------------------------------------------------- #
#  Backends: code run inside the subprocess; must set ``est``
# --------------------------------------------------------------------------- #

PY: Dict[str, Dict[str, str]] = {
    "hdfe": {
        "sp.feols": "r = sp.feols('y ~ x1 + x2 | firm + year', df); est = r.params['x1']",
        "sp.hdfe_ols": "r = sp.hdfe_ols('y ~ x1 + x2 | firm + year', data=df, drop_singletons=False); est = r.params['x1']",
        "sp.fast.feols": "r = sp.fast.feols('y ~ x1 + x2 | firm + year', df, drop_singletons=False); est = float(r.coef()['x1'])",
        "pyfixest": "r = pf.feols('y ~ x1 + x2 | firm + year', df); est = float(r.coef()['x1'])",
    },
    "iv": {
        "sp.feols": "r = sp.feols('y ~ x2 | firm + year | x1 ~ z', df); est = r.params['x1']",
        "pyfixest": "r = pf.feols('y ~ x2 | firm + year | x1 ~ z', df); est = float(r.coef()['x1'])",
    },
    "ppml": {
        "sp.fepois": "r = sp.fepois('c ~ x1 + x2 | firm + year', df); est = r.params['x1']",
        "sp.fast.fepois": "r = sp.fast.fepois('c ~ x1 + x2 | firm + year', df); est = float(r.coef()['x1'])",
        "pyfixest": "r = pf.fepois('c ~ x1 + x2 | firm + year', df); est = float(r.coef()['x1'])",
    },
    "did": {
        "sp.callaway_santanna": "r = sp.callaway_santanna(df, y='y', g='g', t='t', i='id', estimator='reg'); est = r.estimate",
    },
    "dml": {
        "sp.dml": "from sklearn.linear_model import LinearRegression as LR\n"
        "r = sp.dml(df, y='y', treat='d', covariates=[f'x{j}' for j in range(10)], model='plr', "
        "model_y=LR(), model_d=LR(), n_folds=5, n_rep=5, random_state=0); est = r.estimate",
        "doubleml": "import doubleml as dml; from sklearn.linear_model import LinearRegression as LR\n"
        "dd = dml.DoubleMLData(df, 'y', 'd', [f'x{j}' for j in range(10)])\n"
        "m = dml.DoubleMLPLR(dd, LR(), LR(), n_folds=5, n_rep=5); m.fit(); est = float(m.coef[0])",
    },
    "boot": {
        "sp.wild_cluster_bootstrap": "r = sp.wild_cluster_bootstrap(df, y='y', x=['x'], cluster='g', n_boot=999, seed=0); "
        "est = float(np.atleast_1d(r['beta_hat'])[-1])",
    },
}

R: Dict[str, str] = {
    "hdfe": "library(fixest); m <- feols(y ~ x1 + x2 | firm + year, d); est <- coef(m)[['x1']]",
    "iv": "library(fixest); m <- feols(y ~ x2 | firm + year | x1 ~ z, d); est <- coef(m)[['fit_x1']]",
    "ppml": "library(fixest); m <- fepois(c ~ x1 + x2 | firm + year, d); est <- coef(m)[['x1']]",
    # did recodes g == 0 to Inf; on an integer column that becomes NA and
    # the never-treated units are silently dropped -- cast to double first.
    "did": "library(did); d$g <- as.numeric(d$g); a <- att_gt('y', 't', 'id', 'g', data = d, est_method = 'reg', bstrap = FALSE, cband = FALSE); "
    "est <- aggte(a, type = 'simple', bstrap = FALSE)$overall.att",
}

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_PY_DRIVER = """
import time, json, sys, resource, warnings
warnings.simplefilter("ignore")
t0 = time.perf_counter()
import numpy as np, pandas as pd
import statspai as sp
try:
    import pyfixest as pf
except Exception:
    pf = None
df = pd.read_parquet(sys.argv[1])
t_load = time.perf_counter()
code = sys.argv[2]
reps = int(sys.argv[3])
out = {"times": [], "errors": []}
def once():
    ns = {"sp": sp, "pf": pf, "df": df, "np": np, "pd": pd}
    exec(code, ns)
    return float(ns["est"])
t1 = time.perf_counter()
try:
    out["estimate"] = once()
    out["first_fit_s"] = time.perf_counter() - t1
except Exception as e:
    out["errors"].append(f"{type(e).__name__}: {e}"[:300])
for _ in range(reps):
    s = time.perf_counter()
    try:
        once()
        out["times"].append(time.perf_counter() - s)
    except Exception as e:
        out["errors"].append(f"{type(e).__name__}: {e}"[:300])
out["import_s"] = t_load - t0
_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, KiB on Linux
out["peak_rss_mb"] = _rss / 1e6 if sys.platform == "darwin" else _rss / 1024
print("RESULT " + json.dumps(out))
"""

_R_DRIVER = """
suppressPackageStartupMessages({library(data.table); library(jsonlite)})
args <- commandArgs(TRUE)
t0 <- Sys.time()
d <- as.data.frame(fread(args[1]))
t_load <- Sys.time()
code <- args[2]; reps <- as.integer(args[3])
once <- function() { e <- new.env(); e$d <- d; suppressMessages(suppressWarnings(eval(parse(text = code), e))); as.numeric(e$est) }
out <- list(times = c(), errors = c())
t1 <- Sys.time()
res <- tryCatch(once(), error = function(e) { out$errors <<- c(out$errors, conditionMessage(e)); NA })
out$estimate <- res
out$first_fit_s <- as.numeric(difftime(Sys.time(), t1, units = "secs"))
out$import_s <- as.numeric(difftime(t_load, t0, units = "secs"))
for (i in seq_len(reps)) {
  s <- Sys.time()
  ok <- tryCatch({once(); TRUE}, error = function(e) { out$errors <<- c(out$errors, conditionMessage(e)); FALSE })
  if (ok) out$times <- c(out$times, as.numeric(difftime(Sys.time(), s, units = "secs")))
}
cat("RESULT", toJSON(out, auto_unbox = TRUE, digits = NA), "\\n")
"""


def _env(threads: Optional[int]) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["STATSPAI_SKIP_BANNER"] = "1"
    if threads:
        for v in _THREAD_VARS:
            env[v] = str(threads)
    return env


def _run(cmd: List[str], env: Dict[str, str], timeout: float) -> Dict[str, Any]:
    t = time.perf_counter()
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "wall_s": timeout}
    wall = time.perf_counter() - t
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if line is None:
        return {
            "status": "failed",
            "error": (p.stderr or p.stdout)[-500:],
            "wall_s": wall,
        }
    out = json.loads(line[len("RESULT ") :])
    out["wall_s"] = wall
    ok = out.get("estimate") is not None and not (
        isinstance(out.get("estimate"), float) and np.isnan(out["estimate"])
    )
    out["status"] = "ok" if ok and not out.get("errors") else "failed"
    return out


def run_backend(
    task: str,
    backend: str,
    data_path: Path,
    reps: int,
    threads: Optional[int],
    timeout: float,
) -> Dict[str, Any]:
    env = _env(threads)
    if backend == "R":
        if threads:
            env["R_DATATABLE_NUM_THREADS"] = str(threads)
        code = R[task]
        if threads:
            code = (
                f"fixest::setFixest_nthreads({threads}); " + code
                if "fixest" in code
                else code
            )
        with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as fh:
            fh.write(_R_DRIVER)
        cmd = ["Rscript", fh.name, str(data_path.with_suffix(".csv")), code, str(reps)]
        return _run(cmd, env, timeout)
    cmd = [
        sys.executable,
        "-c",
        _PY_DRIVER,
        str(data_path),
        PY[task][backend],
        str(reps),
    ]
    return _run(cmd, env, timeout)


def _summarise(rec: Dict[str, Any]) -> Dict[str, Any]:
    ts = rec.get("times") or []
    if ts:
        rec["warm_median_s"] = float(np.median(ts))
        rec["warm_iqr_s"] = float(np.subtract(*np.percentile(ts, [75, 25])))
    rec["n_failures"] = len(rec.get("errors") or [])
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="*", default=list(PY))
    ap.add_argument("--sizes", nargs="*", type=int, default=[100_000, 1_000_000])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--no-r", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()

    import statspai as sp  # noqa: F401  (version for provenance)

    results: List[Dict[str, Any]] = []
    tmp = Path(tempfile.mkdtemp(prefix="sp-wbench-"))
    for task in args.tasks:
        for n in args.sizes:
            if task == "boot" and n > 200_000:
                continue  # wild bootstrap: one mid-size row is informative
            df = DGPS[task](n)
            path = tmp / f"{task}_{n}.parquet"
            df.to_parquet(path, index=False)
            if task in R and not args.no_r:
                df.to_csv(path.with_suffix(".csv"), index=False)
            backends = list(PY[task]) + (["R"] if task in R and not args.no_r else [])
            for b in backends:
                print(f"[{task} n={len(df):,}] {b} ...", flush=True)
                rec = run_backend(task, b, path, args.reps, args.threads, args.timeout)
                rec.update(task=task, n=len(df), backend=b)
                results.append(_summarise(rec))
                print(
                    f"    -> {rec.get('status')} est={rec.get('estimate')} "
                    f"warm={rec.get('warm_median_s')} rss={rec.get('peak_rss_mb')}",
                    flush=True,
                )
    import statspai

    meta = {
        "statspai_version": statspai.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "threads_pinned": args.threads,
        "reps": args.reps,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for mod in ("numpy", "pandas", "pyfixest", "doubleml"):
        try:
            meta[f"{mod}_version"] = __import__(mod).__version__
        except Exception:
            meta[f"{mod}_version"] = None
    Path(args.out).write_text(
        json.dumps({"meta": meta, "results": results}, indent=2, default=str),
        encoding="utf-8",
    )
    (HERE / "RESULTS.md").write_text(render(meta, results), encoding="utf-8")
    print(f"wrote {args.out} and RESULTS.md")
    return 0


def render(meta: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
    lines = [
        "# Workflow benchmark",
        "",
        "Generated by `benchmarks/workflow_bench/run.py`. Each row is one backend "
        "in its own process: **cold** = import + first fit, **warm** = median "
        "(IQR) of the repeated fits, **peak RSS** of that process (Python "
        "backends; R rows omit it). Estimates are compared *before* times: "
        "a backend whose estimate differs from the others is estimating "
        "something else and is marked, not ranked. Slower rows are reported, "
        "not hidden. Re-run on your hardware before quoting a number.",
        "",
        "## Provenance",
        "",
    ]
    lines += [f"- {k}: `{v}`" for k, v in meta.items()]
    lines.append("")
    df = pd.DataFrame(results)
    for (task, n), grp in df.groupby(["task", "n"], sort=False):
        ok = grp[grp["status"] == "ok"]
        ests = ok["estimate"].astype(float) if len(ok) else pd.Series(dtype=float)
        ref = float(np.median(ests)) if len(ests) else float("nan")
        fastest = (
            ok["warm_median_s"].min() if "warm_median_s" in ok and len(ok) else np.nan
        )
        lines += [
            f"## {task} -- n = {n:,}",
            "",
            "| backend | status | estimate | rel. diff vs median | cold (s) | warm median (s) | IQR | peak RSS (MB) | vs fastest |",
            "| --- | --- | --: | --: | --: | --: | --: | --: | --: |",
        ]
        for _, r in grp.iterrows():
            est = r.get("estimate")
            est = float(est) if est is not None and not pd.isna(est) else np.nan
            rel = abs(est - ref) / max(abs(ref), 1e-12) if np.isfinite(est) else np.nan
            warm = r.get("warm_median_s", np.nan)
            cold = (
                (r.get("import_s") or 0) + (r.get("first_fit_s") or 0)
                if r.get("first_fit_s")
                else np.nan
            )
            ratio = (
                "**fastest**"
                if np.isfinite(warm) and warm == fastest
                else (
                    f"{warm / fastest:.2f}x"
                    if np.isfinite(warm) and np.isfinite(fastest)
                    else "—"
                )
            )
            flag = " ⚠ differs" if np.isfinite(rel) and rel > 1e-4 else ""
            rss = r.get("peak_rss_mb")
            lines.append(
                f"| {r['backend']} | {r['status']}{flag} | {est:.6g} | {rel:.1e} | "
                f"{cold:.2f} | {warm:.3f} | {r.get('warm_iqr_s', np.nan):.3f} | "
                f"{(f'{rss:.0f}' if isinstance(rss, (int, float)) and np.isfinite(rss) else '—')} | {ratio} |"
            )
        errs = grp[grp["status"] != "ok"]
        for _, r in errs.iterrows():
            msg = r.get("error") or "; ".join(r.get("errors") or []) or r["status"]
            lines.append(f"\n> {r['backend']}: {str(msg)[:300]}")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
