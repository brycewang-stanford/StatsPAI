"""Wall-clock benchmark of the optional Rust kernels against the NumPy path.

Times ``sp.fast.feols`` (OLS, absorbed FEs, iid SE) and ``sp.fast.fepois``
on two synthetic panels of one million rows:

* ``easy``: the Track C DGP (``tests/perf/_data.py``): firm and year
  effects assigned at random, so alternating projections converge in a few
  sweeps.
* ``hard``: worker, firm and year effects where only 5% of a worker's
  observations leave the worker's home firm (low mobility, the AKM case),
  so the worker-firm graph mixes slowly.

Each configuration runs in a fresh interpreter: the NumPy path with
``STATSPAI_SKIP_RUST=1``, the Rust path with ``STATSPAI_NUM_THREADS`` set
to 1 and to the CPU count. With ``--fixest`` the same models are timed in
R ``fixest`` with one and with all threads (needs ``Rscript`` with
``fixest`` and ``data.table``). Every other thread pool is pinned to one
thread. Run on an idle machine; the medians of ``--reps`` repetitions after
one warm-up are reported.

Usage::

    python benchmarks/hdfe/rust_kernels_bench.py [--reps 5] [--fixest] [--out FILE.md]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

CHILD = r"""
import json, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
import _data
import statspai as sp

def hard(n=1_000_000, W=100_000, F=5_000, seed=0):
    rng = np.random.default_rng(seed)
    worker = rng.integers(0, W, n)
    home = rng.integers(0, F, W)
    move = rng.random(n) < 0.05
    firm = np.where(move, rng.integers(0, F, n), home[worker])
    year = rng.integers(0, 20, n)
    x1 = rng.normal(size=n) + 0.3 * (firm % 7)
    x2 = rng.normal(size=n)
    eta = 0.1 * x1 - 0.1 * x2 + rng.normal(0, 0.3, W)[worker] + rng.normal(0, 0.3, F)[firm]
    y = x1 - x2 + rng.normal(size=W)[worker] + rng.normal(size=F)[firm] \
        + rng.normal(size=20)[year] + rng.normal(size=n)
    return pd.DataFrame(dict(y=y, x1=x1, x2=x2, worker=worker, firm=firm, year=year,
                             count=rng.poisson(np.exp(eta))))

reps = int(sys.argv[2])
easy = _data.hdfe(1_000_000)
easy["count"] = np.random.default_rng(1).poisson(np.exp(0.2 * easy["x1"]))
data = {"easy": easy, "hard": hard()}
models = {
    "easy": ("y ~ x1 + x2 | firm + year", "count ~ x1 + x2 | firm + year"),
    "hard": ("y ~ x1 + x2 | worker + firm + year", "count ~ x1 + x2 | worker + firm + year"),
}
out = {"backend": sp.fast.backend_info()}
for name, df in data.items():
    for est, f in zip(("feols", "fepois"), models[name]):
        fn = getattr(sp.fast, est)
        fit = fn(f, data=df)
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); fn(f, data=df); ts.append(time.perf_counter() - t)
        out[f"{name}/{est}"] = {"median_s": float(np.median(ts)),
                                "coef": [float(v) for v in fit.coef_vec]}
    if sys.argv[3]:
        df.to_csv(f"{sys.argv[3]}/{name}.csv", index=False)
print("@@" + json.dumps(out))
"""

R_SCRIPT = r"""
suppressMessages({library(fixest); library(data.table)})
args <- commandArgs(TRUE); dir <- args[1]; reps <- as.integer(args[2]); nt <- as.integer(args[3])
setFixest_nthreads(nt)
res <- list()
for (name in c("easy", "hard")) {
  d <- fread(file.path(dir, paste0(name, ".csv")))
  fe <- if (name == "easy") "firm + year" else "worker + firm + year"
  for (est in c("feols", "fepois")) {
    lhs <- if (est == "feols") "y" else "count"
    f <- as.formula(paste(lhs, "~ x1 + x2 |", fe))
    run <- function() if (est == "feols") feols(f, d, vcov = "iid") else fepois(f, d, vcov = "iid")
    fit <- run()
    t <- sapply(seq_len(reps), function(i) system.time(run())[["elapsed"]])
    res[[paste0(name, "/", est)]] <- list(median_s = median(t), coef = unname(coef(fit)))
  }
}
cat("@@", jsonlite::toJSON(res, auto_unbox = TRUE, digits = NA), "\n", sep = "")
"""

PINNED = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}


def _parse(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    raise RuntimeError(f"no result line in:\n{stdout[-2000:]}")


def run_python(label: str, extra_env: dict, reps: int, csv_dir: str) -> dict:
    env = dict(os.environ, **PINNED, **extra_env)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, str(REPO / "tests" / "perf"), str(reps), csv_dir],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    res = _parse(proc.stdout)
    print(f"{label}: done", file=sys.stderr)
    return res


def run_fixest(threads: int, reps: int, csv_dir: str) -> dict:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".R", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(R_SCRIPT)
    proc = subprocess.run(
        ["Rscript", fh.name, csv_dir, str(reps), str(threads)],
        capture_output=True,
        text=True,
        check=True,
    )
    return _parse(proc.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--fixest", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    ncpu = os.cpu_count() or 1

    with tempfile.TemporaryDirectory() as csv_dir:
        cols = {
            "NumPy": run_python(
                "numpy", {"STATSPAI_SKIP_RUST": "1"}, args.reps, csv_dir
            )
        }
        if not cols["NumPy"]:
            return 1
        rust1 = run_python("rust x1", {"STATSPAI_NUM_THREADS": "1"}, args.reps, "")
        if not rust1["backend"]["available"]:
            print(
                "Rust kernels unavailable:", rust1["backend"]["reason"], file=sys.stderr
            )
            return 2
        cols["Rust, 1 thread"] = rust1
        cols[f"Rust, {ncpu} threads"] = run_python(
            f"rust x{ncpu}", {"STATSPAI_NUM_THREADS": str(ncpu)}, args.reps, ""
        )
        if args.fixest and shutil.which("Rscript"):
            cols["fixest, 1 thread"] = run_fixest(1, args.reps, csv_dir)
            cols[f"fixest, {ncpu} threads"] = run_fixest(ncpu, args.reps, csv_dir)

    rows = ["easy/feols", "easy/fepois", "hard/feols", "hard/fepois"]
    lines = [
        f"Machine: {platform.platform()}, {ncpu} logical CPUs, Python "
        f"{platform.python_version()}, statspai-hdfe {rust1['backend']['version']}. "
        f"Median of {args.reps} runs, seconds; n = 1,000,000.",
        "",
        "| model | " + " | ".join(cols) + " |",
        "|---|" + "---:|" * len(cols),
    ]
    for r in rows:
        lines.append(
            f"| {r} | "
            + " | ".join(f"{c[r]['median_s']:.3f}" for c in cols.values())
            + " |"
        )
    base = cols["NumPy"]
    worst = max(
        abs(a - b)
        for c in cols.values()
        for r in rows
        for a, b in zip(c[r]["coef"], base[r]["coef"])
    )
    lines += ["", f"Largest coefficient difference from the NumPy path: {worst:.1e}."]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
