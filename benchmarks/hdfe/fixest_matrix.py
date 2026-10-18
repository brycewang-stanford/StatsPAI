"""StatsPAI against R ``fixest`` on a matrix of fixed-effects regressions.

Each cell is one model on one input. Inputs are drawn once in Python and
written to CSV; R reads the same bytes. Every StatsPAI configuration runs in
a fresh interpreter, every fixest configuration in a fresh ``Rscript``, with
BLAS pinned to one thread. Only the estimation call is timed (no I/O); the
reported number is the minimum over repetitions after one warm-up, since on
a shared machine the minimum is the least contaminated by other load.

Cells (``easy`` is the Track C panel, firm and year effects; ``hard`` is a
low-mobility worker x firm x year panel, the slow case for demeaning)::

    feols_iid     y ~ x1 + x2 | firm + year                 iid
    feols_cl      y ~ x1 + x2 | firm + year                 cluster firm
    feols_hc      y ~ x1 + x2 | firm + year                 hetero
    feols_w       y ~ x1 + x2 | firm + year, weights w      iid
    feols_k10     y ~ x1 + ... + x10 | firm + year          cluster firm
    fepois_iid    count ~ x1 + x2 | firm + year             iid
    fepois_cl     count ~ x1 + x2 | firm + year             cluster firm
    logit_cl      d ~ x1 + x2 | firm + year (binomial)      cluster firm
    hard_feols    y ~ x1 + x2 | worker + firm + year        iid
    hard_fepois   count ~ x1 + x2 | worker + firm + year    iid

StatsPAI entry points: ``sp`` is what a user reaches first (``sp.feols`` /
``sp.fepois`` / ``sp.feglm`` with default arguments); ``pyfixest`` is the
same call with ``backend='pyfixest'``; ``fast`` is ``sp.fast.feols`` /
``sp.fast.fepois``. Run with the Rust extension
importable (``pip install statspai-hdfe``); ``--skip-rust`` times the
pure-Python fallback instead.

Usage::

    python benchmarks/hdfe/fixest_matrix.py --sizes 1e4 1e5 1e6 [--threads 1 8]
        [--entries sp fast] [--cells feols_iid ...] [--no-fixest] [--out F.json]
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

CELLS = {
    # name: (data, lhs, rhs, fe, vcov, weights, family)
    "feols_iid": ("easy", "y", "x1 + x2", "firm + year", "iid", None, "ols"),
    "feols_cl": ("easy", "y", "x1 + x2", "firm + year", "firm", None, "ols"),
    "feols_hc": ("easy", "y", "x1 + x2", "firm + year", "hetero", None, "ols"),
    "feols_w": ("easy", "y", "x1 + x2", "firm + year", "iid", "w", "ols"),
    "feols_k10": (
        "easy",
        "y",
        " + ".join(f"x{i}" for i in range(1, 11)),
        "firm + year",
        "firm",
        None,
        "ols",
    ),
    "fepois_iid": ("easy", "count", "x1 + x2", "firm + year", "iid", None, "pois"),
    "fepois_cl": ("easy", "count", "x1 + x2", "firm + year", "firm", None, "pois"),
    "logit_cl": ("easy", "d", "x1 + x2", "firm + year", "firm", None, "logit"),
    "hard_feols": ("hard", "y", "x1 + x2", "worker + firm + year", "iid", None, "ols"),
    "hard_fepois": (
        "hard",
        "count",
        "x1 + x2",
        "worker + firm + year",
        "iid",
        None,
        "pois",
    ),
}

GEN = r"""
import sys
import numpy as np, pandas as pd

def easy(n, seed=42):
    rng = np.random.default_rng(seed)
    n_firms = max(50, int(np.sqrt(n) * 2))
    firm = rng.integers(0, n_firms, n)
    year = rng.integers(0, 20, n)
    X = rng.normal(size=(n, 10))
    ffe = rng.normal(0, 1.0, n_firms)[firm]
    yfe = rng.normal(0, 0.5, 20)[year]
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + ffe + yfe + rng.normal(0, 0.5, n)
    eta = 0.2 * X[:, 0] - 0.1 * X[:, 1] + 0.3 * ffe
    d = (rng.random(n) < 1 / (1 + np.exp(-(0.5 * X[:, 0] - 0.5 * X[:, 1] + ffe)))).astype(int)
    df = pd.DataFrame({f"x{i + 1}": X[:, i] for i in range(10)})
    df.insert(0, "y", y)
    df["count"] = rng.poisson(np.exp(eta))
    df["d"] = d
    df["w"] = rng.uniform(0.5, 2.0, n)
    df["firm"] = firm
    df["year"] = year
    return df

def hard(n, seed=0):
    rng = np.random.default_rng(seed)
    W, F = max(10, n // 10), max(5, n // 200)
    worker = rng.integers(0, W, n)
    home = rng.integers(0, F, W)
    move = rng.random(n) < 0.05
    firm = np.where(move, rng.integers(0, F, n), home[worker])
    year = rng.integers(0, 20, n)
    x1 = rng.normal(size=n) + 0.3 * (firm % 7)
    x2 = rng.normal(size=n)
    eta = 0.1 * x1 - 0.1 * x2 + rng.normal(0, 0.3, W)[worker] + rng.normal(0, 0.3, F)[firm]
    y = (x1 - x2 + rng.normal(size=W)[worker] + rng.normal(size=F)[firm]
         + rng.normal(size=20)[year] + rng.normal(size=n))
    return pd.DataFrame(dict(y=y, x1=x1, x2=x2, worker=worker, firm=firm, year=year,
                             count=rng.poisson(np.exp(eta))))

kind, n, path = sys.argv[1], int(float(sys.argv[2])), sys.argv[3]
(easy if kind == "easy" else hard)(n).to_csv(path, index=False, float_format="%.17g")
"""

PY = r"""
import json, sys, time, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
import statspai as sp
spec = json.loads(sys.argv[1])
out = {"backend": sp.fast.backend_info()}
frames = {}
for job in spec["jobs"]:
    df = frames.get(job["csv"])
    if df is None:
        df = frames[job["csv"]] = pd.read_csv(job["csv"])
    lhs, rhs, fe, vc, w, fam = job["lhs"], job["rhs"], job["fe"], job["vcov"], job["w"], job["fam"]
    fml = f"{lhs} ~ {rhs} | {fe}"
    entry = job["entry"]
    if entry in ("sp", "pyfixest"):
        vcov = {"CRV1": vc} if vc not in ("iid", "hetero") else vc
        kw = {"vcov": vcov}
        if entry == "pyfixest" and fam != "logit":
            kw["backend"] = "pyfixest"
        if w:
            kw["weights"] = w
        if fam == "ols":
            call = lambda: sp.feols(fml, data=df, **kw)
        elif fam == "pois":
            call = lambda: sp.fepois(fml, data=df, **kw)
        else:
            call = lambda: sp.feglm(fml, data=df, family="logit", **kw)
    else:
        kw = {}
        if vc == "hetero":
            kw["vcov"] = "hc1"
        elif vc != "iid":
            kw["vcov"] = "cr1"
            kw["cluster"] = vc
        if w:
            kw["weights"] = w
        fn = {"ols": getattr(sp.fast, "feols", None), "pois": getattr(sp.fast, "fepois", None),
              "logit": getattr(sp.fast, "feglm", None)}[fam]
        if fn is None:
            out[job["key"]] = {"skipped": "no sp.fast entry"}
            continue
        if fam == "logit":
            kw["family"] = "logit"
        call = lambda: fn(fml, data=df, **kw)
    try:
        fit = call()
    except Exception as exc:
        out[job["key"]] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        continue
    ts = []
    for _ in range(job["reps"]):
        t = time.perf_counter(); call(); ts.append(time.perf_counter() - t)
    params = getattr(fit, "params", None)
    if params is None:
        params = getattr(fit, "coef", None)
        params = params() if callable(params) else params
    se = getattr(fit, "std_errors", None)
    if se is None:
        se = getattr(fit, "se", None)
        se = se() if callable(se) else se
    out[job["key"]] = {"min_s": min(ts), "median_s": float(np.median(ts)),
                       "coef": [float(v) for v in np.asarray(params, dtype=float)],
                       "se": [float(v) for v in np.asarray(se, dtype=float)]}
print("@@" + json.dumps(out))
"""

R = r"""
suppressMessages({library(fixest); library(data.table); library(jsonlite)})
spec <- fromJSON(commandArgs(TRUE)[1], simplifyVector = FALSE)
setFixest_nthreads(as.integer(spec$threads))
frames <- list(); res <- list()
for (job in spec$jobs) {
  if (is.null(frames[[job$csv]])) frames[[job$csv]] <- fread(job$csv)
  d <- frames[[job$csv]]
  f <- as.formula(paste(job$lhs, "~", job$rhs, "|", job$fe))
  vc <- if (job$vcov == "iid") "iid" else if (job$vcov == "hetero") "hetero" else as.formula(paste("~", job$vcov))
  w <- if (is.null(job$w)) NULL else as.formula(paste("~", job$w))
  run <- switch(job$fam,
    ols = function() feols(f, d, vcov = vc, weights = w),
    pois = function() fepois(f, d, vcov = vc, weights = w),
    logit = function() feglm(f, d, family = "logit", vcov = vc, weights = w))
  fit <- run()
  ts <- numeric(job$reps)
  for (i in seq_len(job$reps)) { t0 <- Sys.time(); run(); ts[i] <- as.numeric(Sys.time() - t0, units = "secs") }
  res[[job$key]] <- list(min_s = min(ts), median_s = median(ts), coef = unname(coef(fit)), se = unname(se(fit)))
}
cat("@@", toJSON(res, auto_unbox = TRUE, digits = NA), "\n", sep = "")
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
    raise RuntimeError(f"no result line in:\n{stdout[-3000:]}")


def _reps(n: int) -> int:
    return 30 if n <= 10_000 else 10 if n <= 100_000 else 5 if n <= 1_000_000 else 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--sizes", nargs="+", default=["1e4", "1e5", "1e6"])
    ap.add_argument("--threads", nargs="+", type=int, default=[1, os.cpu_count() or 1])
    ap.add_argument("--entries", nargs="+", default=["sp", "fast"])
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--no-fixest", action="store_true")
    ap.add_argument("--skip-rust", action="store_true")
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    data_dir = args.data_dir or Path(tempfile.mkdtemp(prefix="fixest_matrix_"))
    data_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for size in args.sizes:
        n = int(float(size))
        for cell in args.cells:
            kind, lhs, rhs, fe, vc, w, fam = CELLS[cell]
            csv = data_dir / f"{kind}_{n}.csv"
            if not csv.exists():
                subprocess.run(
                    [sys.executable, "-c", GEN, kind, str(n), str(csv)], check=True
                )
            jobs.append(
                dict(
                    key=f"{cell}@{n}",
                    csv=str(csv),
                    lhs=lhs,
                    rhs=rhs,
                    fe=fe,
                    vcov=vc,
                    w=w,
                    fam=fam,
                    reps=_reps(n),
                )
            )

    results: dict = {
        "machine": f"{platform.platform()}, {os.cpu_count()} logical CPUs",
        "python": platform.python_version(),
        "columns": {},
    }
    for threads in args.threads:
        for entry in args.entries:
            env = dict(os.environ, **PINNED, STATSPAI_NUM_THREADS=str(threads))
            if args.skip_rust:
                env["STATSPAI_SKIP_RUST"] = "1"
            env["PYTHONPATH"] = os.pathsep.join(
                [str(REPO / "src"), env.get("PYTHONPATH", "")]
            )
            spec = {"jobs": [dict(j, entry=entry) for j in jobs]}
            proc = subprocess.run(
                [sys.executable, "-c", PY, json.dumps(spec)],
                env=env,
                capture_output=True,
                text=True,
            )
            if proc.returncode:
                print(proc.stderr[-3000:], file=sys.stderr)
                return 1
            results["columns"][f"{entry} x{threads}"] = _parse(proc.stdout)
            print(f"{entry} x{threads}: done", file=sys.stderr)
        if not args.no_fixest and shutil.which("Rscript"):
            with tempfile.NamedTemporaryFile(
                "w", suffix=".R", delete=False, encoding="utf-8"
            ) as fh:
                fh.write(R)
            spec = {"threads": threads, "jobs": jobs}
            proc = subprocess.run(
                ["Rscript", fh.name, json.dumps(spec)], capture_output=True, text=True
            )
            if proc.returncode:
                print(proc.stderr[-3000:], file=sys.stderr)
                return 1
            results["columns"][f"fixest x{threads}"] = _parse(proc.stdout)
            print(f"fixest x{threads}: done", file=sys.stderr)

    cols = results["columns"]
    keys = [j["key"] for j in jobs]
    header = "| cell | " + " | ".join(cols) + " |"
    lines = [results["machine"], "", header, "|---|" + "---:|" * len(cols)]
    for k in keys:
        cells = []
        for c in cols.values():
            r = c.get(k, {})
            if "min_s" in r:
                cells.append(f"{r['min_s'] * 1000:.1f}")
            else:
                cells.append("err" if "error" in r else "-")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Milliseconds, minimum over repetitions.")
    # coefficient agreement with fixest (single-thread column)
    ref = next((c for n, c in cols.items() if n.startswith("fixest")), None)
    if ref:
        for name, c in cols.items():
            if name.startswith("fixest"):
                continue
            worst_b = worst_se = 0.0
            for k in keys:
                a, b = c.get(k, {}), ref.get(k, {})
                if "coef" in a and "coef" in b and len(a["coef"]) == len(b["coef"]):
                    for u, v in zip(a["coef"], b["coef"]):
                        worst_b = max(worst_b, abs(u - v) / max(abs(v), 1e-12))
                    for u, v in zip(a["se"], b["se"]):
                        worst_se = max(worst_se, abs(u - v) / max(abs(v), 1e-12))
            lines.append(
                f"{name}: max rel. diff to fixest coef {worst_b:.1e}, se {worst_se:.1e}"
            )
    text = "\n".join(lines)
    print(text)
    errors = {
        f"{name}/{k}": r["error"]
        for name, c in cols.items()
        for k, r in c.items()
        if isinstance(r, dict) and "error" in r
    }
    for k, e in errors.items():
        print(f"ERROR {k}: {e}", file=sys.stderr)
    if args.out:
        args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
