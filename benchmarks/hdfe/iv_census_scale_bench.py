"""Census-scale absorbed 2SLS: ``sp.hdfe_ols`` vs ``fixest::feols`` vs ``ivreghdfe``.

The specification is the one Goldman, Gracie and Porter (AER 116(10), 2026)
run on 15-23 million people::

    ivreghdfe married (tight = z) x, absorb(group#tract18#sex) cluster(county18)

which ``sp.from_stata`` translates to::

    sp.hdfe_ols("married ~ x | grp^tract^sex | tight ~ z", data=df, cluster="county")

Synthetic DGP (``gen``), sized like the paper at 20M rows:

* ``n_tracts = n / 285`` tracts (~70k at 20M) with lognormal(0, 1) sizes, so
  a few tracts are large and many are small (some singleton cells);
* tracts nested in ``n_tracts / 23`` counties (~3,000 at 20M), the cluster;
* ``grp`` (2 groups, 15% / 85%) x ``sex`` (2) x tract = the absorbed cell
  (~280k fixed-effect levels at 20M);
* ``x`` an integer age 18-64; ``z`` a county x age shock plus individual
  noise; ``tight = 0.03 z + 0.4 v + cell effect + noise`` with ``v`` an
  unobserved confounder; ``married ~ Bernoulli(0.03 + 0.004 tight +
  0.006 v + cell effect)`` (outcome rate ~3%).

Every float column is rounded to a dyadic rational (``k / 1024``), so its
decimal form in the CSV is exact and every engine parses the same doubles.

Subcommands::

    python iv_census_scale_bench.py gen --n 1000000
    python iv_census_scale_bench.py run --sizes 1000000 5000000 20000000 \
        --engines rust norust fixest stata --threads 4
    python iv_census_scale_bench.py worker --n 1000000 --engine rust   # internal

``run`` launches every (size, engine) in its own process under
``/usr/bin/time -l`` (process peak RSS), sequentially, and writes
``results/iv_census_scale.json``. Each worker loads the CSV, runs a small
warm-up fit (imports, JIT, thread pools), then times one full fit; load
time and estimation time are reported separately.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
R_SCRIPT = HERE / "iv_census_scale_fixest.R"
STATA_DO = HERE / "iv_census_scale_ivreghdfe.do"
STATA_BIN = "/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp"
FORMULA = "married ~ x | grp^tract^sex | tight ~ z"
SEED = 20261028


def csv_path(n: int) -> Path:
    return DATA / f"iv_census_{n}.csv"


# --------------------------------------------------------------------- DGP
def generate(n: int, seed: int = SEED) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + n)
    n_tracts = max(200, n // 285)
    n_counties = max(20, n_tracts // 23)
    size = rng.lognormal(0.0, 1.0, n_tracts)
    cum = np.cumsum(size / size.sum())
    tract = np.searchsorted(cum, rng.random(n), side="right").astype(np.int32)
    tract = np.minimum(tract, n_tracts - 1)
    county_of_tract = rng.integers(0, n_counties, n_tracts).astype(np.int32)
    county = county_of_tract[tract]
    grp = (rng.random(n) < 0.15).astype(np.int8)
    sex = rng.integers(0, 2, n).astype(np.int8)
    x = rng.integers(18, 65, n).astype(np.int16)

    cell = (tract.astype(np.int64) * 2 + grp) * 2 + sex
    n_cells = n_tracts * 4
    cell_d = rng.normal(0.0, 0.5, n_cells)[cell]
    cell_y = rng.normal(0.0, 0.01, n_cells)[cell]
    shock = rng.normal(0.0, 1.0, n_counties * 47)
    z = shock[county.astype(np.int64) * 47 + (x - 18)] + 0.5 * rng.normal(size=n)
    v = rng.normal(size=n)
    tight = 0.03 * z + 0.4 * v + cell_d + rng.normal(size=n)
    p = np.clip(0.03 + 0.004 * tight + 0.006 * v + cell_y, 0.0, 1.0)
    married = (rng.random(n) < p).astype(np.int8)

    def dyadic(a: np.ndarray) -> np.ndarray:
        return np.round(a * 1024.0) / 1024.0

    return {
        "married": married,
        "tight": dyadic(tight),
        "z": dyadic(z),
        "x": x,
        "grp": grp,
        "tract": tract,
        "sex": sex,
        "county": county,
    }


def cmd_gen(n: int) -> Path:
    import pyarrow as pa
    import pyarrow.csv as pacsv

    DATA.mkdir(exist_ok=True)
    out = csv_path(n)
    if out.exists():
        return out
    cols = generate(n)
    table = pa.table(cols)
    tmp = out.with_suffix(".tmp")
    pacsv.write_csv(table, tmp)
    tmp.rename(out)
    meta = {
        "n": n,
        "seed": SEED + n,
        "n_tracts": int(cols["tract"].max()) + 1,
        "n_counties": int(cols["county"].max()) + 1,
        "outcome_rate": float(cols["married"].mean()),
    }
    (DATA / f"iv_census_{n}.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return out


# ------------------------------------------------------------------ worker
def _rss() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss)


def cmd_worker(n: int, engine: str, out: Path) -> None:
    import pandas as pd

    t0 = time.perf_counter()
    df = pd.read_csv(
        csv_path(n),
        engine="pyarrow",
        dtype={
            "married": np.float64,
            "tight": np.float64,
            "z": np.float64,
            "x": np.float64,
            "grp": np.int8,
            "tract": np.int32,
            "sex": np.int8,
            "county": np.int32,
        },
    )
    load_s = time.perf_counter() - t0
    rss_loaded = _rss()

    import statspai as sp
    from statspai import _rust_backend

    if engine == "rust" and _rust_backend.rust is None:
        raise SystemExit("rust engine requested but statspai_hdfe is not loaded")
    if engine == "norust" and _rust_backend.rust is not None:
        raise SystemExit("norust engine requested but STATSPAI_SKIP_RUST is not set")
    warm = df.iloc[:20000]
    sp.hdfe_ols(FORMULA, data=warm, cluster="county")  # imports, JIT, threads
    t0 = time.perf_counter()
    res = sp.hdfe_ols(FORMULA, data=df, cluster="county")
    fit_s = time.perf_counter() - t0
    fs = res.iv_diagnostics["first_stage"]["tight"]
    payload = {
        "engine": engine,
        "n_rows": int(len(df)),
        "load_s": load_s,
        "fit_s": fit_s,
        "rss_after_load_bytes": rss_loaded,
        "coef": {k: float(v) for k, v in res.params.items()},
        "se": {k: float(v) for k, v in res.std_errors.items()},
        "n_obs": int(res.n_obs),
        "n_singletons_dropped": int(res.n_singletons_dropped),
        "n_fe": [int(v) for v in res.n_fe],
        "n_clusters": int(res.cluster_info["n_clusters"][0]),
        "first_stage_F": float(fs["F"]),
        "demean_iters": int(res.iters),
        "statspai_version": sp.__version__,
        "statspai_file": sp.__file__,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --------------------------------------------------------------------- run
def _timed(cmd: List[str], env: Dict[str, str], log: Path) -> Dict[str, Any]:
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.run(
            ["/usr/bin/time", "-l"] + cmd,
            env=env,
            stdout=fh,
            stderr=subprocess.PIPE,
            cwd=str(RESULTS),  # Stata's batch log lands here (gitignored *.log)
        )
    wall = time.perf_counter() - t0
    err = proc.stderr.decode("utf-8", "replace")
    m = re.search(r"(\d+)\s+maximum resident set size", err)
    return {
        "returncode": proc.returncode,
        "process_wall_s": wall,
        "peak_rss_bytes": int(m.group(1)) if m else None,
        "stderr_tail": err[-2000:] if proc.returncode else None,
    }


def machine_info() -> Dict[str, Any]:
    def sysctl(key: str) -> Optional[str]:
        try:
            return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    try:
        load = os.getloadavg()
    except OSError:
        load = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": sysctl("machdep.cpu.brand_string"),
        "ncpu": sysctl("hw.ncpu"),
        "mem_bytes": int(sysctl("hw.memsize") or 0),
        "python": sys.version.split()[0],
        "loadavg_at_start": load,
    }


def run_one(n: int, engine: str, threads: int) -> Dict[str, Any]:
    tmp = RESULTS / f".iv_census_{n}_{engine}.json"
    log = RESULTS / f".iv_census_{n}_{engine}.log"
    if tmp.exists():
        tmp.unlink()
    env = dict(os.environ)
    env["STATSPAI_NUM_THREADS"] = str(threads)
    env["RAYON_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)
    env["NUMBA_NUM_THREADS"] = str(threads)
    if engine in ("rust", "norust"):
        if engine == "norust":
            env["STATSPAI_SKIP_RUST"] = "1"
        else:
            env.pop("STATSPAI_SKIP_RUST", None)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--n",
            str(n),
            "--engine",
            engine,
            "--out",
            str(tmp),
        ]
    elif engine == "fixest":
        cmd = ["Rscript", str(R_SCRIPT), str(csv_path(n)), str(threads), str(tmp)]
    elif engine == "stata":
        # Stata's batch mode writes the log next to the do-file's cwd
        cmd = [
            STATA_BIN,
            "-b",
            "do",
            str(STATA_DO),
            str(csv_path(n)),
            str(tmp),
            str(threads),
        ]
    else:
        raise SystemExit(f"unknown engine {engine}")
    proc = _timed(cmd, env, log)
    if engine == "stata":
        # batch mode with arguments names its log ".log" in the cwd
        batch_log = RESULTS / ".log"
        if batch_log.exists():
            batch_log.replace(RESULTS / f".iv_census_{n}_stata_batch.log")
    rec: Dict[str, Any] = {"size": n, "engine": engine, "threads": threads, **proc}
    if tmp.exists():
        rec.update(json.loads(tmp.read_text(encoding="utf-8")))
        tmp.unlink()
    elif proc["returncode"] == 0:
        rec["error"] = "worker wrote no result"
    return rec


def compare(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Relative differences of coef / SE against the Rust-path StatsPAI fit."""
    out = []
    by = {(r["size"], r["engine"]): r for r in records if "coef" in r}
    for (n, eng), r in sorted(by.items()):
        base = by.get((n, "rust"))
        if base is None or eng == "rust":
            continue
        row = {"size": n, "engine": eng, "vs": "rust"}
        for key in ("coef", "se"):
            for name, val in base[key].items():
                other = r[key].get(name)
                if other is not None:
                    row[f"rel_{key}_{name}"] = abs(other - val) / abs(val)
        out.append(row)
    return out


def cmd_run(sizes: List[int], engines: List[str], threads: int, tag: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    out_file = RESULTS / f"iv_census_scale{tag}.json"
    doc: Dict[str, Any] = {
        "benchmark": "iv_census_scale",
        "formula": FORMULA,
        "stata": "ivreghdfe married x (tight = z), absorb(grp#tract#sex) cluster(county)",
        "r": "feols(married ~ x | grp^tract^sex | tight ~ z, cluster = ~county)",
        "machine": machine_info(),
        "records": [],
    }
    if out_file.exists():
        prev = json.loads(out_file.read_text(encoding="utf-8"))
        doc["records"] = prev.get("records", [])
    for n in sizes:
        cmd_gen(n)
        for eng in engines:
            print(f"[{time.strftime('%H:%M:%S')}] n={n:,} engine={eng}", flush=True)
            rec = run_one(n, eng, threads)
            rec["loadavg_after"] = list(os.getloadavg())
            doc["records"] = [
                r for r in doc["records"] if (r["size"], r["engine"]) != (n, eng)
            ] + [rec]
            doc["comparison"] = compare(doc["records"])
            out_file.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            print(
                f"    fit {rec.get('fit_s')} s  peak RSS "
                f"{(rec.get('peak_rss_bytes') or 0) / 2**30:.2f} GiB  "
                f"rc={rec['returncode']}",
                flush=True,
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--n", type=int, required=True)
    w = sub.add_parser("worker")
    w.add_argument("--n", type=int, required=True)
    w.add_argument("--engine", choices=["rust", "norust"], required=True)
    w.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--sizes", type=int, nargs="+", required=True)
    r.add_argument(
        "--engines",
        nargs="+",
        default=["rust", "norust", "fixest", "stata"],
        choices=["rust", "norust", "fixest", "stata"],
    )
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--tag", default="", help="suffix of the results file")
    a = ap.parse_args()
    if a.cmd == "gen":
        print(cmd_gen(a.n))
    elif a.cmd == "worker":
        cmd_worker(a.n, a.engine, a.out)
    else:
        cmd_run(a.sizes, a.engines, a.threads, a.tag)


if __name__ == "__main__":
    main()
