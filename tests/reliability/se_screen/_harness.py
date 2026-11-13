"""Shared machinery for the standard-error calibration screen.

A family script declares ``ROWS`` (a list of :class:`Row`) and an optional
``MECHANISMS`` dict, then calls :func:`main`. Each row is one estimator call
on one data-generating process whose assumptions the estimator states and
whose true parameter is known. The screen draws ``B`` independent data sets
(seeds ``SeedSequence(base).spawn(B)``, ``base`` derived from the row name),
fits each, and records

* ``coverage``: share of draws whose *reported* 95% interval holds the truth,
* ``se_over_sd``: mean reported SE over the Monte Carlo SD of the estimate,
* ``bias_over_sd``: (mean estimate - truth) over that SD.

A row is flagged when coverage is outside ``[0.918, 0.985]``, ``se_over_sd``
outside ``[0.88, 1.12]`` or ``|bias_over_sd| > 0.25``. Those limits are for
B = 400; for a smaller B the coverage limits widen to the Wilson 99% band
around 0.95 and the SE/SD limits by ``sqrt(400 / B)`` (see :func:`band`).

Nothing here is specific to a family. The fit runs in at most two worker
processes (``spawn`` context, BLAS pinned to one thread).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import signal
import sys
import time
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

GLOBAL_BASE = 20261113
B_SCREEN = 400
B_CONFIRM = 1000
PREFIX = 20
MAX_PROCS = 2
COVERAGE_BAND = (0.918, 0.985)
SE_BAND = (0.88, 1.12)
BIAS_LIMIT = 0.25
DRAW_TIMEOUT = 120  # seconds; a draw that takes longer counts as a failure


class DrawTimeout(Exception):
    """One fit exceeded ``DRAW_TIMEOUT`` seconds."""


def _alarm(signum, frame):
    raise DrawTimeout(f"fit did not return within {DRAW_TIMEOUT} s")


@dataclass
class Row:
    """One screened call.

    ``dgp(rng)`` returns the data; ``fit(data)`` returns
    ``(estimate, se, ci_low, ci_high)`` exactly as the estimator reports
    them. ``B`` overrides the screen size for slow estimators.
    """

    name: str
    estimator: str
    call: str
    n: int
    truth: float
    dgp: Callable[[np.random.Generator], Any]
    fit: Callable[[Any], Tuple[float, float, float, float]]
    B: Optional[int] = None
    design: str = ""
    # The estimator reports a confidence set and no SE (weak-IV robust sets,
    # randomization intervals). ``fit`` then returns one bool: is the truth in
    # the reported set. Only coverage is computed and flagged.
    interval_only: bool = False


def coef(result: Any, term: Optional[str] = None) -> Tuple[float, float, float, float]:
    """Estimate, SE and reported interval for ``term`` (or the headline)."""
    if term is None:
        lo, hi = result.ci
        return float(result.estimate), float(result.se), float(lo), float(hi)
    est = float(result.params[term])
    se = float(result.std_errors[term])
    ci = result.conf_int()
    lo, hi = (float(v) for v in np.asarray(ci.loc[term])[:2])
    return est, se, lo, hi


def wald(est: float, se: float) -> Tuple[float, float, float, float]:
    """Normal 95% interval for results that report an SE but no interval."""
    z = 1.959963984540054
    return float(est), float(se), float(est - z * se), float(est + z * se)


def tidy_row(result: Any, **match: Any) -> Tuple[float, float, float, float]:
    """Row of ``result.tidy()`` selected by column values, as (est, se, lo, hi)."""
    t = result.tidy()
    for k, v in match.items():
        t = t[t[k] == v]
    (row,) = t.to_dict("records")
    return (
        float(row["estimate"]),
        float(row["std_error"]),
        float(row["conf_low"]),
        float(row["conf_high"]),
    )


def wilson(p: float, n: int, z: float = 2.5758293035489) -> Tuple[float, float]:
    """Wilson score interval for a proportion ``p`` observed on ``n`` draws."""
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def band(B: int) -> Dict[str, List[float]]:
    """Flagging limits for a row screened on ``B`` draws."""
    cov = list(COVERAGE_BAND)
    se = list(SE_BAND)
    if B < B_SCREEN:
        lo, hi = wilson(0.95, B)
        cov = [min(cov[0], round(lo, 4)), max(cov[1], round(hi, 4))]
        w = 0.12 * math.sqrt(B_SCREEN / B)
        se = [round(1 - w, 4), round(1 + w, 4)]
    return {
        "coverage": cov,
        "se_over_sd": se,
        "bias_over_sd": [-BIAS_LIMIT, BIAS_LIMIT],
    }


def is_flagged(row: Dict[str, Any]) -> bool:
    """Recompute the flag from the stored numbers of a result row."""
    b = band(int(row["B"]))
    if row.get("se_over_sd") is None:  # interval-only row
        return not b["coverage"][0] <= row["coverage"] <= b["coverage"][1]
    return not (
        b["coverage"][0] <= row["coverage"] <= b["coverage"][1]
        and b["se_over_sd"][0] <= row["se_over_sd"] <= b["se_over_sd"][1]
        and abs(row["bias_over_sd"]) <= BIAS_LIMIT
    )


def row_seeds(name: str, B: int) -> List[np.random.SeedSequence]:
    base = GLOBAL_BASE + zlib.crc32(name.encode("utf-8"))
    return np.random.SeedSequence(base).spawn(B)


_MODULES: Dict[str, Any] = {}


def _load(script: str):
    """Import a family script once per process (the pool reuses its workers)."""
    if script in _MODULES:
        return _MODULES[script]
    path = Path(script)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem + "_screen", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _MODULES[script] = module
    return module


def _draws(row: Row, seeds: Sequence[np.random.SeedSequence]) -> List[Any]:
    out: List[Any] = []
    can_alarm = hasattr(signal, "SIGALRM")
    if can_alarm:
        signal.signal(signal.SIGALRM, _alarm)
    for s in seeds:
        rng = np.random.default_rng(s)
        try:
            if can_alarm:
                signal.alarm(DRAW_TIMEOUT)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = row.dgp(rng)
                if row.interval_only:
                    out.append((float(bool(row.fit(data))),))
                    continue
                vals = tuple(float(v) for v in row.fit(data))
            if not all(math.isfinite(v) for v in vals):
                raise FloatingPointError(f"non-finite result {vals}")
            out.append(vals)
        except Exception as exc:  # recorded, counted and reported per row
            out.append(f"{type(exc).__name__}: {exc}"[:300])
        finally:
            if can_alarm:
                signal.alarm(0)
    return out


def _worker(args: Tuple[str, str, int, int, int]) -> List[Any]:
    script, name, B, start, stop = args
    module = _load(script)
    (row,) = [r for r in module.ROWS if r.name == name]
    return _draws(row, row_seeds(name, B)[start:stop])


def summarise(row: Row, draws: List[Any], B: int, seconds: float) -> Dict[str, Any]:
    ok = np.array([d for d in draws if not isinstance(d, str)], dtype=float)
    errors = [d for d in draws if isinstance(d, str)]
    out: Dict[str, Any] = {
        "name": row.name,
        "estimator": row.estimator,
        "call": row.call,
        "design": row.design,
        "n": row.n,
        "B": B,
        "truth": row.truth,
        "n_failed": len(errors),
        "first_error": errors[0] if errors else "",
        "seconds": round(seconds, 1),
    }
    if len(ok) < max(10, B // 2):
        out.update(
            coverage=float("nan"),
            se_over_sd=float("nan"),
            bias_over_sd=float("nan"),
            flagged=True,
        )
        return out
    if row.interval_only:
        out.update(
            coverage=round(float(ok[:, 0].mean()), 4),
            se_over_sd=None,
            bias_over_sd=None,
            interval_only=True,
        )
        out["prefix"] = {
            "draws": PREFIX,
            "ok": int(sum(not isinstance(d, str) for d in draws[:PREFIX])),
            "sum_estimate": float(
                sum(d[0] for d in draws[:PREFIX] if not isinstance(d, str))
            ),
            "sum_se": 0.0,
        }
        out["flagged"] = bool(is_flagged(out) or len(errors) > 0.05 * B)
        return out
    est, se, lo, hi = ok.T
    sd = float(est.std(ddof=1))
    mad = float(1.4826 * np.median(np.abs(est - np.median(est))))
    out.update(
        coverage=round(float(np.mean((lo <= row.truth) & (row.truth <= hi))), 4),
        se_over_sd=round(float(se.mean() / sd), 4),
        bias_over_sd=round(float((est.mean() - row.truth) / sd), 4),
        mean_estimate=float(est.mean()),
        sd_estimate=sd,
        mean_se=float(se.mean()),
        # Tail-robust versions, for reading rows whose estimator has no moments.
        median_se_over_mad=round(float(np.median(se) / mad), 4) if mad > 0 else None,
        median_bias_over_mad=(
            round(float((np.median(est) - row.truth) / mad), 4) if mad > 0 else None
        ),
        # Wald coverage from the reported SE alone: separates a wrong SE from
        # a wrong reference distribution or interval construction.
        coverage_wald_z=round(
            float(np.mean(np.abs(est - row.truth) <= 1.959963984540054 * se)), 4
        ),
    )
    pre = [d for d in draws[:PREFIX] if not isinstance(d, str)]
    out["prefix"] = {
        "draws": PREFIX,
        "ok": len(pre),
        "sum_estimate": float(np.sum([d[0] for d in pre])),
        "sum_se": float(np.sum([d[1] for d in pre])),
    }
    out["flagged"] = bool(is_flagged(out) or len(errors) > 0.05 * B)
    return out


def run_row(
    script: str, row: Row, B: int, pool: Any = None, procs: int = MAX_PROCS
) -> Dict[str, Any]:
    """Screen one row; ``pool`` is a two-worker pool shared by the whole run."""
    t0 = time.time()
    if pool is None:
        draws = _draws(row, row_seeds(row.name, B))
    else:
        step = max(1, math.ceil(B / (procs * 4)))
        tasks = [(script, row.name, B, a, min(B, a + step)) for a in range(0, B, step)]
        draws = [d for chunk in pool.map(_worker, tasks, chunksize=1) for d in chunk]
    return summarise(row, draws, B, time.time() - t0)


def prefix(script: str, name: str) -> Dict[str, Any]:
    """Recompute the stored ``prefix`` block of one row (used by the test)."""
    module = _load(script)
    (row,) = [r for r in module.ROWS if r.name == name]
    stored_B = None
    res = Path(script).with_name(Path(script).stem + "_results.json")
    for r in json.loads(res.read_text(encoding="utf-8"))["rows"]:
        if r["name"] == name:
            stored_B = int(r["B"])
    draws = _draws(row, row_seeds(name, stored_B)[:PREFIX])
    pre = [d for d in draws if not isinstance(d, str)]
    return {
        "draws": PREFIX,
        "ok": len(pre),
        "sum_estimate": float(np.sum([d[0] for d in pre])),
        "sum_se": 0.0 if row.interval_only else float(np.sum([d[1] for d in pre])),
    }


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    """The README table of one family."""

    def num(v, fmt):
        return "n/a" if v is None else format(v, fmt)

    lines = [
        "| row | call | n | B | coverage | SE/SD | bias/SD | flag |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['name']}` | `{r['call']}` | {r['n']} | {r['B']} | "
            f"{r['coverage']:.3f} | {num(r['se_over_sd'], '.2f')} | "
            f"{num(r['bias_over_sd'], '+.2f')} | "
            f"{'**yes**' if r['flagged'] else ''} |"
        )
    return "\n".join(lines)


def main(script: str, family: str, rows: List[Row], mechanisms: Dict[str, str]) -> None:
    """Command line of a family script."""
    import statspai as sp

    ap = argparse.ArgumentParser(description=f"SE calibration screen: {family}")
    ap.add_argument(
        "--only",
        action="append",
        default=None,
        help="row name or estimator name (repeatable, substring match)",
    )
    ap.add_argument(
        "--B", type=int, default=None, help="draws (default 400, or the row's B)"
    )
    ap.add_argument(
        "--confirm",
        action="store_true",
        help=f"rerun the stored flagged rows at B = {B_CONFIRM}",
    )
    ap.add_argument("--procs", type=int, default=MAX_PROCS, choices=(1, 2))
    ap.add_argument("--dry", action="store_true", help="print, do not write the JSON")
    ap.add_argument(
        "--probe",
        action="store_true",
        help="fit one draw per row, print it with its timing, write nothing",
    )
    ap.add_argument(
        "--missing",
        action="store_true",
        help="run only the rows that have no stored result",
    )
    ap.add_argument(
        "--table",
        action="store_true",
        help="print the stored results as a Markdown table and exit",
    )
    ap.add_argument(
        "--notes",
        action="store_true",
        help="only refresh the mechanism notes in the JSON",
    )
    args = ap.parse_args()

    names = [r.name for r in rows]
    assert len(set(names)) == len(names), "row names must be unique"
    out_path = Path(script).with_name(Path(script).stem + "_results.json")
    stored: Dict[str, Dict[str, Any]] = {}
    if out_path.exists():
        for r in json.loads(out_path.read_text(encoding="utf-8"))["rows"]:
            stored[r["name"]] = r

    if args.table:
        print(markdown_table([stored[n] for n in names if n in stored]))
        return

    todo = rows
    if args.only:
        todo = [
            r for r in rows if any(o == r.estimator or o in r.name for o in args.only)
        ]
    if args.confirm:
        todo = [
            r
            for r in todo
            if stored.get(r.name, {}).get("flagged")
            and stored[r.name]["B"] < B_CONFIRM
            and r.B is None
        ]
    if args.missing:
        todo = [r for r in todo if r.name not in stored]
    if args.notes:
        todo = []
    if args.probe:
        for row in todo:
            t0 = time.time()
            rng = np.random.default_rng(row_seeds(row.name, 1)[0])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vals = row.fit(row.dgp(rng))
                msg = (
                    str(vals)
                    if row.interval_only
                    else " ".join(f"{float(v):.4f}" for v in vals)
                )
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"[:400]
            print(
                f"{row.name:42s} truth={row.truth:.4f} {time.time() - t0:6.2f}s  {msg}",
                flush=True,
            )
        return

    pool = None
    if todo and args.procs > 1:
        import multiprocessing as mp

        pool = mp.get_context("spawn").Pool(args.procs)
    for row in todo:
        B = args.B or (B_CONFIRM if args.confirm else (row.B or B_SCREEN))
        res = run_row(script, row, B, pool, args.procs)
        res["statspai_version"] = sp.__version__
        if args.confirm and row.name in stored:
            prev = stored[row.name]
            res["screen"] = {
                k: prev.get(k) for k in ("B", "coverage", "se_over_sd", "bias_over_sd")
            }
        elif row.name in stored and "screen" in stored[row.name] and B >= B_CONFIRM:
            res["screen"] = stored[row.name]["screen"]
        stored[row.name] = res
        nan = float("nan")
        ratio = res["se_over_sd"] if res["se_over_sd"] is not None else nan
        bias = res["bias_over_sd"] if res["bias_over_sd"] is not None else nan
        print(
            f"{row.name:42s} B={B:5d} cov={res['coverage']:.3f} se/sd={ratio:.3f} "
            f"bias/sd={bias:+.3f} fail={res['n_failed']:3d} "
            f"{'FLAG' if res['flagged'] else 'ok  '} {res['seconds']:.0f}s "
            f"{res['first_error'][:80]}",
            flush=True,
        )
        _write(out_path, family, script, names, stored, mechanisms, args.dry)
    if pool is not None:
        pool.close()
        pool.join()

    _write(out_path, family, script, names, stored, mechanisms, args.dry)


def _write(out_path, family, script, names, stored, mechanisms, dry=False) -> None:
    if dry:
        return
    ordered = []
    for name in names:
        if name in stored:
            r = stored[name]
            r["mechanism"] = mechanisms.get(name, "")
            ordered.append(r)
    payload = {
        "family": family,
        "script": Path(script).name,
        "global_base_seed": GLOBAL_BASE,
        "bands_at_B_400": band(B_SCREEN),
        "rows": ordered,
    }
    out_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
