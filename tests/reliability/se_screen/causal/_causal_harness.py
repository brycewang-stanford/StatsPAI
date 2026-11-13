"""Shared machinery for the standard-error calibration screen.

A family script (``matching.py``, ``weighting.py`` ...) declares a list of
:class:`Row`. Each row names one estimator call, a data-generating process
that satisfies the estimator's stated assumptions, and the true value of
the parameter. The harness draws ``B`` independent data sets, calls the
estimator on each, and records

* ``coverage``: share of replications whose reported interval contains
  the truth;
* ``se_over_sd``: mean reported standard error over the Monte Carlo
  standard deviation of the estimate;
* ``bias_over_sd``: (mean estimate - truth) over the same standard
  deviation.

A row is flagged when coverage leaves the band, when ``se_over_sd`` is
outside [0.88, 1.12], or when ``|bias_over_sd| > 0.25`` (bands widen for
``B < 400``, see :func:`bands`). Seeds are
``SeedSequence(crc32(row id)).spawn(B)``, so a rerun at a larger ``B``
extends the same sequence.

Three row kinds:

``ci``
    A confidence interval for a point-identified parameter.
``set``
    A confidence interval for a partially identified parameter. ``truth``
    is the identified set ``(lower, upper)``. Coverage is the smaller of
    the two endpoint coverages: a valid interval for the parameter must
    cover the parameter wherever it sits in the set, and an Imbens-Manski
    interval is at its nominal level at an endpoint of a wide set.
``prediction``
    A prediction interval. Each replication returns ``cover``, the share
    of fresh test points whose outcome fell inside the interval.

At most two worker processes are used (``spawn`` context; BLAS and numba
are pinned to one thread each).
"""

from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
import zlib  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import (  # noqa: E402
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np  # noqa: E402

MAX_PROCESSES = 2
DEFAULT_B = 400
CONFIRM_B = 1000

# Bands of the brief, for a nominal 95% interval at B >= 400.
COVERAGE_BAND_95 = (0.918, 0.985)
SE_BAND = (0.88, 1.12)
BIAS_LIMIT = 0.25
# A row whose estimator raises on more than this share of draws is flagged.
FAILURE_LIMIT = 0.05


@dataclass
class Row:
    """One estimator call to screen."""

    id: str
    estimator: str
    call: str
    n: int
    truth: Union[float, Tuple[float, float]]
    fit: Callable[[np.random.Generator], Dict[str, Any]]
    B: int = DEFAULT_B
    nominal: float = 0.95
    kind: str = "ci"
    design: str = ""
    # True when the reported standard error is conditional on the sample
    # (sample ATT / sample ATE): coverage, bias and the SD are then taken
    # against each draw's own in-sample truth, returned as ``truth_s``.
    sample_target: bool = False
    # Prediction rows only. True when the interval is conservative by
    # construction (a union bound for an individual effect), so that only
    # under-coverage is a finding.
    conservative: bool = False


def bands(nominal: float, B: int) -> Dict[str, Tuple[float, float]]:
    """Acceptance bands for a row with ``B`` replications.

    Coverage: the brief's [0.918, 0.985] for a 95% interval at ``B >= 400``;
    otherwise the 99% score (Wilson) band around the nominal level, that
    is, every coverage whose Wilson 99% interval contains the nominal
    level. ``se_over_sd`` and ``bias_over_sd`` bands are the brief's at
    ``B >= 400`` and widen with ``sqrt(400 / B)`` below, because the Monte
    Carlo error of both scales with ``1 / sqrt(B)``.
    """
    if abs(nominal - 0.95) < 1e-12 and B >= 400:
        cov = COVERAGE_BAND_95
    else:
        half = 2.5758293035489004 * math.sqrt(nominal * (1 - nominal) / B)
        cov = (nominal - half, min(1.0, nominal + half))
    widen = max(1.0, math.sqrt(400.0 / B))
    return {
        "coverage": (round(cov[0], 4), round(cov[1], 4)),
        "se_over_sd": (1 - 0.12 * widen, 1 + 0.12 * widen),
        "bias_over_sd": (-BIAS_LIMIT * widen, BIAS_LIMIT * widen),
    }


def row_seeds(row_id: str, B: int) -> List[np.random.SeedSequence]:
    base = zlib.crc32(row_id.encode("utf-8"))
    return np.random.SeedSequence(base).spawn(B)


def _load_family(path: str):
    name = "se_screen_" + Path(path).stem
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _row_by_id(module, row_id: str) -> Row:
    (row,) = [r for r in module.ROWS if r.id == row_id]
    return row


def _run_chunk(args) -> List[Dict[str, Any]]:
    path, row_id, B, indices = args
    warnings.filterwarnings("ignore")
    module = _load_family(path)
    row = _row_by_id(module, row_id)
    seeds = row_seeds(row_id, B)
    out = []
    for i in indices:
        rng = np.random.default_rng(seeds[i])
        try:
            rep = dict(row.fit(rng))
            rep["i"] = i
        except Exception as exc:  # recorded and counted, never dropped
            rep = {"i": i, "error": f"{type(exc).__name__}: {exc}"[:300]}
        out.append(rep)
    return out


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


PREFIX = 20


def prefix_sums(reps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Sums over the first ``PREFIX`` replications, for the reproduction test.

    ``reps`` must be sorted by replication index. The sum runs over the
    interval endpoints (or the coverage share of a prediction row), which
    every row kind reports.
    """
    head = [r for r in reps if r["i"] < PREFIX]
    ok = [r for r in head if "error" not in r]
    total = 0.0
    for r in ok:
        if "cover" in r:
            total += _f(r["cover"])
        else:
            total += _f(r["lo"]) + _f(r["hi"])
    return {"draws": len(head), "ok": len(ok), "sum": total}


def run_prefix(path: str, row_id: str) -> Dict[str, Any]:
    """Recompute :func:`prefix_sums` for one row, in this process."""
    module = _load_family(path)
    row = _row_by_id(module, row_id)
    reps = _run_chunk((path, row_id, row.B, list(range(PREFIX))))
    return prefix_sums(reps)


def is_flagged(rec: Dict[str, Any]) -> List[str]:
    """The flag reasons a stored record's own numbers imply."""
    band = bands(rec["nominal"], rec["B"])
    reasons: List[str] = []
    if rec["n_failed"] > FAILURE_LIMIT * rec["B"]:
        reasons.append("failures")
    if rec.get("n_nonfinite", 0) > FAILURE_LIMIT * rec["B"]:
        reasons.append("nonfinite")
    cov = rec["coverage"]
    if cov is None:
        return reasons or ["failures"]
    lo, hi = band["coverage"]
    if cov < lo:
        reasons.append("coverage_low")
    elif cov > hi and not rec.get("conservative_by_construction", False):
        reasons.append("coverage_high")
    if rec["kind"] == "ci":
        if rec["se_over_sd"] is not None:
            s_lo, s_hi = band["se_over_sd"]
            if rec["se_over_sd"] < s_lo:
                reasons.append("se_small")
            elif rec["se_over_sd"] > s_hi:
                reasons.append("se_large")
        if abs(rec["bias_over_sd"]) > band["bias_over_sd"][1]:
            reasons.append("bias")
    return reasons


def summarise(row: Row, reps: Sequence[Dict[str, Any]], seconds: float) -> Dict:
    """Turn the replications of one row into its stored record."""
    B = len(reps)
    ok = [r for r in reps if "error" not in r]
    errors = [r["error"] for r in reps if "error" in r]
    band = bands(row.nominal, B)
    rec: Dict[str, Any] = {
        "id": row.id,
        "estimator": row.estimator,
        "call": row.call,
        "kind": row.kind,
        "design": row.design,
        "n": row.n,
        "B": B,
        "nominal": row.nominal,
        "truth": list(row.truth) if isinstance(row.truth, tuple) else row.truth,
        "n_failed": len(errors),
        "first_error": errors[0] if errors else "",
        "seconds": round(seconds, 1),
        "bands": {k: [round(v[0], 4), round(v[1], 4)] for k, v in band.items()},
    }
    rec["prefix"] = prefix_sums(reps)
    reasons: List[str] = []
    if len(errors) > FAILURE_LIMIT * B:
        reasons.append("failures")
    if not ok:
        rec.update(coverage=None, se_over_sd=None, bias_over_sd=None)
        rec["flag_reasons"] = reasons or ["failures"]
        rec["flagged"] = True
        return rec

    if row.kind == "prediction":
        cover = np.array([_f(r["cover"]) for r in ok])
        rec["coverage"] = round(float(np.nanmean(cover)), 4)
        rec["se_over_sd"] = None
        rec["bias_over_sd"] = None
        if "width" in ok[0]:
            rec["mean_width"] = round(
                float(np.nanmean([_f(r["width"]) for r in ok])), 4
            )
        if "cover2" in ok[0]:
            rec["coverage_secondary"] = round(
                float(np.nanmean([_f(r["cover2"]) for r in ok])), 4
            )
        rec["conservative_by_construction"] = row.conservative
        lo, hi = band["coverage"]
        if rec["coverage"] < lo:
            reasons.append("coverage_low")
        elif rec["coverage"] > hi and not row.conservative:
            reasons.append("coverage_high")
    elif row.kind == "set":
        t_lo, t_hi = row.truth
        lo = np.array([_f(r["lo"]) for r in ok])
        hi = np.array([_f(r["hi"]) for r in ok])
        c_lo = float(np.mean((lo <= t_lo) & (t_lo <= hi)))
        c_hi = float(np.mean((lo <= t_hi) & (t_hi <= hi)))
        rec["coverage_lower_endpoint"] = round(c_lo, 4)
        rec["coverage_upper_endpoint"] = round(c_hi, 4)
        rec["coverage_whole_set"] = round(
            float(np.mean((lo <= t_lo) & (t_hi <= hi))), 4
        )
        rec["coverage"] = round(min(c_lo, c_hi), 4)
        rec["se_over_sd"] = None
        b_lo = np.array([_f(r.get("b_lo")) for r in ok])
        b_hi = np.array([_f(r.get("b_hi")) for r in ok])
        if np.isfinite(b_lo).any():
            rec["bias_over_sd_lower_bound"] = round(
                float((np.nanmean(b_lo) - t_lo) / np.nanstd(b_lo, ddof=1)), 3
            )
            rec["bias_over_sd_upper_bound"] = round(
                float((np.nanmean(b_hi) - t_hi) / np.nanstd(b_hi, ddof=1)), 3
            )
            rec["bias_over_sd"] = max(
                abs(rec["bias_over_sd_lower_bound"]),
                abs(rec["bias_over_sd_upper_bound"]),
            )
        else:
            rec["bias_over_sd"] = None
        if rec["coverage"] < band["coverage"][0]:
            reasons.append("coverage_low")
        elif rec["coverage"] > band["coverage"][1]:
            # the less covered endpoint should sit at the nominal level
            reasons.append("coverage_high")
    else:
        truth = float(row.truth)
        est = np.array([_f(r["est"]) for r in ok])
        se = np.array([_f(r.get("se")) for r in ok])
        lo = np.array([_f(r["lo"]) for r in ok])
        hi = np.array([_f(r["hi"]) for r in ok])
        bad = ~(np.isfinite(est) & np.isfinite(lo) & np.isfinite(hi))
        rec["n_nonfinite"] = int(bad.sum())
        if bad.sum() > FAILURE_LIMIT * B:
            reasons.append("nonfinite")
        est, se, lo, hi = est[~bad], se[~bad], lo[~bad], hi[~bad]
        rec["target"] = "sample" if row.sample_target else "population"
        if row.sample_target:
            # error against the draw's own in-sample parameter
            ts = np.array([_f(r["truth_s"]) for r in ok])[~bad]
            rec["coverage_population_truth"] = round(
                float(np.mean((lo <= truth) & (truth <= hi))), 4
            )
            rec["se_over_sd_population"] = round(
                float(se.mean() / np.std(est, ddof=1)), 3
            )
            lo, hi, est = lo - ts, hi - ts, est - ts
            truth = 0.0
        sd = float(np.std(est, ddof=1))
        rec["mean_estimate"] = round(float(est.mean()), 5)
        rec["sd_estimate"] = round(sd, 5)
        rec["coverage"] = round(float(np.mean((lo <= truth) & (truth <= hi))), 4)
        if np.isfinite(se).all():
            rec["mean_se"] = round(float(se.mean()), 5)
            rec["se_over_sd"] = round(float(se.mean() / sd), 3)
            # the same ratio on scales a few extreme draws cannot move
            iqr = float(np.subtract(*np.percentile(est, [75, 25])) / 1.349)
            if iqr > 0:
                rec["median_se_over_iqr_sd"] = round(float(np.median(se) / iqr), 3)
        else:
            rec["se_over_sd"] = None
        rec["bias_over_sd"] = round(float((est.mean() - truth) / sd), 3)
        # Coverage of the in-sample version of the parameter, where the
        # row supplies one (sample ATT rather than population ATT).
        if "truth_s" in ok[0] and not row.sample_target:
            ts = np.array([_f(r["truth_s"]) for r in ok])[~bad]
            rec["coverage_sample_truth"] = round(
                float(np.mean((lo <= ts) & (ts <= hi))), 4
            )
        c_lo, c_hi = band["coverage"]
        if rec["coverage"] < c_lo:
            reasons.append("coverage_low")
        elif rec["coverage"] > c_hi:
            reasons.append("coverage_high")
        if rec["se_over_sd"] is not None:
            s_lo, s_hi = band["se_over_sd"]
            if rec["se_over_sd"] < s_lo:
                reasons.append("se_small")
            elif rec["se_over_sd"] > s_hi:
                reasons.append("se_large")
        if abs(rec["bias_over_sd"]) > band["bias_over_sd"][1]:
            reasons.append("bias")
    rec["flag_reasons"] = reasons
    rec["flagged"] = bool(reasons)
    return rec


def run_row(path: str, row: Row, B: Optional[int] = None, processes: int = 2) -> Dict:
    """Simulate one row and return its record."""
    B = int(B or row.B)
    processes = max(1, min(int(processes), MAX_PROCESSES))
    chunk = max(1, min(25, B // (4 * processes) or 1))
    jobs = [
        (path, row.id, B, list(range(s, min(s + chunk, B)))) for s in range(0, B, chunk)
    ]
    t0 = time.time()
    reps: List[Dict[str, Any]] = []
    if processes == 1:
        for job in jobs:
            reps.extend(_run_chunk(job))
    else:
        with mp.get_context("spawn").Pool(processes) as pool:
            for part in pool.imap_unordered(_run_chunk, jobs):
                reps.extend(part)
    reps.sort(key=lambda r: r["i"])
    return summarise(row, reps, time.time() - t0)


def load_results(results_path: Path) -> Dict[str, Any]:
    if results_path.exists():
        return json.loads(results_path.read_text(encoding="utf-8"))
    return {"rows": []}


def write_results(
    results_path: Path,
    family: str,
    rows: Sequence[Row],
    records: Dict[str, Dict],
    mechanisms: Dict[str, str],
    history: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    import statspai as sp

    ordered = []
    for row in rows:
        rec = records.get(row.id)
        if rec is None:
            continue
        rec = dict(rec)
        rec.setdefault("statspai_version", sp.__version__)
        rec["mechanism"] = mechanisms.get(row.id, "")
        if history and row.id in history:
            # what the same row measured before a fix made because of it
            rec["before_fix"] = history[row.id]
        ordered.append(rec)
    out = {
        "study": f"se_screen/causal/{family}",
        "flag_rule": (
            "coverage outside the band, se_over_sd outside its band, or "
            "|bias_over_sd| above its limit; bands are stored per row"
        ),
        "statspai_version": sp.__version__,
        "rows": ordered,
    }
    results_path.write_text(
        json.dumps(out, indent=1, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def markdown_table(results_path: Path) -> str:
    """The family's table as it appears in ``tests/reliability/README.md``."""
    rows = load_results(results_path)["rows"]
    lines = [
        "| row | n | B | coverage | SE / SD | bias / SD | flag |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    def cell(v: Any, digits: int) -> str:
        return "" if v is None else f"{v:.{digits}f}"

    for r in rows:
        lines.append(
            f"| `{r['id']}` | {r['n']} | {r['B']} | {cell(r['coverage'], 3)} | "
            f"{cell(r['se_over_sd'], 2)} | {cell(r['bias_over_sd'], 2)} | "
            f"{', '.join(r['flag_reasons'])} |"
        )
    return "\n".join(lines)


def main(
    family_file: str,
    rows: Sequence[Row],
    mechanisms: Dict[str, str],
    history: Optional[Dict[str, Dict[str, Any]]] = None,
    argv: Optional[Sequence[str]] = None,
) -> None:
    """Command-line entry point shared by the family scripts."""
    path = str(Path(family_file).resolve())
    family = Path(path).stem
    results_path = Path(path).with_name(f"{family}_results.json")
    ap = argparse.ArgumentParser(description=f"SE calibration screen: {family}")
    ap.add_argument("--only", action="append", default=[], help="row id or prefix")
    ap.add_argument("--B", type=int, default=None, help="replications per row")
    ap.add_argument("--processes", type=int, default=MAX_PROCESSES)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--table", action="store_true", help="print the stored table")
    ap.add_argument(
        "--missing", action="store_true", help="only rows with no stored record"
    )
    ap.add_argument(
        "--confirm",
        action="store_true",
        help=f"rerun the stored flagged rows at B = {CONFIRM_B}",
    )
    ap.add_argument(
        "--notes", action="store_true", help="rewrite mechanisms only, no runs"
    )
    ns = ap.parse_args(argv)

    if ns.table:
        print(markdown_table(results_path))
        return
    stored = {r["id"]: r for r in load_results(results_path)["rows"]}
    todo = list(rows)
    if ns.only:
        # an argument that names a row selects that row only; otherwise it
        # is a prefix
        ids = {r.id for r in rows}
        todo = [
            r
            for r in todo
            if any(r.id == o or (o not in ids and r.id.startswith(o)) for o in ns.only)
        ]
    if ns.missing:
        todo = [r for r in todo if r.id not in stored]
    if ns.confirm:
        todo = [
            r
            for r in todo
            if stored.get(r.id, {}).get("flagged")
            and stored[r.id]["B"] < min(CONFIRM_B, 2.5 * r.B)
        ]
    if ns.list:
        for r in todo:
            print(f"{r.id:45s} n={r.n:<6d} {r.call}")
        return
    if ns.notes:
        write_results(results_path, family, rows, stored, mechanisms, history)
        return
    for row in todo:
        B = ns.B or (min(CONFIRM_B, int(2.5 * row.B)) if ns.confirm else row.B)
        rec = run_row(path, row, B=B, processes=ns.processes)
        import statspai as sp

        rec["statspai_version"] = sp.__version__
        stored[row.id] = rec
        write_results(results_path, family, rows, stored, mechanisms, history)
        print(
            f"{row.id:45s} B={rec['B']:<5d} cov={rec['coverage']} "
            f"se/sd={rec['se_over_sd']} bias/sd={rec['bias_over_sd']} "
            f"fail={rec['n_failed']} {rec['seconds']}s "
            f"{'FLAG ' + ','.join(rec['flag_reasons']) if rec['flagged'] else ''}",
            flush=True,
        )
        if rec["first_error"]:
            print("    first error:", rec["first_error"], flush=True)


def ci_from(result: Any) -> Dict[str, float]:
    """``est / se / lo / hi`` from a ``CausalResult``-like object."""
    lo, hi = result.ci
    return {
        "est": float(result.estimate),
        "se": float(result.se),
        "lo": float(lo),
        "hi": float(hi),
    }
