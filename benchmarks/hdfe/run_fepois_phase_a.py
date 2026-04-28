"""Phase A benchmark — sp.fast.fepois on the medium HDFE dataset.

Measures the wall-clock impact of routing the IRLS-internal weighted
demean through the Rust kernel (``statspai_hdfe.demean_2d_weighted``,
crate v0.3.0+). Asserts median wall over 3 reps is ≤ 1.5 s on the
project's medium dataset (n=1M, fe1=100k, fe2=1k); failure is a
**hard merge blocker** (see spec §5.7), and the threshold MUST NOT be
silently widened.

Usage::

    python3 benchmarks/hdfe/run_fepois_phase_a.py

Reads ``benchmarks/hdfe/data/medium.csv.gz`` (materialised by
``benchmarks/hdfe/datasets.py`` if missing). Writes
``benchmarks/hdfe/results/medium_statspai_phase_a.json`` for diff
tracking against the Phase 0 baseline (already committed at
``results/medium_statspai.json``: 2.61 s, 6 iters).

Spec: ``docs/superpowers/specs/2026-04-27-native-rust-irls-fepois-design.md`` §5.7.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import statspai as sp
import statspai_hdfe

HERE = Path(__file__).resolve().parent

GATE_SECS = 1.5


def main() -> None:
    csv = HERE / "data" / "medium.csv.gz"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found — run benchmarks/hdfe/datasets.py first."
        )
    dtypes = {"y": np.int64, "x1": np.float64, "x2": np.float64,
              "fe1": np.int32, "fe2": np.int32}
    df = pd.read_csv(csv, dtype=dtypes)

    # Warmup pass (drops singleton/separation pre-pass cost off the timed run
    # and triggers any first-call JIT / mmap that would otherwise skew rep 1).
    _ = sp.fast.fepois("y ~ x1 + x2 | fe1 + fe2", data=df, vcov="iid")

    n_repeats = 3
    timings: list[float] = []
    fit = None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fit = sp.fast.fepois("y ~ x1 + x2 | fe1 + fe2", data=df, vcov="iid")
        timings.append(time.perf_counter() - t0)
    median = float(np.median(timings))

    out = {
        "phase": "A",
        "rust_crate_version": statspai_hdfe.__version__,
        "statspai_version": sp.__version__,
        "dataset": "medium",
        "n": int(df.shape[0]),
        "n_repeats": n_repeats,
        "wall_seconds": timings,
        "wall_seconds_median": median,
        "iterations": int(fit.iterations) if fit is not None else None,
        "converged": bool(fit.converged) if fit is not None else None,
        "gate_seconds": GATE_SECS,
        "passes_gate": median <= GATE_SECS,
    }
    out_path = HERE / "results" / "medium_statspai_phase_a.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    status = "PASS" if median <= GATE_SECS else "FAIL"
    print(f"[Phase A] medium wall median = {median:.3f}s "
          f"(gate ≤ {GATE_SECS}s) — {status}")
    print(f"[Phase A] all reps: {[f'{t:.3f}' for t in timings]}")
    print(f"[Phase A] iterations = {fit.iterations}, converged = {fit.converged}")
    print(f"[Phase A] wrote {out_path}")

    if median > GATE_SECS:
        raise SystemExit(
            f"Phase A merge gate FAILED: {median:.3f}s > {GATE_SECS}s. "
            "Do not merge; record findings in benchmarks/hdfe/AUDIT.md "
            "and return to brainstorming. Do NOT silently widen the threshold."
        )


if __name__ == "__main__":
    main()
