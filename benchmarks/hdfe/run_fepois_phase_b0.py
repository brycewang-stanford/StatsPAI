"""Phase B0 benchmark — sp.fast.fepois on the medium HDFE dataset.

Measures the wall-clock impact of routing the IRLS-internal weighted
demean through the Rust **sort-aware** kernel
(``statspai_hdfe.demean_2d_weighted_sorted``, crate v0.4.0+) plus the
FE-only-plan caching in the dispatcher. Asserts median wall over 3
reps is ≤ 1.5 s on the project's medium dataset (n=1M, fe1=100k,
fe2=1k); failure is a **hard merge blocker** (see spec §5.7), and the
threshold MUST NOT be silently widened.

Phase B0's purpose is to validate, in isolation and before committing
to the larger B1 (full Rust IRLS) work, that sort-by-primary-FE
delivers the projected speedup. PASS → invest in B1. FAIL → STOP and
return to brainstorming.

Usage::

    python3 benchmarks/hdfe/run_fepois_phase_b0.py

Reads ``benchmarks/hdfe/data/medium.csv.gz`` (materialised by
``benchmarks/hdfe/datasets.py`` if missing). Writes
``benchmarks/hdfe/results/medium_statspai_phase_b0.json`` for diff
tracking against:
- Phase 0 baseline (Python np.bincount): ``medium_statspai.json`` 2.61 s
- Phase A (Rust scatter):                ``medium_statspai_phase_a.json`` 2.45 s

Spec: ``docs/superpowers/specs/2026-04-27-native-rust-irls-fepois-design.md``
+ ``docs/superpowers/plans/2026-04-27-phase-b-rust-irls.md`` Task B0.5.
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

    # Sanity check: confirm the sort-aware Rust entry point is loaded.
    has_sorted = hasattr(statspai_hdfe, "demean_2d_weighted_sorted")
    if not has_sorted:
        raise RuntimeError(
            "statspai_hdfe.demean_2d_weighted_sorted is not exported by the "
            f"loaded crate v{statspai_hdfe.__version__}. Rebuild via "
            "(cd rust/statspai_hdfe && python3 -m maturin build --release "
            "--interpreter <python3.13>) and pip-install the wheel."
        )

    # Warmup x 2 (drops singleton/separation pre-pass cost off the timed
    # run; primes the FE-only-plan cache; lets the OS settle).
    for _ in range(2):
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
        "phase": "B0",
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
        "speedup_vs_phase0": 2.61 / median if median > 0 else float("inf"),
        "vs_fixest_064s": median / 0.64 if median > 0 else float("inf"),
    }
    out_path = HERE / "results" / "medium_statspai_phase_b0.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    status = "PASS" if median <= GATE_SECS else "FAIL"
    print(f"[Phase B0] medium wall median = {median:.3f}s "
          f"(gate ≤ {GATE_SECS}s) — {status}")
    print(f"[Phase B0] all reps:       {[f'{t:.3f}' for t in timings]}")
    print(f"[Phase B0] iterations = {fit.iterations}, converged = {fit.converged}")
    print(f"[Phase B0] speedup vs Phase 0 (2.61s): {out['speedup_vs_phase0']:.2f}×")
    print(f"[Phase B0] vs fixest      (0.64s): {out['vs_fixest_064s']:.2f}×")
    print(f"[Phase B0] wrote {out_path}")

    if median > GATE_SECS:
        raise SystemExit(
            f"Phase B0 merge gate FAILED: {median:.3f}s > {GATE_SECS}s. "
            "Do not merge; record findings in benchmarks/hdfe/AUDIT.md "
            "as 'Phase B0 round 1' and surface to user before B1. Do "
            "NOT silently widen the threshold."
        )


if __name__ == "__main__":
    main()
