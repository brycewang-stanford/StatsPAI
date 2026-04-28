"""Phase B1 benchmark — sp.fast.fepois on the medium HDFE dataset.

Measures the wall-clock impact of the native Rust IRLS state machine
(``statspai_hdfe.fepois_irls``, crate v0.5.0+) — single PyO3 call per
fepois eliminates the 12 FFI round-trips per IRLS iter that B0.4 / B0.5
still had. Asserts median wall over 3 reps is ≤ 0.95 s on the project's
medium dataset (n=1M, fe1=100k, fe2=1k); failure is a **hard merge
blocker** (see spec §6.4) and the threshold MUST NOT be silently widened.

Phase B1's goal is to reach ≤ 1.5× of fixest::fepois (0.64 s) with the
full native Rust IRLS port. Intermediate trail:
  Phase 0 (Python np.bincount):                   2.61 s   4.08× fixest
  Phase A (Rust scatter, no cache):               2.45 s   3.83×
  Phase B0 (Rust sequential + dispatcher cache):  1.44 s   2.25×
  Phase B1 (native Rust IRLS, single PyO3 call):  ≤ 0.95 s ≤ 1.5× target

Usage::

    python3 benchmarks/hdfe/run_fepois_phase_b.py

Reads ``benchmarks/hdfe/data/medium.csv.gz``. Writes
``benchmarks/hdfe/results/medium_statspai_phase_b.json``.

Spec: ``docs/superpowers/specs/2026-04-27-native-rust-irls-fepois-design.md`` §6.4
+ ``docs/superpowers/plans/2026-04-27-phase-b-rust-irls.md`` Task B1.8.
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

GATE_SECS = 0.95


def main() -> None:
    csv = HERE / "data" / "medium.csv.gz"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found — run benchmarks/hdfe/datasets.py first."
        )
    dtypes = {"y": np.int64, "x1": np.float64, "x2": np.float64,
              "fe1": np.int32, "fe2": np.int32}
    df = pd.read_csv(csv, dtype=dtypes)

    # Sanity check: confirm the native Rust IRLS entry point is loaded.
    has_native = hasattr(statspai_hdfe, "fepois_irls")
    if not has_native:
        raise RuntimeError(
            "statspai_hdfe.fepois_irls is not exported by the loaded crate "
            f"v{statspai_hdfe.__version__}. Rebuild via "
            "(cd rust/statspai_hdfe && python3 -m maturin build --release "
            "--interpreter <python3.13>) and pip-install the wheel."
        )

    # Warmup x 2 (drops first-call cost and primes the dispatcher cache).
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
        "phase": "B1",
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
        "speedup_vs_phase_b0": 1.441 / median if median > 0 else float("inf"),
        "vs_fixest_064s": median / 0.64 if median > 0 else float("inf"),
    }
    out_path = HERE / "results" / "medium_statspai_phase_b.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    status = "PASS" if median <= GATE_SECS else "FAIL"
    print(f"[Phase B1] medium wall median = {median:.3f}s "
          f"(gate ≤ {GATE_SECS}s) — {status}")
    print(f"[Phase B1] all reps:        {[f'{t:.3f}' for t in timings]}")
    print(f"[Phase B1] iterations = {fit.iterations}, converged = {fit.converged}")
    print(f"[Phase B1] vs Phase 0  (2.61 s): {out['speedup_vs_phase0']:.2f}× speedup")
    print(f"[Phase B1] vs Phase B0 (1.441 s): {out['speedup_vs_phase_b0']:.2f}× speedup")
    print(f"[Phase B1] vs fixest   (0.64 s):  {out['vs_fixest_064s']:.2f}×")
    print(f"[Phase B1] wrote {out_path}")

    if median > GATE_SECS:
        raise SystemExit(
            f"Phase B1 merge gate FAILED: {median:.3f}s > {GATE_SECS}s. "
            "Do not merge; record findings in benchmarks/hdfe/AUDIT.md "
            "as 'Phase B1 round 1' and surface to user. Do NOT silently "
            "widen the threshold."
        )


if __name__ == "__main__":
    main()
