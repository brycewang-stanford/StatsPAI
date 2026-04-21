"""Scaled HDFE benchmark: sp.absorb_ols + sp.feols vs linearmodels PanelOLS.

Extends ``bench_hdfe.py`` with 100k-1M observation scales, where the
StatsPAI native alternating-projection HDFE and the pyfixest-backed
``sp.feols`` pull decisively ahead of the dummy-variable baseline.

Usage
-----

    # Run all three sizes (10k, 100k, 1M) and print wallclocks:
    python benchmarks/bench_hdfe_scaled.py

    # Quick mode (10k, 100k only — skip 1M):
    python benchmarks/bench_hdfe_scaled.py --quick

The 1M-row case takes ~5-10 s per backend on a modern laptop; keep the
``--quick`` flag for CI.  Pass ``--json-out <path>`` to dump the
result dict as JSON (parent dirs are created automatically).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import statspai as sp

# Allow "python bench_hdfe_scaled.py" from repo root or benchmarks/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import bench, fmt_ms, speedup_label  # noqa: E402


# ---------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------

def _make_two_way_panel(n_units: int, n_periods: int,
                        seed: int = 2026) -> pd.DataFrame:
    """Realistic two-way-FE DGP (unit + time effects, 2 covariates)."""
    rng = np.random.default_rng(seed)
    n = n_units * n_periods
    i_ = np.repeat(np.arange(n_units), n_periods)
    t_ = np.tile(np.arange(n_periods), n_units)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    unit_fe = rng.normal(size=n_units)[i_]
    time_fe = rng.normal(size=n_periods)[t_]
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + unit_fe + time_fe + rng.normal(size=n)
    return pd.DataFrame({
        "i": i_, "t": t_, "x1": x1, "x2": x2, "y": y,
    })


# ---------------------------------------------------------------------
# One benchmark row (per size)
# ---------------------------------------------------------------------

def _bench_size(n_units: int, n_periods: int, with_feols: bool,
                with_linearmodels: bool) -> Dict:
    n = n_units * n_periods
    df = _make_two_way_panel(n_units, n_periods)

    # sp.absorb_ols (native HDFE, Numba-JIT'd)
    y_arr = df["y"].values
    X_arr = df[["x1", "x2"]].values
    fe_df = df[["i", "t"]]
    sp_native = bench(
        lambda: sp.absorb_ols(y_arr, X_arr, fe_df),
        n_runs=3,
    )

    row = {
        "n_units": n_units,
        "n_periods": n_periods,
        "n_obs": n,
        "sp_absorb_ols_s": sp_native["mean_s"],
    }

    # sp.feols (pyfixest backend) — only if pyfixest is installed.
    if with_feols:
        try:
            feols_result = bench(
                lambda: sp.feols("y ~ x1 + x2 | i + t", data=df),
                n_runs=2,
            )
            row["sp_feols_s"] = feols_result["mean_s"]
            row["feols_vs_absorb"] = speedup_label(
                sp_native["mean_s"], feols_result["mean_s"],
            )
        except Exception as e:  # pragma: no cover — env-dependent
            row["sp_feols_error"] = f"{type(e).__name__}: {e}"

    # linearmodels PanelOLS baseline.
    if with_linearmodels and n <= 200_000:
        try:
            from linearmodels.panel import PanelOLS
            lm_df = df.set_index(["i", "t"])
            lm_result = bench(
                lambda: PanelOLS.from_formula(
                    "y ~ x1 + x2 + EntityEffects + TimeEffects",
                    data=lm_df,
                ).fit(cov_type="robust"),
                n_runs=2,
            )
            row["linearmodels_s"] = lm_result["mean_s"]
            row["speedup_vs_lm"] = speedup_label(
                sp_native["mean_s"], lm_result["mean_s"],
            )
        except ImportError:  # pragma: no cover
            pass

    return row


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def run(sizes: Optional[List[dict]] = None,
        with_feols: bool = True,
        with_linearmodels: bool = True) -> Dict:
    """Run the scaled HDFE benchmark.

    Parameters
    ----------
    sizes
        List of ``{'n_units': int, 'n_periods': int}`` dicts.  Defaults to
        three rungs: 10k / 100k / 1M observations.
    with_feols
        Include ``sp.feols`` timing (needs pyfixest installed).
    with_linearmodels
        Include ``linearmodels.panel.PanelOLS`` baseline.
    """
    if sizes is None:
        sizes = [
            {"n_units": 1_000, "n_periods": 10},    # 10k
            {"n_units": 10_000, "n_periods": 10},   # 100k
            {"n_units": 100_000, "n_periods": 10},  # 1M
        ]

    rows: List[Dict] = []
    for cfg in sizes:
        row = _bench_size(
            cfg["n_units"], cfg["n_periods"],
            with_feols=with_feols,
            with_linearmodels=with_linearmodels,
        )
        rows.append(row)

        # Pretty print.
        parts = [
            f"n={row['n_obs']:>9,}",
            f"absorb_ols={fmt_ms(row['sp_absorb_ols_s']):<9}",
        ]
        if "sp_feols_s" in row:
            parts.append(
                f"feols={fmt_ms(row['sp_feols_s']):<9} "
                f"({row['feols_vs_absorb']})"
            )
        elif "sp_feols_error" in row:
            parts.append(f"feols=ERR({row['sp_feols_error'][:30]}…)")
        if "linearmodels_s" in row:
            parts.append(
                f"lm={fmt_ms(row['linearmodels_s']):<9} "
                f"({row['speedup_vs_lm']})"
            )
        print("  " + " | ".join(parts))

    return {
        "name": "HDFE scaled (10k / 100k / 1M, two-way FE)",
        "rows": rows,
    }


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Skip the 1M-row case (10k + 100k only).")
    ap.add_argument("--no-feols", action="store_true")
    ap.add_argument("--no-lm", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="Write the result dict to this JSON file.")
    args = ap.parse_args()

    sizes = [
        {"n_units": 1_000, "n_periods": 10},
        {"n_units": 10_000, "n_periods": 10},
    ]
    if not args.quick:
        sizes.append({"n_units": 100_000, "n_periods": 10})

    print("== HDFE scaled benchmark ==")
    result = run(
        sizes=sizes,
        with_feols=not args.no_feols,
        with_linearmodels=not args.no_lm,
    )

    if args.json_out:
        # Ensure the target directory exists before writing — avoids
        # losing a 1M-row benchmark run because the output dir doesn't
        # exist yet (reviewer flagged this as a P0 foot-gun).
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":  # pragma: no cover
    _cli()
