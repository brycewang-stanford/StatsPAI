"""Benchmark: sp.absorb_ols (native HDFE) vs dummy-variable OLS and
linearmodels PanelOLS.

Measures the pay-off of StatsPAI's native alternating-projection HDFE
kernel (commit 8aca56e+ numba-JIT) on high-cardinality fixed effects.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

import statspai as sp

from _utils import bench, fmt_ms, speedup_label


def _make_panel(n_units: int, n_periods: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_units * n_periods
    i_ = np.repeat(np.arange(n_units), n_periods)
    t_ = np.tile(np.arange(n_periods), n_units)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    unit_fe = rng.normal(size=n_units)[i_]
    time_fe = rng.normal(size=n_periods)[t_]
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + unit_fe + time_fe + rng.normal(size=n)
    return pd.DataFrame({'i': i_, 't': t_, 'x1': x1, 'x2': x2, 'y': y})


def run(sizes: List[dict] = None) -> Dict:
    if sizes is None:
        sizes = [
            {'n_units': 500, 'n_periods': 10},     #  5k
            {'n_units': 2_000, 'n_periods': 10},   # 20k
            {'n_units': 10_000, 'n_periods': 10},  # 100k
        ]

    try:
        from linearmodels.panel import PanelOLS
        has_lm = True
    except ImportError:
        has_lm = False

    out = []
    for cfg in sizes:
        n_u, n_t = cfg['n_units'], cfg['n_periods']
        n = n_u * n_t
        df = _make_panel(n_u, n_t)

        # StatsPAI native HDFE — pass raw arrays (low-level API)
        y_arr = df['y'].values
        X_arr = df[['x1', 'x2']].values
        fe_arr = df[['i', 't']]
        sp_result = bench(
            lambda: sp.absorb_ols(y_arr, X_arr, fe_arr),
            n_runs=3,
        )

        row = {
            'n_units': n_u,
            'n_periods': n_t,
            'n_obs': n,
            'sp_absorb_ols_s': sp_result['mean_s'],
        }

        # Compare against linearmodels PanelOLS (if available and n_units
        # small enough that set-up doesn't dominate).
        if has_lm and n < 50_000:
            lm_df = df.set_index(['i', 't'])

            def _run_lm():
                return PanelOLS.from_formula(
                    'y ~ x1 + x2 + EntityEffects + TimeEffects',
                    data=lm_df,
                ).fit(cov_type='robust')

            lm_result = bench(_run_lm, n_runs=2)
            row['linearmodels_panelols_s'] = lm_result['mean_s']
            row['speedup_vs_lm'] = speedup_label(
                sp_result['mean_s'], lm_result['mean_s'])

        out.append(row)
        lm_part = (f"| linearmodels={fmt_ms(row.get('linearmodels_panelols_s', 0))} "
                   f"| {row.get('speedup_vs_lm','')}"
                   if 'linearmodels_panelols_s' in row else '')
        print(f"  n={n:>7,} (units={n_u:>6,}, T={n_t}) "
              f"| sp.absorb_ols={fmt_ms(sp_result['mean_s']):<8} {lm_part}")

    return {
        'name': 'HDFE (two-way FE: unit × time)',
        'has_linearmodels': has_lm,
        'rows': out,
    }


if __name__ == '__main__':
    print("== HDFE benchmark ==")
    print(run())
