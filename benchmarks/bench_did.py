"""Benchmark: sp.callaway_santanna and sp.wooldridge_did scaling."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

import statspai as sp

from _utils import bench, fmt_ms


def _make_staggered(n_units: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cohorts = [3, 5, 7, 0]
    rows = []
    for i in range(n_units):
        g = cohorts[i % 4]
        for t in range(1, 9):
            post = 1 if (g > 0 and t >= g) else 0
            y = 0.2 * t + 1.5 * post + rng.normal(scale=0.8)
            rows.append({'i': i, 't': t, 'g': g, 'y': y})
    return pd.DataFrame(rows)


def run(sizes: List[int] = (200, 1_000, 5_000)) -> Dict:
    out = []
    for n_u in sizes:
        df = _make_staggered(n_u)
        cs = bench(
            lambda: sp.callaway_santanna(df, y='y', g='g', t='t', i='i',
                                          estimator='reg'),
            n_runs=2,
        )
        wool = bench(
            lambda: sp.wooldridge_did(df, y='y', group='i', time='t',
                                       first_treat='g'),
            n_runs=2,
        )
        row = {
            'n_units': n_u,
            'n_obs': n_u * 8,
            'cs2021_s': cs['mean_s'],
            'wooldridge_s': wool['mean_s'],
        }
        out.append(row)
        print(f"  n={n_u:>5,} units ({n_u*8:>6,} obs) | "
              f"CS={fmt_ms(cs['mean_s']):<8} | "
              f"Wooldridge={fmt_ms(wool['mean_s']):<8}")
    return {
        'name': 'Staggered DID (4 cohorts, 8 periods)',
        'rows': out,
    }


if __name__ == '__main__':
    print("== DID benchmark ==")
    print(run())
