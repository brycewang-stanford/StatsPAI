"""Benchmark: sp.regress vs statsmodels OLS at multiple sample sizes."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

import statspai as sp

from _utils import bench, fmt_ms, speedup_label


def _make_data(n: int, p: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = X @ beta + rng.normal(size=n)
    cols = {f'x{i}': X[:, i] for i in range(p)}
    cols['y'] = y
    return pd.DataFrame(cols)


def run(sizes: List[int] = (1_000, 10_000, 100_000)) -> Dict:
    """Run OLS benchmark across sample sizes.

    Returns a dict with one entry per sample size.
    """
    try:
        import statsmodels.api as smapi
        has_sm = True
    except ImportError:
        has_sm = False

    p = 5
    rhs = ' + '.join(f'x{i}' for i in range(p))
    formula = f'y ~ {rhs}'

    out = []
    for n in sizes:
        df = _make_data(n, p=p)
        sp_result = bench(lambda: sp.regress(formula, data=df,
                                              robust='hc1'))
        row = {'n': n, 'sp_regress_s': sp_result['mean_s']}

        if has_sm:
            X = np.column_stack([np.ones(n)] + [df[f'x{i}'].values
                                                for i in range(p)])
            y = df['y'].values
            sm_result = bench(
                lambda: smapi.OLS(y, X).fit(cov_type='HC1')
            )
            row['statsmodels_s'] = sm_result['mean_s']
            row['speedup_vs_sm'] = speedup_label(
                sp_result['mean_s'], sm_result['mean_s'])

        out.append(row)
        print(f"  n={n:>7,} | sp.regress={fmt_ms(sp_result['mean_s']):<8} "
              f"{'| sm.OLS='+fmt_ms(row.get('statsmodels_s',0)) if has_sm else ''} "
              f"{'| '+row.get('speedup_vs_sm','') if has_sm else ''}")

    return {
        'name': 'OLS regression (5 covariates, HC1 SE)',
        'sizes': list(sizes),
        'has_statsmodels': has_sm,
        'rows': out,
    }


if __name__ == '__main__':
    print("== OLS regression benchmark ==")
    print(run())
