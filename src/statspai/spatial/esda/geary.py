"""Geary's C — global spatial autocorrelation."""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from ..weights.core import W
from ._base import SpatialStatistic, permutation_pvalue


def geary(y, w: W, permutations: int = 999, seed: Optional[int] = None) -> SpatialStatistic:
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    S = w.sparse
    S0 = float(S.sum())
    if S0 == 0:
        raise ValueError("Weights matrix has zero total.")
    rows, cols = S.nonzero()
    data = S.data
    diffs = (y[rows] - y[cols]) ** 2
    num = float(np.sum(data * diffs))
    z = y - y.mean()
    den = float(np.sum(z ** 2))
    if den == 0:
        raise ValueError("Variable has zero variance.")
    C = ((n - 1) * num) / (2 * S0 * den)

    EC = 1.0
    sims = None; p_sim = None
    VC = np.nan; z_score = np.nan; p_norm = np.nan
    if permutations and permutations > 0:
        rng = np.random.default_rng(seed)
        sims = np.empty(permutations)
        for k in range(permutations):
            yp = rng.permutation(y)
            d = (yp[rows] - yp[cols]) ** 2
            zp = yp - yp.mean()
            sims[k] = ((n - 1) * np.sum(data * d)) / (2 * S0 * np.sum(zp ** 2))
        p_sim = permutation_pvalue(C, sims)
        VC = float(np.var(sims, ddof=1))
        if VC > 0:
            z_score = (C - 1.0) / np.sqrt(VC)
            p_norm = 2 * (1 - sp_stats.norm.cdf(abs(z_score)))

    return SpatialStatistic(
        name="Geary's C", value=C, expectation=EC,
        variance=VC, z_score=z_score, p_norm=p_norm,
        p_sim=p_sim, simulations=sims,
    )
