"""Getis-Ord G — global and local hotspot statistics."""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from ..weights.core import W
from ._base import SpatialStatistic, permutation_pvalue


def getis_ord_g(y, w: W, permutations: int = 999, seed: Optional[int] = None) -> SpatialStatistic:
    y = np.asarray(y, dtype=float).ravel()
    if np.any(y < 0):
        raise ValueError("Getis-Ord G requires non-negative y.")
    S = w.sparse
    S0 = float(S.sum())
    n = y.size
    diag = S.diagonal()
    num = float(y @ (S @ y) - np.sum(diag * y * y))
    den = float(np.sum(y) ** 2 - np.sum(y ** 2))
    if den == 0:
        raise ValueError("Degenerate y for Getis-Ord G.")
    G = num / den

    EG = S0 / (n * (n - 1)) if n > 1 else np.nan
    sims = None; p_sim = None
    VG = np.nan; z_score = np.nan; p_norm = np.nan
    if permutations and permutations > 0:
        rng = np.random.default_rng(seed)
        sims = np.empty(permutations)
        for k in range(permutations):
            yp = rng.permutation(y)
            sims[k] = (yp @ (S @ yp) - np.sum(diag * yp * yp)) / \
                      (np.sum(yp) ** 2 - np.sum(yp ** 2))
        p_sim = permutation_pvalue(G, sims)
        VG = float(np.var(sims, ddof=1))
        if VG > 0:
            z_score = (G - EG) / np.sqrt(VG)
            p_norm = 2 * (1 - sp_stats.norm.cdf(abs(z_score)))

    return SpatialStatistic(
        name="Getis-Ord G", value=G, expectation=EG,
        variance=VG, z_score=z_score, p_norm=p_norm,
        p_sim=p_sim, simulations=sims,
    )


def getis_ord_local(y, w: W, star: bool = True, permutations: int = 999,
                    seed: Optional[int] = None):
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    S = w.sparse.toarray().copy()
    if star:
        np.fill_diagonal(S, 1.0)
    Wi = S.sum(axis=1)
    sum_y = y.sum()
    mean_y = sum_y / n
    var_y = np.var(y, ddof=0)
    num = S @ y - Wi * mean_y
    denom_core = np.maximum((n * (Wi - Wi ** 2 / n)) / (n - 1), 0)
    denom = np.sqrt(var_y * denom_core)
    denom = np.where(denom == 0, np.nan, denom)
    Gi = num / denom
    return {"Gs": Gi, "z": Gi}
