"""Moran's I — global and local (LISA)."""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from ..weights.core import W
from ._base import SpatialStatistic, permutation_pvalue


def _center(y):
    y = np.asarray(y, dtype=float).ravel()
    return y - y.mean()


def moran(y, w: W, permutations: int = 999, two_tailed: bool = True,
          seed: Optional[int] = None) -> SpatialStatistic:
    z = _center(y)
    n = z.size
    S = w.sparse
    S0 = float(S.sum())
    if S0 == 0:
        raise ValueError("Weights matrix has zero total — cannot compute Moran's I.")
    wz = S @ z
    num = float(z @ wz)
    den = float(z @ z)
    if den == 0:
        raise ValueError("Variable has zero variance.")
    I = (n / S0) * (num / den)

    EI = -1.0 / (n - 1)
    # Cliff-Ord randomisation variance
    S_plus = S + S.T
    S1 = 0.5 * float(S_plus.multiply(S_plus).sum())
    row_sum = np.asarray(S.sum(axis=1)).ravel()
    col_sum = np.asarray(S.sum(axis=0)).ravel()
    S2 = float(np.sum((row_sum + col_sum) ** 2))
    b2 = n * np.sum(z ** 4) / (np.sum(z ** 2) ** 2)
    A = n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
    B = b2 * ((n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)
    C = (n - 1) * (n - 2) * (n - 3) * S0 ** 2
    if C == 0:
        VI = np.nan; z_score = np.nan; p_norm = np.nan
    else:
        VI = (A - B) / C - EI ** 2
        VI = max(VI, 1e-12)
        z_score = (I - EI) / np.sqrt(VI)
        p_norm = (2 * (1 - sp_stats.norm.cdf(abs(z_score)))) if two_tailed \
            else 1 - sp_stats.norm.cdf(z_score)

    sims = None
    p_sim = None
    if permutations and permutations > 0:
        rng = np.random.default_rng(seed)
        sims = np.empty(permutations)
        for k in range(permutations):
            zp = rng.permutation(z)
            sims[k] = (n / S0) * ((zp @ (S @ zp)) / (zp @ zp))
        p_sim = permutation_pvalue(I, sims)

    return SpatialStatistic(
        name="Moran's I", value=I, expectation=EI, variance=VI,
        z_score=z_score, p_norm=p_norm, p_sim=p_sim, simulations=sims,
    )


def moran_local(y, w: W, permutations: int = 999, seed: Optional[int] = None):
    z = _center(y)
    n = z.size
    m2 = np.sum(z ** 2) / n
    if m2 == 0:
        raise ValueError("Variable has zero variance.")
    S = w.sparse
    Ii = z * (S @ z) / m2
    sims = None; p_sim = None
    if permutations and permutations > 0:
        rng = np.random.default_rng(seed)
        sims = np.empty((permutations, n))
        for k in range(permutations):
            zp = rng.permutation(z)
            sims[k] = z * (S @ zp) / m2
        p_sim = np.array([
            permutation_pvalue(Ii[i], sims[:, i]) for i in range(n)
        ])
    return {"Is": Ii, "p_sim": p_sim, "simulations": sims}
