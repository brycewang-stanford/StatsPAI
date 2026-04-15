"""Binary join counts (BB / WW / BW)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..weights.core import W
from ._base import permutation_pvalue


def join_counts(y, w: W, permutations: int = 999, seed: Optional[int] = None):
    y = np.asarray(y).ravel().astype(int)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("join_counts requires binary y (values 0 or 1).")
    S = w.sparse
    rows, cols = S.nonzero()
    data = S.data
    bb = 0.5 * float(np.sum(data * ((y[rows] == 1) & (y[cols] == 1))))
    ww = 0.5 * float(np.sum(data * ((y[rows] == 0) & (y[cols] == 0))))
    bw = float(np.sum(data * (y[rows] != y[cols])))

    sims = None; p_sim = None
    if permutations and permutations > 0:
        rng = np.random.default_rng(seed)
        sims = np.empty(permutations)
        for k in range(permutations):
            yp = rng.permutation(y)
            sims[k] = 0.5 * float(np.sum(data * ((yp[rows] == 1) & (yp[cols] == 1))))
        p_sim = permutation_pvalue(bb, sims)
    return {"BB": bb, "WW": ww, "BW": bw, "p_sim_BB": p_sim, "sims_BB": sims}
