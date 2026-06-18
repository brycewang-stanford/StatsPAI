"""Binary join counts (BB / WW / BW)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..weights.core import W
from ._base import permutation_pvalue


def join_counts(
    y: Any, w: W, permutations: int = 999, seed: Optional[int] = None
) -> Dict[str, Any]:
    """Join-count statistics for a binary spatial variable.

    For a 0/1 ("white"/"black") variable defined over a spatial network,
    counts the number of like-coloured joins (BB, WW) and unlike-coloured
    joins (BW), and tests the BB count against spatial randomness via
    permutation. Excess BB joins indicate positive spatial autocorrelation of
    the "black" (1) category.

    Parameters
    ----------
    y : array_like
        Binary variable, values in ``{0, 1}``, length ``n``.
    w : W
        Spatial weights object defining the join structure.
    permutations : int, default 999
        Number of random permutations for the BB pseudo p-value. ``0`` skips
        the test.
    seed : int, optional
        Seed for the permutation RNG for reproducibility.

    Returns
    -------
    dict
        ``{"BB": float, "WW": float, "BW": float, "p_sim_BB": float | None,
        "sims_BB": ndarray | None}`` — the join counts plus the BB pseudo
        p-value and its permutation distribution.

    Raises
    ------
    ValueError
        If ``y`` contains values other than 0 and 1.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> coords = np.random.default_rng(0).random((40, 2))
    >>> w = sp.knn_weights(coords, k=4)
    >>> y = (np.random.default_rng(6).random(40) > 0.5).astype(int)
    >>> jc = sp.join_counts(y, w, seed=0)
    >>> {"BB", "WW", "BW"} <= set(jc)
    True
    """
    y = np.asarray(y).ravel().astype(int)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("join_counts requires binary y (values 0 or 1).")
    S = w.sparse
    rows, cols = S.nonzero()
    data = S.data
    bb = 0.5 * float(np.sum(data * ((y[rows] == 1) & (y[cols] == 1))))
    ww = 0.5 * float(np.sum(data * ((y[rows] == 0) & (y[cols] == 0))))
    bw = float(np.sum(data * (y[rows] != y[cols])))

    sims = None
    p_sim = None
    if permutations and permutations > 0:
        rng = np.random.default_rng(seed)
        sims = np.empty(permutations)
        for k in range(permutations):
            yp = rng.permutation(y)
            sims[k] = 0.5 * float(np.sum(data * ((yp[rows] == 1) & (yp[cols] == 1))))
        p_sim = permutation_pvalue(bb, sims)
    return {"BB": bb, "WW": ww, "BW": bw, "p_sim_BB": p_sim, "sims_BB": sims}
