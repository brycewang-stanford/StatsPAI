"""Geary's C — global spatial autocorrelation."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy import stats as sp_stats

from ..weights.core import W
from ._base import SpatialStatistic, permutation_pvalue


def geary(
    y: Any, w: W, permutations: int = 999, seed: Optional[int] = None
) -> SpatialStatistic:
    """Global Geary's C — squared-difference measure of spatial autocorrelation.

    Geary's C is based on pairwise squared differences between neighbouring
    values, so it is more sensitive to local (short-range) autocorrelation
    than Moran's I. Values below the expectation of ``1`` indicate positive
    autocorrelation (similar neighbours); values above ``1`` indicate negative
    autocorrelation.

    Parameters
    ----------
    y : array_like
        Variable of interest, length ``n``.
    w : W
        Spatial weights object; ``w.sparse`` must have non-zero total.
    permutations : int, default 999
        Number of random permutations for the pseudo p-value (and the
        empirical variance / z-score). ``0`` skips the permutation test, in
        which case ``variance``, ``z_score`` and ``p_norm`` are ``NaN``.
    seed : int, optional
        Seed for the permutation RNG for reproducibility.

    Returns
    -------
    SpatialStatistic
        Result with ``value`` (C), ``expectation`` (``1.0``), ``variance``,
        ``z_score``, ``p_norm`` and permutation ``p_sim`` / ``simulations``.

    Raises
    ------
    ValueError
        If the weights matrix sums to zero, or ``y`` has zero variance.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> coords = np.random.default_rng(0).random((50, 2))
    >>> w = sp.knn_weights(coords, k=5)
    >>> y = np.random.default_rng(3).standard_normal(50)
    >>> res = sp.geary(y, w, seed=0)
    >>> 0.0 < res.value
    True

    See Also
    --------
    moran : Moran's I, the cross-product alternative.

    References
    ----------
    Geary, R. C. (1954). The contiguity ratio and statistical mapping.
    *The Incorporated Statistician*, 5(3), 115-146. [@geary1954contiguity]
    """
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
    den = float(np.sum(z**2))
    if den == 0:
        raise ValueError("Variable has zero variance.")
    C = ((n - 1) * num) / (2 * S0 * den)

    EC = 1.0
    sims = None
    p_sim = None
    VC = np.nan
    z_score = np.nan
    p_norm = np.nan
    if permutations and permutations > 0:
        rng = np.random.default_rng(seed)
        sims = np.empty(permutations)
        for k in range(permutations):
            yp = rng.permutation(y)
            d = (yp[rows] - yp[cols]) ** 2
            zp = yp - yp.mean()
            sims[k] = ((n - 1) * np.sum(data * d)) / (2 * S0 * np.sum(zp**2))
        p_sim = permutation_pvalue(C, sims)
        VC = float(np.var(sims, ddof=1))
        if VC > 0:
            z_score = (C - 1.0) / np.sqrt(VC)
            p_norm = 2 * (1 - sp_stats.norm.cdf(abs(z_score)))

    return SpatialStatistic(
        name="Geary's C",
        value=C,
        expectation=EC,
        variance=VC,
        z_score=z_score,
        p_norm=p_norm,
        p_sim=p_sim,
        simulations=sims,
    )
