"""Moran's I — global and local (LISA)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy import stats as sp_stats

from ..weights.core import W
from ._base import SpatialStatistic, permutation_pvalue


def _center(y: Any) -> np.ndarray:
    arr: np.ndarray = np.asarray(y, dtype=float).ravel()
    return np.asarray(arr - arr.mean(), dtype=float)


def moran(
    y: Any,
    w: W,
    permutations: int = 999,
    two_tailed: bool = True,
    seed: Optional[int] = None,
) -> SpatialStatistic:
    """Global Moran's I — test for global spatial autocorrelation.

    Moran's I measures whether values of ``y`` at nearby locations (as
    encoded by the spatial weights ``w``) are more similar (positive
    autocorrelation, clustering) or more dissimilar (negative
    autocorrelation, a checkerboard pattern) than would be expected under
    spatial randomness. Inference is provided both analytically (Cliff-Ord
    randomisation variance, giving a normal-approximation z-score) and via a
    conditional permutation test.

    Parameters
    ----------
    y : array_like
        Variable of interest, length ``n`` (one value per spatial unit).
        Flattened to 1-D internally.
    w : W
        Spatial weights object. ``w.sparse`` must be an ``n x n`` matrix whose
        total sum is non-zero. Row-standardised or binary weights both work.
    permutations : int, default 999
        Number of conditional random permutations used for the pseudo
        p-value. Set to ``0`` to skip the permutation test and rely on the
        analytic z-score only.
    two_tailed : bool, default True
        If True the analytic ``p_norm`` is two-tailed; otherwise it is the
        upper-tail (positive autocorrelation) p-value.
    seed : int, optional
        Seed for the permutation RNG (``numpy.random.default_rng``) for
        reproducible pseudo p-values.

    Returns
    -------
    SpatialStatistic
        Result object with ``value`` (Moran's I), ``expectation``
        (``-1/(n-1)``), ``variance`` (randomisation variance), ``z_score``,
        ``p_norm`` (analytic), ``p_sim`` (permutation pseudo p-value) and the
        raw ``simulations`` array.

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
    >>> y = np.random.default_rng(1).standard_normal(50)
    >>> res = sp.moran(y, w, permutations=999, seed=0)
    >>> round(res.value, 3)  # doctest: +SKIP
    -0.041

    See Also
    --------
    moran_local : Local indicators of spatial association (LISA).
    geary : Geary's C, an alternative global autocorrelation statistic.

    References
    ----------
    Moran, P. A. P. (1950). Notes on continuous stochastic phenomena.
    *Biometrika*, 37(1-2), 17-23. [@moran1950notes]
    """
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
    b2 = n * np.sum(z**4) / (np.sum(z**2) ** 2)
    A = n * ((n**2 - 3 * n + 3) * S1 - n * S2 + 3 * S0**2)
    B = b2 * ((n**2 - n) * S1 - 2 * n * S2 + 6 * S0**2)
    C = (n - 1) * (n - 2) * (n - 3) * S0**2
    if C == 0:
        VI = np.nan
        z_score = np.nan
        p_norm = np.nan
    else:
        VI = (A - B) / C - EI**2
        VI = max(VI, 1e-12)
        z_score = (I - EI) / np.sqrt(VI)
        p_norm = (
            (2 * (1 - sp_stats.norm.cdf(abs(z_score))))
            if two_tailed
            else 1 - sp_stats.norm.cdf(z_score)
        )

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
        name="Moran's I",
        value=I,
        expectation=EI,
        variance=VI,
        z_score=z_score,
        p_norm=p_norm,
        p_sim=p_sim,
        simulations=sims,
    )


def moran_local(
    y: Any, w: W, permutations: int = 999, seed: Optional[int] = None
) -> Dict[str, Any]:
    """Local Moran's I (LISA) — per-location spatial association.

    Decomposes global Moran's I into a contribution ``I_i`` for each spatial
    unit, identifying local clusters (high-high / low-low) and spatial
    outliers (high-low / low-high). Each ``I_i`` gets a conditional
    permutation pseudo p-value.

    Parameters
    ----------
    y : array_like
        Variable of interest, length ``n``.
    w : W
        Spatial weights object; ``w.sparse`` is the ``n x n`` weights matrix.
    permutations : int, default 999
        Number of conditional permutations for the per-location pseudo
        p-values. ``0`` skips the permutation test.
    seed : int, optional
        Seed for the permutation RNG for reproducibility.

    Returns
    -------
    dict
        ``{"Is": ndarray, "p_sim": ndarray | None, "simulations": ndarray | None}``
        where ``Is`` holds the ``n`` local statistics and ``p_sim`` the
        matching pseudo p-values.

    Raises
    ------
    ValueError
        If ``y`` has zero variance.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> coords = np.random.default_rng(0).random((40, 2))
    >>> w = sp.knn_weights(coords, k=4)
    >>> y = np.random.default_rng(2).standard_normal(40)
    >>> out = sp.moran_local(y, w, permutations=499, seed=0)
    >>> out["Is"].shape
    (40,)

    See Also
    --------
    moran : Global Moran's I.

    References
    ----------
    Anselin, L. (1995). Local indicators of spatial association---LISA.
    *Geographical Analysis*, 27(2), 93-115. [@anselin1995local]
    """
    z = _center(y)
    n = z.size
    m2 = np.sum(z**2) / n
    if m2 == 0:
        raise ValueError("Variable has zero variance.")
    S = w.sparse
    Ii = z * (S @ z) / m2
    sims = None
    p_sim = None
    if permutations and permutations > 0:
        rng = np.random.default_rng(seed)
        sims = np.empty((permutations, n))
        for k in range(permutations):
            zp = rng.permutation(z)
            sims[k] = z * (S @ zp) / m2
        p_sim = np.array([permutation_pvalue(Ii[i], sims[:, i]) for i in range(n)])
    return {"Is": Ii, "p_sim": p_sim, "simulations": sims}
