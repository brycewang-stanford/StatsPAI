"""Getis-Ord G — global and local hotspot statistics."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy import stats as sp_stats

from ..weights.core import W
from ._base import SpatialStatistic, permutation_pvalue


def getis_ord_g(
    y: Any, w: W, permutations: int = 999, seed: Optional[int] = None
) -> SpatialStatistic:
    """Global Getis-Ord G — overall concentration of high (or low) values.

    The global G statistic summarises whether high values of ``y`` are
    globally concentrated among neighbouring units. It requires non-negative
    ``y``. Larger-than-expected G indicates clustering of high values; smaller
    indicates clustering of low values.

    Parameters
    ----------
    y : array_like
        Non-negative variable of interest, length ``n``.
    w : W
        Spatial weights object defining the neighbourhood structure.
    permutations : int, default 999
        Number of random permutations for the pseudo p-value, empirical
        variance and z-score. ``0`` skips the permutation test.
    seed : int, optional
        Seed for the permutation RNG for reproducibility.

    Returns
    -------
    SpatialStatistic
        Result with ``value`` (G), ``expectation`` (``S0 / (n(n-1))``),
        ``variance``, ``z_score``, ``p_norm`` and ``p_sim`` / ``simulations``.

    Raises
    ------
    ValueError
        If any element of ``y`` is negative, or ``y`` is degenerate.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> coords = np.random.default_rng(0).random((50, 2))
    >>> w = sp.knn_weights(coords, k=5)
    >>> y = np.random.default_rng(4).random(50)  # non-negative
    >>> res = sp.getis_ord_g(y, w, seed=0)
    >>> res.value > 0
    True

    See Also
    --------
    getis_ord_local : Local Gi / Gi* hotspot statistics.

    References
    ----------
    Getis, A. and Ord, J. K. (1992). The analysis of spatial association by
    use of distance statistics. *Geographical Analysis*, 24(3), 189-206.
    [@getis1992analysis]
    """
    y = np.asarray(y, dtype=float).ravel()
    if np.any(y < 0):
        raise ValueError("Getis-Ord G requires non-negative y.")
    S = w.sparse
    S0 = float(S.sum())
    n = y.size
    diag = S.diagonal()
    num = float(y @ (S @ y) - np.sum(diag * y * y))
    den = float(np.sum(y) ** 2 - np.sum(y**2))
    if den == 0:
        raise ValueError("Degenerate y for Getis-Ord G.")
    G = num / den

    EG = S0 / (n * (n - 1)) if n > 1 else np.nan
    sims = None
    p_sim = None
    VG = np.nan
    z_score = np.nan
    p_norm = np.nan
    if permutations and permutations > 0:
        rng = np.random.default_rng(seed)
        sims = np.empty(permutations)
        for k in range(permutations):
            yp = rng.permutation(y)
            sims[k] = (yp @ (S @ yp) - np.sum(diag * yp * yp)) / (
                np.sum(yp) ** 2 - np.sum(yp**2)
            )
        p_sim = permutation_pvalue(G, sims)
        VG = float(np.var(sims, ddof=1))
        if VG > 0:
            z_score = (G - EG) / np.sqrt(VG)
            p_norm = 2 * (1 - sp_stats.norm.cdf(abs(z_score)))

    return SpatialStatistic(
        name="Getis-Ord G",
        value=G,
        expectation=EG,
        variance=VG,
        z_score=z_score,
        p_norm=p_norm,
        p_sim=p_sim,
        simulations=sims,
    )


def getis_ord_local(
    y: Any,
    w: W,
    star: bool = True,
    permutations: int = 999,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Local Getis-Ord Gi / Gi* — hotspot and coldspot detection.

    Computes a standardised local statistic for each unit measuring whether it
    is surrounded by high values (a hotspot, large positive z) or low values
    (a coldspot, large negative z). With ``star=True`` the focal unit is
    included in its own neighbourhood (Gi*); with ``star=False`` it is
    excluded (Gi).

    Parameters
    ----------
    y : array_like
        Variable of interest, length ``n``.
    w : W
        Spatial weights object; densified internally to an ``n x n`` matrix.
    star : bool, default True
        If True compute Gi* (self-included); otherwise Gi (self-excluded).
    permutations : int, default 999
        Accepted for API symmetry; the current implementation returns the
        analytic standardised statistic.
    seed : int, optional
        Seed placeholder for API symmetry.

    Returns
    -------
    dict
        ``{"Gs": ndarray, "z": ndarray}`` — the ``n`` local statistics, which
        are already standardised (so ``Gs`` and ``z`` coincide).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> coords = np.random.default_rng(0).random((40, 2))
    >>> w = sp.knn_weights(coords, k=4)
    >>> y = np.random.default_rng(5).random(40)
    >>> out = sp.getis_ord_local(y, w, star=True)
    >>> out["Gs"].shape
    (40,)

    See Also
    --------
    getis_ord_g : Global G statistic.
    moran_local : LISA, a cross-product local statistic.

    References
    ----------
    Getis, A. and Ord, J. K. (1992). The analysis of spatial association by
    use of distance statistics. *Geographical Analysis*, 24(3), 189-206.
    [@getis1992analysis]
    """
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
    denom_core = np.maximum((n * (Wi - Wi**2 / n)) / (n - 1), 0)
    denom = np.sqrt(var_y * denom_core)
    denom = np.where(denom == 0, np.nan, denom)
    Gi = num / denom
    return {"Gs": Gi, "z": Gi}
