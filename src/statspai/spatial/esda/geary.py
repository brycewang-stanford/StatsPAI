"""Geary's C — global spatial autocorrelation."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy import stats as sp_stats

from ..weights.core import W
from ._base import SpatialStatistic, permutation_pvalue


def geary(
    y: Any,
    w: W,
    permutations: int = 999,
    seed: Optional[int] = None,
    assumption: str = "randomisation",
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
        Number of random permutations for the pseudo p-value. ``0`` skips
        the permutation test; the analytic ``variance`` / ``z_score`` /
        ``p_norm`` are reported either way.
    seed : int, optional
        Seed for the permutation RNG for reproducibility.
    assumption : {'randomisation', 'normality'}, default 'randomisation'
        Null used for the closed-form variance, following Cliff and Ord
        (1981) and matching ``spdep::geary.test``'s ``randomisation=``
        argument. The randomisation null carries the fourth-moment term
        ``k = m4 / m2^2``; the normality null drops it.

        .. versionchanged:: 1.27.0
           ``variance``, ``z_score`` and ``p_norm`` used to be ``NaN``
           whenever ``permutations=0``, even though Geary's C has a
           closed form for all of them -- ``moran`` reported its own
           analytic null in the same situation. The reported ``z_score``
           also changes sign: it is now ``(E[C] - C) / sd``, so a positive
           z means positive spatial autocorrelation, which is
           ``spdep``'s convention and the one ``sp.moran`` already used.
           Two-sided p-values are unaffected.

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
    if assumption not in ("randomisation", "normality"):
        raise ValueError(
            "assumption must be 'randomisation' or 'normality', " f"got {assumption!r}"
        )

    # Closed-form null (Cliff & Ord 1981). S1 and S2 are the standard
    # weight moments; the randomisation null additionally needs the
    # standardised fourth moment k = m4 / m2^2.
    Sd = S.toarray() if hasattr(S, "toarray") else np.asarray(S)
    S1 = 0.5 * float(np.sum((Sd + Sd.T) ** 2))
    S2 = float(np.sum((Sd.sum(axis=1) + Sd.sum(axis=0)) ** 2))
    if n > 3:
        if assumption == "normality":
            VC = ((2 * S1 + S2) * (n - 1) - 4 * S0**2) / (2 * (n + 1) * S0**2)
        else:
            m2 = float(np.sum(z**2)) / n
            m4 = float(np.sum(z**4)) / n
            kk = m4 / m2**2
            VC = (
                (n - 1) * S1 * (n**2 - 3 * n + 3 - (n - 1) * kk)
                - 0.25 * (n - 1) * S2 * (n**2 + 3 * n - 6 - (n**2 - n + 2) * kk)
                + S0**2 * (n**2 - 3 - (n - 1) ** 2 * kk)
            ) / (n * (n - 2) * (n - 3) * S0**2)
    else:  # pragma: no cover - degenerate
        VC = np.nan
    if np.isfinite(VC) and VC > 0:
        # (E - C)/sd, not (C - E)/sd: C below its expectation means POSITIVE
        # autocorrelation, so this orientation makes a positive z mean the
        # same thing it means for Moran's I.
        z_score = (EC - C) / np.sqrt(VC)
        p_norm = 2 * sp_stats.norm.sf(abs(z_score))
    else:  # pragma: no cover - degenerate
        z_score = np.nan
        p_norm = np.nan

    sims = None
    p_sim = None
    if permutations and permutations > 0:
        rng = np.random.default_rng(seed)
        sims = np.empty(permutations)
        for k in range(permutations):
            yp = rng.permutation(y)
            d = (yp[rows] - yp[cols]) ** 2
            zp = yp - yp.mean()
            sims[k] = ((n - 1) * np.sum(data * d)) / (2 * S0 * np.sum(zp**2))
        p_sim = permutation_pvalue(C, sims)

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
