"""Variance estimators for dynamic-panel GMM.

Four VCEs, matching Stata's ``xtabond`` / ``xtabond2`` menu:

===================================  =====================================
one-step, ``robust=False``           ``σ̂² (W'Z A₁ Z'W)^{-1}``
one-step, ``robust=True``            clustered sandwich on the unit
two-step, ``robust=False``           ``(W'Z A₂ Z'W)^{-1}`` — the textbook
                                     efficient-GMM VCE, known to be
                                     severely downward biased in panels
                                     with many moments
two-step, ``robust=True``            Windmeijer (2005) finite-sample
                                     correction
===================================  =====================================

References
----------
Windmeijer, F. (2005). A finite sample correction for the variance of
linear efficient two-step GMM estimators. *Journal of Econometrics*
126(1), 25-51. [@windmeijer2005finite]
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["robust_sandwich", "windmeijer_correction"]


def robust_sandwich(
    Minv: np.ndarray, WZ: np.ndarray, weight: np.ndarray, Omega: np.ndarray
) -> np.ndarray:
    """``(W'ZAZ'W)^{-1} W'ZA Ω AZ'W (W'ZAZ'W)^{-1}`` clustered on the unit."""
    bread = Minv @ WZ @ weight
    return np.asarray(bread @ Omega @ bread.T)


def windmeijer_correction(
    W: np.ndarray,
    Z: np.ndarray,
    WZ: np.ndarray,
    resid1: np.ndarray,
    resid2: np.ndarray,
    A2: np.ndarray,
    Minv2: np.ndarray,
    V1_robust: np.ndarray,
    unit_rows: Sequence[np.ndarray],
    index=None,
) -> np.ndarray:
    """Windmeijer (2005) correction for two-step robust standard errors.

    ``V_corr = V₂ + D V₂ + V₂ D' + D V₁ᵣ D'``, where ``V₂ = Minv2`` is the
    conventional two-step VCE and ``V₁ᵣ`` the one-step robust VCE.  ``D``
    holds ``∂β̂₂/∂β̂₁``, which is non-zero precisely because the efficient
    weight ``A₂ = Ω(ê₁)^{-1}`` is itself estimated from the one-step
    residuals — hence the ``∂Ω/∂β`` term below uses ``ê₁``, not ``ê₂``.

    Validated to machine precision against Stata's ``xtabond, twostep
    vce(robust)``.
    """
    from ._estimate import group_moments

    k = W.shape[1]
    g2 = Z.T @ resid2
    bread2 = Minv2 @ WZ @ A2
    tail = A2 @ g2

    # ``dOmega/dbeta_j = -(Ge' Gw_j + Gw_j' Ge)`` where ``Ge`` and ``Gw_j``
    # stack the per-group moment vectors. Writing it that way replaces
    # ``k x n_groups`` outer products (120k of them on a 20k-unit panel, and
    # 37% of total runtime) with two dense products per parameter.
    Ge = group_moments(Z, resid1, unit_rows, index=index)
    D = np.zeros((k, k))
    for j in range(k):
        Gw = group_moments(Z, W[:, j], unit_rows, index=index)
        dOmega = -(Ge.T @ Gw + Gw.T @ Ge)
        D[:, j] = -(bread2 @ (dOmega @ tail))
    return np.asarray(Minv2 + D @ Minv2 + Minv2 @ D.T + D @ V1_robust @ D.T)
