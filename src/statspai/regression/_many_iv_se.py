"""Many-instrument standard errors for k-class estimators (LIML, Fuller).

[@hansen2008estimation], section 2 (extending [@bekker1994alternative]): for
``delta = (X'PX - a X'X)^-1 (X'Py - a X'y)`` with ``P`` the projection on
all instruments ``Z`` (``T x K``, included exogenous regressors among them)
and ``X`` all ``G`` right-hand-side variables,

    u = y - X d,  s2 = u'u / (T - G),  a~ = u'Pu / u'u,
    X~ = X - u (u'X) / u'u,  V^ = (I - P) X~,  Y^ = P X,
    p_tt = diag(P),  kappa = sum p_tt^2 / K,  tau = K / T,
    H = X'PX - a~ X'X,
    S_B = s2 [(1 - a~)^2 X~'P X~ + a~^2 X~'(I - P) X~],
    A = sum_t (p_tt - tau) Y^_t [sum_t u_t^2 V^_t / T]',
    B = K (kappa - tau) sum_t (u_t^2 - s2) V^_t V^_t' / [T (1 - 2 tau + kappa tau)],
    Var(d) = H^-1 (S_B + A + A' + B) H^-1.

``H^-1 S_B H^-1`` alone is Bekker's (1994) variance; ``A`` and ``B`` carry
the third and fourth moments of non-normal errors. ``p_tt`` is the
leverage of the **instruments**. R ``ivmodel`` (``manyweakSE=TRUE``) takes
it from the projection on the regressors ``[D, X]`` instead (shown on its
output in ``tests/test_liml_many_weak_se.py``); that is not the paper's
quantity and is not copied.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def hhn_vcov(
    y: np.ndarray,
    X: np.ndarray,
    delta: np.ndarray,
    Q: np.ndarray,
    proj: Callable[[np.ndarray], np.ndarray],
    *,
    leverage: Optional[np.ndarray] = None,
    s2_df: Optional[int] = None,
) -> np.ndarray:
    """HHN (2008) variance of a k-class estimate ``delta``.

    ``Q`` is an orthonormal basis of the instrument space and ``proj`` the
    projection on it (applied, never formed). ``leverage`` / ``s2_df``
    replace ``p_tt`` and ``T - G``; they exist so the tests can show what
    another implementation computes, and the estimators never pass them.
    """
    T, G = X.shape
    K = Q.shape[1]
    u = y - X @ delta
    uu = float(u @ u)
    s2 = uu / (T - G if s2_df is None else s2_df)
    a = float(u @ proj(u)) / uu
    Xt = X - np.outer(u, u @ X) / uu
    PXt = proj(Xt)
    Vh = Xt - PXt
    PX = proj(X)
    H = X.T @ PX - a * (X.T @ X)
    S = s2 * ((1.0 - a) ** 2 * (Xt.T @ PXt) + a**2 * (Xt.T @ Vh))
    p = np.einsum("ij,ij->i", Q, Q) if leverage is None else leverage
    tau = K / T
    kap = float(p @ p) / K
    A = np.outer((p - tau) @ PX, (u**2) @ Vh / T)
    B = (
        K
        * (kap - tau)
        * (((u**2 - s2)[:, None] * Vh).T @ Vh)
        / (T * (1.0 - 2.0 * tau + kap * tau))
    )
    Hinv = np.linalg.inv(H)
    V: np.ndarray = Hinv @ (S + A + A.T + B) @ Hinv
    _ret: np.ndarray = (V + V.T) / 2.0
    return _ret


__all__ = ["hhn_vcov"]
