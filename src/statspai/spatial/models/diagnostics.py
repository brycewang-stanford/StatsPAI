"""Spatial LM diagnostics (Anselin 1988 battery).

Five tests, each distributed χ² under H0:
- LM_err     : spatial error in a non-spatial OLS
- LM_lag     : spatial lag in a non-spatial OLS
- Robust_LM_err : LM-err robust to a lag misspecification
- Robust_LM_lag : LM-lag robust to an error misspecification
- SARMA      : joint test for either lag or error (df = 2)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .ml import _coerce_W, _parse_formula


def lm_tests(
    formula: str,
    data: pd.DataFrame,
    W: Any,
    row_normalize: bool = True,
) -> Dict[str, Tuple[float, float]]:
    """Anselin (1988) Lagrange multiplier battery.

    Parameters
    ----------
    formula : str
        ``"y ~ x1 + x2"`` style formula (constant added automatically).
    data : pd.DataFrame
    W : ndarray, scipy.sparse, or ``statspai.spatial.weights.W``
    row_normalize : bool, default True

    Returns
    -------
    dict
        ``{name: (statistic, p_value)}`` for the five tests.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 80
    >>> coords = rng.uniform(size=(n, 2))
    >>> w = sp.knn_weights(coords, k=5)
    >>> x1 = rng.normal(size=n)
    >>> y = 1.0 + 0.5 * x1 + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({"y": y, "x1": x1})
    >>> out = sp.lm_tests("y ~ x1", df, w)
    >>> sorted(out.keys())
    ['LM_err', 'LM_lag', 'Robust_LM_err', 'Robust_LM_lag', 'SARMA']
    """
    y, X, _dep, _indep = _parse_formula(formula, data)
    M = _coerce_W(W, n_expected=len(y), row_normalize=row_normalize)
    n = len(y)

    # OLS residuals and scale
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    s2 = float(e @ e) / n

    We = M @ e
    Wy = M @ y

    # T = tr(W'W + W W) = sum_ij w_ij^2 + sum_ij w_ij w_ji.
    #
    # This line used to read
    #     (M.T.multiply(M)).sum() + (M.multiply(M @ M.T)).sum()
    # whose first term is tr(WW) and whose second term is
    # sum_ij w_ij (W W')_ij -- not tr(W'W). The comment above it already
    # stated the right formula. On a row-standardised rook lattice the old
    # expression returned 28.222 against the correct 56.889, so `LM_err`
    # came out doubled.
    T = float((M.multiply(M)).sum() + (M.T.multiply(M)).sum())

    # Raw LM statistics (Anselin 1988, Anselin & Bera 1998)
    LM_err = ((e @ We) / s2) ** 2 / T

    # RS_lag: (e' W y / s2)^2 / [(WXb)'(M*WXb)/s2 + T]   with M = I - X(X'X)^-1 X'
    #
    # WXb, not Wy. Wy = W X beta_hat + W e, so using it mixes the residual's
    # own spatial structure into the denominator of every lag statistic --
    # which is exactly what the statistic is testing for. Same story as the
    # T term: the comment named the right object and the code substituted a
    # different one.
    WXb = M @ (X @ beta)
    MW_Xb = WXb - X @ (XtX_inv @ (X.T @ WXb))
    J = float(MW_Xb @ MW_Xb) / s2 + T
    LM_lag = ((e @ Wy) / s2) ** 2 / J if J > 0 else np.nan

    # Robust forms (see Anselin et al. 1996):
    # RLM_err = (e'We/s2 - T J^{-1} (e'Wy/s2))^2  /  [T (1 - T J^{-1})]
    ratio = T / J if J > 0 else 0.0
    denom_re = T * max(1.0 - ratio, 1e-12)
    RLM_err = ((e @ We) / s2 - ratio * (e @ Wy) / s2) ** 2 / denom_re

    # RLM_lag = (e'Wy/s2 - e'We/s2)^2 / (J - T)
    denom_rl = max(J - T, 1e-12)
    RLM_lag = ((e @ Wy) / s2 - (e @ We) / s2) ** 2 / denom_rl

    SARMA = LM_err + RLM_lag

    def pv(stat: float, df: int = 1) -> float:
        if not np.isfinite(stat) or stat < 0:
            return np.nan
        return float(sp_stats.chi2.sf(stat, df=df))

    return {
        "LM_err": (float(LM_err), pv(LM_err)),
        "LM_lag": (float(LM_lag), pv(LM_lag)),
        "Robust_LM_err": (float(RLM_err), pv(RLM_err)),
        "Robust_LM_lag": (float(RLM_lag), pv(RLM_lag)),
        "SARMA": (float(SARMA), pv(SARMA, df=2)),
    }


def moran_residuals(
    residuals: np.ndarray,
    W: Any,
    row_normalize: bool = True,
    X: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Moran's I applied to regression residuals (quick LM-err companion).

    Parameters
    ----------
    residuals : ndarray
    W : ndarray, scipy.sparse, or ``statspai.spatial.weights.W``
    row_normalize : bool, default True
    X : ndarray, optional
        The regression's design matrix, **including its constant**. When
        supplied, the p-value uses Cliff and Ord's null for *regression
        residuals* -- the one ``spdep::lm.morantest`` uses -- which depends
        on ``X`` through the hat matrix. Without it the function falls back
        to the raw-variable null, which is not the residuals' null.

        .. versionchanged:: 1.27.0
           The statistic was always exact (5e-16 against ``lm.morantest``).
           The p-value was computed from the observed-variable null, which
           is the wrong reference distribution for a projection of ``y``;
           on a 10x10 lattice it read 3.64e-06 where ``lm.morantest``
           reports 1.56e-06 (two-sided 3.13e-06). Pass ``X`` for the
           correct null.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> coords = rng.uniform(size=(80, 2))
    >>> w = sp.knn_weights(coords, k=5)
    >>> resid = rng.normal(size=80)
    >>> I, p = sp.moran_residuals(resid, w)
    >>> bool(np.isfinite(I) and np.isfinite(p))
    True
    """
    from ..esda.moran import moran

    u = np.asarray(residuals, dtype=float).ravel()
    M = _coerce_W(W, n_expected=len(u), row_normalize=row_normalize)
    if X is None:
        res = moran(u, _from_sparse(M), permutations=0)
        return res.value, res.p_norm

    # Cliff & Ord's regression-residual null, as spdep::lm.morantest builds
    # it: symmetrise the weights, then correct E[I] and Var[I] for the
    # projection that produced the residuals.
    Wd = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    U = 0.5 * (Wd + Wd.T)
    Xd = np.asarray(X, dtype=float)
    n = u.size
    p = Xd.shape[1]
    S0 = float(U.sum())
    S1 = 2.0 * float(np.sum(U * U))
    I = (n / S0) * float(u @ (U @ u)) / float(u @ u)
    XtXinv = np.linalg.inv(Xd.T @ Xd)
    Z = U @ Xd
    C1 = Xd.T @ Z
    C3 = XtXinv @ C1
    trA = float(np.trace(C3))
    trA2 = float(np.trace(C3 @ C3))
    trB = float(np.trace(4.0 * (XtXinv @ (Z.T @ Z))))
    EI = -(n * trA) / ((n - p) * S0)
    VI = (n * n / ((S0 * S0) * (n - p) * (n - p + 2))) * (
        S1 + 2 * trA2 - trB - (2 * trA**2) / (n - p)
    )
    if not np.isfinite(VI) or VI <= 0:  # pragma: no cover - degenerate
        return I, float("nan")
    zi = (I - EI) / np.sqrt(VI)
    return I, float(2.0 * sp_stats.norm.sf(abs(zi)))


def _from_sparse(M: Any) -> Any:
    """Wrap a CSR sparse matrix back into a W-compatible shim for ESDA helpers."""
    from ..weights.core import W as _W

    Md = M.toarray()
    n = Md.shape[0]
    neighbors = {i: np.where(Md[i] != 0)[0].tolist() for i in range(n)}
    weights = {i: Md[i, neighbors[i]].tolist() for i in range(n)}
    return _W(neighbors, weights)
