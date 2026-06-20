"""Spatial LM diagnostics (Anselin 1988 battery).

Five tests, each distributed χ² under H0:
- LM_err     : spatial error in a non-spatial OLS
- LM_lag     : spatial lag in a non-spatial OLS
- Robust_LM_err : LM-err robust to a lag misspecification
- Robust_LM_lag : LM-lag robust to an error misspecification
- SARMA      : joint test for either lag or error (df = 2)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

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
    Wy_centered = Wy - X @ (XtX_inv @ (X.T @ Wy))

    # T = tr(W' W + W W) — computed sparsely to avoid O(n²) dense allocation
    T = float((M.T.multiply(M)).sum() + (M.multiply(M @ M.T)).sum())

    # Raw LM statistics (Anselin 1988, Anselin & Bera 1998)
    LM_err = ((e @ We) / s2) ** 2 / T

    # RS_lag: (e' W y / s2)^2 / [(WXb)'(M*WXb)/s2 + T]   with M = I - X(X'X)^-1 X'
    MW_Xb = Wy_centered  # residual-ised Wy under OLS
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
        return float(1 - sp_stats.chi2.cdf(stat, df=df))

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
) -> Tuple[float, float]:
    """Moran's I applied to regression residuals (quick LM-err companion).

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

    M = _coerce_W(W, n_expected=len(residuals), row_normalize=row_normalize)
    # Build a lightweight W-like wrapper from the sparse matrix
    res = moran(residuals, _from_sparse(M), permutations=0)
    return res.value, res.p_norm


def _from_sparse(M: Any) -> Any:
    """Wrap a CSR sparse matrix back into a W-compatible shim for ESDA helpers."""
    from ..weights.core import W as _W

    Md = M.toarray()
    n = Md.shape[0]
    neighbors = {i: np.where(Md[i] != 0)[0].tolist() for i in range(n)}
    weights = {i: Md[i, neighbors[i]].tolist() for i in range(n)}
    return _W(neighbors, weights)
