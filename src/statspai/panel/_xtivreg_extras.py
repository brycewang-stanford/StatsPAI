"""First differences, first stages and model tests for ``sp.xtivreg``.

Helpers of :func:`statspai.panel.xtivreg.xtivreg` for the options of
Stata's ``xtivreg`` beyond the estimators themselves: ``fd`` (the rows
that form a first difference), ``first`` (the first-stage regressions)
and ``small`` (residual degrees of freedom, and the model test as F
instead of chi2).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import MethodIncompatibility


def fd_pairs(
    codes: np.ndarray, time: pd.Series, name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Rows ``(cur, prev)`` that form a first difference.

    A difference is taken between two observations of a panel at
    consecutive periods, the periods being the distinct values of the time
    variable in the estimation sample (Stata's ``D.`` operator on an
    ``xtset`` panel whose ``delta`` is the spacing of those values; the
    convention of ``sp.panel(method='fd')``). Two observations on either
    side of a gap form no difference.
    """
    tv = time.to_numpy()
    if pd.isna(tv).any():
        raise MethodIncompatibility(f"xtivreg: the time variable {name!r} is missing.")
    uniq = np.sort(pd.unique(tv))
    pos = np.searchsorted(uniq, tv)
    order = np.lexsort((pos, codes))
    a, b = order[:-1], order[1:]
    same = codes[a] == codes[b]
    if np.any(same & (pos[a] == pos[b])):
        raise MethodIncompatibility(
            f"xtivreg: method='fd' needs one row per panel and period; the "
            f"time variable {name!r} repeats within a panel.",
            recovery_hint="Aggregate or drop the duplicated (id, time) rows.",
        )
    ok = same & (pos[b] == pos[a] + 1)
    return b[ok], a[ok]


def ols_fit(
    tsls: Callable[..., Dict[str, np.ndarray]],
    span_index: Callable[[np.ndarray], List[int]],
    target: np.ndarray,
    X: np.ndarray,
    names: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """OLS of ``target`` on the independent columns of ``X``.

    The last column is the constant; it is kept first, so a column that is
    constant in this transformation (a time-invariant variable after the
    within transform) is the one that drops out.
    """
    last = X.shape[1] - 1
    order = [last] + list(range(last))
    idx = sorted(order[j] for j in span_index(X[:, order]))
    Xk = X[:, idx]
    return tsls(target, Xk, Xk), [names[j] for j in idx]


def first_stage(
    fits: Dict[str, Tuple[Dict[str, np.ndarray], List[str]]],
    s2_df: Callable[[int], float],
    cluster_vcov: Optional[Callable[[Dict[str, np.ndarray], int], np.ndarray]],
    nobs: int,
    inference: str,
) -> Dict[str, Dict[str, Any]]:
    """Coefficients and standard errors of each first-stage regression.

    ``s2_df(k)`` is the residual degrees of freedom of a first stage with
    ``k`` columns; ``cluster_vcov(fit, k)`` the cluster-robust variance
    when the main fit clusters.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for endog, (fit, cols) in fits.items():
        k = len(cols)
        if cluster_vcov is None:
            df_r = s2_df(k)
            V = float(fit["e"] @ fit["e"]) / df_r * fit["bread"]
        else:
            V = cluster_vcov(fit, k)
        se = np.sqrt(np.clip(np.diag(V), 0.0, None))
        out[endog] = {
            "params": dict(zip(cols, map(float, fit["d"]))),
            "std_errors": dict(zip(cols, map(float, se))),
            "nobs": int(nobs),
            "inference": inference,
        }
    return out


def model_test(
    d: np.ndarray, V: np.ndarray, small: bool, df_r: Optional[float]
) -> Dict[str, float]:
    """Wald test that every coefficient but the constant (last) is zero:
    chi2 by default, ``chi2 / q`` against F(q, df_r) with ``small``."""
    q = len(d) - 1
    if q < 1:
        return {}
    b = d[:-1]
    Vs = V[:-1, :-1]
    try:
        w = float(b @ np.linalg.solve(Vs, b))
    except np.linalg.LinAlgError:
        w = float(b @ np.linalg.pinv(Vs) @ b)
    if small and df_r is not None:
        F = w / q
        return {"F": F, "F_p": float(stats.f.sf(F, q, df_r)), "df_m": q}
    return {"chi2": w, "chi2_p": float(stats.chi2.sf(w, q)), "df_m": q}
