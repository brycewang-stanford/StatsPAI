"""Sanderson-Windmeijer conditional first-stage statistics for ``sp.hdfe_ols``.

With two or more endogenous regressors the plain first-stage F of each one
says nothing about whether the instruments separate it from the others.
The Sanderson-Windmeijer statistic [@sanderson2016weak] conditions on the
other endogenous regressors: for regressor ``j``,

1. ``delta`` is the 2SLS coefficient of ``d_j`` on the other endogenous
   regressors ``D_-j`` with the excluded instruments (every variable already
   swept of the absorbed fixed effects and partialled of the included
   exogenous regressors, which is FWL-equivalent to keeping them in);
2. ``v = d_j - D_-j delta`` uses the actual, not the fitted, ``D_-j``;
3. ``v`` is regressed on the excluded instruments and their coefficients
   are tested with a Wald statistic -- classical, heteroskedasticity-robust
   or cluster-robust, the same variance ``ivreg2`` uses for the ordinary
   first-stage F.

``p - 1`` directions of the instrument space are spent on ``delta``, so the
statistic has ``L1 - p + 1`` degrees of freedom (``L1`` excluded
instruments, ``p`` endogenous regressors). The finite-sample scaling is
``ivreg2``'s (the ``ivreghdfe`` default, always ``small``): with ``N`` the
observations, ``L`` all instruments (excluded plus included), ``sdof`` the
absorbed degrees of freedom (fixed effects nested in the cluster excluded)
and ``G`` the clusters,

* ``SW chi2`` is the Wald statistic with no small-sample factor (under iid
  the residual variance is ``u'u / N``);
* ``SW F = chi2 / (L1 - p + 1) / q`` with ``q = N / (N - L - sdof)``
  (iid, robust) or ``q = (N - 1) / (N - L - sdof) * G / (G - 1)`` (cluster),
  referred to ``F(L1 - p + 1, N - L - sdof)`` or ``F(L1 - p + 1, G - 1)``;
* the partial R-squared is ``1 - u'u / v'v`` from the classical fit whatever
  the variance (``ivreg2``'s ``SWr2``).

With one endogenous regressor ``D_-j`` is empty, ``v = d_1`` and every
number equals the ordinary first-stage statistic. Verified against
``ivreg2`` 4.1.12 and ``ivreghdfe`` 1.1.4 in
``tests/reference_parity/test_iv_sw_first_stage_parity.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats


def sw_first_stage(
    Dt: np.ndarray,
    Zt: np.ndarray,
    names: List[str],
    *,
    groups: Optional[np.ndarray],
    robust: bool,
    n_obs: int,
    n_instruments_total: int,
    sdof: int,
) -> Dict[str, Dict[str, Any]]:
    """Sanderson-Windmeijer statistics for each column of ``Dt``.

    Parameters
    ----------
    Dt, Zt : ndarray
        Endogenous regressors ``(N, p)`` and excluded instruments
        ``(N, L1)``, both swept of the fixed effects and partialled of the
        included exogenous regressors.
    names : list of str
        Labels of the endogenous regressors.
    groups : ndarray or None
        Cluster labels on the estimation sample (one-way).
    robust : bool
        Heteroskedasticity-robust variance when not clustered.
    n_obs, n_instruments_total, sdof : int
        ``N``, ``L`` (excluded plus included instruments) and the absorbed
        degrees of freedom, as in the rest of the ``ivreg2`` statistics.

    Returns
    -------
    dict
        ``name -> {'sw_F', 'sw_F_df', 'sw_F_p', 'sw_chi2', 'sw_chi2_df',
        'sw_chi2_p', 'sw_partial_r2'}``.
    """
    from ._hdfe_iv import _meat

    N, p = Dt.shape
    L1 = Zt.shape[1]
    df1 = L1 - p + 1
    dfr = n_obs - n_instruments_total - sdof
    if groups is not None:
        G = int(len(np.unique(groups)))
        q = (n_obs - 1) / dfr * G / (G - 1)
        df2 = G - 1
    else:
        q = n_obs / dfr
        df2 = dfr
    ZZ = Zt.T @ Zt
    ZZ_inv = np.linalg.inv(ZZ)
    # projection of the endogenous regressors on the instruments, used by
    # the 2SLS step of every j
    Dhat = Zt @ (ZZ_inv @ (Zt.T @ Dt))

    out: Dict[str, Dict[str, Any]] = {}
    for j, nm in enumerate(names):
        others = [k for k in range(p) if k != j]
        v = Dt[:, j].copy()
        if others:
            Do, Dho = Dt[:, others], Dhat[:, others]
            delta = np.linalg.solve(Dho.T @ Dho, Dho.T @ Dt[:, j])
            v = v - Do @ delta
        pi = ZZ_inv @ (Zt.T @ v)
        u = v - Zt @ pi
        uu = float(u @ u)
        ess = float(pi @ ZZ @ pi)
        if groups is None and not robust:
            chi2 = ess / (uu / n_obs)
        else:
            Vn = ZZ_inv @ _meat(Zt, u, groups) @ ZZ_inv
            chi2 = float(pi @ np.linalg.solve(Vn, pi))
        F = chi2 / df1 / q
        vv = float(v @ v)
        out[nm] = {
            "sw_F": float(F),
            "sw_F_df": (int(df1), int(df2)),
            "sw_F_p": float(stats.f.sf(F, df1, df2)),
            "sw_chi2": float(chi2),
            "sw_chi2_df": int(df1),
            "sw_chi2_p": float(stats.chi2.sf(chi2, df1)),
            "sw_partial_r2": float(1.0 - uu / vv) if vv > 0 else float("nan"),
        }
    return out
