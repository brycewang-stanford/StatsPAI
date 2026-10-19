"""Historical decomposition of a structural VAR.

With structural shocks ``u_t = P^{-1} e_t`` (``e_t`` the VAR residuals) the
data satisfy, period by period,

    y_t = baseline_t + sum_j sum_{s=0}^{t-1} Theta_s[:, j] u_{j, t-s},

``Theta_s = Phi_s P`` the structural responses and ``baseline_t`` the path
from the initial values and the deterministic terms when every shock is
zero. The shock sums are what ``svars::hd`` reports (it accumulates from
the first residual and plots them against the demeaned series); here the
baseline is returned as well, so the decomposition adds up exactly.

A proxy identification pins down one column ``b`` of ``P``: its shock is
``u_t = b' Sigma^{-1} e_t / (b' Sigma^{-1} b)`` scaled to the impact column
used, and the rest of ``e_t`` is reported as ``other``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility


def historical_decomposition(result: Any) -> pd.DataFrame:
    from .svar import ma_coefficients

    if result._draws is not None:
        raise MethodIncompatibility(
            "svar: sign restrictions identify a set of impact matrices; the "
            "historical decomposition of their pointwise median is not a "
            "decomposition of the data.",
            recovery_hint="Use a point-identified model (short-run, long-run "
            "or proxy).",
        )
    vr = result._var_result
    if vr is None or vr._levels is None:
        raise MethodIncompatibility(
            "svar: this result does not carry its VAR; re-run sp.svar.",
            recovery_hint="Fit with sp.svar(sp.var(...), ...).",
        )
    resid = np.asarray(vr.residuals, float)  # (T, K)
    levels = np.asarray(vr._levels, float)
    T, k = resid.shape
    P = np.asarray(result._impact, float)  # (K, m)
    if result.identification == "proxy":
        sigma = np.asarray(result._sigma, float)
        w = np.linalg.solve(sigma, P[:, 0])
        shocks = (resid @ w / (P[:, 0] @ w))[:, None]
    else:
        shocks = np.linalg.solve(P, resid.T).T
    phi = ma_coefficients(result._lag_matrices, T - 1)
    theta = np.einsum("sij,jk->sik", phi, P)  # (T, K, m)
    m = P.shape[1]
    contrib = np.zeros((T, k, m))
    for t in range(T):
        # sum_{s=0}^{t} theta[s] * shocks[t - s]
        contrib[t] = np.einsum("sik,sk->ik", theta[: t + 1], shocks[t::-1])
    actual = levels[len(levels) - T :]
    other = None
    if result.identification == "proxy":
        e_left = resid - shocks @ P.T
        other = np.zeros((T, k))
        for t in range(T):
            other[t] = np.einsum("sij,sj->i", phi[: t + 1], e_left[t::-1])
    total = contrib.sum(axis=2) + (0.0 if other is None else other)
    baseline = actual - total
    start = len(levels) - T
    rows = []
    for t in range(T):
        for i, name in enumerate(result.var_names):
            row = {
                "period": start + t,
                "variable": name,
                "actual": actual[t, i],
                "baseline": baseline[t, i],
            }
            for j, shock in enumerate(result.shock_names):
                row[shock] = contrib[t, i, j]
            if other is not None:
                row["other"] = other[t, i]
            rows.append(row)
    return pd.DataFrame(rows)
