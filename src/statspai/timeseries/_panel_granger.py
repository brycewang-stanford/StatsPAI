"""Dumitrescu-Hurlin panel Granger non-causality test.

For every unit ``i`` the regression

    y_it = a_i + sum_{k=1}^K g_ik y_{i,t-k} + sum_{k=1}^K b_ik x_{i,t-k} + e_it

is fitted by OLS on that unit's own series, and ``W_i`` is the Wald
statistic of ``b_i1 = ... = b_iK = 0`` with the OLS variance
``s^2 (X'X)^{-1}``, ``s^2 = e'e / (n_i - 2K - 1)`` (``W_i = K F_i``). The null
is that ``x`` Granger-causes ``y`` in no unit; the alternative, in at least
some. With ``W = mean(W_i)``:

* ``Z_bar = sqrt(N / (2K)) (W - K)``, asymptotically N(0, 1) as ``T``
  then ``N`` grow;
* ``Z_tilde`` corrects for a fixed ``T``. Balanced:
  ``sqrt(N/(2K) (T-3K-5)/(T-2K-3)) ((T-3K-3)/(T-3K-1) W - K)``, ``T`` the
  number of periods per unit; unbalanced, the moment-matched form with
  each unit's ``T_i``.

This is ``plm::pgrangertest`` (all three statistics) and Stata's
``xtgcause`` (which writes ``Z_tilde`` with ``T - K`` regression
observations; the two expressions are the same). P-values are two-sided,
as both report them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from ..exceptions import DataInsufficient, MethodIncompatibility


def panel_granger(
    data: pd.DataFrame,
    caused: str,
    causing: Union[str, Sequence[str]],
    lags: int,
    id: str,
    time: str,
) -> Dict[str, Any]:
    if not isinstance(causing, str):
        causing_list = list(causing)
        if len(causing_list) != 1:
            raise MethodIncompatibility(
                "granger_causality: the panel (Dumitrescu-Hurlin) test takes "
                "one causing variable.",
                recovery_hint="Test each causing variable separately.",
            )
        causing = causing_list[0]
    if caused == causing:
        raise MethodIncompatibility(
            "granger_causality: 'causing' must not be the caused variable.",
            recovery_hint="Name two different columns.",
        )
    K = int(lags)
    if K < 1:
        raise MethodIncompatibility(
            f"granger_causality: lags={lags!r}; the test needs K >= 1.",
            recovery_hint="Pass lags=1 (or more).",
        )
    check_columns(
        data,
        {"caused": caused, "causing": causing, "id": id, "time": time},
        function="sp.granger_causality",
    )
    df = data[[id, time, caused, causing]].sort_values([id, time], kind="mergesort")
    if df.duplicated([id, time]).any():
        raise MethodIncompatibility(
            "granger_causality: repeated (id, time) pairs.",
            recovery_hint="Each unit must have one row per period.",
        )
    if df[[caused, causing]].isna().any().any():
        raise MethodIncompatibility(
            "granger_causality: missing values in the panel; the lag "
            "structure of a unit with a hole is not defined.",
            recovery_hint="Drop the units with missing periods or fill them.",
        )
    periods = np.sort(df[time].unique())
    rank = {t: r for r, t in enumerate(periods)}
    rows: List[Dict[str, Any]] = []
    Ti: List[int] = []
    for unit, g in df.groupby(id, sort=True):
        r = g[time].map(rank).to_numpy()
        if len(r) > 1 and np.any(np.diff(r) != 1):
            raise MethodIncompatibility(
                f"granger_causality: unit {unit!r} skips periods; its lags "
                "would join non-adjacent dates.",
                recovery_hint="Use units observed in consecutive periods.",
                diagnostics={"unit": unit},
            )
        y = g[caused].to_numpy(dtype=float)
        x = g[causing].to_numpy(dtype=float)
        n_t = len(y)
        n = n_t - K
        if n - 2 * K - 1 <= 0:
            raise DataInsufficient(
                f"granger_causality: unit {unit!r} has {n_t} periods, too "
                f"few for {K} lags.",
                recovery_hint="Use fewer lags or drop short units.",
                diagnostics={"unit": unit, "periods": n_t},
            )
        cols = [np.ones(n)]
        cols += [y[K - k : n_t - k] for k in range(1, K + 1)]
        cols += [x[K - k : n_t - k] for k in range(1, K + 1)]
        X = np.column_stack(cols)
        yy = y[K:]
        xtx_inv = np.linalg.inv(X.T @ X)
        b = xtx_inv @ (X.T @ yy)
        e = yy - X @ b
        s2 = float(e @ e) / (n - X.shape[1])
        idx = slice(1 + K, 1 + 2 * K)
        bk = b[idx]
        W = float(bk @ np.linalg.solve(s2 * xtx_inv[idx, idx], bk))
        rows.append(
            {
                id: unit,
                "wald": W,
                "p_value": float(stats.f.sf(W / K, K, n - X.shape[1])),
                "n": n,
            }
        )
        Ti.append(n_t)
    N = len(rows)
    Wi = np.array([r["wald"] for r in rows])
    Ti_arr = np.array(Ti, dtype=float)
    Wbar = float(Wi.mean())
    balanced = bool(np.all(Ti_arr == Ti_arr[0]))
    zbar = float(np.sqrt(N / (2.0 * K)) * (Wbar - K)) if balanced else float("nan")
    if balanced:
        T = Ti_arr[0]
        if T <= 3 * K + 5:
            ztilde = float("nan")
        else:
            ztilde = float(
                np.sqrt(N / (2.0 * K) * (T - 3 * K - 5) / (T - 2 * K - 3))
                * ((T - 3 * K - 3) / (T - 3 * K - 1) * Wbar - K)
            )
    else:
        if np.any(Ti_arr <= 3 * K + 5):
            ztilde = float("nan")
        else:
            mean_w = np.mean(K * (Ti_arr - 3 * K - 1) / (Ti_arr - 3 * K - 3))
            var_w = np.mean(
                2
                * K
                * (Ti_arr - 3 * K - 1) ** 2
                * (Ti_arr - 2 * K - 3)
                / ((Ti_arr - 3 * K - 3) ** 2 * (Ti_arr - 3 * K - 5))
            )
            ztilde = float(np.sqrt(N) * (Wbar - mean_w) / np.sqrt(var_w))
    p_zbar = float(2 * stats.norm.sf(abs(zbar))) if np.isfinite(zbar) else float("nan")
    p_zt = (
        float(2 * stats.norm.sf(abs(ztilde))) if np.isfinite(ztilde) else float("nan")
    )
    return {
        "test": "Dumitrescu-Hurlin",
        "W_bar": Wbar,
        "Z_bar": zbar,
        "Z_bar_p_value": p_zbar,
        "Z_tilde": ztilde,
        "Z_tilde_p_value": p_zt,
        "p_value": p_zt,
        "lags": K,
        "n_units": N,
        "balanced": balanced,
        "periods": int(Ti_arr.max()),
        "caused": caused,
        "causing": causing,
        "individual": pd.DataFrame(rows),
        "reject": bool(p_zt < 0.05) if np.isfinite(p_zt) else None,
    }
