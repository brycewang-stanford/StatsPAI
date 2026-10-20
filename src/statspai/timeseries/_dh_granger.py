"""Dumitrescu-Hurlin test of Granger non-causality in heterogeneous panels.

Each unit ``i`` gets its own regression on its ``T_i`` consecutive periods,

    y_t = a_i + sum_{k=1}^{K} g_{ik} y_{t-k} + sum_{k=1}^{K} b_{ik} x_{t-k} + e_t,

fitted on the ``n_i = T_i - K`` periods with complete lags, and the Wald
statistic of ``b_{i1} = ... = b_{iK} = 0``,

    W_i = (RSS_r - RSS_u) / (RSS_u / (n_i - 2K - 1)).

Under the null of no causality for any unit [@dumitrescu2012testing]:

* ``W_bar = mean_i W_i``;
* ``Z_bar = sqrt(N / (2K)) (W_bar - K)``, standard normal as ``T`` then
  ``N`` grow (balanced panels only);
* ``Z_tilde = (sum_i W_i - sum_i E_i) / sqrt(sum_i V_i)``, standard normal
  as ``N`` grows for fixed ``T``, where ``E_i`` and ``V_i`` are the exact
  mean and variance of ``K F(K, n_i - 2K - 1)`` (the small-sample law of
  ``W_i`` with normal errors):

      E_i = K (n_i - 2K - 1) / (n_i - 2K - 3),
      V_i = 2K (n_i - 2K - 1)^2 (n_i - K - 3)
            / ((n_i - 2K - 3)^2 (n_i - 2K - 5)).

  In terms of the periods ``T_i = n_i + K`` the variance is finite only for
  ``T_i > 5 + 3K``. On a balanced panel the expression reduces to the
  paper's closed form in ``T``.

The per-unit p-value refers ``W_i / K`` to ``F(K, n_i - 2K - 1)``, the law
behind ``E_i`` and ``V_i`` (Stata ``xtgcause``'s ``r(PVi)``; ``plm``'s
``indgranger`` reports the asymptotic ``chi2(K)`` value instead).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from ..core._collinear import independent_columns
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._lpx_panel import period_index

FN = "granger_causality"


def _refuse(message: str, hint: str, **diag: Any) -> MethodIncompatibility:
    return MethodIncompatibility(
        f"{FN}: {message}", recovery_hint=hint, diagnostics=diag
    )


def _wald_parts(Xu: np.ndarray, y: np.ndarray, k_r: int) -> tuple:
    """``(RSS_r - RSS_u, RSS_u)`` for the nested fits on the first ``k_r``
    columns and on all of ``Xu``.

    With ``Xu = QR`` the restricted fit spans the first ``k_r`` columns of
    ``Q``, so the drop in the residual sum of squares is the squared norm of
    the remaining coordinates of ``Q'y``; computing it that way avoids the
    cancellation in ``RSS_r - RSS_u`` when the drop is small.
    """
    q, _ = np.linalg.qr(Xu, mode="reduced")
    qy = q.T @ y
    e = y - q @ qy
    return float(qy[k_r:] @ qy[k_r:]), float(e @ e)


def dumitrescu_hurlin(
    data: pd.DataFrame,
    *,
    caused: Any,
    causing: Any,
    lags: Optional[int],
    id: str,
    time: str,
) -> Dict[str, Any]:
    """Run the test; see the module docstring. Returns a dict."""
    if isinstance(causing, (list, tuple)):
        if len(causing) != 1:
            raise _refuse(
                "the panel (Dumitrescu-Hurlin) test takes one causing variable.",
                "Test each candidate separately, e.g. causing='x'.",
                causing=list(causing),
            )
        causing = causing[0]
    if not isinstance(caused, str) or not isinstance(causing, str):
        raise _refuse(
            "caused= and causing= must name one column each.",
            "e.g. caused='y', causing='x'.",
        )
    if caused == causing:
        raise _refuse(
            "the causing variable is the caused variable.",
            "Name two different columns.",
        )
    K = 1 if lags is None else lags
    if isinstance(K, (bool, np.bool_)) or not isinstance(K, (int, np.integer)) or K < 1:
        raise _refuse(
            f"lags={lags!r}; the test needs an integer K >= 1.",
            "Pass lags=1 (or more).",
        )
    K = int(K)
    check_columns(
        data,
        {"caused": caused, "causing": causing, "id": id, "time": time},
        function="sp.granger_causality",
    )
    frame = data[[id, time, caused, causing]].reset_index(drop=True)
    values = {}
    for c in (caused, causing):
        try:
            values[c] = frame[c].to_numpy(dtype=float)
        except (TypeError, ValueError):
            raise _refuse(
                f"column {c!r} is not numeric.", "Convert it to numbers first."
            ) from None
        if not np.isfinite(values[c]).all():
            raise _refuse(
                f"{c!r} has missing values; a unit's lags would join "
                "non-adjacent periods.",
                "Drop the incomplete units or fill the gaps first.",
                column=c,
            )
    unit_codes, unit_labels = pd.factorize(frame[id])
    periods = period_index(frame[time], "time")

    units: List[Dict[str, Any]] = []
    for u in range(len(unit_labels)):
        rows = np.flatnonzero(unit_codes == u)
        rows = rows[np.argsort(periods[rows], kind="stable")]
        steps = np.diff(periods[rows])
        if np.any(steps == 0):
            raise _refuse(
                f"unit {unit_labels[u]!r} has the same period twice.",
                "Aggregate or drop the duplicate rows first.",
            )
        if np.any(steps != 1):
            raise _refuse(
                f"unit {unit_labels[u]!r} skips periods; its lags would join "
                "non-adjacent dates.",
                "Use units observed in consecutive periods (fill or drop the " "gaps).",
                unit=unit_labels[u],
            )
        units.append({"label": unit_labels[u], "rows": rows, "T": len(rows)})

    need = 3 * K + 6
    short = [d["label"] for d in units if d["T"] < need]
    if short:
        raise DataInsufficient(
            f"sp.{FN}: {len(short)} of {len(units)} units have T <= 5 + 3K = "
            f"{need - 1} periods (K = {K}); the fixed-T variance of the "
            "Wald statistic does not exist there (Stata xtgcause and plm "
            "refuse too).",
            recovery_hint="Use fewer lags or drop the short units.",
            diagnostics={
                "min_periods_needed": need,
                "short_units": [str(s) for s in short][:20],
                "lags": K,
            },
        )

    y_all, x_all = values[caused], values[causing]
    wald = np.empty(len(units))
    pval = np.empty(len(units))
    n_reg = np.empty(len(units), dtype=int)
    for i, d in enumerate(units):
        y, x = y_all[d["rows"]], x_all[d["rows"]]
        T = d["T"]
        lhs = y[K:]
        ylags = [y[K - k : T - k] for k in range(1, K + 1)]
        xlags = [x[K - k : T - k] for k in range(1, K + 1)]
        Xr = np.column_stack([np.ones(T - K), *ylags])
        Xu = np.column_stack([Xr, *xlags])
        labels = ["_cons"] + [f"L{k}.{caused}" for k in range(1, K + 1)]
        labels += [f"L{k}.{causing}" for k in range(1, K + 1)]
        _, dependent = independent_columns(Xu, labels)
        if dependent:
            raise DataInsufficient(
                f"sp.{FN}: the regression of unit {d['label']!r} is singular "
                "("
                + "; ".join(f"{o['variable']}: {o['reason']}" for o in dependent)
                + "), so its Wald statistic does not exist.",
                recovery_hint="Drop units whose series are constant or "
                "collinear over their periods.",
                diagnostics={"unit": str(d["label"]), "omitted": dependent},
            )
        drop, rss_u = _wald_parts(Xu, lhs, Xr.shape[1])
        dof = (T - K) - 2 * K - 1
        wald[i] = drop / (rss_u / dof)
        pval[i] = stats.f.sf(wald[i] / K, K, dof)
        n_reg[i] = T - K

    N = len(units)
    T_i = np.array([d["T"] for d in units], dtype=float)
    balanced = bool(np.all(T_i == T_i[0]))
    n_i = T_i - K
    mean_i = K * (n_i - 2 * K - 1) / (n_i - 2 * K - 3)
    var_i = (
        2
        * K
        * (n_i - 2 * K - 1) ** 2
        * (n_i - K - 3)
        / ((n_i - 2 * K - 3) ** 2 * (n_i - 2 * K - 5))
    )
    w_bar = float(wald.mean())
    z_tilde = float((wald.sum() - mean_i.sum()) / np.sqrt(var_i.sum()))
    p_tilde = float(2 * stats.norm.sf(abs(z_tilde)))
    if balanced:
        z_bar = float(np.sqrt(N / (2 * K)) * (w_bar - K))
        p_bar = float(2 * stats.norm.sf(abs(z_bar)))
    else:
        z_bar = p_bar = float("nan")
    out: Dict[str, Any] = {
        "test": "Dumitrescu-Hurlin",
        "W_bar": w_bar,
        "Z_bar": z_bar,
        "Z_bar_p_value": p_bar,
        "Z_tilde": z_tilde,
        "Z_tilde_p_value": p_tilde,
        "p_value": p_tilde,
        "lags": K,
        "n_units": N,
        "balanced": balanced,
        "periods": int(T_i.max()),
        "caused": caused,
        "causing": causing,
        "individual": pd.DataFrame(
            {"id": list(unit_labels), "wald": wald, "p_value": pval, "n": n_reg}
        ),
        "reject": bool(p_tilde < 0.05),
    }
    if not balanced:
        out["note"] = (
            "Unbalanced panel: Z_bar (asymptotic in T with a common T) is not "
            "reported; Z_tilde uses each unit's own T."
        )
    return out
