"""
Malmquist productivity index for parametric stochastic frontiers.

Given panel data, fit a separate frontier ``F^t`` for each period ``t``
and compute the Fare-Grosskopf-Lindgren-Roos (1994) Malmquist index
between adjacent periods::

    M_O^{t -> t+1}
        = [ D^t(x^{t+1}, y^{t+1}) / D^t(x^t, y^t) ]
        x [ D^{t+1}(x^{t+1}, y^{t+1}) / D^{t+1}(x^t, y^t) ]^{1/2}

The canonical output-oriented decomposition is ``M = EC x TC``:

* **EC** (efficiency change) = ``D^{t+1}(x^{t+1}, y^{t+1}) / D^t(x^t, y^t)``
* **TC** (technical change) = ``[ D^t(x^{t+1}, y^{t+1}) / D^{t+1}(x^{t+1}, y^{t+1})
        x D^t(x^t, y^t) / D^{t+1}(x^t, y^t) ]^{1/2}``

We use the frontier-distance definition ``D^s(x, y) = y / exp(x' beta_s)``
(deterministic approximation standard in parametric Malmquist work;
Coelli-Rao 2005, "Total Factor Productivity Growth in Agriculture").

``M > 1`` = productivity growth, ``EC > 1`` = catch-up to own-period
frontier, ``TC > 1`` = outward frontier shift.

References
----------
Fare, R., Grosskopf, S., Lindgren, B. & Roos, P. (1994).  "Productivity
    changes in Swedish pharmacies 1980-1989: A non-parametric Malmquist
    approach."  J. Productivity Analysis 3, 85-101.
Coelli, T.J. & Rao, D.S.P. (2005).  "Total factor productivity growth in
    agriculture: a Malmquist index analysis of 93 countries, 1980-2000."
    Agricultural Economics 32, 115-134.
Kumbhakar, S.C., Wang, H.J. & Horncastle, A.P. (2015).  A Practitioner's
    Guide to Stochastic Frontier Analysis, Cambridge U.P., Chapter 14.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .sfa import FrontierResult, frontier as _frontier


@dataclass
class MalmquistResult:
    """Container for Malmquist productivity index decomposition."""

    index_table: pd.DataFrame
    """Wide table: one row per (id, period pair) with columns
    ``['m_index', 'ec', 'tc']`` plus the original id / period columns."""

    period_frontiers: Dict[Any, FrontierResult]
    """Frontier fit per period."""

    summary_by_period: pd.DataFrame
    """Mean M / EC / TC per period transition."""

    data_info: Dict[str, Any]

    def summary(self) -> str:
        lines = [
            "=" * 80,
            "Malmquist Productivity Index (Fare-Grosskopf-Lindgren-Roos 1994)",
            "=" * 80,
            f"Periods : {self.data_info['periods']}",
            f"N units : {self.data_info['n_units']}",
            f"Total transitions: {len(self.index_table)}",
            "",
            "Mean Malmquist components by period transition:",
            self.summary_by_period.round(4).to_string(),
            "",
            "Interpretation: M>1 = productivity growth, EC>1 = catch-up,",
            "TC>1 = frontier moved outward.",
        ]
        return "\n".join(lines)


def malmquist(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id: str,
    time: str,
    *,
    dist: str = "half-normal",
    cost: bool = False,
    **frontier_kwargs,
) -> MalmquistResult:
    """Compute the Malmquist productivity index via period-by-period SFA.

    Parameters
    ----------
    data : pandas.DataFrame
    y, x, id, time : str / list of str
    dist, cost : forwarded to :func:`frontier`
    **frontier_kwargs : forwarded to per-period :func:`frontier`
        (e.g., ``usigma``, ``vsigma``, ``emean``).

    Returns
    -------
    :class:`MalmquistResult`

    Notes
    -----
    Assumes the dependent variable ``y`` is already in log form for a
    log-linear / Cobb-Douglas / translog frontier.  The distance
    function used is ``D^s(x, y) = y / exp(x' beta_s)``, so for the
    fitted period ``D < 1`` corresponds to technically inefficient
    firms and ``D = 1`` on the frontier.
    """
    required = [y] + list(x) + [id, time]
    df = data[required].dropna().copy()
    df = df.sort_values([id, time]).reset_index(drop=True)

    periods = sorted(df[time].unique())
    if len(periods) < 2:
        raise ValueError("Malmquist index requires at least two periods.")

    # Fit a frontier per period.
    period_frontiers: Dict[Any, FrontierResult] = {}
    period_betas: Dict[Any, np.ndarray] = {}
    for t in periods:
        sub = df[df[time] == t].copy()
        if len(sub) < len(x) + 3:
            raise ValueError(
                f"Period {t!r} has only {len(sub)} observations; need at least "
                f"{len(x) + 3} to identify the frontier."
            )
        res = _frontier(sub, y=y, x=x, dist=dist, cost=cost, **frontier_kwargs)
        period_frontiers[t] = res
        period_betas[t] = res.params.loc[["_cons"] + list(x)].to_numpy()

    def _log_distance(xmat: np.ndarray, y_vals: np.ndarray,
                      beta: np.ndarray) -> np.ndarray:
        """log D^s(x, y) = log y - x' beta_s (for production; flip for cost)."""
        return y_vals - xmat @ beta

    # Walk all firms through adjacent period pairs.
    rows = []
    for unit_id, grp in df.groupby(id, sort=False):
        grp = grp.sort_values(time)
        times_seen = grp[time].to_numpy()
        X_seen = np.column_stack([np.ones(len(grp)), grp[x].to_numpy()])
        y_seen = grp[y].to_numpy()

        for t_idx in range(len(times_seen) - 1):
            t1 = times_seen[t_idx]
            t2 = times_seen[t_idx + 1]
            if t2 not in period_frontiers or t1 not in period_frontiers:
                continue
            # Require consecutive periods in the global list to avoid
            # mixing non-adjacent observations.
            if periods.index(t2) != periods.index(t1) + 1:
                continue
            x1 = X_seen[t_idx]
            x2 = X_seen[t_idx + 1]
            yv1 = y_seen[t_idx]
            yv2 = y_seen[t_idx + 1]
            beta1 = period_betas[t1]
            beta2 = period_betas[t2]

            log_D_t_xt_yt   = _log_distance(x1.reshape(1, -1), np.array([yv1]), beta1)[0]
            log_D_tp_xtp_ytp = _log_distance(x2.reshape(1, -1), np.array([yv2]), beta2)[0]
            log_D_t_xtp_ytp  = _log_distance(x2.reshape(1, -1), np.array([yv2]), beta1)[0]
            log_D_tp_xt_yt   = _log_distance(x1.reshape(1, -1), np.array([yv1]), beta2)[0]

            # Output-oriented Malmquist index.
            if cost:
                # Cost orientation: reciprocal for distance-to-cost-frontier.
                log_D_t_xt_yt, log_D_tp_xtp_ytp, log_D_t_xtp_ytp, log_D_tp_xt_yt = (
                    -log_D_t_xt_yt, -log_D_tp_xtp_ytp,
                    -log_D_t_xtp_ytp, -log_D_tp_xt_yt,
                )

            log_M = 0.5 * (
                (log_D_t_xtp_ytp - log_D_t_xt_yt)
                + (log_D_tp_xtp_ytp - log_D_tp_xt_yt)
            )
            log_EC = log_D_tp_xtp_ytp - log_D_t_xt_yt
            log_TC = log_M - log_EC

            rows.append(
                {
                    id: unit_id,
                    f"{time}_from": t1,
                    f"{time}_to": t2,
                    "m_index": float(np.exp(log_M)),
                    "ec": float(np.exp(log_EC)),
                    "tc": float(np.exp(log_TC)),
                }
            )

    index_table = pd.DataFrame(rows)
    if len(index_table) == 0:
        raise RuntimeError(
            "No consecutive-period observations found; Malmquist index is empty."
        )

    by_period = index_table.groupby(f"{time}_to")[["m_index", "ec", "tc"]].agg(
        ["mean", "std", "count"]
    )

    return MalmquistResult(
        index_table=index_table,
        period_frontiers=period_frontiers,
        summary_by_period=by_period,
        data_info={
            "periods": periods,
            "n_units": df[id].nunique(),
            "n_obs": len(df),
            "dep_var": y,
            "regressors": list(x),
            "id_col": id,
            "time_col": time,
            "orientation": "cost" if cost else "output",
        },
    )


# ---------------------------------------------------------------------------
# Translog design helper
# ---------------------------------------------------------------------------


def translog_design(
    data: pd.DataFrame,
    inputs: List[str],
    *,
    include_interactions: bool = True,
    include_squares: bool = True,
    interaction_prefix: str = "",
) -> pd.DataFrame:
    """Build a translog design matrix from Cobb-Douglas inputs.

    Translog is ``log y = alpha + sum_k beta_k * log x_k
                          + 0.5 sum_k sum_l gamma_{kl} * log x_k * log x_l``.

    This helper takes input columns (already in log form) and returns a
    DataFrame with the original columns plus squares ``x_k^2 / 2`` and
    cross-products ``x_k * x_l`` that can be fed straight to
    :func:`frontier` / :func:`xtfrontier` as additional regressors.

    Parameters
    ----------
    data : pandas.DataFrame
    inputs : list of str
        Columns containing ``log x_k`` terms (already log-transformed).
    include_interactions : bool, default True
        If True, adds ``x_k * x_l`` for k < l.
    include_squares : bool, default True
        If True, adds ``0.5 * x_k^2`` terms (translog convention).
    interaction_prefix : str, default ""
        Optional prefix for the generated columns (e.g., ``"tl_"``).

    Returns
    -------
    pandas.DataFrame
        Original data + appended translog terms.  Two lists are stored
        on ``df.attrs`` for convenience (neither is auto-consumed by
        :func:`frontier` or :func:`xtfrontier` — the user must pass one
        of them explicitly as ``x=``):

        - ``df.attrs['translog_terms']`` — *all* regressors for a
          translog frontier: original inputs + squares + interactions.
          Pass this directly to ``sp.frontier(..., x=terms)``.
        - ``df.attrs['translog_added_terms']`` — only the *new* columns
          appended by this helper (squares and interactions). Use this
          if you already have ``inputs`` in your ``x`` list and just
          want to extend it.

    Examples
    --------
    >>> df_tl = translog_design(df, inputs=["log_k", "log_l"])
    >>> # Option A — one-liner, pass the full translog regressor list:
    >>> terms = df_tl.attrs["translog_terms"]
    >>> sp.frontier(df_tl, y="log_y", x=terms)
    >>> # Option B — extend an existing x list without double-counting:
    >>> base = ["log_k", "log_l"]
    >>> sp.frontier(df_tl, y="log_y",
    ...             x=base + df_tl.attrs["translog_added_terms"])
    """
    if not inputs:
        raise ValueError("inputs must be non-empty.")
    df = data.copy()
    added: List[str] = []  # only the columns this helper creates

    if include_squares:
        for k in inputs:
            col = f"{interaction_prefix}{k}_sq"
            df[col] = 0.5 * df[k] ** 2
            added.append(col)

    if include_interactions:
        for i, k in enumerate(inputs):
            for l in inputs[i + 1:]:
                col = f"{interaction_prefix}{k}_x_{l}"
                df[col] = df[k] * df[l]
                added.append(col)

    # The full translog regressor list = original inputs + newly added terms.
    df.attrs["translog_terms"] = list(inputs) + added
    df.attrs["translog_added_terms"] = added
    return df


__all__ = ["malmquist", "MalmquistResult", "translog_design"]
