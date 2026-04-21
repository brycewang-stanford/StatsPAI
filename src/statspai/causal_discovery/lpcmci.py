"""
LPCMCI: Latent-PCMCI for time-series causal discovery with hidden confounders.

Gerhardus & Runge (2020, NeurIPS) relax PCMCI's no-latent-confounders
assumption by learning a **time-series Maximal Ancestral Graph (MAG)**
with lag information. Edges are typed:

- ``'-->'`` directed (``X → Y``)
- ``'<->'`` bidirected (``X ↔ Y``, shared latent ancestor)
- ``'o->'`` / ``'o-o'`` uncertain orientation

This implementation is a *lightweight* version suitable for small-to-mid
size systems (≤30 variables, ≤8 lags). For industrial-grade systems with
hundreds of variables, delegate to ``tigramite`` (not a hard dependency).

References
----------
Gerhardus, A. & Runge, J. (2020).
"High-recall causal discovery for autocorrelated time series with latent
confounders." NeurIPS 2020.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .pcmci import partial_corr_pvalue


__all__ = ["lpcmci", "LPCMCIResult"]


@dataclass
class LPCMCIResult:
    """Output of :func:`lpcmci`."""
    variables: List[str]
    tau_max: int
    alpha: float
    # Edge-type tensor [lag, i, j]: '-->', '<->', 'o->', 'o-o', or '' (none)
    edge_types: np.ndarray
    # p-values from the MCI test for the skeleton
    p_values: np.ndarray
    # Per-variable parent sets after PC1
    parents: Dict[str, List[tuple]]

    def summary(self) -> str:
        lines = [
            "LPCMCI — Latent-PCMCI for time-series causal discovery",
            "=" * 60,
            f"  variables : {self.variables}",
            f"  tau_max   : {self.tau_max}",
            f"  alpha     : {self.alpha}",
        ]
        n_d = int((self.edge_types == "-->").sum())
        n_b = int((self.edge_types == "<->").sum())
        n_u = int(np.isin(self.edge_types, ["o->", "o-o"]).sum())
        lines.append(f"  edges     : directed={n_d}, bidirected={n_b}, uncertain={n_u}")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Long-format edges DataFrame."""
        rows = []
        N = len(self.variables)
        for lag in range(self.tau_max + 1):
            for i in range(N):
                for j in range(N):
                    et = self.edge_types[lag, i, j]
                    if et:
                        rows.append({
                            "lag": lag,
                            "from": self.variables[i],
                            "to": self.variables[j],
                            "type": et,
                            "p_value": float(self.p_values[lag, i, j]),
                        })
        return pd.DataFrame(rows)


def _lag_matrix(data: np.ndarray, lag: int) -> np.ndarray:
    """Return (T-lag)-row matrix of ``X_{t-lag}`` aligned with ``X_t``."""
    if lag == 0:
        return data.copy()
    return data[:-lag].copy()


def _present(data: np.ndarray, lag: int) -> np.ndarray:
    """Return rows aligned with ``_lag_matrix(data, lag)`` for the present."""
    if lag == 0:
        return data.copy()
    return data[lag:].copy()


def lpcmci(
    data: pd.DataFrame,
    *,
    variables: Optional[Sequence[str]] = None,
    tau_max: int = 3,
    alpha: float = 0.05,
    ci_test: Optional[Callable] = None,
    max_cond_dim: int = 3,
) -> LPCMCIResult:
    """Latent-PCMCI for stationary time-series causal discovery.

    Parameters
    ----------
    data : DataFrame
        Long-format time series, one row per time stamp. Rows must be in
        chronological order; index is ignored.
    variables : sequence of str, optional
        Subset of columns to analyse. Defaults to all numeric columns.
    tau_max : int, default 3
        Maximum lag to consider.
    alpha : float, default 0.05
        Significance level for CI tests.
    ci_test : callable, optional
        Custom CI test ``(x, y, Z) -> p_value``. Defaults to partial
        correlation (Gaussian).
    max_cond_dim : int, default 3
        Maximum size of conditioning set in the PC1 stage.

    Returns
    -------
    LPCMCIResult

    Notes
    -----
    Algorithm sketch (Gerhardus & Runge 2020, simplified):

    1. **PC1** — standard PCMCI parent selection.
    2. **MCI** — for each pair ``(i, τ, j)`` with ``τ > 0``, test
       ``X_i^{t-τ} ⟂ X_j^t | parents(X_j^t) ∪ parents(X_i^{t-τ})``. If
       rejected, keep the edge.
    3. **Orientation** — directed edges are kept as ``'-->'``. A
       contemporaneous edge (``τ=0``) that survives MCI with a non-empty
       conditioning set on both sides is marked ``'<->'`` (bidirected)
       as a proxy for latent-confounder detection. Uncertain
       contemporaneous edges become ``'o-o'``.

    The full LPCMCI paper uses a richer orientation rule-set; this
    implementation gives users the edge typology without the full
    ancestral-graph orientation search.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    if variables is None:
        variables = [c for c in data.columns if np.issubdtype(data[c].dtype, np.number)]
    variables = list(variables)
    if len(variables) < 2:
        raise ValueError("Need at least 2 variables for LPCMCI.")
    X = data[variables].to_numpy(dtype=float)
    T, N = X.shape
    if T <= tau_max + 5:
        raise ValueError(
            f"Time series too short (T={T}) relative to tau_max={tau_max}. "
            "Need T > tau_max + 5."
        )
    if ci_test is None:
        ci_test = partial_corr_pvalue

    # Stage 1: PC1 parent selection (like PCMCI but simplified).
    parents: Dict[str, List[tuple]] = {v: [] for v in variables}
    for j, vj in enumerate(variables):
        candidates = [(i, tau) for tau in range(1, tau_max + 1) for i in range(N)]
        # Marginal screening
        kept = []
        for (i, tau) in candidates:
            x = _lag_matrix(X[:, i:i + 1], tau).ravel()
            y = _present(X[:, j:j + 1], tau).ravel()
            p = float(ci_test(x, y, None))
            if p < alpha:
                kept.append(((i, tau), p))
        kept.sort(key=lambda r: r[1])
        # Iteratively condition on growing parent sets
        current = [k for k, _ in kept]
        for depth in range(1, max_cond_dim + 1):
            new_current = []
            for (i, tau) in current:
                # Build conditioning set from top-depth other parents
                others = [c for c in current if c != (i, tau)][:depth]
                if not others:
                    new_current.append((i, tau))
                    continue
                lag_max_used = max(tau, *(o[1] for o in others))
                y = X[lag_max_used:, j]
                x = X[lag_max_used - tau: T - tau, i]
                Z = np.column_stack([
                    X[lag_max_used - ot: T - ot, oi]
                    for (oi, ot) in others
                ])
                p = float(ci_test(x, y, Z))
                if p < alpha:
                    new_current.append((i, tau))
            current = new_current
        parents[vj] = current

    # Stage 2: MCI test + edge typing
    edge_types = np.full((tau_max + 1, N, N), "", dtype=object)
    p_values = np.full((tau_max + 1, N, N), np.nan)

    for j, vj in enumerate(variables):
        for i, vi in enumerate(variables):
            for tau in range(tau_max + 1):
                if tau == 0 and i == j:
                    continue
                if tau > 0 and i == j:
                    pass  # autoregressive self-loop allowed
                # Conditioning: parents of j (excluding this edge) + parents of i
                Z_parents = [p for p in parents[vj] if p != (i, tau)]
                Z_parents += [(p0, p1 + tau) for (p0, p1) in parents[vi]]
                Z_parents = [p for p in Z_parents if 0 < p[1] <= tau_max]
                lag_max_used = max([tau] + [p[1] for p in Z_parents] + [1])
                if T - lag_max_used < 5:
                    continue
                y = X[lag_max_used:, j]
                x = X[lag_max_used - tau: T - tau, i] if tau > 0 else X[lag_max_used:, i]
                if Z_parents:
                    Z = np.column_stack([
                        X[lag_max_used - pt: T - pt, pi]
                        for (pi, pt) in Z_parents
                    ])
                else:
                    Z = None
                p = float(ci_test(x, y, Z))
                p_values[tau, i, j] = p
                if p < alpha:
                    if tau > 0:
                        # Directed via time order
                        edge_types[tau, i, j] = "-->"
                    else:
                        # Contemporaneous — type depends on symmetry
                        # If also significant in the reverse direction with
                        # both sides' parent sets, flag bidirected.
                        edge_types[tau, i, j] = "o-o"

    # Convert symmetric contemporaneous 'o-o' → '<->' when both directions
    # survive after conditioning on *both* parent unions (proxy for latent
    # confounder).
    for i in range(N):
        for j in range(i + 1, N):
            if edge_types[0, i, j] == "o-o" and edge_types[0, j, i] == "o-o":
                edge_types[0, i, j] = "<->"
                edge_types[0, j, i] = "<->"

    return LPCMCIResult(
        variables=variables,
        tau_max=tau_max,
        alpha=alpha,
        edge_types=edge_types,
        p_values=p_values,
        parents=parents,
    )
