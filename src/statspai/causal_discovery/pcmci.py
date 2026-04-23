"""
PCMCI: causal discovery for time-series (Runge et al. 2019).

A two-stage algorithm specialised for stationary multivariate time-series
with auto-correlation and cross-correlation:

1. **PC1 stage (parent selection)** — for each variable ``X_j^t``, iteratively
   select a parent set ``\\hat{P}(X_j^t)`` from the lagged predictors
   ``{X_k^{t-τ} : k, τ ∈ 1..τ_max}`` using conditional-independence tests.
2. **MCI stage (momentary conditional independence)** — for each candidate
   causal link ``X_i^{t-τ} → X_j^t``, test conditional independence while
   conditioning on **both** parent sets. This controls false positives
   driven by autocorrelation that PC alone can't suppress.

The output is a lag-specific adjacency tensor ``A[lag, i, j]`` of
p-values for the ``X_i^{t-lag} → X_j^t`` edge, plus a binary decision
matrix at the user-specified significance level.

Conditional-independence test
-----------------------------
A **partial-correlation** CI test (Gaussian assumption) is the default —
fast, analytically tractable, and sufficient for linear time-series
systems. For nonlinear dependencies users can plug in any callable
``(x, y, Z) -> p_value``.

References
----------
Runge, J., Nowack, P., Kretschmer, M., Flaxman, S. & Sejdinovic, D.
(2019). "Detecting and quantifying causal associations in large
nonlinear time series datasets." *Science Advances*, 5(11). [@runge2019detecting]

Runge, J. (2020). "Discovering contemporaneous and lagged causal
relations in autocorrelated nonlinear time series datasets."
*Conference on Uncertainty in Artificial Intelligence (UAI)*.

Spirtes, P., Glymour, C. & Scheines, R. (2000). *Causation,
Prediction, and Search*, 2nd ed. [@spirtes2000causation]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════
#  Conditional-independence test (default: partial correlation)
# ═══════════════════════════════════════════════════════════════════════

def partial_corr_pvalue(
    x: np.ndarray, y: np.ndarray, Z: Optional[np.ndarray] = None,
) -> float:
    """
    Partial-correlation p-value for H0: ``X ⟂ Y | Z``.

    Residualises X and Y on Z via OLS, then applies a Fisher-z
    transform to the residual correlation with df = n - |Z| - 2.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = len(x)
    if Z is None or (hasattr(Z, "size") and Z.size == 0):
        r, _ = stats.pearsonr(x, y)
        k = 0
    else:
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        k = Z.shape[1]
        # Regress x and y on Z; correlate residuals
        Z_aug = np.column_stack([np.ones(n), Z])
        beta_x, *_ = np.linalg.lstsq(Z_aug, x, rcond=None)
        beta_y, *_ = np.linalg.lstsq(Z_aug, y, rcond=None)
        rx = x - Z_aug @ beta_x
        ry = y - Z_aug @ beta_y
        # Pearson on residuals
        if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
            return 1.0
        r, _ = stats.pearsonr(rx, ry)
    # Fisher-z — SE uses effective sample size (n - |Z| - 3), not df - 1.
    # For partial correlation with k conditioning variables, the
    # standard error of z = atanh(r) is 1 / sqrt(n - k - 3).
    r = np.clip(r, -0.999999, 0.999999)
    n_eff = n - k - 3
    if n_eff <= 0:
        return 1.0
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(n_eff)
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return p


# ═══════════════════════════════════════════════════════════════════════
#  Lag-matrix helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_lagged(X: np.ndarray, tau_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given X of shape (T, d), return (target_mat, lagged_mat) where:

    * target_mat has shape (T - tau_max, d) — aligned X_t.
    * lagged_mat has shape (T - tau_max, d * tau_max) with columns in
      the order: [var0_lag1, var1_lag1, ..., vard_lag1, var0_lag2, ...].
    """
    T, d = X.shape
    if tau_max < 1:
        raise ValueError("tau_max must be >= 1")
    if T <= tau_max + 1:
        raise ValueError(f"Need at least tau_max+2 = {tau_max+2} time points.")
    Y = X[tau_max:]  # (T - tau_max, d)
    blocks = []
    for lag in range(1, tau_max + 1):
        blocks.append(X[tau_max - lag : T - lag])
    lagged = np.hstack(blocks)  # (T - tau_max, d * tau_max)
    return Y, lagged


def _coord(var: int, lag: int, d: int) -> int:
    """Column index in lagged_mat for variable ``var`` at lag ``lag``."""
    return (lag - 1) * d + var


# ═══════════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PCMCIResult:
    """PCMCI output — lag-specific adjacency + discovered links."""

    variables: List[str]
    tau_max: int
    alpha: float
    p_matrix: np.ndarray   # shape (tau_max + 1, d, d)  p[lag, i, j]: edge X_i^{t-lag} → X_j^t
    val_matrix: np.ndarray # partial correlation strength, same shape
    adjacency: np.ndarray  # boolean, same shape
    n_effective: int
    method: str = "PCMCI (Runge et al. 2019)"

    def discovered_links(self) -> pd.DataFrame:
        """Return a DataFrame of significant links sorted by strength."""
        rows = []
        tau_max, d, _ = self.p_matrix.shape
        for lag in range(1, tau_max):
            for i in range(d):
                for j in range(d):
                    if not self.adjacency[lag, i, j]:
                        continue
                    rows.append({
                        "source": self.variables[i],
                        "target": self.variables[j],
                        "lag": lag,
                        "partial_corr": float(self.val_matrix[lag, i, j]),
                        "p_value": float(self.p_matrix[lag, i, j]),
                    })
        out = pd.DataFrame(rows)
        if len(out):
            out = out.sort_values(
                by=["target", "p_value"],
            ).reset_index(drop=True)
        return out

    def summary(self) -> str:
        n_links = int(self.adjacency.sum())
        return (
            f"{self.method}\n"
            f"  variables : {self.variables}\n"
            f"  tau_max   : {self.tau_max}\n"
            f"  α          : {self.alpha}\n"
            f"  N (effective): {self.n_effective}\n"
            f"  significant links: {n_links}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()


# ═══════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════

def pcmci(
    data: pd.DataFrame,
    variables: Optional[Sequence[str]] = None,
    *,
    tau_max: int = 3,
    pc_alpha: float = 0.05,
    mci_alpha: Optional[float] = None,
    ci_test: Optional[Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]] = None,
    max_conds_dim: Optional[int] = None,
    verbose: bool = False,
) -> PCMCIResult:
    """
    PCMCI causal discovery for stationary time-series.

    Parameters
    ----------
    data : DataFrame
        Time-series DataFrame with one row per time step. Must be
        strictly ordered (rows are consecutive time points).
    variables : list of str, optional
        Columns to use. Defaults to all columns.
    tau_max : int, default 3
        Maximum lag to consider for parent candidates.
    pc_alpha : float, default 0.05
        Significance threshold used during the PC1 selection stage.
    mci_alpha : float, optional
        Significance threshold for the final MCI adjacency. Defaults
        to ``pc_alpha``.
    ci_test : callable, optional
        Custom CI test ``(x, y, Z) -> p_value``. Defaults to
        :func:`partial_corr_pvalue`.
    max_conds_dim : int, optional
        Hard cap on the conditioning-set size during PC1. ``None``
        means no cap (the algorithm stops automatically when no
        predictors remain).
    verbose : bool, default False

    Returns
    -------
    PCMCIResult

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.pcmci(df_ts, variables=['gdp', 'inflation', 'rates'],
    ...                tau_max=4, pc_alpha=0.01)
    >>> res.discovered_links()        # DataFrame of all significant links
    """
    if variables is None:
        variables = list(data.columns)
    else:
        variables = list(variables)

    X = data[variables].to_numpy(dtype=float)
    T, d = X.shape
    if mci_alpha is None:
        mci_alpha = pc_alpha
    ci = ci_test or partial_corr_pvalue

    Y, L = _make_lagged(X, tau_max)
    n_eff = len(Y)

    # -----------------------------------------------------------
    # PC1 stage: for each target variable j at lag 0, iteratively
    # prune candidate parents (i, τ) whose CI test with j becomes
    # non-significant conditional on the currently retained parents.
    # -----------------------------------------------------------
    parents: List[List[Tuple[int, int]]] = [[] for _ in range(d)]
    for j in range(d):
        # Initial candidate set = all (i, τ) with τ ∈ 1..tau_max
        cand = [(i, tau) for tau in range(1, tau_max + 1) for i in range(d)]

        # Marginal test to seed ordering
        marg_pvals = {}
        for (i, tau) in cand:
            p = ci(L[:, _coord(i, tau, d)], Y[:, j], None)
            marg_pvals[(i, tau)] = p
        cand = [c for c in cand if marg_pvals[c] < pc_alpha]
        cand.sort(key=lambda c: marg_pvals[c])

        if verbose:  # pragma: no cover
            print(f"[PC1] j={j} initial candidates: {len(cand)}")

        # Iteratively test conditional on a growing conditioning set
        # (topological discovery: at each iteration, use the s-strongest
        # currently retained parents as conditioning set for the next
        # candidate). We stop when no candidate can be dropped.
        changed = True
        pass_num = 0
        while changed:
            changed = False
            pass_num += 1
            for c in list(cand):
                cond = [cc for cc in cand if cc != c]
                if max_conds_dim is not None:
                    cond = cond[:max_conds_dim]
                if not cond:
                    break
                Z = np.column_stack(
                    [L[:, _coord(i, t, d)] for (i, t) in cond]
                )
                p = ci(L[:, _coord(c[0], c[1], d)], Y[:, j], Z)
                if p >= pc_alpha:
                    cand.remove(c)
                    changed = True
                    if verbose:  # pragma: no cover
                        print(f"[PC1] pass {pass_num}: drop {c} (p={p:.3g})")
            # Refresh candidate ordering periodically — not strictly
            # necessary but improves the practical convergence rate.
        parents[j] = cand

    # -----------------------------------------------------------
    # MCI stage: for each candidate link X_i^{t-τ} → X_j^t, test
    # conditional on parents(j) ∪ parents(i at lag τ).
    # Parents(i at lag τ) in lagged form are the PC1 parents of i
    # shifted by τ — i.e. (k, σ + τ) for each (k, σ) ∈ parents(i).
    # -----------------------------------------------------------
    p_matrix = np.ones((tau_max + 1, d, d))
    val_matrix = np.zeros((tau_max + 1, d, d))

    for j in range(d):
        for i in range(d):
            for tau in range(1, tau_max + 1):
                # Conditioning: parents(j) ∪ shifted parents(i by τ)
                cond_coords: List[Tuple[int, int]] = list(parents[j])
                for (k, sigma) in parents[i]:
                    new_lag = sigma + tau
                    if 1 <= new_lag <= tau_max and (k, new_lag) not in cond_coords:
                        cond_coords.append((k, new_lag))
                # Remove the candidate itself if it slipped in
                cond_coords = [c for c in cond_coords if c != (i, tau)]

                x_arr = L[:, _coord(i, tau, d)]
                y_arr = Y[:, j]
                if cond_coords:
                    Z = np.column_stack(
                        [L[:, _coord(k, s, d)] for (k, s) in cond_coords]
                    )
                else:
                    Z = None

                p = ci(x_arr, y_arr, Z)
                p_matrix[tau, i, j] = p

                # Also store the residual correlation magnitude
                if Z is None:
                    r, _ = stats.pearsonr(x_arr, y_arr)
                else:
                    Z_aug = np.column_stack([np.ones(len(x_arr)), Z])
                    bx, *_ = np.linalg.lstsq(Z_aug, x_arr, rcond=None)
                    by, *_ = np.linalg.lstsq(Z_aug, y_arr, rcond=None)
                    rx = x_arr - Z_aug @ bx
                    ry = y_arr - Z_aug @ by
                    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
                        r = 0.0
                    else:
                        r, _ = stats.pearsonr(rx, ry)
                val_matrix[tau, i, j] = r

    adjacency = p_matrix < mci_alpha
    adjacency[0] = False  # lag-0 is not a causal link here

    return PCMCIResult(
        variables=list(variables),
        tau_max=tau_max,
        alpha=mci_alpha,
        p_matrix=p_matrix,
        val_matrix=val_matrix,
        adjacency=adjacency,
        n_effective=n_eff,
    )


__all__ = ["pcmci", "PCMCIResult", "partial_corr_pvalue"]
