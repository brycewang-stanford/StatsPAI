"""
PCMCI: causal discovery for time-series (Runge et al. 2019).

A two-stage algorithm specialised for stationary multivariate time-series
with auto-correlation and cross-correlation:

1. **PC1 stage (parent selection)**. For each variable ``X_j^t`` start from
   every lagged predictor ``{X_k^{t-τ} : k, τ ∈ 1..τ_max}``. For
   ``p = 0, 1, 2, ...`` test each remaining candidate given the ``p``
   strongest *other* candidates (one conditioning set per candidate, hence
   PC1), drop the candidates that are not significant at ``pc_alpha`` once
   the pass is over, and re-sort the survivors by the smallest absolute
   partial correlation seen so far. Stop when fewer than ``p`` other
   candidates are left.
2. **MCI stage (momentary conditional independence)**. For each candidate
   link ``X_i^{t-τ} → X_j^t``, test conditional independence given the
   parents of ``X_j^t`` and the parents of ``X_i^{t-τ}``, that is the
   parents of ``X_i`` shifted by ``τ``, which reach back to lag
   ``2 τ_max``. This controls false positives driven by autocorrelation
   that PC alone cannot suppress.

Every test of both stages uses the same ``T - 2 τ_max`` time points, so
p-values are comparable across links. With a t-reference partial
correlation test the parents, p-values and partial correlations are those
of ``tigramite`` (``PCMCI.run_pcmci`` with ``ParCorr``;
``tests/reference_parity/test_pcmci_tigramite_parity.py``).

The output is a lag-specific tensor ``p_matrix[lag, i, j]`` of
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
nonlinear time series datasets." *Science Advances*, 5(11).
[@runge2019detecting]

Runge, J. (2020). "Discovering contemporaneous and lagged causal
relations in autocorrelated nonlinear time series datasets."
*Conference on Uncertainty in Artificial Intelligence (UAI)*.

Spirtes, P., Glymour, C. & Scheines, R. (2000). *Causation,
Prediction, and Search*, 2nd ed. [@spirtes2000causation]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin

# ═══════════════════════════════════════════════════════════════════════
#  Conditional-independence test (default: partial correlation)
# ═══════════════════════════════════════════════════════════════════════


def partial_corr_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    Z: Optional[np.ndarray] = None,
) -> float:
    """
    Partial-correlation p-value for H0: ``X ⟂ Y | Z``.

    Residualises X and Y on Z via OLS, then applies a Fisher-z
    transform to the residual correlation with df = n - |Z| - 2.

    Examples
    --------
    Two variables sharing a common driver ``z`` are marginally
    correlated but conditionally independent given ``z``:

    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> z = rng.normal(size=200)
    >>> x = z + rng.normal(size=200)
    >>> y = z + rng.normal(size=200)
    >>> bool(sp.partial_corr_pvalue(x, y) < 0.05)        # marginally dependent
    True
    >>> bool(sp.partial_corr_pvalue(x, y, z) > 0.05)     # independent given z
    True
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
    p = float(2.0 * stats.norm.sf(abs(z)))
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
        raise ValueError(f"Need at least tau_max+2 = {tau_max + 2} time points.")
    Y = X[tau_max:]  # (T - tau_max, d)
    blocks = []
    for lag in range(1, tau_max + 1):
        blocks.append(X[tau_max - lag : T - lag])
    lagged = np.hstack(blocks)  # (T - tau_max, d * tau_max)
    return Y, lagged


def _partial_corr(x: np.ndarray, y: np.ndarray, Z: Optional[np.ndarray]) -> float:
    """Correlation of ``x`` and ``y`` after a least-squares fit on ``Z``."""
    if Z is None:
        rx, ry = x - x.mean(), y - y.mean()
    else:
        Z_aug = np.column_stack([np.ones(len(x)), Z])
        rx = x - Z_aug @ np.linalg.lstsq(Z_aug, x, rcond=None)[0]
        ry = y - Z_aug @ np.linalg.lstsq(Z_aug, y, rcond=None)[0]
    sx, sy = float(np.sqrt(rx @ rx)), float(np.sqrt(ry @ ry))
    scale = max(float(np.sqrt(x @ x)), float(np.sqrt(y @ y)), 1e-300)
    if sx <= 1e-12 * scale or sy <= 1e-12 * scale:
        return 0.0
    return float((rx @ ry) / (sx * sy))


def _coord(var: int, lag: int, d: int) -> int:
    """Column index in lagged_mat for variable ``var`` at lag ``lag``."""
    return (lag - 1) * d + var


# ═══════════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PCMCIResult(ResultProtocolMixin):
    """PCMCI output — lag-specific adjacency + discovered links.

    Returned by :func:`pcmci`. Bundles the lag-specific p-value tensor
    (``p_matrix``), the partial-correlation strengths (``val_matrix``),
    the boolean ``adjacency`` decision tensor, and the effective sample
    size. Call :meth:`discovered_links` for a tidy DataFrame of the
    significant lagged links and :meth:`summary` for a one-screen report.

    Examples
    --------
    A two-variable system where lagged GDP drives inflation:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> T = 120
    >>> gdp = np.zeros(T)
    >>> inflation = np.zeros(T)
    >>> eg = rng.normal(0, 1, T)
    >>> ei = rng.normal(0, 1, T)
    >>> for t in range(1, T):
    ...     gdp[t] = 0.5 * gdp[t - 1] + eg[t]
    ...     inflation[t] = 0.4 * gdp[t - 1] + 0.3 * inflation[t - 1] + ei[t]
    >>> df = pd.DataFrame({"gdp": gdp, "inflation": inflation})
    >>> res = sp.pcmci(df, tau_max=2, pc_alpha=0.05)
    >>> isinstance(res, sp.PCMCIResult)
    True
    >>> links = res.discovered_links()
    >>> list(links.columns)
    ['source', 'target', 'lag', 'partial_corr', 'p_value']
    >>> bool(((links["source"] == "gdp") &
    ...       (links["target"] == "inflation")).any())
    True
    """

    variables: List[str]
    tau_max: int
    alpha: float
    # shape (tau_max + 1, d, d); p[lag, i, j] tests X_i^{t-lag} -> X_j^t
    p_matrix: np.ndarray
    val_matrix: np.ndarray  # partial correlation strength, same shape
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
                    rows.append(
                        {
                            "source": self.variables[i],
                            "target": self.variables[j],
                            "lag": lag,
                            "partial_corr": float(self.val_matrix[lag, i, j]),
                            "p_value": float(self.p_matrix[lag, i, j]),
                        }
                    )
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
    ci_test: Optional[
        Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
    ] = None,
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
        :func:`partial_corr_pvalue`, the Fisher z test. ``tigramite``'s
        ``ParCorr`` refers the same partial correlation to a t
        distribution with ``n - dim(Z) - 2`` degrees of freedom; pass
        that test here to reproduce its p-values. Candidates are ranked
        by absolute partial correlation whatever the test.
    max_conds_dim : int, optional
        Hard cap on the conditioning-set size during PC1. ``None``
        means no cap (the algorithm stops automatically when fewer
        candidates than conditions remain).
    verbose : bool, default False

    Returns
    -------
    PCMCIResult

    Examples
    --------
    >>> import statspai as sp, numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> df_ts = pd.DataFrame(rng.normal(size=(120, 3)),
    ...                      columns=["gdp", "inflation", "rates"])
    >>> res = sp.pcmci(df_ts, variables=["gdp", "inflation", "rates"],
    ...                tau_max=2, pc_alpha=0.1)
    >>> links = res.discovered_links()  # DataFrame of all significant links
    >>> list(links.columns)
    ['source', 'target', 'lag', 'partial_corr', 'p_value']
    """
    if variables is None:
        variables = list(data.columns)
    else:
        variables = list(variables)

    from ._data import check_alpha, check_no_missing

    check_no_missing(data[variables], "pcmci")
    X = data[variables].to_numpy(dtype=float)
    T, d = X.shape
    if mci_alpha is None:
        mci_alpha = pc_alpha
    check_alpha(pc_alpha, "pcmci", "pc_alpha")
    check_alpha(mci_alpha, "pcmci", "mci_alpha")
    ci = ci_test or partial_corr_pvalue
    if T <= 2 * tau_max + 2:
        from ..exceptions import DataInsufficient

        raise DataInsufficient(
            f"pcmci: {T} time points are too few for tau_max={tau_max}. The "
            "parents of a lagged variable reach back to lag 2 * tau_max, so "
            f"more than {2 * tau_max + 2} time points are needed.",
            recovery_hint="Lower tau_max or use a longer series.",
            diagnostics={"n_time": int(T), "tau_max": int(tau_max)},
        )
    # One sample for every test: the time points with 2 * tau_max lags
    # behind them. A link at lag tau is tested given the parents of the
    # lagged variable, which reach back to lag tau + tau_max.
    Y, L = _make_lagged(X, 2 * tau_max)
    n_eff = len(Y)

    def col(i: int, tau: int) -> np.ndarray:
        return L[:, _coord(i, tau, d)]

    def stack(coords: List[Tuple[int, int]]) -> Optional[np.ndarray]:
        if not coords:
            return None
        return np.column_stack([col(k, s) for (k, s) in coords])

    # -----------------------------------------------------------
    # PC1 stage (Runge et al. 2019, Algorithm S1). For p = 0, 1, ...
    # each remaining candidate is tested given the p strongest other
    # candidates; non-significant ones leave after the pass, and the
    # rest are re-sorted by the smallest |partial correlation| seen.
    # -----------------------------------------------------------
    parents: List[List[Tuple[int, int]]] = [[] for _ in range(d)]
    for j in range(d):
        cand = [(i, tau) for i in range(d) for tau in range(1, tau_max + 1)]
        val_min = {c: np.inf for c in cand}
        p_dim = 0
        while len(cand) - 1 >= p_dim:
            if max_conds_dim is not None and p_dim > max_conds_dim:
                break
            dropped = []
            for c in cand:
                cond = [o for o in cand if o != c][:p_dim]
                Z = stack(cond)
                x_arr = col(c[0], c[1])
                val_min[c] = min(val_min[c], abs(_partial_corr(x_arr, Y[:, j], Z)))
                if ci(x_arr, Y[:, j], Z) > pc_alpha:
                    dropped.append(c)
            if verbose and dropped:  # pragma: no cover
                print(f"[PC1] j={j} p={p_dim}: drop {dropped}")
            cand = [c for c in cand if c not in dropped]
            cand.sort(key=lambda c: -val_min[c])
            p_dim += 1
        parents[j] = cand

    # -----------------------------------------------------------
    # MCI stage: for each candidate link X_i^{t-τ} → X_j^t, test
    # conditional on parents(j) and on parents(i) shifted by τ, that
    # is (k, σ + τ) for each (k, σ) ∈ parents(i).
    # -----------------------------------------------------------
    p_matrix = np.ones((tau_max + 1, d, d))
    val_matrix = np.zeros((tau_max + 1, d, d))
    for j in range(d):
        for i in range(d):
            for tau in range(1, tau_max + 1):
                cond_coords = [c for c in parents[j] if c != (i, tau)]
                for k, sigma in parents[i]:
                    shifted = (k, sigma + tau)
                    if shifted not in cond_coords:
                        cond_coords.append(shifted)
                x_arr = col(i, tau)
                y_arr = Y[:, j]
                Z_cond = stack(cond_coords)
                p_matrix[tau, i, j] = ci(x_arr, y_arr, Z_cond)
                val_matrix[tau, i, j] = _partial_corr(x_arr, y_arr, Z_cond)

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
