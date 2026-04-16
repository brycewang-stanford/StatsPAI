"""
Weak-IV-robust confidence *sets* for a single endogenous regressor.

These routines invert weak-IV-robust tests across a grid of candidate
values of β₀ to yield confidence sets that remain valid regardless of
instrument strength. Three variants:

- :func:`anderson_rubin_ci` — AR (1949) test by grid inversion.
- :func:`conditional_lr_ci` — Moreira (2003) CLR test by grid inversion
  (uniformly most powerful invariant in the single-endogenous case).
- :func:`k_test_ci`         — Kleibergen (2002, 2005) K/K-J test by grid
  inversion; faster than CLR and also weak-IV-robust.

When the first-stage F is large, all three sets collapse to the usual
normal Wald CI; when identification is weak, they can be much wider
(or even unbounded / disconnected), which is the *correct* behaviour.

References
----------
Anderson, T.W. and Rubin, H. (1949).
    "Estimation of the Parameters of a Single Equation in a Complete
    System of Stochastic Equations." *AMS*, 20(1), 46-63.

Moreira, M.J. (2003). "A conditional likelihood ratio test for structural
    models." *Econometrica*, 71(4), 1027-1048.

Kleibergen, F. (2002). "Pivotal statistics for testing structural
    parameters in instrumental variables regression."
    *Econometrica*, 70(5), 1781-1803.

Kleibergen, F. (2005). "Testing parameters in GMM without assuming
    that they are identified." *Econometrica*, 73(4), 1103-1123.

Andrews, I., Stock, J.H. and Sun, L. (2019). "Weak Instruments in IV
    Regression: Theory and Practice." *Annual Review of Economics*, 11.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class WeakIVConfidenceSet:
    method: str
    level: float
    beta_grid: np.ndarray
    statistic: np.ndarray
    critical_value: np.ndarray
    in_set: np.ndarray
    lower: float
    upper: float
    is_empty: bool
    is_connected: bool
    is_unbounded: bool
    extra: dict

    def as_intervals(self) -> List[Tuple[float, float]]:
        """Return the CI as a list of (lo, hi) intervals (handles disconnection)."""
        if self.is_empty:
            return []
        intervals = []
        in_set = self.in_set
        grid = self.beta_grid
        i = 0
        n = len(grid)
        while i < n:
            if in_set[i]:
                j = i
                while j + 1 < n and in_set[j + 1]:
                    j += 1
                intervals.append((float(grid[i]), float(grid[j])))
                i = j + 1
            else:
                i += 1
        return intervals

    def summary(self) -> str:
        intervals = self.as_intervals()
        lines = [
            f"{self.method} — weak-IV-robust confidence set",
            "-" * 60,
            f"  level                : {int(self.level * 100)}%",
            f"  grid                 : {len(self.beta_grid)} points on "
            f"[{self.beta_grid[0]:.3f}, {self.beta_grid[-1]:.3f}]",
        ]
        if self.is_empty:
            lines.append("  confidence set       : EMPTY  ← mis-specification?")
        elif len(intervals) == 1:
            lo, hi = intervals[0]
            lines.append(f"  confidence set       : [{lo:.4f}, {hi:.4f}]")
        else:
            pieces = " ∪ ".join(f"[{lo:.3f}, {hi:.3f}]" for lo, hi in intervals)
            lines.append(f"  confidence set       : {pieces}   (disconnected!)")
        if self.is_unbounded:
            lines.append("  NOTE                 : CI touches grid boundary — may be unbounded.")
        return "\n".join(lines)


def _grab(v, data, cols=False):
    if isinstance(v, str):
        return data[v].values.astype(float)
    if cols and isinstance(v, list) and all(isinstance(x, str) for x in v):
        return data[v].values.astype(float)
    return np.asarray(v, dtype=float)


def _residualize(M: np.ndarray, W: np.ndarray) -> np.ndarray:
    if W.size == 0 or W.shape[1] == 0:
        return M
    b, *_ = np.linalg.lstsq(W, M, rcond=None)
    return M - W @ b


def _prep(y, endog, instruments, exog, data, add_const):
    Y = _grab(y, data).reshape(-1)
    D = _grab(endog, data).reshape(-1)
    Z = _grab(instruments, data, cols=True)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = len(Y)
    if exog is None:
        W = np.ones((n, 1)) if add_const else np.empty((n, 0))
    else:
        Wx = _grab(exog, data, cols=True)
        if Wx.ndim == 1:
            Wx = Wx.reshape(-1, 1)
        W = np.column_stack([np.ones(n), Wx]) if add_const else Wx
    Yt = _residualize(Y.reshape(-1, 1), W).ravel()
    Dt = _residualize(D.reshape(-1, 1), W).ravel()
    Zt = _residualize(Z, W)
    return Yt, Dt, Zt, W.shape[1], n


def _default_grid(Yt: np.ndarray, Dt: np.ndarray, Zt: np.ndarray,
                  n_points: int) -> np.ndarray:
    """β grid centered on 2SLS ± 10 × conservative SE."""
    PZ = Zt @ np.linalg.solve(Zt.T @ Zt, Zt.T)
    D_hat = PZ @ Dt
    denom = float(D_hat @ Dt)
    if abs(denom) < 1e-12:
        # instruments totally irrelevant — center grid on 0 with broad span
        se = np.std(Yt) / (np.std(Dt) + 1e-12)
        return np.linspace(-10 * se, 10 * se, n_points)
    b2sls = float(D_hat @ Yt) / denom
    se = np.std(Yt - b2sls * Dt) / (np.std(D_hat) + 1e-12) / np.sqrt(len(Yt))
    return np.linspace(b2sls - 10 * se, b2sls + 10 * se, n_points)


# ═══════════════════════════════════════════════════════════════════════
#  AR confidence set
# ═══════════════════════════════════════════════════════════════════════

def anderson_rubin_ci(
    y: Union[np.ndarray, pd.Series, str],
    endog: Union[np.ndarray, pd.Series, str],
    instruments: Union[np.ndarray, pd.DataFrame, List[str]],
    exog: Optional[Union[np.ndarray, pd.DataFrame, List[str]]] = None,
    data: Optional[pd.DataFrame] = None,
    level: float = 0.95,
    n_grid: int = 401,
    beta_grid: Optional[np.ndarray] = None,
    add_const: bool = True,
) -> WeakIVConfidenceSet:
    """
    Anderson-Rubin (1949) confidence set by grid inversion.

    For each candidate β₀, compute the AR F-statistic

        AR(β₀) = (u₀' P_Z u₀ / k) / (u₀' M_Z u₀ / (n - k - kW))
        u₀ = y - β₀ · d  (partialled out of exogenous controls)

    and include β₀ in the CI whenever ``AR(β₀) ≤ F_{k, n-k-kW}^{1-α}``.

    Valid under any instrument strength. Under weak identification the
    set can be disconnected or unbounded — we flag both.
    """
    Yt, Dt, Zt, kW, n = _prep(y, endog, instruments, exog, data, add_const)
    k = Zt.shape[1]
    dfd = max(n - kW - k, 1)
    crit = stats.f.ppf(level, k, dfd)

    if beta_grid is None:
        beta_grid = _default_grid(Yt, Dt, Zt, n_grid)

    stats_arr = np.empty_like(beta_grid, dtype=float)
    for i, b0 in enumerate(beta_grid):
        u0 = Yt - b0 * Dt
        pi, *_ = np.linalg.lstsq(Zt, u0, rcond=None)
        u_hat = Zt @ pi
        rss_full = float((u0 - u_hat) @ (u0 - u_hat))
        rss_red = float(u0 @ u0)
        if rss_full > 0:
            stats_arr[i] = ((rss_red - rss_full) / k) / (rss_full / dfd)
        else:
            stats_arr[i] = np.inf
    in_set = stats_arr <= crit

    return _build_set("Anderson-Rubin (AR)", level, beta_grid, stats_arr,
                      np.full_like(stats_arr, crit), in_set,
                      extra={"df_num": k, "df_denom": dfd})


# ═══════════════════════════════════════════════════════════════════════
#  CLR (Moreira 2003) confidence set by grid inversion
# ═══════════════════════════════════════════════════════════════════════

def conditional_lr_ci(
    y: Union[np.ndarray, pd.Series, str],
    endog: Union[np.ndarray, pd.Series, str],
    instruments: Union[np.ndarray, pd.DataFrame, List[str]],
    exog: Optional[Union[np.ndarray, pd.DataFrame, List[str]]] = None,
    data: Optional[pd.DataFrame] = None,
    level: float = 0.95,
    n_grid: int = 201,
    beta_grid: Optional[np.ndarray] = None,
    n_sim: int = 5000,
    add_const: bool = True,
    random_state: Optional[int] = None,
) -> WeakIVConfidenceSet:
    """
    Moreira (2003) CLR confidence set by grid inversion.

    At each candidate β₀, compute the CLR statistic and its conditional
    critical value via Monte-Carlo given ``T'T``; include β₀ iff
    ``CLR(β₀) ≤ c(T'T, 1-α)``.

    Uniformly most powerful invariant under normal errors with a single
    endogenous regressor. Tight under strong ID, wide under weak ID.
    """
    Yt, Dt, Zt, kW, n = _prep(y, endog, instruments, exog, data, add_const)
    k = Zt.shape[1]
    # Orthonormalise instruments
    L = np.linalg.cholesky(Zt.T @ Zt)
    Zs = np.linalg.solve(L.T, Zt.T).T  # Zs'Zs = I_k

    if beta_grid is None:
        beta_grid = _default_grid(Yt, Dt, Zt, n_grid)

    rng = np.random.default_rng(random_state)
    m = int(n_sim)
    # Pre-sample a k × m normal matrix once and reuse for each β₀
    S_sim_base = rng.standard_normal((m, k))

    stat_arr = np.empty(len(beta_grid))
    crit_arr = np.empty(len(beta_grid))

    for i, b0 in enumerate(beta_grid):
        ustar = Yt - b0 * Dt
        # Reduced-form Sigma under H0
        M_Z = np.eye(n) - Zs @ Zs.T
        YD = np.column_stack([ustar, Dt])
        df_r = max(n - kW - k, 1)
        Sigma = YD.T @ M_Z @ YD / df_r
        suu = float(Sigma[0, 0])
        svv = float(Sigma[1, 1])
        suv = float(Sigma[0, 1])
        if suu <= 0 or svv <= 0:
            stat_arr[i] = 0.0
            crit_arr[i] = np.inf
            continue
        S = Zs.T @ ustar / np.sqrt(suu)
        d_perp = Dt - (suv / suu) * ustar
        sperp = max(svv - suv ** 2 / suu, 1e-12)
        T = Zs.T @ d_perp / np.sqrt(sperp)
        ar = float(S @ S)
        qt = float(T @ T)
        lm = float((S @ T) ** 2 / max(qt, 1e-12))
        clr = 0.5 * (ar - qt + np.sqrt(max((ar + qt) ** 2 - 4 * (ar * qt - lm * qt), 0.0)))

        # Conditional critical value at the observed qt
        # Simulate S ~ N(0, I_k) independent of T (under H0)
        # LM_sim = (S'T)^2 / qt = s1_sim^2 where s1 is coord along T direction
        T_dir = T / max(np.linalg.norm(T), 1e-12)
        s1 = S_sim_base @ T_dir
        s_rest_sq = np.sum(S_sim_base ** 2, axis=1) - s1 ** 2
        ar_sim = s1 ** 2 + s_rest_sq
        lm_sim = s1 ** 2
        clr_sim = 0.5 * (
            ar_sim - qt + np.sqrt(
                np.maximum((ar_sim + qt) ** 2 - 4 * (ar_sim * qt - lm_sim * qt), 0.0)
            )
        )
        crit = float(np.quantile(clr_sim, level))
        stat_arr[i] = clr
        crit_arr[i] = crit

    in_set = stat_arr <= crit_arr
    return _build_set("Moreira CLR", level, beta_grid, stat_arr, crit_arr,
                      in_set, extra={"n_sim": n_sim})


# ═══════════════════════════════════════════════════════════════════════
#  Kleibergen K test CI
# ═══════════════════════════════════════════════════════════════════════

def k_test_ci(
    y: Union[np.ndarray, pd.Series, str],
    endog: Union[np.ndarray, pd.Series, str],
    instruments: Union[np.ndarray, pd.DataFrame, List[str]],
    exog: Optional[Union[np.ndarray, pd.DataFrame, List[str]]] = None,
    data: Optional[pd.DataFrame] = None,
    level: float = 0.95,
    n_grid: int = 401,
    beta_grid: Optional[np.ndarray] = None,
    add_const: bool = True,
) -> WeakIVConfidenceSet:
    """
    Kleibergen (2002) K-test confidence set by grid inversion.

    The K-statistic projects the AR score onto the (estimated) score
    direction of β, giving a 1-df χ²-valued pivot even under weak ID.

        K(β₀)  =  n · (score_β|β₀)² / var
               ≈  (S'T)² / (T'T)

    where S and T are the AR score and "T-statistic" from Moreira (2003).
    Faster than CLR but slightly less powerful; still weak-IV-robust.
    """
    Yt, Dt, Zt, kW, n = _prep(y, endog, instruments, exog, data, add_const)
    k = Zt.shape[1]
    L = np.linalg.cholesky(Zt.T @ Zt)
    Zs = np.linalg.solve(L.T, Zt.T).T

    if beta_grid is None:
        beta_grid = _default_grid(Yt, Dt, Zt, n_grid)

    crit = stats.chi2.ppf(level, df=1)

    stat_arr = np.empty(len(beta_grid))
    M_Z = np.eye(n) - Zs @ Zs.T

    for i, b0 in enumerate(beta_grid):
        ustar = Yt - b0 * Dt
        YD = np.column_stack([ustar, Dt])
        df_r = max(n - kW - k, 1)
        Sigma = YD.T @ M_Z @ YD / df_r
        suu = float(Sigma[0, 0])
        svv = float(Sigma[1, 1])
        suv = float(Sigma[0, 1])
        if suu <= 0 or svv <= 0:
            stat_arr[i] = 0.0
            continue
        S = Zs.T @ ustar / np.sqrt(suu)
        d_perp = Dt - (suv / suu) * ustar
        sperp = max(svv - suv ** 2 / suu, 1e-12)
        T = Zs.T @ d_perp / np.sqrt(sperp)
        qt = float(T @ T)
        K_stat = float((S @ T) ** 2 / max(qt, 1e-12))
        stat_arr[i] = K_stat

    crit_arr = np.full_like(stat_arr, crit)
    in_set = stat_arr <= crit
    return _build_set("Kleibergen K", level, beta_grid, stat_arr, crit_arr,
                      in_set, extra={"df": 1})


# ═══════════════════════════════════════════════════════════════════════
#  Shared result builder
# ═══════════════════════════════════════════════════════════════════════

def _build_set(method, level, beta_grid, stat_arr, crit_arr, in_set,
               extra: dict) -> WeakIVConfidenceSet:
    if not in_set.any():
        lo = hi = np.nan
        is_empty = True
    else:
        lo = float(beta_grid[in_set].min())
        hi = float(beta_grid[in_set].max())
        is_empty = False

    # Detect disconnection: the set {β : in_set[i]} should be a contiguous run
    idx = np.where(in_set)[0]
    is_connected = (not is_empty) and (len(idx) == idx.max() - idx.min() + 1)
    # Unbounded hint: first or last grid point is in the set
    is_unbounded = (not is_empty) and (bool(in_set[0]) or bool(in_set[-1]))

    return WeakIVConfidenceSet(
        method=method,
        level=level,
        beta_grid=np.asarray(beta_grid, dtype=float),
        statistic=stat_arr.astype(float),
        critical_value=crit_arr.astype(float),
        in_set=in_set.astype(bool),
        lower=lo, upper=hi,
        is_empty=is_empty,
        is_connected=is_connected,
        is_unbounded=is_unbounded,
        extra=extra,
    )


__all__ = [
    "anderson_rubin_ci",
    "conditional_lr_ci",
    "k_test_ci",
    "WeakIVConfidenceSet",
]
