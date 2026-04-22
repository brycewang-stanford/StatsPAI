"""
Beyond the Average: Distributional Effects under Imperfect Compliance
(Byambadalai, Hirata, Oka & Yasui 2025, arXiv 2509.15594).

Estimates the distributional treatment effect on compliers when
treatment is partially observed (imperfect compliance, à la LATE).
Combines Imbens-Rubin (1997) Wald-style decomposition with quantile
indicators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BeyondAverageResult:
    """Distributional LATE on compliers."""
    quantiles: np.ndarray
    late_q: np.ndarray
    se_q: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    complier_share: float
    n_obs: int

    def summary(self) -> str:
        rows = [
            "Beyond-the-Average: Distributional LATE",
            "=" * 42,
            f"  N           : {self.n_obs}",
            f"  Complier sh.: {self.complier_share:.3f}",
            "  Quantile  LATE      SE       95% CI",
        ]
        for q, l, s, lo, hi in zip(
            self.quantiles, self.late_q, self.se_q,
            self.ci_low, self.ci_high
        ):
            rows.append(
                f"  {q:.2f}     {l:+.4f}  {s:.4f}  [{lo:+.4f}, {hi:+.4f}]"
            )
        return "\n".join(rows)


def beyond_average_late(
    data: pd.DataFrame,
    y: str,
    treat: str,
    instrument: str,
    quantiles: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 0,
) -> BeyondAverageResult:
    """
    Distributional LATE on compliers under imperfect compliance.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat, instrument : str
    quantiles : array-like, optional
    alpha : float
    n_boot : int
    seed : int

    Returns
    -------
    BeyondAverageResult
    """
    if quantiles is None:
        quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    df = data[[y, treat, instrument]].dropna().reset_index(drop=True)
    Y = df[y].to_numpy(float)
    D = df[treat].to_numpy(int)
    Z = df[instrument].to_numpy(int)
    n = len(df)
    rng = np.random.default_rng(seed)

    if Z.max() != 1 or Z.min() != 0:
        raise ValueError("Instrument must be binary (0/1).")

    complier_share = float(D[Z == 1].mean() - D[Z == 0].mean())
    if complier_share <= 0:
        raise ValueError(
            "Estimated complier share ≤ 0 — instrument fails monotonicity."
        )

    # ------------------------------------------------------------------ #
    #  Abadie (2002) κ-weighted complier-subpopulation CDFs + quantile
    #  inversion.  Without covariates, Abadie's κ reduces to the
    #  Imbens-Angrist Wald identity per CDF:
    #
    #    F_{Y_1 | c}(y) = [P(Y <= y, D = 1 | Z = 1)
    #                       - P(Y <= y, D = 1 | Z = 0)] / Δp
    #    F_{Y_0 | c}(y) = [P(Y <= y, D = 0 | Z = 0)
    #                       - P(Y <= y, D = 0 | Z = 1)] / (-Δp_0)
    #
    #  where Δp = P(D = 1 | Z = 1) - P(D = 1 | Z = 0) (complier share).
    #  The complier QTE at level q is Q_{1,c}(q) - Q_{0,c}(q).
    # ------------------------------------------------------------------ #

    def _complier_cdfs(Yi: np.ndarray, Di: np.ndarray, Zi: np.ndarray):
        """Return (grid, F1_c, F0_c) monotone CDFs on a shared y-grid."""
        dp = (Di[Zi == 1].mean() - Di[Zi == 0].mean())
        if abs(dp) < 1e-8:
            return None
        # Shared y-grid at unique observed values, sorted.
        grid = np.sort(np.unique(Yi))
        # Empirical joint CDFs: F^{z, d}(y) = P(Y <= y, D = d | Z = z)
        p_z1 = float(np.mean(Zi == 1))
        p_z0 = float(np.mean(Zi == 0))
        if p_z1 < 1e-8 or p_z0 < 1e-8:
            return None
        # Build F1_c(y) = [sum(Y<=y, D=1, Z=1)/n_z1 - sum(Y<=y, D=1, Z=0)/n_z0] / dp
        # by sorted cumsum over the y-grid.
        order = np.argsort(Yi)
        y_s = Yi[order]
        d_s = Di[order]
        z_s = Zi[order]
        n_z1 = int(np.sum(Zi == 1))
        n_z0 = int(np.sum(Zi == 0))
        if n_z1 == 0 or n_z0 == 0:
            return None
        # Cumulative counts
        cum_d1z1 = np.cumsum((d_s == 1) & (z_s == 1))
        cum_d1z0 = np.cumsum((d_s == 1) & (z_s == 0))
        cum_d0z0 = np.cumsum((d_s == 0) & (z_s == 0))
        cum_d0z1 = np.cumsum((d_s == 0) & (z_s == 1))
        # For each grid value, pick the last-y-index <= grid value
        idxs = np.searchsorted(y_s, grid, side='right') - 1
        idxs = np.clip(idxs, 0, len(y_s) - 1)
        F1_c = (cum_d1z1[idxs] / n_z1 - cum_d1z0[idxs] / n_z0) / dp
        F0_c = (cum_d0z0[idxs] / n_z0 - cum_d0z1[idxs] / n_z1) / dp
        # Enforce monotonicity + [0, 1] bounds (isotonic clip).
        F1_c = np.clip(np.maximum.accumulate(F1_c), 0.0, 1.0)
        F0_c = np.clip(np.maximum.accumulate(F0_c), 0.0, 1.0)
        return grid, F1_c, F0_c

    def _invert_cdf(grid: np.ndarray, F: np.ndarray, q: float) -> float:
        """Empirical quantile: smallest y such that F(y) >= q."""
        if not len(grid):
            return np.nan
        idx = np.searchsorted(F, q, side='left')
        idx = min(int(idx), len(grid) - 1)
        return float(grid[idx])

    def _late_q(Yi, Di, Zi, q):
        cdfs = _complier_cdfs(Yi, Di, Zi)
        if cdfs is None:
            return np.nan
        grid, F1, F0 = cdfs
        q1 = _invert_cdf(grid, F1, q)
        q0 = _invert_cdf(grid, F0, q)
        return float(q1 - q0)

    late_q = np.array([_late_q(Y, D, Z, q) for q in quantiles])

    # Bootstrap SE
    boot = np.full((n_boot, len(quantiles)), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for j, q in enumerate(quantiles):
            try:
                boot[b, j] = _late_q(Y[idx], D[idx], Z[idx], q)
            except Exception:
                pass
    se_q = np.nanstd(boot, axis=0, ddof=1)
    se_q = np.where(np.isfinite(se_q) & (se_q > 0), se_q, 1e-6)

    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci_low = late_q - z_crit * se_q
    ci_high = late_q + z_crit * se_q

    return BeyondAverageResult(
        quantiles=quantiles,
        late_q=late_q,
        se_q=se_q,
        ci_low=ci_low,
        ci_high=ci_high,
        complier_share=complier_share,
        n_obs=n,
    )
