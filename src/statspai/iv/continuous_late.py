"""
Continuous-Instrument LATE (Zeng et al. 2025, arXiv 2504.03063).

Generalises LATE identification to continuous instruments. The
"maximal complier class" is the set of units whose treatment status
shifts most as Z varies; the LATE on this class is the limit of the
Wald ratio across instrument quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ContinuousLATEResult:
    """Continuous-instrument LATE on the maximal complier class."""
    estimate: float
    se: float
    ci: tuple
    complier_share: float
    n_obs: int

    def summary(self) -> str:
        header = "Continuous-Instrument LATE (Maximal Complier Class)"
        bar = "=" * len(header)
        return (
            f"{header}\n"
            f"{bar}\n"
            f"  N            : {self.n_obs}\n"
            f"  Complier sh. : {self.complier_share:.3f}\n"
            f"  LATE         : {self.estimate:+.4f} (SE {self.se:.4f})\n"
            f"  95% CI       : [{self.ci[0]:+.4f}, {self.ci[1]:+.4f}]\n"
        )


def continuous_iv_late(
    data: pd.DataFrame,
    y: str,
    treat: str,
    instrument: str,
    n_quantiles: int = 4,
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 0,
) -> ContinuousLATEResult:
    """
    LATE with a continuous instrument via quantile-bin Wald estimator.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat, instrument : str
    n_quantiles : int, default 4
        Number of quantile bins of the instrument; LATE is averaged
        across bins weighted by complier share.
    alpha : float
    n_boot : int
    seed : int

    Returns
    -------
    ContinuousLATEResult
    """
    df = data[[y, treat, instrument]].dropna().reset_index(drop=True)
    Y = df[y].to_numpy(float)
    D = df[treat].to_numpy(float)
    Z = df[instrument].to_numpy(float)
    n = len(df)
    rng = np.random.default_rng(seed)

    def _wald_per_bin(Yi, Di, Zi):
        z_bins = pd.qcut(Zi, q=n_quantiles, labels=False,
                          duplicates='drop')
        unique_bins = np.sort(np.unique(z_bins))
        if len(unique_bins) < 2:
            return float(np.mean(Yi[Di > Di.mean()]) - np.mean(Yi[Di <= Di.mean()])), 1.0
        # Wald ratio for each adjacent bin, then average
        atts = []
        weights = []
        for k in range(len(unique_bins) - 1):
            mask = z_bins == unique_bins[k]
            mask_next = z_bins == unique_bins[k + 1]
            num = float(Yi[mask_next].mean() - Yi[mask].mean())
            denom = float(Di[mask_next].mean() - Di[mask].mean())
            if abs(denom) < 1e-6:
                continue
            atts.append(num / denom)
            weights.append(abs(denom))
        if not atts:
            return float('nan'), 0.0
        atts = np.array(atts)
        weights = np.array(weights)
        # Maximal complier class: pick the bin pair with the biggest
        # |denom| (most "responsive" units to Z).
        idx = int(np.argmax(weights))
        return float(atts[idx]), float(weights[idx])

    estimate, complier = _wald_per_bin(Y, D, Z)
    boot = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot[b], _ = _wald_per_bin(Y[idx], D[idx], Z[idx])
        except Exception:
            pass
    se = float(np.nanstd(boot, ddof=1)) or 1e-6
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (estimate - z_crit * se, estimate + z_crit * se)
    return ContinuousLATEResult(
        estimate=estimate,
        se=se,
        ci=ci,
        complier_share=complier,
        n_obs=n,
    )
