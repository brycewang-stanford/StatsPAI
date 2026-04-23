"""
General-purpose bootstrap inference framework.

Works with any StatsPAI estimator — pass a fitting function and get
bootstrap confidence intervals, standard errors, and p-values.

Supports:
- **Nonparametric bootstrap** (resample rows)
- **Pairs cluster bootstrap** (resample clusters)
- **Block bootstrap** (for time-series / panel data)
- **Percentile, BCa, and normal** confidence intervals

References
----------
Efron, B. and Tibshirani, R.J. (1993).
*An Introduction to the Bootstrap*. Chapman & Hall.

Cameron, A.C., Gelbach, J.B. and Miller, D.L. (2008).
"Bootstrap-Based Improvements for Inference with Clustered Errors."
*Review of Economics and Statistics*, 90(3), 414-427. [@cameron2008bootstrap]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


@dataclass
class BootstrapResult:
    """Container for bootstrap inference results."""

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    alpha: float
    n_boot: int
    boot_distribution: np.ndarray
    ci_method: str

    def summary(self) -> str:
        lines = [
            "Bootstrap Inference",
            f"  Estimate:   {self.estimate:.6f}",
            f"  Std. Error: {self.se:.6f}",
            f"  CI ({1-self.alpha:.0%}):    [{self.ci_lower:.6f}, {self.ci_upper:.6f}]  ({self.ci_method})",
            f"  p-value:    {self.pvalue:.4f}",
            f"  Replications: {self.n_boot}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def bootstrap(
    data: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    n_boot: int = 1000,
    cluster: Optional[str] = None,
    block: Optional[str] = None,
    ci_method: str = "percentile",
    alpha: float = 0.05,
    seed: Optional[int] = None,
    null_value: float = 0.0,
) -> BootstrapResult:
    """
    General-purpose bootstrap inference.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    statistic : callable
        A function ``f(df) -> float`` that computes the statistic of
        interest on a (possibly resampled) DataFrame.
    n_boot : int, default 1000
        Number of bootstrap replications.
    cluster : str, optional
        Column name for cluster bootstrap (resample clusters, not rows).
    block : str, optional
        Column name for block bootstrap (resample blocks, preserve
        within-block ordering). Useful for panel/time-series data.
    ci_method : str, default 'percentile'
        CI method: ``'percentile'``, ``'bca'``, or ``'normal'``.
    alpha : float, default 0.05
        Significance level.
    seed : int, optional
        Random seed.
    null_value : float, default 0.0
        Value under H0 for p-value computation.

    Returns
    -------
    BootstrapResult

    Examples
    --------
    >>> def did_effect(df):
    ...     result = sp.did(df, y='wage', treat='treated', time='post')
    ...     return result.estimate
    >>> boot = sp.bootstrap(df, did_effect, n_boot=500, cluster='firm')
    >>> print(boot)

    >>> # Quick bootstrap for any custom statistic
    >>> boot = sp.bootstrap(df, lambda d: d['y'].mean() - d['x'].mean(),
    ...                     n_boot=2000)
    """
    rng = np.random.RandomState(seed)

    # Original estimate
    theta_hat = statistic(data)

    # Bootstrap distribution
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        boot_data = _resample(data, rng, cluster=cluster, block=block)
        try:
            boot_stats[b] = statistic(boot_data)
        except Exception:
            boot_stats[b] = np.nan

    # Remove failed replications
    boot_stats = boot_stats[~np.isnan(boot_stats)]
    n_valid = len(boot_stats)

    if n_valid < 10:
        raise RuntimeError(
            f"Only {n_valid}/{n_boot} bootstrap replications succeeded. "
            "Check that your statistic function handles resampled data."
        )

    # Standard error
    se = float(np.std(boot_stats, ddof=1))

    # Confidence interval
    if ci_method == "percentile":
        ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    elif ci_method == "normal":
        z = sp_stats.norm.ppf(1 - alpha / 2)
        ci_lower = theta_hat - z * se
        ci_upper = theta_hat + z * se
    elif ci_method == "bca":
        ci_lower, ci_upper = _bca_ci(theta_hat, boot_stats, data, statistic,
                                     alpha, rng, cluster, block)
    else:
        raise ValueError(f"Unknown ci_method: {ci_method}. Use 'percentile', 'normal', or 'bca'.")

    # Two-sided p-value (fraction of boot dist more extreme than null)
    centered = boot_stats - theta_hat  # center at estimate
    pvalue = float(np.mean(np.abs(centered) >= abs(theta_hat - null_value)))
    pvalue = max(pvalue, 1 / (n_valid + 1))  # minimum p-value

    return BootstrapResult(
        estimate=theta_hat,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pvalue=pvalue,
        alpha=alpha,
        n_boot=n_valid,
        boot_distribution=boot_stats,
        ci_method=ci_method,
    )


# ====================================================================== #
#  Resampling strategies
# ====================================================================== #

def _resample(
    data: pd.DataFrame,
    rng: np.random.RandomState,
    cluster: Optional[str] = None,
    block: Optional[str] = None,
) -> pd.DataFrame:
    """Resample data according to the specified strategy."""
    n = len(data)

    if cluster is not None:
        # Cluster bootstrap: resample clusters, keep all rows within
        clusters = data[cluster].unique()
        boot_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        frames = []
        for i, c in enumerate(boot_clusters):
            chunk = data[data[cluster] == c].copy()
            # Relabel to avoid duplicate indices
            chunk.index = range(len(frames) * 0 + len(chunk))  # will be reset
            frames.append(chunk)
        return pd.concat(frames, ignore_index=True)

    elif block is not None:
        # Block bootstrap: resample blocks (e.g., time periods)
        blocks = data[block].unique()
        boot_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        frames = [data[data[block] == b] for b in boot_blocks]
        return pd.concat(frames, ignore_index=True)

    else:
        # Nonparametric: resample rows
        idx = rng.choice(n, size=n, replace=True)
        return data.iloc[idx].reset_index(drop=True)


# ====================================================================== #
#  BCa confidence interval
# ====================================================================== #

def _bca_ci(
    theta_hat: float,
    boot_stats: np.ndarray,
    data: pd.DataFrame,
    statistic: Callable,
    alpha: float,
    rng: np.random.RandomState,
    cluster: Optional[str],
    block: Optional[str],
) -> tuple:
    """Bias-corrected and accelerated (BCa) confidence interval."""
    n_boot = len(boot_stats)

    # Bias correction factor z0
    prop_less = np.mean(boot_stats < theta_hat)
    z0 = sp_stats.norm.ppf(max(min(prop_less, 0.999), 0.001))

    # Acceleration factor a (jackknife)
    n = len(data)
    jack_stats = np.empty(n)
    for i in range(min(n, 200)):  # cap jackknife iterations
        jack_data = data.drop(data.index[i]).reset_index(drop=True)
        try:
            jack_stats[i] = statistic(jack_data)
        except Exception:
            jack_stats[i] = theta_hat

    jack_stats = jack_stats[:min(n, 200)]
    jack_mean = np.mean(jack_stats)
    diff = jack_mean - jack_stats
    a = np.sum(diff ** 3) / (6 * (np.sum(diff ** 2)) ** 1.5) if np.sum(diff ** 2) > 0 else 0

    # Adjusted quantiles
    z_alpha = sp_stats.norm.ppf(alpha / 2)
    z_1alpha = sp_stats.norm.ppf(1 - alpha / 2)

    a1 = sp_stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    a2 = sp_stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

    ci_lower = float(np.percentile(boot_stats, 100 * np.clip(a1, 0.001, 0.999)))
    ci_upper = float(np.percentile(boot_stats, 100 * np.clip(a2, 0.001, 0.999)))

    return ci_lower, ci_upper
