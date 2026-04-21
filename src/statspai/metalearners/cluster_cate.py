"""
Cluster-Based CATE (Hartford et al. 2025, arXiv 2409.08773).

Estimates conditional ATE by first clustering units in covariate
space, then computing a within-cluster DIM (or DR-Learner). Returns
a per-cluster CATE table — interpretable as discrete groups instead
of a continuous τ(x) function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ClusterCATEResult:
    """Per-cluster CATE table."""
    cluster_table: pd.DataFrame      # cols: cluster, n, cate, se, ci_low, ci_high
    n_clusters: int
    n_obs: int

    def summary(self) -> str:
        rows = [
            "Cluster-Based CATE",
            "=" * 42,
            f"  N clusters : {self.n_clusters}",
            f"  N total    : {self.n_obs}",
            "",
            self.cluster_table.to_string(index=False),
        ]
        return "\n".join(rows)


def cluster_cate(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    n_clusters: int = 4,
    alpha: float = 0.05,
    seed: int = 0,
) -> ClusterCATEResult:
    """
    Cluster-based CATE estimator.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat : str
    covariates : list of str
    n_clusters : int, default 4
        Number of K-means clusters.
    alpha : float
    seed : int

    Returns
    -------
    ClusterCATEResult
    """
    from sklearn.cluster import KMeans

    df = data[[y, treat] + list(covariates)].dropna().reset_index(drop=True)
    if df[treat].nunique() != 2:
        raise ValueError("Cluster-CATE requires binary treatment.")
    Y = df[y].to_numpy(float)
    D = df[treat].to_numpy(int)
    X = df[covariates].to_numpy(float)

    km = KMeans(n_clusters=n_clusters, random_state=seed,
                n_init=10).fit(X)
    labels = km.labels_

    rows = []
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() < 5 or (D[mask] == 1).sum() == 0 or (D[mask] == 0).sum() == 0:
            continue
        treated = Y[mask & (D == 1)]
        control = Y[mask & (D == 0)]
        cate_k = float(treated.mean() - control.mean())
        se_k = float(np.sqrt(treated.var(ddof=1) / max(len(treated), 1)
                              + control.var(ddof=1) / max(len(control), 1)))
        z_crit = float(stats.norm.ppf(1 - alpha / 2))
        rows.append({
            'cluster': k, 'n': int(mask.sum()),
            'cate': cate_k, 'se': se_k,
            'ci_low': cate_k - z_crit * se_k,
            'ci_high': cate_k + z_crit * se_k,
        })

    return ClusterCATEResult(
        cluster_table=pd.DataFrame(rows),
        n_clusters=len(rows),
        n_obs=len(df),
    )
