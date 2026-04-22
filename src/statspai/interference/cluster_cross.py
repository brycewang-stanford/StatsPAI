"""
Cluster RCT with Cross-Cluster Interference (Leung 2023,
arXiv 2310.18836).

When clusters are not perfectly isolated — e.g. neighbouring villages
share a market or workforce — the standard cluster-RCT estimator is
biased. This module adjusts for cross-cluster spillover by including
a measure of "exposure to treated neighbours" as a regressor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CrossClusterRCTResult:
    """Output of cross-cluster RCT with interference correction."""
    direct_effect: float
    direct_se: float
    spillover_effect: float
    spillover_se: float
    n_clusters: int
    method: str = "Cluster RCT with Cross-Cluster Interference"

    def summary(self) -> str:
        return (
            f"{self.method}\n"
            "=" * 42 + "\n"
            f"  N clusters    : {self.n_clusters}\n"
            f"  Direct effect : {self.direct_effect:+.4f} "
            f"(SE {self.direct_se:.4f})\n"
            f"  Spillover     : {self.spillover_effect:+.4f} "
            f"(SE {self.spillover_se:.4f})\n"
        )


def cluster_cross_interference(
    data: pd.DataFrame,
    y: str,
    cluster: str,
    treat: str,
    neighbour_treat_share: str,
    alpha: float = 0.05,
) -> CrossClusterRCTResult:
    """
    Cluster RCT with explicit interference adjustment.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Individual-level outcome.
    cluster : str
        Cluster identifier.
    treat : str
        Cluster-level binary treatment.
    neighbour_treat_share : str
        Share of treated neighbours per cluster (precomputed by user
        from spatial / network adjacency).
    alpha : float

    Returns
    -------
    CrossClusterRCTResult
    """
    df = data[[y, cluster, treat, neighbour_treat_share]].dropna() \
        .reset_index(drop=True)
    # Aggregate to cluster level for inference
    cl = df.groupby(cluster).agg({
        y: 'mean', treat: 'first',
        neighbour_treat_share: 'first',
    }).reset_index()

    # OLS: Y_c = α + β * D_c + γ * NShare_c + ε_c
    Y = cl[y].to_numpy(float)
    D = cl[treat].to_numpy(float)
    NS = cl[neighbour_treat_share].to_numpy(float)
    X = np.column_stack([np.ones_like(Y), D, NS])
    try:
        beta = np.linalg.solve(X.T @ X, X.T @ Y)
        resid = Y - X @ beta
        sigma2 = float(np.sum(resid ** 2) / max(len(Y) - X.shape[1], 1))
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        direct = float(beta[1])
        se_direct = float(np.sqrt(max(cov[1, 1], 0.0)))
        spillover = float(beta[2])
        se_spillover = float(np.sqrt(max(cov[2, 2], 0.0)))
    except np.linalg.LinAlgError:
        direct = float('nan'); se_direct = float('nan')
        spillover = float('nan'); se_spillover = float('nan')

    return CrossClusterRCTResult(
        direct_effect=direct,
        direct_se=se_direct,
        spillover_effect=spillover,
        spillover_se=se_spillover,
        n_clusters=len(cl),
    )
