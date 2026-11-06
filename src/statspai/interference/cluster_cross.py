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

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, IdentificationFailure, MethodIncompatibility


@dataclass
class CrossClusterRCTResult(ResultProtocolMixin):
    """Output of cross-cluster RCT with interference correction.

    Produced by :func:`cluster_cross_interference`. Holds the estimated
    ``direct_effect`` and ``spillover_effect`` (with standard errors), the
    number of clusters, and a formatted ``.summary()``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for c in range(40):
    ...     d = int(rng.integers(0, 2))
    ...     nshare = float(rng.random())
    ...     base = 1.0 * d + 0.8 * nshare
    ...     for _ in range(int(rng.integers(5, 15))):
    ...         rows.append({"cluster": c, "treat": d,
    ...                      "nshare": nshare, "y": base + rng.normal()})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.cluster_cross_interference(df, y="y", cluster="cluster",
    ...                                     treat="treat",
    ...                                     neighbour_treat_share="nshare")
    >>> type(res).__name__
    'CrossClusterRCTResult'
    >>> res.n_clusters
    40
    >>> isinstance(res.summary(), str)
    True
    """

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

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for c in range(40):
    ...     d = int(rng.integers(0, 2))
    ...     nshare = float(rng.random())
    ...     base = 1.0 * d + 0.8 * nshare
    ...     for _ in range(int(rng.integers(5, 15))):
    ...         rows.append({"cluster": c, "treat": d,
    ...                      "nshare": nshare, "y": base + rng.normal()})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.cluster_cross_interference(df, y="y", cluster="cluster",
    ...                                     treat="treat",
    ...                                     neighbour_treat_share="nshare")
    >>> res.n_clusters
    40
    >>> isinstance(res.direct_effect, float)
    True
    >>> isinstance(res.summary(), str)
    True
    """
    df = (
        data[[y, cluster, treat, neighbour_treat_share]].dropna().reset_index(drop=True)
    )
    # Treatment and exposure are cluster-level quantities; a column that
    # varies inside a cluster would otherwise be read from its first row.
    for col in (treat, neighbour_treat_share):
        n_distinct = df.groupby(cluster)[col].nunique()
        varying = n_distinct[n_distinct > 1]
        if len(varying):
            raise MethodIncompatibility(
                f"sp.cluster_cross_interference: {col!r} varies within "
                f"{len(varying)} cluster(s) (e.g. {list(varying.index[:3])}); "
                "it must be constant within each cluster.",
                recovery_hint=(
                    f"Assign {col!r} at the cluster level, or use an "
                    "individual-level estimator such as sp.network_exposure."
                ),
                diagnostics={"column": col, "n_varying_clusters": int(len(varying))},
            )

    # Aggregate to cluster level for inference
    cl = (
        df.groupby(cluster)
        .agg(
            {
                y: "mean",
                treat: "first",
                neighbour_treat_share: "first",
            }
        )
        .reset_index()
    )

    # OLS: Y_c = alpha + beta * D_c + gamma * NShare_c + e_c
    Y = cl[y].to_numpy(float)
    D = cl[treat].to_numpy(float)
    NS = cl[neighbour_treat_share].to_numpy(float)
    X = np.column_stack([np.ones_like(Y), D, NS])
    n_cl, k = X.shape
    if n_cl <= k:
        raise DataInsufficient(
            f"sp.cluster_cross_interference: {n_cl} cluster(s) for {k} "
            "cluster-level coefficients leaves no residual degrees of freedom.",
            recovery_hint="Use more clusters (at least four).",
            diagnostics={"n_clusters": int(n_cl), "n_coefficients": int(k)},
        )
    rank = int(np.linalg.matrix_rank(X))
    if rank < k:
        raise IdentificationFailure(
            "sp.cluster_cross_interference: the cluster-level design "
            f"[1, {treat}, {neighbour_treat_share}] has rank {rank} < {k}; "
            "the direct and spillover effects are not separately identified "
            "(e.g. every cluster has the same treatment, the same neighbour "
            "share, or the two are collinear).",
            recovery_hint=(
                "Check that treatment and the neighbour share both vary "
                "across clusters and are not collinear."
            ),
            diagnostics={"rank": rank, "n_clusters": int(n_cl)},
        )
    beta = np.linalg.solve(X.T @ X, X.T @ Y)
    resid = Y - X @ beta
    sigma2 = float(np.sum(resid**2) / (n_cl - k))
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    direct = float(beta[1])
    se_direct = float(np.sqrt(max(cov[1, 1], 0.0)))
    spillover = float(beta[2])
    se_spillover = float(np.sqrt(max(cov[2, 2], 0.0)))

    return CrossClusterRCTResult(
        direct_effect=direct,
        direct_se=se_direct,
        spillover_effect=spillover,
        spillover_se=se_spillover,
        n_clusters=len(cl),
    )
