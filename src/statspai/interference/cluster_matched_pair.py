"""
Matched-Pair Cluster RCT (Bai et al. 2022 v-2025, arXiv 2211.14903).

In a matched-pair cluster RCT, clusters are paired on baseline
covariates and one cluster of each pair is randomly assigned to
treatment. This module implements the weighted DIM estimator and
its single-variance estimator under the matched-pair design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class MatchedPairResult:
    """Output of matched-pair cluster RCT estimation."""
    estimate: float
    se: float
    ci: tuple
    n_pairs: int
    n_clusters: int
    method: str = "Matched-Pair Cluster RCT"

    def summary(self) -> str:
        return (
            f"{self.method}\n"
            "=" * 42 + "\n"
            f"  N pairs    : {self.n_pairs}\n"
            f"  N clusters : {self.n_clusters}\n"
            f"  Estimate   : {self.estimate:+.4f} (SE {self.se:.4f})\n"
            f"  95% CI     : [{self.ci[0]:+.4f}, {self.ci[1]:+.4f}]\n"
        )


def cluster_matched_pair(
    data: pd.DataFrame,
    y: str,
    cluster: str,
    treat: str,
    pair: str,
    alpha: float = 0.05,
) -> MatchedPairResult:
    """
    Matched-pair cluster RCT estimator (weighted DIM + Bai SE).

    Parameters
    ----------
    data : pd.DataFrame
        Individual-level data.
    y : str
        Outcome.
    cluster : str
        Cluster identifier.
    treat : str
        Cluster-level binary treatment.
    pair : str
        Pair identifier (each pair contains exactly two clusters).
    alpha : float

    Returns
    -------
    MatchedPairResult
    """
    df = data[[y, cluster, treat, pair]].dropna().reset_index(drop=True)
    # Cluster-level means
    cl = df.groupby([pair, cluster, treat])[y].mean().reset_index()
    pair_diffs = []
    for p, sub in cl.groupby(pair):
        if len(sub) != 2:
            continue
        try:
            yt = sub.loc[sub[treat] == 1, y].iloc[0]
            yc = sub.loc[sub[treat] == 0, y].iloc[0]
            pair_diffs.append(yt - yc)
        except Exception:
            continue
    pair_diffs = np.array(pair_diffs)
    if len(pair_diffs) < 2:
        raise ValueError(
            f"Need at least 2 valid pairs (got {len(pair_diffs)})."
        )
    estimate = float(pair_diffs.mean())
    # Bai (2022) single-variance estimator: across-pair variance / n_pairs
    se = float(pair_diffs.std(ddof=1) / np.sqrt(len(pair_diffs)))
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (estimate - z_crit * se, estimate + z_crit * se)
    return MatchedPairResult(
        estimate=estimate,
        se=se,
        ci=ci,
        n_pairs=len(pair_diffs),
        n_clusters=int(cl[cluster].nunique()),
    )
