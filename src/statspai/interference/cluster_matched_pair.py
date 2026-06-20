"""
Matched-Pair Cluster RCT (Bai et al. 2022 v-2025, arXiv 2211.14903).

In a matched-pair cluster RCT, clusters are paired on baseline
covariates and one cluster of each pair is randomly assigned to
treatment. This module implements the weighted DIM estimator and
its single-variance estimator under the matched-pair design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin


@dataclass
class MatchedPairResult(ResultProtocolMixin):
    """Output of matched-pair cluster RCT estimation.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(7)
    >>> rows = []
    >>> for p in range(20):                       # 20 matched pairs
    ...     base = rng.normal(0, 1)               # shared pair-level baseline
    ...     for arm in (0, 1):                    # one control, one treated cluster
    ...         cid = p * 2 + arm
    ...         for _ in range(15):
    ...             y = base + 0.5 * arm + rng.normal(0, 0.3)
    ...             rows.append({"y": y, "cluster": cid, "treat": arm, "pair": p})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.cluster_matched_pair(df, y="y", cluster="cluster",
    ...                               treat="treat", pair="pair")
    >>> isinstance(res, sp.MatchedPairResult)
    True
    >>> res.n_pairs
    20
    >>> res.n_clusters
    40
    """

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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(7)
    >>> rows = []
    >>> for p in range(20):                       # 20 matched pairs
    ...     base = rng.normal(0, 1)               # shared pair-level baseline
    ...     for arm in (0, 1):                    # one control, one treated cluster
    ...         cid = p * 2 + arm
    ...         for _ in range(15):
    ...             y = base + 0.5 * arm + rng.normal(0, 0.3)
    ...             rows.append({"y": y, "cluster": cid, "treat": arm, "pair": p})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.cluster_matched_pair(df, y="y", cluster="cluster",
    ...                               treat="treat", pair="pair")
    >>> res.n_pairs
    20
    >>> "Matched-Pair" in res.summary()
    True
    """
    df = data[[y, cluster, treat, pair]].dropna().reset_index(drop=True)
    # Cluster-level means
    cl = df.groupby([pair, cluster, treat])[y].mean().reset_index()
    pair_diff_list = []
    for p, sub in cl.groupby(pair):
        if len(sub) != 2:
            continue
        try:
            yt = sub.loc[sub[treat] == 1, y].iloc[0]
            yc = sub.loc[sub[treat] == 0, y].iloc[0]
            pair_diff_list.append(yt - yc)
        except Exception:
            continue
    pair_diffs = np.array(pair_diff_list)
    if len(pair_diffs) < 2:
        raise ValueError(f"Need at least 2 valid pairs (got {len(pair_diffs)}).")
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
