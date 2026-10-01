"""
Staggered-Rollout Cluster RCT (Chen & Li 2025, arXiv 2502.10939).

Cluster RCT where treatment is rolled out across clusters over time
(staggered adoption). Estimates the dynamic ATT averaging over the
event-time × cohort matrix, robust to the standard staggered-DiD
contamination.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin


@dataclass
class StaggeredClusterRCTResult(ResultProtocolMixin):
    """Staggered-rollout cluster RCT output.

    Attributes
    ----------
    overall_att : float
        Mean post-treatment dynamic ATT across event times.
    overall_se : float
        Cluster-bootstrap standard error of ``overall_att``.
    event_study : pandas.DataFrame
        Per relative-time ATTs with ``se``, ``ci_low`` and ``ci_high``.
    n_clusters : int
        Number of clusters in the design.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for c in range(12):
    ...     ft = 4 if c < 4 else (6 if c < 8 else 0)  # 0 = never-treated
    ...     fe = rng.normal()
    ...     for t in range(8):
    ...         d = 1 if (ft > 0 and t >= ft) else 0
    ...         y = 1.0 + fe + 0.2 * t + 1.5 * d + rng.normal(scale=0.3)
    ...         rows.append({"cluster": c, "time": t, "first_treat": ft, "y": y})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.cluster_staggered_rollout(
    ...     df, y="y", cluster="cluster", time="time", first_treat="first_treat")
    >>> isinstance(res, sp.StaggeredClusterRCTResult)
    True
    >>> res.n_clusters
    12
    """

    overall_att: float
    overall_se: float
    event_study: pd.DataFrame
    n_clusters: int
    method: str = "Staggered-Rollout Cluster RCT"

    def summary(self) -> str:
        return (
            f"{self.method}\n"
            "=" * 42 + "\n"
            f"  N clusters    : {self.n_clusters}\n"
            f"  Overall ATT   : {self.overall_att:+.4f} "
            f"(SE {self.overall_se:.4f})\n"
            f"  Event-study (pre-mean): "
            f"{self.event_study[self.event_study['rel_time'] < 0]['att'].mean():+.4f}\n"
            f"  Event-study (post-mean): "
            f"{self.event_study[self.event_study['rel_time'] >= 0]['att'].mean():+.4f}\n"
        )


def cluster_staggered_rollout(
    data: pd.DataFrame,
    y: str,
    cluster: str,
    time: str,
    first_treat: str,
    leads: int = 2,
    lags: int = 4,
    alpha: float = 0.05,
) -> StaggeredClusterRCTResult:
    """
    Staggered-rollout cluster RCT estimator.

    Parameters
    ----------
    data : pd.DataFrame
        Panel: cluster × time × outcome.
    y, cluster, time, first_treat : str
        first_treat = first calendar time the cluster is treated
        (0 / NaN for never-treated).
    leads, lags : int
    alpha : float

    Returns
    -------
    StaggeredClusterRCTResult

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for c in range(12):  # 4 cohorts at t=4, 4 at t=6, 4 never-treated
    ...     ft = 4 if c < 4 else (6 if c < 8 else 0)
    ...     fe = rng.normal()
    ...     for t in range(8):
    ...         d = 1 if (ft > 0 and t >= ft) else 0
    ...         y = 1.0 + fe + 0.2 * t + 1.5 * d + rng.normal(scale=0.3)
    ...         rows.append({"cluster": c, "time": t, "first_treat": ft, "y": y})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.cluster_staggered_rollout(
    ...     df, y="y", cluster="cluster", time="time", first_treat="first_treat")
    >>> res.event_study.columns.tolist()
    ['rel_time', 'att', 'se', 'ci_low', 'ci_high']
    >>> res.n_clusters
    12
    """
    df = data[[y, cluster, time, first_treat]].dropna().reset_index(drop=True)
    cl = (
        df.groupby([cluster, time]).agg({y: "mean", first_treat: "first"}).reset_index()
    )

    cohorts = sorted(cl.loc[cl[first_treat] > 0, first_treat].unique())
    wide = cl.pivot(index=cluster, columns=time, values=y)
    g_by_cluster = cl.groupby(cluster)[first_treat].first().reindex(wide.index)
    Yw = wide.to_numpy(float)
    g_arr = g_by_cluster.to_numpy()
    pos = {t: j for j, t in enumerate(wide.columns)}
    if not np.any(g_arr == 0):
        raise ValueError("No never-treated control clusters available.")

    rel_times = list(range(-leads, lags + 1))

    def _event_study(idx: np.ndarray) -> np.ndarray:
        """ATT by relative time on the clusters ``idx`` (with multiplicity).

        Each cohort is compared with the never-treated clusters only.

        correctness fix (2026-10): the comparison group used to be every
        cluster outside the cohort, so a cohort that switched on between
        the reference period and ``t`` entered as a control. With two
        cohorts and a constant effect of 1.5 the later event times came
        out at 0.8 to 0.9.
        """
        Yb = Yw[idx]
        gb = g_arr[idx]
        ctrl = Yb[gb == 0]
        out = np.full((len(cohorts), len(rel_times)), np.nan)
        if len(ctrl) == 0:
            return np.full(len(rel_times), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m_ctrl = np.nanmean(ctrl, axis=0)
            for i, c in enumerate(cohorts):
                trt = Yb[gb == c]
                if len(trt) == 0 or (c - 1) not in pos:
                    continue
                m_trt = np.nanmean(trt, axis=0)
                base = m_trt[pos[c - 1]] - m_ctrl[pos[c - 1]]
                for j, k in enumerate(rel_times):
                    t = c + k
                    if t in pos:
                        out[i, j] = (m_trt[pos[t]] - m_ctrl[pos[t]]) - base
            # Simple mean across cohorts at each relative time.
            return np.nanmean(out, axis=0)

    def _overall(es_vec: np.ndarray) -> float:
        post = es_vec[[j for j, k in enumerate(rel_times) if k >= 0]]
        post = post[np.isfinite(post)]
        return float(post.mean()) if len(post) else float("nan")

    n_clusters = len(wide)
    es_vec = _event_study(np.arange(n_clusters))
    keep = np.isfinite(es_vec)
    if not keep.any():
        raise ValueError("Could not estimate any event-time ATTs.")

    # Cluster bootstrap: one joint draw gives every event time and the
    # post-period average, so each row carries its own standard error.
    n_boot = 100
    rng = np.random.default_rng(0)
    boot_es = np.full((n_boot, len(rel_times)), np.nan)
    boot_overall = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n_clusters, size=n_clusters)
        boot_es[b] = _event_study(idx)
        boot_overall[b] = _overall(boot_es[b])

    from ..core._bootstrap import bootstrap_se

    label = "interference.cluster_staggered_rollout"
    se = bootstrap_se(boot_overall, label=label)
    overall = _overall(es_vec)
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    es = pd.DataFrame({"rel_time": rel_times, "att": es_vec})[keep].reset_index(
        drop=True
    )
    es["se"] = [
        bootstrap_se(boot_es[:, j], label=label, warn=False)
        for j in np.flatnonzero(keep)
    ]
    es["ci_low"] = es["att"] - z_crit * es["se"]
    es["ci_high"] = es["att"] + z_crit * es["se"]

    return StaggeredClusterRCTResult(
        overall_att=overall,
        overall_se=se,
        event_study=es,
        n_clusters=int(n_clusters),
    )
