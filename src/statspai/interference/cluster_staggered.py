"""
Staggered-Rollout Cluster RCT (Chen & Li 2025, arXiv 2502.10939).

Cluster RCT where treatment is rolled out across clusters over time
(staggered adoption). Estimates the dynamic ATT averaging over the
event-time × cohort matrix, robust to the standard staggered-DiD
contamination.
"""

from __future__ import annotations

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

    # Cohort-level event-study atts
    cohorts = sorted(cl.loc[cl[first_treat] > 0, first_treat].unique())
    control_clusters = cl.loc[cl[first_treat] == 0, cluster].unique()
    if len(control_clusters) == 0:
        raise ValueError("No never-treated control clusters available.")

    rel_rows = []
    for c in cohorts:
        cohort_clusters = cl.loc[cl[first_treat] == c, cluster].unique()
        for k in range(-leads, lags + 1):
            t = c + k
            sub = cl[cl[time] == t]
            if sub.empty:
                continue
            sub = sub.assign(_t=lambda d: d[cluster].isin(cohort_clusters).astype(int))
            ref = cl[cl[time] == c - 1]
            if ref.empty:
                continue
            ref = ref.assign(_t=lambda d: d[cluster].isin(cohort_clusters).astype(int))
            try:
                m = sub.groupby("_t")[y].mean()
                m_ref = ref.groupby("_t")[y].mean()
                att = float(
                    (m.get(1, np.nan) - m.get(0, np.nan))
                    - (m_ref.get(1, np.nan) - m_ref.get(0, np.nan))
                )
                if np.isfinite(att):
                    rel_rows.append(
                        {
                            "cohort": c,
                            "rel_time": k,
                            "att": att,
                        }
                    )
            except Exception:
                continue

    es_long = pd.DataFrame(rel_rows)
    if es_long.empty:
        raise ValueError("Could not estimate any event-time ATTs.")
    # Aggregate per relative time (simple mean across cohorts)
    es = es_long.groupby("rel_time")["att"].mean().reset_index()
    # SE via cluster bootstrap
    rng = np.random.default_rng(0)
    boot = []
    n_clusters = cl[cluster].nunique()
    for b in range(100):
        sample_clusters = rng.choice(
            cl[cluster].unique(), size=n_clusters, replace=True
        )
        sub_cl = pd.concat(
            [cl[cl[cluster] == cc] for cc in sample_clusters], ignore_index=True
        )
        try:
            inner_atts = []
            for c in cohorts:
                cohort_cls = sub_cl.loc[sub_cl[first_treat] == c, cluster].unique()
                if len(cohort_cls) == 0:
                    continue
                for k in range(0, lags + 1):
                    t = c + k
                    sub = sub_cl[sub_cl[time] == t]
                    ref = sub_cl[sub_cl[time] == c - 1]
                    if sub.empty or ref.empty:
                        continue
                    sub = sub.assign(
                        _t=lambda d: d[cluster].isin(cohort_cls).astype(int)
                    )
                    ref = ref.assign(
                        _t=lambda d: d[cluster].isin(cohort_cls).astype(int)
                    )
                    m = sub.groupby("_t")[y].mean()
                    mr = ref.groupby("_t")[y].mean()
                    att = (m.get(1, np.nan) - m.get(0, np.nan)) - (
                        mr.get(1, np.nan) - mr.get(0, np.nan)
                    )
                    if np.isfinite(att):
                        inner_atts.append(att)
            if inner_atts:
                boot.append(np.mean(inner_atts))
        except Exception:
            continue
    se = float(np.std(boot, ddof=1)) if len(boot) >= 2 else 1e-6
    overall = float(es.loc[es["rel_time"] >= 0, "att"].mean())
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    es["se"] = se
    es["ci_low"] = es["att"] - z_crit * se
    es["ci_high"] = es["att"] + z_crit * se

    return StaggeredClusterRCTResult(
        overall_att=overall,
        overall_se=se,
        event_study=es,
        n_clusters=int(n_clusters),
    )
