"""
Metafrontier analysis (O'Donnell-Rao-Battese 2008).

Given ``K`` groups with potentially different technologies, fit a group
frontier per group and then solve for a *metafrontier* ``beta^*`` that
envelopes every fitted group frontier from above (for a production
frontier) or from below (for a cost frontier), while staying as close
as possible to each group's estimates.

Decomposition (production):
    TE_i^{meta}  =  TE_i^{group}  x  TGR_i
where
    TGR_i = exp(x_i' beta^{group(i)} - x_i' beta^{meta})   in  (0, 1]
is the technology-gap ratio between i's group frontier and the
metafrontier.

We implement the LP formulation (O'Donnell-Rao-Battese 2008, Eq. 13):

    minimise_{beta_meta}   sum_{i in pooled sample}  x_i' (beta_meta - beta^{k(i)})
    subject to             x_i' beta_meta  >=  x_i' beta^k   for all i, all k.

(Sum of *positive* gaps because the constraints force every difference
to be non-negative for production.)  For cost frontiers we flip the
inequality direction.

References
----------
O'Donnell, C.J., Rao, D.S.P. & Battese, G.E. (2008).  "Metafrontier
    frameworks for the study of firm-level efficiencies and technology
    ratios."  Empirical Economics 34, 231-255.
Battese, G.E., Rao, D.S.P. & O'Donnell, C.J. (2004).  "A Metafrontier
    Production Function for Estimation of Technical Efficiencies and
    Technology Gaps for Firms Operating Under Different Technologies."
    J. Productivity Analysis 21, 91-103.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .sfa import FrontierResult, frontier as _frontier


@dataclass
class MetafrontierResult:
    """Container for a metafrontier fit."""
    beta_meta: pd.Series
    beta_groups: Dict[Any, pd.Series]
    group_frontiers: Dict[Any, FrontierResult]
    tgr: pd.Series                # technology-gap ratio per obs
    te_meta: pd.Series            # TE_meta per obs
    te_group: pd.Series           # TE_group per obs
    data_info: Dict[str, Any]
    lp_status: str

    def summary(self) -> str:
        lines = [
            "=" * 80,
            "Metafrontier (O'Donnell-Rao-Battese 2008)",
            "=" * 80,
            f"Groups        : {list(self.group_frontiers.keys())}",
            f"N observations: {self.data_info['n_obs']}",
            f"LP status     : {self.lp_status}",
            "",
            "Metafrontier coefficients:",
        ]
        lines.append(self.beta_meta.to_string())
        lines.append("")
        lines.append("Group-specific mean TGR / TE_group / TE_meta:")
        group_summary = pd.DataFrame({
            "mean_tgr":   self.tgr.groupby(self.data_info["group_vec"]).mean(),
            "mean_te_group": self.te_group.groupby(self.data_info["group_vec"]).mean(),
            "mean_te_meta":  self.te_meta.groupby(self.data_info["group_vec"]).mean(),
        })
        lines.append(group_summary.round(4).to_string())
        return "\n".join(lines)


def metafrontier(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    group: str,
    *,
    dist: str = "half-normal",
    cost: bool = False,
    te_method: str = "bc",
    **frontier_kwargs,
) -> MetafrontierResult:
    """Estimate a metafrontier across ``K`` groups.

    Parameters
    ----------
    data : pandas.DataFrame
    y, x, group : str / list of str
    dist : inefficiency distribution for the group-level frontiers.
    cost : bool, default False
    te_method : {'bc', 'jlms'}
    frontier_kwargs : forwarded to :func:`frontier` for each group
        (e.g., ``usigma``, ``vsigma``, ``emean``, ``vce``, ``cluster``).

    Returns
    -------
    :class:`MetafrontierResult`
    """
    if group not in data.columns:
        raise KeyError(f"{group!r} is not a column in data.")
    required = [y] + list(x) + [group]
    df = data[required].dropna().copy()

    # ------------------------------------------------------------------
    # Step 1: fit a frontier per group.
    # ------------------------------------------------------------------
    group_ids = df[group].unique()
    group_frontiers: Dict[Any, FrontierResult] = {}
    beta_groups: Dict[Any, pd.Series] = {}
    for g in group_ids:
        sub = df[df[group] == g].copy()
        if len(sub) < len(x) + 3:
            raise ValueError(
                f"Group {g!r} has only {len(sub)} observations; need at least {len(x) + 3}."
            )
        res = _frontier(sub, y=y, x=x, dist=dist, cost=cost,
                        te_method=te_method, **frontier_kwargs)
        group_frontiers[g] = res
        beta_groups[g] = res.params.loc[["_cons"] + list(x)].copy()

    # ------------------------------------------------------------------
    # Step 2: assemble design matrices and solve the LP for beta_meta.
    # ------------------------------------------------------------------
    n = len(df)
    X = np.concatenate([np.ones((n, 1)), df[x].to_numpy(dtype=float)], axis=1)
    p = X.shape[1]

    # Each obs i has a group g(i): x_i' beta^{g(i)} is its own-group frontier.
    own_beta = np.vstack(
        [beta_groups[g].to_numpy() for g in df[group].to_numpy()]
    )
    own_frontier = np.einsum("ij,ij->i", X, own_beta)

    # Objective: min sum_i (x_i' beta_meta - own_frontier_i).
    # For cost (sign=+1), flip to min sum_i (own_frontier_i - x_i' beta_meta).
    if cost:
        c = -X.sum(axis=0)        # maximise x_i' beta_meta, which is min(-c'beta).
    else:
        c = X.sum(axis=0)         # minimise x_i' beta_meta.

    # Constraints: for every i and every group k, x_i' beta_meta  >= x_i' beta^k
    # (flip sign for cost).
    A_ub_rows = []
    b_ub_rows = []
    for g in group_ids:
        bk = beta_groups[g].to_numpy()
        Xbk = X @ bk
        if cost:
            # x_i' beta_meta <= x_i' beta^k    =>    X @ beta_meta - Xbk <= 0
            A_ub_rows.append(X)
            b_ub_rows.append(Xbk)
        else:
            # x_i' beta_meta >= x_i' beta^k    =>    -X @ beta_meta + Xbk <= 0
            A_ub_rows.append(-X)
            b_ub_rows.append(-Xbk)
    A_ub = np.concatenate(A_ub_rows, axis=0)
    b_ub = np.concatenate(b_ub_rows, axis=0)

    # beta_meta unbounded; default linprog bounds are (0, None), override to (None, None)
    bounds = [(None, None)] * p

    lp = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )
    if not lp.success:
        raise RuntimeError(
            f"Metafrontier LP failed: {lp.message}. Try different data or "
            "fewer groups."
        )
    beta_meta_arr = lp.x
    beta_meta = pd.Series(beta_meta_arr, index=beta_groups[group_ids[0]].index,
                          name="beta_meta")

    # ------------------------------------------------------------------
    # Step 3: technology-gap ratios and meta-efficiencies.
    # ------------------------------------------------------------------
    meta_frontier_hat = X @ beta_meta_arr
    # TGR per obs: ratio of group frontier to meta frontier.
    # Production: frontier = max output.  In log units, TGR_i = exp(log_y_group_frontier - log_y_meta_frontier).
    gap = own_frontier - meta_frontier_hat  # negative for production, positive for cost (with sign flip enforced)
    tgr = np.exp(gap)
    tgr = np.clip(tgr, 0.0, 1.0)
    tgr_series = pd.Series(tgr, index=df.index, name="tgr")

    # TE_group = the BC efficiency from the group frontier (aligned to df row order).
    te_group_arr = np.empty(n)
    for g in group_ids:
        mask_df = df[group] == g
        res_g = group_frontiers[g]
        te_g = res_g.efficiency(method=te_method)
        te_group_arr[mask_df.to_numpy()] = te_g.values
    te_group_series = pd.Series(te_group_arr, index=df.index, name="te_group")
    te_meta_series = pd.Series(te_group_arr * tgr, index=df.index, name="te_meta")

    data_info = {
        "n_obs": n,
        "group_col": group,
        "group_vec": df[group].to_numpy(),
        "regressors": list(x),
        "dep_var": y,
    }
    return MetafrontierResult(
        beta_meta=beta_meta,
        beta_groups=beta_groups,
        group_frontiers=group_frontiers,
        tgr=tgr_series,
        te_meta=te_meta_series,
        te_group=te_group_series,
        data_info=data_info,
        lp_status=str(lp.message),
    )


__all__ = ["metafrontier", "MetafrontierResult"]
