"""Group-time aggregation for ``sp.didregress(aggregate=...)``.

Stata 18 ``didregress`` / ``xtdidregress`` ``aggregate()``: with few
groups, collapse the data to one row per group-period cell and fit the DiD
on the cells [@donald2007inference].

Stage 1 removes the covariates that vary within cells. ``'dlang'`` and
``'standard'`` regress the outcome on them with cell fixed effects (one
slope for all cells) and take the cell mean of ``y - x'b`` as the cell's
outcome; ``'dlang_varying'`` fits a separate regression in each cell and
takes its intercept. Stage 2 regresses the cell outcomes on the treatment,
the covariates that are constant within cells, and group and period
effects. ``'dlang'`` uses the ordinary least-squares variance with a t
reference on the residual degrees of freedom of that regression;
``'standard'`` clusters on the group.

Matches Stata 18 to 1e-10 (estimate and standard error). One departure:
Stata refers the dlang t statistic to ``cells - (regressors + periods)``
degrees of freedom, leaving out the group effects it absorbed; the exact
finite-sample distribution of the OLS t statistic under normal errors has
``cells - all parameters``, which is the default here
(``dlang_df='stata'`` reproduces Stata).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["AGGREGATE_METHODS", "aggregate_cells"]

AGGREGATE_METHODS = ("standard", "dlang", "dlang_varying")


def _cell_codes(clean: pd.DataFrame, group: str, time: str) -> np.ndarray:
    return pd.factorize(pd.MultiIndex.from_frame(clean[[group, time]]))[0]


def _split_covariates(clean: pd.DataFrame, covariates: List[str], cells: np.ndarray):
    """Covariates varying within at least one cell (stage 1) or none (stage 2)."""
    within, between = [], []
    for c in covariates:
        nuniq = clean[c].groupby(cells).nunique()
        (within if (nuniq > 1).any() else between).append(c)
    return within, between


def aggregate_cells(
    clean: pd.DataFrame,
    *,
    y: str,
    treat: str,
    group: str,
    time: str,
    covariates: List[str],
    method: str,
    context: str,
) -> Dict[str, Any]:
    """Collapse to group-period cells; returns the stage-2 inputs."""
    cells = _cell_codes(clean, group, time)
    n_cells = int(cells.max()) + 1
    within, between = _split_covariates(clean, covariates, cells)
    yv = clean[y].to_numpy(dtype=float)
    first_stage: Dict[str, float] = {}
    if not within:
        c_val = pd.Series(yv).groupby(cells).mean().to_numpy()
    elif method == "dlang_varying":
        Z = clean[within].to_numpy(dtype=float)
        c_val = np.empty(n_cells)
        k = 1 + Z.shape[1]
        for j in range(n_cells):
            rows = cells == j
            X = np.column_stack([np.ones(rows.sum()), Z[rows]])
            if rows.sum() <= k or np.linalg.matrix_rank(X) < k:
                g, t = clean.loc[rows, [group, time]].iloc[0]
                raise DataInsufficient(
                    f"{context}: aggregate='dlang_varying' fits a regression in "
                    f"every group-period cell; cell ({group}={g}, {time}={t}) "
                    f"has {int(rows.sum())} observations for {k} parameters or "
                    "collinear covariates.",
                    recovery_hint=(
                        "Use aggregate='dlang' (one slope for all cells), or "
                        "drop covariates."
                    ),
                    diagnostics={"cell_obs": int(rows.sum()), "parameters": k},
                )
            beta, *_ = np.linalg.lstsq(X, yv[rows], rcond=None)
            c_val[j] = beta[0]
    else:
        Z = clean[within].to_numpy(dtype=float)
        frame = pd.DataFrame(np.column_stack([yv, Z]))
        swept = (frame - frame.groupby(cells).transform("mean")).to_numpy()
        Zs = swept[:, 1:]
        if np.linalg.matrix_rank(Zs) < Zs.shape[1]:
            raise DataInsufficient(
                f"{context}: the within-cell covariates are collinear once "
                "group-period cell effects are removed.",
                recovery_hint="Drop one of the collinear covariates.",
            )
        beta, *_ = np.linalg.lstsq(Zs, swept[:, 0], rcond=None)
        first_stage = dict(zip(within, map(float, beta)))
        c_val = pd.Series(yv - Z @ beta).groupby(cells).mean().to_numpy()

    keys = clean[[group, time, treat, *between]].groupby(cells).first()
    agg = keys.reset_index(drop=True)
    agg["__c"] = c_val
    agg["__n"] = np.bincount(cells)
    return {
        "cells": agg,
        "within": within,
        "between": between,
        "first_stage": first_stage,
        "n_cells": n_cells,
    }


def check_cluster_constant(
    clean: pd.DataFrame, cluster: str, group: str, time: str, context: str
) -> None:
    cells = _cell_codes(clean, group, time)
    if (clean[cluster].groupby(cells).nunique() > 1).any():
        raise MethodIncompatibility(
            f"{context}: with aggregate= the cluster variable must be constant "
            "within each group-period cell (the cells are the observations).",
            recovery_hint=f"Cluster on {group!r} (the default) or a coarser level.",
        )
