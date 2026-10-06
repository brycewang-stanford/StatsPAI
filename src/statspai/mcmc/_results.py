"""Posterior summary table shared by the samplers of this package."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility
from .diagnostics import gelman_rubin, mcmc_summary


def posterior_table(
    d_df: pd.DataFrame, chain_idx: np.ndarray, chains: int, level: float
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Summary of stacked draws: the ``table`` of a result and the
    convergence numbers that go into its diagnostics."""
    summ = mcmc_summary(d_df, quantiles=())
    if chains > 1:
        ess = np.zeros(d_df.shape[1])
        for c in range(chains):
            ess += mcmc_summary(d_df.loc[chain_idx == c], quantiles=())[
                "ess"
            ].to_numpy()
        summ["ess"] = ess
        with np.errstate(divide="ignore", invalid="ignore"):
            summ["ts_se"] = summ["sd"] / np.sqrt(ess)
    lo = (1.0 - level) / 2.0
    table = pd.DataFrame(
        {
            "mean": summ["mean"],
            "sd": summ["sd"],
            "mcse": summ["ts_se"],
            "ess": summ["ess"],
            "lower": d_df.quantile(lo),
            "median": d_df.quantile(0.5),
            "upper": d_df.quantile(1.0 - lo),
            "prob_positive": (d_df > 0).mean(),
        }
    )
    diag: Dict[str, Any] = {"warnings": [], "min_ess": float(table["ess"].min())}
    try:
        gr = gelman_rubin([d_df.loc[chain_idx == c] for c in range(chains)], split=True)
        diag["max_split_rhat"] = float(gr.table["psrf"].max())
    except (MethodIncompatibility, DataInsufficient):
        diag["max_split_rhat"] = float("nan")
    return table, diag
