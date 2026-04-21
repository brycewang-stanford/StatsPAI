"""
Clone-Censor-Weight (CCW) for sustained-treatment strategies.

When the target trial contrasts *sustained* treatment strategies
(e.g. "start statin at time 0 and continue forever" vs "never start"),
each person-time row in the observational data is compatible with
zero, one, or two strategies until they deviate. CCW solves this by:

1. **Clone** each subject once per compatible strategy.
2. **Censor** a clone at the moment they deviate from their assigned
   strategy.
3. **Re-weight** uncensored clones via IPCW using a censoring model
   that conditions on post-baseline covariates predicting deviation.

This removes the selection bias that artificial censoring would
otherwise introduce, as long as the censoring model captures all
time-varying confounders.

References
----------
* Hernan et al. (2016) Target Trial Emulation.
* Cain et al. (2010) When to Start Antiretroviral Therapy: A Dynamic
  Regime Approach.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np
import pandas as pd


@dataclass
class CloneCensorWeightResult:
    cloned_data: pd.DataFrame
    strategies: list[str]
    n_originals: int
    n_clones: int
    weights_summary: dict

    def __repr__(self) -> str:
        return (
            f"CloneCensorWeightResult(n_originals={self.n_originals}, "
            f"n_clones={self.n_clones}, strategies={self.strategies})"
        )


def clone_censor_weight(
    data: pd.DataFrame,
    id_col: str,
    time_col: str,
    treatment_col: str,
    strategies: dict[str, Callable[[pd.DataFrame], np.ndarray]],
    censor_covariates: Sequence[str] | None = None,
    stabilize: bool = True,
) -> CloneCensorWeightResult:
    """Clone-censor-weight each subject across target-trial strategies.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format (one row per subject-time).
    id_col, time_col, treatment_col : str
        Column names identifying subject, time, and treatment exposure.
    strategies : dict[str, Callable]
        Map strategy name → predicate taking a subject's DataFrame and
        returning a boolean np.ndarray (True where the observed
        treatment is *consistent* with the strategy at that time).
    censor_covariates : list[str], optional
        Covariates used to estimate IP-of-censoring weights after
        cloning. Defaults to all non-key columns.
    stabilize : bool, default True
        Use stabilized IPC weights.

    Returns
    -------
    CloneCensorWeightResult
        ``cloned_data`` holds one row per (id, time, strategy) surviving
        artificial censoring, with an ``_ipcw`` weight column.
    """
    from .ccw_internal import _artificial_censor, _compute_ipcw

    required = {id_col, time_col, treatment_col}
    if not required.issubset(data.columns):
        raise KeyError(f"data must contain columns {required}")

    long = data.sort_values([id_col, time_col]).reset_index(drop=True)
    clones_frames: list[pd.DataFrame] = []

    for strat_name, predicate in strategies.items():
        df_s = long.copy()
        df_s["_strategy"] = strat_name
        df_s["_consistent"] = False
        for subject_id, block in df_s.groupby(id_col, sort=False):
            mask = predicate(block)
            df_s.loc[block.index, "_consistent"] = mask
        df_s = _artificial_censor(df_s, id_col=id_col, time_col=time_col)
        clones_frames.append(df_s)

    cloned = pd.concat(clones_frames, ignore_index=True)

    if censor_covariates is None:
        censor_covariates = [
            c
            for c in data.columns
            if c not in {id_col, time_col, treatment_col}
        ]

    cloned = _compute_ipcw(
        cloned,
        id_col=id_col,
        time_col=time_col,
        censor_covariates=list(censor_covariates),
        stabilize=stabilize,
    )

    return CloneCensorWeightResult(
        cloned_data=cloned,
        strategies=list(strategies.keys()),
        n_originals=data[id_col].nunique(),
        n_clones=int(cloned.shape[0]),
        weights_summary={
            "mean": float(cloned["_ipcw"].mean()),
            "max": float(cloned["_ipcw"].max()),
            "min": float(cloned["_ipcw"].min()),
        },
    )
