"""Internal helpers for clone-censor-weight."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _artificial_censor(
    df: pd.DataFrame, id_col: str, time_col: str
) -> pd.DataFrame:
    """Censor each clone the moment they deviate from the strategy."""
    kept_rows: list[pd.DataFrame] = []
    for _, block in df.groupby(id_col, sort=False):
        block = block.sort_values(time_col)
        consistent = block["_consistent"].to_numpy()
        if consistent.any() and not consistent[0]:
            continue
        first_inconsistent = np.argmax(~consistent) if (~consistent).any() else len(consistent)
        if not consistent[0]:
            first_inconsistent = 0
        if first_inconsistent == 0 and not consistent[0]:
            continue
        kept = block.iloc[:first_inconsistent].copy()
        kept["_censored"] = int(first_inconsistent < len(consistent))
        kept_rows.append(kept)
    if not kept_rows:
        return df.iloc[0:0].assign(_censored=pd.Series(dtype=int))
    return pd.concat(kept_rows, ignore_index=True)


def _compute_ipcw(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    censor_covariates: list[str],
    stabilize: bool,
) -> pd.DataFrame:
    """Attach an ``_ipcw`` column using a pooled-logistic censoring
    model fit once per strategy."""
    from statspai.censoring.ipcw import _fit_logit, _sigmoid

    df = df.copy()
    df["_ipcw"] = 1.0

    for strat, group in df.groupby("_strategy", sort=False):
        Xcols = [c for c in censor_covariates if c in group.columns]
        if not Xcols:
            continue
        X = group[Xcols].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(group)), X])
        y = 1.0 - group["_censored"].to_numpy(dtype=float)
        beta = _fit_logit(y, X)
        p_uncensored = np.clip(_sigmoid(X @ beta), 1e-6, 1.0)

        if stabilize:
            y_marg = y.mean()
            y_marg = max(min(y_marg, 1 - 1e-6), 1e-6)
            w = y_marg / p_uncensored
        else:
            w = 1.0 / p_uncensored
        df.loc[group.index, "_ipcw"] = w
    return df
