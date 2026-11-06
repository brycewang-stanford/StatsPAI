"""Internal helpers for clone-censor-weight."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _artificial_censor(df: pd.DataFrame, id_col: str, time_col: str) -> pd.DataFrame:
    """Rows at risk of artificial censoring, one clone per ``id``.

    A clone is followed from its first row up to and including the first
    row where its observed treatment is inconsistent with the strategy;
    that row carries ``_cens_event = 1`` (artificially censored at that
    time) and every earlier row ``_cens_event = 0``. A clone inconsistent
    at its first row is censored at time zero. ``_censored`` flags clones
    that are censored at some time (constant within a clone).

    The censoring rows are needed to fit the censoring model; they are
    removed after the weights are computed (:func:`_compute_ipcw`).
    """
    kept_rows: list[pd.DataFrame] = []
    for _, block in df.groupby(id_col, sort=False):
        block = block.sort_values(time_col)
        consistent = np.asarray(block["_consistent"].to_numpy(), dtype=bool)
        if consistent.all():
            kept = block.copy()
            kept["_cens_event"] = 0
            kept["_censored"] = 0
        else:
            k = int(np.argmax(~consistent))
            kept = block.iloc[: k + 1].copy()
            event = np.zeros(k + 1, dtype=int)
            event[k] = 1
            kept["_cens_event"] = event
            kept["_censored"] = 1
        kept_rows.append(kept)
    if not kept_rows:
        return df.iloc[0:0].assign(
            _cens_event=pd.Series(dtype=int), _censored=pd.Series(dtype=int)
        )
    return pd.concat(kept_rows, ignore_index=True)


def _compute_ipcw(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    censor_covariates: list[str],
    stabilize: bool,
) -> pd.DataFrame:
    """Attach an ``_ipcw`` column and drop the censoring rows.

    For each strategy a pooled logistic model for remaining uncensored,
    ``P(C_t = 0 | L_t)``, is fitted on every row at risk of censoring
    (the rows before a clone's deviation and the deviation row itself).
    The weight of an uncensored row at time ``t`` is the cumulative
    product over the clone's rows ``j <= t`` of ``1 / P(C_j = 0 | L_j)``;
    stabilised weights multiply each factor by the pooled share of
    uncensored at-risk rows (an intercept-only numerator model). This is
    the inverse probability of censoring weight of Cole and Hernan (2008),
    the quantity ``ipw::ipwtm(type = "cens", family = "binomial")``
    returns. With no censoring covariates no model is fitted and every
    weight is 1.
    """
    from statspai.censoring.ipcw import _fit_logit, _sigmoid

    df = df.copy()
    df["_ipcw"] = 1.0

    for strat, group in df.groupby("_strategy", sort=False):
        Xcols = [c for c in censor_covariates if c in group.columns]
        if not Xcols:
            continue
        X = group[Xcols].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(group)), X])
        y = 1.0 - group["_cens_event"].to_numpy(dtype=float)
        beta = _fit_logit(y, X)
        p_uncensored = np.clip(_sigmoid(X @ beta), 1e-6, 1.0)

        if stabilize:
            y_marg = y.mean()
            y_marg = max(min(y_marg, 1 - 1e-6), 1e-6)
            factor = y_marg / p_uncensored
        else:
            factor = 1.0 / p_uncensored
        # rows are sorted by (id, time) within each clone
        w = (
            pd.Series(factor, index=group.index)
            .groupby(group[id_col].to_numpy(), sort=False)
            .cumprod()
        )
        df.loc[group.index, "_ipcw"] = w.to_numpy()
    out = df.loc[df["_cens_event"] == 0].drop(columns="_cens_event")
    return out.reset_index(drop=True)
