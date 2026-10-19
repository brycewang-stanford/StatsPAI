"""Stabilized inverse-probability-of-censoring weights for ``sp.msm``.

Person-period data: one row per (unit, period) up to and including the
period in which the unit is censored, ``censor = 1`` on that row (its
outcome is not observed) and 0 elsewhere. For each row the per-period
ratio is

    P(C_t = 0 | A_{t-1}, V) / P(C_t = 0 | A_{t-1}, V, L_t),

both pooled logistic regressions over every row at risk of censoring,
and the weight of an uncensored row is the cumulative product of the
ratios up to its period, as R ``ipw::ipwtm(family = "binomial", type =
"first")`` builds it. The MSM fits the uncensored rows with the product of
these weights and the treatment weights.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility


def censoring_weights(
    data: pd.DataFrame,
    censor: str,
    treat: str,
    id: str,
    time: str,
    numerator: List[str],
    denominator: List[str],
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Cumulative stabilized censoring weight of every row (indexed like
    ``data``) and a diagnostics dict."""
    from .msm import _logit_proba

    df = data.sort_values([id, time])
    c = df[censor]
    if c.isna().any() or not set(np.unique(c.to_numpy(dtype=float))) <= {0.0, 1.0}:
        raise MethodIncompatibility(
            f"sp.msm: the censoring indicator {censor!r} must be 0/1 with no "
            "missing values.",
            recovery_hint="Code 1 on the period in which the unit is censored, 0 "
            "elsewhere.",
        )
    c_arr = c.to_numpy(dtype=float)
    ids = df[id].to_numpy()
    # censoring must be the unit's last row
    after = (
        pd.Series(c_arr, index=df.index).groupby(ids, sort=False).cumsum().to_numpy()
        - c_arr
    )
    if (after > 0).any():
        raise MethodIncompatibility(
            f"sp.msm: some units have rows after their censoring period "
            f"({int((after > 0).sum())} rows).",
            recovery_hint="Drop every row after the period in which censor = 1.",
        )
    a_lag = df.groupby(id, sort=False)[treat].shift(1).fillna(0.0).to_numpy(dtype=float)
    cols_num = df[numerator].to_numpy(dtype=float) if numerator else None
    cols_den = df[denominator].to_numpy(dtype=float) if denominator else None
    parts_num = [a_lag[:, None]] + ([cols_num] if cols_num is not None else [])
    parts_den = parts_num + ([cols_den] if cols_den is not None else [])
    X_num = np.column_stack(parts_num)
    X_den = np.column_stack(parts_den)
    ok = np.isfinite(X_den).all(axis=1)
    if not ok.all():
        raise DataInsufficient(
            "sp.msm: the censoring model's covariates are missing on "
            f"{int((~ok).sum())} rows (censored rows too must carry them).",
            recovery_hint="Fill or drop those rows; the censoring model needs "
            "the lagged treatment, the baseline covariates and "
            "censor_covariates on every row at risk of censoring.",
        )
    if c_arr.sum() == 0:
        raise DataInsufficient(
            "sp.msm: censor= was given but no row is censored.",
            recovery_hint="Drop censor=.",
        )
    p_num = _logit_proba(X_num, c_arr)
    p_den = _logit_proba(X_den, c_arr)
    ratio = (1.0 - p_num) / (1.0 - p_den)
    cw = pd.Series(ratio, index=df.index).groupby(ids, sort=False).cumprod()
    unc = c_arr == 0
    diag = {
        "censor": censor,
        "n_censored_units": int(c_arr.sum()),
        "n_rows_at_risk": int(len(df)),
        "cw_mean": float(cw[unc].mean()),
        "cw_min": float(cw[unc].min()),
        "cw_max": float(cw[unc].max()),
        "numerator": ["lag(" + treat + ")"] + list(numerator),
        "denominator": ["lag(" + treat + ")"] + list(numerator) + list(denominator),
    }
    return cw.reindex(data.index), diag
