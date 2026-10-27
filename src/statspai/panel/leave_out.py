"""Leave-out (peer) means -- ``sp.leave_out_mean``.

The mean of a variable over the *other* members of a row's group: the
peer average of coworkers, classmates or neighbours. Rows flagged by
``exclude`` (e.g. the focal entrants) are kept out of every peer pool but
still receive the mean of their peers. A row with no peer left gets NaN,
counted and reported, never a silent zero.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..exceptions import ColumnNotFound, MethodIncompatibility, StatsPAIWarning

__all__ = ["leave_out_mean"]


def _flag(
    data: pd.DataFrame, exclude: Union[None, str, Sequence[bool], np.ndarray, pd.Series]
) -> np.ndarray:
    n = len(data)
    if exclude is None:
        return np.zeros(n, dtype=bool)
    if isinstance(exclude, str):
        if exclude not in data.columns:
            raise ColumnNotFound(
                f"leave_out_mean: exclude={exclude!r} is not a column.",
                recovery_hint="Pass a boolean column name or a boolean array "
                "of the data's length.",
            )
        raw = data[exclude]
    else:
        raw = exclude
    arr = pd.Series(np.asarray(raw)) if not isinstance(raw, pd.Series) else raw
    if len(arr) != n:
        raise MethodIncompatibility(
            f"leave_out_mean: exclude has {len(arr)} entries, data has {n} rows.",
            recovery_hint="Pass one flag per row, in row order.",
        )
    if arr.isna().any():
        raise MethodIncompatibility(
            "leave_out_mean: exclude has missing values; whether those rows "
            "are peers is undefined.",
            recovery_hint="Fill the flag (True / False) for every row.",
            diagnostics={"n_missing": int(arr.isna().sum())},
        )
    vals = arr.to_numpy()
    if vals.dtype != bool:
        uniq = set(np.unique(vals).tolist())
        if not uniq <= {0, 1}:
            raise MethodIncompatibility(
                f"leave_out_mean: exclude must be boolean (or 0/1); got values "
                f"{sorted(uniq)[:5]}.",
                recovery_hint="Pass a True / False flag.",
            )
        vals = vals.astype(bool)
    return vals


def leave_out_mean(
    data: pd.DataFrame,
    value: str,
    group: Union[str, Sequence[str]],
    *,
    exclude: Union[None, str, Sequence[bool], np.ndarray, pd.Series] = None,
    weights: Optional[str] = None,
) -> pd.Series:
    """Mean of ``value`` over the other rows of the same group.

    For row ``i`` in group ``g`` the result is ``sum_j w_j v_j / sum_j w_j``
    over rows ``j != i`` of ``g`` that are not excluded and have a
    non-missing ``value`` (and weight). Excluded rows get the mean of their
    group's non-excluded rows.

    Parameters
    ----------
    data : DataFrame
    value : str
        Column to average (e.g. the standardised worker fixed effect).
    group : str or list of str
        Group column(s); several columns define one group per combination
        (e.g. ``['firm', 'occupation', 'entry_year']``). Rows with a missing
        group key belong to no group and get NaN.
    exclude : str, bool array or Series, optional
        Rows kept out of every peer pool (a column name or one flag per
        row), e.g. the focal entrants.
    weights : str, optional
        Non-negative weight column.

    Returns
    -------
    pd.Series
        Aligned with ``data`` (same index, duplicates included), named
        ``f"{value}_loo"``. ``.attrs`` holds ``n_no_peers`` (rows with no
        peer left, which are NaN) and ``n_missing_group``.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> d = pd.DataFrame({"firm": [1, 1, 1, 2, 2], "fe": [1.0, 2.0, 6.0, 3.0, 5.0],
    ...                   "entrant": [False, False, True, False, True]})
    >>> sp.leave_out_mean(d, "fe", "firm", exclude="entrant").tolist()
    [2.0, 1.0, 1.5, nan, 3.0]

    References
    ----------
    None: the result is an arithmetic identity, not an estimator. A typical
    use is the quality of a worker's coworkers measured by their estimated
    fixed effects (``sp.akm``) with the focal entrants left out.
    """
    groups = [group] if isinstance(group, str) else list(group)
    if not groups:
        raise MethodIncompatibility(
            "leave_out_mean: group= needs at least one column.",
            recovery_hint="Pass the column(s) that define the peer groups.",
        )
    need = [value] + groups + ([weights] if weights else [])
    missing = [c for c in need if c not in data.columns]
    if missing:
        raise ColumnNotFound(
            f"leave_out_mean: column(s) {missing} not in data.",
            recovery_hint="Pass existing column names.",
            diagnostics={"missing": missing},
        )
    n = len(data)
    excl = _flag(data, exclude)
    v = pd.to_numeric(data[value], errors="coerce").to_numpy(dtype=float)
    if weights:
        w = pd.to_numeric(data[weights], errors="coerce").to_numpy(dtype=float)
        if np.any(w[np.isfinite(w)] < 0):
            raise MethodIncompatibility(
                "leave_out_mean: weights must be non-negative.",
                recovery_hint="Drop or fix the negative weights.",
            )
    else:
        w = np.ones(n)
    keys = data[groups]
    has_group = keys.notna().all(axis=1).to_numpy()
    if len(groups) == 1:
        codes, _ = pd.factorize(keys.iloc[:, 0])
    else:
        codes = keys.groupby(groups, sort=False, dropna=True).ngroup().to_numpy()
    codes = np.where(has_group, codes, -1)
    G = int(codes.max()) + 1 if (codes >= 0).any() else 0
    # a row contributes as a peer when it is in a group, not excluded and
    # has a usable value and weight
    peer = (codes >= 0) & ~excl & np.isfinite(v) & np.isfinite(w)
    cg = np.where(peer, codes, 0)
    S = np.bincount(cg, weights=np.where(peer, w * v, 0.0), minlength=G)
    W = np.bincount(cg, weights=np.where(peer, w, 0.0), minlength=G)
    out = np.full(n, np.nan)
    in_g = codes >= 0
    num = S[codes[in_g]] - np.where(peer[in_g], w[in_g] * v[in_g], 0.0)
    den = W[codes[in_g]] - np.where(peer[in_g], w[in_g], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[in_g] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    no_peer = int(np.sum(in_g & ~np.isfinite(out)))
    n_missing_group = int(np.sum(~in_g))
    if no_peer or n_missing_group:
        warnings.warn(
            StatsPAIWarning(
                f"leave_out_mean: {no_peer} row(s) have no peer left in their "
                f"group and {n_missing_group} row(s) have a missing group key; "
                "their leave-out mean is NaN.",
                recovery_hint="Drop those rows or use a coarser group; check "
                "result.attrs['n_no_peers'].",
                diagnostics={
                    "n_no_peers": no_peer,
                    "n_missing_group": n_missing_group,
                },
            ),
            stacklevel=2,
        )
    res = pd.Series(out, index=data.index, name=f"{value}_loo")
    res.attrs["n_no_peers"] = no_peer
    res.attrs["n_missing_group"] = n_missing_group
    return res
