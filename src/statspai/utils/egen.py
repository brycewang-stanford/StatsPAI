"""
Stata-style egen row-wise and rank helpers.

Ports the core ``pyegen`` convenience functions to StatsPAI so users
migrating from PyStataR keep a familiar API. All functions accept a
DataFrame plus a list of variables and return a pandas Series aligned
to the DataFrame's index.

Equivalent to Stata's::

    egen newvar = rowmean(var1 var2 var3)
    egen newvar = rowtotal(var1 var2 var3)
    egen newvar = rowmax(var1 var2 var3)
    egen newvar = rowmin(var1 var2 var3)
    egen newvar = rowsd(var1 var2 var3)
    egen newvar = rownonmiss(var1 var2 var3)
    egen newvar = rank(var), [by(group)] [field|track|unique]
"""

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd


def _select(data: pd.DataFrame, vars: Sequence[str]) -> pd.DataFrame:
    missing = [v for v in vars if v not in data.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    return data[list(vars)]


def rowmean(data: pd.DataFrame, vars: Sequence[str]) -> pd.Series:
    """Row-wise mean across ``vars``, ignoring NaN (Stata ``egen rowmean``)."""
    return _select(data, vars).mean(axis=1, skipna=True)


def rowtotal(data: pd.DataFrame, vars: Sequence[str]) -> pd.Series:
    """Row-wise sum across ``vars``, treating NaN as 0 (Stata ``egen rowtotal``)."""
    return _select(data, vars).sum(axis=1, skipna=True)


def rowmax(data: pd.DataFrame, vars: Sequence[str]) -> pd.Series:
    """Row-wise max across ``vars``, ignoring NaN."""
    return _select(data, vars).max(axis=1, skipna=True)


def rowmin(data: pd.DataFrame, vars: Sequence[str]) -> pd.Series:
    """Row-wise min across ``vars``, ignoring NaN."""
    return _select(data, vars).min(axis=1, skipna=True)


def rowsd(data: pd.DataFrame, vars: Sequence[str]) -> pd.Series:
    """Row-wise standard deviation across ``vars`` (Stata ``egen rowsd``)."""
    return _select(data, vars).std(axis=1, skipna=True, ddof=1)


def rowcount(data: pd.DataFrame, vars: Sequence[str]) -> pd.Series:
    """Row-wise count of non-missing values (Stata ``egen rownonmiss``)."""
    return _select(data, vars).notna().sum(axis=1).astype(int)


def rank(
    data: pd.DataFrame,
    var: str,
    by: Optional[Union[str, Sequence[str]]] = None,
    method: str = 'average',
    ascending: bool = True,
) -> pd.Series:
    """
    Rank a variable, optionally within groups.

    Equivalent to Stata's ``egen newvar = rank(var), by(group)`` with
    ``method`` mapping to Stata's tie-handling options:

    - ``'average'`` → Stata default
    - ``'min'``     → ``field``
    - ``'dense'``   → ``track``
    - ``'first'``   → ``unique``

    Parameters
    ----------
    data : pd.DataFrame
    var : str
        Column to rank.
    by : str or list of str, optional
        Group variables. If given, ranks are computed within each group.
    method : {'average', 'min', 'max', 'dense', 'first'}, default 'average'
    ascending : bool, default True

    Returns
    -------
    pd.Series aligned to ``data.index``.
    """
    if var not in data.columns:
        raise ValueError(f"Column '{var}' not found")

    if by is None:
        return data[var].rank(method=method, ascending=ascending)

    by_cols = [by] if isinstance(by, str) else list(by)
    missing = [c for c in by_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Group columns not found: {missing}")

    return data.groupby(by_cols, dropna=False)[var].rank(
        method=method, ascending=ascending
    )


def outlier_indicator(
    data: pd.DataFrame,
    vars: Sequence[str],
    cuts: tuple = (1, 99),
    combined: bool = False,
) -> pd.DataFrame:
    """
    Flag outlier observations by percentile cutoffs.

    Returns 0/1 indicator columns for values outside ``[cuts[0], cuts[1]]``
    percentile bounds. Mirrors the ``pywinsor2.outlier_indicator`` API.

    Parameters
    ----------
    data : pd.DataFrame
    vars : list of str
        Columns to check.
    cuts : tuple of (float, float), default (1, 99)
        Lower and upper percentile cutoffs.
    combined : bool, default False
        If True, also add a column ``_outlier_any`` marking rows that are
        outliers on *any* variable.

    Returns
    -------
    pd.DataFrame
        Copy of ``data`` with added ``{var}_outlier`` 0/1 columns.
    """
    df = data.copy()
    lo_pct, hi_pct = cuts
    flag_cols = []

    for v in vars:
        if v not in df.columns:
            raise ValueError(f"Column '{v}' not found")
        col = df[v].astype(float)
        valid = col.notna()
        if valid.any():
            lo = np.nanpercentile(col[valid], lo_pct)
            hi = np.nanpercentile(col[valid], hi_pct)
            flag = ((col < lo) | (col > hi)).astype(int)
        else:
            flag = pd.Series(0, index=col.index, dtype=int)
        flag[~valid] = 0
        name = f'{v}_outlier'
        df[name] = flag
        flag_cols.append(name)

    if combined and flag_cols:
        df['_outlier_any'] = (df[flag_cols].sum(axis=1) > 0).astype(int)

    return df
