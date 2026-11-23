"""
Data manipulation utilities — Stata-style convenience functions.

Provides:
- pwcorr: Pairwise correlation matrix with significance stars
- winsor: Winsorize variables at specified percentiles
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._frame_input import returns_input_frame
from .._input_validation import check_columns
from ..exceptions import MethodIncompatibility, StatsPAIWarning
from ..exceptions import warn as _warn


def pwcorr(
    data: pd.DataFrame,
    vars: Optional[List[str]] = None,
    stars: bool = True,
    method: str = "pearson",
    decimals: int = 3,
    output: str = "text",
    listwise: bool = False,
    obs: bool = False,
) -> Union[str, pd.DataFrame]:
    """
    Pairwise correlation matrix with significance stars.

    Equivalent to Stata's ``pwcorr var1 var2 var3, star(0.05)``.

    Missing values are handled **pairwise** by default, exactly as Stata's
    ``pwcorr``: each correlation uses every row on which *that pair* is
    observed, so a missing value in a third variable never changes the
    correlation between two complete variables.  Pass ``listwise=True``
    for Stata's ``pwcorr, listwise`` (one common estimation sample).

    Parameters
    ----------
    data : pd.DataFrame
    vars : list of str, optional
        Variables to correlate. Default: all numeric columns.
    stars : bool, default True
        Show significance stars (* p<0.1, ** p<0.05, *** p<0.01).
    method : str, default 'pearson'
        ``'pearson'``, ``'spearman'``, or ``'kendall'``.
    decimals : int, default 3
        Decimal places.
    output : str, default 'text'
        ``'text'``, ``'dataframe'``, ``'latex'``, ``'html'``.
    listwise : bool, default False
        Drop every row with a missing value in any of ``vars`` before
        computing all correlations (Stata ``listwise``).
    obs : bool, default False
        Print the number of observations used for each pair beneath the
        coefficient (Stata ``obs``) in text / LaTeX / HTML output.

    Returns
    -------
    str or pd.DataFrame
        Formatted correlation matrix. With ``output='dataframe'`` the
        correlation matrix is returned and the p-values and pairwise
        observation counts are attached as ``.attrs['pvalues']`` and
        ``.attrs['nobs']`` (both DataFrames).  A pair with fewer than three
        jointly observed rows (or a constant variable) has a NaN
        correlation rather than a fabricated number.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> out = sp.pwcorr(df, vars=['log_wage', 'education', 'experience'])
    >>> isinstance(out, str)
    True
    >>> bool('***' in out)   # significant pairwise correlations are starred
    True
    >>> # Raw correlation matrix instead of formatted text
    >>> cm = sp.pwcorr(df, vars=['log_wage', 'education'], output='dataframe')
    >>> round(float(cm.loc['log_wage', 'education']), 3)
    0.335
    >>> int(cm.attrs['nobs'].loc['log_wage', 'education']) == len(df)
    True
    """
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError(
            f"method must be 'pearson', 'spearman', " f"or 'kendall', got '{method}'"
        )
    if vars is None:
        vars = list(data.select_dtypes(include=[np.number]).columns)
    check_columns(data, {"vars": list(vars)}, function="sp.pwcorr")

    df = data[vars]
    if listwise:
        df = df.dropna()
    k = len(vars)

    corr_matrix = np.full((k, k), np.nan)
    pval_matrix = np.full((k, k), np.nan)
    nobs_matrix = np.zeros((k, k), dtype=int)

    for i in range(k):
        for j in range(i, k):
            pair = df[[vars[i], vars[j]]] if i != j else df[[vars[i]]]
            pair = pair.dropna()
            n_ij = len(pair)
            nobs_matrix[i, j] = nobs_matrix[j, i] = n_ij
            if i == j:
                corr_matrix[i, i] = 1.0 if n_ij > 0 else np.nan
                pval_matrix[i, i] = 0.0 if n_ij > 0 else np.nan
                continue
            x = pair.iloc[:, 0].to_numpy(dtype=float)
            y_val = pair.iloc[:, 1].to_numpy(dtype=float)
            if n_ij < 3 or np.ptp(x) == 0 or np.ptp(y_val) == 0:
                r, p = np.nan, np.nan
            elif method == "pearson":
                r, p = stats.pearsonr(x, y_val)
            elif method == "spearman":
                r, p = stats.spearmanr(x, y_val)
            else:
                r, p = stats.kendalltau(x, y_val)
            corr_matrix[i, j] = corr_matrix[j, i] = float(r)
            pval_matrix[i, j] = pval_matrix[j, i] = float(p)

    if output == "dataframe":
        out = pd.DataFrame(corr_matrix, index=vars, columns=vars)
        out.attrs["pvalues"] = pd.DataFrame(pval_matrix, index=vars, columns=vars)
        out.attrs["nobs"] = pd.DataFrame(nobs_matrix, index=vars, columns=vars)
        out.attrs["missing"] = "listwise" if listwise else "pairwise"
        return out

    # Format with stars (lower triangle only, like Stata)
    def _fmt(val: float, pval: float, show_stars: bool) -> str:
        if not np.isfinite(val):
            return "."
        s = f"{val:.{decimals}f}"
        if show_stars and pval < 0.01:
            s += "***"
        elif show_stars and pval < 0.05:
            s += "**"
        elif show_stars and pval < 0.1:
            s += "*"
        return s

    display = pd.DataFrame("", index=vars, columns=vars)
    for i in range(k):
        for j in range(i + 1):
            cell = _fmt(corr_matrix[i, j], pval_matrix[i, j], stars and i != j)
            if obs:
                cell = f"{cell} (N={nobs_matrix[i, j]})"
            display.iloc[i, j] = cell

    if output == "latex":
        return _pwcorr_latex(display, vars, stars)

    if output == "html":
        return display.to_html()

    lines = [display.to_string()]
    if stars:
        lines.append("")
        lines.append("* p<0.1, ** p<0.05, *** p<0.01")
    off = nobs_matrix[~np.eye(k, dtype=bool)] if k > 1 else nobs_matrix.ravel()
    n_lo, n_hi = (int(off.min()), int(off.max())) if off.size else (0, 0)
    if listwise or n_lo == n_hi:
        lines.append(f"N = {n_hi}")
    else:
        lines.append(f"N = {n_lo} to {n_hi} (pairwise deletion; obs=True shows each)")
    return "\n".join(lines)


def _pwcorr_latex(
    display: pd.DataFrame,
    vars: List[str],
    stars: bool,
) -> str:
    """LaTeX output for pwcorr."""
    k = len(vars)
    spec = "l" + "c" * k
    lines = [
        f"\\begin{{tabular}}{{{spec}}}",
        "\\hline\\hline",
        " & " + " & ".join(vars) + " \\\\",
        "\\hline",
    ]
    for i, var in enumerate(vars):
        row_vals = [display.iloc[i, j] for j in range(k)]
        lines.append(var + " & " + " & ".join(row_vals) + " \\\\")
    lines.append("\\hline\\hline")
    lines.append("\\end{tabular}")
    if stars:
        lines.append("\\\\")
        lines.append("\\footnotesize{* p<0.1, ** p<0.05, *** p<0.01}")
    return "\n".join(lines)


@returns_input_frame
def winsor(
    data: pd.DataFrame,
    vars: Optional[List[str]] = None,
    cuts: Tuple[float, float] = (1, 99),
    replace: bool = False,
    suffix: str = "_w",
    method: str = "stata",
    subset: Optional[Union[str, pd.Series, np.ndarray]] = None,
    by: Optional[Union[str, List[str]]] = None,
    trim: bool = False,
) -> pd.DataFrame:
    """
    Winsorize variables at specified percentiles.

    Equivalent to Stata's ``winsor2 var1 var2, cuts(1 99)``, including its
    percentile definition, ``if`` qualifier (``subset=``) and ``by()``.

    Parameters
    ----------
    data : pd.DataFrame
    vars : list of str, optional
        Variables to winsorize. Default: all numeric columns.
    cuts : tuple of (float, float), default (1, 99)
        Lower and upper percentile cutoffs.
    replace : bool, default False
        If True, overwrite original columns. If False, create new
        columns with ``suffix``.
    suffix : str, default '_w'
        Suffix for new winsorized columns (when ``replace=False``).
    method : str, default 'stata'
        Percentile definition. ``'stata'`` is Stata's ``_pctile``
        default, used by ``winsor2`` (the empirical
        quantile, averaging the two order statistics when ``n p / 100``
        is an integer; numpy's ``'averaged_inverted_cdf'``). Any other
        value is passed to :func:`numpy.percentile` as its ``method``;
        ``'linear'`` reproduces StatsPAI before 1.33.
    subset : str, boolean Series or array, optional
        Rows used to compute the cutoffs and winsorized, as ``winsor2 ...
        if <cond>``: a :meth:`pandas.DataFrame.query` string such as
        ``"year >= 2007 & year <= 2020"`` or a boolean mask. Rows outside
        keep their value when ``replace=True`` and get NaN in the new
        column otherwise (Stata's ``generate ... if``).
    by : str or list of str, optional
        Compute the cutoffs within groups, as ``winsor2 ..., by(g)``.
    trim : bool, default False
        Set the values beyond the cutoffs to missing instead of pulling
        them in (``winsor2 ..., trim``; truncation). The new columns then
        take the suffix ``'_tr'`` unless ``suffix`` is given. Trimming one
        variable at a time removes a different set of rows for each; a
        regression on several trimmed variables loses every row trimmed in
        any of them.

    Returns
    -------
    pd.DataFrame
        DataFrame with winsorized variables. A polars ``DataFrame`` or
        ``LazyFrame`` comes back as one, with the other columns exactly as
        they were passed and missing values as null, so the call fits a
        chain: ``df.pipe(sp.winsor, vars=["x"], replace=True)``.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> # Winsorize at 1st and 99th percentile -> adds 'log_wage_w'
    >>> out = sp.winsor(df, vars=['log_wage'], cuts=(1, 99))
    >>> 'log_wage_w' in out.columns
    True
    >>> bool(out['log_wage_w'].max() <= df['log_wage'].max())  # tails clipped
    True
    >>> len(out) == len(df)   # winsorizing does not drop rows
    True
    >>> # Replace in place instead of adding a suffixed column
    >>> repl = sp.winsor(df, vars=['log_wage'], replace=True)
    >>> 'log_wage_w' in repl.columns
    False

    Notes
    -----
    Winsorization replaces values below the ``cuts[0]``-th percentile
    with that percentile value, and values above the ``cuts[1]``-th
    percentile with that percentile value. Unlike trimming, winsorization
    does not remove observations.

    The ``'stata'`` percentile is also R's ``quantile(type = 2)``, so
    ``sp.winsor(df, [v], cuts=(1, 99))`` equals ``farr::winsorize(v, 0.01)``
    and ``trim=True`` equals ``farr::truncate(v, 0.01)``. With ``by=``
    it equals ``group_by(g) |> mutate(across(v, winsorize))``, with two
    exceptions that each raise a warning: infinite values are left out
    of the cut-offs (farr counts them and pulls them in), and rows with
    a missing group key are left alone (dplyr winsorizes them as one
    more group).

    Before 1.33 the cutoffs used numpy's linear interpolation, which
    differs from ``winsor2`` on every variable whose ``n p / 100`` is not
    an integer; on a long-tailed variable the 99th percentile moved by
    0.02% (51,003.45 against Stata's 51,015).
    """
    df = data.copy()

    if vars is None:
        vars = list(df.select_dtypes(include=[np.number]).columns)
    np_method = "averaged_inverted_cdf" if str(method).lower() == "stata" else method
    if trim and suffix == "_w":
        suffix = "_tr"

    check_columns(df, {"vars": list(vars), "by": by}, function="sp.winsor")
    lo_pct, hi_pct = cuts
    if subset is None:
        in_sample = np.ones(len(df), dtype=bool)
    elif isinstance(subset, str):
        in_sample = df.index.isin(df.query(subset).index)
    else:
        in_sample = np.asarray(subset, dtype=bool)
        if in_sample.shape != (len(df),):
            raise MethodIncompatibility(
                "subset mask must have one entry per row of data"
            )
    if by is None:
        groups = np.zeros(len(df), dtype=np.int64)
    else:
        by_cols = [by] if isinstance(by, str) else list(by)
        groups = df.groupby(by_cols, dropna=False, sort=False).ngroup().to_numpy()
        key_missing = df[by_cols].isna().any(axis=1).to_numpy() & in_sample
        if key_missing.any():
            left = "keep their value" if replace else "get NaN in the new columns"
            _warn(
                StatsPAIWarning,
                f"sp.winsor: {int(key_missing.sum())} row(s) have a missing "
                f"value in by={by_cols}; they get no cut-offs and {left}, as "
                "in Stata's winsor2 by(). dplyr's group_by() (e.g. with "
                "farr::winsorize) treats the missing key as a group of its "
                "own and winsorizes it.",
                stacklevel=3,
                recovery_hint="Drop or fill the rows with a missing group key "
                "before calling sp.winsor.",
                diagnostics={"rows_missing_by": int(key_missing.sum())},
            )
        in_sample = in_sample & ~key_missing

    for var in vars:
        col = df[var].values.astype(float)
        valid = np.isfinite(col)
        use = valid & in_sample
        n_inf = int((np.isinf(col) & in_sample).sum())
        if n_inf:
            left = "stay infinite" if replace else f"get NaN in {var + suffix}"
            _warn(
                StatsPAIWarning,
                f"sp.winsor: {var} has {n_inf} infinite value(s); they are "
                f"left out of the cut-offs and {left}. farr::winsorize (R) "
                "counts them in the quantile and pulls them in to the "
                "cut-offs, which also moves the cut-offs; Stata has no "
                "infinite values.",
                stacklevel=3,
                recovery_hint="Set infinite values to NaN (e.g. from a "
                "division by zero) before winsorizing, or clip them first if "
                "they should count as extreme observations.",
                diagnostics={"variable": var, "n_infinite": n_inf},
            )

        winsorized = col.copy()
        # One stable sort on the group code instead of one mask per group:
        # a mask per group is O(groups x rows) (150 s for 20,000 groups on
        # 2 million rows). The stable sort keeps the rows of a group in
        # data order, so each group reaches np.percentile exactly as before.
        used = np.flatnonzero(use)
        used = used[np.argsort(groups[used], kind="stable")]
        codes = groups[used]
        starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])[: len(used)]
        stops = np.r_[starts[1:], len(used)]
        for start, stop in zip(starts, stops):
            rows = used[start:stop]
            vals = col[rows]
            lo_val, hi_val = np.percentile(vals, [lo_pct, hi_pct], method=np_method)
            if trim:
                winsorized[rows] = np.where(
                    (vals < lo_val) | (vals > hi_val), np.nan, vals
                )
            else:
                winsorized[rows] = np.clip(vals, lo_val, hi_val)

        if replace:
            df[var] = winsorized
        else:
            # Stata's `generate ... if`: rows outside the sample are missing.
            df[var + suffix] = np.where(use, winsorized, np.nan)

    return df
