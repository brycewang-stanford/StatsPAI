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

from ..exceptions import MethodIncompatibility


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


def winsor(
    data: pd.DataFrame,
    vars: Optional[List[str]] = None,
    cuts: Tuple[float, float] = (1, 99),
    replace: bool = False,
    suffix: str = "_w",
    method: str = "stata",
    subset: Optional[Union[str, pd.Series, np.ndarray]] = None,
    by: Optional[Union[str, List[str]]] = None,
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

    Returns
    -------
    pd.DataFrame
        DataFrame with winsorized variables.

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

    Before 1.33 the cutoffs used numpy's linear interpolation, which
    differs from ``winsor2`` on every variable whose ``n p / 100`` is not
    an integer; on a long-tailed variable the 99th percentile moved by
    0.02% (51,003.45 against Stata's 51,015).
    """
    df = data.copy()

    if vars is None:
        vars = list(df.select_dtypes(include=[np.number]).columns)
    np_method = "averaged_inverted_cdf" if str(method).lower() == "stata" else method

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
        in_sample = in_sample & df[by_cols].notna().all(axis=1).to_numpy()

    for var in vars:
        if var not in df.columns:
            raise ValueError(f"Column '{var}' not found")

        col = df[var].values.astype(float)
        valid = np.isfinite(col)
        use = valid & in_sample

        winsorized = col.copy()
        for gcode in np.unique(groups[use]):
            rows = use & (groups == gcode)
            lo_val, hi_val = np.percentile(
                col[rows], [lo_pct, hi_pct], method=np_method
            )
            winsorized[rows] = np.clip(col[rows], lo_val, hi_val)

        if replace:
            df[var] = winsorized
        else:
            # Stata's `generate ... if`: rows outside the sample are missing.
            df[var + suffix] = np.where(use, winsorized, np.nan)

    return df
