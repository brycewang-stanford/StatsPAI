"""
Data manipulation utilities — Stata-style convenience functions.

Provides:
- pwcorr: Pairwise correlation matrix with significance stars
- winsor: Winsorize variables at specified percentiles
"""

from typing import Optional, List, Dict, Union, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def pwcorr(
    data: pd.DataFrame,
    vars: Optional[List[str]] = None,
    stars: bool = True,
    method: str = 'pearson',
    decimals: int = 3,
    output: str = 'text',
) -> Union[str, pd.DataFrame]:
    """
    Pairwise correlation matrix with significance stars.

    Equivalent to Stata's ``pwcorr var1 var2 var3, star(0.05)``.

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

    Returns
    -------
    str or pd.DataFrame
        Formatted correlation matrix.

    Examples
    --------
    >>> print(sp.pwcorr(df, vars=['wage', 'education', 'experience']))
                    wage    education  experience
    wage           1.000
    education     0.523***   1.000
    experience    0.412***   0.156**     1.000
    """
    if vars is None:
        vars = list(data.select_dtypes(include=[np.number]).columns)

    df = data[vars].dropna()
    n = len(df)
    k = len(vars)

    # Compute correlations and p-values
    corr_matrix = np.zeros((k, k))
    pval_matrix = np.zeros((k, k))

    for i in range(k):
        for j in range(k):
            if i == j:
                corr_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            elif j < i:
                # Already computed
                corr_matrix[i, j] = corr_matrix[j, i]
                pval_matrix[i, j] = pval_matrix[j, i]
            else:
                x, y_val = df[vars[i]].values, df[vars[j]].values
                if method == 'pearson':
                    r, p = stats.pearsonr(x, y_val)
                elif method == 'spearman':
                    r, p = stats.spearmanr(x, y_val)
                elif method == 'kendall':
                    r, p = stats.kendalltau(x, y_val)
                else:
                    raise ValueError(f"method must be 'pearson', 'spearman', "
                                     f"or 'kendall', got '{method}'")
                corr_matrix[i, j] = r
                pval_matrix[i, j] = p

    # Format with stars (lower triangle only, like Stata)
    def _fmt(val, pval, show_stars):
        s = f'{val:.{decimals}f}'
        if show_stars and pval < 0.01:
            s += '***'
        elif show_stars and pval < 0.05:
            s += '**'
        elif show_stars and pval < 0.1:
            s += '*'
        return s

    # Build display matrix (lower triangular)
    display = pd.DataFrame('', index=vars, columns=vars)
    for i in range(k):
        for j in range(k):
            if j > i:
                display.iloc[i, j] = ''  # upper triangle empty
            elif i == j:
                display.iloc[i, j] = f'{1.0:.{decimals}f}'
            else:
                display.iloc[i, j] = _fmt(
                    corr_matrix[i, j], pval_matrix[i, j], stars)

    if output == 'dataframe':
        # Return raw correlation + pvalue DataFrames
        return pd.DataFrame(corr_matrix, index=vars, columns=vars)

    if output == 'latex':
        return _pwcorr_latex(display, vars, stars)

    if output == 'html':
        return display.to_html()

    # Text output
    lines = []
    lines.append(display.to_string())
    if stars:
        lines.append('')
        lines.append('* p<0.1, ** p<0.05, *** p<0.01')
    lines.append(f'N = {n}')
    return '\n'.join(lines)


def _pwcorr_latex(display, vars, stars):
    """LaTeX output for pwcorr."""
    k = len(vars)
    spec = 'l' + 'c' * k
    lines = [
        f'\\begin{{tabular}}{{{spec}}}',
        '\\hline\\hline',
        ' & ' + ' & '.join(vars) + ' \\\\',
        '\\hline',
    ]
    for i, var in enumerate(vars):
        row_vals = [display.iloc[i, j] for j in range(k)]
        lines.append(var + ' & ' + ' & '.join(row_vals) + ' \\\\')
    lines.append('\\hline\\hline')
    lines.append('\\end{tabular}')
    if stars:
        lines.append('\\\\')
        lines.append('\\footnotesize{* p<0.1, ** p<0.05, *** p<0.01}')
    return '\n'.join(lines)


def winsor(
    data: pd.DataFrame,
    vars: Optional[List[str]] = None,
    cuts: Tuple[float, float] = (1, 99),
    replace: bool = False,
    suffix: str = '_w',
) -> pd.DataFrame:
    """
    Winsorize variables at specified percentiles.

    Equivalent to Stata's ``winsor2 var1 var2, cuts(1 99)``.

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

    Returns
    -------
    pd.DataFrame
        DataFrame with winsorized variables.

    Examples
    --------
    >>> # Winsorize at 1st and 99th percentile
    >>> df = sp.winsor(df, vars=['wage', 'income'], cuts=(1, 99))
    >>> # Now df has 'wage_w' and 'income_w' columns

    >>> # Replace in place
    >>> df = sp.winsor(df, vars=['wage'], replace=True)

    Notes
    -----
    Winsorization replaces values below the ``cuts[0]``-th percentile
    with that percentile value, and values above the ``cuts[1]``-th
    percentile with that percentile value. Unlike trimming, winsorization
    does not remove observations.
    """
    df = data.copy()

    if vars is None:
        vars = list(df.select_dtypes(include=[np.number]).columns)

    lo_pct, hi_pct = cuts

    for var in vars:
        if var not in df.columns:
            raise ValueError(f"Column '{var}' not found")

        col = df[var].values.astype(float)
        valid = np.isfinite(col)

        lo_val = np.nanpercentile(col[valid], lo_pct)
        hi_val = np.nanpercentile(col[valid], hi_pct)

        winsorized = np.clip(col, lo_val, hi_val)
        # Preserve NaN
        winsorized = np.where(valid, winsorized, np.nan)

        if replace:
            df[var] = winsorized
        else:
            df[var + suffix] = winsorized

    return df
