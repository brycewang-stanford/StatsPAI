"""
Variable label system for pandas DataFrames.

Brings Stata's ``label variable`` functionality to Python. Labels are
stored as DataFrame metadata (``df.attrs['_labels']``) and are
automatically used by StatsPAI output functions (modelsummary, outreg2,
binscatter, etc.).

Usage
-----
>>> sp.label_var(df, 'wage', 'Monthly wage (CNY)')
>>> sp.label_var(df, 'edu', 'Years of education')
>>> sp.get_label(df, 'wage')
'Monthly wage (CNY)'

>>> # Bulk labeling
>>> sp.label_vars(df, {'wage': 'Monthly wage', 'edu': 'Education (years)'})

>>> # Stata-style describe
>>> sp.describe(df)
"""

from typing import Optional, Dict

import pandas as pd


def label_var(df: pd.DataFrame, var: str, label: str) -> None:
    """
    Attach a human-readable label to a variable.

    Equivalent to Stata's ``label variable wage "Monthly wage (CNY)"``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to label (modified in place).
    var : str
        Column name.
    label : str
        Human-readable label.

    Examples
    --------
    >>> sp.label_var(df, 'wage', 'Monthly wage (CNY)')
    >>> sp.label_var(df, 'edu', 'Years of education')
    """
    if var not in df.columns:
        raise ValueError(f"Column '{var}' not found in DataFrame")

    if '_labels' not in df.attrs:
        df.attrs['_labels'] = {}
    df.attrs['_labels'][var] = label


def label_vars(df: pd.DataFrame, labels: Dict[str, str]) -> None:
    """
    Attach labels to multiple variables at once.

    Parameters
    ----------
    df : pd.DataFrame
    labels : dict
        {column_name: label_string} mapping.

    Examples
    --------
    >>> sp.label_vars(df, {
    ...     'wage': 'Monthly wage (CNY)',
    ...     'edu': 'Years of education',
    ...     'exp': 'Work experience (years)',
    ... })
    """
    for var, label in labels.items():
        label_var(df, var, label)


def get_label(df: pd.DataFrame, var: str) -> str:
    """
    Get the label for a variable, falling back to the column name.

    Parameters
    ----------
    df : pd.DataFrame
    var : str

    Returns
    -------
    str
        The label if set, otherwise the column name itself.
    """
    labels = df.attrs.get('_labels', {})
    return labels.get(var, var)


def get_labels(df: pd.DataFrame) -> Dict[str, str]:
    """
    Get all variable labels as a dictionary.

    Returns
    -------
    dict
        {column_name: label} for all labeled columns.
        Unlabeled columns map to their own name.
    """
    labels = df.attrs.get('_labels', {})
    return {col: labels.get(col, col) for col in df.columns}


def describe(
    df: pd.DataFrame,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Stata-style ``describe`` — variable names, types, labels, and
    non-missing counts in one table.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list, optional
        Subset of columns. Default: all.

    Returns
    -------
    pd.DataFrame
        Columns: variable, type, n, n_missing, label.

    Examples
    --------
    >>> sp.describe(df)
       variable    type     n  n_missing              label
    0      wage  float64  1000         0  Monthly wage (CNY)
    1       edu    int64  1000         0  Years of education
    2    female    int64   998         2            female
    """
    cols = columns or list(df.columns)
    labels = df.attrs.get('_labels', {})

    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        rows.append({
            'variable': col,
            'type': str(df[col].dtype),
            'n': int(df[col].notna().sum()),
            'n_missing': int(df[col].isna().sum()),
            'label': labels.get(col, ''),
        })

    return pd.DataFrame(rows)
