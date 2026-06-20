"""
Variable label system for pandas DataFrames.

Brings Stata's ``label variable`` functionality to Python. Labels are
stored as DataFrame metadata (``df.attrs['_labels']``) and are
automatically used by StatsPAI output functions (modelsummary, outreg2,
binscatter, etc.).

Usage
-----
>>> import statspai as sp
>>> import pandas as pd
>>> df = pd.DataFrame({'wage': [10.0, 20.0], 'edu': [12, 16]})
>>> sp.label_var(df, 'wage', 'Monthly wage (CNY)')
>>> sp.label_var(df, 'edu', 'Years of education')
>>> sp.get_label(df, 'wage')
'Monthly wage (CNY)'

>>> # Bulk labeling
>>> sp.label_vars(df, {'wage': 'Monthly wage', 'edu': 'Education (years)'})

>>> # Stata-style describe
>>> tbl = sp.describe(df)
>>> list(tbl.columns)
['variable', 'type', 'n', 'n_missing', 'label']
"""

from typing import Optional, Dict, cast

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
    >>> import statspai as sp
    >>> import pandas as pd
    >>> df = pd.DataFrame({'wage': [10.0, 20.0], 'edu': [12, 16]})
    >>> sp.label_var(df, 'wage', 'Monthly wage (CNY)')
    >>> sp.label_var(df, 'edu', 'Years of education')
    >>> sp.get_label(df, 'wage')
    'Monthly wage (CNY)'
    """
    if var not in df.columns:
        raise ValueError(f"Column '{var}' not found in DataFrame")

    if "_labels" not in df.attrs:
        df.attrs["_labels"] = {}
    df.attrs["_labels"][var] = label


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
    >>> import statspai as sp
    >>> import pandas as pd
    >>> df = pd.DataFrame({'wage': [10.0, 20.0],
    ...                    'edu': [12, 16],
    ...                    'exp': [5, 8]})
    >>> sp.label_vars(df, {
    ...     'wage': 'Monthly wage (CNY)',
    ...     'edu': 'Years of education',
    ...     'exp': 'Work experience (years)',
    ... })
    >>> sp.get_label(df, 'exp')
    'Work experience (years)'
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

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> df = pd.DataFrame({'wage': [10.0, 20.0], 'edu': [12, 16]})
    >>> sp.label_var(df, 'wage', 'Monthly wage (CNY)')
    >>> sp.get_label(df, 'wage')
    'Monthly wage (CNY)'
    >>> sp.get_label(df, 'edu')   # unlabeled -> falls back to column name
    'edu'
    """
    labels = cast(Dict[str, str], df.attrs.get("_labels", {}))
    return labels.get(var, var)


def get_labels(df: pd.DataFrame) -> Dict[str, str]:
    """
    Get all variable labels as a dictionary.

    Returns
    -------
    dict
        {column_name: label} for all labeled columns.
        Unlabeled columns map to their own name.

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> df = pd.DataFrame({'wage': [10.0, 20.0], 'edu': [12, 16]})
    >>> sp.label_var(df, 'wage', 'Monthly wage (CNY)')
    >>> sp.get_labels(df)
    {'wage': 'Monthly wage (CNY)', 'edu': 'edu'}
    """
    labels = cast(Dict[str, str], df.attrs.get("_labels", {}))
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
    >>> import statspai as sp
    >>> import pandas as pd
    >>> df = pd.DataFrame({'wage': [10.0, 20.0, 15.0],
    ...                    'edu': [12, 16, 14]})
    >>> sp.label_var(df, 'wage', 'Monthly wage (CNY)')
    >>> tbl = sp.describe(df)
    >>> list(tbl.columns)
    ['variable', 'type', 'n', 'n_missing', 'label']
    >>> tbl['label'].tolist()
    ['Monthly wage (CNY)', '']
    """
    cols = columns or list(df.columns)
    labels = cast(Dict[str, str], df.attrs.get("_labels", {}))

    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        rows.append(
            {
                "variable": col,
                "type": str(df[col].dtype),
                "n": int(df[col].notna().sum()),
                "n_missing": int(df[col].isna().sum()),
                "label": labels.get(col, ""),
            }
        )

    return pd.DataFrame(rows)
