"""
Smart data I/O with variable label preservation.

Reads Stata .dta, SAS .sas7bdat, SPSS .sav, CSV, Excel, and Parquet
files, automatically preserving variable labels when available.

For Stata .dta files, variable labels are stored in ``df.attrs['_labels']``
and are used by StatsPAI output functions (modelsummary, outreg2, etc.).

References
----------
This addresses the #1 pain point for Stata → Python migration:
pandas' ``read_stata()`` loses variable labels silently.
"""

from pathlib import Path
from typing import Any, Optional

import pandas as pd


def read_data(
    path: str,
    encoding: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read data from any common format, preserving variable labels.

    Automatically detects format from file extension and stores
    Stata/SPSS/SAS variable labels in ``df.attrs['_labels']``.

    Parameters
    ----------
    path : str
        File path. Supported: .dta, .csv, .xlsx, .xls, .parquet,
        .sas7bdat, .sav, .feather, .json.
    encoding : str, optional
        Character encoding (for CSV).
    **kwargs
        Passed to the underlying pandas reader.

    Returns
    -------
    pd.DataFrame
        With variable labels in ``df.attrs['_labels']`` if available.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.read_data('survey.dta')  # doctest: +SKIP
    >>> sp.describe(df)  # shows variable labels from Stata  # doctest: +SKIP
    >>> sp.get_label(df, 'wage')  # doctest: +SKIP
    'Monthly wage in CNY'

    >>> df = sp.read_data('data.csv')  # CSV has no labels  # doctest: +SKIP
    >>> sp.label_vars(df, {'wage': 'Monthly wage', 'edu': 'Education'})  # doctest: +SKIP
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".dta":
        df = _read_stata(path, **kwargs)
    elif ext in (".csv", ".tsv"):
        df = pd.read_csv(path, encoding=encoding, **kwargs)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, **kwargs)
    elif ext == ".parquet":
        df = pd.read_parquet(path, **kwargs)
    elif ext == ".feather":
        df = pd.read_feather(path, **kwargs)
    elif ext == ".json":
        df = pd.read_json(path, **kwargs)
    elif ext == ".sas7bdat":
        df = _read_sas(path, **kwargs)
    elif ext == ".sav":
        df = _read_spss(path, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported: .dta, .csv, .xlsx, .parquet, .sas7bdat, .sav"
        )

    return df


def _read_stata(path: str, **kwargs: Any) -> pd.DataFrame:
    """Read .dta with variable and value labels preserved.

    ``pyreadstat`` (``pip install statspai[io]``) is preferred.  Without it
    the pandas reader is used; it keeps variable labels and value labels as
    well, and value-labelled columns keep their numeric codes (as with
    pyreadstat) rather than being converted to categoricals, so the two
    paths return the same frame layout.
    """
    try:
        import pyreadstat
    except ImportError:
        return _read_stata_pandas(path, **kwargs)

    df, meta = pyreadstat.read_dta(path, **kwargs)
    if meta.column_names_to_labels:
        df.attrs["_labels"] = {
            k: v for k, v in meta.column_names_to_labels.items() if v
        }
    if meta.variable_value_labels:
        df.attrs["_value_labels"] = meta.variable_value_labels
    return df


def _read_stata_pandas(path: str, **kwargs: Any) -> pd.DataFrame:
    """pandas fallback for .dta that still carries labels into ``attrs``."""
    kwargs.setdefault("convert_categoricals", False)
    with pd.read_stata(path, iterator=True, **kwargs) as reader:
        df = reader.read()
        var_labels = reader.variable_labels()
        label_sets = reader.value_labels()
        # value_labels() is keyed by label-set name; map sets to variables.
        lbl_names = getattr(reader, "_lbllist", None) or []
    labels = {k: v for k, v in var_labels.items() if v}
    if labels:
        df.attrs["_labels"] = labels
    if not lbl_names:
        # Private attribute gone in a future pandas: fall back to the common
        # Stata convention of naming a label set after its variable.
        lbl_names = [c if c in label_sets else "" for c in df.columns]
    value_labels = {}
    for col, lbl in zip(df.columns, lbl_names):
        if lbl and lbl in label_sets:
            value_labels[col] = {
                (k.item() if hasattr(k, "item") else k): v
                for k, v in label_sets[lbl].items()
            }
    if value_labels:
        df.attrs["_value_labels"] = value_labels
    return df


def _read_sas(path: str, **kwargs: Any) -> pd.DataFrame:
    """Read SAS .sas7bdat with labels."""
    try:
        import pyreadstat

        df, meta = pyreadstat.read_sas7bdat(path, **kwargs)
        if meta.column_names_to_labels:
            df.attrs["_labels"] = {
                k: v for k, v in meta.column_names_to_labels.items() if v
            }
        return df
    except ImportError:
        return pd.read_sas(path, **kwargs)


def _read_spss(path: str, **kwargs: Any) -> pd.DataFrame:
    """Read SPSS .sav with labels."""
    try:
        import pyreadstat

        df, meta = pyreadstat.read_sav(path, **kwargs)
        if meta.column_names_to_labels:
            df.attrs["_labels"] = {
                k: v for k, v in meta.column_names_to_labels.items() if v
            }
        return df
    except ImportError:
        return pd.read_spss(path, **kwargs)
