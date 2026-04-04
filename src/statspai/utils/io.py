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

from typing import Optional, Dict
from pathlib import Path

import pandas as pd


def read_data(
    path: str,
    encoding: Optional[str] = None,
    **kwargs,
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
    >>> df = sp.read_data('survey.dta')
    >>> sp.describe(df)  # shows variable labels from Stata
    >>> sp.get_label(df, 'wage')
    'Monthly wage in CNY'

    >>> df = sp.read_data('data.csv')  # CSV has no labels
    >>> sp.label_vars(df, {'wage': 'Monthly wage', 'edu': 'Education'})
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == '.dta':
        df = _read_stata(path, **kwargs)
    elif ext in ('.csv', '.tsv'):
        df = pd.read_csv(path, encoding=encoding, **kwargs)
    elif ext in ('.xlsx', '.xls'):
        df = pd.read_excel(path, **kwargs)
    elif ext == '.parquet':
        df = pd.read_parquet(path, **kwargs)
    elif ext == '.feather':
        df = pd.read_feather(path, **kwargs)
    elif ext == '.json':
        df = pd.read_json(path, **kwargs)
    elif ext == '.sas7bdat':
        df = _read_sas(path, **kwargs)
    elif ext == '.sav':
        df = _read_spss(path, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported: .dta, .csv, .xlsx, .parquet, .sas7bdat, .sav"
        )

    return df


def _read_stata(path, **kwargs):
    """Read .dta with variable labels preserved."""
    import pyreadstat

    try:
        df, meta = pyreadstat.read_dta(path, **kwargs)
        # Store variable labels
        if meta.column_names_to_labels:
            df.attrs['_labels'] = {
                k: v for k, v in meta.column_names_to_labels.items()
                if v  # skip empty labels
            }
        # Store value labels
        if meta.variable_value_labels:
            df.attrs['_value_labels'] = meta.variable_value_labels
        return df
    except ImportError:
        # Fallback to pandas (loses labels)
        import warnings
        warnings.warn(
            "pyreadstat not installed. Variable labels from .dta will be lost. "
            "Install: pip install pyreadstat",
            UserWarning,
        )
        return pd.read_stata(path, **kwargs)


def _read_sas(path, **kwargs):
    """Read SAS .sas7bdat with labels."""
    try:
        import pyreadstat
        df, meta = pyreadstat.read_sas7bdat(path, **kwargs)
        if meta.column_names_to_labels:
            df.attrs['_labels'] = {
                k: v for k, v in meta.column_names_to_labels.items() if v
            }
        return df
    except ImportError:
        return pd.read_sas(path, **kwargs)


def _read_spss(path, **kwargs):
    """Read SPSS .sav with labels."""
    try:
        import pyreadstat
        df, meta = pyreadstat.read_sav(path, **kwargs)
        if meta.column_names_to_labels:
            df.attrs['_labels'] = {
                k: v for k, v in meta.column_names_to_labels.items() if v
            }
        return df
    except ImportError:
        return pd.read_spss(path, **kwargs)
