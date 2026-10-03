"""
Smart data I/O with variable label preservation.

Reads Stata .dta, SAS .sas7bdat, SPSS .sav, CSV, Excel, and Parquet
files, automatically preserving variable labels when available.

For Stata .dta files, variable labels are stored in ``df.attrs['_labels']``
and are used by StatsPAI output functions (modelsummary, outreg2, etc.).
Value labels go to ``df.attrs['_value_labels']`` and the dataset label to
``df.attrs['_data_label']``.  :func:`write_data` writes all three back, so
``read_data`` -> edit -> ``write_data`` keeps a .dta file's labels.

References
----------
This addresses the #1 pain point for Stata → Python migration:
pandas' ``read_stata()`` loses variable labels silently.
"""

import re
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from ..exceptions import MethodIncompatibility, MissingDependencyError

# Stata's limit on a variable label and on the dataset label (``help limits``).
_STATA_LABEL_MAX = 80


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
        _restore_value_label_keys(df)
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
    if getattr(meta, "file_label", None):
        df.attrs["_data_label"] = meta.file_label
    formats = _informative_formats(getattr(meta, "original_variable_types", None) or {})
    if formats:
        df.attrs["_formats"] = formats
    return df


def _read_stata_pandas(path: Any, **kwargs: Any) -> pd.DataFrame:
    """pandas fallback for .dta that still carries labels into ``attrs``.

    ``path`` may be a file path or a binary buffer.
    """
    kwargs.setdefault("convert_categoricals", False)
    with pd.read_stata(path, iterator=True, **kwargs) as reader:
        df = reader.read()
        attrs = _stata_reader_attrs(reader, list(df.columns))
    df.attrs.update(attrs)
    return df


def stata_label_attrs(path: Any, columns: Optional[list] = None) -> Dict[str, Any]:
    """Label metadata of a .dta file, without loading its rows.

    Returns the ``attrs`` entries :func:`read_data` would set
    (``_labels`` / ``_value_labels`` / ``_data_label`` / ``_formats``, each
    only when present), restricted to ``columns`` if given.  Used by readers that
    stream the rows in chunks and so cannot go through ``read_data``.
    """
    with pd.read_stata(
        path, iterator=True, columns=columns, convert_categoricals=False
    ) as reader:
        if columns is None:
            columns = list(reader.variable_labels())
        return _stata_reader_attrs(reader, list(columns))


def _per_variable(reader: Any, attribute: str, columns: list) -> Dict[Any, Any]:
    """A private per-variable list of a pandas ``StataReader``, keyed by name.

    pandas keeps ``_lbllist`` / ``_fmtlist`` in file order for every variable
    until rows are read; a read with ``columns=`` then narrows them to the
    selected columns, in the order requested, while ``_varlist`` stays whole.
    Both layouts are handled; anything else (the attribute is gone in a
    future pandas) yields ``{}`` and the caller degrades.
    """
    values = getattr(reader, attribute, None) or []
    all_names = getattr(reader, "_varlist", None) or []
    if len(values) == len(all_names):
        return dict(zip(all_names, values))
    if len(values) == len(columns):
        return dict(zip(columns, values))
    return {}


def _stata_reader_attrs(reader: Any, columns: list) -> Dict[str, Any]:
    """``attrs`` for the variables in ``columns``, read from an open reader."""
    var_labels = reader.variable_labels()
    label_sets = reader.value_labels()
    # value_labels() is keyed by label-set name; map sets to variables.
    set_of = _per_variable(reader, "_lbllist", columns)
    if not set_of:
        # Fall back to the common Stata convention of naming a label set
        # after its variable.
        set_of = {c: c for c in columns if c in label_sets}
    attrs: Dict[str, Any] = {}
    labels = {c: var_labels[c] for c in columns if var_labels.get(c)}
    if labels:
        attrs["_labels"] = labels
    value_labels = {}
    for col in columns:
        lbl = set_of.get(col)
        if lbl and lbl in label_sets:
            value_labels[col] = {
                (k.item() if hasattr(k, "item") else k): v
                for k, v in label_sets[lbl].items()
            }
    if value_labels:
        attrs["_value_labels"] = value_labels
    data_label = getattr(reader, "data_label", "")
    if data_label:
        attrs["_data_label"] = data_label
    formats = _informative_formats(
        {
            c: f
            for c, f in _per_variable(reader, "_fmtlist", columns).items()
            if c in columns
        }
    )
    if formats:
        attrs["_formats"] = formats
    return attrs


#: Formats that say nothing beyond the storage type: Stata's defaults for
#: numerics (``%9.0g``, ``%8.0g``, ``%10.0g``, ``%12.0g``) and for strings
#: (``%9s``, ``%18s``).  Anything else is worth keeping: ``%td`` / ``%tm`` /
#: ``%tq`` name the time unit of a date, ``%12.2fc`` marks money.
_DEFAULT_FORMAT_RE = re.compile(r"^%-?\d+(\.0g|s)$")


def _informative_formats(formats: Dict[str, Any]) -> Dict[str, str]:
    """Keep only display formats that carry information (see above)."""
    return {
        str(col): str(fmt)
        for col, fmt in formats.items()
        if isinstance(fmt, str) and fmt and not _DEFAULT_FORMAT_RE.match(fmt)
    }


def _restore_value_label_keys(df: pd.DataFrame) -> None:
    """Turn value-label codes back into integers after a Parquet round trip.

    Parquet stores ``df.attrs`` as JSON, whose object keys are strings, so
    ``{0: 'male'}`` comes back as ``{'0': 'male'}``.
    """
    stored = df.attrs.get("_value_labels")
    if not isinstance(stored, dict):
        return
    restored = {}
    for col, mapping in stored.items():
        if not isinstance(mapping, dict):
            return
        fixed = {}
        for code, text in mapping.items():
            try:
                fixed[int(code)] = text
            except (TypeError, ValueError):
                return  # not written by write_data; leave untouched
        restored[col] = fixed
    df.attrs["_value_labels"] = restored


def write_data(
    data: pd.DataFrame,
    path: Union[str, Path],
    *,
    labels: Optional[Dict[str, str]] = None,
    value_labels: Optional[Dict[str, Dict[Any, str]]] = None,
    data_label: Optional[str] = None,
    **kwargs: Any,
) -> Path:
    """
    Write data to a file, keeping variable and value labels.

    The counterpart of :func:`read_data`.  For Stata .dta the variable
    labels (``data.attrs['_labels']``), value labels
    (``data.attrs['_value_labels']``) and dataset label
    (``data.attrs['_data_label']``) are written into the file, so Stata's
    ``describe`` / ``label list`` and :func:`read_data` see them again.
    ``pandas.DataFrame.to_stata`` writes none of these unless each is
    passed by hand.

    Parameters
    ----------
    data : pd.DataFrame
        Data to write.  The row index is not written.
    path : str or Path
        Output file.  The extension picks the format: .dta, .csv, .tsv,
        .xlsx, .parquet, .feather, .json.
    labels : dict, optional
        ``{column: label}``.  Added to (and overriding) the labels already
        in ``data.attrs['_labels']``.
    value_labels : dict, optional
        ``{column: {code: text}}``.  Added to (and overriding) those in
        ``data.attrs['_value_labels']``.
    data_label : str, optional
        Dataset label (Stata's ``label data``).  Default: the one in
        ``data.attrs['_data_label']``, if any.
    **kwargs
        Passed to the underlying pandas writer.  For .dta, ``version``
        defaults to 118 (Stata 14 and later), the oldest format that
        stores labels as UTF-8; pass ``version=114`` for older Stata, in
        which case labels must be Latin-1.

    Returns
    -------
    pathlib.Path
        The file written.

    Raises
    ------
    MethodIncompatibility
        (a ``ValueError``.)  Unsupported extension; ``labels`` / ``value_labels`` naming a
        column that is not in ``data``; a variable or dataset label longer
        than Stata's 80 characters; value labels on a non-numeric column
        or on non-integer codes.

    Warns
    -----
    UserWarning
        When ``data`` carries labels and the format cannot store them
        (.csv, .tsv, .xlsx, .feather, .json).  Parquet stores them.

    Notes
    -----
    Labels for columns that are not in ``data`` (dropped since the labels
    were attached) are skipped.  Stata notes, display formats and
    characteristics are not part of ``data.attrs`` and are not written.

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> df = pd.DataFrame({'wage': [10.0, 20.0], 'female': [0, 1]})
    >>> sp.label_var(df, 'wage', 'Monthly wage (CNY)')
    >>> out = sp.write_data(  # doctest: +SKIP
    ...     df, 'survey.dta',
    ...     value_labels={'female': {0: 'male', 1: 'female'}},
    ... )
    >>> sp.get_label(sp.read_data('survey.dta'), 'wage')  # doctest: +SKIP
    'Monthly wage (CNY)'
    """
    p = Path(path)
    ext = p.suffix.lower()

    var_lab = dict(data.attrs.get("_labels") or {})
    var_lab.update(labels or {})
    val_lab = dict(data.attrs.get("_value_labels") or {})
    val_lab.update(value_labels or {})
    for name, given in (("labels", labels), ("value_labels", value_labels)):
        unknown = [c for c in (given or {}) if c not in data.columns]
        if unknown:
            raise MethodIncompatibility(
                f"{name} names columns not in the data: {unknown}"
            )
    var_lab = {c: v for c, v in var_lab.items() if c in data.columns and v}
    val_lab = {c: v for c, v in val_lab.items() if c in data.columns and v}
    if data_label is None:
        data_label = data.attrs.get("_data_label") or None

    if ext == ".dta":
        _write_stata(p, data, var_lab, val_lab, data_label, **kwargs)
        return p

    if ext == ".parquet":
        out = data
        if labels or value_labels or data_label:
            out = data.copy(deep=False)
            if var_lab:
                out.attrs["_labels"] = var_lab
            if val_lab:
                out.attrs["_value_labels"] = val_lab
            if data_label:
                out.attrs["_data_label"] = data_label
        kwargs.setdefault("index", False)
        out.to_parquet(p, **kwargs)
        return p

    if ext in (".csv", ".tsv"):
        kwargs.setdefault("index", False)
        if ext == ".tsv":
            kwargs.setdefault("sep", "\t")
        data.to_csv(p, **kwargs)
    elif ext == ".xlsx":
        kwargs.setdefault("index", False)
        data.to_excel(p, **kwargs)
    elif ext == ".feather":
        data.to_feather(p, **kwargs)
    elif ext == ".json":
        data.to_json(p, **kwargs)
    else:
        raise MethodIncompatibility(
            f"Unsupported file format: '{ext}'. "
            f"Supported: .dta, .csv, .tsv, .xlsx, .parquet, .feather, .json"
        )
    if var_lab or val_lab or data_label:
        warnings.warn(
            f"'{ext}' files cannot store variable or value labels; "
            f"{p.name} was written without them. Use .dta or .parquet to "
            f"keep them.",
            UserWarning,
            stacklevel=2,
        )
    return p


def _write_stata(
    p: Path,
    df: pd.DataFrame,
    var_lab: Dict[str, str],
    val_lab: Dict[str, Dict[Any, str]],
    data_label: Optional[str],
    **kwargs: Any,
) -> None:
    """Write .dta with labels, checking Stata's limits by variable name."""
    too_long = [c for c, v in var_lab.items() if len(str(v)) > _STATA_LABEL_MAX]
    if too_long:
        raise MethodIncompatibility(
            f"Stata variable labels hold at most {_STATA_LABEL_MAX} "
            f"characters; too long for: {too_long}. Shorten them, e.g. "
            f"with sp.label_var."
        )
    if data_label is not None and len(data_label) > _STATA_LABEL_MAX:
        raise MethodIncompatibility(
            f"Stata dataset labels hold at most {_STATA_LABEL_MAX} "
            f"characters; got {len(data_label)}."
        )
    kwargs.setdefault("version", 118)
    kwargs.setdefault("write_index", False)
    if val_lab:
        # pandas < 1.4 has no value_labels argument; only pass it when needed.
        clean: Dict[str, Dict[int, str]] = {}
        for c, m in val_lab.items():
            bad = [k for k in m if int(k) != k]
            if bad:
                raise MethodIncompatibility(
                    f"Stata value labels attach to integer codes only; "
                    f"'{c}' has non-integer codes {bad}."
                )
            clean[c] = {int(k): str(v) for k, v in m.items()}
        kwargs["value_labels"] = clean
    try:
        df.to_stata(
            p,
            variable_labels={c: str(v) for c, v in var_lab.items()} or None,
            data_label=data_label,
            **kwargs,
        )
    except TypeError as e:
        if "value_labels" not in str(e):
            raise
        raise MissingDependencyError(
            "Writing value labels to .dta needs pandas >= 1.4; "
            f"found pandas {pd.__version__}.",
            recovery_hint='pip install -U "pandas>=1.4"',
        ) from e


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
