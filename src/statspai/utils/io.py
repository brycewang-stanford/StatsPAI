"""
Smart data I/O with variable label preservation.

Reads Stata .dta, SAS .sas7bdat, SPSS .sav, CSV, Excel, and Parquet
files, automatically preserving variable labels when available.

For Stata .dta files, variable labels are stored in ``df.attrs['_labels']``
and are used by StatsPAI output functions (modelsummary, outreg2, etc.).
Value labels go to ``df.attrs['_value_labels']`` and the dataset label to
``df.attrs['_data_label']``.  :func:`write_data` writes all three back, so
``read_data`` -> edit -> ``write_data`` keeps a .dta file's labels.
Labels on Stata's extended missing values (``.a`` ... ``.z``) are kept apart
in ``df.attrs['_missing_labels']``, keyed ``'.a'``; they are not codes a row
can hold once the missing values are ``NaN``.

References
----------
This addresses the #1 pain point for Stata → Python migration:
pandas' ``read_stata()`` loses variable labels silently.
"""

import re
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility, MissingDependencyError

# Stata's limit on a variable label and on the dataset label (``help limits``).
_STATA_LABEL_MAX = 80

# In a .dta value-label table the missing values sit above the largest
# ``long``: ``.`` is 2147483621 and ``.a`` ... ``.z`` follow (``help dta``).
_STATA_MISSING_CODE = 2147483621
_STATA_MISSING_NAMES = ("",) + tuple("abcdefghijklmnopqrstuvwxyz")
_EXTENDED_MISSING_MODES = ("nan", "column")
#: Suffix of the column that holds a variable's ``.a`` ... ``.z`` codes when
#: a file is read with ``extended_missing='column'``.
MISSING_CODE_SUFFIX = "__miss"


def read_data(
    path: str,
    encoding: Optional[str] = None,
    extended_missing: str = "nan",
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
    extended_missing : {'nan', 'column'}, default 'nan'
        What to do with Stata's extended missing values ``.a`` ... ``.z``
        in a .dta file (survey codes such as "refused" or "don't know").
        ``'nan'`` reads them as ``NaN``, like ``.``; which of them a row
        held is then gone.  ``'column'`` also reads them as ``NaN`` and
        adds, for every variable that has any, a column ``<var>__miss``
        holding ``'.a'`` ... ``'.z'`` on those rows and missing elsewhere.
        Either way the labels attached to these codes are in
        ``df.attrs['_missing_labels']``.
    **kwargs
        Passed to the underlying pandas reader.

    Returns
    -------
    pd.DataFrame
        With variable labels in ``df.attrs['_labels']`` if available.

    Warns
    -----
    UserWarning
        When a .dta file labels an extended missing value and
        ``extended_missing='nan'``: the rows that held it are no longer
        told apart from ``.``.

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
    if extended_missing not in _EXTENDED_MISSING_MODES:
        raise MethodIncompatibility(
            f"extended_missing must be one of {_EXTENDED_MISSING_MODES}; "
            f"got {extended_missing!r}"
        )

    if ext == ".dta":
        df = _read_stata(path, extended_missing=extended_missing, **kwargs)
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


def _read_stata(
    path: str, extended_missing: str = "nan", **kwargs: Any
) -> pd.DataFrame:
    """Read .dta with variable and value labels preserved.

    ``pyreadstat`` (``pip install statspai[io]``) is preferred.  Without it
    the pandas reader is used; it keeps variable labels and value labels as
    well, and value-labelled columns keep their numeric codes (as with
    pyreadstat) rather than being converted to categoricals.  Stata dates
    (``%td``, ``%tc``, ``%tw``, ``%tm``, ``%tq``, ``%th``, ``%ty``) are
    ``datetime64`` on both paths, and numeric columns are ``int64`` or
    ``float64`` on both, whatever storage type the file uses.
    """
    try:
        import pyreadstat
    except ImportError:
        return _read_stata_pandas(
            path, extended_missing=extended_missing, _stacklevel=5, **kwargs
        )

    convert_dates = not kwargs.get("disable_datetime_conversion", False)
    if convert_dates:
        kwargs.setdefault("dates_as_pandas_datetime", True)
    keep_codes = extended_missing == "column"
    df, meta = pyreadstat.read_dta(path, **kwargs)
    # pyreadstat returns a numeric variable that holds .a ... .z as an
    # ``object`` column of numbers and NaN.
    storage = getattr(meta, "readstat_variable_types", None) or {}
    for col in df.columns:
        if df[col].dtype == object and storage.get(col, "string") != "string":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    holders = [c for c in df.columns if storage.get(c, "string") != "string"]
    if keep_codes and holders:
        # A second pass for the letters only.  With user_missing pyreadstat
        # (1.3.4) returns garbage for a plain '.' in an integer variable, so
        # the values above come from the default read.
        lettered, _ = pyreadstat.read_dta(
            path, **{**kwargs, "usecols": holders, "user_missing": True}
        )
        for col in holders:
            if lettered[col].dtype != object:
                continue
            _insert_missing_codes(
                df,
                col,
                lettered[col].map(lambda v: "." + v if isinstance(v, str) else None),
            )
    if convert_dates:
        _convert_stata_period_dates(
            df, getattr(meta, "original_variable_types", None) or {}
        )
    if meta.column_names_to_labels:
        df.attrs["_labels"] = {
            k: v for k, v in meta.column_names_to_labels.items() if v
        }
    value_labels, missing_labels = _split_value_labels(meta.variable_value_labels or {})
    if value_labels:
        df.attrs["_value_labels"] = value_labels
    if missing_labels:
        df.attrs["_missing_labels"] = missing_labels
    if getattr(meta, "file_label", None):
        df.attrs["_data_label"] = meta.file_label
    formats = _informative_formats(getattr(meta, "original_variable_types", None) or {})
    if formats:
        df.attrs["_formats"] = formats
    if not keep_codes:
        _warn_missing_labels(path, missing_labels)
    return df


def _missing_name(code: Any) -> Optional[str]:
    """``'.a'`` for the key of an extended missing value, else ``None``.

    The key is the table's integer (pandas), the bare letter (pyreadstat),
    or already ``'.a'`` (a label handed to :func:`write_data`).
    """
    if isinstance(code, str):
        letter = code[1:] if code.startswith(".") else code
        if letter in _STATA_MISSING_NAMES:
            return "." + letter
        return None
    try:
        offset = int(code) - _STATA_MISSING_CODE
    except (TypeError, ValueError, OverflowError):
        return None
    if 0 <= offset < len(_STATA_MISSING_NAMES) and int(code) == code:
        return "." + _STATA_MISSING_NAMES[offset]
    return None


def _split_value_labels(
    by_variable: Dict[Any, Dict[Any, str]],
) -> "tuple[Dict[Any, Dict[Any, str]], Dict[Any, Dict[str, str]]]":
    """Separate the labels of ``.a`` ... ``.z`` from those of real codes."""
    regular: Dict[Any, Dict[Any, str]] = {}
    missing: Dict[Any, Dict[str, str]] = {}
    for col, mapping in by_variable.items():
        codes: Dict[Any, str] = {}
        gaps: Dict[str, str] = {}
        for code, text in mapping.items():
            name = _missing_name(code)
            if name is None:
                codes[code.item() if hasattr(code, "item") else code] = text
            else:
                gaps[name] = text
        if codes:
            regular[col] = codes
        if gaps:
            missing[col] = gaps
    return regular, missing


def _warn_missing_labels(
    path: Any, missing_labels: Dict[Any, Dict[str, str]], stacklevel: int = 4
) -> None:
    if not missing_labels:
        return
    shown = "; ".join(
        f"{col}: " + ", ".join(f"{k} {v!r}" for k, v in m.items())
        for col, m in list(missing_labels.items())[:5]
    )
    more = len(missing_labels) - 5
    name = Path(path).name if isinstance(path, (str, Path)) else "the file"
    warnings.warn(
        f"{name} labels extended missing values ({shown}"
        + (f"; and {more} more variables" if more > 0 else "")
        + "). They are read as NaN, so a row that held .a can no longer be "
        "told apart from one that held '.'. The label texts are in "
        "df.attrs['_missing_labels']; read with extended_missing='column' "
        f"to keep each row's code in a '<var>{MISSING_CODE_SUFFIX}' column.",
        UserWarning,
        stacklevel=stacklevel,
    )


def _insert_missing_codes(df: pd.DataFrame, col: Any, codes: pd.Series) -> None:
    """Put ``codes`` (``'.a'`` ... ``'.z'`` or ``None``) right after ``col``.

    In place, as ``<col>__miss``; nothing is added when no row has a code.
    """
    if not codes.notna().any():
        return
    name = f"{col}{MISSING_CODE_SUFFIX}"
    if name in df.columns:
        raise MethodIncompatibility(
            f"extended_missing='column' needs the column name '{name}', "
            f"which the file already uses."
        )
    values = codes.where(codes.notna(), None).astype(object)
    values.index = df.index
    df.insert(df.columns.get_loc(col) + 1, name, values)


#: Stata date formats whose stored value is a count of periods since 1960
#: (the year itself for ``%ty``).  pyreadstat converts ``%td`` and ``%tc``
#: and returns these as the raw count; pandas converts them all.
_STATA_PERIOD_UNITS = ("tw", "tm", "tq", "th", "ty")


def _stata_date_unit(fmt: Any) -> Optional[str]:
    """``'td'``, ``'tm'``, ... for a Stata date display format, else ``None``.

    Follows the pandas reader: the format must start with the unit, with or
    without the leading ``%`` (``%tm``, ``%tmCCYY!mNN``), and ``%d`` is the
    old spelling of ``%td``.
    """
    if not isinstance(fmt, str):
        return None
    body = fmt[1:] if fmt.startswith("%") else fmt
    for unit in ("tc", "td", "tw", "tm", "tq", "th", "ty"):
        if body.startswith(unit):
            return unit
    if body.startswith("d"):
        return "td"
    return None


def _convert_stata_period_dates(df: pd.DataFrame, formats: Dict[str, Any]) -> None:
    """Turn ``%tw`` / ``%tm`` / ``%tq`` / ``%th`` / ``%ty`` counts into dates.

    In place, to the first day of the period, as ``pd.read_stata`` does, so
    a monthly date is the same value whether or not pyreadstat is installed.
    A column whose dates fall outside the ``datetime64`` range keeps its
    counts and warns.
    """
    for col, fmt in formats.items():
        unit = _stata_date_unit(fmt)
        if unit not in _STATA_PERIOD_UNITS or col not in df.columns:
            continue
        values = df[col]
        if not pd.api.types.is_numeric_dtype(values):
            continue
        missing = values.isna()
        # a placeholder count for missing rows, masked back to NaT below
        n = values.fillna(1960 if unit == "ty" else 0).astype("int64")
        days = None
        if unit == "ty":
            year, month = n, 1
        elif unit == "th":
            year, month = 1960 + n // 2, (n % 2) * 6 + 1
        elif unit == "tq":
            year, month = 1960 + n // 4, (n % 4) * 3 + 1
        elif unit == "tm":
            year, month = 1960 + n // 12, n % 12 + 1
        else:  # tw: 52 weeks to a year, week 52 absorbs the extra days
            year, month, days = 1960 + n // 52, 1, (n % 52) * 7
        try:
            dates = pd.to_datetime({"year": year, "month": month, "day": 1})
            if days is not None:
                dates = dates + pd.to_timedelta(days, unit="D")
        except (ValueError, OverflowError) as exc:
            warnings.warn(
                f"'{col}' has Stata format {fmt} but its dates could not be "
                f"converted ({exc}); the column keeps Stata's period counts.",
                UserWarning,
                stacklevel=4,
            )
            continue
        df[col] = dates.where(~missing)


def _read_stata_pandas(
    path: Any, extended_missing: str = "nan", _stacklevel: int = 3, **kwargs: Any
) -> pd.DataFrame:
    """pandas fallback for .dta that still carries labels into ``attrs``.

    ``path`` may be a file path or a binary buffer.
    """
    kwargs.setdefault("convert_categoricals", False)
    keep_codes = extended_missing == "column"
    if keep_codes:
        kwargs["convert_missing"] = True
    with pd.read_stata(path, iterator=True, **kwargs) as reader:
        df = reader.read()
        attrs = _stata_reader_attrs(reader, list(df.columns))
    if keep_codes:
        # pandas hands back every missing value as a StataMissingValue, in an
        # ``object`` column; '.' itself needs no code.
        from pandas.io.stata import StataMissingValue

        def code_of(v: Any) -> Optional[str]:
            if isinstance(v, StataMissingValue) and v.string != ".":
                return str(v.string)
            return None

        for col in list(df.columns):
            values = df[col]
            if values.dtype != object:
                continue
            marker = values.map(lambda v: isinstance(v, StataMissingValue))
            if not marker.any():
                continue
            df[col] = pd.to_numeric(values.where(~marker), errors="coerce").astype(
                "float64"
            )
            _insert_missing_codes(df, col, values.map(code_of))
    else:
        _warn_missing_labels(path, attrs.get("_missing_labels") or {}, _stacklevel)
    df.attrs.update(attrs)
    widen_stata_numerics(df)
    # pandas returns a %ty column with a missing value as ``object``
    # (Timestamps and NaT); make it datetime64 like every other date.
    for col, fmt in (attrs.get("_formats") or {}).items():
        if _stata_date_unit(fmt) and df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError, OverflowError):
                # dates outside the datetime64 range stay as pandas gave them
                continue
    return df


def widen_stata_numerics(df: pd.DataFrame) -> None:
    """Widen Stata's storage types to ``int64`` / ``float64``, in place.

    pandas reads a ``byte`` as ``int8``, an ``int`` as ``int16``, a ``long``
    as ``int32`` and a ``float`` as ``float32``.  Stata itself computes in
    double precision, so the storage type never limits a result there; in
    numpy it does, without a word: ``age ** 2`` on an ``int8`` column wraps
    past 127.  Widening also matches what pyreadstat returns.
    """
    for col in df.columns:
        dtype = df[col].dtype
        if not isinstance(dtype, np.dtype):
            continue
        if dtype.kind in "iu" and dtype.itemsize < 8:
            df[col] = df[col].astype("int64")
        elif dtype.kind == "f" and dtype.itemsize < 8:
            df[col] = df[col].astype("float64")


def stata_label_attrs(path: Any, columns: Optional[list] = None) -> Dict[str, Any]:
    """Label metadata of a .dta file, without loading its rows.

    Returns the ``attrs`` entries :func:`read_data` would set
    (``_labels`` / ``_value_labels`` / ``_missing_labels`` / ``_data_label`` /
    ``_formats``, each
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
            value_labels[col] = dict(label_sets[lbl])
    value_labels, missing_labels = _split_value_labels(value_labels)
    if value_labels:
        attrs["_value_labels"] = value_labels
    if missing_labels:
        attrs["_missing_labels"] = missing_labels
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
        ``data.attrs['_value_labels']``.  A code may be ``'.a'`` ... ``'.z'``
        to label an extended missing value.
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
    were attached) are skipped.  Labels of extended missing values
    (``data.attrs['_missing_labels']``) are written into the file's label
    tables, but every missing value in the data is written as ``.``: a
    ``<var>__miss`` column from ``read_data(extended_missing='column')`` is
    written as the string column it is, not folded back into ``<var>``.
    Stata notes and characteristics are not part of ``data.attrs`` and are
    not written.  Of the display formats in
    ``data.attrs['_formats']`` only the unit of a date is used: a
    ``datetime64`` column read from a ``%td`` / ``%tm`` / ``%tq`` variable is
    written back with that unit unless ``convert_dates=`` says otherwise.
    Each numeric column is stored in the smallest Stata type that holds its
    values exactly (as Stata's ``compress`` does); no value is rounded.

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
    val_lab = {c: dict(m) for c, m in (data.attrs.get("_value_labels") or {}).items()}
    for c, m in (data.attrs.get("_missing_labels") or {}).items():
        val_lab.setdefault(c, {}).update(m)
    for c, m in (value_labels or {}).items():
        # as documented, a column passed here replaces what attrs had for it
        val_lab[c] = dict(m)
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
            regular, gaps = _split_value_labels(val_lab)
            if var_lab:
                out.attrs["_labels"] = var_lab
            if regular:
                out.attrs["_value_labels"] = regular
            if gaps:
                out.attrs["_missing_labels"] = gaps
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
    df = _compress_for_stata(df)
    if "convert_dates" not in kwargs:
        # A date read from a %td / %tm / %tq column goes back with that unit;
        # pandas would otherwise write every datetime column as %tc.
        units = {
            c: _stata_date_unit(f)
            for c, f in (df.attrs.get("_formats") or {}).items()
            if c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c])
        }
        units = {c: u for c, u in units.items() if u}
        if units:
            kwargs["convert_dates"] = units
    if val_lab:
        # pandas < 1.4 has no value_labels argument; only pass it when needed.
        clean: Dict[str, Dict[int, str]] = {}
        for c, m in val_lab.items():
            codes: Dict[int, str] = {}
            bad = []
            for k, v in m.items():
                name = _missing_name(k) if isinstance(k, str) else None
                if name is not None:
                    # back to the table's own code for .a ... .z
                    k = _STATA_MISSING_CODE + _STATA_MISSING_NAMES.index(name[1:])
                try:
                    whole = int(k) == k
                except (TypeError, ValueError):
                    whole = False
                if not whole:
                    bad.append(k)
                else:
                    codes[int(k)] = str(v)
            if bad:
                raise MethodIncompatibility(
                    f"Stata value labels attach to integer codes and to "
                    f"'.a' ... '.z' only; '{c}' has {bad}."
                )
            clean[c] = codes
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


#: Largest non-missing value of each Stata integer type (``help data types``);
#: the codes above it are the missing values ``.``, ``.a`` ... ``.z``.
_STATA_INT_RANGES = (
    ("int8", -127, 100),
    ("int16", -32767, 32740),
    ("int32", -2147483647, 2147483620),
)


def _compress_for_stata(df: pd.DataFrame) -> pd.DataFrame:
    """Give each numeric column the smallest Stata type that holds it exactly.

    What Stata's ``compress`` does for integers, plus ``float`` for a
    ``float64`` column whose every value is a float32 (one read from a Stata
    ``float``).  Nothing is rounded.  :func:`read_data` widens every column
    on the way in, so without this a file read and written back would come
    out several times its size.  Returns a shallow copy when anything
    changed; the caller's frame is not modified.
    """
    narrowed: Dict[Any, Any] = {}
    for col in df.columns:
        values = df[col]
        dtype = values.dtype
        if not isinstance(dtype, np.dtype) or len(values) == 0:
            continue
        if dtype.kind in "iu":
            lo, hi = values.min(), values.max()
            for name, low, high in _STATA_INT_RANGES:
                if low <= lo and hi <= high:
                    if np.dtype(name).itemsize < dtype.itemsize:
                        narrowed[col] = values.astype(name)
                    break
        elif dtype == np.float64:
            arr = values.to_numpy()
            with np.errstate(over="ignore"):
                as_float = arr.astype("float32")
            if np.array_equal(as_float.astype("float64"), arr, equal_nan=True):
                narrowed[col] = values.astype("float32")
    if not narrowed:
        return df
    out = df.copy(deep=False)
    for col, values in narrowed.items():
        out[col] = values
    out.attrs = df.attrs
    return out


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
    """Read SPSS .sav with variable labels and value labels.

    Value labels go to ``attrs['_value_labels']`` for numeric variables whose
    labelled codes are whole numbers, the kind :func:`write_data` can store
    in a .dta and ``sp.decode`` can show.  SPSS also allows labels on string
    values and on fractions; those are left out.
    """
    try:
        import pyreadstat
    except ImportError:
        return pd.read_spss(path, **kwargs)

    df, meta = pyreadstat.read_sav(path, **kwargs)
    if meta.column_names_to_labels:
        df.attrs["_labels"] = {
            k: v for k, v in meta.column_names_to_labels.items() if v
        }
    value_labels: Dict[Any, Dict[int, str]] = {}
    for col, mapping in (meta.variable_value_labels or {}).items():
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        codes: Dict[int, str] = {}
        for code, text in mapping.items():
            if isinstance(code, str) or code != code or int(code) != code:
                codes = {}
                break
            codes[int(code)] = str(text)
        if codes:
            value_labels[col] = codes
    if value_labels:
        df.attrs["_value_labels"] = value_labels
    return df
