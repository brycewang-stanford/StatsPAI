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
['variable', 'type', 'n', 'n_missing', 'label', 'value_labels']

Value labels (Stata's ``label define`` + ``label values``) live in
``df.attrs['_value_labels']`` as ``{column: {code: text}}``; the labels of
Stata's extended missing values ``.a`` ... ``.z`` are kept apart in
``df.attrs['_missing_labels']``.

>>> df['edu'] = [1, 2]
>>> sp.label_values(df, 'edu', {1: 'primary', 2: 'secondary'})
>>> sp.decode(df, 'edu')['edu'].tolist()
['primary', 'secondary']
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union, cast

import pandas as pd

from .._aliases import accepts_aliases
from ..exceptions import MethodIncompatibility

#: Suffix of the column holding a variable's ``.a`` ... ``.z`` codes; the
#: same one ``sp.read_data(extended_missing='column')`` writes.
_MISSING_CODE_SUFFIX = "__miss"
_MISSING_CODE_RE = re.compile(r"^\.[a-z]$")


@accepts_aliases(data="df")
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


#: ``attrs`` entries that map column name -> metadata for that column.
_PER_COLUMN_ATTRS = (
    "_labels",
    "_value_labels",
    "_missing_labels",
    "_formats",
    "_value_label_names",
    "_notes",
    "_characteristics",
)


@accepts_aliases(data="df")
def label_vars(
    df: pd.DataFrame,
    labels: Union[Dict[str, str], pd.DataFrame],
    rename: Optional[Dict[str, str]] = None,
) -> None:
    """
    Attach labels to multiple variables at once, or copy them from a frame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to label (modified in place).
    labels : dict or pd.DataFrame
        ``{column_name: label_string}``, or a labelled DataFrame to copy
        from.  Copying carries variable labels, value labels, the labels
        of extended missing values and display formats for every column
        the two frames share by name, and leaves labels ``df`` already has
        alone.  It is the way to get labels back after a pandas operation
        that drops them: ``merge``, ``pd.get_dummies``, ``pivot``, and
        ``pd.concat`` of frames whose labels differ all return a frame
        with empty ``attrs``.
    rename : dict, optional
        ``{old_name: new_name}``, only when copying: the columns of
        ``labels`` that go by another name in ``df``.  ``DataFrame.rename``
        keeps ``attrs`` but leaves the labels under the old names.

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

    A merge drops the labels; copy them back from the frame that had them:

    >>> merged = df.merge(pd.DataFrame({'edu': [12, 16], 'z': [0, 1]}), on='edu')
    >>> sp.get_label(merged, 'wage')
    'wage'
    >>> sp.label_vars(merged, df)
    >>> sp.get_label(merged, 'wage')
    'Monthly wage (CNY)'

    >>> short = df.rename(columns={'wage': 'w'})
    >>> sp.label_vars(short, df, rename={'wage': 'w'})
    >>> sp.get_label(short, 'w')
    'Monthly wage (CNY)'
    """
    if isinstance(labels, pd.DataFrame):
        _copy_labels(df, labels, rename or {})
        return
    if rename is not None:
        raise MethodIncompatibility("rename= applies only when labels is a DataFrame")
    for var, label in labels.items():
        label_var(df, var, label)


def _copy_labels(
    df: pd.DataFrame, source: pd.DataFrame, rename: Dict[str, str]
) -> None:
    unknown = [old for old in rename if old not in source.columns]
    if unknown:
        raise MethodIncompatibility(
            f"rename names columns not in the source frame: {unknown}"
        )
    for key in _PER_COLUMN_ATTRS:
        theirs = source.attrs.get(key)
        if not isinstance(theirs, dict) or not theirs:
            continue
        # a fresh dict: frames that came out of one another share attrs
        mine = dict(df.attrs.get(key) or {})
        for col, meta in theirs.items():
            target = rename.get(col, col)
            if target in df.columns and target not in mine:
                mine[target] = dict(meta) if isinstance(meta, dict) else meta
        if mine:
            df.attrs[key] = mine
    if "_data_label" in source.attrs and "_data_label" not in df.attrs:
        df.attrs["_data_label"] = source.attrs["_data_label"]


@accepts_aliases(data="df")
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


@accepts_aliases(data="df")
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


@accepts_aliases(data="df")
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
    ['variable', 'type', 'n', 'n_missing', 'label', 'value_labels']
    >>> tbl['label'].tolist()
    ['Monthly wage (CNY)', '']
    >>> sp.label_values(df, 'edu', {12: 'high school', 16: 'college'})
    >>> sp.describe(df)['value_labels'].tolist()
    ['', '12=high school, 16=college']
    """
    cols = columns or list(df.columns)
    labels = cast(Dict[str, str], df.attrs.get("_labels", {}))
    value_labels = df.attrs.get("_value_labels") or {}
    missing_labels = df.attrs.get("_missing_labels") or {}

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
                "value_labels": _value_label_summary(
                    {**(value_labels.get(col) or {}), **(missing_labels.get(col) or {})}
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=["variable", "type", "n", "n_missing", "label", "value_labels"],
    )


def _value_label_summary(mapping: Dict[Any, str], width: int = 60) -> str:
    """``'0=No, 1=Yes'``, cut at ``width`` characters with the count left out."""
    if not mapping:
        return ""
    parts = [f"{code}={text}" for code, text in mapping.items()]
    out = ", ".join(parts)
    if len(out) <= width:
        return out
    kept: List[str] = []
    for part in parts:
        if len(", ".join(kept + [part])) > width - 12:
            break
        kept.append(part)
    return ", ".join(kept) + f", ... ({len(parts)} codes)"


@accepts_aliases(data="df")
def label_values(
    df: pd.DataFrame,
    var: Union[str, Sequence[str]],
    labels: Optional[Dict[Any, str]],
) -> None:
    """
    Attach value labels to the codes of one or more variables.

    Equivalent to Stata's ``label define yn 0 "No" 1 "Yes"`` followed by
    ``label values female union yn``.  The data are not changed: the
    columns keep their numeric codes, and the labels are used by
    :func:`describe`, ``sp.tab``, :func:`decode`, ``sp.regtable(labels=)``
    and written to .dta / Parquet by ``sp.write_data``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to label (modified in place).
    var : str or list of str
        Column(s) the labels apply to.
    labels : dict or None
        ``{code: text}``.  Codes are integers; ``'.a'`` ... ``'.z'`` label
        Stata's extended missing values.  The mapping replaces any labels
        the variable already has; ``None`` (or ``{}``) removes them.

    Raises
    ------
    MethodIncompatibility
        (a ``ValueError``.)  A column that is not in ``df``, a non-numeric
        column, or a code
        that is neither an integer nor ``'.a'`` ... ``'.z'``.

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> df = pd.DataFrame({'female': [0, 1, 1], 'union': [1, 0, 1]})
    >>> sp.label_values(df, ['female', 'union'], {0: 'No', 1: 'Yes'})
    >>> df.attrs['_value_labels']['union']
    {0: 'No', 1: 'Yes'}
    >>> sp.decode(df)['female'].tolist()
    ['No', 'Yes', 'Yes']
    """
    names = [var] if isinstance(var, str) else list(var)
    regular: Dict[int, str] = {}
    missing: Dict[str, str] = {}
    for code, text in (labels or {}).items():
        if isinstance(code, str):
            if not _MISSING_CODE_RE.match(code):
                raise MethodIncompatibility(
                    f"value label code {code!r} is neither an integer nor "
                    f"one of '.a' ... '.z'"
                )
            missing[code] = str(text)
            continue
        try:
            whole = int(code) == code
        except (TypeError, ValueError):
            whole = False
        if not whole:
            raise MethodIncompatibility(
                f"value labels attach to integer codes only; got {code!r}"
            )
        regular[int(code)] = str(text)
    for name in names:
        if name not in df.columns:
            raise MethodIncompatibility(f"Column '{name}' not found in DataFrame")
        if not pd.api.types.is_numeric_dtype(df[name]):
            raise MethodIncompatibility(
                f"Column '{name}' is {df[name].dtype}, not numeric; value "
                f"labels describe numeric codes"
            )
    for key, mapping in (("_value_labels", regular), ("_missing_labels", missing)):
        # a fresh dict, so a frame that shares attrs with another is not touched
        store = dict(df.attrs.get(key) or {})
        for name in names:
            if mapping:
                store[name] = dict(mapping)
            else:
                store.pop(name, None)
        if store:
            df.attrs[key] = store
        else:
            df.attrs.pop(key, None)
    if not regular and not missing and df.attrs.get("_value_label_names"):
        # the set a file attached to these variables goes with its labels
        kept = {
            k: v for k, v in df.attrs["_value_label_names"].items() if k not in names
        }
        if kept:
            df.attrs["_value_label_names"] = kept
        else:
            df.attrs.pop("_value_label_names", None)


def _code_text(code: Any) -> str:
    """How an unlabelled code is shown: ``3`` for 3.0, else as is."""
    try:
        if float(code) == int(code):
            return str(int(code))
    except (TypeError, ValueError, OverflowError):
        pass
    return str(code)


def _decode_series(
    values: pd.Series,
    mapping: Dict[Any, str],
    missing_codes: Optional[pd.Series] = None,
    missing_mapping: Optional[Dict[str, str]] = None,
) -> pd.Series:
    """``values`` with each code replaced by its label, as an ordered categorical.

    The categories follow the order of the codes, not of the label texts, so
    a table or a set of dummies built from the result is laid out as one
    built from the codes would be.  A code without a label is shown as the
    number; two codes that share a text are told apart by ``text (code)``.
    """
    # categories: every labelled code (present or not) and every present code
    present = set(pd.unique(values.dropna()).tolist())
    order: List[Any] = sorted(set(mapping) | present, key=float)
    text_of: Dict[Any, str] = {}
    for code in order:
        key = int(code) if float(code) == int(code) else code
        text_of[code] = mapping.get(key, _code_text(code))
    counts: Dict[str, int] = {}
    for text in text_of.values():
        counts[text] = counts.get(text, 0) + 1
    for code, text in list(text_of.items()):
        if counts[text] > 1:
            text_of[code] = f"{text} ({_code_text(code)})"
    categories = list(dict.fromkeys(text_of.values()))
    out = values.map(lambda v: text_of.get(v) if pd.notna(v) else None).astype(object)
    if missing_codes is not None and missing_mapping:
        shown = missing_codes.map(lambda c: missing_mapping.get(c) if c else None)
        fill = out.isna() & shown.notna()
        out = out.where(~fill, shown)
        for text in missing_mapping.values():
            if text not in categories:
                categories.append(text)
    return pd.Series(
        pd.Categorical(out, categories=categories, ordered=True),
        index=values.index,
        name=values.name,
    )


@accepts_aliases(data="df")
def decode(
    df: pd.DataFrame,
    columns: Union[str, Sequence[str], None] = None,
    missing: bool = False,
) -> pd.DataFrame:
    """
    Replace value-labelled codes by their label texts.

    The counterpart of Stata's ``decode``.  A .dta file read by
    ``sp.read_data`` keeps its labelled variables as numeric codes (what
    Stata itself stores and estimates on); this gives the readable version
    for tables, plots and exports.

    Parameters
    ----------
    df : pd.DataFrame
        Data with value labels in ``df.attrs['_value_labels']``.
    columns : str or list of str, optional
        Columns to decode.  Default: every column that has value labels.
    missing : bool, default False
        Also show the labels of extended missing values (``.a`` "Refused")
        instead of leaving those rows missing.  Needs the ``<var>__miss``
        column that ``sp.read_data(extended_missing='column')`` adds.

    Returns
    -------
    pd.DataFrame
        A copy in which each decoded column is an ordered categorical whose
        categories follow the codes.  A code without a label is shown as
        the number.  Decoded columns no longer appear in
        ``attrs['_value_labels']``; variable labels are kept.

    Raises
    ------
    MethodIncompatibility
        (a ``ValueError``.)  A requested column is not in ``df`` or has no
        value labels.

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> df = pd.DataFrame({'edu': [3, 1, 2, 1], 'wage': [30., 10., 20., 12.]})
    >>> sp.label_values(df, 'edu', {1: 'primary', 2: 'secondary', 3: 'tertiary'})
    >>> out = sp.decode(df)
    >>> out['edu'].tolist()
    ['tertiary', 'primary', 'secondary', 'primary']
    >>> list(out['edu'].cat.categories)   # code order, not alphabetical
    ['primary', 'secondary', 'tertiary']
    >>> df['edu'].tolist()                # the input is not modified
    [3, 1, 2, 1]
    """
    value_labels = dict(df.attrs.get("_value_labels") or {})
    missing_labels = df.attrs.get("_missing_labels") or {}
    if columns is None:
        names = [c for c in df.columns if value_labels.get(c)]
    else:
        names = [columns] if isinstance(columns, str) else list(columns)
        for name in names:
            if name not in df.columns:
                raise MethodIncompatibility(f"Column '{name}' not found in DataFrame")
            if not value_labels.get(name):
                raise MethodIncompatibility(
                    f"Column '{name}' has no value labels; attach them with "
                    f"sp.label_values"
                )
    out = df.copy()
    for name in names:
        companion = f"{name}{_MISSING_CODE_SUFFIX}"
        use_missing = missing and companion in df.columns
        out[name] = _decode_series(
            df[name],
            value_labels[name],
            df[companion] if use_missing else None,
            missing_labels.get(name) if use_missing else None,
        )
        value_labels.pop(name, None)
    out.attrs = dict(df.attrs)
    if value_labels:
        out.attrs["_value_labels"] = value_labels
    else:
        out.attrs.pop("_value_labels", None)
    return out


# A factor level inside a coefficient name: ``C(edu)[T.2]``, ``C(edu)[2]``,
# ``edu[T.2]`` (formula terms) and ``edu::2`` (fixest ``i()``).
_FACTOR_TERM_RE = re.compile(
    r"^(?:C\(\s*(?P<cvar>[^,)\s]+)[^)]*\)|(?P<var>[^\[\]]+))"
    r"\[(?:T\.)?(?P<level>[^\]]+)\]$"
)
_FIXEST_TERM_RE = re.compile(r"^(?P<var>[^:\[\]]+)::(?P<level>[^:]+)$")
_INTERACTION_SPLIT_RE = re.compile(r"(?<!:):(?!:)")


def _level_text(level: str, mapping: Dict[Any, str]) -> Optional[str]:
    try:
        number = float(level)
        if number == int(number) and int(number) in mapping:
            return str(mapping[int(number)])
    except (TypeError, ValueError, OverflowError):
        pass
    return str(mapping[level]) if level in mapping else None


def term_labels(names: Iterable[Any], data: pd.DataFrame) -> Dict[str, str]:
    """Readable labels for coefficient names, from a frame's labels.

    ``wage`` becomes its variable label; ``C(edu)[T.2]`` and ``edu::2``
    become ``"<label of edu>: <label of code 2>"``; the parts of an
    interaction (``a:b``) are labelled one by one and joined by ``" × "``.
    Names for which the frame has nothing to say are left out.
    """
    var_labels = data.attrs.get("_labels") or {}
    value_labels = data.attrs.get("_value_labels") or {}

    def one(part: str) -> str:
        match = _FIXEST_TERM_RE.match(part) or _FACTOR_TERM_RE.match(part)
        if match is None:
            return str(var_labels.get(part, part))
        groups = match.groupdict()
        var = groups.get("cvar") or groups.get("var") or ""
        level = groups["level"]
        text = _level_text(level, value_labels.get(var) or {})
        if text is None and var not in var_labels:
            return part
        return f"{var_labels.get(var, var)}: {text if text is not None else level}"

    out: Dict[str, str] = {}
    for raw in names:
        name = str(raw)
        label = " \u00d7 ".join(one(p) for p in _INTERACTION_SPLIT_RE.split(name))
        if label != name:
            out[name] = label
    return out
