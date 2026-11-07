"""Package-level fail-loud input-validation primitives.

A recurring reliability defect across StatsPAI estimators is the pattern::

    df = data[[col_a, col_b, *covs]].dropna()
    ... linear algebra on df ...

which (a) surfaces a typo'd column name as a cryptic
``KeyError: "['x'] not in index"`` rather than a named, actionable error,
and (b) silently returns a NaN/garbage estimate when the frame is empty,
all-NaN, or has fewer complete rows than parameters — the "吞异常返回 NaN"
anti-pattern design principle 7 forbids.

These two helpers centralise the guard (CLAUDE.md §4: shared primitive, not
re-implemented per estimator).  Every failure raises
:class:`~statspai.exceptions.StatsPAIError` (the agent-native umbrella,
carrying a ``recovery_hint`` and ``diagnostics``); the too-few-rows case
uses :class:`~statspai.exceptions.DataInsufficient`, which is *also* a
:class:`ValueError`.  So ``except sp.StatsPAIError`` catches every case and
``except ValueError`` still catches the data-size failures.
"""

from __future__ import annotations

import re
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import pandas as pd

from .exceptions import (
    ColumnNotFound,
    DataInsufficient,
    MethodIncompatibility,
    StatsPAIError,
)


def require_columns(
    data: pd.DataFrame, columns: Sequence[str], *, function: str
) -> None:
    """Validate that ``data`` is a frame carrying every required column.

    A wrong type raises :class:`~statspai.exceptions.StatsPAIError`; a missing
    column raises :class:`~statspai.exceptions.ColumnNotFound` (a
    :class:`~statspai.exceptions.DataInsufficient`, hence a ``ValueError``
    *and* a ``StatsPAIError``, with ``diagnostics["did_you_mean"]``) whose
    message contains the
    conventional ``"Missing columns"`` phrase plus the offending names — far
    more actionable than the bare ``KeyError: "['typo'] not in index"`` pandas
    would otherwise raise, while staying back-compatible with callers that
    ``except ValueError`` on the long-standing missing-column contract.
    """
    if not isinstance(data, pd.DataFrame):
        raise StatsPAIError(
            f"{function}: `data` must be a pandas DataFrame, got "
            f"{type(data).__name__}.",
            recovery_hint="Pass a pandas DataFrame whose columns include the "
            "variables named in the call.",
            diagnostics={"received_type": type(data).__name__},
        )
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ColumnNotFound.from_missing(
            missing,
            list(data.columns),
            message=f"{function}: Missing columns {missing} — not found in data.",
        )


def missing_named_columns(
    columns: Union[Mapping[str, Any], Sequence[Any]], available: Iterable[Any]
) -> Dict[str, Any]:
    """``{label: column}`` for every requested column absent from ``available``.

    ``columns`` is either ``{argument_label: column_or_columns}`` or a plain
    sequence of column names (each labelled by itself).  ``None`` values are
    skipped; a list / tuple value under ``label`` is expanded element-wise to
    ``label[i]`` so the label says which entry of the user's list is wrong.
    Unhashable values (an array passed where a name is also accepted) are
    not column names and are skipped.
    """
    avail = set(available)

    def _absent(col: Any) -> bool:
        # arrays / Series passed in place of a name are not column names
        try:
            return col is not None and col not in avail
        except TypeError:
            return False

    if isinstance(columns, Mapping):
        items = list(columns.items())
    else:
        items = [(str(c), c) for c in columns]
    missing: Dict[str, Any] = {}
    for label, value in items:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for i, col in enumerate(value):
                if _absent(col):
                    missing[f"{label}[{i}]"] = col
        elif _absent(value):
            missing[str(label)] = value
    return missing


def check_columns(
    data: pd.DataFrame,
    columns: Union[Mapping[str, Any], Sequence[Any]],
    *,
    function: Optional[str] = None,
    message: Optional[str] = None,
    recovery_hint: Optional[str] = None,
    extra_hint: Optional[str] = None,
    extra_diagnostics: Optional[Dict[str, Any]] = None,
) -> None:
    """Raise :class:`~statspai.exceptions.ColumnNotFound` for absent columns.

    The column-name guard every estimator needs, with ``did_you_mean``
    suggestions.  ``columns`` follows :func:`missing_named_columns`:
    ``{"y": y, "treat": treat, "covariates": covariates}`` labels each
    missing column with the argument the caller typed.

    Examples
    --------
    >>> import pandas as pd
    >>> from statspai._input_validation import check_columns
    >>> from statspai.exceptions import ColumnNotFound
    >>> df = pd.DataFrame({"wage": [1.0], "x1": [0.0]})
    >>> try:
    ...     check_columns(df, {"y": "wage", "covariates": ["x11"]})
    ... except ColumnNotFound as e:
    ...     print(e.diagnostics["did_you_mean"])
    {'covariates[0]': 'x1'}
    """
    missing = missing_named_columns(columns, data.columns)
    if missing:
        raise ColumnNotFound.from_missing(
            missing,
            list(data.columns),
            message=message,
            recovery_hint=recovery_hint,
            context=function,
            extra_hint=extra_hint,
            extra_diagnostics=extra_diagnostics,
        )


#: The one message every estimator gives for ``inf`` in the data, so the
#: same input is refused the same way whichever estimator reads it.
INFINITE_VALUES_HINT = (
    "Infinite values are not missing values, so they are neither dropped "
    "nor passed to the estimator. Fix the step that produced them (a log "
    "of zero, a division by zero), or recode them as missing with "
    "data.replace([np.inf, -np.inf], np.nan) to have those rows dropped "
    "like any other missing value."
)


def _infinite_count(values: Any) -> int:
    """Number of ``+inf`` / ``-inf`` entries in a column (NaN not counted)."""
    import numpy as np

    if isinstance(values, pd.Series):
        dtype = values.dtype
        if isinstance(dtype, np.dtype):
            if dtype.kind not in "fc":
                if dtype.kind != "O":
                    return 0  # integers, bools, datetimes hold no inf
                return int(
                    sum(
                        1
                        for v in values.to_numpy()
                        if isinstance(v, float) and np.isinf(v)
                    )
                )
            arr = values.to_numpy()
        else:
            if not pd.api.types.is_float_dtype(dtype):
                return 0
            arr = values.to_numpy(dtype=float, na_value=np.nan)
    else:
        arr = np.asarray(values)
        if arr.dtype.kind not in "fc":
            return 0
    if arr.size == 0 or np.isfinite(arr).all():
        return 0
    return int(np.isinf(arr).sum())


def infinite_values_by_column(
    data: pd.DataFrame, columns: Iterable[Any]
) -> Dict[str, int]:
    """``{column: number of ±inf values}`` for the named columns that hold any.

    Names that are ``None`` or not columns of ``data`` are skipped (the
    caller's column check reports those). Missing values are not counted:
    they are the caller's listwise-deletion business, ``inf`` is not.
    """
    out: Dict[str, int] = {}
    if not isinstance(data, pd.DataFrame):
        return out
    seen = set()
    for col in columns:
        try:
            if col is None or col in seen or col not in data.columns:
                continue
        except TypeError:  # an unhashable array passed in place of a name
            continue
        seen.add(col)
        series = data[col]
        if isinstance(series, pd.DataFrame):  # duplicated column name
            continue
        n_inf = _infinite_count(series)
        if n_inf:
            out[str(col)] = n_inf
    return out


def raise_infinite(
    counts: Mapping[str, int], *, function: str, n_rows: Optional[int] = None
) -> None:
    """Raise the shared refusal for ``counts`` (``{column: n_inf}``) if any."""
    if not counts:
        return
    from .exceptions import DataInsufficient

    total = int(sum(counts.values()))
    cols = ", ".join(f"{c!r} ({n})" for c, n in counts.items())
    diagnostics: Dict[str, Any] = {
        "infinite_by_column": dict(counts),
        "n_infinite": total,
    }
    if n_rows is not None:
        diagnostics["n_rows_infinite"] = int(n_rows)
    raise DataInsufficient(
        f"{function}: {total} infinite value(s) in the data -- by column: " f"{cols}.",
        recovery_hint=INFINITE_VALUES_HINT,
        diagnostics=diagnostics,
    )


def require_finite(
    data: pd.DataFrame, columns: Iterable[Any], *, function: str
) -> None:
    """Refuse ``+inf`` / ``-inf`` in the columns an estimator uses.

    The package-wide policy for non-finite values: a missing value (NaN) is
    handled by the estimator's own missing-data rule (usually listwise
    deletion with an ``AssumptionWarning``), while an infinite value raises
    :class:`~statspai.exceptions.DataInsufficient` naming each column and its
    count. ``inf`` almost always comes from a transformation that failed
    (``log(0)``, ``x / 0``), so dropping it silently would hide the error and
    propagating it returns a meaningless number.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from statspai._input_validation import require_finite
    >>> from statspai.exceptions import DataInsufficient
    >>> df = pd.DataFrame({"y": [1.0, np.inf, 2.0], "x": [0.0, 1.0, np.nan]})
    >>> try:
    ...     require_finite(df, ["y", "x"], function="demo")
    ... except DataInsufficient as e:
    ...     print(e.diagnostics["infinite_by_column"])
    {'y': 1}
    """
    counts = infinite_values_by_column(data, columns)
    if not counts:
        return
    import numpy as np

    hit = np.zeros(len(data), dtype=bool)
    for col in counts:
        values = pd.to_numeric(data[col], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        hit |= np.isinf(values)
    raise_infinite(counts, function=function, n_rows=int(hit.sum()))


def clean_frame(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    function: str,
    n_params: int,
    missing: str = "drop",
) -> pd.DataFrame:
    """Validate columns, drop incomplete rows, enforce an identification floor.

    Returns the complete-case sub-frame (index reset).  Raises
    :class:`~statspai.exceptions.DataInsufficient` when ``n_params`` or fewer
    complete rows remain — the point below which the downstream linear algebra
    is under-determined and would otherwise emit a silent NaN estimate — and
    when a used column holds ``inf`` (:func:`require_finite`; infinite values
    are refused, not dropped as missing). ``missing="raise"`` refuses
    incomplete rows instead of dropping them (:func:`raise_on_missing`).
    """
    require_columns(data, columns, function=function)
    require_finite(data, columns, function=function)
    if check_missing_policy(missing, function=function) == "raise":
        raise_on_missing(data, columns, function=function)
    df = data[list(columns)].dropna().reset_index(drop=True)
    if len(df) <= n_params:
        raise DataInsufficient(
            f"{function}: only {len(df)} complete row(s) after dropping "
            f"missing values, but the model estimates {n_params} parameter(s) "
            "— need strictly more complete observations than parameters.",
            recovery_hint="Provide more non-missing observations or drop "
            "covariates.",
            diagnostics={"n_complete": int(len(df)), "n_params": int(n_params)},
        )
    return df


# --------------------------------------------------------------------- #
#  Missing-data policy and the shared estimator input contract
# --------------------------------------------------------------------- #


class TooFewClusters(DataInsufficient, MethodIncompatibility):
    """A cluster variable with fewer than two values.

    Both a :class:`DataInsufficient` and a :class:`MethodIncompatibility`,
    because the estimators whose own checks it replaced raised one or the
    other; every existing ``except`` clause keeps catching it.
    """

    code = "data_insufficient"


#: Values of the shared ``missing=`` option.
MISSING_POLICIES = ("drop", "raise")

#: Roles a ``missing=`` mapping may name. ``"response"`` is the outcome
#: (the left side of a formula, or the outcome argument); ``"predictors"``
#: is every other column the call uses: regressors, covariates, treatment,
#: instruments, weights, cluster, unit and time identifiers.
MISSING_ROLES = ("response", "predictors")

#: The annotation of the shared ``missing=`` parameter: one rule for every
#: used column, or a rule per role (``{"response": "raise"}``).
MissingPolicy = Union[Literal["drop", "raise"], Mapping[str, Literal["drop", "raise"]]]

#: The hint every ``missing="raise"`` refusal gives.
MISSING_RAISE_HINT = (
    "missing='raise' refuses incomplete rows instead of deleting them. "
    "Handle the missing values first (impute them with sp.mice, or drop the "
    "rows yourself with data.dropna(subset=[...]) so the sample is a "
    "decision you made), or pass missing='drop' to have the estimator "
    "delete incomplete rows listwise and report the count."
)


def check_missing_policy(missing: Any, *, function: str) -> str:
    """Validate a ``missing=`` value; returns it. Raises on anything else."""
    if isinstance(missing, str) and missing in MISSING_POLICIES:
        return missing
    from .exceptions import MethodIncompatibility

    raise MethodIncompatibility(
        f"{function}: missing must be 'drop' or 'raise', got {missing!r}.",
        recovery_hint="Pass missing='drop' (the default: delete incomplete "
        "rows listwise with a warning) or missing='raise' (refuse data with "
        "missing values in the columns the call uses).",
        diagnostics={"missing": repr(missing), "allowed": list(MISSING_POLICIES)},
    )


_ROLE_POLICY_HINT = (
    "Pass missing='drop', missing='raise', or a dict giving 'drop' or "
    "'raise' per role, e.g. missing={'response': 'raise', 'predictors': "
    "'drop'}. The roles are 'response' (the outcome) and 'predictors' "
    "(every other column the call uses, weights and cluster included); a "
    "role left out of the dict follows the default, 'drop'."
)


def resolve_missing_policy(
    missing: Any, *, function: str
) -> Union[str, Dict[str, str]]:
    """Validate ``missing=`` in either form.

    A string must be ``"drop"`` or ``"raise"`` and is returned as is. A
    mapping must name at least one role of :data:`MISSING_ROLES`, each with
    ``"drop"`` or ``"raise"``; it is returned with every role filled in (a
    role left out is ``"drop"``, the default). Anything else raises
    :class:`~statspai.exceptions.MethodIncompatibility`.

    Examples
    --------
    >>> from statspai._input_validation import resolve_missing_policy
    >>> resolve_missing_policy({"response": "raise"}, function="demo")
    {'response': 'raise', 'predictors': 'drop'}
    """
    if isinstance(missing, str):
        if missing in MISSING_POLICIES:
            return missing
    elif isinstance(missing, Mapping):
        unknown = [k for k in missing if k not in MISSING_ROLES]
        bad = {
            str(k): v
            for k, v in missing.items()
            if k in MISSING_ROLES and not (isinstance(v, str) and v in MISSING_POLICIES)
        }
        if missing and not unknown and not bad:
            return {role: str(missing.get(role, "drop")) for role in MISSING_ROLES}
        if not missing:
            problem = "the dict names no role"
        elif unknown:
            problem = f"unknown role(s) {[str(k) for k in unknown]}"
        else:
            problem = f"role value(s) {bad} are not 'drop' or 'raise'"
        raise MethodIncompatibility(
            f"{function}: missing={dict(missing)!r} is not a valid rule per "
            f"role: {problem}.",
            recovery_hint=_ROLE_POLICY_HINT,
            diagnostics={
                "missing": repr(dict(missing)),
                "allowed": list(MISSING_POLICIES),
                "allowed_roles": list(MISSING_ROLES),
                "unknown_roles": [str(k) for k in unknown],
                "invalid_values": {k: repr(v) for k, v in bad.items()},
            },
        )
    raise MethodIncompatibility(
        f"{function}: missing must be 'drop', 'raise' or a dict of those per "
        f"role, got {missing!r}.",
        recovery_hint=_ROLE_POLICY_HINT,
        diagnostics={
            "missing": repr(missing),
            "allowed": list(MISSING_POLICIES),
            "allowed_roles": list(MISSING_ROLES),
        },
    )


def missing_values_by_column(
    data: pd.DataFrame, columns: Iterable[Any]
) -> Dict[str, int]:
    """``{column: number of missing values}`` for the named columns with any.

    Missing means ``NaN`` / ``None`` / ``pd.NA`` / ``NaT`` (``isna``);
    ``inf`` is not missing (see :func:`require_finite`). Names that are
    ``None`` or not columns of ``data`` are skipped.
    """
    out: Dict[str, int] = {}
    if not isinstance(data, pd.DataFrame):
        return out
    seen = set()
    for col in columns:
        try:
            if col is None or col in seen or col not in data.columns:
                continue
        except TypeError:
            continue
        seen.add(col)
        series = data[col]
        if isinstance(series, pd.DataFrame):  # duplicated column name
            continue
        n = int(series.isna().sum())
        if n:
            out[str(col)] = n
    return out


def raise_on_missing(
    data: pd.DataFrame,
    columns: Iterable[Any],
    *,
    function: str,
    extra: Optional[Mapping[str, int]] = None,
    roles: Optional[Mapping[str, Sequence[str]]] = None,
    policy: Optional[Mapping[str, str]] = None,
) -> None:
    """The ``missing="raise"`` refusal: name each column and its count.

    ``extra`` adds counts for inputs that are not columns of ``data`` (a
    weights array, say), keyed by the argument name. Does nothing when no
    value is missing. With a rule per role, ``roles`` maps each role set to
    ``"raise"`` to its columns (and ``extra`` keys), and ``policy`` is the
    full rule; the diagnostics then add ``missing_by_role`` and
    ``missing_roles``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from statspai._input_validation import raise_on_missing
    >>> from statspai.exceptions import DataInsufficient
    >>> df = pd.DataFrame({"y": [1.0, 2.0, np.nan], "x": [np.nan, 1.0, 2.0]})
    >>> try:
    ...     raise_on_missing(df, ["y", "x"], function="demo")
    ... except DataInsufficient as e:
    ...     print(e.diagnostics["missing_by_column"], e.diagnostics["n_rows_missing"])
    {'y': 1, 'x': 1} 2
    """
    columns = [c for c in dict.fromkeys(c for c in columns if c is not None)]
    counts = missing_values_by_column(data, columns)
    extra_counts = {str(k): int(v) for k, v in (extra or {}).items() if v}
    if not counts and not extra_counts:
        return
    present = [c for c in columns if str(c) in counts]
    n_rows = int(data[present].isna().any(axis=1).sum()) if present else 0
    by_column = {**counts, **extra_counts}
    total = int(sum(by_column.values()))
    listing = ", ".join(f"{c!r} ({n})" for c, n in by_column.items())
    diagnostics: Dict[str, Any] = {
        "missing_policy": "raise" if policy is None else dict(policy),
        "missing_by_column": by_column,
        "n_missing": total,
        "n_rows_missing": n_rows,
    }
    where = "the columns this call uses"
    if roles is not None:
        by_role = {
            role: {str(c): by_column[str(c)] for c in cols if str(c) in by_column}
            for role, cols in roles.items()
        }
        by_role = {role: cols for role, cols in by_role.items() if cols}
        diagnostics["missing_by_role"] = by_role
        diagnostics["missing_roles"] = list(by_role)
        where = "the " + " and ".join(by_role) + " column(s)"
    raise DataInsufficient(
        f"{function}: missing="
        + ("'raise'" if policy is None else repr(dict(policy)))
        + f" and the data hold {total} missing value(s) in {where}"
        + (f", in {n_rows} row(s)" if n_rows else "")
        + f" -- by column: {listing}.",
        recovery_hint=MISSING_RAISE_HINT,
        diagnostics=diagnostics,
    )


#: Parameters that select an option, never a column. Their values are not
#: scanned for column names (``crit_dist="t"`` must not pick up a column
#: called ``t``).
_OPTION_PARAMS = frozenset(
    {
        "aggregation",
        "alpha",
        "backend",
        "balance",
        "base_period",
        "bootstrap",
        "bwselect",
        "control_group",
        "crit_dist",
        "engine",
        "estimand",
        "estimator",
        "family",
        "inference",
        "information",
        "kernel",
        "link",
        "marginal_effects",
        "masspoints",
        "method",
        "missing",
        "model",
        "outcome_model",
        "parity",
        "ps_model",
        "pretest",
        "pretrend_method",
        "scale",
        "se_method",
        "ssc",
        "weights_as",
        "window_rule",
    }
)

_NAME_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_QUOTED_NAME = re.compile(r"""Q\(\s*(['"])(.+?)\1\s*\)|`([^`]+)`""")


def _named_columns(value: Any, columns: Any) -> list:
    """Column names of ``columns`` that ``value`` refers to.

    A string is a column name, or a formula / option string whose name
    tokens (including ``Q("...")`` and backtick-quoted names) are matched
    against the columns. Lists, tuples and dict values are searched
    element-wise; anything else (numbers, arrays, learners) names nothing.
    """
    found: list = []
    if isinstance(value, str):
        if value in columns:
            return [value]
        for m in _QUOTED_NAME.finditer(value):
            name = m.group(2) or m.group(3)
            if name in columns:
                found.append(name)
        for token in _NAME_TOKEN.findall(value):
            if token in columns:
                found.append(token)
        return found
    if isinstance(value, Mapping):
        for v in value.values():
            found += _named_columns(v, columns)
        return found
    if isinstance(value, (list, tuple)):
        for v in value:
            found += _named_columns(v, columns)
    return found


def _is_option_parameter(param: Any) -> bool:
    """A parameter with a string default or a ``Literal`` annotation."""
    import inspect

    default = param.default
    if isinstance(default, str) and default:
        return True
    annotation = param.annotation
    if annotation is inspect.Parameter.empty:
        return False
    return "Literal" in str(annotation)


def _array_missing(value: Any, n_rows: int) -> int:
    """Missing values in a row-aligned array argument (a weights vector)."""
    import numpy as np

    if isinstance(value, pd.Series) or (
        isinstance(value, np.ndarray) and value.ndim == 1
    ):
        if len(value) != n_rows:
            return 0
        return int(pd.isna(pd.Series(np.asarray(value, dtype=object))).sum())
    return 0


def _outcome_columns(value: Any, columns: Any) -> list:
    """Columns an outcome argument names: the left side of a formula."""
    if isinstance(value, str) and "~" in value and value not in columns:
        return _named_columns(value.split("~", 1)[0], columns)
    return _named_columns(value, columns)


def _require_numeric_outcome(
    data: pd.DataFrame, columns: Sequence[str], *, function: str
) -> None:
    """Refuse a text outcome where the estimator needs numbers.

    Numeric, boolean and categorical columns pass (a categorical is the
    estimator's business); an ``object`` / string column passes only when
    every non-missing value is a number (``Decimal`` from SQL, Python
    ``int`` from JSON), as the formula layer reads those as numbers.
    """
    from .core._formula_dtypes import _is_number_object
    from .exceptions import MethodIncompatibility

    for col in columns:
        series = data[col]
        if isinstance(series, pd.DataFrame):
            continue
        dtype = series.dtype
        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            continue
        if isinstance(dtype, pd.CategoricalDtype):
            continue
        if not (
            pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
        ):
            continue
        if pd.api.types.is_object_dtype(dtype) and _is_number_object(series.to_numpy()):
            continue
        examples = list(pd.unique(series.dropna()))[:3]
        raise MethodIncompatibility(
            f"{function}: the outcome {col!r} holds text (dtype {dtype}), "
            f"e.g. {examples}; this estimator needs a numeric outcome.",
            recovery_hint=(
                "Convert the outcome to numbers first, e.g. "
                "pd.to_numeric(data[col]) for numbers stored as text, or "
                "(data[col] == 'yes').astype(int) for a two-level label. "
                "For an ordered outcome use sp.ologit / sp.oprobit with an "
                "ordered pd.Categorical."
            ),
            diagnostics={"column": str(col), "dtype": str(dtype)},
        )


def check_estimator_input(
    data: Any,
    arguments: Mapping[str, Any],
    *,
    function: str,
    signature: Any = None,
    missing: Union[str, Mapping[str, str]] = "drop",
    outcome: Sequence[str] = (),
    nan_ok: Sequence[str] = (),
    inf_ok: Sequence[str] = (),
    data_arg: str = "data",
) -> Optional[Dict[str, Any]]:
    """The input checks every core estimator runs before it fits.

    ``arguments`` maps parameter names to the values of the call (the
    bound arguments, ``**kwargs`` flattened in). The columns the call uses
    are read from them: plain column names, lists of names, formula tokens
    and the cluster spellings ``cluster=`` / ``clustervars=`` /
    ``vce="cluster v"`` / ``vcov={"CRV1": "v"}``. Option parameters (a
    string default, a ``Literal`` annotation, or a name in
    ``_OPTION_PARAMS``) are skipped. In order, the call is refused when

    1. ``data`` has no rows;
    2. a used column holds no non-missing value;
    3. an ``outcome`` column holds text;
    4. a used column holds ``inf`` (:func:`require_finite`);
    5. a cluster column takes fewer than two values;
    6. ``missing="raise"`` and a used column holds a missing value.

    ``missing`` may instead be a rule per role, as returned by
    :func:`resolve_missing_policy`: the ``outcome`` columns are the
    ``"response"`` role and every other used column (array arguments such
    as a weights vector included) is ``"predictors"``. Step 6 then applies
    to the roles set to ``"raise"``, and the return value records the
    missing values of the roles set to ``"drop"``, which the estimator
    deletes by its own rule: ``{"policy": ..., "missing_by_role": {role:
    {column: count}}, "n_rows_missing": ...}``. A string policy returns
    ``None``.

    Columns named by a parameter in ``nan_ok`` are exempt from 2 and 6
    (``NaN`` there carries meaning: a never-treated cohort, an unobserved
    outcome in a selection model); ``inf_ok`` exempts from 4 (``g = inf``
    for never treated). Only ``DataFrame`` input is checked.
    """
    if not isinstance(data, pd.DataFrame):
        return None
    import inspect

    by_role = isinstance(missing, Mapping)
    collect_arrays = by_role or missing == "raise"

    params = signature.parameters if signature is not None else {}
    columns = data.columns
    used: Dict[str, list] = {}  # parameter -> columns
    array_missing: Dict[str, int] = {}
    for name, value in arguments.items():
        if name == data_arg or name in _OPTION_PARAMS or value is None:
            continue
        spec = params.get(name)
        if spec is not None and spec.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if spec is not None and _is_option_parameter(spec):
            continue
        cols = _named_columns(value, columns)
        if cols:
            used[name] = list(dict.fromkeys(cols))
        elif collect_arrays and name not in nan_ok:
            n_missing = _array_missing(value, len(data))
            if n_missing:
                array_missing[name] = n_missing

    from .core._vcov_spec import _named_clusters, _named_clusters_plain

    clusters: list = []
    for key in ("robust", "vce", "vcov"):
        clusters += _named_clusters(arguments.get(key))
    for key in ("cluster", "clustervars"):
        clusters += _named_clusters_plain(arguments.get(key))
    clusters = [c for c in dict.fromkeys(clusters) if c in columns]
    if clusters:
        used.setdefault("cluster", [])
        used["cluster"] = list(dict.fromkeys(used["cluster"] + clusters))

    def _columns(exclude: Sequence[str]) -> list:
        out: list = []
        for name, cols in used.items():
            if name not in exclude:
                out += cols
        return list(dict.fromkeys(out))

    # 1. zero rows
    if len(data) == 0:
        raise DataInsufficient(
            f"{function}: `data` is empty (no rows); a fit needs at least one row.",
            recovery_hint="Check the filter or merge that produced the frame; "
            "it removed every row.",
            diagnostics={"n_rows": 0, "columns_used": _columns(())},
        )
    # 2. an entirely missing column
    nan_cols = _columns(nan_ok)
    empty = [
        c
        for c in nan_cols
        if not isinstance(data[c], pd.DataFrame) and bool(data[c].isna().all())
    ]
    if empty:
        roles = {c: name for name, cols in reversed(list(used.items())) for c in cols}
        listing = ", ".join(f"{c!r} ({roles.get(c, 'used')}=)" for c in empty)
        raise DataInsufficient(
            f"{function}: column(s) {listing} hold no non-missing value (entirely "
            "NaN), so no row can enter the estimation.",
            recovery_hint="An entirely missing column usually means a failed "
            "merge or a misspelt key upstream. Check how the column was "
            "built, or leave it out of the call.",
            diagnostics={"all_missing_columns": [str(c) for c in empty]},
        )
    # 3. a text outcome
    outcome_cols: list = []
    for name in outcome:
        value = arguments.get(name)
        if value is not None:
            outcome_cols += _outcome_columns(value, columns)
    if outcome_cols:
        _require_numeric_outcome(
            data, list(dict.fromkeys(outcome_cols)), function=function
        )
    # 4. infinite values
    require_finite(data, _columns(inf_ok), function=function)
    # 5. a single cluster
    for c in clusters:
        n_levels = int(data[c].nunique(dropna=True))
        if n_levels < 2:
            raise TooFewClusters(
                f"{function}: the cluster variable {c!r} takes {n_levels} "
                "value(s); cluster-robust standard errors need at least two "
                "clusters.",
                recovery_hint="Cluster on a variable with several groups, or "
                "drop cluster= for heteroskedasticity-robust errors. With "
                "few clusters (under about 30) see sp.wild_cluster_bootstrap.",
                diagnostics={"cluster": str(c), "n_clusters": n_levels},
            )
    # 6. the missing-data policy
    if not by_role:
        if missing == "raise":
            raise_on_missing(data, nan_cols, function=function, extra=array_missing)
        return None
    assert isinstance(missing, Mapping)
    response = set(outcome_cols)
    role_columns: Dict[str, list] = {
        "response": [c for c in nan_cols if c in response]
        + [a for a in array_missing if a in outcome],
        "predictors": [c for c in nan_cols if c not in response]
        + [a for a in array_missing if a not in outcome],
    }
    strict = {r: cols for r, cols in role_columns.items() if missing[r] == "raise"}
    if strict:
        raise_on_missing(
            data,
            [c for cols in strict.values() for c in cols if c in columns],
            function=function,
            extra={
                a: n
                for a, n in array_missing.items()
                if any(a in v for v in strict.values())
            },
            roles=strict,
            policy=missing,
        )
    lenient = {r: cols for r, cols in role_columns.items() if missing[r] == "drop"}
    record_by_role: Dict[str, Dict[str, int]] = {}
    lenient_frame_cols: List[Any] = []
    for role, cols in lenient.items():
        frame_cols = [c for c in cols if c in columns]
        lenient_frame_cols += frame_cols
        counts = missing_values_by_column(data, frame_cols)
        counts.update({str(a): array_missing[a] for a in cols if a in array_missing})
        record_by_role[role] = counts
    lenient_frame_cols = list(dict.fromkeys(lenient_frame_cols))
    n_rows = (
        int(data[lenient_frame_cols].isna().any(axis=1).sum())
        if lenient_frame_cols
        else 0
    )
    return {
        "policy": dict(missing),
        "missing_by_role": record_by_role,
        "n_rows_missing": n_rows,
    }


def checks_input(
    function: str,
    *,
    outcome: Sequence[str] = (),
    nan_ok: Sequence[str] = (),
    inf_ok: Sequence[str] = (),
    data_arg: str = "data",
) -> Any:
    """Decorator: run :func:`check_estimator_input` before the estimator.

    The wrapped function declares ``missing: MissingPolicy = "drop"`` in
    its signature; the decorator validates the value and applies the
    policy, so the function body never looks at it. The registry gives
    every decorated function the shared ``missing`` parameter card
    (``__statspai_checks_input__`` marks the wrapper). With a rule per
    role (``missing={"response": "raise"}``) the decorator also writes the
    record of :func:`check_estimator_input` to
    ``result.diagnostics["missing_by_role"]``. Place it
    below ``accepts_aliases`` / ``accepts_vce`` (it reads canonical names)
    and above decorators that rewrite the data (``markout_clusters``,
    ``_expands_categorical``, ``_calendar_time_aware``), so it sees the
    caller's frame.
    """
    import functools
    import inspect

    def deco(fn: Any) -> Any:
        sig = inspect.signature(fn)
        if "missing" not in sig.parameters:
            raise TypeError(f"{function}: checks_input needs a missing= parameter")
        default = sig.parameters["missing"].default

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                bound = sig.bind_partial(*args, **kwargs)
            except TypeError:
                return fn(*args, **kwargs)
            arguments: Dict[str, Any] = {}
            for name, value in bound.arguments.items():
                if sig.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
                    arguments.update(value)
                else:
                    arguments[name] = value
            policy = resolve_missing_policy(
                arguments.get("missing", default), function=function
            )
            record = check_estimator_input(
                arguments.get(data_arg),
                arguments,
                function=function,
                signature=sig,
                missing=policy,
                outcome=outcome,
                nan_ok=nan_ok,
                inf_ok=inf_ok,
                data_arg=data_arg,
            )
            result = fn(*args, **kwargs)
            if record is not None:
                diagnostics = getattr(result, "diagnostics", None)
                if isinstance(diagnostics, dict):
                    diagnostics["missing_by_role"] = record
            return result

        wrapper.__statspai_checks_input__ = True  # type: ignore[attr-defined]
        return wrapper

    return deco
