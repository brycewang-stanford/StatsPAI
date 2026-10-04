"""
Utility functions for formula parsing and data processing
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from patsy import dmatrices

_BARE_NAME_RE = re.compile(r"^[A-Za-z_]\w*$")

# Functions a formula may call without the ``np.`` prefix, so that a formula
# written for R or Stata (``log(wage) ~ educ + sqrt(exper)``) reads the same
# here. A data column of the same name takes precedence, as in patsy.
_FORMULA_FUNCTIONS: Dict[str, Any] = {
    "np": np,
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "log1p": np.log1p,
    "exp": np.exp,
    "sqrt": np.sqrt,
}


def formula_eval_env() -> Any:
    """The namespace in which every StatsPAI formula is evaluated."""
    from patsy import EvalEnvironment

    return EvalEnvironment([_FORMULA_FUNCTIONS])


def evaluate_formula_expression(expr: str, data: pd.DataFrame) -> pd.Series:
    """One formula expression (``np.log(wage)``, ``I(a / b)``) as a column.

    Rows on which it cannot be evaluated are missing in the result. An
    expression that expands to several columns (``C(g)``, ``a * b``) is
    refused: an outcome or an endogenous regressor is a single variable.
    """
    from patsy import NAAction, dmatrix

    from ..exceptions import MethodIncompatibility

    frame = _coerce_string_extension_dtypes(data)
    try:
        design = dmatrix(
            "0 + " + expr,
            frame,
            eval_env=formula_eval_env(),
            return_type="dataframe",
            NA_action=NAAction(on_NA="drop"),
        )
    except Exception as exc:
        raise MethodIncompatibility(
            f"Could not evaluate {expr!r} on the data: {exc}",
            recovery_hint=(
                "Check the column names; log, exp, sqrt and np.<function> "
                "are available, and arithmetic goes inside I(...)."
            ),
            diagnostics={"expression": expr},
        ) from exc
    if design.shape[1] != 1:
        raise MethodIncompatibility(
            f"{expr!r} expands to {design.shape[1]} columns "
            f"({list(design.columns)}); a single variable is needed here.",
            recovery_hint="Create the variable as a column first.",
            diagnostics={"expression": expr},
        )
    return design.iloc[:, 0].astype(float).reindex(data.index)


def formula_to_columns(
    formula: str, data: pd.DataFrame
) -> Tuple[pd.DataFrame, str, List[str]]:
    """Evaluate ``y ~ terms`` and return ``(frame, y_name, x_names)``.

    For estimators that work from column names: the transformed outcome and
    every built regressor (``np.log(x)``, ``I(x**2)``, ``C(g)``, ``a:b``)
    become columns of a copy of ``data``, restricted to the rows on which
    the formula could be evaluated. The intercept is not among ``x_names``.
    """
    y_df, X_df = create_design_matrices(formula, data)
    frame = data.loc[X_df.index].copy()
    y_name = str(y_df.columns[0])
    frame[y_name] = np.asarray(y_df.iloc[:, 0], dtype=float)
    x_names = [str(c) for c in X_df.columns if c != "Intercept"]
    for col in x_names:
        frame[col] = np.asarray(X_df[col], dtype=float)
    return frame, y_name, x_names


def _split_additive_formula_terms(part: str) -> List[str]:
    """Split simple ``+`` formulas, preserving common intercept controls."""
    normalized = re.sub(r"(?<!^)\s*-\s*(?=[01](?:\b|$))", "+-", part)
    terms: List[str] = []
    for raw in normalized.split("+"):
        term = raw.strip()
        if not term:
            continue
        compact = re.sub(r"\s+", "", term)
        if compact in {"1", "+1"}:
            terms.append("1")
        elif compact == "-1":
            terms.append("-1")
        elif compact in {"0", "+0", "-0"}:
            terms.append("0")
        else:
            terms.append(term)
    return terms


def _try_simple_numeric_design_matrices(
    formula: str,
    data: pd.DataFrame,
    return_type: str = "dataframe",
) -> Optional[Tuple[Any, Any]]:
    """Fast path for plain numeric additive formulas.

    Patsy remains the compatibility path for categorical transforms,
    interactions, functions, quoting, and any non-numeric column. The common
    benchmark/user path ``y ~ x1 + x2`` can be built directly with the same
    column names, intercept convention, and NA-drop row index.
    """
    if return_type not in {"dataframe", "array"}:
        return None
    if formula.count("~") != 1 or "|" in formula:
        return None

    lhs, rhs = (part.strip() for part in formula.split("~", 1))
    if not _BARE_NAME_RE.match(lhs) or lhs not in data.columns:
        return None
    if not rhs:
        return None

    # Reject Patsy syntax and non-bare column names. A leading '-' is only
    # supported for the intercept-removal idiom, not variable subtraction.
    if re.search(r"[():*/\[\]{}]", rhs):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_\s+\-]+", rhs):
        return None

    intercept = True
    rhs_terms = [
        term.strip()
        for term in re.sub(r"(?<!^)-", "+-", rhs).split("+")
        if term.strip()
    ]
    x_names: List[str] = []
    seen = set()
    for term in rhs_terms:
        term = re.sub(r"\s+", "", term)
        if term in {"1", "+1"}:
            continue
        if term in {"0", "+0", "-0", "-1"}:
            intercept = False
            continue
        if term.startswith("-"):
            return None
        if not _BARE_NAME_RE.match(term) or term not in data.columns:
            return None
        if not pd.api.types.is_numeric_dtype(data[term]):
            return None
        if term not in seen:
            x_names.append(term)
            seen.add(term)

    if not x_names and not intercept:
        return None
    if not pd.api.types.is_numeric_dtype(data[lhs]):
        return None

    col_arrays = [
        data[col].to_numpy(dtype=float, na_value=np.nan, copy=False)
        for col in [lhs] + x_names
    ]
    complete = np.ones(len(data), dtype=bool)
    for col_arr in col_arrays:
        complete &= ~np.isnan(col_arr)
    if not bool(complete.any()):
        return None
    all_complete = bool(complete.all())
    if all_complete:
        index = data.index
    else:
        index = data.index[complete]

    y_values = col_arrays[0] if all_complete else col_arrays[0][complete]
    y_arr = y_values.reshape(-1, 1)
    x_cols: List[str] = []
    n_rows = y_arr.shape[0]
    n_cols = int(intercept) + len(x_names)
    X_arr = np.empty((n_rows, n_cols), dtype=float)
    offset = 0
    if intercept:
        X_arr[:, 0] = 1.0
        x_cols.append("Intercept")
        offset = 1
    if x_names:
        for pos, col_arr in enumerate(col_arrays[1:], start=offset):
            X_arr[:, pos] = col_arr if all_complete else col_arr[complete]
        x_cols.extend(x_names)

    if return_type == "array":
        return y_arr, X_arr
    y_df = pd.DataFrame(y_arr, columns=[lhs], index=index)
    X_df = pd.DataFrame(X_arr, columns=x_cols, index=index)
    return y_df, X_df


def parse_formula(formula: str) -> Dict[str, Any]:
    """
    Parse econometric formula into components

    Supports formulas like:
    - "y ~ x1 + x2"  (basic regression)
    - "y ~ x1 + x2 | fe1 + fe2"  (fixed effects)
    - "y ~ (x1 ~ z1 + z2) + x3"  (instrumental variables)

    Parameters
    ----------
    formula : str
        Formula string

    Returns
    -------
    Dict[str, Any]
        Parsed formula components
    """
    result: Dict[str, Any] = {
        "dependent": None,
        "exogenous": [],
        "endogenous": [],
        "instruments": [],
        "fixed_effects": [],
        "has_constant": True,
    }

    # Split by | for fixed effects
    if "|" in formula:
        main_formula, fe_part = formula.split("|", 1)
        result["fixed_effects"] = [var.strip() for var in fe_part.split("+")]
    else:
        main_formula = formula

    # Split dependent and independent variables
    if "~" not in main_formula:
        raise ValueError(
            "Formula must contain '~' to separate dependent and "
            "independent variables"
        )

    dependent_part, independent_part = main_formula.split("~", 1)
    result["dependent"] = dependent_part.strip()

    # Parse instrumental variables (in parentheses)
    iv_pattern = r"\(([^)]+)\)"
    iv_matches = re.findall(iv_pattern, independent_part)

    if iv_matches:
        for iv_spec in iv_matches:
            if "~" in iv_spec:
                endog, instruments = iv_spec.split("~", 1)
                result["endogenous"].extend([var.strip() for var in endog.split("+")])
                result["instruments"].extend(
                    [var.strip() for var in instruments.split("+")]
                )
            else:
                result["exogenous"].extend([var.strip() for var in iv_spec.split("+")])

        # Remove IV specifications from independent part
        independent_part = re.sub(iv_pattern, "", independent_part)

    # Parse remaining exogenous variables
    remaining_vars = _split_additive_formula_terms(independent_part)
    result["exogenous"].extend(remaining_vars)

    # Check for constant term
    if "1" in result["exogenous"]:
        result["exogenous"] = [var for var in result["exogenous"] if var != "1"]
    if "-1" in result["exogenous"] or "0" in result["exogenous"]:
        result["has_constant"] = False
        result["exogenous"] = [
            var for var in result["exogenous"] if var not in ["-1", "0"]
        ]

    return result


def _coerce_string_extension_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    """Cast pandas string-extension columns to ``object`` for patsy.

    pandas >= 3.0 makes ``StringDtype`` the default for text columns. patsy's
    categorical sniffer calls ``np.issubdtype(col.dtype, np.bool_)``, which
    raises ``TypeError: Cannot interpret '<StringDtype(...)>' as a data type``
    on any pandas extension dtype. Casting the offending columns back to
    ``object`` restores the pandas < 3.0 code path (object -> categorical) with
    identical downstream design matrices. Numeric / bool / datetime columns are
    left untouched, and the frame is copied only when a conversion is needed.
    """
    offenders = {}
    for col in data.columns:
        dtype = data[col].dtype
        if isinstance(dtype, pd.StringDtype) or (
            pd.api.types.is_extension_array_dtype(dtype)
            and pd.api.types.is_string_dtype(dtype)
        ):
            offenders[col] = data[col].astype(object)
    if offenders:
        data = data.assign(**offenders)
    return _widen_narrow_integers(data)


def _widen_narrow_integers(data: pd.DataFrame) -> pd.DataFrame:
    """Cast ``int8`` / ``int16`` / ``int32`` (and unsigned) columns to ``int64``.

    ``pd.read_stata`` and ``pd.read_parquet`` keep the storage type of the
    file, so a Stata ``int`` arrives as ``int16``. numpy arithmetic stays in
    that type and wraps around without a warning: ``I(x**2)`` on an ``int16``
    column with values above 181 is a column of garbage, and the regression
    built on it has no error to show for it. Stata and R do arithmetic in
    double precision, so the same formula there is right. ``int64`` keeps the
    names of ``C(x)`` levels unchanged (a cast to float would rename ``T.2``
    to ``T.2.0``). A column that only holds -1, 0 and 1 cannot overflow in
    a product or a power and is left as it is, so the indicator columns of a
    large frame are not copied; the frame itself is copied only when a
    column is widened.
    """
    if not isinstance(data, pd.DataFrame) or not data.columns.is_unique:
        return data
    narrow = {}
    for col, dtype in data.dtypes.items():
        if not (
            isinstance(dtype, np.dtype) and dtype.kind in "iu" and dtype.itemsize < 8
        ):
            continue
        values = data[col].to_numpy()
        if values.size and (values.max() > 1 or values.min() < -1):
            narrow[col] = values.astype(np.int64)
    if narrow:
        data = data.assign(**narrow)
    return data


def create_design_matrices(
    formula: str, data: pd.DataFrame, return_type: str = "dataframe"
) -> Tuple[Any, Any]:
    """
    Create design matrices from formula and data

    Parameters
    ----------
    formula : str
        Regression formula
    data : pd.DataFrame
        Input data
    return_type : str, default 'dataframe'
        Return type ('dataframe' or 'array')

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (y, X) matrices
    """
    fast = _try_simple_numeric_design_matrices(formula, data, return_type)
    if fast is not None:
        return fast

    data = _coerce_string_extension_dtypes(data)

    try:
        y, X = dmatrices(
            formula, data, eval_env=formula_eval_env(), return_type=return_type
        )
        return y, X
    except Exception as patsy_error:
        # Fallback to manual parsing if patsy fails
        parsed = parse_formula(formula)
        unknown = [
            name
            for name in [parsed["dependent"]] + list(parsed["exogenous"])
            if name not in data.columns
        ]
        if unknown:
            # the manual parser reads column names only; without them the
            # patsy message is the one that says what is wrong
            from ..exceptions import MethodIncompatibility

            raise MethodIncompatibility(
                f"Could not evaluate the formula {formula!r}: {patsy_error}",
                recovery_hint=(
                    "Check the column names and the functions the formula "
                    "calls; log, exp, sqrt and np.<function> are available."
                ),
                diagnostics={"formula": formula, "unknown_terms": unknown},
            ) from patsy_error

        y = data[parsed["dependent"]].values
        if return_type == "dataframe":
            y = pd.DataFrame(
                y,
                columns=[parsed["dependent"]],
                index=data.index,
            )

        X_cols = parsed["exogenous"].copy()
        if parsed["has_constant"]:
            X_cols = ["Intercept"] + X_cols

        if parsed["has_constant"]:
            X = np.column_stack(
                [np.ones(len(data))] + [data[col].values for col in parsed["exogenous"]]
            )
        else:
            X = np.column_stack([data[col].values for col in parsed["exogenous"]])

        if return_type == "dataframe":
            X = pd.DataFrame(X, columns=X_cols, index=data.index)

        return y, X


def prepare_data(
    data: pd.DataFrame,
    dependent: str,
    independent: List[str],
    weights: Optional[str] = None,
    subset: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Prepare data for econometric estimation

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    dependent : str
        Dependent variable name
    independent : List[str]
        Independent variable names
    weights : str, optional
        Weight variable name
    subset : pd.Series, optional
        Boolean series for subsetting data

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
        (y, X, weights) arrays
    """
    # Apply subset if provided
    if subset is not None:
        data = data[subset].copy()

    # Drop missing values
    all_vars = [dependent] + independent
    if weights:
        all_vars.append(weights)

    data_clean = data[all_vars].dropna()

    # Extract arrays
    y = data_clean[dependent].values
    X = data_clean[independent].values
    w = data_clean[weights].values if weights else None

    return y, X, w


def add_constant(X: np.ndarray, has_constant: bool = True) -> np.ndarray:
    """
    Add constant term to design matrix

    Parameters
    ----------
    X : np.ndarray
        Design matrix
    has_constant : bool, default True
        Whether to add constant

    Returns
    -------
    np.ndarray
        Design matrix with constant if requested
    """
    if has_constant:
        return np.column_stack([np.ones(X.shape[0]), X])
    return X


def get_variable_names(
    formula: str, data: pd.DataFrame, include_constant: bool = True
) -> List[str]:
    """
    Get variable names from formula

    Parameters
    ----------
    formula : str
        Regression formula
    data : pd.DataFrame
        Input data
    include_constant : bool, default True
        Whether to include constant in names

    Returns
    -------
    List[str]
        Variable names
    """
    parsed = parse_formula(formula)

    names = []
    if include_constant and parsed["has_constant"]:
        names.append("const")

    names.extend(parsed["exogenous"])

    return names
