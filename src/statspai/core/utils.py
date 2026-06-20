"""
Utility functions for formula parsing and data processing
"""

from typing import Tuple, List, Optional, Dict, Any
import pandas as pd
import numpy as np
import re
from patsy import dmatrices

_BARE_NAME_RE = re.compile(r"^[A-Za-z_]\w*$")


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

    try:
        y, X = dmatrices(formula, data, return_type=return_type)
        return y, X
    except Exception:
        # Fallback to manual parsing if patsy fails
        parsed = parse_formula(formula)

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
