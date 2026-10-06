"""Putting regression inputs on a common scale."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import special

from ..exceptions import MethodIncompatibility

_BINARY_RULES = ("center", "full", "0/1", "-0.5/0.5", "none")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


def invlogit(x: Any) -> Any:
    """Inverse logit, ``1 / (1 + exp(-x))``.

    Maps a linear index to a probability. Stable for large ``|x|``.

    Parameters
    ----------
    x : float or array-like

    Returns
    -------
    float, ndarray or Series

    Examples
    --------
    >>> import statspai as sp
    >>> float(sp.invlogit(0.0))
    0.5
    """
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return special.expit(x.astype(float))
    out = special.expit(np.asarray(x, dtype=float))
    return float(out) if np.ndim(out) == 0 else out


def _rescale(col: pd.Series, binary: str, divisor: float) -> tuple:
    values = col.to_numpy(dtype=float)
    ok = np.isfinite(values)
    distinct = np.unique(values[ok])
    if distinct.size < 2:
        return col.astype(float), {"rule": "constant", "center": 0.0, "scale": 1.0}
    mean = float(values[ok].mean())
    sd = float(values[ok].std(ddof=1))
    if distinct.size == 2:
        lo, hi = float(distinct[0]), float(distinct[1])
        if binary == "none":
            return col.astype(float), {"rule": "none", "center": 0.0, "scale": 1.0}
        if binary == "center":
            return col - mean, {"rule": "center", "center": mean, "scale": 1.0}
        if binary == "0/1":
            return (col - lo) / (hi - lo), {
                "rule": "0/1",
                "center": lo,
                "scale": hi - lo,
            }
        if binary == "-0.5/0.5":
            mid = 0.5 * (lo + hi)
            return (col - mid) / (hi - lo), {
                "rule": "-0.5/0.5",
                "center": mid,
                "scale": hi - lo,
            }
    return (col - mean) / (divisor * sd), {
        "rule": f"{divisor:g} sd",
        "center": mean,
        "scale": divisor * sd,
    }


def _formula_inputs(formula: str, data: pd.DataFrame) -> List[str]:
    if "~" not in formula:
        raise MethodIncompatibility(f"formula must look like 'y ~ x'; got {formula!r}.")
    rhs = formula.split("~", 1)[1]
    seen: List[str] = []
    for name in _NAME.findall(rhs):
        if name in data.columns and name not in seen:
            seen.append(name)
    return seen


def standardize(
    data: Union[pd.DataFrame, pd.Series, Any],
    columns: Optional[Sequence[str]] = None,
    formula: Optional[str] = None,
    exclude: Optional[Sequence[str]] = None,
    binary: str = "center",
    divisor: float = 2.0,
) -> Any:
    """Centre numeric inputs and divide them by two standard deviations.

    After the rescaling a coefficient is the change in the outcome for a
    move from one standard deviation below the mean of an input to one
    above. That is directly comparable with the coefficient of a binary
    input, which compares its two values: a 0 / 1 variable with equal
    shares has standard deviation 0.5, so two standard deviations are
    its whole range. Dividing by one standard deviation, the usual
    z-score, makes continuous inputs look half as important as binary
    ones.

    Parameters
    ----------
    data : DataFrame or Series
        A Series (or array) is rescaled and returned.
    columns : list of str, optional
        Columns to rescale. Default: every numeric column, or the inputs
        of ``formula``.
    formula : str, optional
        ``'y ~ x1 + x2'``: rescale the numeric columns named on the
        right-hand side and leave the outcome alone. Refit the same
        formula on the returned data; interactions are then products of
        centred inputs, so main effects are effects at the mean.
    exclude : list of str, optional
        Columns to leave untouched.
    binary : {'center', 'full', '0/1', '-0.5/0.5', 'none'}, default 'center'
        Treatment of inputs with two distinct values: subtract the mean;
        treat like any numeric input; recode to 0 / 1; recode to
        -0.5 / 0.5; or leave as is.
    divisor : float, default 2
        Number of standard deviations to divide by. ``1`` gives z-scores.

    Returns
    -------
    DataFrame or Series
        A copy with the chosen columns rescaled. For a DataFrame,
        ``attrs['standardize']`` maps each column to the centre and
        scale used, so new data can be put on the same scale:
        ``(x - center) / scale``.

    Notes
    -----
    Categorical and string columns are never touched; their indicator
    coefficients already compare groups. A transformed term such as
    ``log(x)`` is not standardized through ``formula`` (R
    ``arm::standardize`` rescales ``x`` inside the logarithm, which
    changes the model): create the transformed column first and
    standardize that.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(10, 3, 200), "d": rng.integers(0, 2, 200)})
    >>> out = sp.standardize(df)
    >>> round(float(out["x"].std()), 3)
    0.5
    >>> abs(float(out["d"].mean())) < 1e-12
    True

    References
    ----------
    gelman2008scaling
    """
    if binary not in _BINARY_RULES:
        raise MethodIncompatibility(
            f"binary must be one of {_BINARY_RULES}; got {binary!r}."
        )
    if not divisor > 0:
        raise MethodIncompatibility("divisor must be positive.")
    if not isinstance(data, pd.DataFrame):
        series = data if isinstance(data, pd.Series) else pd.Series(np.asarray(data))
        if not pd.api.types.is_numeric_dtype(series):
            raise MethodIncompatibility("Only numeric values can be standardized.")
        out, _ = _rescale(series.astype(float), binary, divisor)
        return out if isinstance(data, pd.Series) else out.to_numpy()
    if columns is not None and formula is not None:
        raise MethodIncompatibility("Pass columns or formula, not both.")
    if formula is not None:
        names = _formula_inputs(formula, data)
    elif columns is not None:
        names = [str(c) for c in columns]
        missing = [c for c in names if c not in data.columns]
        if missing:
            raise MethodIncompatibility(f"Columns not in data: {missing}.")
    else:
        names = list(data.columns)
    skip = set(exclude or [])
    out = data.copy()
    record: Dict[str, Dict[str, Any]] = {}
    for name in names:
        col = data[name]
        if name in skip:
            continue
        numeric = pd.api.types.is_numeric_dtype(col) and not isinstance(
            col.dtype, pd.CategoricalDtype
        )
        if not numeric or pd.api.types.is_bool_dtype(col):
            if columns is not None:
                raise MethodIncompatibility(
                    f"Column {name!r} is not numeric and cannot be standardized."
                )
            continue
        out[name], record[name] = _rescale(col.astype(float), binary, divisor)
    out.attrs["standardize"] = record
    return out


__all__ = ["invlogit", "standardize"]
