"""Input checks ``sp.feols`` / ``sp.fepois`` / ``sp.feglm`` run before any backend.

The fixest entry points dispatch to the native HDFE engine or to pyfixest.
When the native planner declined a call, pyfixest received degenerate input
unchecked and its own errors reached the user: "All variables are collinear"
on a frame with no rows, ``ZeroDivisionError`` when there were more
regressors than observations, and rows with ``inf`` were dropped (with a
pyfixest warning) where ``sp.regress`` refuses them. These checks run first,
whatever the backend, and give the same typed errors ``sp.regress`` gives:

* no rows, or no complete row over the columns the formula uses;
* ``inf`` in a used column (the package-wide policy of
  :func:`statspai._input_validation.require_finite`);
* for a formula without fixed effects made of bare column names, no more
  complete rows than parameters (``nobs <= k`` leaves no residual degrees of
  freedom).

:func:`translate_backend_failure` is the net under these checks: the
pyfixest failures that mean "degenerate design" become typed errors that
name the entry point.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd

from .._input_validation import require_finite
from ..exceptions import DataInsufficient, NumericalInstability

__all__ = ["referenced_columns", "check_fixest_input", "translate_backend_failure"]

# a name that is not called as a function: ``x`` but not ``log(``
_TOKEN = re.compile(r"(?<![\w.])([A-Za-z_][A-Za-z0-9_.]*)\b(?!\s*\()")
_BARE_RHS = re.compile(r"^\s*[A-Za-z_]\w*(\s*\+\s*[A-Za-z_]\w*)*\s*$")


def referenced_columns(fml: Any, data: pd.DataFrame, extra: Iterable[Any]) -> List[str]:
    """Data columns the formula (and ``extra`` named columns) refer to."""
    cols = set(map(str, data.columns))
    found: List[str] = []
    if isinstance(fml, str):
        for name in _TOKEN.findall(fml):
            if name in cols and name not in found:
                found.append(name)
    for name in extra:
        if isinstance(name, str) and name in cols and name not in found:
            found.append(name)
    return found


def _n_params_plain(fml: str) -> Optional[int]:
    """Parameter count of ``y ~ x1 + x2`` (bare names, no fixed effects)."""
    if fml.count("~") != 1 or "|" in fml:
        return None
    lhs, rhs = fml.split("~", 1)
    rhs = rhs.strip()
    intercept = True
    for drop in ("- 1", "-1", "+ 0", "+0"):
        if rhs.endswith(drop):
            rhs = rhs[: -len(drop)].strip()
            intercept = False
    if not _BARE_RHS.match(rhs):
        return None
    terms = {t.strip() for t in rhs.split("+") if t.strip()}
    terms.discard("1")
    return len(terms) + int(intercept)


def check_fixest_input(
    fml: Any,
    data: Any,
    *,
    entry: str,
    weights: Optional[str] = None,
    cluster: Any = None,
    extra: Iterable[Any] = (),
) -> None:
    """Refuse degenerate input before ``entry`` dispatches to a backend."""
    if not isinstance(data, pd.DataFrame) or not isinstance(fml, str):
        return  # the backends' own type checks report these
    if len(data) == 0:
        raise DataInsufficient(
            f"{entry}: the data have no rows, so there is nothing to estimate.",
            recovery_hint="Check the filter or merge that produced the frame.",
            diagnostics={"n_rows": 0},
        )
    clusters = (
        [cluster]
        if isinstance(cluster, str)
        else (
            [c for c in cluster if isinstance(c, str)]
            if isinstance(cluster, (list, tuple))
            else []
        )
    )
    used = referenced_columns(fml, data, [weights, *clusters, *extra])
    if not used:
        return
    require_finite(data, used, function=entry)
    if not data.columns.is_unique:
        return  # pyfixest names the duplicated columns
    # rows fixest keeps: complete on every column the model reads
    model_cols = [c for c in used if c not in clusters]
    if not model_cols:
        return
    complete = int(data[model_cols].notna().all(axis=1).sum())
    if complete == 0:
        empty = [c for c in model_cols if data[c].isna().all()]
        detail = (
            f"; entirely missing: {empty}"
            if empty
            else "; every row misses a value in at least one of them"
        )
        raise DataInsufficient(
            f"{entry}: no row is complete on the columns the model uses "
            f"{model_cols}{detail}.",
            recovery_hint=(
                "Drop the entirely-missing variables from the formula, or "
                "fill the missing values (sp.mice) before fitting."
            ),
            diagnostics={
                "n_rows": int(len(data)),
                "n_complete": 0,
                "all_missing_columns": empty,
            },
        )
    k = _n_params_plain(fml)
    if k is not None and complete <= k:
        raise DataInsufficient(
            f"{entry} requires more observations than parameters to estimate "
            "residual variance and standard errors: "
            f"nobs={complete}, parameters={k}, residual df={complete - k}.",
            recovery_hint="Provide more complete observations or drop regressors.",
            diagnostics={"n_complete": complete, "n_params": k},
        )


def translate_backend_failure(exc: BaseException, entry: str) -> None:
    """Re-raise a pyfixest degenerate-design failure as a typed error.

    Returns (so the caller re-raises ``exc`` unchanged) for anything else.
    """
    text = str(exc)
    if isinstance(exc, ZeroDivisionError) or (
        isinstance(exc, ValueError) and "All variables are collinear" in text
    ):
        raise DataInsufficient(
            f"{entry}: the design has no estimable regressor after pyfixest "
            "removed collinear and absorbed columns, or no residual degrees "
            f"of freedom are left ({type(exc).__name__}: {text.strip()}).",
            recovery_hint=(
                "Check that the regressors vary within the fixed effects and "
                "that there are more complete observations than parameters."
            ),
            diagnostics={"backend": "pyfixest", "backend_error": text.strip()},
        ) from exc
    if isinstance(exc, (ValueError, TypeError)) and (
        "dependent variable must" in text.lower()
    ):
        from ..exceptions import MethodIncompatibility

        raise MethodIncompatibility(
            f"{entry}: the outcome does not fit the model family " f"({text.strip()}).",
            recovery_hint=(
                "A logit / probit family needs a 0/1 outcome and a Poisson "
                "one a non-negative count; recode the outcome or pick the "
                "family that matches it (feols for a continuous outcome)."
            ),
            diagnostics={"backend": "pyfixest", "backend_error": text.strip()},
        ) from exc
    if isinstance(exc, np.linalg.LinAlgError):
        raise NumericalInstability(
            f"{entry}: the pyfixest backend could not invert the design "
            f"({text.strip()}); a regressor is (numerically) collinear with "
            "the others or with the fixed effects.",
            recovery_hint=(
                "Drop or rescale the offending regressor; a column of very "
                "large integers (IDs near 2**62) loses its variation when "
                "converted to float64."
            ),
            diagnostics={"backend": "pyfixest", "backend_error": text.strip()},
        ) from exc
