"""Replicate-weight versions of :func:`sp.svymean`, :func:`sp.svytotal`
and :func:`sp.svyglm` (R ``svymean`` / ``svytotal`` / ``svyglm`` on a
``svrepdesign``): the statistic is recomputed with every replicate's
weights and its variance is :meth:`ReplicateDesign.variance`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..exceptions import MethodIncompatibility

if TYPE_CHECKING:
    from .estimators import SurveyResult
    from .replicate import ReplicateDesign


def _result(
    names: List[str], est: np.ndarray, V: np.ndarray, df: float, alpha: float
) -> "SurveyResult":
    from .estimators import SurveyResult

    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    crit = sp_stats.t.ppf(1 - alpha / 2, df=df)
    return SurveyResult(
        estimate=pd.Series(est, index=names),
        std_error=pd.Series(se, index=names),
        ci_lower=pd.Series(est - crit * se, index=names),
        ci_upper=pd.Series(est + crit * se, index=names),
        deff=pd.Series(np.full(len(names), np.nan), index=names),
        dof=df,
        alpha=alpha,
    )


def _domain_weights(design: "ReplicateDesign", subpop: Any) -> np.ndarray:
    """``(n, 1 + R)`` weights, zero outside the domain."""
    from .estimators import _domain_mask

    W = design.all_weights()
    dom = _domain_mask(design, subpop)  # type: ignore[arg-type]
    return W if dom is None else W * dom[:, None]


def rep_mean_total(
    variables: Union[str, List[str]],
    design: "ReplicateDesign",
    alpha: float,
    subpop: Any,
    total: bool,
) -> "SurveyResult":
    from .estimators import _resolve_vars

    names, vals = _resolve_vars(
        variables, design.data, "sp.svytotal" if total else "sp.svymean"
    )
    if not np.all(np.isfinite(vals)):
        raise MethodIncompatibility(
            "replicate design: the variables have missing values.",
            recovery_hint="Drop or impute them, or use subpop= for a domain.",
        )
    W = _domain_weights(design, subpop)
    sums = W.T @ vals  # (1 + R, p)
    stats = sums if total else sums / W.sum(axis=0)[:, None]
    V = design.variance(stats[0], stats[1:])
    return _result(names, stats[0], V, design.df, alpha)


def rep_glm(
    y: np.ndarray,
    X: np.ndarray,
    names: List[str],
    pos: np.ndarray,
    design: "ReplicateDesign",
    family: str,
    alpha: float,
    dof: str,
    dom: Optional[np.ndarray],
) -> "SurveyResult":
    """Refit the GLM with every replicate's weights (R svyglm on svrep)."""
    from .estimators import _irls_fit, _wls_fit

    W = design.all_weights()
    if dom is not None:
        W = W * dom[:, None]
    W = W[pos]

    def fit(w: np.ndarray) -> np.ndarray:
        keep = w > 0
        if family == "gaussian":
            return np.asarray(_wls_fit(y[keep], X[keep], w[keep])[0])
        if family in ("binomial", "poisson"):
            return np.asarray(_irls_fit(y[keep], X[keep], w[keep], family=family)[0])
        raise ValueError(
            f"Unknown family: {family}. Use 'gaussian', 'binomial', or 'poisson'."
        )

    full = fit(W[:, 0])
    reps = np.vstack([fit(W[:, r]) for r in range(1, W.shape[1])])
    V = design.variance(full, reps)
    df = design.df
    if dof == "residual":
        df = df + 1 - X.shape[1]
    elif dof != "design":
        raise MethodIncompatibility(f"dof must be 'design' or 'residual'; got {dof!r}")
    return _result(names, full, V, df, alpha)
