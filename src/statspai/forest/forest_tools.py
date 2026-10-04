"""Forest-agnostic post-estimation: ``variable_importance``,
``best_linear_projection`` and ``get_scores``.

These are the grf functions of the same names, lifted to one entry point
that accepts any GRF-engine forest StatsPAI fits:

* :func:`statspai.causal_forest` (GRF engine),
* :func:`statspai.iv_forest`, :func:`statspai.multi_arm_forest`,
  :func:`statspai.causal_survival_forest` (doubly-robust scores),
* :func:`statspai.lm_forest`, :func:`statspai.regression_forest`,
  :func:`statspai.probability_forest`, :func:`statspai.quantile_forest`,
  :func:`statspai.survival_forest` (variable importance only).

References
----------
[@athey2019generalized], [@semenova2021debiased]
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from ..exceptions import MethodIncompatibility
from ._grf_family import (
    GRFFamilyForest,
    importance_from_split_frequencies,
    validate_alpha,
    validate_vcov_type,
)

_SCORE_FORESTS = (
    "sp.causal_forest",
    "sp.iv_forest",
    "sp.multi_arm_forest",
    "sp.causal_survival_forest",
)


def _is_causal_forest(forest: Any) -> bool:
    from .causal_forest import CausalForest

    return isinstance(forest, CausalForest)


def _projection_design(
    A: Any, forest: Any, context: str
) -> Tuple[Optional[np.ndarray], List[str]]:
    """``(A matrix or None, coefficient names)`` for a projection on the
    forest's training rows."""
    n = int(forest.n_obs)
    if A is None:
        return None, ["Intercept"]
    if isinstance(A, pd.Series):
        A = A.to_frame()
    if isinstance(A, pd.DataFrame):
        names = [str(c) for c in A.columns]
        mat = A.to_numpy(dtype=float)
    else:
        mat = np.asarray(A, dtype=float)
        if mat.ndim == 1:
            mat = mat[:, None]
        names = [f"A{j + 1}" for j in range(mat.shape[1])]
    if mat.ndim != 2 or mat.shape[0] != n:
        raise MethodIncompatibility(
            f"{context}: A must have one row per training observation "
            f"({n} rows after missing-value removal).",
            recovery_hint=(
                "The doubly-robust scores are defined on the training rows; "
                "pass covariates aligned with them."
            ),
            diagnostics={"shape": list(mat.shape), "n_obs": n},
        )
    if not np.isfinite(mat).all():
        raise MethodIncompatibility(
            f"{context}: A has non-finite values.",
            recovery_hint="Drop or impute missing projection covariates.",
        )
    return mat, ["Intercept"] + names


def variable_importance(
    forest: Any, decay_exponent: float = 2.0, max_depth: int = 4
) -> pd.Series:
    """
    Split-frequency variable importance of a forest (``grf::variable_importance``).

    For depths ``d = 1..max_depth`` the share of splits made on each
    covariate at that depth is weighted by ``d^(-decay_exponent)``; the
    weighted shares are summed and divided by the total weight, so the
    importances sum to one.  This measures how often the forest *uses* a
    covariate near the root -- for a causal forest, how often it splits on
    it to separate effect heterogeneity -- and is not a test of anything.

    Parameters
    ----------
    forest : fitted GRF-engine forest
        Any of ``sp.causal_forest`` (GRF engine), ``sp.iv_forest``,
        ``sp.multi_arm_forest``, ``sp.lm_forest``, ``sp.regression_forest``,
        ``sp.multi_regression_forest``, ``sp.probability_forest``,
        ``sp.quantile_forest``, ``sp.survival_forest``,
        ``sp.causal_survival_forest``.
    decay_exponent : float, default 2
        How quickly deeper splits lose weight.
    max_depth : int, default 4
        Deepest level counted.

    Returns
    -------
    pd.Series
        Importance per covariate (sums to one), in covariate order.

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(1000, 3)); T = rng.binomial(1, 0.5, 1000)
    >>> Y = X[:, 0] * T + rng.normal(size=1000)
    >>> cf = sp.causal_forest(Y=Y, T=T, X=X, n_estimators=200, random_state=1)
    >>> sp.variable_importance(cf).idxmax()  # doctest: +SKIP
    'X0'

    References
    ----------
    [@athey2019generalized]
    """
    if int(max_depth) < 1:
        raise MethodIncompatibility(
            "variable_importance(): max_depth must be >= 1.",
            recovery_hint="Use max_depth=4 (grf's default).",
        )
    if isinstance(forest, GRFFamilyForest):
        return forest.variable_importance(decay_exponent, max_depth)
    if _is_causal_forest(forest):
        if getattr(forest, "_engine", None) is None:
            raise MethodIncompatibility(
                "variable_importance(): the forest was not fitted with the "
                "GRF engine.",
                recovery_hint="Refit sp.causal_forest with the default split_rule.",
            )
        counts = forest._engine.split_frequencies(int(max_depth))
        imp = importance_from_split_frequencies(counts, decay_exponent)
        names = forest._feature_names or [f"X{j}" for j in range(counts.shape[1])]
        return pd.Series(imp, index=list(names), name="importance")
    raise MethodIncompatibility(
        f"variable_importance(): unsupported object {type(forest).__name__}.",
        recovery_hint="Pass a forest fitted by a StatsPAI GRF-family function.",
    )


def get_scores(forest: Any, **kwargs: Any) -> np.ndarray:
    """
    Doubly-robust scores of a forest's average effect (``grf::get_scores``).

    The mean of the scores is the forest's doubly-robust average effect;
    regressing them on covariates is the best linear projection.  Every
    score uses the forest's *out-of-bag* conditional effect.

    Parameters
    ----------
    forest : fitted forest
        ``sp.causal_forest`` (GRF engine, binary or continuous treatment),
        ``sp.iv_forest`` (average conditional LATE), ``sp.multi_arm_forest``
        (one column per contrast), ``sp.causal_survival_forest``.
    **kwargs
        Forwarded to the forest's own ``get_scores`` (e.g.
        ``compliance_score=`` for an instrumental forest).

    Returns
    -------
    np.ndarray
        ``(n,)`` scores, or ``(n, K-1)`` for a multi-arm forest.

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(500, 2)); T = rng.binomial(1, 0.5, 500)
    >>> Y = (1 + X[:, 0]) * T + rng.normal(size=500)
    >>> cf = sp.causal_forest(Y=Y, T=T, X=X, n_estimators=100, random_state=1)
    >>> sp.get_scores(cf).shape
    (500,)

    References
    ----------
    [@athey2019generalized], [@robins1994estimation]
    """
    if _is_causal_forest(forest):
        from . import _grf_inference as gi

        if not gi.is_grf_forest(forest):
            raise MethodIncompatibility(
                "get_scores(): the causal forest was not fitted with the GRF "
                "engine.",
                recovery_hint="Refit sp.causal_forest with the default split_rule.",
            )
        gi._require_dr(forest, "get_scores()")
        gi.require_finite_oob(forest, "get_scores()")
        scores, *_ = gi.dr_scores(forest, np.asarray(forest._oob_tau, dtype=float))
        return np.asarray(scores, dtype=float)
    fn = getattr(forest, "get_scores", None)
    if fn is None:
        raise MethodIncompatibility(
            f"get_scores(): {type(forest).__name__} has no doubly-robust scores.",
            recovery_hint="Scores exist for " + ", ".join(_SCORE_FORESTS) + ".",
        )
    return np.asarray(fn(**kwargs), dtype=float)


@accepts_aliases(_strict=True, vcov_type="vce")
def best_linear_projection(
    forest: Any,
    A: Any = None,
    vce: str = "HC3",
    alpha: float = 0.05,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Best linear projection of the conditional effect on covariates.

    Regresses the forest's doubly-robust scores on ``(1, A)`` with a
    heteroskedasticity- (cluster-, when the forest has clusters) robust
    covariance -- ``grf::best_linear_projection``.  The coefficients are
    the population best linear predictor of ``tau(X)`` given ``A``
    (Semenova and Chernozhukov 2021); ``A=None`` gives the doubly-robust
    average effect.

    Parameters
    ----------
    forest : fitted forest, or a DML result
        ``sp.causal_forest`` (GRF engine), ``sp.iv_forest``,
        ``sp.multi_arm_forest`` (one projection per contrast, stacked),
        ``sp.causal_survival_forest``. Also the result of
        ``sp.dml(model='irm')`` with the ATE score: its cross-fitted
        doubly-robust scores are projected the same way (``DoubleML``'s
        ``cate()`` / ``gate()`` with ``A`` a basis or group indicators),
        and the joint covariance is in ``.attrs['vcov']``.
    A : array-like or DataFrame, optional
        Projection covariates, one row per training observation (after
        missing-value removal).
    vce : {"HC3", "HC0", "HC1", "HC2"}, default "HC3"
        Covariance type (``sandwich::vcovCL`` conventions).
    alpha : float, default 0.05
        Significance level for the intervals.
    **kwargs
        Forwarded to the forest's score construction.

    Returns
    -------
    pd.DataFrame
        Rows ``Intercept`` and one per column of ``A`` with ``coef``,
        ``se``, ``t``, ``p``, ``ci_lower``, ``ci_upper``.

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(1000, 3)); T = rng.binomial(1, 0.5, 1000)
    >>> Y = (1 + X[:, 0]) * T + rng.normal(size=1000)
    >>> cf = sp.causal_forest(Y=Y, T=T, X=X, n_estimators=400, random_state=1)
    >>> sp.best_linear_projection(cf, A=X[:, :1])  # doctest: +SKIP

    References
    ----------
    [@semenova2021debiased], [@athey2019generalized]
    """
    alpha = validate_alpha(alpha, "best_linear_projection()")
    vce = validate_vcov_type(vce, "best_linear_projection()")
    info = getattr(forest, "model_info", None)
    if isinstance(info, dict) and "dml_model" in info:
        return _dml_best_linear_projection(info, A, vce, alpha)
    if _is_causal_forest(forest):
        from . import _grf_inference as gi

        if not gi.is_grf_forest(forest):
            raise MethodIncompatibility(
                "best_linear_projection(): the causal forest was not fitted "
                "with the GRF engine.",
                recovery_hint="Refit sp.causal_forest with the default split_rule.",
            )

        class _N:
            n_obs = int(len(forest._Y_original))

        mat, names = _projection_design(A, _N, "best_linear_projection()")
        return gi.best_linear_projection(
            forest, mat, names, alpha=alpha, clip=0.0, vcov_type=vce
        )
    fn = getattr(forest, "best_linear_projection", None)
    if fn is None or not isinstance(forest, GRFFamilyForest):
        raise MethodIncompatibility(
            "best_linear_projection(): unsupported object " f"{type(forest).__name__}.",
            recovery_hint="Supported: " + ", ".join(_SCORE_FORESTS) + ".",
        )
    return fn(A=A, vce=vce, alpha=alpha, **kwargs)


def _dml_best_linear_projection(
    info: dict, A: Any, vce: str, alpha: float
) -> pd.DataFrame:
    """Project the cross-fitted doubly-robust scores of ``sp.dml(model='irm')``."""
    from ._grf_inference import _coef_table, cluster_robust_vcov

    context = "best_linear_projection()"
    scores = info.get("_dr_scores")
    if scores is None or info.get("cluster") is not None:
        raise MethodIncompatibility(
            f"{context}: needs the doubly-robust scores of an unweighted, "
            "unclustered sp.dml(model='irm') fit with the ATE score; got "
            f"model={str(info.get('dml_model')).lower()!r}"
            + (", clustered" if info.get("cluster") is not None else "")
            + ".",
            recovery_hint=(
                "Fit sp.dml(..., model='irm') without sample_weight / cluster, "
                "or use sp.causal_forest for clustered data."
            ),
        )
    scores = np.asarray(scores, dtype=float)
    n = len(scores)

    class _N:
        n_obs = n

    mat, names = _projection_design(A, _N, context)
    design = np.ones((n, 1)) if mat is None else np.column_stack([np.ones(n), mat])
    beta = np.linalg.lstsq(design, scores, rcond=None)[0]
    resid = scores - design @ beta
    V = cluster_robust_vcov(
        design, resid, weights=np.ones(n), clusters=None, vcov_type=vce
    )
    out = _coef_table(beta, V, names, alpha)
    out = out.drop(columns=["null"]).rename(
        columns={"ci_low": "ci_lower", "ci_high": "ci_upper"}
    )
    out.attrs["method"] = f"cross-fitted AIPW scores of sp.dml(model='irm'), {vce} SE"
    out.attrs["vcov"] = pd.DataFrame(V, index=names, columns=names)
    return out


__all__ = ["variable_importance", "best_linear_projection", "get_scores"]
