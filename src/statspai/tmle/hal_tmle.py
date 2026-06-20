r"""HAL-TMLE — TMLE with an L1-penalised step-function (HAL-style) nuisance.

TMLE is doubly-robust and semiparametrically efficient *given* good nuisance
estimates.  When those nuisances are rich and non-smooth, generic ML learners
such as random forests can under-regularise (overfit near the boundary) or
over-smooth (miss step-like heterogeneity), degrading finite-sample coverage.

**Implementation note — main-effects HAL only.** The full Highly Adaptive
Lasso (Benkeser & van der Laan 2016) uses **all subset-product** indicator
basis functions :math:`\phi_S(x) = \prod_{j\in S}\mathbb I\{x_j \le a_j\}`
across :math:`S \subseteq \{1,\ldots,p\}` — that basis is rich enough
to approximate any càdlàg function of bounded variation. Computing it
requires :math:`O(n \cdot 2^p)` columns and is impractical without
sparse-tensor tricks. This module implements the **main-effects-only**
restriction: per-feature step functions
:math:`\mathbb I\{x_j \le a_j\}` only, with :math:`O(np)` columns, fit
via L1-penalised regression. This is **L1-penalised additive piecewise-
constant regression**, not full HAL — it lacks HAL's universal càdlàg
approximation guarantee, but it shares HAL's flexibility on
additively-separable signals and is the variant most production HAL-TMLE
implementations actually ship. ``max_anchors_per_col`` further caps the
basis when a feature has many distinct values; quantile anchors are
substituted above the cap.

A single scalar ``lambda_`` controls the :math:`L_1` penalty — when ``None``
we pick it via 5-fold CV.

References
----------
Benkeser, D. & van der Laan, M. J. (2016). The Highly Adaptive Lasso
    Estimator. *2016 IEEE Int. Conf. on Data Science and Advanced
    Analytics (DSAA)*, 689–696. [@benkeser2016highly]
Li, Y., Qiu, S., Wang, Z. & van der Laan, M. J. (2025). Regularized
    Targeted Maximum Likelihood Estimation in Highly Adaptive Lasso
    Implied Working Models.  arXiv:2506.17214. [@li2025regularized]
van der Laan, M. J., Benkeser, D. & Cai, W. (2023). Efficient estimation
    of pathwise differentiable target parameters with the undersmoothed
    highly adaptive lasso. *International Journal of Biostatistics*,
    19(1), 261–289.  doi 10.1515/ijb-2019-0092. [@vanderlaan2023efficient]
"""

from __future__ import annotations

import inspect
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["hal_tmle", "HALRegressor", "HALClassifier"]


# ---------------------------------------------------------------------------
# Minimal duck-typed sklearn-estimator base.
#
# Step 1D of the cold-start budget: ``HALRegressor`` and ``HALClassifier``
# previously subclassed ``sklearn.base.BaseEstimator`` plus a Mixin, which
# pulled ~39 ``sklearn.*`` submodules into ``sys.modules`` for every
# ``import statspai`` — the only remaining sklearn footprint after Steps
# 1B/1C lazy-loaded ``forest`` and the 18 estimator files.  Inheriting
# from sklearn's base classes is gratuitous here: ``super_learner.fit``
# only needs ``sklearn.base.clone(learner)`` (which is duck-typed —
# ``get_params(deep=False)`` + ``cls(**params)`` reconstruction) plus
# ``.fit`` / ``.predict`` / ``.predict_proba``.  No code path on the HAL
# classes calls ``.score(...)``, ``is_classifier(...)``, or
# ``is_regressor(...)``.
#
# ``_BaseHAL`` reproduces the slice of ``BaseEstimator`` that ``clone()``
# actually uses, derived from sklearn 1.x:
#
#   - ``get_params(deep=True)``: introspect ``__init__`` signature, return
#     ``{name: getattr(self, name)}``.  Identity is preserved (no copy)
#     so sklearn's clone post-clone sanity check (``param1 is param2``)
#     passes.
#   - ``set_params(**params)``: ``setattr`` for each.
#   - ``__repr__``: ``ClassName(k=v, ...)`` matching sklearn's style.
#
# ``_estimator_type`` is set on the subclasses so ``sklearn.base.is_regressor``
# / ``is_classifier`` keep returning the right answer if any future
# external caller tries them.
# ---------------------------------------------------------------------------


class _BaseHAL:
    """Minimal sklearn-compatible duck-typed estimator base.

    Provides the ``get_params`` / ``set_params`` / ``__repr__`` slice of
    ``sklearn.base.BaseEstimator`` — sufficient for
    ``sklearn.base.clone()`` round-trip and standard cross-fitting
    pipelines — without forcing sklearn at module-load time.
    """

    def get_params(self, deep: bool = True) -> dict:
        # Mirror ``sklearn.base.BaseEstimator.get_params``: introspect
        # ``__init__`` and return ``{name: getattr(self, name)}`` with
        # the original object identity preserved.  ``deep`` is accepted
        # for sklearn-protocol compatibility but ignored — HAL params
        # are scalars, not nested estimators.
        del deep
        params: dict = {}
        for name in inspect.signature(type(self).__init__).parameters:
            if name == "self":
                continue
            params[name] = getattr(self, name)
        return params

    def set_params(self, **params: Any) -> "_BaseHAL":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        params = self.get_params(deep=False)
        items = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{type(self).__name__}({items})"


def _hal_basis(
    X: np.ndarray,
    anchors: Optional[np.ndarray] = None,
    max_anchors_per_col: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """Main-effects HAL basis: column-wise step functions at anchor points.

    ``anchors`` is a flat 2-column array ``[[j, value], ...]`` of (feature
    index, breakpoint) pairs — emitted when first called on training data and
    reused on prediction time.  If ``anchors`` is None we generate them from
    the sorted values of each column, capped at ``max_anchors_per_col`` to
    keep the basis manageable on larger samples.
    """
    n, p = X.shape
    if anchors is None:
        cols: list = []
        vals: list = []
        for j in range(p):
            xv = np.unique(X[:, j])
            if len(xv) > max_anchors_per_col:
                q = np.linspace(0, 1, max_anchors_per_col + 1)[1:-1]
                xv = np.quantile(X[:, j], q)
            for v in xv:
                cols.append(j)
                vals.append(v)
        anchors = np.column_stack(
            [np.asarray(cols, dtype=int), np.asarray(vals, dtype=float)]
        )

    B = np.zeros((n, anchors.shape[0]))
    for k in range(anchors.shape[0]):
        j = int(anchors[k, 0])
        v = float(anchors[k, 1])
        B[:, k] = (X[:, j] <= v).astype(float)
    return B, anchors


def _validate_hal_positive_int(value: Any, *, name: str) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, bool)
        or int(value) < 1
    ):
        raise MethodIncompatibility(
            f"{name} must be a positive integer.",
            recovery_hint=f"Use {name} >= 1.",
            diagnostics={name: value},
        )
    return int(value)


def _coerce_hal_matrix(
    X: Any,
    *,
    context: str,
    expected_features: Optional[int] = None,
) -> np.ndarray:
    try:
        X_arr = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: X must be numeric.",
            recovery_hint="Convert HAL feature columns to numeric values.",
        ) from exc
    if X_arr.ndim == 1:
        if (
            expected_features is not None
            and expected_features > 1
            and X_arr.size == expected_features
        ):
            X_arr = X_arr.reshape(1, -1)
        elif expected_features in (None, 1):
            X_arr = X_arr.reshape(-1, 1)
        else:
            raise MethodIncompatibility(
                f"{context}: one-dimensional X does not match fit features.",
                recovery_hint="Pass X shaped (n_samples, n_features).",
                diagnostics={
                    "n_values": int(X_arr.size),
                    "expected_features": expected_features,
                },
            )
    elif X_arr.ndim != 2:
        raise MethodIncompatibility(
            f"{context}: X must be a 1D or 2D numeric array.",
            recovery_hint="Pass X shaped (n_samples, n_features).",
            diagnostics={"x_ndim": int(X_arr.ndim)},
        )
    if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
        raise DataInsufficient(
            f"{context}: X must contain at least one row and one feature.",
            recovery_hint="Pass a non-empty HAL feature matrix.",
            diagnostics={
                "nobs": int(X_arr.shape[0]),
                "n_features": int(X_arr.shape[1]),
            },
        )
    if expected_features is not None and X_arr.shape[1] != expected_features:
        raise MethodIncompatibility(
            f"{context}: feature count does not match fit().",
            recovery_hint="Use the same number and order of HAL features.",
            diagnostics={
                "expected_features": expected_features,
                "observed_features": int(X_arr.shape[1]),
            },
        )
    if not np.isfinite(X_arr).all():
        raise MethodIncompatibility(
            f"{context}: X contains NaN or infinite values.",
            recovery_hint="Drop or impute non-finite HAL feature rows.",
        )
    return X_arr


def _coerce_hal_target(y: Any, *, context: str) -> np.ndarray:
    try:
        y_arr = np.asarray(y, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: y must be numeric.",
            recovery_hint="Convert the HAL target to numeric values.",
        ) from exc
    if y_arr.shape[0] == 0:
        raise DataInsufficient(
            f"{context}: y must contain at least one row.",
            recovery_hint="Pass a non-empty HAL target vector.",
        )
    if not np.isfinite(y_arr).all():
        raise MethodIncompatibility(
            f"{context}: y contains NaN or infinite values.",
            recovery_hint="Drop or impute non-finite HAL target rows.",
        )
    return y_arr


class HALRegressor(_BaseHAL):
    """L1-penalised HAL regressor (sklearn-compatible duck-typed API).

    Fits an L1-penalised regression on a main-effects HAL basis of
    per-feature step functions. Exposes the ``.fit`` / ``.predict``
    interface so it can be dropped into cross-fitting pipelines such as
    :func:`sp.tmle` and :func:`sp.hal_tmle`.

    Parameters
    ----------
    lambda_ : float, optional
        L1 penalty. ``None`` selects it via cross-validation.
    max_anchors_per_col : int, default 40
        Cap on step-function anchor points per feature.
    cv : int, default 5
        Folds for the CV penalty search (used only when ``lambda_`` is None).
    random_state : int, default 0

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3))
    >>> y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=200)
    >>> reg = sp.HALRegressor(max_anchors_per_col=10).fit(X, y)
    >>> reg.predict(X).shape
    (200,)
    """

    _estimator_type = "regressor"  # for sklearn.base.is_regressor compatibility

    def __init__(
        self,
        lambda_: Optional[float] = None,
        max_anchors_per_col: int = 40,
        cv: int = 5,
        random_state: int = 0,
    ):
        self.lambda_ = lambda_
        self.max_anchors_per_col = max_anchors_per_col
        self.cv = cv
        self.random_state = random_state
        self.n_features_in_: Optional[int] = None

    def fit(self, X: Any, y: Any) -> "HALRegressor":
        max_anchors = _validate_hal_positive_int(
            self.max_anchors_per_col,
            name="max_anchors_per_col",
        )
        cv = _validate_hal_positive_int(self.cv, name="cv")
        if cv < 2:
            raise MethodIncompatibility(
                "cv must be an integer >= 2.",
                recovery_hint="Use at least two HAL penalty-search folds.",
                diagnostics={"cv": self.cv},
            )
        if self.lambda_ is not None:
            try:
                lambda_ = float(self.lambda_)
            except (TypeError, ValueError) as exc:
                raise MethodIncompatibility(
                    "lambda_ must be a finite non-negative scalar.",
                    recovery_hint="Use lambda_=None or lambda_ >= 0.",
                    diagnostics={"lambda_": self.lambda_},
                ) from exc
            if not np.isfinite(lambda_) or lambda_ < 0:
                raise MethodIncompatibility(
                    "lambda_ must be a finite non-negative scalar.",
                    recovery_hint="Use lambda_=None or lambda_ >= 0.",
                    diagnostics={"lambda_": self.lambda_},
                )
            self.lambda_ = lambda_
        X = _coerce_hal_matrix(X, context="HALRegressor.fit()")
        y = _coerce_hal_target(y, context="HALRegressor.fit()")
        if X.shape[0] != y.shape[0]:
            raise MethodIncompatibility(
                "HALRegressor.fit(): X and y must have the same row count.",
                recovery_hint="Align HAL features and target to the same sample.",
                diagnostics={"n_x": int(X.shape[0]), "n_y": int(y.shape[0])},
            )
        if X.shape[0] < 2:
            raise DataInsufficient(
                "HALRegressor.fit() needs at least 2 observations.",
                recovery_hint="Pass at least two complete rows.",
                diagnostics={"nobs": int(X.shape[0])},
            )
        B, anchors = _hal_basis(
            X,
            anchors=None,
            max_anchors_per_col=max_anchors,
        )
        from sklearn.linear_model import Lasso, LassoCV
        from ..compat.sklearn import lasso_cv_alphas_kwargs

        if self.lambda_ is None:
            cv = int(max(2, min(cv, max(2, len(y) // 20))))
            model = LassoCV(
                cv=cv,
                random_state=self.random_state,
                max_iter=5000,
                **lasso_cv_alphas_kwargs(20),
            )
        else:
            model = Lasso(
                alpha=self.lambda_, max_iter=5000, random_state=self.random_state
            )
        model.fit(B, y)
        self._model = model
        self._anchors = anchors
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "_model") or not hasattr(self, "_anchors"):
            raise MethodIncompatibility(
                "HALRegressor.predict() requires a fitted model.",
                recovery_hint="Call fit() before predict().",
            )
        X = _coerce_hal_matrix(
            X,
            context="HALRegressor.predict()",
            expected_features=self.n_features_in_,
        )
        B, _ = _hal_basis(X, anchors=self._anchors)
        return np.asarray(self._model.predict(B))


class HALClassifier(_BaseHAL):
    """L1-penalised HAL logistic classifier (sklearn-compatible duck-typed API).

    Fits an L1-penalised logistic regression on a main-effects HAL basis of
    per-feature step functions. Exposes ``.fit`` / ``.predict`` /
    ``.predict_proba`` so it can serve as the propensity learner in
    :func:`sp.tmle` and :func:`sp.hal_tmle`.

    Parameters
    ----------
    C : float, default 1.0
        Inverse L1 penalty (larger = less shrinkage), as in scikit-learn.
    max_anchors_per_col : int, default 40
        Cap on step-function anchor points per feature.
    random_state : int, default 0

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3))
    >>> y = (X[:, 0] + rng.normal(size=200) > 0).astype(int)
    >>> clf = sp.HALClassifier(max_anchors_per_col=10).fit(X, y)
    >>> clf.predict_proba(X).shape
    (200, 2)
    >>> [int(c) for c in clf.classes_]
    [0, 1]
    """

    _estimator_type = "classifier"  # for sklearn.base.is_classifier compatibility

    def __init__(
        self,
        C: float = 1.0,
        max_anchors_per_col: int = 40,
        random_state: int = 0,
    ):
        self.C = C
        self.max_anchors_per_col = max_anchors_per_col
        self.random_state = random_state
        self.n_features_in_: Optional[int] = None

    def fit(self, X: Any, y: Any) -> "HALClassifier":
        max_anchors = _validate_hal_positive_int(
            self.max_anchors_per_col,
            name="max_anchors_per_col",
        )
        try:
            C = float(self.C)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "C must be a finite positive scalar.",
                recovery_hint="Use C > 0 for HALClassifier.",
                diagnostics={"C": self.C},
            ) from exc
        if not np.isfinite(C) or C <= 0:
            raise MethodIncompatibility(
                "C must be a finite positive scalar.",
                recovery_hint="Use C > 0 for HALClassifier.",
                diagnostics={"C": self.C},
            )
        self.C = C
        X = _coerce_hal_matrix(X, context="HALClassifier.fit()")
        y_float = _coerce_hal_target(y, context="HALClassifier.fit()")
        if X.shape[0] != y_float.shape[0]:
            raise MethodIncompatibility(
                "HALClassifier.fit(): X and y must have the same row count.",
                recovery_hint="Align HAL features and binary target.",
                diagnostics={"n_x": int(X.shape[0]), "n_y": int(y_float.shape[0])},
            )
        classes = np.unique(y_float)
        if len(classes) < 2:
            raise DataInsufficient(
                "HALClassifier.fit() needs both outcome classes.",
                recovery_hint="Provide both 0 and 1 outcomes.",
                diagnostics={"classes": classes.tolist()},
            )
        if len(classes) > 2 or not np.array_equal(classes, np.array([0.0, 1.0])):
            raise MethodIncompatibility(
                "HALClassifier.fit() expects binary y coded as 0/1.",
                recovery_hint=(
                    "Recode the negative class to 0 and positive class to 1."
                ),
                diagnostics={"classes": classes.tolist()},
            )
        y = y_float.astype(int)
        B, anchors = _hal_basis(
            X,
            anchors=None,
            max_anchors_per_col=max_anchors,
        )
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=self.C,
            max_iter=2000,
            random_state=self.random_state,
        )
        model.fit(B, y)
        self._model = model
        self._anchors = anchors
        self.classes_ = model.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "_model") or not hasattr(self, "_anchors"):
            raise MethodIncompatibility(
                "HALClassifier.predict() requires a fitted model.",
                recovery_hint="Call fit() before predict().",
            )
        X = _coerce_hal_matrix(
            X,
            context="HALClassifier.predict()",
            expected_features=self.n_features_in_,
        )
        B, _ = _hal_basis(X, anchors=self._anchors)
        return np.asarray(self._model.predict(B))

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "_model") or not hasattr(self, "_anchors"):
            raise MethodIncompatibility(
                "HALClassifier.predict_proba() requires a fitted model.",
                recovery_hint="Call fit() before predict_proba().",
            )
        X = _coerce_hal_matrix(
            X,
            context="HALClassifier.predict_proba()",
            expected_features=self.n_features_in_,
        )
        B, _ = _hal_basis(X, anchors=self._anchors)
        return np.asarray(self._model.predict_proba(B))


def hal_tmle(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: Sequence[str],
    variant: str = "delta",
    lambda_outcome: Optional[float] = None,
    C_propensity: float = 1.0,
    max_anchors_per_col: int = 40,
    n_folds: int = 5,
    estimand: str = "ATE",
    alpha: float = 0.05,
    propensity_bounds: Tuple[float, float] = (0.025, 0.975),
    random_state: int = 42,
) -> CausalResult:
    """TMLE with Highly Adaptive Lasso (HAL) nuisance learners.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Binary or continuous outcome.
    treat : str
        Binary treatment (0/1).
    covariates : list of str
    variant : {"delta"}, default "delta"
        ``"delta"`` plugs HAL into the standard TMLE targeting step. The
        ``"projection"`` variant from Li-Qiu-Wang-vdL (2025) is **not yet
        implemented** — earlier versions of this module accepted it but
        the implementation was a no-op (a heuristic ε-shrinkage that did
        not feed back into ``result.estimate``). It now raises
        :class:`NotImplementedError`. To keep the API stable while we
        port the proper Riesz-projection step, please file an issue if
        this blocks you.
    lambda_outcome : float, optional
        Outcome-model L1 penalty; None selects it via 5-fold CV.
    C_propensity : float, default 1.0
        Inverse L1 penalty for the propensity classifier (larger = less
        shrinkage).
    max_anchors_per_col : int, default 40
        Cap on the number of HAL anchor points per covariate.  The full
        cumulative-distribution anchors are used up to this cap; above it
        quantile anchors are substituted.
    n_folds : int, default 5
        Cross-fitting folds passed to :func:`sp.tmle`.
    estimand : {"ATE", "ATT"}, default "ATE"
    alpha : float, default 0.05
    propensity_bounds : tuple, default (0.025, 0.975)
        Truncation bounds for the propensity score.
    random_state : int, default 42

    Returns
    -------
    CausalResult
        Standard TMLE result object with ``.model_info['variant']`` set.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x1, x2, x3 = (rng.normal(size=n) for _ in range(3))
    >>> ps = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
    >>> d = (rng.uniform(size=n) < ps).astype(int)
    >>> y = 1.0 + 0.8 * d + x1 + 0.5 * x2 + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2, "x3": x3})
    >>> r = sp.hal_tmle(df, y="y", treat="d", covariates=["x1", "x2", "x3"],
    ...                 max_anchors_per_col=10)
    >>> bool(np.isfinite(r.estimate))
    True
    >>> r.model_info["variant"]
    'delta'

    References
    ----------
    [@benkeser2016highly] [@vanderlaan2023efficient] [@li2025regularized]
    """
    if variant == "projection":
        # The projection-variant block in v1.11.x and earlier shrunk the
        # targeting ε by an ad-hoc ``1 / (1 + log(1+max_anchors))`` factor
        # AFTER ``result.estimate`` had already been computed, so the
        # estimate was unchanged — the variant flag was effectively a
        # no-op that mutated only ``model_info["eps"]``. Rather than
        # ship a misleading variant we raise until the proper
        # Riesz-projection step (Li-Qiu-Wang-vdL 2025 §3.2) is ported.
        # ``docs/rfc/hal_tmle_projection.md`` sketches the algorithm
        # and the parity-test gate that has to clear before the variant
        # can be promoted to stable.
        raise NotImplementedError(
            "hal_tmle(variant='projection') is not yet implemented. "
            "Use variant='delta' (the standard HAL-TMLE plug-in) for "
            "production work; the v1.11.x projection code path was a "
            "no-op on the point estimate (see CHANGELOG). Roadmap and "
            "parity gates: docs/rfc/hal_tmle_projection.md. If you "
            "need this variant urgently, file an issue with the "
            "publication's headline number you'd like to match."
        )
    if variant != "delta":
        raise ValueError(
            f"variant must be 'delta' (got {variant!r}); "
            "'projection' is currently NotImplemented."
        )
    if estimand not in {"ATE", "ATT"}:
        raise ValueError("estimand must be 'ATE' or 'ATT'")

    # Lazy import to avoid circular dependency at module load.
    from .tmle import tmle as _tmle

    hal_q = HALRegressor(
        lambda_=lambda_outcome,
        max_anchors_per_col=max_anchors_per_col,
        random_state=random_state,
    )
    hal_g = HALClassifier(
        C=C_propensity,
        max_anchors_per_col=max_anchors_per_col,
        random_state=random_state,
    )

    result = _tmle(
        data=data,
        y=y,
        treat=treat,
        covariates=list(covariates),
        outcome_library=[hal_q],
        propensity_library=[hal_g],
        n_folds=n_folds,
        estimand=estimand,
        alpha=alpha,
        propensity_bounds=propensity_bounds,
        random_state=random_state,
    )
    # Record HAL-specific metadata
    result.method = f"HAL-TMLE ({variant} variant)"
    info = result.model_info or {}
    info.update(
        {
            "nuisance": "Highly Adaptive Lasso (main-effects basis only)",
            "variant": variant,
            "max_anchors_per_col": max_anchors_per_col,
            "citation": (
                "Li, Y., Qiu, S., Wang, Z. and van der Laan, M. J. (2025). "
                "Regularized Targeted Maximum Likelihood Estimation in "
                "Highly Adaptive Lasso Implied Working Models. "
                "arXiv:2506.17214."
            ),
        }
    )

    result.model_info = info
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            result,
            function="sp.tmle.hal_tmle",
            params={
                "y": y,
                "treat": treat,
                "covariates": list(covariates),
                "variant": variant,
                "lambda_outcome": lambda_outcome,
                "C_propensity": C_propensity,
                "max_anchors_per_col": max_anchors_per_col,
                "n_folds": n_folds,
                "estimand": estimand,
                "alpha": alpha,
                "propensity_bounds": list(propensity_bounds),
                "random_state": random_state,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return result
