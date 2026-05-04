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
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.results import CausalResult


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
        params = {}
        for name in inspect.signature(self.__init__).parameters:
            if name == "self":
                continue
            params[name] = getattr(self, name)
        return params

    def set_params(self, **params) -> "_BaseHAL":
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
) -> np.ndarray:
    """Main-effects HAL basis: column-wise step functions at anchor points.

    ``anchors`` is a flat 2-column array ``[[j, value], ...]`` of (feature
    index, breakpoint) pairs — emitted when first called on training data and
    reused on prediction time.  If ``anchors`` is None we generate them from
    the sorted values of each column, capped at ``max_anchors_per_col`` to
    keep the basis manageable on larger samples.
    """
    n, p = X.shape
    if anchors is None:
        cols, vals = [], []
        for j in range(p):
            xv = np.unique(X[:, j])
            if len(xv) > max_anchors_per_col:
                q = np.linspace(0, 1, max_anchors_per_col + 1)[1:-1]
                xv = np.quantile(X[:, j], q)
            for v in xv:
                cols.append(j)
                vals.append(v)
        anchors = np.column_stack([np.asarray(cols, dtype=int),
                                    np.asarray(vals, dtype=float)])

    B = np.zeros((n, anchors.shape[0]))
    for k in range(anchors.shape[0]):
        j = int(anchors[k, 0])
        v = float(anchors[k, 1])
        B[:, k] = (X[:, j] <= v).astype(float)
    return B, anchors


class HALRegressor(_BaseHAL):
    """L1-penalised HAL regressor (sklearn-compatible duck-typed API)."""

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

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        B, anchors = _hal_basis(X, anchors=None,
                                 max_anchors_per_col=self.max_anchors_per_col)
        from sklearn.linear_model import Lasso, LassoCV
        if self.lambda_ is None:
            cv = int(max(2, min(self.cv, max(2, len(y) // 20))))
            model = LassoCV(
                cv=cv, random_state=self.random_state,
                n_alphas=20, max_iter=5000,
            )
        else:
            model = Lasso(alpha=self.lambda_, max_iter=5000,
                          random_state=self.random_state)
        model.fit(B, y)
        self._model = model
        self._anchors = anchors
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        B, _ = _hal_basis(X, anchors=self._anchors)
        return self._model.predict(B)


class HALClassifier(_BaseHAL):
    """L1-penalised HAL logistic classifier (sklearn-compatible duck-typed API)."""

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

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel().astype(int)
        B, anchors = _hal_basis(X, anchors=None,
                                 max_anchors_per_col=self.max_anchors_per_col)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            penalty="l1", solver="liblinear", C=self.C,
            max_iter=2000, random_state=self.random_state,
        )
        model.fit(B, y)
        self._model = model
        self._anchors = anchors
        self.classes_ = model.classes_
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        B, _ = _hal_basis(X, anchors=self._anchors)
        return self._model.predict(B)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        B, _ = _hal_basis(X, anchors=self._anchors)
        return self._model.predict_proba(B)


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
    propensity_bounds=(0.025, 0.975),
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
    >>> r = sp.hal_tmle(df, y="y", treat="d", covariates=["x1","x2","x3"])
    >>> r.summary()
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
        data=data, y=y, treat=treat, covariates=list(covariates),
        outcome_library=[hal_q],
        propensity_library=[hal_g],
        n_folds=n_folds, estimand=estimand, alpha=alpha,
        propensity_bounds=propensity_bounds,
        random_state=random_state,
    )
    # Record HAL-specific metadata
    result.method = f"HAL-TMLE ({variant} variant)"
    info = result.model_info or {}
    info.update({
        "nuisance": "Highly Adaptive Lasso (main-effects basis only)",
        "variant": variant,
        "max_anchors_per_col": max_anchors_per_col,
        "citation": (
            "Li, Y., Qiu, S., Wang, Z. and van der Laan, M. J. (2025). "
            "Regularized Targeted Maximum Likelihood Estimation in "
            "Highly Adaptive Lasso Implied Working Models. "
            "arXiv:2506.17214."
        ),
    })

    result.model_info = info
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            result,
            function="sp.tmle.hal_tmle",
            params={
                "y": y, "treat": treat,
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
