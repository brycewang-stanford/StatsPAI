"""Linear-model forest (``grf::lm_forest``) and its multi-arm specialisation.

The forest estimates the coefficients of the conditionally linear model

.. math::

    Y = c(x) + h_1(x) W_1 + \\dots + h_K(x) W_K + \\varepsilon,

(``Y`` possibly vector-valued) by the forest-weighted local least squares
of the nuisance-centred ``Y`` on the nuisance-centred ``W`` -- the
multivariate R-learner of Nie and Wager (2021) with GRF kernel weights.
Trees split on the vector-valued gradient pseudo-outcome of all
coefficients (ATW 2019, eq. 20).  With ``W`` the arm indicators of a
``K``-arm treatment this is the multi-arm causal forest
(:func:`statspai.multi_arm_forest`).

References
----------
[@athey2019generalized], [@nie2021quasi]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from ..exceptions import MethodIncompatibility
from . import _grf_engine as engine
from ._grf_family import (
    ForestOptions,
    GRFFamilyForest,
    cluster_codes,
    require_finite,
    resolve_inputs,
    sample_weights,
    user_nuisance,
)


class LMForestResult(GRFFamilyForest):
    """A fitted linear-model forest.

    Attributes
    ----------
    coefficients : np.ndarray
        Out-of-bag ``h_k(X_i)``, shape ``(n, K, q)``.
    coefficient_variance : np.ndarray or None
        Little-bag variances, ``(n, K, q)`` (single outcome only).
    regressor_names, outcome_names : list of str

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(500, 2)); W = rng.normal(size=(500, 1))
    >>> Y = (1 + X[:, 0]) * W[:, 0] + rng.normal(size=500)
    >>> sp.lm_forest(y=Y, regressors=W, covariates=X,
    ...              n_estimators=100).regressor_names
    ['regressors']
    """

    _context = "lm_forest"

    def __init__(self) -> None:
        self.coefficients = np.zeros(0)
        self.coefficient_variance: Optional[np.ndarray] = None
        self.regressor_names: List[str] = []
        self.outcome_names: List[str] = []
        self.feature_names: List[str] = []
        self.n_obs = 0
        self.detail: Dict[str, Any] = {}

    def _reshape(self, flat: np.ndarray) -> np.ndarray:
        k = len(self.regressor_names)
        q = len(self.outcome_names)
        return np.asarray(flat, dtype=float).reshape(flat.shape[0], k, q)

    def predict(
        self, newdata: Any = None, estimate_variance: bool = False
    ) -> Dict[str, np.ndarray]:
        """``{"predictions": (m, K, q)[, "variance_estimates": (m, K, q)]}``."""
        if newdata is None:
            pred = self.coefficients
            var = self.coefficient_variance
        else:
            if estimate_variance:
                self._require_little_bags()
            p, v = self._engine.predict(
                self._new_X(newdata), estimate_variance=bool(estimate_variance)
            )
            pred = self._reshape(p)
            var = self._reshape(v)
        out = {"predictions": pred}
        if estimate_variance:
            if var is None:
                raise MethodIncompatibility(
                    f"{self._context}: variance estimates need ci_group_size >= 2.",
                    recovery_hint="Refit with ci_group_size=2.",
                )
            out["variance_estimates"] = var
        return out

    def summary(self) -> str:
        lines = [
            "Linear-model forest (GRF engine)",
            f"  N      : {self.n_obs}",
            f"  trees  : {self.num_trees}",
        ]
        for b, rn in enumerate(self.regressor_names):
            for a, on in enumerate(self.outcome_names):
                h = self.coefficients[:, b, a]
                lines.append(
                    f"  h[{rn} -> {on}] (OOB): mean {np.mean(h):.4f}, "
                    f"sd {np.std(h):.4f}"
                )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"LMForestResult(K={len(self.regressor_names)}, "
            f"q={len(self.outcome_names)}, n={self.n_obs})"
        )


def fit_multi_causal(
    context: str,
    X: np.ndarray,
    Yc: np.ndarray,
    Wc: np.ndarray,
    opts: ForestOptions,
    common: Dict[str, Any],
) -> Any:
    """Train the multi-causal forest on centred ``Y`` ``(n, q)`` and ``W``
    ``(n, K)``; returns ``(forest, coef (n,K,q), var (n,K,q) or None)``."""
    n, q = Yc.shape
    k = Wc.shape[1]
    forest = engine.train_forest(
        X,
        np.zeros(n),
        kind=engine.KIND_MULTI_CAUSAL,
        num_trees=opts.n_estimators,
        mtry=opts.mtry,
        min_node_size=int(opts.min_samples_leaf),
        stabilize_splits=bool(opts.stabilize_splits),
        ci_group_size=int(opts.ci_group_size),
        max_depth=opts.max_depth,
        M=np.ascontiguousarray(np.column_stack([Yc, Wc])),
        params=np.array([float(q), float(k)]),
        **common,
    )
    want_var = opts.ci_group_size > 1 and q == 1
    pred, var = forest.predict_oob(X, estimate_variance=want_var)
    require_finite(pred, "coefficient", context)
    coef = pred.reshape(n, k, q)
    return forest, coef, (var.reshape(n, k, q) if want_var else None)


def oob_multi_regression(
    X: np.ndarray,
    Ymat: np.ndarray,
    opts: ForestOptions,
    common: Dict[str, Any],
    name: str,
    context: str,
) -> np.ndarray:
    """Out-of-bag ``E[Y | X]`` for each column of ``Ymat`` from one
    multi-task regression forest (``max(50, n_estimators // 4)`` trees)."""
    from ._grf_family import nuisance_trees, with_stream

    mtry = None if opts.mtry is None else int(min(max(int(opts.mtry), 1), X.shape[1]))
    if Ymat.shape[1] == 1:
        forest = engine.train_forest(
            X,
            Ymat[:, 0],
            kind=engine.KIND_REGRESSION,
            num_trees=nuisance_trees(opts.n_estimators),
            mtry=mtry,
            min_node_size=5,
            ci_group_size=1,
            **with_stream(common, name),
        )
        pred, _ = forest.predict_oob(X)
        pred = np.asarray(pred, dtype=float)[:, None]
    else:
        forest = engine.train_forest(
            X,
            np.zeros(X.shape[0]),
            kind=engine.KIND_MULTI_REG,
            num_trees=nuisance_trees(opts.n_estimators),
            mtry=mtry,
            min_node_size=5,
            ci_group_size=1,
            M=np.ascontiguousarray(Ymat),
            params=np.array([float(Ymat.shape[1])]),
            **with_stream(common, name),
        )
        pred, _ = forest.predict_oob(X)
    require_finite(pred, name, context)
    return np.asarray(pred, dtype=float)


@accepts_aliases(_strict=True, n_trees="n_estimators", min_leaf="min_samples_leaf")
def lm_forest(
    data: Optional[pd.DataFrame] = None,
    y: Any = None,
    regressors: Any = None,
    covariates: Any = None,
    *,
    clusters: Any = None,
    weights: Any = None,
    equalize_cluster_weights: bool = False,
    Y_hat: Any = None,
    W_hat: Any = None,
    n_estimators: int = 2000,
    min_samples_leaf: int = 5,
    max_samples: float = 0.5,
    mtry: Optional[int] = None,
    honest: bool = True,
    honesty_fraction: float = 0.5,
    honesty_prune_leaves: bool = True,
    split_alpha: float = 0.05,
    imbalance_penalty: float = 0.0,
    stabilize_splits: bool = False,
    ci_group_size: int = 2,
    max_depth: Optional[int] = None,
    random_state: Optional[int] = 42,
    n_jobs: int = 1,
) -> LMForestResult:
    """
    Linear-model forest: covariate-varying coefficients ``h_k(x)`` in
    ``Y = c(x) + sum_k h_k(x) W_k`` (``grf::lm_forest``).

    Parameters
    ----------
    data : pd.DataFrame, optional
        Input data; when omitted, the other inputs are arrays.
    y : str, list of str or array-like
        Outcome(s); several outcomes are modelled jointly.
    regressors : str, list of str or array-like
        The regressors ``W_1..W_K`` whose coefficients vary with ``x``.
    covariates : str, list of str or 2-D array
        The covariates ``X`` along which the coefficients vary.
    clusters, weights, equalize_cluster_weights
        As in :func:`statspai.causal_forest`.
    Y_hat, W_hat : array-like, optional
        Precomputed ``E[Y|X]`` ``(n, q)`` and ``E[W|X]`` ``(n, K)``; by
        default out-of-bag (multi-task) regression forests.
    n_estimators, min_samples_leaf, max_samples, mtry, honest,
    honesty_fraction, honesty_prune_leaves, split_alpha, imbalance_penalty,
    ci_group_size, max_depth, random_state, n_jobs
        Forest options (grf names: ``num.trees``, ``min.node.size``,
        ``sample.fraction``, ..., ``alpha``).
    stabilize_splits : bool, default False
        Apply the causal-forest split constraints to every regressor.

    Returns
    -------
    LMForestResult
        ``.coefficients`` ``(n, K, q)`` out-of-bag, ``.predict(newdata)``,
        ``.variable_importance()``.

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 600; X = rng.normal(size=(n, 3)); W = rng.normal(size=(n, 2))
    >>> Y = X[:, 1] + (1 + X[:, 0]) * W[:, 0] - W[:, 1] + rng.normal(size=n)
    >>> lmf = sp.lm_forest(y=Y, regressors=W, covariates=X, n_estimators=200)
    >>> lmf.coefficients.shape
    (600, 2, 1)

    References
    ----------
    [@athey2019generalized], [@nie2021quasi]
    """
    ctx = "lm_forest"
    opts = ForestOptions(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_samples=max_samples,
        mtry=mtry,
        honest=honest,
        honesty_fraction=honesty_fraction,
        honesty_prune_leaves=honesty_prune_leaves,
        split_alpha=split_alpha,
        imbalance_penalty=imbalance_penalty,
        stabilize_splits=stabilize_splits,
        ci_group_size=ci_group_size,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    opts.validate(ctx)
    arrays, names, keep, n_input = resolve_inputs(
        ctx,
        data,
        {"y": y, "regressors": regressors, "covariates": covariates},
        {"clusters": clusters, "weights": weights},
    )
    X = arrays["covariates"]
    Y = arrays["y"]
    W = arrays["regressors"]
    n = X.shape[0]
    cl = cluster_codes(arrays.get("clusters"), ctx)
    sw = sample_weights(arrays.get("weights"), equalize_cluster_weights, ctx)
    common = opts.engine_kwargs(cl, equalize_cluster_weights, sw)
    y_hat = (
        oob_multi_regression(X, Y, opts, common, "Y_hat", ctx)
        if Y_hat is None
        else user_nuisance(_rows(Y_hat, keep, n_input), n, Y.shape[1], "Y_hat", ctx)
    )
    w_hat = (
        oob_multi_regression(X, W, opts, common, "W_hat", ctx)
        if W_hat is None
        else user_nuisance(_rows(W_hat, keep, n_input), n, W.shape[1], "W_hat", ctx)
    )
    forest, coef, var = fit_multi_causal(ctx, X, Y - y_hat, W - w_hat, opts, common)
    res = LMForestResult()
    res._engine = forest
    res._X = X
    res._Y, res._W, res._y_hat, res._w_hat = Y, W, y_hat, w_hat
    res.coefficients = coef
    res.coefficient_variance = var
    res.regressor_names = names["regressors"]
    res.outcome_names = names["y"]
    res.feature_names = names["covariates"]
    res.n_obs = n
    res.detail = {
        "nuisance_source": {
            "Y_hat": (
                "user-supplied" if Y_hat is not None else "regression forest (OOB)"
            ),
            "W_hat": (
                "user-supplied" if W_hat is not None else "regression forest (OOB)"
            ),
        },
        "n_dropped_missing": int(n_input - n),
        "n_clusters": None if cl is None else int(cl.max()) + 1,
        "options": dict(opts.__dict__),
    }
    return res


def _rows(values: Any, keep: np.ndarray, n_input: int) -> np.ndarray:
    """Align a user nuisance given for the input rows with the kept rows."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim >= 1 and arr.shape[0] == n_input and n_input != int(keep.sum()):
        return np.asarray(arr[keep], dtype=float)
    return arr


__all__ = ["lm_forest", "LMForestResult", "fit_multi_causal", "oob_multi_regression"]
