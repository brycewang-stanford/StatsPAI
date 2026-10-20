"""Default (GRF) fitting path for :class:`statspai.forest.CausalForest`.

``grf::causal_forest`` fits two regression forests for the nuisances
``Y.hat = E[Y | X]`` and ``W.hat = E[W | X]`` (``num.trees = max(50,
num.trees / 4)``, ``ci.group.size = 1``, ``min.node.size = 5``, the same
seed, sampling options and clusters), takes their out-of-bag predictions,
and trains the causal forest on ``Y - Y.hat`` and ``W - W.hat``.  This
module reproduces that pipeline on top of :mod:`._grf_engine` and stores
everything the inference helpers need (OOB CATEs and their variances,
nuisances, cluster codes, observation weights).

User-supplied nuisances are honoured in the two ways grf allows:
precomputed ``Y_hat`` / ``W_hat`` vectors, or scikit-learn estimators,
which are cross-fitted with folds that never split a cluster.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..exceptions import DataInsufficient, MethodIncompatibility
from . import _grf_engine as engine
from ._grf_family import with_stream


def _validate_nuisance_vector(values: Any, name: str, n: int) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 1:
        arr = np.full(n, float(arr[0]))
    if arr.size != n:
        raise MethodIncompatibility(
            f"CausalForest.fit(): {name} must have one value per row.",
            recovery_hint=f"Pass {name} aligned with Y (length {n}) or a scalar.",
            diagnostics={f"n_{name}": int(arr.size), "n": int(n)},
        )
    if not np.isfinite(arr).all():
        raise MethodIncompatibility(
            f"CausalForest.fit(): {name} contains NaN or infinite values.",
            recovery_hint=f"Pass finite {name} predictions.",
        )
    return arr


def _validate_clusters(clusters: Any, n: int) -> Optional[np.ndarray]:
    if clusters is None:
        return None
    arr = np.asarray(clusters).ravel()
    if arr.size != n:
        raise MethodIncompatibility(
            "CausalForest.fit(): clusters must have one label per row.",
            recovery_hint=f"Pass a cluster id vector of length {n}.",
            diagnostics={"n_clusters_labels": int(arr.size), "n": int(n)},
        )
    if arr.dtype.kind in "fc" and not np.isfinite(arr.astype(float)).all():
        raise MethodIncompatibility(
            "CausalForest.fit(): clusters contain missing values.",
            recovery_hint="Drop rows with a missing cluster id before fitting.",
        )
    if arr.dtype == object and any(v is None for v in arr):
        raise MethodIncompatibility(
            "CausalForest.fit(): clusters contain missing values.",
            recovery_hint="Drop rows with a missing cluster id before fitting.",
        )
    _, codes = np.unique(arr, return_inverse=True)
    n_clusters = int(codes.max()) + 1
    if n_clusters < 2:
        raise DataInsufficient(
            "CausalForest.fit(): clustered forests need at least two clusters.",
            recovery_hint="Pass clusters with more than one distinct id.",
            diagnostics={"n_clusters": n_clusters},
        )
    return codes.astype(np.int64).ravel()


def observation_weights(
    clusters: Optional[np.ndarray], equalize_cluster_weights: bool, n: int
) -> np.ndarray:
    """``grf:::observation_weights`` (no sample weights), normalised to 1."""
    if clusters is None or not equalize_cluster_weights:
        raw = np.ones(n)
    else:
        counts = np.bincount(clusters)
        raw = 1.0 / counts[clusters]
    return np.asarray(raw / raw.sum(), dtype=float)


def _crossfit(
    model: Any,
    features: np.ndarray,
    target: np.ndarray,
    clusters: Optional[np.ndarray],
    folds: int,
    seed: Optional[int],
    proba: bool,
) -> np.ndarray:
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold, KFold

    n = target.size
    out = np.empty(n)
    if clusters is not None:
        n_groups = int(np.unique(clusters).size)
        splitter: Any = GroupKFold(n_splits=min(folds, n_groups))
        split_iter = splitter.split(features, target, groups=clusters)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(features, target)
    for train_idx, test_idx in split_iter:
        fitted = clone(model).fit(features[train_idx], target[train_idx])
        if proba:
            from ..core._learners import positive_class_proba

            # (n,) or (n, 2) by classes_; 0 when the fold held no treated rows
            out[test_idx] = positive_class_proba(
                fitted, features[test_idx], context="causal_forest model_t"
            )
        else:
            out[test_idx] = np.asarray(fitted.predict(features[test_idx])).ravel()
    return out


def fit_grf(
    cf: Any,
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    W: Optional[np.ndarray],
    *,
    clusters: Any = None,
    Y_hat: Any = None,
    W_hat: Any = None,
) -> Dict[str, Any]:
    """Fit ``cf`` (a :class:`CausalForest`) the way ``grf::causal_forest`` does.

    Mutates ``cf`` with the engine forest and the stored inference inputs;
    returns a diagnostics dictionary.
    """
    n = Y.size
    cluster_codes = _validate_clusters(clusters, n)
    y_hat_arr = _validate_nuisance_vector(Y_hat, "Y_hat", n)
    w_hat_arr = _validate_nuisance_vector(W_hat, "W_hat", n)

    seed = 0 if cf.random_state is None else int(cf.random_state)
    common: Dict[str, Any] = dict(
        clusters=cluster_codes,
        equalize_cluster_weights=bool(cf.equalize_cluster_weights),
        sample_fraction=float(cf.max_samples),
        honesty=bool(cf.honest),
        honesty_fraction=float(cf.honesty_fraction),
        honesty_prune_leaves=bool(cf.honesty_prune_leaves),
        alpha=float(cf.alpha),
        imbalance_penalty=float(cf.imbalance_penalty),
        seed=seed,
        n_jobs=cf.n_jobs,
    )
    nuisance_features = X if W is None else np.hstack([X, W])
    nuisance_trees = max(50, int(cf.n_estimators) // 4)
    nuisance_source = {}

    if y_hat_arr is None:
        if cf._user_model_y:
            y_hat_arr = _crossfit(
                cf.model_y,
                nuisance_features,
                Y,
                cluster_codes,
                cf.nuisance_folds,
                cf.random_state,
                proba=False,
            )
            nuisance_source["Y_hat"] = f"cross-fitted {type(cf.model_y).__name__}"
        else:
            forest_y = engine.train_forest(
                nuisance_features,
                Y,
                kind=engine.KIND_REGRESSION,
                num_trees=nuisance_trees,
                mtry=_nuisance_mtry(cf.mtry, nuisance_features.shape[1]),
                min_node_size=5,
                ci_group_size=1,
                **with_stream(common, "Y_hat"),
            )
            y_hat_arr, _ = forest_y.predict_oob(nuisance_features)
            nuisance_source["Y_hat"] = "grf regression forest (OOB)"
    else:
        nuisance_source["Y_hat"] = "user-supplied"

    if w_hat_arr is None:
        if cf._user_model_t:
            y_proba = bool(cf.discrete_treatment) and hasattr(
                cf.model_t, "predict_proba"
            )
            w_hat_arr = _crossfit(
                cf.model_t,
                nuisance_features,
                T,
                cluster_codes,
                cf.nuisance_folds,
                cf.random_state,
                proba=y_proba,
            )
            nuisance_source["W_hat"] = f"cross-fitted {type(cf.model_t).__name__}"
        else:
            forest_w = engine.train_forest(
                nuisance_features,
                T,
                kind=engine.KIND_REGRESSION,
                num_trees=nuisance_trees,
                mtry=_nuisance_mtry(cf.mtry, nuisance_features.shape[1]),
                min_node_size=5,
                ci_group_size=1,
                **with_stream(common, "W_hat"),
            )
            w_hat_arr, _ = forest_w.predict_oob(nuisance_features)
            nuisance_source["W_hat"] = "grf regression forest (OOB)"
    else:
        nuisance_source["W_hat"] = "user-supplied"

    for name, arr in (("Y_hat", y_hat_arr), ("W_hat", w_hat_arr)):
        if not np.isfinite(arr).all():
            raise DataInsufficient(
                f"CausalForest.fit(): the {name} nuisance forest left rows "
                "without an out-of-bag prediction.",
                recovery_hint=(
                    "Increase n_estimators so every row is out-of-bag for "
                    "some nuisance tree, or pass precomputed Y_hat / W_hat."
                ),
                diagnostics={"n_missing": int(np.sum(~np.isfinite(arr)))},
            )

    forest = engine.train_forest(
        X,
        Y - y_hat_arr,
        T - w_hat_arr,
        kind=engine.KIND_CAUSAL,
        num_trees=int(cf.n_estimators),
        mtry=cf.mtry,
        min_node_size=int(cf.min_samples_leaf),
        stabilize_splits=bool(cf.stabilize_splits),
        ci_group_size=int(cf.ci_group_size),
        max_depth=cf.max_depth,
        **common,
    )
    tau_oob, var_oob = forest.predict_oob(X, estimate_variance=cf.ci_group_size > 1)
    n_missing = warn_missing_oob(tau_oob)

    cf._engine = forest
    cf._forest = None
    cf._m_insample = y_hat_arr
    cf._e_insample = w_hat_arr
    cf._oob_tau = tau_oob
    cf._oob_var = var_oob if cf.ci_group_size > 1 else None
    cf._clusters = cluster_codes
    cf._observation_weight = observation_weights(
        cluster_codes, bool(cf.equalize_cluster_weights), n
    )
    return {
        "n_rows_without_oob_prediction": n_missing,
        "nuisance_source": nuisance_source,
        "n_clusters": None if cluster_codes is None else int(cluster_codes.max() + 1),
        "num_trees": forest.num_trees,
        "mtry": forest.options.get("mtry_value"),
        "empty_leaf_share": (
            float(1.0 - np.mean(forest.leaf_nonempty[forest.left == -1]))
            if np.any(forest.left == -1)
            else 0.0
        ),
    }


def warn_missing_oob(tau_oob: np.ndarray) -> int:
    """Warn (as grf returns NaN) when some rows have no OOB prediction."""
    import warnings

    from ..exceptions import AssumptionWarning

    n_missing = int(np.sum(~np.isfinite(tau_oob)))
    if n_missing:
        warnings.warn(
            f"CausalForest: {n_missing} training row(s) are in the subsample "
            "of every tree (or only reach uninformative leaves), so they have "
            "no out-of-bag prediction. predict() returns NaN for them and "
            "in-sample inference (ATE, calibration, RATE, BLP) will refuse to "
            "run. Increase n_estimators.",
            AssumptionWarning,
            stacklevel=4,
        )
    return n_missing


def _nuisance_mtry(mtry: Optional[int], p: int) -> Optional[int]:
    """grf passes the causal forest's ``mtry`` to the nuisance forests."""
    if mtry is None:
        return None
    return int(min(max(int(mtry), 1), p))
