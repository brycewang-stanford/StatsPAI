"""Causal forest with fixed effects (panel data).

``CausalForest(fe="unit" | "twoway")`` estimates heterogeneous effects in

    Y_it = alpha_i + gamma_t + tau(X_it) D_it + e_it

without letting the fixed effects masquerade as heterogeneity.  A pooled
forest (or one run on globally demeaned data) splits on covariates that are
correlated with ``alpha_i`` and reports level differences as effect
differences; and global two-way demeaning of ``tau(X_i) D_it`` mixes the
effects of different units, so the residualized regression is misspecified
whenever ``tau`` varies.  Removing the fixed effects *within each node* --
where units are similar in ``X`` and ``tau`` is close to constant -- avoids
both problems (Kattenberg, Scheer and Thiel 2023, Sec. 3).

Implementation choices (all documented, none copied from other code):

* Splitting uses the GRF gradient criterion on the node-residualized
  outcome and treatment, with the residualized treatment as the instrument
  of the stabilised splitting rule.
* Leaves are residualized again on their honest estimation sample and
  store the within moments ``sum Y~ D~`` and ``sum D~^2``; predictions are
  the ratio of their tree averages, so the GRF prediction and little-bag
  variance apply unchanged.  A leaf whose residualized treatment has no
  variation is treated as uninformative (excluded), never as a zero effect.
* The two-way within transformation iterates alternating projections to a
  tolerance, so unbalanced nodes are exact up to ``1e-10`` rather than an
  arbitrary fixed number of sweeps.
* Trees are sampled, split into honest halves and cross-fitted by unit
  (or by a coarser cluster that nests units).
* ``Y`` and ``D`` are first centred by out-of-bag regression forests on
  the covariates, as in the authors' implementation; for time-invariant
  covariates this centring is absorbed by the unit effect.

The identifying assumption is conditional parallel trends with no
anticipation; staggered adoption with heterogeneous dynamic effects should
use :func:`statspai.did_forest`, which builds clean group-time comparisons.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..exceptions import DataInsufficient, MethodIncompatibility
from . import _grf_engine as engine
from ._grf_family import with_stream
from ._grf_fit import (
    _crossfit,
    _nuisance_mtry,
    _validate_nuisance_vector,
    observation_weights,
    warn_missing_oob,
)


def _codes(values: Any, name: str, n: int) -> np.ndarray:
    arr = np.asarray(values).ravel()
    if arr.size != n:
        raise MethodIncompatibility(
            f"CausalForest.fit(): {name} must have one id per row.",
            recovery_hint=f"Pass {name} aligned with Y (length {n}).",
            diagnostics={f"n_{name}": int(arr.size), "n": int(n)},
        )
    if arr.dtype.kind in "fc" and not np.isfinite(arr.astype(float)).all():
        raise MethodIncompatibility(
            f"CausalForest.fit(): {name} contains missing values.",
            recovery_hint=f"Drop rows with a missing {name} id.",
        )
    if arr.dtype == object and any(v is None for v in arr):
        raise MethodIncompatibility(
            f"CausalForest.fit(): {name} contains missing values.",
            recovery_hint=f"Drop rows with a missing {name} id.",
        )
    _, codes = np.unique(arr, return_inverse=True)
    return codes.astype(np.int64).ravel()


def _within_sum_of_squares(values: np.ndarray, groups: np.ndarray) -> float:
    counts = np.bincount(groups)
    sums = np.bincount(groups, weights=values)
    means = sums / np.maximum(counts, 1)
    resid = values - means[groups]
    return float(resid @ resid)


def _twoway_within_share(
    values: np.ndarray, unit: np.ndarray, time: np.ndarray
) -> float:
    """Share of the variation in ``values`` left by unit and period effects.

    Residual sum of squares of the least-squares projection on unit and
    period dummies (exact on unbalanced panels too), over the centred total
    sum of squares.
    """
    from scipy import sparse
    from scipy.sparse.linalg import lsqr

    n = values.size
    centred = values - values.mean()
    total = float(centred @ centred)
    if total <= 0.0:
        return 0.0
    n_units = int(unit.max()) + 1
    rows = np.concatenate([np.arange(n), np.arange(n)])
    cols = np.concatenate([unit, n_units + time])
    dummies = sparse.csr_matrix(
        (np.ones(2 * n), (rows, cols)), shape=(n, n_units + int(time.max()) + 1)
    )
    resid = centred - dummies @ lsqr(dummies, centred, atol=1e-14, btol=1e-14)[0]
    return float(resid @ resid) / total


def fit_fe(
    cf: Any,
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    W: Optional[np.ndarray],
    *,
    clusters: Any,
    Y_hat: Any,
    W_hat: Any,
    unit: Any,
    time: Any,
) -> Dict[str, Any]:
    n = Y.size
    if unit is None:
        raise MethodIncompatibility(
            f"CausalForest(fe={cf.fe!r}).fit() requires unit= ids.",
            recovery_hint="Pass unit=<panel unit ids> (and time= for 'twoway').",
        )
    unit_codes = _codes(unit, "unit", n)
    if cf.fe == "twoway":
        if time is None:
            raise MethodIncompatibility(
                "CausalForest(fe='twoway').fit() requires time= ids.",
                recovery_hint="Pass time=<period ids>, or use fe='unit'.",
            )
        time_codes = _codes(time, "time", n)
    else:
        time_codes = np.zeros(n, dtype=np.int64)

    n_units = int(unit_codes.max()) + 1
    if n_units < 2:
        raise DataInsufficient(
            "CausalForest(fe=...) needs at least two units.",
            recovery_hint="Pass a panel with more than one unit.",
        )
    if cf.fe == "twoway":
        dup = np.unique(unit_codes * (int(time_codes.max()) + 1) + time_codes)
        if dup.size != n:
            raise MethodIncompatibility(
                "CausalForest(fe='twoway'): (unit, time) pairs must be unique.",
                recovery_hint=(
                    "Aggregate to one row per unit-period, or use fe='unit' "
                    "for repeated observations within a unit."
                ),
                diagnostics={"n_rows": int(n), "n_unique_pairs": int(dup.size)},
            )

    if clusters is None:
        cluster_codes = unit_codes
    else:
        cluster_codes = _codes(clusters, "clusters", n)
        per_unit = np.full(n_units, -1, dtype=np.int64)
        for u, c in zip(unit_codes, cluster_codes):
            if per_unit[u] == -1:
                per_unit[u] = c
            elif per_unit[u] != c:
                raise MethodIncompatibility(
                    "CausalForest(fe=...): every unit must belong to a single "
                    "cluster (clusters must nest units).",
                    recovery_hint=(
                        "Cluster at the unit level (the default) or at a level "
                        "that contains whole units."
                    ),
                )
        if int(cluster_codes.max()) + 1 < 2:
            raise DataInsufficient(
                "CausalForest(fe=...): clustered forests need at least two "
                "clusters.",
                recovery_hint="Pass clusters with more than one distinct id.",
            )

    # Treatment must vary within units (and, for two-way FE, net of periods)
    # or the within estimator has nothing to identify the effect from.
    within_ss = _within_sum_of_squares(T.astype(float), unit_codes)
    if within_ss <= 1e-10:
        raise DataInsufficient(
            "CausalForest(fe=...): the treatment does not vary within units, "
            "so unit fixed effects absorb it entirely.",
            recovery_hint=(
                "Use a pooled forest with clusters=unit, or a design where "
                "units switch treatment status."
            ),
        )
    if cf.fe == "twoway":
        # Net of unit *and* period effects. A treatment that is a function
        # of the period alone (every unit adopts at the same date) varies
        # within units but is absorbed by the period effects, and the forest
        # would be fitted to rounding noise. The test is relative to the
        # treatment's own variation: an absorbed treatment leaves a share at
        # rounding level (1e-20 and below), while one cell out of line in a
        # panel of n rows leaves about 1 / (n var(T)), so 1e-10 separates
        # the two for any panel that fits in memory.
        share = _twoway_within_share(T.astype(float), unit_codes, time_codes)
        if share <= 1e-10:
            raise DataInsufficient(
                "CausalForest(fe='twoway'): the treatment does not vary net "
                "of unit and period effects (it is a function of the period "
                "within every unit, e.g. all units adopt at the same date), "
                "so the fixed effects absorb it entirely and the effect is "
                "not identified.",
                recovery_hint=(
                    "Use a design in which units are treated at different "
                    "dates or some are never treated; fe='unit' identifies "
                    "the effect only by assuming there are no period effects."
                ),
                diagnostics={"twoway_within_share_of_variation": share},
            )
    unit_has_variation = np.zeros(n_units, dtype=bool)
    t_by_unit_min = np.full(n_units, np.inf)
    t_by_unit_max = np.full(n_units, -np.inf)
    np.minimum.at(t_by_unit_min, unit_codes, T)
    np.maximum.at(t_by_unit_max, unit_codes, T)
    unit_has_variation = t_by_unit_max > t_by_unit_min

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
    features = X if W is None else np.hstack([X, W])
    nuisance_trees = max(50, int(cf.n_estimators) // 4)
    source: Dict[str, str] = {}
    for name, target, user_flag, model, arr in (
        ("Y_hat", Y, cf._user_model_y, cf.model_y, y_hat_arr),
        ("W_hat", T.astype(float), cf._user_model_t, cf.model_t, w_hat_arr),
    ):
        if arr is not None:
            source[name] = "user-supplied"
            continue
        if user_flag:
            fitted = _crossfit(
                model,
                features,
                target,
                cluster_codes,
                cf.nuisance_folds,
                cf.random_state,
                proba=False,
            )
            source[name] = f"cross-fitted {type(model).__name__}"
        else:
            forest = engine.train_forest(
                features,
                target,
                kind=engine.KIND_REGRESSION,
                num_trees=nuisance_trees,
                mtry=_nuisance_mtry(cf.mtry, features.shape[1]),
                min_node_size=5,
                ci_group_size=1,
                **with_stream(common, name),
            )
            fitted, _ = forest.predict_oob(features)
            if not np.isfinite(fitted).all():
                raise DataInsufficient(
                    f"CausalForest.fit(): the {name} nuisance forest left rows "
                    "without an out-of-bag prediction.",
                    recovery_hint=(
                        "Increase n_estimators, or pass precomputed Y_hat / W_hat."
                    ),
                )
            source[name] = "grf regression forest (OOB)"
        if name == "Y_hat":
            y_hat_arr = fitted
        else:
            w_hat_arr = fitted
    assert y_hat_arr is not None and w_hat_arr is not None

    forest = engine.train_forest(
        X,
        Y - y_hat_arr,
        T - w_hat_arr,
        kind=engine.KIND_CAUSAL_FE,
        num_trees=int(cf.n_estimators),
        mtry=cf.mtry,
        min_node_size=int(cf.min_samples_leaf),
        stabilize_splits=bool(cf.stabilize_splits),
        ci_group_size=int(cf.ci_group_size),
        max_depth=cf.max_depth,
        unit=unit_codes,
        time=time_codes,
        tau_split=str(cf.split_rule).lower().strip() == "cffe",
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
    cf._fe_unit = unit_codes
    cf._fe_time = time_codes
    cf._fe_W = None if W is None else np.asarray(W, dtype=float)
    cf._fe_imputation = None  # imputation designs are rebuilt on refit
    cf._observation_weight = observation_weights(
        cluster_codes, bool(cf.equalize_cluster_weights), n
    )
    leaves = forest.left == -1
    return {
        "n_rows_without_oob_prediction": n_missing,
        "nuisance_source": source,
        "n_clusters": int(cluster_codes.max()) + 1,
        "n_units": n_units,
        "n_periods": int(time_codes.max()) + 1 if cf.fe == "twoway" else None,
        "share_units_switching_treatment": float(unit_has_variation.mean()),
        "split_rule": str(cf.split_rule).lower().strip(),
        "num_trees": forest.num_trees,
        "mtry": forest.options.get("mtry_value"),
        "uninformative_leaf_share": (
            float(1.0 - np.mean(forest.leaf_nonempty[leaves]))
            if np.any(leaves)
            else 0.0
        ),
    }
