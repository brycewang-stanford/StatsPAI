"""
Cluster Synthetic Control Method.

Clusters donor units by pre-treatment trajectory similarity, then builds
SCM within the best-matching cluster.  This reduces the donor pool to the
most relevant units, improving pre-treatment fit and reducing interpolation
bias.

Three clustering strategies are supported:

* **kmeans** (default) — K-means on standardised pre-treatment outcomes.
* **spectral** — Spectral clustering on the correlation matrix.
* **hierarchical** — Agglomerative clustering with Ward linkage.

The number of clusters can be set manually or selected automatically via
silhouette scores.  An optional *augment* mode adds the best out-of-cluster
donors (by Euclidean distance) to the selected cluster.

References
----------
Rho, S. (2024). "Cluster Synthetic Control Methods."
Billmeier, A. and Nannicini, T. (2013). "Assessing Economic Liberalization
    Episodes: A Synthetic Control Approach."
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

# sklearn is imported lazily inside the helpers that need it so that
# ``import statspai`` doesn't pull ~245 sklearn submodules through this
# file when the user never touches cluster_synth.

from ..core.results import CausalResult

__all__ = ["cluster_synth"]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cluster_synth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    n_clusters: Optional[int] = None,
    cluster_method: str = "kmeans",
    augment: bool = False,
    max_augment: int = 3,
    covariates: Optional[List[str]] = None,
    placebo: bool = True,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> CausalResult:
    """
    Cluster Synthetic Control estimator.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    outcome : str
        Name of the outcome column.
    unit : str
        Name of the unit-identifier column.
    time : str
        Name of the time-period column.
    treated_unit : any
        Identifier of the single treated unit.
    treatment_time : any
        First treatment period (inclusive).
    n_clusters : int or None
        Number of clusters.  ``None`` selects automatically via silhouette
        score (k from 2 to min(J-1, 10)).
    cluster_method : {'kmeans', 'spectral', 'hierarchical'}
        Clustering algorithm.
    augment : bool
        If ``True``, augment the selected cluster with the closest donors
        from other clusters.
    max_augment : int
        Maximum number of additional donors when *augment* is ``True``.
    covariates : list of str or None
        Additional columns to include in the clustering feature matrix.
    placebo : bool
        Run in-space placebo permutation inference.
    alpha : float
        Significance level for confidence intervals.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    CausalResult
    """
    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    _validate_inputs(data, outcome, unit, time, treated_unit, treatment_time,
                     cluster_method)

    # ------------------------------------------------------------------
    # 2. Reshape to wide panel
    # ------------------------------------------------------------------
    panel = data.pivot_table(index=unit, columns=time, values=outcome)
    all_times = sorted(panel.columns)
    pre_times = [t for t in all_times if t < treatment_time]
    post_times = [t for t in all_times if t >= treatment_time]

    if len(pre_times) < 2:
        raise ValueError("Need at least 2 pre-treatment periods.")
    if len(post_times) == 0:
        raise ValueError("No post-treatment periods found.")

    Y1_pre = panel.loc[treated_unit, pre_times].values.astype(np.float64)
    Y1_post = panel.loc[treated_unit, post_times].values.astype(np.float64)

    donors = [u for u in panel.index if u != treated_unit]
    if len(donors) < 3:
        raise ValueError(
            f"Need at least 3 donor units for clustering; got {len(donors)}."
        )

    Y0_pre = panel.loc[donors, pre_times].values.astype(np.float64)
    Y0_post = panel.loc[donors, post_times].values.astype(np.float64)

    # ------------------------------------------------------------------
    # 3. Build clustering feature matrix
    # ------------------------------------------------------------------
    X_donors, X_treated = _build_features(
        Y0_pre, Y1_pre, data, donors, treated_unit, covariates, unit,
        pre_times, time,
    )

    # ------------------------------------------------------------------
    # 4. Cluster donors
    # ------------------------------------------------------------------
    cluster_labels, centers, sil_scores = _cluster_donors(
        X_donors, n_clusters, cluster_method, seed,
    )

    # ------------------------------------------------------------------
    # 5. Assign treated unit to nearest cluster
    # ------------------------------------------------------------------
    treated_cluster = _find_treated_cluster(X_treated, centers, cluster_labels)

    # ------------------------------------------------------------------
    # 6. Select donors from the treated unit's cluster
    # ------------------------------------------------------------------
    in_cluster_mask = cluster_labels == treated_cluster
    selected_idx = list(np.where(in_cluster_mask)[0])

    if augment:
        aug_idx = _augment_donors(
            X_donors, X_treated, cluster_labels, treated_cluster, max_augment,
        )
        selected_idx = sorted(set(selected_idx) | set(aug_idx))

    selected_donors = [donors[i] for i in selected_idx]
    Y0_pre_sel = Y0_pre[selected_idx]
    Y0_post_sel = Y0_post[selected_idx]

    if len(selected_donors) < 2:
        raise ValueError(
            "Selected cluster has fewer than 2 donors.  Try fewer clusters "
            "or enable augment=True."
        )

    # ------------------------------------------------------------------
    # 7. Solve SCM weights within selected donors
    # ------------------------------------------------------------------
    weights = _scm_weights(Y1_pre, Y0_pre_sel)

    # ------------------------------------------------------------------
    # 8. Compute synthetic control and treatment effects
    # ------------------------------------------------------------------
    synth_pre = Y0_pre_sel.T @ weights
    synth_post = Y0_post_sel.T @ weights

    gaps_pre = Y1_pre - synth_pre
    gaps_post = Y1_post - synth_post

    att = float(np.mean(gaps_post))
    pre_rmspe = float(np.sqrt(np.mean(gaps_pre ** 2)))
    post_rmspe = float(np.sqrt(np.mean(gaps_post ** 2)))

    # ------------------------------------------------------------------
    # 9. Placebo inference (optional)
    # ------------------------------------------------------------------
    se = np.nan
    pvalue = np.nan
    ci = (np.nan, np.nan)
    placebo_info: Dict[str, Any] = {}

    if placebo:
        placebo_info = _run_placebos(
            Y0_pre, Y0_post, Y1_pre, donors, cluster_labels, treated_cluster,
            augment, max_augment, X_donors, X_treated, seed,
        )
        if len(placebo_info["atts"]) > 0:
            placebo_atts = np.array(placebo_info["atts"])
            se = float(np.std(placebo_atts, ddof=1))
            # Two-sided rank-based p-value
            n_extreme = np.sum(np.abs(placebo_atts) >= np.abs(att))
            pvalue = float((n_extreme + 1) / (len(placebo_atts) + 1))
            z = stats.norm.ppf(1 - alpha / 2)
            ci = (att - z * se, att + z * se)

    # ------------------------------------------------------------------
    # 10. Build effects DataFrame
    # ------------------------------------------------------------------
    effects_df = pd.DataFrame({
        "time": list(pre_times) + list(post_times),
        "treated": np.concatenate([Y1_pre, Y1_post]),
        "synthetic": np.concatenate([synth_pre, synth_post]),
        "effect": np.concatenate([gaps_pre, gaps_post]),
        "period": ["pre"] * len(pre_times) + ["post"] * len(post_times),
    })

    # ------------------------------------------------------------------
    # 11. Build cluster label mapping
    # ------------------------------------------------------------------
    cluster_label_map = {donors[i]: int(cluster_labels[i])
                         for i in range(len(donors))}
    weight_map = {selected_donors[i]: float(weights[i])
                  for i in range(len(selected_donors))}

    model_info: Dict[str, Any] = {
        "cluster_labels": cluster_label_map,
        "treated_cluster": int(treated_cluster),
        "n_clusters": int(len(np.unique(cluster_labels))),
        "cluster_method": cluster_method,
        "selected_donors": selected_donors,
        "weights": weight_map,
        "silhouette_scores": sil_scores,
        "pre_rmspe": pre_rmspe,
        "post_rmspe": post_rmspe,
        "effects_by_period": effects_df,
        "pre_times": pre_times,
        "post_times": post_times,
        "donors": donors,
        "augmented": augment,
        "placebo": placebo_info,
    }

    return CausalResult(
        method="Cluster Synthetic Control",
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        detail=effects_df,
        model_info=model_info,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    cluster_method: str,
) -> None:
    """Raise on invalid arguments."""
    for col in [outcome, unit, time]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")
    if treated_unit not in data[unit].values:
        raise ValueError(f"Treated unit '{treated_unit}' not in data.")
    if treatment_time not in data[time].values:
        raise ValueError(f"Treatment time '{treatment_time}' not in data.")
    valid_methods = {"kmeans", "spectral", "hierarchical"}
    if cluster_method not in valid_methods:
        raise ValueError(
            f"cluster_method must be one of {valid_methods}, "
            f"got '{cluster_method}'."
        )


def _build_features(
    Y0_pre: np.ndarray,
    Y1_pre: np.ndarray,
    data: pd.DataFrame,
    donors: List[Any],
    treated_unit: Any,
    covariates: Optional[List[str]],
    unit: str,
    pre_times: List[Any],
    time: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build and standardise clustering feature matrices.

    Returns
    -------
    X_donors : ndarray, shape (J, n_features)
    X_treated : ndarray, shape (n_features,)
    """
    from sklearn.preprocessing import StandardScaler
    # Base features: standardised pre-treatment trajectories
    scaler = StandardScaler()
    # Stack treated + donors, fit scaler on all
    all_pre = np.vstack([Y1_pre.reshape(1, -1), Y0_pre])
    all_scaled = scaler.fit_transform(all_pre)

    X_treated_base = all_scaled[0]
    X_donors_base = all_scaled[1:]

    if covariates is None:
        return X_donors_base, X_treated_base

    # Append covariate means over pre-treatment periods
    pre_data = data[data[time].isin(pre_times)]
    cov_list_donors = []
    for donor in donors:
        mask = pre_data[unit] == donor
        cov_list_donors.append(
            pre_data.loc[mask, covariates].mean(axis=0).values.astype(np.float64)
        )
    cov_donors = np.array(cov_list_donors)

    mask_tr = pre_data[unit] == treated_unit
    cov_treated = (
        pre_data.loc[mask_tr, covariates].mean(axis=0).values.astype(np.float64)
    )

    cov_scaler = StandardScaler()
    all_cov = np.vstack([cov_treated.reshape(1, -1), cov_donors])
    all_cov_scaled = cov_scaler.fit_transform(all_cov)

    X_treated_full = np.concatenate([X_treated_base, all_cov_scaled[0]])
    X_donors_full = np.hstack([X_donors_base, all_cov_scaled[1:]])

    return X_donors_full, X_treated_full


def _auto_n_clusters(
    X: np.ndarray,
    method: str,
    seed: Optional[int],
) -> Tuple[int, Dict[int, float]]:
    """
    Select number of clusters via silhouette score.

    Tests k from 2 to min(J-1, 10) and returns the best k.
    """
    from sklearn.metrics import silhouette_score
    J = X.shape[0]
    k_max = min(J - 1, 10)
    if k_max < 2:
        return 2, {2: 0.0}

    scores: Dict[int, float] = {}
    for k in range(2, k_max + 1):
        labels = _fit_cluster(X, k, method, seed)
        # Guard against degenerate clustering (all in one cluster)
        if len(np.unique(labels)) < 2:
            scores[k] = -1.0
            continue
        scores[k] = float(silhouette_score(X, labels))

    best_k = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best_k, scores


def _fit_cluster(
    X: np.ndarray,
    n_clusters: int,
    method: str,
    seed: Optional[int],
) -> np.ndarray:
    """Run one clustering algorithm and return labels."""
    from sklearn.cluster import (
        AgglomerativeClustering,
        KMeans,
        SpectralClustering,
    )
    if method == "kmeans":
        model = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=seed,
        )
        return model.fit_predict(X)

    if method == "spectral":
        n_samples = X.shape[0]
        n_neighbors = max(2, min(n_samples - 1, 10))
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            random_state=seed,
            assign_labels="kmeans",
        )
        return model.fit_predict(X)

    # hierarchical
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
    )
    return model.fit_predict(X)


def _compute_centers(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute cluster centroids from labels."""
    unique_labels = np.unique(labels)
    centers = np.zeros((len(unique_labels), X.shape[1]))
    for i, lab in enumerate(unique_labels):
        centers[i] = X[labels == lab].mean(axis=0)
    return centers


def _cluster_donors(
    X_donors: np.ndarray,
    n_clusters: Optional[int],
    method: str,
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, Dict[int, float]]:
    """
    Cluster the donor feature matrix.

    Returns
    -------
    labels : ndarray of int, shape (J,)
    centers : ndarray, shape (K, n_features)
    sil_scores : dict  (empty if n_clusters was provided manually)
    """
    sil_scores: Dict[int, float] = {}

    if n_clusters is None:
        n_clusters, sil_scores = _auto_n_clusters(X_donors, method, seed)

    # Clamp n_clusters to feasible range
    J = X_donors.shape[0]
    n_clusters = max(2, min(n_clusters, J - 1))

    labels = _fit_cluster(X_donors, n_clusters, method, seed)
    centers = _compute_centers(X_donors, labels)

    return labels, centers, sil_scores


def _find_treated_cluster(
    X_treated: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
) -> int:
    """Assign the treated unit to its nearest cluster centroid."""
    dists = np.linalg.norm(centers - X_treated, axis=1)
    return int(np.unique(labels)[np.argmin(dists)])


def _augment_donors(
    X_donors: np.ndarray,
    X_treated: np.ndarray,
    cluster_labels: np.ndarray,
    treated_cluster: int,
    max_augment: int,
) -> List[int]:
    """
    Return indices of the closest out-of-cluster donors to the treated unit.
    """
    out_mask = cluster_labels != treated_cluster
    out_idx = np.where(out_mask)[0]

    if len(out_idx) == 0:
        return []

    dists = np.linalg.norm(X_donors[out_idx] - X_treated, axis=1)
    n_take = min(max_augment, len(out_idx))
    top_idx = out_idx[np.argsort(dists)[:n_take]]
    return list(top_idx)


def _scm_weights(Y1_pre: np.ndarray, Y0_pre: np.ndarray) -> np.ndarray:
    """
    Standard SCM: non-negative weights minimising
    ||Y1 - Y0^T w||^2   s.t.  sum(w) = 1,  w >= 0.
    """
    from ._core import solve_simplex_weights
    return solve_simplex_weights(Y1_pre, Y0_pre.T)


def _run_placebos(
    Y0_pre: np.ndarray,
    Y0_post: np.ndarray,
    Y1_pre: np.ndarray,
    donors: List[Any],
    cluster_labels: np.ndarray,
    treated_cluster: int,
    augment: bool,
    max_augment: int,
    X_donors: np.ndarray,
    X_treated: np.ndarray,
    seed: Optional[int],
) -> Dict[str, Any]:
    """
    In-space placebo test: pretend each donor in the treated cluster is the
    treated unit and re-run cluster SCM.

    Returns
    -------
    dict with 'atts', 'pre_rmspes', 'post_rmspes', 'ratios', 'units'.
    """
    atts: List[float] = []
    pre_rmspes: List[float] = []
    post_rmspes: List[float] = []
    ratios: List[float] = []
    units: List[Any] = []

    J = Y0_pre.shape[0]

    # All donor pre/post data stacked with treated at position 0
    all_pre = np.vstack([Y1_pre.reshape(1, -1), Y0_pre])   # (J+1, T0)
    all_post = np.vstack([
        np.zeros((1, Y0_post.shape[1])),  # placeholder — filled per iter
        Y0_post,
    ])
    # We need treated post as well for the full pool
    # Treated post is not in Y0_post; we omit the treated from placebos
    # and only iterate over donors.

    for i in range(J):
        placebo_unit = donors[i]
        # The "treated" in this iteration is donor i
        Y_p_pre = Y0_pre[i]
        Y_p_post = Y0_post[i]

        # Remaining donors: all other donors + the actual treated unit
        other_idx = [j for j in range(J) if j != i]
        # Pool: other donors' pre/post
        Y_pool_pre = Y0_pre[other_idx]
        Y_pool_post = Y0_post[other_idx]

        if Y_pool_pre.shape[0] < 2:
            continue

        # Determine which cluster this placebo belongs to and select donors
        # from that cluster (mirroring the main algorithm)
        in_cluster = cluster_labels[other_idx]
        placebo_cluster = cluster_labels[i]
        sel = np.where(in_cluster == placebo_cluster)[0]

        if augment and len(sel) < Y_pool_pre.shape[0]:
            out_mask = in_cluster != placebo_cluster
            out_idx = np.where(out_mask)[0]
            if len(out_idx) > 0:
                dists = np.linalg.norm(
                    X_donors[np.array(other_idx)[out_idx]] - X_donors[i],
                    axis=1,
                )
                n_take = min(max_augment, len(out_idx))
                extra = out_idx[np.argsort(dists)[:n_take]]
                sel = np.sort(np.concatenate([sel, extra]))

        if len(sel) < 1:
            continue

        Y_sel_pre = Y_pool_pre[sel]
        Y_sel_post = Y_pool_post[sel]

        try:
            w = _scm_weights(Y_p_pre, Y_sel_pre)
            synth_pre = Y_sel_pre.T @ w
            synth_post = Y_sel_post.T @ w

            gap_pre = Y_p_pre - synth_pre
            gap_post = Y_p_post - synth_post

            pre_mspe = float(np.mean(gap_pre ** 2))
            post_mspe = float(np.mean(gap_post ** 2))
            att_p = float(np.mean(gap_post))
            ratio = (
                np.sqrt(post_mspe) / np.sqrt(pre_mspe)
                if pre_mspe > 1e-10
                else 0.0
            )

            atts.append(att_p)
            pre_rmspes.append(np.sqrt(pre_mspe))
            post_rmspes.append(np.sqrt(post_mspe))
            ratios.append(float(ratio))
            units.append(placebo_unit)
        except Exception:
            continue

    return {
        "atts": atts,
        "pre_rmspes": pre_rmspes,
        "post_rmspes": post_rmspes,
        "ratios": ratios,
        "units": units,
    }
