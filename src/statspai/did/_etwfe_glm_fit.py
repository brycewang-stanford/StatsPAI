"""Fitting helpers of the nonlinear ETWFE: unit-FE PPML, cluster sandwich,
collinearity screen.  Used by :func:`._etwfe_nonlinear.etwfe_glm`."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from ..exceptions import ConvergenceFailure, DataInsufficient


def _cluster_sandwich(
    scores: np.ndarray,
    bread_inv: np.ndarray,
    cl_codes: np.ndarray,
    n_clusters: int,
    factor: float,
) -> np.ndarray:
    """``factor * A^{-1} (sum_g s_g s_g') A^{-1}`` with per-cluster sums."""
    G = int(cl_codes.max()) + 1 if cl_codes.size else 0
    S = np.column_stack(
        [
            np.bincount(cl_codes, weights=scores[:, j], minlength=G)
            for j in range(scores.shape[1])
        ]
    )
    meat = S.T @ S
    return factor * (bread_inv @ meat @ bread_inv)


def _independent_columns(
    X: np.ndarray, groups: Optional[np.ndarray] = None, tol: float = 1e-9
) -> np.ndarray:
    """Indices of the columns kept after dropping linear dependencies.

    Columns are screened in order and a column is dropped when it is (to
    ``tol`` relative) a linear combination of the ones already kept --
    Stata ``_rmcoll``'s rule, which ``ppmlhdfe`` applies after partialling
    out the absorbed effects.  With ``groups`` the columns are demeaned
    within those groups first (the absorbed unit effect).  Works on the
    ``K x K`` cross-product, so it costs one pass over the data.
    """
    Xd = np.asarray(X, dtype=float)
    if groups is not None and Xd.shape[1]:
        G = int(groups.max()) + 1
        cnt = np.bincount(groups, minlength=G).astype(float)
        means = np.column_stack(
            [
                np.bincount(groups, weights=Xd[:, j], minlength=G) / cnt
                for j in range(Xd.shape[1])
            ]
        )
        Xd = Xd - means[groups]
    A = Xd.T @ Xd
    kept: List[int] = []
    for j in range(A.shape[0]):
        ajj = A[j, j]
        if not np.isfinite(ajj) or ajj <= 0.0:
            continue
        if kept:
            Akk = A[np.ix_(kept, kept)]
            akj = A[kept, j]
            try:
                resid = ajj - akj @ np.linalg.solve(Akk, akj)
            except np.linalg.LinAlgError:  # pragma: no cover - kept is independent
                resid = 0.0
            if resid <= tol * ajj:
                continue
        kept.append(j)
    return np.asarray(kept, dtype=int)


def _fit_poisson_unit_fe(
    y: np.ndarray,
    X: np.ndarray,
    unit_codes: np.ndarray,
    cl_codes: np.ndarray,
    n_clusters_full: int,
    n_full: int,
    count_separated: bool = True,
) -> Dict[str, Any]:
    """PPML with the unit effect absorbed; ``fe='unit'``.

    Separated rows (units whose outcome is zero in every period, and rows a
    single regressor predicts to be zero) are removed before IRLS, as in
    Stata ``ppmlhdfe``; they carry no information about the slopes and their
    fitted mean is exactly zero.  They stay in ``N`` and in the cluster count
    that enters the ``G/(G-1)`` factor, which is how ``jwdid`` reports its
    ``ppmlhdfe`` fits (the replication-package tables print the full-sample
    ``N``).

    The small-sample factor is ``G/(G-1)`` alone, Stata ``ppmlhdfe``'s
    clustered convention (and ``sp.ppmlhdfe``'s): with 26 clusters and 18
    regressors a ``(N-1)/(N-K)`` term would inflate every SE by 3.5%
    relative to ``jwdid``.  ``count_separated=False`` counts only the
    clusters with a non-separated row, as ``ppmlhdfe`` does whenever it
    flags the separation (``etwfe(separated='drop')``).
    """
    from ..regression.count import _ppml_hdfe_irls, _ppml_separation_mask

    keep, sep_counts = _ppml_separation_mask(y, X, [unit_codes])
    if not keep.any():
        raise DataInsufficient(
            "etwfe(family='poisson', fe='unit'): every unit has an all-zero "
            "outcome; nothing to estimate.",
            recovery_hint="Check the outcome column.",
            diagnostics={"n_separated": int(len(y))},
        )
    Xk = X[keep]
    live = np.flatnonzero(np.any(Xk != 0, axis=0))
    # Collinear columns (e.g. a covariate's level dummies that sum to one
    # inside a cohort's cells) are omitted as ppmlhdfe does; the
    # aggregates are estimable functions and do not depend on which one.
    live = live[_independent_columns(Xk[:, live], groups=unit_codes[keep])]
    beta_live, mu, converged, n_iter, X_dm = _ppml_hdfe_irls(
        y[keep],
        Xk[:, live],
        fe_indices_list=[unit_codes[keep]],
        maxiter=1000,
        tol=1e-10,
    )
    if not converged:
        warnings.warn(
            "etwfe(family='poisson', fe='unit'): PPML did not converge; "
            "estimates may be unreliable.",
            stacklevel=3,
        )
    yk = y[keep]
    bread = X_dm.T @ (mu[:, None] * X_dm)
    try:
        bread_inv = np.linalg.inv(bread)
    except np.linalg.LinAlgError as exc:
        raise ConvergenceFailure(
            "etwfe(family='poisson', fe='unit'): singular information matrix "
            "(collinear cohort x period cells or controls).",
            recovery_hint="Drop collinear controls or check the cohort " "timing.",
            diagnostics={"n_obs": int(keep.sum())},
        ) from exc
    k = len(live)
    G = int(n_clusters_full) if count_separated else int(np.unique(cl_codes[keep]).size)
    factor = G / (G - 1.0) if G > 1 else 1.0
    vcov = _cluster_sandwich(
        X_dm * (yk - mu)[:, None], bread_inv, cl_codes[keep], G, factor
    )
    # ppmlhdfe also reports ``_cons``: the weighted means are added back
    # before the last IRLS step, so the index is ``(x - xbar)'b + a`` with
    # ``xbar`` the mu-weighted sample mean and ``a = xbar'b + _cons``.  Its
    # score is ``y - mu`` and ``X_dm`` is mu-orthogonal to a constant, so the
    # bread is block diagonal; ``vcov_cons`` is the covariance of
    # ``(b, a)``.  When the clusters nest the units the ``a`` block is
    # exactly zero (the Poisson FOC makes ``y - mu`` sum to zero by unit).
    resid = yk - mu
    mu_tot = float(mu.sum())
    bread_c = np.zeros((k + 1, k + 1))
    bread_c[:k, :k] = bread_inv
    bread_c[k, k] = 1.0 / mu_tot
    vcov_cons = _cluster_sandwich(
        np.column_stack([X_dm * resid[:, None], resid]),
        bread_c,
        cl_codes[keep],
        G,
        factor,
    )
    # Cluster-summed scores (in the full cluster coding) and the bread, for
    # the unconditional variance of the aggregates (response_se=
    # 'unconditional'): vcov == factor * bread_inv S'S bread_inv.
    score_cl = np.column_stack(
        [
            np.bincount(
                cl_codes[keep],
                weights=X_dm[:, j] * resid,
                minlength=int(cl_codes.max()) + 1,
            )
            for j in range(k)
        ]
    )
    return {
        "keep": keep,
        "live": live,
        "beta": beta_live,
        "vcov": vcov,
        "vcov_cons": vcov_cons,
        "score_cl": score_cl,
        "bread_inv": bread_inv,
        "mu": mu,
        "converged": bool(converged),
        "n_iter": int(n_iter),
        "sep_counts": sep_counts,
        "ssc": {"n": int(n_full), "k": int(k), "G": G, "factor": float(factor)},
    }
