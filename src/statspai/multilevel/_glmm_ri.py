"""
Vectorised random-intercept (q = 1) kernel for the GLMM likelihoods.

Every parity specification and most applied GLMMs have a single scalar
random intercept.  For that case the per-group Newton iterations for the
conditional modes, the Laplace terms and the adaptive Gauss-Hermite sums
are all elementwise in the group index, so they can run for every group at
once with ``np.bincount`` segment sums instead of a Python loop over
groups.  The numbers are the same as the per-block path in
:mod:`.glmm` (same Newton step, same damping, same extra
quadratic-convergence step); only the bookkeeping differs.

The kernel is shared by :func:`.glmm.meglm` (exponential-family GLMMs)
and :func:`._ordinal.meologit` through a pair of callables:

``curv_score(eta) -> (W, s)``
    observed (or Fisher) information and score on η per observation.
``loglik_vec(eta) -> ll``
    per-observation conditional log-likelihood.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy import special

_LOG_2PI = float(np.log(2.0 * np.pi))
_EPS = 1e-12


def ri_mode(
    eta_fixed: np.ndarray,
    gidx: np.ndarray,
    n_groups: int,
    sigma2: float,
    curv_score: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    u0: np.ndarray,
    max_inner: int = 50,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Conditional modes û_g and curvatures H_g for all groups at once.

    Newton on ``Σ_i log f(y_i | η_i + u_g) − u_g² / (2σ²)`` for every g
    simultaneously, with the per-block path's step damping; once every
    group meets the step criterion one further (quadratically convergent)
    step is taken so û is at machine precision.
    """
    inv_s2 = 1.0 / max(sigma2, _EPS)
    u = u0.astype(float).copy()
    all_done = False
    converged = False
    for _ in range(max_inner):
        W, s = curv_score(eta_fixed + u[gidx])
        grad = np.bincount(gidx, weights=s, minlength=n_groups) - inv_s2 * u
        H = np.bincount(gidx, weights=W, minlength=n_groups) + inv_s2
        if not np.all(H > 0):
            break
        step = grad / H
        u_abs = np.abs(u) + 1e-12
        cap = 5.0 * np.maximum(u_abs, 1.0)
        step = np.where(np.abs(step) > cap, np.sign(step) * cap, step)
        u = u + step
        if all_done:
            converged = True
            break
        all_done = bool(np.all(np.abs(step) < tol * (1.0 + u_abs)))
    W, _ = curv_score(eta_fixed + u[gidx])
    H = np.bincount(gidx, weights=W, minlength=n_groups) + inv_s2
    if not np.all(H > 0):
        converged = False
    return u, H, converged


def ri_marginal_loglik(
    eta_fixed: np.ndarray,
    gidx: np.ndarray,
    n_groups: int,
    sigma2: float,
    curv_score: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    loglik_vec: Callable[[np.ndarray], np.ndarray],
    u_cache: np.ndarray,
    gh_nodes: Optional[np.ndarray] = None,
    gh_log_weights: Optional[np.ndarray] = None,
) -> float:
    """Laplace (``gh_nodes is None``) or AGHQ marginal log-likelihood.

    Updates ``u_cache`` in place with the new modes (warm start for the
    next call).  Returns ``-inf`` when a curvature is non-positive.
    """
    u, H, _ = ri_mode(eta_fixed, gidx, n_groups, sigma2, curv_score, u_cache)
    u_cache[:] = u
    if not np.all(H > 0):
        return -np.inf
    log_s2 = np.log(max(sigma2, _EPS))
    if gh_nodes is None:
        ll_data = float(np.sum(loglik_vec(eta_fixed + u[gidx])))
        return ll_data - 0.5 * float(
            np.sum(log_s2 + u**2 / max(sigma2, _EPS) + np.log(H))
        )
    # AGHQ: u_gk = û_g + √2 σ̂_g x_k,  σ̂_g = H_g^{-1/2}.
    sig_hat = 1.0 / np.sqrt(H)
    K = gh_nodes.shape[0]
    log_terms = np.empty((n_groups, K))
    for k in range(K):
        u_k = u + np.sqrt(2.0) * sig_hat * gh_nodes[k]
        ll_k = np.bincount(
            gidx, weights=loglik_vec(eta_fixed + u_k[gidx]), minlength=n_groups
        )
        log_prior = -0.5 * (_LOG_2PI + log_s2) - 0.5 * u_k**2 / max(sigma2, _EPS)
        log_terms[:, k] = (
            ll_k
            + log_prior
            + gh_nodes[k] ** 2
            + gh_log_weights[k]
            + 0.5 * (np.log(2.0) + 2.0 * np.log(sig_hat))
        )
    return float(np.sum(special.logsumexp(log_terms, axis=1)))
