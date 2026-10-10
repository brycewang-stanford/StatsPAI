"""Shared helpers for ML estimators that need a covariance matrix.

The asymptotic covariance of an MLE is the inverse observed Fisher
information at the optimum,

    V = [ -H(theta_hat) ]^{-1},

where H is the Hessian of the log-likelihood. Several `sp.regression`
modules previously read `scipy.optimize.minimize(method="BFGS").hess_inv`
as a stand-in. That object is BFGS's running quasi-Newton update of
the inverse Hessian — it is good enough to drive the optimiser, but it
is NOT a reliable estimator of the true Hessian at the optimum.

Parity comparisons against R / Stata for `sp.tobit`, `sp.ologit`,
`sp.mlogit`, and friends showed SE errors of 13-30% (and 26% for the
main coefficient of ordered logit) traceable to this approximation.

This module provides a single helper, ``numerical_hessian``, that
computes the Hessian by central finite differences. The result is
inverted (with a Moore-Penrose fallback) and returned as the
covariance matrix.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient


def require_two_clusters(clusters: Any, function: str) -> int:
    """Refuse a cluster-robust variance with fewer than two clusters.

    At the MLE the scores sum to zero over the sample, so with a single
    cluster the clustered meat ``sum_g s_g s_g'`` is zero up to rounding and
    ``G/(G-1)`` is undefined: the variance is not estimable. ``sp.regress``,
    ``sp.glm`` and ``sp.iv`` refuse this input; the likelihood estimators
    used to report standard errors of order 1e-16 (or a 1e-10 floor).
    Returns the number of clusters.
    """
    n_clusters = int(len(pd.unique(np.asarray(clusters).ravel())))
    if n_clusters < 2:
        raise DataInsufficient(
            f"{function}: cluster-robust standard errors need at least two "
            f"clusters; the cluster variable has {n_clusters} distinct value"
            f"{'' if n_clusters == 1 else 's'} in the estimation sample.",
            recovery_hint=(
                "Cluster on a variable with several groups, or use "
                "robust='robust' for heteroskedasticity-robust standard "
                "errors."
            ),
            diagnostics={"n_clusters": n_clusters},
        )
    return n_clusters


def robust_convergence(opt_result: Any, grad_tol: float = 1e-3) -> Tuple[bool, float]:
    """Robust convergence flag for a ``scipy.optimize.minimize`` MLE fit.

    SciPy's BFGS frequently returns ``success=False`` with status 2
    ("Desired error not necessarily achieved due to precision loss") at a
    perfectly good optimum of a flat log-likelihood — a line-search
    artefact, not a real failure (Nelder-Mead reaches the identical
    objective and coefficients). Trusting ``success`` alone makes MLE
    estimators (``tobit``, ``truncreg``, ``zip``/``zinb``, ...) report
    ``converged=False`` on correct fits.

    This treats a small gradient norm at the optimum as convergence:
    ``success`` **or** ``norm(grad) < grad_tol`` with a finite objective. It
    only ever relaxes a *false negative* — a genuinely non-converged run
    leaves a large gradient and still reports ``False``, so the flag never
    wrongly claims convergence.

    Parameters
    ----------
    opt_result : scipy.optimize.OptimizeResult
        Object returned by ``scipy.optimize.minimize``. Only its
        ``success``, ``jac`` and ``fun`` attributes are read.
    grad_tol : float, default 1e-3
        Gradient-norm threshold below which a non-``success`` optimum is
        accepted as converged.

    Returns
    -------
    (converged, gradient_norm) : tuple of (bool, float)
        ``gradient_norm`` is ``inf`` when no gradient is available.
    """
    grad = getattr(opt_result, "jac", None)
    grad_norm = (
        float(np.linalg.norm(np.asarray(grad, dtype=float)))
        if grad is not None
        else float("inf")
    )
    success = bool(getattr(opt_result, "success", False))
    fun = getattr(opt_result, "fun", np.inf)
    converged = bool(success or (np.isfinite(fun) and grad_norm < grad_tol))
    return converged, grad_norm


def numerical_hessian(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float | None = None,
) -> np.ndarray:
    """Central-difference Hessian of a scalar-valued ``func`` at ``x``.

    The (i, j) entry is

        H[i, j] = (f(x + e_i + e_j) - f(x + e_i - e_j)
                   - f(x - e_i + e_j) + f(x - e_i - e_j)) / (4 h_i h_j)

    with a per-coordinate step ``h_i = eps * max(|x_i|, 1)`` (Press et
    al., *Numerical Recipes*, §5.7). The matrix is then symmetrised.

    Parameters
    ----------
    func : callable
        Returns a scalar (e.g. negative log-likelihood at ``x``).
    x : np.ndarray
        Point at which to evaluate the Hessian. Length ``k`` triggers
        ``2*k*(k+1)`` function evaluations.
    eps : float, optional
        Relative step size. Default ``(machine_eps)**(1/3) ≈ 6e-6``,
        which minimises truncation + round-off error for a smooth
        twice-differentiable ``func``.

    Returns
    -------
    H : np.ndarray of shape (k, k)
        Symmetric numerical Hessian.
    """
    x = np.asarray(x, dtype=float).copy()
    k = x.size
    if eps is None:
        eps = float(np.finfo(float).eps ** (1.0 / 3.0))
    h = eps * np.maximum(np.abs(x), 1.0)

    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            dx_i = np.zeros(k)
            dx_i[i] = h[i]
            dx_j = np.zeros(k)
            dx_j[j] = h[j]
            fpp = func(x + dx_i + dx_j)
            fpm = func(x + dx_i - dx_j)
            fmp = func(x - dx_i + dx_j)
            fmm = func(x - dx_i - dx_j)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j])
            H[j, i] = H[i, j]
    return H


def hessian_cov(
    neg_loglik: Callable[[np.ndarray], float],
    theta_hat: np.ndarray,
    eps: float | None = None,
    ridge: float = 1e-10,
) -> np.ndarray:
    """Covariance matrix from the numerical observed information.

    Inverts ``H_neg = numerical_hessian(neg_loglik, theta_hat)`` (which
    equals ``-H(loglik)``). Falls back to a small ridge plus pseudo-
    inverse when the Hessian is singular or indefinite. The caller is
    expected to clip negative diagonal entries before taking the
    sqrt for standard errors.

    Parameters
    ----------
    neg_loglik : callable
        Negative log-likelihood evaluated at ``theta``.
    theta_hat : np.ndarray
        MLE estimate.
    eps : float, optional
        Step size override forwarded to :func:`numerical_hessian`.
    ridge : float
        Diagonal ridge added before pinv fallback.

    Returns
    -------
    V : np.ndarray of shape (k, k)
        Estimated covariance matrix of ``theta_hat``.
    """
    H_neg = numerical_hessian(neg_loglik, theta_hat, eps=eps)
    try:
        V = np.linalg.inv(H_neg)
    except np.linalg.LinAlgError:
        V = np.linalg.pinv(H_neg + ridge * np.eye(H_neg.shape[0]))
    return V


# ---------------------------------------------------------------------------
# Per-observation scores and the observed information, to machine precision
# ---------------------------------------------------------------------------
#
# ``numerical_hessian`` differences the *total* log-likelihood twice, so its
# round-off error is ~ |ll| * eps_mach / h^2: with ll ~ 1e3 and h ~ 1e-5 that
# is ~1e-3 in absolute terms, visible in the fifth digit of a standard error.
# The helpers below differentiate a *per-observation* log-likelihood written
# with complex-safe primitives (``np.exp``, ``np.log``, ``scipy.special.ndtr``
# / ``log_ndtr`` / ``loggamma``): the complex-step derivative
# ``Im f(theta + i h e_j) / h`` has no subtractive cancellation, so the scores
# are exact to rounding, and the Hessian -- one central difference of those
# exact scores -- is accurate to ~1e-10 relative.  Robust and cluster
# sandwiches also need the per-observation scores, which ``numerical_hessian``
# cannot supply.

ObsLoglik = Callable[[np.ndarray], np.ndarray]


def complex_step_scores(
    obs_loglik: ObsLoglik, theta: np.ndarray, h: float = 1e-30
) -> np.ndarray:
    """Per-observation scores ``d l_i / d theta`` by the complex-step method.

    Parameters
    ----------
    obs_loglik : callable
        Maps a (possibly complex) parameter vector to the ``(n,)`` vector of
        per-observation log-likelihood contributions. It must be built from
        complex-safe operations (no ``np.clip``, no ``astype(float)``).
    theta : np.ndarray
        Real parameter vector of length ``k``.
    h : float, default 1e-30
        Imaginary step; any value far below ``sqrt(eps_mach)`` gives the same
        result because there is no cancellation.

    Returns
    -------
    np.ndarray of shape (n, k)
    """
    theta = np.asarray(theta, dtype=float)
    columns = []
    for j in range(theta.size):
        tc = theta.astype(complex)
        tc[j] += 1j * h
        columns.append(np.imag(obs_loglik(tc)) / h)
    return np.column_stack(columns)


def ml_scores_hessian(
    obs_loglik: ObsLoglik, theta: np.ndarray, rel_step: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-observation scores and the Hessian of the total log-likelihood.

    The Hessian is the central difference of the complex-step total score
    with step ``rel_step * max(1, |theta_j|)``, symmetrised.

    Returns
    -------
    (scores, H)
        ``scores`` has shape ``(n, k)``; ``H`` is the ``(k, k)`` Hessian of
        the log-likelihood (negative definite at a maximum), so the
        observed information is ``-H``.
    """
    theta = np.asarray(theta, dtype=float)
    scores = complex_step_scores(obs_loglik, theta)
    k = theta.size
    H = np.empty((k, k))
    for j in range(k):
        step = rel_step * max(1.0, abs(float(theta[j])))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += step
        tm[j] -= step
        H[:, j] = (
            complex_step_scores(obs_loglik, tp).sum(axis=0)
            - complex_step_scores(obs_loglik, tm).sum(axis=0)
        ) / (2.0 * step)
    return scores, 0.5 * (H + H.T)


def ml_newton_polish(
    obs_loglik: ObsLoglik,
    theta: np.ndarray,
    maxiter: int = 50,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Newton-Raphson from a nearby start to the maximum of ``sum(obs_loglik)``.

    Quasi-Newton optimisers stop on a gradient tolerance that leaves the
    estimates several digits short of the optimum Stata's Newton-Raphson
    reaches; a few exact Newton steps close the gap. Each step is halved
    until the log-likelihood does not decrease, and iteration stops when the
    largest step is below ``tol * (1 + max|theta|)``. If no improving step
    exists the input is returned unchanged.

    Returns
    -------
    (theta, scores, H, n_steps)
        The polished estimate with its per-observation scores and Hessian
        (see :func:`ml_scores_hessian`).
    """
    theta = np.asarray(theta, dtype=float).copy()

    def total(t: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            value = float(np.sum(np.real(obs_loglik(t))))
        return value if np.isfinite(value) else -np.inf

    scores, H = ml_scores_hessian(obs_loglik, theta)
    ll = total(theta)
    steps = 0
    for _ in range(maxiter):
        try:
            step = np.linalg.solve(H, scores.sum(axis=0))
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        for _halving in range(40):
            candidate = theta - step
            ll_new = total(candidate)
            if ll_new >= ll - 1e-12 * abs(ll):
                break
            step = step / 2.0
        else:
            break
        theta, ll = candidate, ll_new
        scores, H = ml_scores_hessian(obs_loglik, theta)
        steps += 1
        if np.max(np.abs(step)) < tol * (1.0 + np.max(np.abs(theta))):
            break
    return theta, scores, H, steps


def newton_converged(scores: np.ndarray, H: np.ndarray, tol: float = 1e-5) -> bool:
    """Stata's Newton-Raphson convergence rule at a polished optimum.

    ``g' (-H)^{-1} g < tol`` with ``g`` the total score -- ``ml``'s default
    ``nrtolerance(1e-5)``. Judge convergence here, after
    :func:`ml_newton_polish`, rather than on the quasi-Newton run that only
    supplied the start: BFGS often stops on a line-search failure short of
    the optimum the polish then reaches, and reporting its flag said
    ``converged=False`` for correct estimates (``sp.oprobit`` on the
    Princelings replication, QJE 2019).
    """
    g = np.asarray(scores, dtype=float)
    g = g.sum(axis=0) if g.ndim == 2 else g
    if not np.all(np.isfinite(g)):
        return False
    try:
        v = float(g @ np.linalg.solve(-np.asarray(H, dtype=float), g))
    except np.linalg.LinAlgError:
        return False
    return bool(np.isfinite(v) and abs(v) < tol)


def inverse_information(H: np.ndarray) -> np.ndarray:
    """``(-H)^{-1}`` with a pseudo-inverse fallback for a singular Hessian."""
    try:
        return np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(-H)


def se_from_vcov(V: np.ndarray) -> np.ndarray:
    """Standard errors from a covariance matrix; negative variances become NaN.

    ``sqrt(abs(diag))`` would report a finite standard error for a variance
    the optimiser got wrong in sign, which is the failure that should be
    visible.
    """
    d = np.diag(np.asarray(V, dtype=float))
    with np.errstate(invalid="ignore"):
        return np.where(d >= 0, np.sqrt(np.maximum(d, 0.0)), np.nan)
