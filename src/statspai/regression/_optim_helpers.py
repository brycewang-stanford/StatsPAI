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
