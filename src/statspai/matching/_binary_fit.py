"""Maximum-likelihood fit of a logit or probit propensity model.

One Newton-Raphson, shared by the matching estimators. It runs on the
standardised design (each covariate centred and divided by its standard
deviation) and returns the coefficients on the covariates as given.

A covariate that is constant, or a linear combination of the ones before
it, is left out and its coefficient reported as zero: Stata's ``note: x
omitted because of collinearity``. The fitted score does not depend on
which of two collinear columns is kept.

Why both steps are needed. The fit used to run on the columns as given and
solve the Newton step whatever the Hessian was. With a redundant covariate
the Hessian is singular; ``numpy.linalg.solve`` returns a step anyway, and
its error lies along the redundant direction. That direction leaves the
score alone when the columns are of similar size. Propensity models are
told to include powers and interactions (``age^3``, ``re74^2``,
``education * re74``), though, and next to a column of order 1e9 the error
leaks into the fitted score: on one specification with a repeated dummy
the scores were off in the third decimal and the matched estimate by a
third. Standardising makes the redundant column detectable at a fixed
tolerance and the remaining Hessian well conditioned.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from ..exceptions import ConvergenceWarning

__all__ = ["fit_binary_index"]

#: a column counts as collinear when what is left of it, after the columns
#: before it are projected out, is this fraction of its length
_COLLINEAR = 1e-9


def _pieces(model: str, eta: np.ndarray, t: np.ndarray) -> tuple:
    """Log-likelihood, score residual and curvature at the index ``eta``."""
    if model == "probit":
        eta = np.clip(eta, -37.0, 37.0)
        log_p, log_q = norm.logcdf(eta), norm.logcdf(-eta)
        loglik = float(np.sum(np.where(t == 1, log_p, log_q)))
        # T f / Phi(eta) - (1 - T) f / Phi(-eta), stable in both tails
        logpdf = norm.logpdf(eta)
        lam = np.where(t == 1, np.exp(logpdf - log_p), -np.exp(logpdf - log_q))
        return loglik, lam, lam * (lam + eta)
    loglik = float(np.sum(t * eta - np.logaddexp(0.0, eta)))
    p = expit(eta)
    return loglik, t - p, p * (1.0 - p)


def _independent_columns(design: np.ndarray) -> np.ndarray:
    """Columns of ``design`` that are not combinations of earlier ones."""
    r = np.linalg.qr(design, mode="r")
    length = np.sqrt(np.sum(design**2, axis=0))
    diag = np.abs(np.diag(r))
    return np.asarray(diag > _COLLINEAR * np.maximum(length, 1e-300))


def fit_binary_index(
    X: np.ndarray,
    T: np.ndarray,
    model: str = "logit",
    *,
    max_iter: int = 200,
) -> Dict[str, Any]:
    """Fit ``P(T = 1 | X) = F(a + X b)`` by maximum likelihood.

    Parameters
    ----------
    X : ndarray, shape (n, k)
        Covariates, without a constant.
    T : ndarray, shape (n,)
        The 0/1 outcome.
    model : {'logit', 'probit'}
    max_iter : int
        Newton steps allowed before a :class:`ConvergenceWarning`.

    Returns
    -------
    dict
        ``beta`` -- coefficients, constant first, zero for an omitted
        covariate; ``vcov`` -- inverse of the observed information, in the
        same coordinates (zero rows and columns for an omitted covariate);
        ``omitted`` -- positions in ``X`` of the omitted covariates;
        ``loglik``; ``iterations``; ``converged``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    t = np.asarray(T, dtype=float)
    n, k = X.shape
    mean = X.mean(axis=0) if k else np.zeros(0)
    sd = X.std(axis=0) if k else np.zeros(0)
    varies = sd > 1e-12 * np.maximum(np.abs(mean), 1.0)
    scale = np.where(varies, sd, 1.0)
    Z = (X - mean) / scale
    design = np.column_stack([np.ones(n), Z])
    keep = np.r_[True, varies]
    keep[keep] = _independent_columns(design[:, keep])
    keep[0] = True
    A = design[:, keep]

    gamma = np.zeros(A.shape[1])
    loglik, resid, curv = _pieces(model, A @ gamma, t)
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        grad = A.T @ resid
        H = (A * curv[:, None]).T @ A
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, grad, rcond=None)[0]
        decrement = float(grad @ delta)
        step = 1.0
        for _ in range(40):
            trial = gamma + step * delta
            new = _pieces(model, A @ trial, t)
            if np.isfinite(new[0]) and new[0] >= loglik - 1e-12 * abs(loglik):
                break
            step /= 2.0
        gain = new[0] - loglik
        gamma, (loglik, resid, curv) = trial, new
        if abs(decrement) < 1e-10 or abs(gain) <= 1e-13 * (abs(loglik) + 1.0):
            converged = True
            break
    if not converged:
        warnings.warn(
            f"the {model} propensity model did not converge in {max_iter} "
            "Newton steps; the fitted scores are not the maximum-likelihood "
            "ones. Check for a covariate that predicts treatment perfectly.",
            ConvergenceWarning,
            stacklevel=3,
        )

    # back to the covariates as given: b_j = g_j / sd_j, a = g_0 - sum b_j m_j
    kept = np.flatnonzero(keep[1:])
    back = np.zeros((k + 1, A.shape[1]))
    back[0, 0] = 1.0
    for col, j in enumerate(kept, start=1):
        back[j + 1, col] = 1.0 / scale[j]
        back[0, col] = -mean[j] / scale[j]
    H = (A * curv[:, None]).T @ A
    try:
        inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(H)
    return {
        "beta": back @ gamma,
        "vcov": back @ inv @ back.T,
        "omitted": [j for j in range(k) if not keep[j + 1]],
        "loglik": loglik,
        "iterations": iterations,
        "converged": converged,
    }
