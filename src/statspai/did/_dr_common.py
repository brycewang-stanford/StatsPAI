"""Shared algebra for the two-period doubly robust DiD estimators.

Clean-room implementation (2026-10-20). Written from Sant'Anna and Zhao
(2020) [@santanna2020doubly] and the public DRDID / did documentation;
the reference implementations were used only as black boxes (their
outputs on our data), never read. The specification, with the black-box
evidence for every convention, is ``docs/dev/cleanroom/did_drdid.md``.

Every estimator in :mod:`statspai.did._rcs` and the panel engines in
:mod:`statspai.did.callaway_santanna` is built from the same four pieces:

* a Hájek mean ``theta = E_n[a Z] / E_n[a]`` of some outcome ``Z`` under
  non-negative weights ``a`` (:class:`HajekMean`), whose influence
  function is ``a (Z - theta) / E_n[a]``;
* the asymptotic linear representation of a weighted least-squares fit
  (:func:`wls_influence`);
* the asymptotic linear representation of a weighted logit maximum
  likelihood fit (:func:`logit_influence`);
* the inverse probability tilting fit of the propensity score
  (:func:`ipt_fit`), whose first-order conditions make the estimation
  effect of the nuisance parameters vanish (Sant'Anna and Zhao 2020,
  Sections 3.1-3.2).

The influence function of an estimator is the sum of the Hájek terms
plus, for each estimated nuisance parameter vector, ``l_i' g`` where
``l_i`` is that parameter's influence function and ``g`` the gradient of
the estimator with respect to the parameter (the delta method).

Influence functions here are on the scale where
``theta_hat - theta ~ mean(IF)``; the analytic standard error of a
sample of ``n`` rows is ``sqrt(mean(IF**2) / n)``.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

from ..exceptions import ConvergenceWarning

__all__ = [
    "HajekMean",
    "design_matrix",
    "ipt_fit",
    "logit_fit",
    "logit_influence",
    "normalized_weights",
    "se_from_influence",
    "spanning_columns",
    "wls_coef",
    "wls_influence",
]


def normalized_weights(weights: Optional[np.ndarray], n: int) -> np.ndarray:
    """Return ``weights / mean(weights)`` (ones when ``weights`` is None)."""
    if weights is None:
        return np.ones(n, dtype=float)
    w = np.asarray(weights, dtype=float)
    return w / w.mean()


def design_matrix(covariates: Optional[np.ndarray], n: int) -> np.ndarray:
    """Covariates with a leading intercept column (intercept only if None)."""
    if covariates is None:
        return np.ones((n, 1), dtype=float)
    x = np.asarray(covariates, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 0:
        return np.ones((n, 1), dtype=float)
    return np.column_stack([np.ones(n, dtype=float), x])


def se_from_influence(influence: np.ndarray) -> float:
    """``sqrt(mean(IF**2) / n)``: the plug-in variance of the estimator."""
    inf = np.asarray(influence, dtype=float)
    n = inf.shape[0]
    return float(np.sqrt(np.mean(inf**2) / n))


class HajekMean:
    """``theta = E_n[a Z] / E_n[a]`` with its influence function.

    Parameters
    ----------
    a : ndarray
        Non-negative weights (one per row; zero outside the arm).
    z : ndarray
        The quantity being averaged.
    """

    def __init__(self, a: np.ndarray, z: np.ndarray) -> None:
        self.a = np.asarray(a, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.mass = float(self.a.mean())
        self.theta = float(np.mean(self.a * self.z) / self.mass)
        self.influence = self.a * (self.z - self.theta) / self.mass

    def grad_through_z(self, dz: np.ndarray) -> np.ndarray:
        """Gradient of ``theta`` when ``Z`` moves by ``dz @ delta``.

        ``dz`` is the ``n x k`` derivative of ``Z`` with respect to a
        parameter vector of length ``k``.
        """
        return (self.a[:, None] * dz).mean(axis=0) / self.mass

    def grad_through_a(self, da: np.ndarray) -> np.ndarray:
        """Gradient of ``theta`` when the weights move by ``da @ delta``."""
        return (da * (self.z - self.theta)[:, None]).mean(axis=0) / self.mass


def spanning_columns(x: np.ndarray, rows: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Columns of ``x`` kept for a least-squares fit on ``rows``.

    Columns are screened in order and one is dropped when, on the rows with
    positive weight, it is a linear combination of the columns already kept
    (so a covariate that is constant within the fitted rows is absorbed by
    the intercept). Predictions then use the kept columns only, the way a
    regression that reports aliased coefficients as missing predicts.
    """
    from ._etwfe_glm_fit import _independent_columns

    live = np.asarray(rows, dtype=bool) & (w > 0)
    return np.asarray(_independent_columns(x[live]), dtype=int)


def wls_coef(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """Weighted least squares of ``y`` on ``x`` over the selected rows."""
    rows = np.asarray(rows, dtype=bool)
    sw = np.sqrt(w[rows])
    coef, *_ = np.linalg.lstsq(x[rows] * sw[:, None], y[rows] * sw, rcond=None)
    return coef


def wls_influence(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    rows: np.ndarray,
    coef: np.ndarray,
) -> np.ndarray:
    """Influence function of a WLS coefficient vector (``n x k``).

    Row ``i`` is ``(E_n[w 1_rows x x'])^{-1} w_i 1_rows,i x_i e_i`` with
    ``e_i`` the residual; rows outside the fit contribute zero.
    """
    rows = np.asarray(rows, dtype=bool)
    n = x.shape[0]
    wr = np.where(rows, w, 0.0)
    gram = (x.T * wr) @ x / n
    resid = np.where(rows, y - x @ coef, 0.0)
    score = x * (wr * resid)[:, None]
    return np.linalg.solve(gram, score.T).T


def logit_fit(
    d: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    *,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted logit maximum likelihood by Newton's method.

    Returns ``(coef, fitted probabilities)``. Warns with
    :class:`~statspai.exceptions.ConvergenceWarning` when the iterations
    do not settle (typically separation).
    """
    coef = np.zeros(x.shape[1])
    converged = False
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-(x @ coef)))
        hess = (x.T * (w * p * (1.0 - p))) @ x
        step = np.linalg.solve(hess, x.T @ (w * (d - p)))
        coef = coef + step
        if np.max(np.abs(step)) <= tol * (1.0 + np.max(np.abs(coef))):
            converged = True
            break
    if not converged:
        warnings.warn(
            "propensity-score logit did not converge (separated or "
            "collinear covariates); the fitted scores are unreliable.",
            ConvergenceWarning,
            stacklevel=3,
        )
    return coef, 1.0 / (1.0 + np.exp(-(x @ coef)))


def logit_influence(
    x: np.ndarray, d: np.ndarray, w: np.ndarray, p: np.ndarray
) -> np.ndarray:
    """Influence function of the weighted logit MLE (``n x k``).

    Row ``i`` is ``H^{-1} w_i (d_i - p_i) x_i`` with
    ``H = E_n[w p (1 - p) x x']``.
    """
    n = x.shape[0]
    hess = (x.T * (w * p * (1.0 - p))) @ x / n
    score = x * (w * (d - p))[:, None]
    return np.linalg.solve(hess, score.T).T


def ipt_fit(
    d: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse probability tilting fit of a logit propensity score.

    Maximises ``E_n[w (D x'g - (1 - D) exp(x'g))]``, a strictly concave
    objective (Sant'Anna and Zhao 2020, Section 3.1). Its first-order
    condition ``E_n[w (D - (1 - D) exp(x'g)) x] = 0`` balances every
    covariate between the treated arm and the odds-weighted comparison
    arm. Newton's method with step halving, started at the logit MLE.

    Returns ``(coef, Lambda(x'g))``.
    """
    coef, _ = logit_fit(d, x, w)

    def objective(g: np.ndarray) -> float:
        lin = x @ g
        return float(np.sum(w * (d * lin - (1.0 - d) * np.exp(lin))))

    converged = False
    for _ in range(max_iter):
        odds = np.exp(x @ coef)
        grad = x.T @ (w * (d - (1.0 - d) * odds))
        hess = (x.T * (w * (1.0 - d) * odds)) @ x
        step = np.linalg.solve(hess, grad)
        base = objective(coef)
        t = 1.0
        while not (objective(coef + t * step) >= base) and t > 1e-10:
            t *= 0.5
        coef = coef + t * step
        if np.max(np.abs(t * step)) <= tol * (1.0 + np.max(np.abs(coef))):
            converged = True
            break
    if not converged:
        warnings.warn(
            "inverse probability tilting did not converge; the propensity "
            "scores of the improved DR estimator are unreliable.",
            ConvergenceWarning,
            stacklevel=3,
        )
    return coef, 1.0 / (1.0 + np.exp(-(x @ coef)))
