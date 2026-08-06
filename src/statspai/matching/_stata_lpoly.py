"""Stata ``lpoly``-compatible local polynomial smoother.

This exists for exactly one reason: ``psmatch2 ..., llr`` with its default
Epanechnikov kernel does not run local linear regression *matching*.  It
rewrites the request as nearest-neighbour matching on an ``lpoly``-smoothed
outcome (``psmatch2.ado``: "do nearest neighbor if llr with tricube"), so
reproducing a published ``psmatch2 ..., llr`` number requires reproducing
Stata's ``lpoly``.

Stata's Epanechnikov kernel — the ``lpoly`` default, and *not* the same
parameterisation psmatch2 uses internally for ``kernel`` matching — is

.. math:: K(u) = \\frac{3}{4}\\left(1 - \\frac{u^2}{5}\\right)\\Big/\\sqrt{5},
          \\qquad |u| \\le \\sqrt{5}

which has unit variance, so its effective support is :math:`\\pm\\sqrt5 h`
rather than :math:`\\pm h`.  Using psmatch2's compact ``epan`` here instead
silently changes the bandwidth by a factor of :math:`\\sqrt5`.

References
----------
Fan, J. and Gijbels, I. (1996). *Local Polynomial Modelling and Its
    Applications*. Chapman & Hall.  [``fan1996local``]
"""

from __future__ import annotations

import numpy as np

#: Half-width of Stata's unit-variance Epanechnikov kernel, in bandwidths.
_EPAN_SUPPORT = np.sqrt(5.0)


def stata_epanechnikov(u: np.ndarray) -> np.ndarray:
    """Stata's unit-variance Epanechnikov kernel ``K(u)``."""
    u = np.asarray(u, dtype=float)
    inside = np.abs(u) <= _EPAN_SUPPORT
    out: np.ndarray = np.zeros_like(u)
    out[inside] = 0.75 * (1.0 - u[inside] ** 2 / 5.0) / _EPAN_SUPPORT
    return out


def lpoly_predict(
    x: np.ndarray,
    y: np.ndarray,
    at: np.ndarray,
    bandwidth: float,
    degree: int = 1,
) -> np.ndarray:
    """Local polynomial fit of ``y`` on ``x``, evaluated at ``at``.

    Reproduces ``lpoly y x, degree(degree) at(at) bwidth(bandwidth)`` with
    Stata's default Epanechnikov kernel: at each evaluation point solve the
    kernel-weighted least squares problem

    .. math::

        \\min_{b} \\sum_j K\\!\\left(\\frac{x_j - x_0}{h}\\right)
            \\left(y_j - \\sum_{p=0}^{d} b_p (x_j - x_0)^p \\right)^2

    and return the intercept :math:`b_0`, which is the estimate of
    :math:`E[y \\mid x = x_0]`.

    Parameters
    ----------
    x, y : ndarray
        The fitting sample (Stata restricts this with ``if``; the caller is
        responsible for having subset it).
    at : ndarray
        Evaluation points.
    bandwidth : float
        Kernel bandwidth ``h``.
    degree : int, default 1
        Polynomial degree.  ``1`` is local linear, which is what psmatch2
        requests.

    Returns
    -------
    ndarray
        Fitted values at ``at``.  Entries where the local design is empty or
        rank-deficient come back ``nan`` — the same points where Stata's
        ``lpoly`` leaves the generated variable missing.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    at = np.asarray(at, dtype=float)
    h = float(bandwidth)
    if h <= 0:
        raise ValueError(f"lpoly bandwidth must be positive, got {bandwidth!r}")

    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]

    out: np.ndarray = np.full(at.shape, np.nan, dtype=float)
    if x.size == 0:
        return out

    d = max(int(degree), 0)
    for i, x0 in enumerate(at):
        if not np.isfinite(x0):
            continue
        w = stata_epanechnikov((x - x0) / h)
        nz = w > 0
        if nz.sum() < d + 1:
            # Too few donors inside the window to identify the local fit.
            continue
        dx = x[nz] - x0
        # Design [1, dx, dx^2, ...]; the intercept is the fitted value.
        design = np.vander(dx, N=d + 1, increasing=True)
        wt = w[nz]
        xtwx = design.T @ (design * wt[:, None])
        xtwy = design.T @ (y[nz] * wt)
        try:
            beta = np.linalg.solve(xtwx, xtwy)
        except np.linalg.LinAlgError:
            continue
        out[i] = float(beta[0])
    return out


def psmatch2_llr_smoothed_outcome(
    outcome: np.ndarray,
    pscore: np.ndarray,
    treated: np.ndarray,
    support: np.ndarray,
    bandwidth: float,
    degree: int = 1,
) -> np.ndarray:
    """psmatch2's ``_s_y``: the control-fitted local linear outcome curve.

    Mirrors::

        lpoly y _pscore if _treated==0 & _support==1, ///
              nograph deg(1) at(_pscore) gen(_s_y) bw(...)

    i.e. the smoother is fitted on the **on-support controls only** but
    evaluated at *every* unit's propensity score.
    """
    outcome = np.asarray(outcome, dtype=float)
    pscore = np.asarray(pscore, dtype=float)
    fit_rows = (np.asarray(treated) == 0) & np.asarray(support, dtype=bool)
    smoothed: np.ndarray = lpoly_predict(
        pscore[fit_rows],
        outcome[fit_rows],
        pscore,
        bandwidth=bandwidth,
        degree=degree,
    )
    return smoothed
