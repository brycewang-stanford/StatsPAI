"""Asymptotic (estimating-equation) variance of SIMEX.

Carroll, Kuechenhoff, Lombard and Stefanski stack the estimating equations
of the naive fit and of the ``B``-averaged fits at every ``lambda``, take
the sandwich of that system, and carry it through the least-squares fit of
the extrapolant by the delta method. The layout follows R ``simex``
(``asymptotic = TRUE``) line by line so that the two agree on the same
pseudo-errors:

* ``PSI`` is ``n x k (L + 1)``: the naive scores, then for every ``lambda``
  the per-observation scores averaged over the ``B`` simulated fits;
* ``A`` is block diagonal with blocks ``-(1 / n) sum_i w_i d_i^2 / v_i x_i
  x_i'`` (averaged over ``B`` for the simulated blocks);
* ``Sigma = A^{-1} cov(PSI) A^{-T}``, ``Sigma_gamma = D^{-1} S Sigma S'
  D^{-1}`` with ``D = S S'`` and ``S`` the derivative of the extrapolant
  in its parameters, and ``Var = G' Sigma_gamma G / n`` with ``G`` the
  derivative of the extrapolant's value at ``lambda = -1``.

The score is the likelihood score ``w x (y - mu) d / v``. R uses
``x (y - mu)`` and ignores prior weights; the two coincide for unweighted
least squares, logit and Poisson (canonical links) and differ for probit.

References
----------
[@carroll1996asymptotics]
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy import stats


def score_parts(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, family: str, beta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-observation scores (``n x k``) and the summed Jacobian (``k x k``)."""
    eta = X @ beta
    if family == "gaussian":
        resid = y - eta
        return X * (w * resid)[:, None], -(X.T * w) @ X
    eps = 1e-10
    if family == "logit":
        m = 1.0 / (1.0 + np.exp(-eta))
        d = m * (1.0 - m)
        v = d
    elif family == "probit":
        m = stats.norm.cdf(eta)
        d = stats.norm.pdf(eta)
        v = m * (1.0 - m)
    else:  # poisson
        m = np.exp(eta)
        d = m
        v = m
    d = np.maximum(d, eps)
    v = np.maximum(v, eps)
    psi = X * (w * (y - m) * d / v)[:, None]
    return psi, -(X.T * (w * d**2 / v)) @ X


def _curve_derivatives(
    method: str, grid: np.ndarray, params: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """``S_j`` rows (``n_gamma x (L + 1)``) and ``G_j`` (``n_gamma``).

    Signs follow R ``simex:::construct.s`` and the ``g`` vectors of
    ``simex``; they cancel in the variance.
    """
    if method == "quadratic":
        S = -np.vstack([np.ones_like(grid), grid, grid**2])
        return S, np.array([1.0, -1.0, 1.0])
    if method == "linear":
        S = -np.vstack([np.ones_like(grid), grid])
        return S, np.array([1.0, -1.0])
    if method == "nonlinear":
        _, b, c = params
        S = np.vstack([np.ones_like(grid), 1.0 / (c + grid), -b / (c + grid) ** 2])
        return S, np.array([-1.0, -1.0 / (c - 1.0), b / (c - 1.0) ** 2])
    # loglinear: exp(a + b lambda) - shift
    a, b = params[0], params[1]
    e = np.exp(a + b * grid)
    S = np.vstack([e, e * grid])
    ev = np.exp(a - b)
    return S, np.array([ev, -ev])


def asymptotic_covariance(
    psi_blocks: List[np.ndarray],
    jac_blocks: List[np.ndarray],
    grid: np.ndarray,
    curves: Sequence[Tuple[str, Sequence[float]]],
) -> np.ndarray:
    """Delta-method covariance of the extrapolated coefficients.

    ``psi_blocks[g]`` is the ``n x k`` score matrix at ``grid[g]`` (averaged
    over the simulations), ``jac_blocks[g]`` the matching ``k x k``
    Jacobian divided by ``n``; ``curves[j]`` names the extrapolant of
    coefficient ``j`` and its fitted parameters.
    """
    n, k = psi_blocks[0].shape
    n_grid = grid.size
    PSI = np.hstack(psi_blocks)
    c11 = np.cov(PSI, rowvar=False, ddof=1)
    a_inv = np.zeros((k * n_grid, k * n_grid))
    for g, block in enumerate(jac_blocks):
        sl = slice(g * k, (g + 1) * k)
        a_inv[sl, sl] = np.linalg.inv(block)
    sigma = a_inv @ c11 @ a_inv.T

    s_rows: List[np.ndarray] = []
    g_cols: List[np.ndarray] = []
    for j, (method, params) in enumerate(curves):
        S_j, g_j = _curve_derivatives(method, grid, params)
        rows = np.zeros((S_j.shape[0], k * n_grid))
        rows[:, j::k] = S_j
        s_rows.append(rows)
        g_cols.append(g_j)
    S = np.vstack(s_rows)
    G = np.zeros((S.shape[0], k))
    start = 0
    for j, g_j in enumerate(g_cols):
        G[start : start + g_j.size, j] = g_j
        start += g_j.size
    d_inv = np.linalg.inv(S @ S.T)
    sigma_gamma = d_inv @ S @ sigma @ S.T @ d_inv
    out: np.ndarray = G.T @ sigma_gamma @ G / n
    sym: np.ndarray = (out + out.T) / 2.0
    return sym
