"""Compiled Frank-Wolfe loop for the synthetic-DiD weight problems.

Imported on the first weight solve of ``sp.sdid``, so that ``import
statspai`` does not import numba.

The loop is the one in ``sdid._sc_weight_fw_loop`` statement for statement.
Inner products go through ``np.dot`` (the BLAS routines NumPy itself calls)
and element-wise updates keep their original association, so the iterates,
and with them the weights, the estimate and every placebo / bootstrap draw,
are the ones the interpreted loop produces.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _matvec(M: np.ndarray, x: np.ndarray) -> np.ndarray:  # pragma: no cover - numba
    # NumPy's ``M @ x`` is a dot product, not gemv, when ``M`` has one row;
    # the two round differently, so follow NumPy.
    if M.shape[0] == 1:
        out = np.empty(1)
        out[0] = np.dot(M[0], x)
        return out
    full: np.ndarray = np.dot(M, x)
    return full


@njit(cache=True)
def fw_loop(
    A: np.ndarray,
    At: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray,
    ax: np.ndarray,
    eta: float,
    zeta2: float,
    min_dec2: float,
    max_iter: int,
) -> np.ndarray:  # pragma: no cover - numba
    """Frank-Wolfe on ``||A w - b||^2 / n + zeta^2 ||w||^2`` over the simplex.

    ``A`` and ``At`` (its transpose) are C-contiguous, ``ax = A @ weights``.
    ``weights`` and ``ax`` are overwritten; the final weights are returned.
    """
    n_rows, n_weights = A.shape
    resid = np.empty(n_rows)
    err_dir = np.empty(n_rows)
    half_grad = np.empty(n_weights)
    prev = 0.0
    have_prev = False
    for _ in range(max_iter):
        for r in range(n_rows):
            resid[r] = ax[r] - b[r]
        atr = _matvec(At, resid)
        for j in range(n_weights):
            half_grad[j] = atr[j] + eta * weights[j]
        vertex = np.argmin(half_grad)
        w_i = weights[vertex]
        ww = np.dot(weights, weights)
        dir_sq = ww - 2.0 * w_i + 1.0  # ||e_i - w||^2
        if dir_sq == 0.0:
            val = zeta2 * ww + np.dot(resid, resid) / n_rows
        else:
            for r in range(n_rows):
                err_dir[r] = A[r, vertex] - ax[r]
            denom = np.dot(err_dir, err_dir) + eta * dir_sq
            num = half_grad[vertex] - np.dot(half_grad, weights)
            step = 0.0 if denom <= 0 else -num / denom
            # min(1.0, max(0.0, step)) with Python's comparison order
            if not step > 0.0:
                step = 0.0
            if not step < 1.0:
                step = 1.0
            shrink = 1.0 - step
            for j in range(n_weights):
                weights[j] = weights[j] * shrink
            weights[vertex] += step
            for r in range(n_rows):
                ax[r] = ax[r] + step * err_dir[r]
                resid[r] = ax[r] - b[r]
            val = zeta2 * np.dot(weights, weights) + np.dot(resid, resid) / n_rows
        if have_prev and prev - val <= min_dec2:
            break
        prev = val
        have_prev = True
    return weights
