"""Compiled coordinate descent for the sparse synthetic-control LASSO.

Imported on the first LASSO solve of ``sp.synth(method='sparse')``, so that
``import statspai`` does not import numba.

The sweep is ``sparse._coordinate_descent_loop`` statement for statement.
Its one inner product, ``X[:, j] @ r``, is rounded by the BLAS ``ddot`` that
NumPy calls with the column's own stride, and a strided ``ddot`` does not
round like a contiguous one. The kernel therefore calls ``ddot`` itself on
the column in place (through SciPy's BLAS pointers) instead of ``np.dot``,
which would copy a strided column first. That keeps the weights identical
to the interpreted loop for every memory layout of ``X``.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np
from numba import njit
from numba.extending import get_cython_function_address

_PTR = ctypes.c_void_p
# double ddot(int *n, double *x, int *incx, double *y, int *incy)
ddot = ctypes.CFUNCTYPE(ctypes.c_double, _PTR, _PTR, _PTR, _PTR, _PTR)(
    get_cython_function_address("scipy.linalg.cython_blas", "ddot")
)


@njit(cache=True)
def cd_loop(
    dot: Any,
    X: np.ndarray,
    y: np.ndarray,
    col_norm_sq: np.ndarray,
    lambda_pen: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:  # pragma: no cover - numba
    """Cyclic coordinate descent on ``0.5 ||y - X w||^2 + lambda ||w||_1``.

    ``dot`` is the BLAS ``ddot`` above (a ctypes pointer has to arrive as an
    argument for the compiled code to be cached on disk). ``X`` needs
    positive strides that are multiples of 8 bytes.
    """
    T, J = X.shape
    w = np.zeros(J)
    w_old = np.empty(J)
    r = y.copy()
    n = np.array([T], dtype=np.int32)
    inc = np.array([X.strides[0] // 8], dtype=np.int32)
    one = np.array([1], dtype=np.int32)
    base = X.ctypes.data
    col_stride = X.strides[1]
    for _ in range(max_iter):
        for j in range(J):
            w_old[j] = w[j]
        for j in range(J):
            cn = col_norm_sq[j]
            if cn < 1e-12:
                continue
            wj = w[j]
            for i in range(T):
                r[i] += X[i, j] * wj
            rho = dot(
                n.ctypes.data,
                base + j * col_stride,
                inc.ctypes.data,
                r.ctypes.data,
                one.ctypes.data,
            )
            # sign(x) * maximum(|x| - lam, 0), as NumPy evaluates it
            x = rho / cn
            excess = abs(x) - lambda_pen / cn
            if not (excess >= 0.0 or np.isnan(excess)):
                excess = 0.0
            wj = np.sign(x) * excess
            w[j] = wj
            for i in range(T):
                r[i] -= X[i, j] * wj
        # np.max(np.abs(w - w_old)) < tol; a NaN makes the comparison false
        delta = 0.0
        for j in range(J):
            d = abs(w[j] - w_old[j])
            if np.isnan(d):
                delta = d
                break
            if d > delta:
                delta = d
        if delta < tol:
            break
    return w
