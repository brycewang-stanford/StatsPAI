"""Exact Gaussian likelihood of a stationary, invertible ARMA process by
the innovations algorithm.

``sp.arima`` maximises this likelihood on the differenced series. It is the
same function a Kalman filter started from the stationary distribution
evaluates (the two agree to rounding error), at a fraction of the cost: the
transformed process is a moving average of order ``q`` after the first
``max(p, q)`` observations, so each step costs ``O(q^2)`` instead of a
state-covariance update (Brockwell and Davis, 1991, sections 5.2 and 8.7;
bib key ``brockwell1991time``).

Seasonal models enter through their expanded polynomials.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import njit


@njit(cache=True)
def expand(
    v: np.ndarray, p: int, q: int, P: int, Q: int, s: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Coefficients of the expanded AR and MA polynomials.

    ``v`` holds ``ar, ma, seasonal ar, seasonal ma``. Returns ``phi`` with
    ``x_t = sum phi_j x_{t-j} + ...`` and ``theta`` with
    ``... + e_t + sum theta_j e_{t-j}``.
    """
    ar = np.zeros(p + P * s + 1)
    ar[0] = 1.0
    for j in range(p):
        ar[j + 1] = -v[j]
    if P > 0:
        out = np.zeros(p + P * s + 1)
        for i in range(p + 1):
            out[i] += ar[i]
            for k in range(P):
                out[i + (k + 1) * s] -= ar[i] * v[p + q + k]
        ar = out
    ma = np.zeros(q + Q * s + 1)
    ma[0] = 1.0
    for j in range(q):
        ma[j + 1] = v[p + j]
    if Q > 0:
        out = np.zeros(q + Q * s + 1)
        for i in range(q + 1):
            out[i] += ma[i]
            for k in range(Q):
                out[i + (k + 1) * s] += ma[i] * v[p + q + P + k]
        ma = out
    return -ar[1:], ma[1:]


@njit(cache=True)
def arma_loglike(
    x: np.ndarray, phi: np.ndarray, theta: np.ndarray
) -> Tuple[float, float]:
    """Concentrated exact log likelihood and the innovation variance.

    Returns ``(loglik, sigma2)`` with ``sigma2 = S / n`` the maximum
    likelihood innovation variance given the coefficients; ``(-inf, nan)``
    when the recursion breaks down.
    """
    n = x.shape[0]
    p = phi.shape[0]
    q = theta.shape[0]
    m = max(p, q)
    bad = (-np.inf, np.nan)
    if m == 0:
        s0 = 0.0
        for t in range(n):
            s0 += x[t] * x[t]
        if not s0 > 0.0:
            return bad
        sig = s0 / n
        return -0.5 * n * (np.log(2.0 * np.pi * sig) + 1.0), sig
    th = np.zeros(q + 1)
    th[0] = 1.0
    for j in range(q):
        th[j + 1] = theta[j]
    # psi weights up to q
    psi = np.zeros(q + 1)
    psi[0] = 1.0
    for j in range(1, q + 1):
        acc = th[j]
        for k in range(1, min(j, p) + 1):
            acc += phi[k - 1] * psi[j - k]
        psi[j] = acc
    # b_h = Cov(phi(B) X_t, X_{t-h}) = sum_{j >= h} theta_j psi_{j-h}
    b = np.zeros(max(m, q) + 2)
    for h in range(q + 1):
        acc = 0.0
        for j in range(h, q + 1):
            acc += th[j] * psi[j - h]
        b[h] = acc
    # autocovariances gamma(0..m) with unit innovation variance
    gam = np.zeros(m + 1)
    A = np.zeros((p + 1, p + 1))
    rhs = np.zeros(p + 1)
    for k in range(p + 1):
        A[k, k] += 1.0
        for j in range(1, p + 1):
            A[k, abs(k - j)] -= phi[j - 1]
        rhs[k] = b[k] if k <= q else 0.0
    if not abs(np.linalg.det(A)) > 1e-300:
        return bad  # a unit root: no stationary covariance
    g0 = np.linalg.solve(A, rhs)
    for k in range(min(p, m) + 1):
        gam[k] = g0[k]
    for k in range(p + 1, m + 1):
        acc = b[k] if k <= q else 0.0
        for j in range(1, p + 1):
            acc += phi[j - 1] * gam[k - j]
        gam[k] = acc
    if not gam[0] > 0.0:
        return bad
    # autocovariances of the moving-average part
    c = np.zeros(q + 1)
    for h in range(q + 1):
        acc = 0.0
        for r in range(q - h + 1):
            acc += th[r] * th[r + h]
        c[h] = acc
    L = max(m - 1, q)  # longest lag an innovations coefficient can have
    coef = np.zeros((n, L + 1))  # coef[k, j] = theta_{k, j}
    v = np.zeros(n)
    e = np.zeros(n)
    v[0] = gam[0]
    e[0] = x[0]
    ssq = e[0] * e[0] / v[0]
    slog = np.log(v[0])
    for nn in range(1, n):
        # row nn: coefficients theta_{nn, nn - k}, k = lo .. nn - 1
        lo = 0 if nn < m else max(0, nn - q)
        i1 = nn + 1
        for k in range(lo, nn):
            j1 = k + 1
            h = i1 - j1
            if i1 <= m:
                kap = gam[h]
            elif j1 <= m:
                kap = b[h] if h <= q else 0.0
            else:
                kap = c[h] if h <= q else 0.0
            acc = kap
            jlo = lo
            if k >= m:
                jlo = max(jlo, k - q)
            for j in range(jlo, k):
                acc -= coef[k, k - j] * coef[nn, nn - j] * v[j]
            coef[nn, nn - k] = acc / v[k]
        if i1 <= m:
            vn = gam[0]
        else:
            vn = c[0]
        for j in range(lo, nn):
            vn -= coef[nn, nn - j] * coef[nn, nn - j] * v[j]
        if not vn > 0.0 or not np.isfinite(vn):
            return bad
        v[nn] = vn
        pred = 0.0
        if nn >= m:
            for r in range(p):
                pred += phi[r] * x[nn - 1 - r]
        for j in range(1, nn - lo + 1):
            pred += coef[nn, j] * e[nn - j]
        e[nn] = x[nn] - pred
        ssq += e[nn] * e[nn] / vn
        slog += np.log(vn)
    if not ssq > 0.0 or not np.isfinite(ssq):
        return bad
    sig = ssq / n
    return -0.5 * (n * (np.log(2.0 * np.pi * sig) + 1.0) + slog), sig
