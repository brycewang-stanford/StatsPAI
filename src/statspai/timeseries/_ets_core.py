"""Numerical kernels of the ETS (error, trend, seasonal) state space models.

The recursions are those of the innovations state space framework of
Hyndman, Koehler, Snyder and Grose (2002) in the parameterisation of
Hyndman, Koehler, Ord and Snyder (2008): one level, an optional growth
term (additive or multiplicative, possibly damped) and an optional
seasonal term (additive or multiplicative) driven by a single source of
error that is additive or multiplicative.

Integer codes keep the kernels inside numba: error ``0`` additive, ``1``
multiplicative; trend and season ``0`` none, ``1`` additive, ``2``
multiplicative.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numba import njit

_TOL = 1.0e-10
#: value of the criterion at parameters where the recursion breaks down
BAD = 1.0e300


@njit(cache=True)
def ets_filter(
    y: np.ndarray,
    m: int,
    error: int,
    trend: int,
    season: int,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    level0: float,
    growth0: float,
    season0: np.ndarray,
    states: np.ndarray,
    fitted: np.ndarray,
    resid: np.ndarray,
) -> Tuple[float, float, int]:
    """Run the filter; fill ``states`` (n + 1 rows), ``fitted``, ``resid``.

    ``season0`` holds ``s_0, s_{-1}, ..., s_{-m+1}``; ``states`` columns are
    the level, the growth term and the seasonal terms most recent first.

    Returns ``(sum of squared innovations, sum of log |fitted|, status)``;
    a non-zero status marks a breakdown (a zero forecast under a
    multiplicative term, a non-positive multiplicative growth).
    """
    n = y.shape[0]
    lev = level0
    b = growth0
    ns = m if season > 0 else 0
    s = np.empty(max(ns, 1))
    for j in range(ns):
        s[j] = season0[j]
    states[0, 0] = lev
    states[0, 1] = b
    for j in range(ns):
        states[0, 2 + j] = s[j]
    sse = 0.0
    sumlog = 0.0
    for t in range(n):
        if trend == 0:
            phib = 0.0
            q = lev
        elif trend == 1:
            phib = phi * b
            q = lev + phib
        else:
            if b <= 0.0:
                return sse, sumlog, 1
            phib = b**phi
            q = lev * phib
        if season == 0:
            f = q
        elif season == 1:
            f = q + s[ns - 1]
        else:
            f = q * s[ns - 1]
        if not np.isfinite(f):
            return sse, sumlog, 1
        if abs(f) < _TOL and (error == 1):
            return sse, sumlog, 1
        if error == 0:
            e = y[t] - f
        else:
            e = (y[t] - f) / f
        fitted[t] = f
        resid[t] = e
        sse += e * e
        sumlog += np.log(abs(f)) if error == 1 else 0.0
        # level
        if season == 0:
            p = y[t]
        elif season == 1:
            p = y[t] - s[ns - 1]
        else:
            if abs(s[ns - 1]) < _TOL:
                return sse, sumlog, 1
            p = y[t] / s[ns - 1]
        new_l = q + alpha * (p - q)
        # growth
        if trend > 0:
            if trend == 1:
                r = new_l - lev
            else:
                if abs(lev) < _TOL:
                    return sse, sumlog, 1
                r = new_l / lev
            b = phib + (beta / alpha) * (r - phib)
        # season
        if season > 0:
            if season == 1:
                tt = y[t] - q
            else:
                if abs(q) < _TOL:
                    return sse, sumlog, 1
                tt = y[t] / q
            new_s = s[ns - 1] + gamma * (tt - s[ns - 1])
            for j in range(ns - 1, 0, -1):
                s[j] = s[j - 1]
            s[0] = new_s
        lev = new_l
        states[t + 1, 0] = lev
        states[t + 1, 1] = b
        for j in range(ns):
            states[t + 1, 2 + j] = s[j]
    return sse, sumlog, 0


def neg2loglik(sse: float, sumlog: float, n: int, error: int) -> float:
    """Minus twice the concentrated log likelihood, up to the constant
    that Hyndman et al. (2008, eq. 5.3) drop: ``n log(sum e^2)``, plus
    ``2 sum log|fitted|`` under a multiplicative error."""
    if not sse > 0.0:
        return -BAD
    out = n * np.log(sse)
    if error == 1:
        out += 2.0 * sumlog
    return float(out)


@njit(cache=True)
def _seasonal_roots_inside(
    alpha: float, beta: float, gamma: float, phi: float, m: int
) -> bool:
    """Whether the characteristic polynomial of a seasonal model has all
    its roots inside the unit circle.

    The Schur-Cohn step-down recursion settles it in ``O(m^2)`` when no
    root is near the circle; otherwise the companion-matrix eigenvalues
    decide, with the tolerance ``1 + 1e-10`` on the largest modulus.
    """
    d = m + 1
    # monic polynomial z^d + c[d-1] z^{d-1} + ... + c[0]
    c = np.empty(d)
    c[0] = phi * (1.0 - alpha - gamma)
    c[1] = alpha + beta - alpha * phi + gamma - 1.0
    for j in range(2, m):
        c[j] = alpha + beta - alpha * phi
    c[m] = alpha + beta - phi
    # a[0] = 1 (leading), a[j] = coefficient of z^{d-j}
    a = np.empty(d + 1)
    a[0] = 1.0
    for j in range(1, d + 1):
        a[j] = c[d - j]
    clear = True
    work = np.empty(d + 1)
    for order in range(d, 0, -1):
        k = a[order]
        if not abs(k) < 1.0 - 1e-7:
            clear = False
            break
        den = 1.0 - k * k
        for j in range(1, order):
            work[j] = (a[j] - k * a[order - j]) / den
        for j in range(1, order):
            a[j] = work[j]
    if clear:
        return True
    comp = np.zeros((d, d), dtype=np.complex128)
    for j in range(d):
        comp[0, j] = -c[d - 1 - j]
    for j in range(1, d):
        comp[j, j - 1] = 1.0
    ev = np.linalg.eigvals(comp)
    return bool(np.max(np.abs(ev)) <= 1.0 + 1e-10)


@njit(cache=True)
def _admissible(
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    has_beta: bool,
    has_gamma: bool,
    m: int,
) -> bool:
    if phi < 0.0 or phi > 1.0 + 1e-8:
        return False
    if not has_gamma:
        if alpha < 1.0 - 1.0 / phi or alpha > 1.0 + 1.0 / phi:
            return False
        if has_beta:
            if beta < alpha * (phi - 1.0) or beta > (1.0 + phi) * (2.0 - alpha):
                return False
        return True
    if m > 1:
        be = beta if has_beta else 0.0
        if gamma < max(1.0 - 1.0 / phi - alpha, 0.0):
            return False
        if gamma > 1.0 + 1.0 / phi - alpha:
            return False
        if alpha < 1.0 - 1.0 / phi - gamma * (1.0 - m + phi + phi * m) / (
            2.0 * phi * m
        ):
            return False
        if be < -(1.0 - phi) * (gamma / m + alpha):
            return False
        if not _seasonal_roots_inside(alpha, be, gamma, phi, m):
            return False
    return True


def admissible(
    alpha: float,
    beta: Optional[float],
    gamma: Optional[float],
    phi: Optional[float],
    m: int,
) -> bool:
    """Whether the smoothing parameters give a forecastable (stable) model.

    The conditions are those of Hyndman, Akram and Archibald (2008) for the
    linear models: closed-form inequalities, and for seasonal models a
    check that the roots of the characteristic polynomial lie inside the
    unit circle.
    """
    return bool(
        _admissible(
            float(alpha),
            0.0 if beta is None else float(beta),
            0.0 if gamma is None else float(gamma),
            1.0 if phi is None else float(phi),
            beta is not None,
            gamma is not None,
            int(m),
        )
    )


@njit(cache=True)
def ets_objective(
    x: np.ndarray,
    y: np.ndarray,
    m: int,
    error: int,
    trend: int,
    season: int,
    pos: np.ndarray,
    fixed: np.ndarray,
    has: np.ndarray,
    bounds: int,
    lower: np.ndarray,
    upper: np.ndarray,
    states: np.ndarray,
    fitted: np.ndarray,
    resid: np.ndarray,
) -> float:
    """Minus twice the concentrated log likelihood at the free vector
    ``x``, or ``BAD`` outside the parameter region.

    ``pos[i]`` is the position in ``x`` of smoothing parameter ``i``
    (alpha, beta, gamma, phi), ``-1`` when it is fixed at ``fixed[i]``;
    ``has[i]`` says whether the model has it. The initial states follow
    the smoothing parameters. ``bounds``: 0 both, 1 usual, 2 admissible.
    """
    par = np.empty(4)
    k = 0
    for i in range(4):
        if pos[i] >= 0:
            par[i] = x[pos[i]]
            k += 1
        else:
            par[i] = fixed[i]
    alpha, beta, gamma, phi = par[0], par[1], par[2], par[3]
    if not has[3]:
        phi = 1.0
    if bounds != 2:
        if alpha < lower[0] or alpha > upper[0]:
            return BAD
        if has[1] and (beta < lower[1] or beta > alpha or beta > upper[1]):
            return BAD
        if has[3] and (phi < lower[3] or phi > upper[3]):
            return BAD
        if has[2] and (gamma < lower[2] or gamma > 1.0 - alpha or gamma > upper[2]):
            return BAD
    if bounds != 1:
        if not _admissible(alpha, beta, gamma, phi, has[1], has[2], m):
            return BAD
    l0 = x[k]
    k += 1
    b0 = 0.0
    if trend > 0:
        b0 = x[k]
        k += 1
        if trend == 2 and b0 <= 0.0:
            return BAD
    s0 = np.zeros(max(m, 1))
    if season > 0:
        tot = 0.0
        for j in range(m - 1):
            s0[j] = x[k + j]
            tot += s0[j]
        s0[m - 1] = (m - tot) if season == 2 else -tot
        if season == 2:
            for j in range(m):
                if s0[j] < 0.0:
                    return BAD
    sse, sumlog, bad = ets_filter(
        y,
        m,
        error,
        trend,
        season,
        alpha,
        beta,
        gamma,
        phi,
        l0,
        b0,
        s0,
        states,
        fitted,
        resid,
    )
    if bad != 0 or not np.isfinite(sse) or not sse > 0.0:
        return BAD
    val = y.shape[0] * np.log(sse)
    if error == 1:
        val += 2.0 * sumlog
    if not np.isfinite(val):
        return BAD
    return float(max(val, -1.0e10))


def point_forecast(
    h: int,
    m: int,
    trend: int,
    season: int,
    phi: float,
    last_state: np.ndarray,
) -> np.ndarray:
    """Forecasts from the last state with every future innovation at zero."""
    lev = float(last_state[0])
    b = float(last_state[1])
    s = np.asarray(last_state[2 : 2 + m], dtype=float) if season > 0 else None
    out = np.empty(h)
    phisum = 0.0
    for k in range(1, h + 1):
        phisum += phi**k
        if trend == 0:
            q = lev
        elif trend == 1:
            q = lev + phisum * b
        else:
            q = lev * b**phisum
        if season == 0 or s is None:
            out[k - 1] = q
        else:
            # s[0] is s_n, s[m - 1] is s_{n - m + 1}
            sj = s[m - 1 - ((k - 1) % m)]
            out[k - 1] = q + sj if season == 1 else q * sj
    return out


def _cvals(
    h: int,
    m: int,
    trend: int,
    season: int,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
) -> np.ndarray:
    """``c_j`` of Hyndman et al. (2008, Table 6.2) for ``j = 1..h``."""
    j = np.arange(1, h + 1, dtype=float)
    c: np.ndarray = np.full(h, alpha, dtype=float)
    if trend == 1:
        c = c + beta * np.cumsum(phi**j)
    if season > 0:
        c = c + gamma * ((np.arange(1, h + 1) % m) == 0)
    return c


def class1_variance(
    h: int,
    m: int,
    trend: int,
    season: int,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    sigma2: float,
) -> np.ndarray:
    """Forecast variance of the linear homoscedastic models (additive
    error, no multiplicative component)."""
    c = _cvals(h, m, trend, season, alpha, beta, gamma, phi)
    cum = np.concatenate([[0.0], np.cumsum(c[: h - 1] ** 2)])
    return sigma2 * (1.0 + cum)


def class2_variance(
    mu: np.ndarray,
    m: int,
    trend: int,
    season: int,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    sigma2: float,
) -> np.ndarray:
    """Forecast variance of the linear heteroscedastic models
    (multiplicative error, additive trend and season)."""
    h = mu.shape[0]
    c = _cvals(h, m, trend, season, alpha, beta, gamma, phi)
    theta = np.empty(h)
    theta[0] = mu[0] ** 2
    for k in range(1, h):
        theta[k] = mu[k] ** 2 + sigma2 * float(np.sum(c[:k] ** 2 * theta[k - 1 :: -1]))
    return np.asarray((1.0 + sigma2) * theta - mu**2, dtype=float)


def class3_moments(
    h: int,
    m: int,
    trend: int,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    sigma2: float,
    last_state: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and variance of the forecasts of the models with a
    multiplicative error and a multiplicative season (Hyndman et al.,
    2008, section 6.4.4): the level-growth block and the seasonal block
    are propagated as a Kronecker product."""
    k1 = 2 if trend > 0 else 1
    H1 = np.ones((1, k1))
    H2 = np.zeros((1, m))
    H2[0, m - 1] = 1.0
    if trend == 0:
        F1 = np.array([[1.0]])
        G1 = np.array([[alpha]])
    else:
        F1 = np.array([[1.0, 1.0], [0.0, phi]])
        G1 = np.array([[alpha, alpha], [beta, beta]])
    F2 = np.zeros((m, m))
    F2[0, m - 1] = 1.0
    F2[1:, : m - 1] = np.eye(m - 1)
    G2 = np.zeros((m, m))
    G2[0, m - 1] = gamma
    x1 = np.asarray(last_state[:k1], dtype=float).reshape(-1, 1)
    x2 = np.asarray(last_state[2 : 2 + m], dtype=float).reshape(1, -1)
    Mh = x1 @ x2
    Vh = np.zeros((Mh.size, Mh.size))
    H21 = np.kron(H2, H1)
    F21 = np.kron(F2, F1)
    G21 = np.kron(G2, G1)
    K = np.kron(G2, F1) + np.kron(F2, G1)
    mu = np.empty(h)
    var = np.empty(h)
    for i in range(h):
        mu[i] = (H1 @ Mh @ H2.T).item()
        var[i] = (1.0 + sigma2) * (H21 @ Vh @ H21.T).item() + sigma2 * mu[i] ** 2
        vec = Mh.reshape(-1, 1, order="F")
        outer = vec @ vec.T
        Vh = F21 @ Vh @ F21.T + sigma2 * (
            F21 @ Vh @ G21.T
            + G21 @ Vh @ F21.T
            + K @ (Vh + outer) @ K.T
            + sigma2 * G21 @ (3.0 * Vh + 2.0 * outer) @ G21.T
        )
        Mh = F1 @ Mh @ F2.T + G1 @ Mh @ G2.T * sigma2
    return mu, var


def simulate_paths(
    h: int,
    m: int,
    error: int,
    trend: int,
    season: int,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    last_state: np.ndarray,
    innovations: np.ndarray,
) -> np.ndarray:
    """Future sample paths, one per row of ``innovations`` (paths by h)."""
    n_paths = innovations.shape[0]
    lev = np.full(n_paths, float(last_state[0]))
    b = np.full(n_paths, float(last_state[1]))
    ns = m if season > 0 else 0
    s = np.tile(np.asarray(last_state[2 : 2 + ns], dtype=float), (n_paths, 1))
    out = np.empty((n_paths, h))
    with np.errstate(all="ignore"):
        for k in range(h):
            if trend == 0:
                phib = np.zeros(n_paths)
                q = lev
            elif trend == 1:
                phib = phi * b
                q = lev + phib
            else:
                phib = np.where(b > 0, b, np.nan) ** phi
                q = lev * phib
            if season == 0:
                f = q
            elif season == 1:
                f = q + s[:, ns - 1]
            else:
                f = q * s[:, ns - 1]
            e = innovations[:, k]
            yk = f + e if error == 0 else f * (1.0 + e)
            out[:, k] = yk
            if season == 0:
                p = yk
            elif season == 1:
                p = yk - s[:, ns - 1]
            else:
                p = yk / s[:, ns - 1]
            new_l = q + alpha * (p - q)
            if trend > 0:
                r = new_l - lev if trend == 1 else new_l / lev
                b = phib + (beta / alpha) * (r - phib)
            if season > 0:
                tt = yk - q if season == 1 else yk / q
                new_s = s[:, ns - 1] + gamma * (tt - s[:, ns - 1])
                s = np.column_stack([new_s, s[:, : ns - 1]])
            lev = new_l
    return out
