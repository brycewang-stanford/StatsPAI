"""Gibbs kernels of the TVP-VAR with stochastic volatility.

Model, for ``t = 1 .. T`` (``K`` variables, ``k = K p + 1`` regressors)::

    y_t = Z_t b_t + A_t^{-1} diag(sigma_t) eps_t,     eps_t ~ N(0, I)
    b_t = b_{t-1} + nu_t,            nu_t   ~ N(0, Q)
    a_t = a_{t-1} + zeta_t,          zeta_t ~ N(0, S)   (S block diagonal)
    h_t = h_{t-1} + eta_t,           eta_t  ~ N(0, W),   h_t = log sigma_t

``b_t`` stacks the rows of the ``K x k`` coefficient matrix (equation by
equation, lags first, constant last), ``Z_t = I_K (x) x_t'``, and ``a_t``
stacks the free elements of the unit lower-triangular ``A_t`` row by row.
The priors are on the states of the first date: ``b_1 ~ N(b0, PB)``,
``a_1 ~ N(a0, PA)``, ``h_1 ~ N(h0, PH)``, so each hyperparameter update
sees ``T - 1`` increments.

Everything that loops over dates is compiled by numba on first use; the
plain functions (suffix ``_py``) are the source and are never called
directly. The seven-normal approximation of the log chi-square(1) density
is the one of :mod:`statspai.mcmc.sv`, imported from there.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

__all__ = [
    "kernels",
    "mixture",
    "indicator_probs",
    "iw_posterior",
    "training_prior",
    "simple_prior",
    "lag_matrix",
]

_K: Dict[str, Any] = {}
# rebound to the compiled functions by kernels() before anything that calls
# them is compiled (numba reads a function's globals when it compiles it)
_chol: Any = None
_ck: Any = None
_riw: Any = None
_draw_ind: Any = None
_explosive: Any = None
_sweep: Any = None


def mixture() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weights, means and variances of the seven-normal approximation."""
    from ..mcmc.sv import _MIX_M, _MIX_Q, _MIX_V

    return (
        np.ascontiguousarray(_MIX_Q, dtype=float),
        np.ascontiguousarray(_MIX_M, dtype=float),
        np.ascontiguousarray(_MIX_V, dtype=float),
    )


def indicator_probs(resid: np.ndarray) -> np.ndarray:
    """Posterior probabilities of the seven components.

    ``resid`` is ``log(ystar^2 + offset) - 2 h``; the result has one more
    axis, of length 7, and sums to one along it.
    """
    q, mm, vv = mixture()
    d = np.asarray(resid, dtype=float)[..., None] - mm
    logp = np.log(q) - 0.5 * np.log(vv) - 0.5 * d * d / vv
    logp -= logp.max(axis=-1, keepdims=True)
    p = np.exp(logp)
    out: np.ndarray = p / p.sum(axis=-1, keepdims=True)
    return out


def iw_posterior(
    scale: np.ndarray, df: float, path: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Inverse-Wishart posterior of a random-walk innovation covariance.

    Prior ``IW(scale, df)`` and a state path of ``T`` rows give
    ``IW(scale + sum_t d_t d_t', df + T - 1)`` with ``d_t`` the ``T - 1``
    increments of the path.
    """
    d = np.diff(np.asarray(path, dtype=float), axis=0)
    return np.asarray(scale, dtype=float) + d.T @ d, float(df) + d.shape[0]


# --------------------------------------------------------------------------
# Kernels (compiled lazily)
# --------------------------------------------------------------------------
def _chol_py(A: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor of a positive semi-definite matrix.

    A pivot that is not positive (a direction with no variance, or
    rounding) gives a zero column instead of an error.
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    for j in range(n):
        d = A[j, j]
        for k in range(j):
            d -= L[j, k] * L[j, k]
        if d <= 0.0:
            continue
        d = np.sqrt(d)
        L[j, j] = d
        for i in range(j + 1, n):
            v = A[i, j]
            for k in range(j):
                v -= L[i, k] * L[j, k]
            L[i, j] = v / d
    return L


def _ck_py(
    rng: Any,
    y: np.ndarray,
    Z: np.ndarray,
    R: np.ndarray,
    Qc: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
) -> np.ndarray:
    """One draw of a random-walk state path (Carter-Kohn).

    ``y_t = Z_t x_t + N(0, R_t)``, ``x_t = x_{t-1} + N(0, Qc)``,
    ``x_1 ~ N(m0, P0)``. Shapes: ``y`` (T, n), ``Z`` (T, n, m), ``R``
    (T, n, n). Forward Kalman filter, then the states are drawn backwards
    from ``x_t | x_{t+1}, y_1..t``.
    """
    T = y.shape[0]
    m = m0.shape[0]
    mf = np.zeros((T, m))
    Pf = np.zeros((T, m, m))
    mp = m0.copy()
    Pp = P0.copy()
    for t in range(T):
        if t > 0:
            mp = mf[t - 1].copy()
            Pp = Pf[t - 1] + Qc
        Zt = np.ascontiguousarray(Z[t])
        PZ = Pp @ Zt.T.copy()
        F = Zt @ PZ + R[t]
        F = 0.5 * (F + F.T)
        v = y[t] - Zt @ mp
        G = np.linalg.solve(F, PZ.T.copy())  # F^{-1} Z P
        mf[t] = mp + G.T.copy() @ v
        Pn = Pp - PZ @ G
        Pf[t] = 0.5 * (Pn + Pn.T)
    out = np.zeros((T, m))
    out[T - 1] = mf[T - 1] + _chol(Pf[T - 1]) @ rng.standard_normal(m)
    for t in range(T - 2, -1, -1):
        Pt = np.ascontiguousarray(Pf[t])
        G = np.linalg.solve(Pt + Qc, Pt)  # (P + Q)^{-1} P
        mean = mf[t] + G.T.copy() @ (out[t + 1] - mf[t])
        V = Pt - Pt @ G
        V = 0.5 * (V + V.T)
        out[t] = mean + _chol(V) @ rng.standard_normal(m)
    return out


def _riw_py(rng: Any, df: float, scale: np.ndarray) -> np.ndarray:
    """One draw from IW(scale, df), mean ``scale / (df - p - 1)``."""
    p = scale.shape[0]
    inv = np.linalg.inv(scale)
    L = _chol(0.5 * (inv + inv.T))
    Am = np.zeros((p, p))
    for i in range(p):
        Am[i, i] = np.sqrt(rng.chisquare(df - i))
        for j in range(i):
            Am[i, j] = rng.standard_normal()
    LA = L @ Am
    out = np.linalg.inv(LA @ LA.T)
    return 0.5 * (out + out.T)


def _draw_ind_py(
    rng: Any,
    resid: np.ndarray,
    q: np.ndarray,
    mm: np.ndarray,
    vv: np.ndarray,
    s: np.ndarray,
) -> None:
    """Mixture indicators given ``resid = log(ystar^2 + c) - 2 h``."""
    T, K = resid.shape
    nc = q.shape[0]
    p = np.zeros(nc)
    for t in range(T):
        for i in range(K):
            top = -np.inf
            for j in range(nc):
                d = resid[t, i] - mm[j]
                p[j] = np.log(q[j]) - 0.5 * np.log(vv[j]) - 0.5 * d * d / vv[j]
                if p[j] > top:
                    top = p[j]
            tot = 0.0
            for j in range(nc):
                p[j] = np.exp(p[j] - top)
                tot += p[j]
            u = rng.random() * tot
            acc = 0.0
            pick = nc - 1
            for j in range(nc):
                acc += p[j]
                if u < acc:
                    pick = j
                    break
            s[t, i] = pick


def _explosive_py(B: np.ndarray, K: int, lags: int) -> bool:
    """True when the VAR frozen at some date has a root on or outside
    the unit circle. ``B`` is (T, K k), rows of the coefficient matrix."""
    k = K * lags + 1
    n = K * lags
    comp = np.zeros((n, n), dtype=np.complex128)
    for i in range(K, n):
        comp[i, i - K] = 1.0
    for t in range(B.shape[0]):
        for i in range(K):
            for j in range(n):
                comp[i, j] = B[t, i * k + j]
        if np.abs(np.linalg.eigvals(comp)).max() >= 1.0:
            return True
    return False


def _sweep_py(
    rng: Any,
    Y: np.ndarray,
    Z: np.ndarray,
    B: np.ndarray,
    a: np.ndarray,
    h: np.ndarray,
    s: np.ndarray,
    Qc: np.ndarray,
    Sc: np.ndarray,
    Wc: np.ndarray,
    b0: np.ndarray,
    PB: np.ndarray,
    a0: np.ndarray,
    PA: np.ndarray,
    h0: np.ndarray,
    PH: np.ndarray,
    Qs: np.ndarray,
    Qdf: float,
    Ss: np.ndarray,
    Sdf: np.ndarray,
    Ws: np.ndarray,
    Wdf: float,
    q: np.ndarray,
    mm: np.ndarray,
    vv: np.ndarray,
    offset: float,
    ordering: int,
    stationary: bool,
    lags: int,
    max_tries: int,
) -> int:
    """One sweep; the state arrays are updated in place.

    ``ordering`` 0: coefficients, covariances, indicators, volatilities,
    hyperparameters (the indicators are drawn immediately before the
    volatilities that condition on them). ``ordering`` 1 is the order of
    the 2005 paper, kept only so that the tests can show it is wrong:
    coefficients, covariances, volatilities given the indicators of the
    previous sweep, indicators, hyperparameters.

    Returns the number of coefficient paths drawn (0 when ``stationary``
    and none of ``max_tries`` paths was stable: the previous one is kept).
    """
    T, K = Y.shape
    # ---- 1. coefficient path given A^T, Sigma^T, Q
    R = np.zeros((T, K, K))
    Ainv = np.zeros((T, K, K))
    for t in range(T):
        A = np.eye(K)
        pos = 0
        for i in range(1, K):
            for j in range(i):
                A[i, j] = a[t, pos]
                pos += 1
        Ai = np.linalg.inv(A)
        for i in range(K):
            sd = np.exp(h[t, i])
            for j in range(K):
                Ai[j, i] *= sd  # A^{-1} diag(sigma)
        Ainv[t] = Ai
        R[t] = Ai @ Ai.T
    tries = 0
    if stationary:
        found = False
        for _ in range(max_tries):
            cand = _ck(rng, Y, Z, R, Qc, b0, PB)
            tries += 1
            if not _explosive(cand, K, lags):
                B[:, :] = cand
                found = True
                break
        if not found:
            tries = 0
    else:
        B[:, :] = _ck(rng, Y, Z, R, Qc, b0, PB)
        tries = 1
    # ---- 2. covariance states, one equation at a time
    yhat = np.zeros((T, K))
    for t in range(T):
        yhat[t] = Y[t] - np.ascontiguousarray(Z[t]) @ B[t]
    pos = 0
    for i in range(1, K):
        yi = np.ascontiguousarray(yhat[:, i : i + 1])
        Zi = np.zeros((T, 1, i))
        Ri = np.zeros((T, 1, 1))
        for t in range(T):
            for j in range(i):
                Zi[t, 0, j] = -yhat[t, j]
            Ri[t, 0, 0] = np.exp(2.0 * h[t, i])
        blk = _ck(
            rng,
            yi,
            Zi,
            Ri,
            np.ascontiguousarray(Sc[pos : pos + i, pos : pos + i]),
            np.ascontiguousarray(a0[pos : pos + i]),
            np.ascontiguousarray(PA[pos : pos + i, pos : pos + i]),
        )
        a[:, pos : pos + i] = blk
        pos += i
    # ---- 3. volatilities by the mixture approximation
    ystar2 = np.zeros((T, K))
    for t in range(T):
        pos = 0
        for i in range(K):
            v = yhat[t, i]
            for j in range(i):
                v += a[t, pos] * yhat[t, j]
                pos += 1
            ystar2[t, i] = np.log(v * v + offset)
    if ordering == 0:
        _draw_ind(rng, ystar2 - 2.0 * h, q, mm, vv, s)
    Zh = np.zeros((T, K, K))
    Rh = np.zeros((T, K, K))
    yh = np.zeros((T, K))
    for t in range(T):
        for i in range(K):
            Zh[t, i, i] = 2.0
            Rh[t, i, i] = vv[s[t, i]]
            yh[t, i] = ystar2[t, i] - mm[s[t, i]]
    h[:, :] = _ck(rng, yh, Zh, Rh, Wc, h0, PH)
    if ordering != 0:
        _draw_ind(rng, ystar2 - 2.0 * h, q, mm, vv, s)
    # ---- 4. hyperparameters
    d = B[1:] - B[:-1]
    Qc[:, :] = _riw(rng, Qdf + T - 1, Qs + d.T @ d)
    pos = 0
    for i in range(1, K):
        d = np.ascontiguousarray(a[1:, pos : pos + i] - a[:-1, pos : pos + i])
        blk = _riw(
            rng,
            Sdf[i - 1] + T - 1,
            np.ascontiguousarray(Ss[pos : pos + i, pos : pos + i]) + d.T @ d,
        )
        Sc[pos : pos + i, pos : pos + i] = blk
        pos += i
    d = h[1:] - h[:-1]
    Wc[:, :] = _riw(rng, Wdf + T - 1, Ws + d.T @ d)
    return tries


def _run_py(
    rng: Any,
    burnin: int,
    keep: int,
    thin: int,
    Y: np.ndarray,
    Z: np.ndarray,
    B: np.ndarray,
    a: np.ndarray,
    h: np.ndarray,
    s: np.ndarray,
    Qc: np.ndarray,
    Sc: np.ndarray,
    Wc: np.ndarray,
    b0: np.ndarray,
    PB: np.ndarray,
    a0: np.ndarray,
    PA: np.ndarray,
    h0: np.ndarray,
    PH: np.ndarray,
    Qs: np.ndarray,
    Qdf: float,
    Ss: np.ndarray,
    Sdf: np.ndarray,
    Ws: np.ndarray,
    Wdf: float,
    q: np.ndarray,
    mm: np.ndarray,
    vv: np.ndarray,
    offset: float,
    ordering: int,
    stationary: bool,
    lags: int,
    max_tries: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int
]:
    """The chain: ``burnin + keep * thin`` sweeps, every ``thin``-th kept."""
    T, K = Y.shape
    m = B.shape[1]
    na = a.shape[1]
    Bd = np.zeros((keep, T, m), dtype=np.float32)
    ad = np.zeros((keep, T, na), dtype=np.float32)
    hd = np.zeros((keep, T, K), dtype=np.float32)
    Qd = np.zeros((keep, m))
    Sd = np.zeros((keep, na, na))
    Wd = np.zeros((keep, K, K))
    kept = 0
    stuck = 0
    paths = 0
    for it in range(burnin + keep * thin):
        tries = _sweep(
            rng,
            Y,
            Z,
            B,
            a,
            h,
            s,
            Qc,
            Sc,
            Wc,
            b0,
            PB,
            a0,
            PA,
            h0,
            PH,
            Qs,
            Qdf,
            Ss,
            Sdf,
            Ws,
            Wdf,
            q,
            mm,
            vv,
            offset,
            ordering,
            stationary,
            lags,
            max_tries,
        )
        if it >= burnin:
            if tries == 0:
                stuck += 1
                paths += max_tries
            else:
                paths += tries
            if (it - burnin) % thin == 0:
                Bd[kept] = B
                ad[kept] = a
                hd[kept] = h
                for j in range(m):
                    Qd[kept, j] = Qc[j, j]
                Sd[kept] = Sc
                Wd[kept] = Wc
                kept += 1
    return Bd, ad, hd, Qd, Sd, Wd, stuck, paths


def kernels() -> Dict[str, Any]:
    """The compiled kernels: ``chol``, ``ck``, ``riw``, ``draw_ind``,
    ``explosive``, ``sweep``, ``run``.

    numba is imported here and not at module import: ``import statspai``
    must stay light.
    """
    if not _K:
        from numba import njit  # type: ignore[import-untyped]

        g = globals()
        for name in ("chol", "ck", "riw", "draw_ind", "explosive", "sweep"):
            g["_" + name] = njit(cache=True)(g[f"_{name}_py"])
            _K[name] = g["_" + name]
        _K["run"] = njit(cache=True)(_run_py)
    return _K


# --------------------------------------------------------------------------
# Data and priors
# --------------------------------------------------------------------------
def lag_matrix(Y: np.ndarray, lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """``Y[lags:]`` and its regressors: lags first, constant last."""
    n = Y.shape[0]
    parts = [Y[lags - lag : n - lag] for lag in range(1, lags + 1)]
    X = np.column_stack(parts + [np.ones(n - lags)])
    return np.ascontiguousarray(Y[lags:]), np.ascontiguousarray(X)


def training_prior(
    Y: np.ndarray,
    lags: int,
    k_B: float,
    k_A: float,
    k_sig: float,
    k_Q: float,
    k_S: float,
    k_W: float,
) -> Dict[str, Any]:
    """Prior from a constant-coefficient VAR on a training sample.

    OLS on ``Y`` (the training rows) gives ``B_ols`` with covariance
    ``Sigma (x) (X'X)^{-1}``; the residuals are then regressed equation by
    equation on (minus) the residuals of the equations above, which gives
    the free elements of ``A`` with their covariance and the standard
    deviations of the orthogonal shocks. Variances divide by the degrees
    of freedom of each regression.
    """
    y, X = lag_matrix(Y, lags)
    n, K = y.shape
    k = X.shape[1]
    m = K * k
    XtXi = np.linalg.inv(X.T @ X)
    Bols = (XtXi @ X.T @ y).T  # K x k
    U = y - X @ Bols.T
    Sigma = U.T @ U / (n - k)
    VB = np.kron(Sigma, XtXi)
    na = K * (K - 1) // 2
    a0 = np.zeros(na)
    VA = np.zeros((na, na))
    sig = np.zeros(K)
    sig[0] = np.sqrt(U[:, 0] @ U[:, 0] / (n - k))
    pos = 0
    for i in range(1, K):
        Zi = -U[:, :i]
        ZtZi = np.linalg.inv(Zi.T @ Zi)
        ai = ZtZi @ Zi.T @ U[:, i]
        e = U[:, i] - Zi @ ai
        s2 = float(e @ e) / (n - k - i)
        a0[pos : pos + i] = ai
        VA[pos : pos + i, pos : pos + i] = s2 * ZtZi
        sig[i] = np.sqrt(s2)
        pos += i
    tau = Y.shape[0]
    Qdf = float(max(tau, m + 2))
    Ss = np.zeros((na, na))
    Sdf = np.zeros(max(K - 1, 0))
    pos = 0
    for i in range(1, K):
        Sdf[i - 1] = i + 1.0
        Ss[pos : pos + i, pos : pos + i] = (
            k_S**2 * (i + 1.0) * VA[pos : pos + i, pos : pos + i]
        )
        pos += i
    return {
        "b0": Bols.reshape(-1),
        "PB": k_B * VB,
        "a0": a0,
        "PA": k_A * VA,
        "h0": np.log(sig),
        "PH": k_sig * np.eye(K),
        "Qs": k_Q**2 * Qdf * VB,
        "Qdf": Qdf,
        "Ss": Ss,
        "Sdf": Sdf,
        "Ws": k_W**2 * (K + 1.0) * np.eye(K),
        "Wdf": K + 1.0,
    }


def simple_prior(
    Y: np.ndarray, lags: int, k_Q: float, k_S: float, k_W: float
) -> Dict[str, Any]:
    """A prior that uses no training sample.

    Coefficients and covariance states of the first date are centred at
    zero with variance 10; the log standard deviations at the log sample
    standard deviation of each variable's first difference, variance 10.
    The innovation covariances are inverse Wishart with the smallest
    degrees of freedom that give a proper prior with a finite mean and
    identity scale matrices times ``k^2 df``.
    """
    K = Y.shape[1]
    m = K * (K * lags + 1)
    na = K * (K - 1) // 2
    sd = np.diff(Y, axis=0).std(axis=0, ddof=1)
    Ss = np.zeros((na, na))
    Sdf = np.zeros(max(K - 1, 0))
    pos = 0
    for i in range(1, K):
        Sdf[i - 1] = i + 2.0
        Ss[pos : pos + i, pos : pos + i] = k_S**2 * (i + 2.0) * np.eye(i)
        pos += i
    return {
        "b0": np.zeros(m),
        "PB": 10.0 * np.eye(m),
        "a0": np.zeros(na),
        "PA": 10.0 * np.eye(na),
        "h0": np.log(sd),
        "PH": 10.0 * np.eye(K),
        "Qs": k_Q**2 * (m + 2.0) * np.eye(m),
        "Qdf": m + 2.0,
        "Ss": Ss,
        "Sdf": Sdf,
        "Ws": k_W**2 * (K + 2.0) * np.eye(K),
        "Wdf": K + 2.0,
    }
