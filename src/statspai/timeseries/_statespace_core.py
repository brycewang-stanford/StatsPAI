"""Kalman filter and fixed-interval smoother for a linear state space model.

Shared by :mod:`statspai.timeseries.statespace`. The model is

    X_t = F_t X_{t-1} + V_t,          V_t ~ (0, Q_t)
    Y_t = A_t + G_t X_t + W_t,        W_t ~ (0, R_t)

with ``X_0 ~ (x0, P0)``. The recursions work on arrays of fixed dimension:
a missing element of ``Y_t`` has its row of ``G_t`` set to zero and its row
and column of ``R_t`` replaced by a unit variance, which leaves the state
moments unchanged and adds nothing to the likelihood once its normal
constant is left out. That keeps the loops free of index selection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize

from ..exceptions import MethodIncompatibility

_LOG_2PI = float(np.log(2.0 * np.pi))


class NotStationary(MethodIncompatibility):
    """``init='stationary'`` was asked for a transition matrix that is not stable."""


FilterOut = Tuple[
    np.ndarray,  # predicted state, (T, m)
    np.ndarray,  # predicted covariance, (T, m, m)
    np.ndarray,  # filtered state
    np.ndarray,  # filtered covariance
    np.ndarray,  # prediction errors, (T, n), 0 where missing
    np.ndarray,  # their covariance, (T, n, n)
    np.ndarray,  # inverse of that covariance
    np.ndarray,  # gain P G' S^{-1}, (T, m, n)
    np.ndarray,  # standardised prediction errors, (T, n)
    np.ndarray,  # log-likelihood contributions, (T,)
]


def filter_py(
    y: np.ndarray,
    mask: np.ndarray,
    A: np.ndarray,
    G: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
) -> FilterOut:
    """Kalman filter. Every system array carries a leading time axis.

    The time axis has length ``T`` or 1 (constant). ``y`` holds 0 where
    ``mask`` is False. Raises ``numpy.linalg.LinAlgError`` when a
    prediction-error covariance is not positive definite.
    """
    T, n = y.shape
    m = x0.shape[0]
    xp = np.zeros((T, m))
    Pp = np.zeros((T, m, m))
    xf = np.zeros((T, m))
    Pf = np.zeros((T, m, m))
    v = np.zeros((T, n))
    S = np.zeros((T, n, n))
    Sinv = np.zeros((T, n, n))
    K = np.zeros((T, m, n))
    e = np.zeros((T, n))
    ll = np.zeros(T)
    eye = np.eye(m)
    x = x0.copy()
    P = P0.copy()
    for t in range(T):
        Ft = F[t if F.shape[0] > 1 else 0]
        Qt = Q[t if Q.shape[0] > 1 else 0]
        x = Ft @ x
        P = Ft @ P @ Ft.T + Qt
        P = 0.5 * (P + P.T)
        xp[t] = x
        Pp[t] = P
        Gt = G[t if G.shape[0] > 1 else 0].copy()
        Rt = R[t if R.shape[0] > 1 else 0].copy()
        At = A[t if A.shape[0] > 1 else 0]
        nobs = 0
        for i in range(n):
            if mask[t, i]:
                nobs += 1
            else:
                Gt[i, :] = 0.0
                Rt[i, :] = 0.0
                Rt[:, i] = 0.0
                Rt[i, i] = 1.0
        if nobs == 0:
            xf[t] = x
            Pf[t] = P
            continue
        vt = y[t] - At - Gt @ x
        for i in range(n):
            if not mask[t, i]:
                vt[i] = 0.0
        GP = Gt @ P
        St = GP @ Gt.T + Rt
        St = 0.5 * (St + St.T)
        L = np.linalg.cholesky(St)
        et = np.linalg.solve(L, vt)
        Kt = np.linalg.solve(St, GP).T
        logdet = 0.0
        quad = 0.0
        for i in range(n):
            logdet += 2.0 * np.log(L[i, i])
            quad += et[i] * et[i]
        ll[t] = -0.5 * (nobs * _LOG_2PI + logdet + quad)
        # Joseph form: stays symmetric and non-negative definite when R or
        # Q is singular and under a near-diffuse P0
        IKG = eye - Kt @ Gt
        x = x + Kt @ vt
        P = IKG @ P @ IKG.T + Kt @ Rt @ Kt.T
        P = 0.5 * (P + P.T)
        xf[t] = x
        Pf[t] = P
        v[t] = vt
        S[t] = St
        Sinv[t] = np.linalg.inv(St)
        K[t] = Kt
        e[t] = et
    return xp, Pp, xf, Pf, v, S, Sinv, K, e, ll


def smooth_py(
    xp: np.ndarray,
    Pp: np.ndarray,
    v: np.ndarray,
    Sinv: np.ndarray,
    K: np.ndarray,
    G: np.ndarray,
    mask: np.ndarray,
    F: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-interval smoother by the backward recursion in ``r`` and ``N``.

    ``r_{t-1} = G' S^{-1} v_t + L_t' r_t`` and
    ``N_{t-1} = G' S^{-1} G + L_t' N_t L_t`` with
    ``L_t = F_{t+1} (I - K_t G_t)``; the smoothed moments are
    ``x_{t|t-1} + P_{t|t-1} r_{t-1}`` and
    ``P_{t|t-1} - P_{t|t-1} N_{t-1} P_{t|t-1}``. No predicted covariance is
    inverted, so a singular ``Q`` needs no special case. Also returns the
    last ``r`` and ``N``, those of the first date.
    """
    T, m = xp.shape
    n = v.shape[1]
    xs = np.zeros((T, m))
    Ps = np.zeros((T, m, m))
    r = np.zeros(m)
    N = np.zeros((m, m))
    eye = np.eye(m)
    for t in range(T - 1, -1, -1):
        Gt = G[t if G.shape[0] > 1 else 0].copy()
        for i in range(n):
            if not mask[t, i]:
                Gt[i, :] = 0.0
        tn = t + 1 if t + 1 < T else t
        Fn = F[tn if F.shape[0] > 1 else 0]
        Lt = Fn @ (eye - K[t] @ Gt)
        GS = Gt.T @ Sinv[t]
        r = GS @ v[t] + Lt.T @ r
        N = GS @ Gt + Lt.T @ N @ Lt
        N = 0.5 * (N + N.T)
        xs[t] = xp[t] + Pp[t] @ r
        V = Pp[t] - Pp[t] @ N @ Pp[t]
        Ps[t] = 0.5 * (V + V.T)
    return xs, Ps, r, N


_KERNELS: Dict[str, Any] = {}


def kernels(compiled: bool) -> Tuple[Any, Any]:
    """The filter and the smoother, compiled by numba when asked for.

    numba is imported here and not at module import, and its absence is not
    an error: the recursions then run as plain NumPy.
    """
    if not compiled:
        return filter_py, smooth_py
    if not _KERNELS:
        try:
            from numba import njit  # type: ignore[import-untyped]
        except ImportError:
            _KERNELS["filter"] = filter_py
            _KERNELS["smooth"] = smooth_py
        else:
            _KERNELS["filter"] = njit(cache=True)(filter_py)
            _KERNELS["smooth"] = njit(cache=True)(smooth_py)
    return _KERNELS["filter"], _KERNELS["smooth"]


def as_stack(
    value: Any, shape: Tuple[int, ...], T: int, name: str, *, symmetric: bool = False
) -> np.ndarray:
    """Turn a constant or time-varying system matrix into ``(T or 1, *shape)``."""
    arr = np.array(value, dtype=float)
    one_row = len(shape) == 2 and shape[0] == 1
    if arr.ndim == 0 and int(np.prod(shape)) == 1:
        arr = arr.reshape((1,) + shape)
    elif arr.shape == shape:
        arr = arr[None]
    elif one_row and arr.shape == (shape[1],):
        arr = arr.reshape((1,) + shape)
    elif one_row and arr.shape == (T, shape[1]):
        # one observable with a loading that changes over time
        arr = arr[:, None, :]
    elif int(np.prod(shape)) == 1 and arr.shape == (T,):
        arr = arr.reshape((T,) + shape)
    if arr.ndim != len(shape) + 1 or arr.shape[1:] != shape:
        raise MethodIncompatibility(
            f"{name} has shape {np.shape(value)}; expected {shape} or "
            f"({T},) + {shape} for a time-varying matrix.",
            recovery_hint=(
                "Time-varying system matrices carry the time axis first; "
                f"entry t of {name} belongs to observation t."
            ),
        )
    if arr.shape[0] not in (1, T):
        raise MethodIncompatibility(
            f"{name} has {arr.shape[0]} time slices for {T} observations."
        )
    if not np.all(np.isfinite(arr)):
        raise MethodIncompatibility(f"{name} contains NaN or infinite entries.")
    if symmetric:
        if not np.allclose(arr, np.swapaxes(arr, 1, 2), rtol=1e-10, atol=1e-12):
            raise MethodIncompatibility(f"{name} must be symmetric.")
        arr = 0.5 * (arr + np.swapaxes(arr, 1, 2))
    return np.ascontiguousarray(arr)


def stationary_cov(F: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Solve ``P = F P F' + Q`` (``vec(P) = (I - F kron F)^{-1} vec(Q)``)."""
    m = F.shape[0]
    lhs = np.eye(m * m) - np.kron(F, F)
    P = np.linalg.solve(lhs, Q.reshape(-1)).reshape(m, m)
    return 0.5 * (P + P.T)


def initial_state(
    F: np.ndarray,
    Q: np.ndarray,
    x0: Optional[Any],
    P0: Optional[Any],
    init: str,
    kappa: float,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Moments of ``X_0`` and the rule that produced the covariance."""
    m = F.shape[1]
    if init not in ("auto", "stationary", "diffuse"):
        raise MethodIncompatibility(
            f"init={init!r} is not one of 'auto', 'stationary', 'diffuse', " "'exact'."
        )
    mean = np.zeros(m) if x0 is None else np.array(x0, dtype=float).reshape(-1)
    if mean.shape != (m,):
        raise MethodIncompatibility(f"x0 must have {m} entries.")
    if P0 is not None:
        cov = np.array(P0, dtype=float)
        if cov.ndim == 0 and m == 1:
            cov = cov.reshape(1, 1)
        if cov.shape != (m, m) or not np.allclose(cov, cov.T, atol=1e-12):
            raise MethodIncompatibility(f"P0 must be a symmetric {m} x {m} matrix.")
        return mean, 0.5 * (cov + cov.T), "user"
    radius = float(np.max(np.abs(np.linalg.eigvals(F[0]))))
    stable = radius < 1.0 - 1e-9
    if init == "stationary" or (init == "auto" and stable):
        if not stable:
            raise NotStationary(
                "init='stationary' needs every eigenvalue of F inside the unit "
                f"circle; the largest modulus is {radius:.6g}.",
                recovery_hint="Use init='diffuse' or pass P0.",
            )
        if F.shape[0] > 1 or Q.shape[0] > 1:
            raise MethodIncompatibility(
                "The stationary initial covariance is defined for constant F " "and Q.",
                recovery_hint="Pass P0 (and x0), or use init='diffuse'.",
            )
        return mean, stationary_cov(F[0], Q[0]), "stationary"
    if not kappa > 0:
        raise MethodIncompatibility("kappa must be positive.")
    return mean, kappa * np.eye(m), "diffuse"


def forecast_moments(
    steps: int,
    x: np.ndarray,
    P: np.ndarray,
    A: np.ndarray,
    G: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """State and observation forecasts with their mean squared errors.

    Each system array has a leading axis of length ``steps`` or 1.
    """
    m = x.shape[0]
    n = A.shape[1]
    xs = np.zeros((steps, m))
    Ps = np.zeros((steps, m, m))
    ys = np.zeros((steps, n))
    Ss = np.zeros((steps, n, n))
    for h in range(steps):
        Fh = F[h if F.shape[0] > 1 else 0]
        Gh = G[h if G.shape[0] > 1 else 0]
        x = Fh @ x
        P = Fh @ P @ Fh.T + Q[h if Q.shape[0] > 1 else 0]
        P = 0.5 * (P + P.T)
        xs[h] = x
        Ps[h] = P
        ys[h] = A[h if A.shape[0] > 1 else 0] + Gh @ x
        Sh = Gh @ P @ Gh.T + R[h if R.shape[0] > 1 else 0]
        Ss[h] = 0.5 * (Sh + Sh.T)
    return xs, Ps, ys, Ss


def num_gradient(f: Any, x: np.ndarray, rel: float = 1e-6) -> np.ndarray:
    """Central-difference gradient with a step scaled to each parameter."""
    g = np.zeros(x.size)
    for i in range(x.size):
        h = rel * max(abs(float(x[i])), 1.0)
        up = x.copy()
        dn = x.copy()
        up[i] += h
        dn[i] -= h
        g[i] = (f(up) - f(dn)) / (2.0 * h)
    return g


def num_jacobian(f: Any, x: np.ndarray, rel: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of a vector function, ``(len(f), len(x))``."""
    cols = []
    for i in range(x.size):
        h = rel * max(abs(float(x[i])), 1.0)
        up = x.copy()
        dn = x.copy()
        up[i] += h
        dn[i] -= h
        cols.append((np.asarray(f(up), float) - np.asarray(f(dn), float)) / (2.0 * h))
    return np.column_stack(cols)


def num_hessian(
    f: Any, x: np.ndarray, rel: float = 1e-4, diagonal: bool = False
) -> np.ndarray:
    """Central-difference Hessian from function values.

    The step for parameter ``i`` is ``rel * max(|x_i|, 1)``; ``rel`` near the
    fourth root of machine precision balances truncation against rounding.
    """
    k = x.size
    h = np.array([rel * max(abs(float(v)), 1.0) for v in x])
    f0 = f(x)
    H = np.zeros((k, k))
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = h[i]
        H[i, i] = (f(x + ei) - 2.0 * f0 + f(x - ei)) / h[i] ** 2
        if diagonal:
            continue
        for j in range(i + 1, k):
            ej = np.zeros(k)
            ej[j] = h[j]
            val = (
                f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)
            ) / (4.0 * h[i] * h[j])
            H[i, j] = val
            H[j, i] = val
    return H


REPORTED = (
    "gradient",
    "scaled_gradient",
    "hessian_pd",
    "hessian_min_eigenvalue",
    "steps",
)


def maximise(
    nll: Any,
    contributions: Any,
    theta0: np.ndarray,
    method: str,
    vce: str,
    maxiter: int,
    tol: float,
) -> Dict[str, Any]:
    """Minimise ``nll`` from ``theta0``, check the optimum, and price it.

    ``contributions(theta)`` returns the per-date log-likelihood terms and
    is used for the outer product of scores. The search runs in parameters
    rescaled by the curvature of ``nll`` along each axis at ``theta0``.
    """
    k = theta0.size

    def grad(theta: np.ndarray) -> np.ndarray:
        return num_gradient(nll, theta)

    ref = np.maximum(np.abs(theta0), 1.0)
    curv = np.diag(num_hessian(nll, theta0, rel=1e-3, diagonal=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(curv > 0, 1.0 / np.sqrt(np.abs(curv)), ref)
    scale = np.clip(scale, 1e-6 * ref, 1e3 * ref)

    def nll_z(z: np.ndarray) -> float:
        return float(nll(z * scale))

    def grad_z(z: np.ndarray) -> np.ndarray:
        return num_gradient(nll_z, z)

    def scaled(theta: np.ndarray, fval: float) -> float:
        g = np.abs(grad(theta)) * np.maximum(np.abs(theta), 1.0)
        return float(np.max(g) / max(abs(fval), 1.0))

    steps: List[Dict[str, Any]] = []

    def run(name: str, x: np.ndarray) -> Tuple[np.ndarray, float]:
        if name == "Nelder-Mead":
            options: Dict[str, Any] = {"maxiter": maxiter * k, "maxfev": maxiter * k}
            options.update(xatol=1e-8, fatol=1e-10, adaptive=True)
            opt = optimize.minimize(nll_z, x / scale, method=name, options=options)
        else:
            options = {"maxiter": maxiter, "gtol": 1e-7}
            opt = optimize.minimize(
                nll_z, x / scale, jac=grad_z, method=name, options=options
            )
        xn = np.asarray(opt.x, dtype=float) * scale
        step: Dict[str, Any] = {"optimizer": name, "loglik": -float(opt.fun)}
        step.update(success=bool(opt.success), message=str(opt.message))
        step.update(iterations=int(getattr(opt, "nit", 0)))
        step.update(scaled_gradient=scaled(xn, float(opt.fun)))
        steps.append(step)
        return xn, float(opt.fun)

    best, fbest = run(method, theta0)
    if method == "Nelder-Mead":
        cand, fc = run("BFGS", best)
        if fc <= fbest:
            best, fbest = cand, fc
    if scaled(best, fbest) > tol:
        cand, fc = run("Nelder-Mead", best)
        cand, fc = run("BFGS", cand)
        if fc <= fbest:
            best, fbest = cand, fc
    sg = scaled(best, fbest)

    H = num_hessian(nll, best)
    eig = np.linalg.eigvalsh(0.5 * (H + H.T))
    hessian_pd = bool(np.all(eig > 0))
    cov: np.ndarray = np.full((k, k), np.nan)
    opg: np.ndarray = np.zeros((k, k))
    if vce in ("opg", "robust"):
        scores = num_jacobian(contributions, best)
        opg = scores.T @ scores
    if vce == "opg":
        cov = np.linalg.pinv(opg)
    elif hessian_pd:
        Hinv = np.linalg.inv(H)
        cov = Hinv if vce == "hessian" else Hinv @ opg @ Hinv
    notes: List[str] = []
    if not hessian_pd:
        notes.append(
            "The Hessian of the negative log-likelihood is not positive "
            f"definite (smallest eigenvalue {eig.min():.3g}): the point is not "
            "a regular interior maximum."
        )
    if sg > tol:
        notes.append(f"The scaled gradient {sg:.3g} exceeds tol = {tol:g}.")
    return {
        "theta": best,
        "fun": fbest,
        "cov": cov,
        "notes": notes,
        "converged": bool(sg <= tol and hessian_pd),
        "gradient": [float(-g) for g in grad(best)],
        "scaled_gradient": sg,
        "hessian_pd": hessian_pd,
        "hessian_min_eigenvalue": float(eig.min()),
        "steps": steps,
    }
