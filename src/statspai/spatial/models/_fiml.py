"""Full-information maximum likelihood shared by the spatial ML fits.

``y = rho W y + Z g + u, u = lambda W u + e`` (``rho`` fixed at 0 when
``lag=False``). ``(g, sigma^2)`` are profiled out; the spatial parameters
maximise the concentrated log-likelihood; standard errors invert the
observed information of the full log-likelihood in
``(g, [rho], lambda, sigma^2)``, which is Stata ``spregress, ml`` with its
default ``vce(oim)``. Used by ``sp.sdem`` / ``sp.gns`` (``Z = [X, WX]``) and
``sp.sac`` (``Z = X``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np
from scipy.optimize import minimize

from ...exceptions import ConvergenceFailure, NumericalInstability

__all__ = ["_hessian", "_fit"]


def _hessian(f: Callable[[np.ndarray], float], p: np.ndarray) -> np.ndarray:
    """Central-difference Hessian, Richardson-extrapolated (error O(h^4))."""

    def at_step(h: float) -> np.ndarray:
        m = len(p)
        H = np.zeros((m, m))
        steps = h * np.maximum(1.0, np.abs(p))
        for i in range(m):
            for j in range(i, m):

                def val(a: int, c: int) -> float:
                    q = p.copy()
                    q[i] += a * steps[i]
                    q[j] += c * steps[j]
                    return f(q)

                H[i, j] = H[j, i] = (
                    val(1, 1) - val(1, -1) - val(-1, 1) + val(-1, -1)
                ) / (4.0 * steps[i] * steps[j])
        return H

    return (4.0 * at_step(5e-5) - at_step(1e-4)) / 3.0


def _fit(y: np.ndarray, Z: np.ndarray, Wd: np.ndarray, lag: bool) -> Dict[str, Any]:
    n = len(y)
    omega = np.linalg.eigvals(Wd)
    real = np.abs(omega.imag) < 1e-10 * np.max(np.abs(omega))
    w_min = float(np.min(omega.real[real])) if real.any() else -1.0
    w_max = float(np.max(omega.real[real])) if real.any() else 1.0
    lo, hi = 1.0 / w_min + 1e-6, 1.0 / w_max - 1e-6
    Wy = Wd @ y
    eye = np.eye(n)

    def logdet(r: float) -> float:
        return float(np.sum(np.log(1.0 - r * omega)).real)

    def unpack(t: np.ndarray) -> Tuple[float, float]:
        return (float(t[0]), float(t[1])) if lag else (0.0, float(t[0]))

    def conc(t: np.ndarray) -> Tuple[float, np.ndarray, float]:
        rho, lam = unpack(t)
        if not (lo < lam < hi and lo < rho < hi):
            return 1e20, np.zeros(Z.shape[1]), np.nan
        B = eye - lam * Wd
        yt = B @ (y - rho * Wy)
        Zt = B @ Z
        g, *_ = np.linalg.lstsq(Zt, yt, rcond=None)
        e = yt - Zt @ g
        s2 = float(e @ e) / n
        val = -n / 2.0 * np.log(s2) + logdet(lam) + (logdet(rho) if lag else 0.0)
        return -val, g, s2

    def grad(t: np.ndarray) -> np.ndarray:
        # analytic gradient of the concentrated log-likelihood (the
        # regression coefficients drop out by the envelope theorem)
        rho, lam = unpack(t)
        B = eye - lam * Wd
        _, g, s2 = conc(t)
        u = y - rho * Wy - Z @ g
        e = B @ u
        d_lam = (e @ (Wd @ u)) / s2 - float(np.sum(omega / (1.0 - lam * omega)).real)
        if not lag:
            return np.array([-d_lam])
        d_rho = (e @ (B @ Wy)) / s2 - float(np.sum(omega / (1.0 - rho * omega)).real)
        return -np.array([d_rho, d_lam])

    def _polish(t: np.ndarray) -> np.ndarray:
        # Nelder-Mead stops where its simplex shrinks below the tolerance;
        # on the flat rho-lambda ridge of a GNS likelihood that point moved
        # with the platform's rounding (1e-6 in a coefficient between two CI
        # runners). Newton steps on the analytic gradient converge to the
        # stationary point itself; a step that worsens the objective, or
        # leaves the parameter space, is not taken.
        for _ in range(20):
            gr = grad(t)
            h = 1e-6
            J = np.empty((t.size, t.size))
            for i in range(t.size):
                d = np.zeros(t.size)
                d[i] = h
                J[:, i] = (grad(t + d) - grad(t - d)) / (2.0 * h)
            try:
                step = np.linalg.solve(J, gr)
            except np.linalg.LinAlgError:
                break
            cand = t - step
            if not np.all(np.isfinite(cand)) or conc(cand)[0] > conc(t)[
                0
            ] + 1e-12 * abs(conc(t)[0]):
                break
            t = cand
            if np.max(np.abs(step)) < 1e-14:
                break
        return t

    k = 2 if lag else 1
    best = None
    for start in ([0.0] * k, [0.5] * k, [-0.5] * k):
        res = minimize(
            lambda t: conc(t)[0],
            np.asarray(start, dtype=float),
            method="Nelder-Mead",
            options={"xatol": 1e-11, "fatol": 1e-13, "maxiter": 20000},
        )
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not np.isfinite(best.fun) or best.fun >= 1e19:
        raise ConvergenceFailure(
            "spatial ML: the concentrated log-likelihood has no interior maximum.",
            recovery_hint="Check W (row-normalised, no isolated units).",
        )
    t = _polish(np.asarray(best.x, dtype=float))
    _, g, s2 = conc(t)
    rho, lam = unpack(t)

    def full(p: np.ndarray) -> float:
        gg = p[: Z.shape[1]]
        j = Z.shape[1]
        r = p[j] if lag else 0.0
        j += int(lag)
        lm, s = p[j], p[j + 1]
        e = (eye - lm * Wd) @ (y - r * Wy - Z @ gg)
        return float(
            -n / 2.0 * np.log(2.0 * np.pi * s)
            + (logdet(r) if lag else 0.0)
            + logdet(lm)
            - (e @ e) / (2.0 * s)
        )

    p = np.concatenate([g, [rho] if lag else [], [lam, s2]])
    H = _hessian(full, p)
    try:
        V = np.linalg.inv(-H)
    except np.linalg.LinAlgError as exc:
        raise NumericalInstability(
            "spatial ML: the observed information is singular.",
            recovery_hint="Drop collinear regressors (X and WX).",
        ) from exc
    resid = (eye - lam * Wd) @ (y - rho * Wy - Z @ g)
    return {
        "g": g,
        "rho": rho,
        "lam": lam,
        "sigma2": s2,
        "V": V,
        "loglik": full(p),
        "resid_u": y - rho * Wy - Z @ g,
        "resid_e": resid,
    }
