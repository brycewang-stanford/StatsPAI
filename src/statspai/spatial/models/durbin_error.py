"""Spatial Durbin error (SDEM) and general nesting (GNS) models by ML.

SDEM:  ``y = X b + W X t + u,          u = lambda W u + e``
GNS:   ``y = rho W y + X b + W X t + u, u = lambda W u + e``

with ``e ~ N(0, sigma^2 I)``. ``(b, t, sigma^2)`` are profiled out of the
log-likelihood

    l = -n/2 log(2 pi sigma^2) + log|I - rho W| + log|I - lambda W|
        - e'e / (2 sigma^2),   e = (I - lambda W)((I - rho W) y - Z g)

and the spatial parameters maximise the concentrated function. Standard
errors invert the observed information of the full log-likelihood in
``(b, t, rho, lambda, sigma^2)`` -- Stata ``spregress, ml`` (its default
``vce(oim)``), which the estimates and standard errors equal to 1e-6 on
the fixture (``tests/reference_parity/test_spatial_durbin_error_Stata_parity.py``).
The Hessian is taken by central differences with Richardson
extrapolation; the log-determinants use the eigenvalues of ``W``, so
``n`` up to a few thousand.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ...core.results import EconometricResults
from ...exceptions import (
    ConvergenceFailure,
    MethodIncompatibility,
    NumericalInstability,
)
from .ml import ArrayOrW, _coerce_W, _make_results, _parse_formula

__all__ = ["sdem", "gns"]


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
    t = best.x
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


def _durbin_model(
    W: ArrayOrW,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool,
    alpha: float,
    lag: bool,
) -> EconometricResults:
    y, X, dep_var, indep = _parse_formula(formula, data)
    n = len(y)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise MethodIncompatibility(
            "spatial ML: missing or infinite values; every unit enters W.",
            recovery_hint="Impute or drop units (and their rows / columns of W).",
        )
    M = _coerce_W(W, n_expected=n, row_normalize=row_normalize)
    Wd = M.toarray()
    Z = np.column_stack([X, Wd @ X[:, 1:]])
    if np.linalg.matrix_rank(Z) < Z.shape[1]:
        raise NumericalInstability(
            "spatial ML: X and WX are collinear.",
            recovery_hint="Drop regressors whose spatial lag is (nearly) constant.",
        )
    fit = _fit(y, Z, Wd, lag)
    names: List[str] = ["const", *indep, *(f"W_{v}" for v in indep)]
    if lag:
        names.append("rho")
    names.append("lambda")
    k = len(names)
    params = np.concatenate([fit["g"], [fit["rho"]] if lag else [], [fit["lam"]]])
    se = np.sqrt(np.clip(np.diag(fit["V"])[:k], 0.0, None))
    res = _make_results(
        alpha=alpha,
        model_type="gns" if lag else "sdem",
        spatial_param_name="rho,lambda" if lag else "lambda",
        spatial_param_value=fit["rho"] if lag else fit["lam"],
        var_names=names,
        params_vec=params,
        se_vec=se,
        sigma2=fit["sigma2"],
        resid=fit["resid_e"],
        fitted=y - fit["resid_u"],
        n=n,
        dep_var=dep_var,
        log_lik=fit["loglik"],
        extra_diag={
            "se_sigma2": float(np.sqrt(fit["V"][-1, -1])),
            "vce": "oim",
        },
        _w_sparse=M,
    )
    res.model_info["model_type"] = (
        "GNS (Spatial Lag + Durbin + Error)" if lag else "SDEM (Spatial Durbin Error)"
    )
    res.model_info["sigma2"] = float(fit["sigma2"])
    res.model_info["se_sigma2"] = float(np.sqrt(fit["V"][-1, -1]))
    res.model_info["loglik"] = float(fit["loglik"])
    res.model_info["vcov"] = pd.DataFrame(fit["V"][:k, :k], index=names, columns=names)
    return res


def sdem(
    W: ArrayOrW,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool = True,
    alpha: float = 0.05,
) -> EconometricResults:
    """Spatial Durbin error model: ``y = Xb + WXt + u, u = lambda W u + e``.

    Parameters
    ----------
    W : ndarray, scipy.sparse matrix, or statspai.spatial.weights.W
        n x n spatial weights, rows in the order of ``data``.
    data : pd.DataFrame
    formula : str
        ``"y ~ x1 + x2"``; a constant is added, and the spatial lag of each
        regressor (not of the constant) enters as ``W_<name>``.
    row_normalize : bool, default True
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        ``params`` ``[const, x..., W_x..., lambda]`` with OIM standard
        errors (Stata ``spregress, ml errorlag(W) ivarlag(W: x...)``);
        ``diagnostics['sigma2']`` and its SE ``se_sigma2``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 40
    >>> W = np.zeros((n, n))
    >>> for i in range(n):
    ...     W[i, (i - 1) % n] = W[i, (i + 1) % n] = 1.0
    >>> x = rng.normal(size=n)
    >>> df = pd.DataFrame({"y": 1 + x + rng.normal(size=n), "x": x})
    >>> list(sp.sdem(W, df, "y ~ x").params.index)
    ['const', 'x', 'W_x', 'lambda']
    """
    return _durbin_model(W, data, formula, row_normalize, alpha, lag=False)


def gns(
    W: ArrayOrW,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool = True,
    alpha: float = 0.05,
) -> EconometricResults:
    """General nesting spatial model (Manski-type): spatial lag of ``y``,
    of the regressors, and of the error.

    ``y = rho W y + Xb + WXt + u, u = lambda W u + e`` by maximum
    likelihood, OIM standard errors as Stata ``spregress, ml dvarlag(W)
    errorlag(W) ivarlag(W: x...)``. The model nests SAR, SEM, SDM, SDEM and
    SAC; it is weakly identified in small samples (``rho`` and ``lambda``
    trade off), so read the standard errors of the two before the point
    estimates.

    Parameters
    ----------
    W : ndarray, scipy.sparse matrix, or statspai.spatial.weights.W
    data : pd.DataFrame
    formula : str
    row_normalize : bool, default True
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        ``params`` ``[const, x..., W_x..., rho, lambda]``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 40
    >>> W = np.zeros((n, n))
    >>> for i in range(n):
    ...     W[i, (i - 1) % n] = W[i, (i + 1) % n] = 1.0
    >>> x = rng.normal(size=n)
    >>> df = pd.DataFrame({"y": 1 + x + rng.normal(size=n), "x": x})
    >>> list(sp.gns(W, df, "y ~ x").params.index)
    ['const', 'x', 'W_x', 'rho', 'lambda']
    """
    return _durbin_model(W, data, formula, row_normalize, alpha, lag=True)
