"""Random-effects spatial panel by maximum likelihood (Stata ``spxtregress, re``).

For a balanced panel with entities ``i`` and periods ``t``::

    y_t = lambda W y_t + X_t b + W X_t g + c + e_t,   e_t = rho W e_t + v_t

with ``c ~ N(0, sigma_u^2 I_N)`` constant over time and ``v_t ~
N(0, sigma_e^2 I_N)``. ``lambda`` is present for ``model='sar' | 'sdm' |
'sac'``, ``rho`` for ``'sem' | 'sac'``, the ``W X`` terms for ``'sdm'``.
With ``A = I - lambda W`` and ``B = I - rho W``, ``r_t = A y_t - X_t b``
has covariance ``J_T (x) sigma_u^2 I + I_T (x) sigma_e^2 (B'B)^-1``.
Splitting it into the between part ``M_1 = T sigma_u^2 I + sigma_e^2
(B'B)^-1`` and the within part ``M_2 = sigma_e^2 (B'B)^-1``::

    l = -NT/2 log 2pi - 1/2 (log|M_1| + (T-1) log|M_2|) + T log|A|
        - 1/2 (T rbar' M_1^-1 rbar + sum_t (r_t - rbar)' M_2^-1 (r_t - rbar))

This is the model of Stata ``spxtregress, re`` (without ``sarpanel``);
estimates, the two standard deviations and OIM standard errors equal
Stata 18 to 1e-6 (``tests/reference_parity/test_spatial_panel_re_Stata_parity.py``).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy.optimize import minimize

from ...exceptions import ConvergenceFailure, NumericalInstability
from ..models.durbin_error import _hessian

__all__ = ["fit_re"]


def fit_re(
    Y: np.ndarray,
    X_mats: List[np.ndarray],
    names: List[str],
    Wd: np.ndarray,
    model: str,
) -> Dict[str, Any]:
    """Fit on ``(N, T)`` arrays; ``X_mats`` holds one ``(N, T)`` per regressor.

    Returns ``params`` / ``se`` (constant first, then the regressors, their
    spatial lags for SDM, then ``lambda`` / ``rho``), ``sigma_u``,
    ``sigma_e`` with SEs, the log-likelihood and the residuals ``r_t``.
    """
    N, T = Y.shape
    lag = model in ("sar", "sdm", "sac")
    err = model in ("sem", "sac")
    cols = [np.ones((N, T)), *X_mats]
    labels = ["const", *names]
    if model == "sdm":
        cols += [Wd @ x for x in X_mats]
        labels += [f"W_{v}" for v in names]
    Xs = np.stack(cols, axis=2)  # (N, T, k)
    k = Xs.shape[2]
    omega = np.linalg.eigvals(Wd)
    real = np.abs(omega.imag) < 1e-10
    lo = 1.0 / float(np.min(omega.real[real])) + 1e-9
    hi = 1.0 / float(np.max(omega.real[real])) - 1e-9
    eye = np.eye(N)

    def unpack(p: np.ndarray):
        b = p[:k]
        j = k
        lam = p[j] if lag else 0.0
        j += int(lag)
        rho = p[j] if err else 0.0
        j += int(err)
        return b, lam, rho, p[j], p[j + 1]

    def loglik(p: np.ndarray) -> float:
        b, lam, rho, su, se = unpack(p)
        if not (lo < lam < hi and lo < rho < hi) or su <= 0 or se <= 0:
            return -1e20
        A = eye - lam * Wd
        B = eye - rho * Wd
        R = A @ Y - Xs @ b  # (N, T)
        BtB_inv = np.linalg.inv(B.T @ B)
        M2 = se**2 * BtB_inv
        M1 = T * su**2 * eye + M2
        rbar = R.mean(axis=1)
        D = R - rbar[:, None]
        _, ld1 = np.linalg.slogdet(M1)
        _, ld2 = np.linalg.slogdet(M2)
        quad = T * rbar @ np.linalg.solve(M1, rbar) + float(
            np.sum(D * np.linalg.solve(M2, D))
        )
        log_a = float(np.sum(np.log(1.0 - lam * omega)).real)
        return float(
            -N * T / 2.0 * np.log(2.0 * np.pi)
            - 0.5 * (ld1 + (T - 1) * ld2)
            + T * log_a
            - 0.5 * quad
        )

    # Start: pooled OLS of y on X, residual SDs split evenly.
    Xf = Xs.reshape(N * T, k)
    b0, *_ = np.linalg.lstsq(Xf, Y.reshape(N * T), rcond=None)
    s0 = float(np.std(Y.reshape(N * T) - Xf @ b0)) / np.sqrt(2.0)
    p0 = np.concatenate([b0, [0.0] * (int(lag) + int(err)), [s0, s0]])
    neg = lambda p: -loglik(p)  # noqa: E731
    res = minimize(neg, p0, method="Nelder-Mead", options={"maxiter": 40000})
    for _ in range(3):
        res = minimize(
            neg,
            res.x,
            method="Nelder-Mead",
            options={"xatol": 1e-11, "fatol": 1e-13, "maxiter": 60000},
        )
    if not np.isfinite(res.fun) or res.fun >= 1e19:
        raise ConvergenceFailure(
            "spatial_panel(effects='re'): the likelihood has no interior maximum.",
            recovery_hint="Check W and that the panel varies over time.",
        )
    p = res.x
    H = _hessian(loglik, p)
    try:
        V = np.linalg.inv(-H)
    except np.linalg.LinAlgError as exc:
        raise NumericalInstability(
            "spatial_panel(effects='re'): singular observed information.",
            recovery_hint="Drop collinear regressors.",
        ) from exc
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    b, lam, rho, su, sv = unpack(p)
    # package convention (sp.sac): rho on Wy, lambda on the error
    spatial = (["rho"] if lag else []) + (["lambda"] if err else [])
    m = k + len(spatial)
    A = eye - lam * Wd
    return {
        "names": labels + spatial,
        "params": p[:m],
        "se": se[:m],
        "sigma_u": float(su),
        "sigma_e": float(sv),
        "se_sigma_u": float(se[m]),
        "se_sigma_e": float(se[m + 1]),
        "loglik": float(loglik(p)),
        # package naming: rho on Wy, lambda on the error
        "rho": float(lam),
        "lambda": float(rho),
        "residuals": A @ Y - Xs @ b,
        "vcov": V[:m, :m],
    }
