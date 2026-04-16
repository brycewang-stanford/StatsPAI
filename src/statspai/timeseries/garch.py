"""GARCH(p,q) volatility models (Bollerslev, 1986).

Maximum-likelihood estimation of conditional variance models:

    r_t = μ + ε_t,   ε_t = σ_t z_t,   z_t ~ N(0,1)
    σ²_t = ω + Σ α_i ε²_{t-i} + Σ β_j σ²_{t-j}

The most common specification is GARCH(1,1) where
    σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}

and α + β < 1 for stationarity.

This module provides:
- :func:`garch` — fit GARCH(p,q) by MLE (conditional Gaussian)
- Result with volatility path, standardised residuals, forecast
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class GARCHResult:
    omega: float
    alpha: np.ndarray        # (q,)
    beta: np.ndarray         # (p,)
    mu: float
    sigma2: np.ndarray       # conditional variance path (T,)
    residuals: np.ndarray    # ε_t = r_t - μ
    std_residuals: np.ndarray  # z_t = ε_t / σ_t
    log_likelihood: float
    aic: float
    bic: float
    n: int
    p: int
    q: int

    @property
    def persistence(self) -> float:
        return float(self.alpha.sum() + self.beta.sum())

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Multi-step ahead variance forecast (analytic recursion)."""
        s2 = np.empty(horizon)
        # last-period values
        eps2_last = float(self.residuals[-1] ** 2)
        s2_last = float(self.sigma2[-1])
        for h in range(horizon):
            s2_h = self.omega
            if self.q >= 1:
                s2_h += self.alpha[0] * (eps2_last if h == 0 else s2[h - 1])
            if self.p >= 1:
                s2_h += self.beta[0] * (s2_last if h == 0 else s2[h - 1])
            s2[h] = s2_h
        return s2

    def summary(self) -> str:
        k = 1 + len(self.alpha) + len(self.beta) + 1
        lines = [
            f"GARCH({self.p},{self.q})",
            "-" * 40,
            f"n              : {self.n}",
            f"Log-Lik        : {self.log_likelihood:.4f}",
            f"AIC            : {self.aic:.4f}",
            f"BIC            : {self.bic:.4f}",
            f"Persistence    : {self.persistence:.4f}",
            "",
            "Parameters:",
            f"  mu    = {self.mu: .6f}",
            f"  omega = {self.omega: .6f}",
        ]
        for i, a in enumerate(self.alpha):
            lines.append(f"  alpha[{i + 1}] = {a: .6f}")
        for j, b in enumerate(self.beta):
            lines.append(f"  beta[{j + 1}] = {b: .6f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def garch(
    y,
    p: int = 1,
    q: int = 1,
    mean: bool = True,
) -> GARCHResult:
    """Fit GARCH(p,q) by conditional Gaussian MLE.

    Parameters
    ----------
    y : array-like
        Return series (or log-return, etc.).
    p : int, default 1
        Number of GARCH (lagged σ²) terms.
    q : int, default 1
        Number of ARCH (lagged ε²) terms.
    mean : bool, default True
        Estimate a constant mean μ; if False, μ = 0.
    """
    y = np.asarray(y, dtype=float).ravel()
    T = len(y)
    if T < max(p, q) + 10:
        raise ValueError("Time series too short for GARCH estimation.")
    y_mean = float(y.mean()) if mean else 0.0

    def neg_ll(theta: np.ndarray) -> float:
        mu = float(theta[0]) if mean else 0.0
        omega = float(theta[int(mean)])
        alpha = theta[int(mean) + 1: int(mean) + 1 + q]
        beta = theta[int(mean) + 1 + q: int(mean) + 1 + q + p]
        if omega <= 0 or np.any(alpha < 0) or np.any(beta < 0):
            return 1e15
        if alpha.sum() + beta.sum() >= 1.0:
            return 1e15
        eps = y - mu
        s2 = np.empty(T)
        s2_init = float(eps.var())
        for t in range(T):
            s2[t] = omega
            for i in range(q):
                s2[t] += alpha[i] * (eps[t - 1 - i] ** 2 if t - 1 - i >= 0 else s2_init)
            for j in range(p):
                s2[t] += beta[j] * (s2[t - 1 - j] if t - 1 - j >= 0 else s2_init)
            s2[t] = max(s2[t], 1e-12)
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + eps ** 2 / s2)
        return -ll

    # Initial guesses
    eps0 = y - y_mean
    var0 = float(eps0.var())
    omega0 = var0 * 0.05
    alpha0 = [0.1] * q
    beta0 = [0.85 / max(p, 1)] * p
    x0 = ([y_mean] if mean else []) + [omega0] + alpha0 + beta0
    x0 = np.array(x0)

    opt = minimize(neg_ll, x0, method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10})
    theta = opt.x
    mu = float(theta[0]) if mean else 0.0
    omega = float(theta[int(mean)])
    alpha = theta[int(mean) + 1: int(mean) + 1 + q]
    beta = theta[int(mean) + 1 + q: int(mean) + 1 + q + p]

    eps = y - mu
    s2 = np.empty(T)
    s2_init = float(eps.var())
    for t in range(T):
        s2[t] = omega
        for i in range(q):
            s2[t] += alpha[i] * (eps[t - 1 - i] ** 2 if t - 1 - i >= 0 else s2_init)
        for j in range(p):
            s2[t] += beta[j] * (s2[t - 1 - j] if t - 1 - j >= 0 else s2_init)
        s2[t] = max(s2[t], 1e-12)

    ll = float(-opt.fun)
    k_params = int(mean) + 1 + q + p
    aic = -2 * ll + 2 * k_params
    bic = -2 * ll + k_params * np.log(T)
    std_resid = eps / np.sqrt(s2)

    return GARCHResult(
        omega=omega, alpha=alpha, beta=beta, mu=mu,
        sigma2=s2, residuals=eps, std_residuals=std_resid,
        log_likelihood=ll, aic=aic, bic=bic, n=T, p=p, q=q,
    )
