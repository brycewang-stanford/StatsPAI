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
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..regression._optim_helpers import hessian_cov


@dataclass
class GARCHResult:
    """Fitted GARCH(p,q) model returned by :func:`garch`.

    Holds the conditional-variance parameters, the volatility path, and
    standardised residuals, plus :meth:`forecast` for multi-step variance.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> T = 400
    >>> eps = np.zeros(T)
    >>> s2 = np.ones(T)
    >>> omega, a1, b1 = 0.05, 0.1, 0.85
    >>> for t in range(1, T):
    ...     s2[t] = omega + a1 * eps[t - 1] ** 2 + b1 * s2[t - 1]
    ...     eps[t] = np.sqrt(s2[t]) * rng.standard_normal()
    >>> res = sp.garch(eps, p=1, q=1)
    >>> bool(res.persistence < 1.0)   # alpha + beta < 1 => stationary
    True
    >>> res.forecast(horizon=3).shape
    (3,)
    """

    omega: float
    alpha: np.ndarray  # (q,)
    beta: np.ndarray  # (p,)
    mu: float
    sigma2: np.ndarray  # conditional variance path (T,)
    residuals: np.ndarray  # ε_t = r_t - μ
    std_residuals: np.ndarray  # z_t = ε_t / σ_t
    log_likelihood: float
    aic: float
    bic: float
    n: int
    p: int
    q: int
    coef: Optional[np.ndarray] = None  # parameter vector (param_names order)
    se_vec: Optional[np.ndarray] = None  # asymptotic SEs (same order)
    param_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Agent-native accessors (params / std_errors / t / p), so GARCH
    # supports inference like every other estimator. Standard errors come
    # from the inverse observed-information (numerical Hessian) at the MLE.
    # ------------------------------------------------------------------
    @property
    def params(self) -> pd.Series:
        if self.coef is None or self.param_names is None:
            return pd.Series(dtype=float)
        return pd.Series(
            np.asarray(self.coef, float),
            index=list(self.param_names),
        )

    @property
    def std_errors(self) -> pd.Series:
        if self.se_vec is None or self.param_names is None:
            return pd.Series(dtype=float)
        return pd.Series(
            np.asarray(self.se_vec, float),
            index=list(self.param_names),
        )

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.std_errors

    @property
    def pvalues(self) -> pd.Series:
        from scipy import stats

        z = (self.params / self.std_errors).to_numpy(float)
        return pd.Series(
            2.0 * (1.0 - stats.norm.cdf(np.abs(z))), index=self.params.index
        )

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
        ]
        if self.param_names is not None and self.se_vec is not None:
            pr, se = self.params, self.std_errors
            tv, pv = self.tvalues, self.pvalues
            lines.append(
                f"  {'':<10s}{'coef':>11s}{'std err':>11s}" f"{'z':>9s}{'P>|z|':>9s}"
            )
            for nm in self.param_names:
                lines.append(
                    f"  {nm:<10s}{pr[nm]:11.6f}{se[nm]:11.6f}"
                    f"{tv[nm]:9.3f}{pv[nm]:9.4f}"
                )
        else:
            lines.append(f"  mu    = {self.mu: .6f}")
            lines.append(f"  omega = {self.omega: .6f}")
            for i, a in enumerate(self.alpha):
                lines.append(f"  alpha[{i + 1}] = {a: .6f}")
            for j, b in enumerate(self.beta):
                lines.append(f"  beta[{j + 1}] = {b: .6f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def garch(
    y: object,
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

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> T = 400
    >>> eps = np.zeros(T)
    >>> s2 = np.ones(T)
    >>> omega, a1, b1 = 0.05, 0.1, 0.85
    >>> for t in range(1, T):
    ...     s2[t] = omega + a1 * eps[t - 1] ** 2 + b1 * s2[t - 1]
    ...     eps[t] = np.sqrt(s2[t]) * rng.standard_normal()
    >>> res = sp.garch(eps, p=1, q=1)
    >>> isinstance(res, sp.GARCHResult)
    True
    >>> bool(res.persistence < 1.0)   # alpha + beta < 1 => stationary
    True
    >>> res.sigma2.shape
    (400,)
    >>> res.forecast(horizon=3).shape
    (3,)
    >>> bool(np.isfinite(res.aic))
    True
    >>> print(res.summary())  # doctest: +SKIP
    """
    y = np.asarray(y, dtype=float).ravel()
    T = len(y)
    if T < max(p, q) + 10:
        raise ValueError("Time series too short for GARCH estimation.")
    y_mean = float(y.mean()) if mean else 0.0

    def neg_ll(theta: np.ndarray) -> float:
        mu = float(theta[0]) if mean else 0.0
        omega = float(theta[int(mean)])
        alpha = theta[int(mean) + 1 : int(mean) + 1 + q]
        beta = theta[int(mean) + 1 + q : int(mean) + 1 + q + p]
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
                eps_lag = eps[t - 1 - i] ** 2 if t - 1 - i >= 0 else s2_init
                s2[t] += alpha[i] * eps_lag
            for j in range(p):
                s2_lag = s2[t - 1 - j] if t - 1 - j >= 0 else s2_init
                s2[t] += beta[j] * s2_lag
            s2[t] = max(s2[t], 1e-12)
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + eps**2 / s2)
        return float(-ll)

    # Initial guesses
    eps0 = y - y_mean
    var0 = float(eps0.var())
    omega0 = var0 * 0.05
    alpha0 = [0.1] * q
    beta0 = [0.85 / max(p, 1)] * p
    x0_parts = ([y_mean] if mean else []) + [omega0] + alpha0 + beta0
    x0 = np.asarray(x0_parts, dtype=float)

    opt = minimize(
        neg_ll,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10},
    )
    theta = np.asarray(opt.x, dtype=float)
    mu = float(theta[0]) if mean else 0.0
    omega = float(theta[int(mean)])
    alpha = theta[int(mean) + 1 : int(mean) + 1 + q]
    beta = theta[int(mean) + 1 + q : int(mean) + 1 + q + p]

    eps = y - mu
    s2 = np.empty(T)
    s2_init = float(eps.var())
    for t in range(T):
        s2[t] = omega
        for i in range(q):
            eps_lag = eps[t - 1 - i] ** 2 if t - 1 - i >= 0 else s2_init
            s2[t] += alpha[i] * eps_lag
        for j in range(p):
            s2_lag = s2[t - 1 - j] if t - 1 - j >= 0 else s2_init
            s2[t] += beta[j] * s2_lag
        s2[t] = max(s2[t], 1e-12)

    param_names = (
        (["mu"] if mean else [])
        + ["omega"]
        + [f"alpha[{i + 1}]" for i in range(q)]
        + [f"beta[{j + 1}]" for j in range(p)]
    )
    try:
        V = hessian_cov(neg_ll, theta)
        se_vec = np.sqrt(np.clip(np.diag(V), 0.0, None))
        if not np.all(np.isfinite(se_vec)):
            se_vec = np.full(len(theta), np.nan)
    except Exception:  # pragma: no cover - singular Hessian at a boundary
        se_vec = np.full(len(theta), np.nan)

    ll = float(-opt.fun)
    k_params = int(mean) + 1 + q + p
    aic = -2 * ll + 2 * k_params
    bic = -2 * ll + k_params * np.log(T)
    std_resid = eps / np.sqrt(s2)

    _result = GARCHResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        mu=mu,
        sigma2=s2,
        residuals=eps,
        std_residuals=std_resid,
        log_likelihood=ll,
        aic=aic,
        bic=bic,
        n=T,
        p=p,
        q=q,
        coef=np.asarray(theta, float),
        se_vec=se_vec,
        param_names=param_names,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.timeseries.garch",
            params={"p": p, "q": q, "mean": mean},
            data=None,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
