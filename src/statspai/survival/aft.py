"""Accelerated Failure Time (AFT) parametric survival models.

AFT models the **log survival time** directly as a linear function:

    log T_i = X_i β + σ ε_i

where the distribution of ε determines the family:

- **exponential**: ε ~ standard extreme-value (Gumbel)
- **weibull**: ε ~ standard extreme-value (Gumbel)
- **lognormal**: ε ~ N(0,1)
- **loglogistic**: ε ~ standard logistic

Estimation is by maximum likelihood with right-censoring.

References
----------
Kalbfleisch, J.D. & Prentice, R.L. (2002). *The Statistical Analysis
  of Failure Time Data*, 2nd ed. Wiley.
Klein, J.P. & Moeschberger, M.L. (2003). *Survival Analysis:
  Techniques for Censored and Truncated Data*, 2nd ed. Springer. [@kalbfleisch2002statistical]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import minimize

from .models import _parse_formula


AFTFamily = Literal["exponential", "weibull", "lognormal", "loglogistic"]


def _aft_log_likelihood(
    beta: np.ndarray, sigma: float,
    X: np.ndarray, logT: np.ndarray, E: np.ndarray,
    family: AFTFamily,
) -> float:
    """AFT log-likelihood with right-censoring."""
    eta = X @ beta
    z = (logT - eta) / sigma
    if family in ("exponential", "weibull"):
        # Gumbel (minimum): f(z) = exp(z - exp(z)), S(z) = exp(-exp(z))
        log_f = z - np.exp(z) - np.log(sigma)
        log_S = -np.exp(z)
    elif family == "lognormal":
        log_f = sp_stats.norm.logpdf(z) - np.log(sigma)
        log_S = sp_stats.norm.logsf(z)
    elif family == "loglogistic":
        log_f = sp_stats.logistic.logpdf(z) - np.log(sigma)
        log_S = sp_stats.logistic.logsf(z)
    else:
        raise ValueError(f"unknown family {family!r}")
    ll = float(np.sum(E * log_f + (1 - E) * log_S))
    return ll


@dataclass
class AFTResult:
    beta: np.ndarray
    se: np.ndarray
    sigma: float
    var_names: List[str]
    family: AFTFamily
    log_likelihood: float
    aic: float
    bic: float
    n: int
    n_events: int

    def summary(self) -> str:
        lines = [
            f"AFT Model ({self.family})",
            "-" * 45,
            f"n = {self.n}, events = {self.n_events}",
            f"log σ = {np.log(self.sigma):.4f}  (σ = {self.sigma:.4f})",
            f"Log-Lik = {self.log_likelihood:.4f}",
            f"AIC = {self.aic:.2f}, BIC = {self.bic:.2f}",
            "",
            "Coefficients (on log-time scale):",
        ]
        for nm, b, s in zip(self.var_names, self.beta, self.se):
            t = b / s if s > 0 else np.nan
            p = 2 * (1 - sp_stats.norm.cdf(abs(t)))
            lines.append(
                f"  {nm:<15s}  {b: .4f}  (SE {s: .4f}, z {t: .3f}, p {p: .4f})"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def aft(
    formula: str,
    data: pd.DataFrame,
    family: AFTFamily = "weibull",
) -> AFTResult:
    """Fit an Accelerated Failure Time model by MLE.

    Parameters
    ----------
    formula : str
        ``"duration + event ~ x1 + x2"``.
    family : {"exponential", "weibull", "lognormal", "loglogistic"}
    """
    lhs_str, covariates = _parse_formula(formula)
    lhs_parts = [s.strip() for s in lhs_str.split("+")]
    if len(lhs_parts) != 2:
        raise ValueError("formula LHS must be 'duration + event'")
    dur_col, event_col = lhs_parts

    df = data[[dur_col, event_col] + covariates].dropna()
    T_raw = df[dur_col].to_numpy(float)
    E = df[event_col].to_numpy(float).astype(int)
    X = np.column_stack([np.ones(len(df))] +
                        [df[c].to_numpy(float) for c in covariates])
    n, k = X.shape
    logT = np.log(np.maximum(T_raw, 1e-12))
    n_events = int(E.sum())

    fix_sigma = family == "exponential"

    def neg_ll(theta):
        beta = theta[:k]
        sigma = 1.0 if fix_sigma else np.exp(theta[k])
        return -_aft_log_likelihood(beta, sigma, X, logT, E, family)

    beta0 = np.linalg.lstsq(X, logT, rcond=None)[0]
    sigma0 = float(np.std(logT - X @ beta0))
    x0 = np.concatenate([beta0, [] if fix_sigma else [np.log(max(sigma0, 0.1))]])
    opt = minimize(neg_ll, x0, method="L-BFGS-B", options={"maxiter": 500})
    beta = opt.x[:k]
    sigma = 1.0 if fix_sigma else float(np.exp(opt.x[k]))

    # SE from Hessian
    from scipy.optimize import approx_fprime
    n_params = len(opt.x)
    H = np.zeros((n_params, n_params))
    h = 1e-5
    for i in range(n_params):
        e = np.zeros(n_params); e[i] = h
        H[i] = (approx_fprime(opt.x + e, neg_ll, h) -
                 approx_fprime(opt.x - e, neg_ll, h)) / (2 * h)
    try:
        V = np.linalg.inv(H)
        se = np.sqrt(np.maximum(np.diag(V), 0))[:k]
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)

    ll = float(-opt.fun)
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + n_params * np.log(n)

    _result = AFTResult(
        beta=beta, se=se, sigma=sigma,
        var_names=["Intercept"] + covariates,
        family=family, log_likelihood=ll,
        aic=aic, bic=bic, n=n, n_events=n_events,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.survival.aft",
            params={"formula": formula, "family": family},
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
