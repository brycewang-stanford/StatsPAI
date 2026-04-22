"""MR-RAPS: Robust Adjusted Profile Score for two-sample summary-data MR.

Reference
---------
Zhao, Q., Wang, J., Hemani, G., Bowden, J. & Small, D.S. (2020).
"Statistical inference in two-sample summary-data Mendelian
randomization using robust adjusted profile score."
*The Annals of Statistics*, 48(3), 1742-1769.

Complements :func:`grapple` (same profile-likelihood family) by using
Tukey's biweight robust loss in place of the Gaussian quadratic.  This
makes the estimator resistant to a small fraction of gross pleiotropy
outliers (heavy-tailed departures from balanced pleiotropy) while
retaining the weak-instrument correction :math:`\\beta^2 s_x^2`.

Model
-----
Same structural model as GRAPPLE:

.. math::

   \\beta_{y,i} = \\beta \\beta_{x,i} + u_i,
   \\qquad u_i \\sim \\mathcal{N}(0, s_{y,i}^2 + \\beta^2 s_{x,i}^2 + \\tau^2).

MR-RAPS replaces the sum of log-Gaussian contributions with a sum of
Tukey biweight contributions on the standardised residual:

.. math::

   \\ell_i(\\beta,\\tau^2) = \\rho_c\\!\\Big(
     \\frac{\\beta_{y,i} - \\beta\\beta_{x,i}}{\\sqrt{s_{y,i}^2 + \\beta^2 s_{x,i}^2 + \\tau^2}}
     \\Big)

with :math:`\\rho_c(r)` the Tukey biweight with tuning constant c.
Small-c → more robust (95 %→ c=4.685).  SE from sandwich formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from ._common import as_float_arrays, harmonize_signs


__all__ = ["MRRapsResult", "mr_raps"]


@dataclass
class MRRapsResult:
    """Output of :func:`mr_raps`.

    Attributes
    ----------
    estimate : float
    se : float
        Sandwich SE from the robust M-estimator.
    ci_lower, ci_upper : float
    p_value : float
    tau2 : float
    loglik_robust : float
        Final robust (Tukey) loss value.
    converged : bool
    tuning_c : float
    n_snps : int
    """
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    tau2: float
    loglik_robust: float
    converged: bool
    tuning_c: float
    n_snps: int

    def summary(self) -> str:
        ci = f"[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]"
        conv = "converged" if self.converged else "DID NOT CONVERGE"
        return (
            "MR-RAPS (Robust Adjusted Profile Score)\n"
            + "=" * 62 + "\n"
            f"  n SNPs        : {self.n_snps}\n"
            f"  Tukey c       : {self.tuning_c:.2f}\n"
            f"  causal β      : {self.estimate:+.4f}   SE = {self.se:.4f}\n"
            f"  95% CI        : {ci}\n"
            f"  p-value       : {self.p_value:.4g}\n"
            f"  pleiotropy τ² : {self.tau2:.4g}\n"
            f"  robust loss   : {self.loglik_robust:.3f}  ({conv})"
        )


def _tukey_rho(r: np.ndarray, c: float) -> np.ndarray:
    """Tukey biweight ρ(r) — zero outside |r| > c (outlier rejection)."""
    out = np.zeros_like(r)
    inner = np.abs(r) <= c
    u = r[inner] / c
    out[inner] = (c ** 2 / 6.0) * (1.0 - (1.0 - u ** 2) ** 3)
    out[~inner] = c ** 2 / 6.0
    return out


def _tukey_psi(r: np.ndarray, c: float) -> np.ndarray:
    """Tukey biweight ψ(r) = ρ'(r): zero outside |r|>c."""
    out = np.zeros_like(r)
    inner = np.abs(r) <= c
    u = r[inner] / c
    out[inner] = r[inner] * (1.0 - u ** 2) ** 2
    return out


def _tukey_psi_prime(r: np.ndarray, c: float) -> np.ndarray:
    """ψ'(r) — needed for sandwich J matrix."""
    out = np.zeros_like(r)
    inner = np.abs(r) <= c
    u = r[inner] / c
    out[inner] = (1.0 - u ** 2) ** 2 - 4.0 * (u ** 2) * (1.0 - u ** 2)
    return out


def _raps_loss(
    params: np.ndarray,
    bx: np.ndarray, by: np.ndarray,
    vx: np.ndarray, vy: np.ndarray,
    c: float,
) -> float:
    """Robust adjusted profile score objective.

    L(β, τ²) = Σ [ρ_c(r_i) + 0.5 log σ_i²(β, τ²)]

    The log-variance term is the profile-likelihood adjustment that
    keeps τ² from blowing up to infinity (where ρ_c saturates at
    c²/6 and the robust loss is flat).  Without it the objective has
    a degenerate minimum at τ² → ∞; with it, the penalty on large
    σ² restores identifiability.
    """
    beta, log_tau2 = params
    tau2 = float(np.exp(log_tau2))
    sigma2 = vy + beta ** 2 * vx + tau2
    if np.any(sigma2 <= 0):
        return 1e12
    sigma = np.sqrt(sigma2)
    resid = (by - beta * bx) / sigma
    return float(
        np.sum(_tukey_rho(resid, c)) + 0.5 * np.sum(np.log(sigma2))
    )


def mr_raps(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_exposure: np.ndarray,
    se_outcome: np.ndarray,
    *,
    tuning_c: float = 4.685,
    alpha: float = 0.05,
    beta_init: Optional[float] = None,
    tau2_init: float = 1e-4,
) -> MRRapsResult:
    """Robust Adjusted Profile Score MR (Zhao et al. 2020).

    Profile-likelihood MR with a Tukey biweight loss instead of
    Gaussian, making the estimator resistant to a small fraction of
    gross pleiotropy outliers.  Same weak-instrument correction as
    :func:`grapple` (i.e. includes :math:`\\beta^2 s_x^2`).

    Parameters
    ----------
    beta_exposure, beta_outcome : ndarray
    se_exposure, se_outcome : ndarray
    tuning_c : float, default 4.685
        Tukey tuning constant.  4.685 gives 95% asymptotic efficiency
        under Gaussian errors; smaller values → more robust but less
        efficient.  Zhao et al. 2020 recommend c=4.685 as default.
    alpha : float, default 0.05
    beta_init : float, optional
        Default IVW.
    tau2_init : float, default 1e-4

    Returns
    -------
    :class:`MRRapsResult`

    Notes
    -----
    SE is computed from the sandwich formula for M-estimators:

    .. math::

       \\mathrm{Var}(\\hat\\beta) \\approx J^{-1} V J^{-1},

    where :math:`J = -\\sum \\psi'(r_i) \\partial r_i / \\partial\\beta`
    and :math:`V = \\sum \\psi(r_i)^2 (\\partial r_i/\\partial\\beta)^2`.
    Evaluated numerically at the MLE with :math:`\\tau^2` profiled out.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.mr_raps(bx, by, sx, sy)
    >>> print(res.summary())
    """
    bx, by, sx, sy = as_float_arrays(
        beta_exposure, beta_outcome, se_exposure, se_outcome
    )
    bx, by = harmonize_signs(bx, by)
    vx = sx ** 2
    vy = sy ** 2

    if tuning_c <= 0:
        raise ValueError(f"tuning_c must be > 0; got {tuning_c}")

    # IVW warm start
    if beta_init is None:
        w = 1.0 / vy
        denom = float(np.sum(w * bx ** 2))
        beta_init = float(np.sum(w * bx * by) / denom) if denom > 0 else 0.0

    x0 = np.array([beta_init, np.log(max(tau2_init, 1e-10))])
    res = minimize(
        _raps_loss, x0, args=(bx, by, vx, vy, tuning_c),
        method="L-BFGS-B",
        options={"maxiter": 500, "gtol": 1e-8},
    )

    beta_hat = float(res.x[0])
    tau2_hat = float(np.exp(res.x[1]))

    # ---- Sandwich SE via M-estimator formula --------------------------
    sigma2 = vy + beta_hat ** 2 * vx + tau2_hat
    sigma = np.sqrt(sigma2)
    r = (by - beta_hat * bx) / sigma
    # ∂r/∂β at fixed τ²:
    #   r = (by - β bx) / sqrt(vy + β² vx + τ²)
    #   dr/dβ = [-bx * sigma - (by - β bx) * (β vx / sigma)] / sigma²
    drdb = (-bx * sigma - (by - beta_hat * bx) * (beta_hat * vx / sigma)) / sigma2
    psi = _tukey_psi(r, tuning_c)
    psi_prime = _tukey_psi_prime(r, tuning_c)
    J = float(np.sum(psi_prime * drdb ** 2))
    V = float(np.sum(psi ** 2 * drdb ** 2))
    if abs(J) > 1e-12 and V > 0:
        se_hat = float(np.sqrt(V) / abs(J))
    else:
        se_hat = float("nan")

    z_crit = stats.norm.ppf(1 - alpha / 2)
    if np.isfinite(se_hat) and se_hat > 0:
        z = beta_hat / se_hat
        p_value = float(2.0 * stats.norm.sf(abs(z)))
        lo = beta_hat - z_crit * se_hat
        hi = beta_hat + z_crit * se_hat
    else:
        p_value = float("nan")
        lo = hi = float("nan")

    return MRRapsResult(
        estimate=beta_hat,
        se=se_hat,
        ci_lower=lo,
        ci_upper=hi,
        p_value=p_value,
        tau2=tau2_hat,
        loglik_robust=-float(res.fun),
        converged=bool(res.success),
        tuning_c=float(tuning_c),
        n_snps=len(bx),
    )
