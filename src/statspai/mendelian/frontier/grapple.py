"""GRAPPLE: profile-likelihood MR with weak-instrument + pleiotropy robustness.

Reference
---------
Wang, J., Zhao, Q., Bowden, J., Hemani, G., Smith, G.D., Small, D.S.
& Dickhaus, T. (2021).
"Causal inference for heritable phenotypic risk factors using
heterogeneous genetic instruments." *PLoS Genetics*, 17(6), e1009575.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from ._common import as_float_arrays, harmonize_signs


__all__ = ["GrappleResult", "grapple"]


@dataclass
class GrappleResult:
    """Output of :func:`grapple`.

    Attributes
    ----------
    estimate : float
        Profile-likelihood MLE of the causal effect β.
    se : float
        SE from observed Fisher information.
    ci_lower, ci_upper : float
    p_value : float
    tau2 : float
        Estimated pleiotropy variance (balanced pleiotropy SD² = τ²).
    loglik : float
        Profile log-likelihood at the MLE.
    converged : bool
    n_snps : int
    """
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    tau2: float
    loglik: float
    converged: bool
    n_snps: int

    def summary(self) -> str:
        ci = f"[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]"
        conv = "converged" if self.converged else "DID NOT CONVERGE"
        return (
            "GRAPPLE (profile-likelihood MR)\n"
            + "=" * 62 + "\n"
            f"  n SNPs        : {self.n_snps}\n"
            f"  causal β      : {self.estimate:+.4f}   SE = {self.se:.4f}\n"
            f"  95% CI        : {ci}\n"
            f"  p-value       : {self.p_value:.4g}\n"
            f"  pleiotropy τ² : {self.tau2:.4g}\n"
            f"  log-lik       : {self.loglik:.3f}  ({conv})"
        )


def _grapple_nll(
    params: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
) -> float:
    """Negative log-likelihood for GRAPPLE (single-exposure).

    Model:  β_y = β * β_x + u,  u ~ N(0, vy + β² vx + τ²)
    """
    beta, log_tau2 = params
    tau2 = float(np.exp(log_tau2))
    sigma2 = vy + beta ** 2 * vx + tau2
    if np.any(sigma2 <= 0):
        return 1e12
    resid = by - beta * bx
    return 0.5 * float(
        np.sum(resid ** 2 / sigma2 + np.log(2 * np.pi * sigma2))
    )


def grapple(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_exposure: np.ndarray,
    se_outcome: np.ndarray,
    *,
    alpha: float = 0.05,
    beta_init: Optional[float] = None,
    tau2_init: float = 1e-4,
) -> GrappleResult:
    """Profile-likelihood MR with weak-instrument + pleiotropy robustness.

    Implements the single-exposure GRAPPLE estimator of Wang, Zhao,
    Bowden et al. (2021).  Model:

    .. math::

       \\beta_{y,i} = \\beta \\, \\beta_{x,i} + u_i,
       \\qquad u_i \\sim \\mathcal{N}(0,\\ s_{y,i}^2 + \\beta^2 s_{x,i}^2 + \\tau^2)

    The variance term :math:`\\beta^2 s_{x,i}^2` is the weak-instrument
    measurement-error correction (cf. Bowden 2019 "MR with measurement
    error"); :math:`\\tau^2` absorbs balanced directional pleiotropy.
    Both β and τ² are profiled jointly via quasi-Newton (L-BFGS-B on the
    negative log-likelihood with ``log τ²`` as a free parameter).

    SE is obtained from the observed Fisher information at the MLE
    (diagonal block corresponding to β, marginal over τ²).

    Parameters
    ----------
    beta_exposure, beta_outcome : ndarray
    se_exposure, se_outcome : ndarray
    alpha : float, default 0.05
    beta_init : float, optional
        Initial β.  Defaults to the IVW estimate.
    tau2_init : float, default 1e-4
        Initial pleiotropy variance.

    Returns
    -------
    :class:`GrappleResult`

    Notes
    -----
    This implements the single-exposure GRAPPLE; the multi-exposure
    variant (joint causal effects of K correlated exposures) is a
    natural extension and will land in ``sp.mr_grapple_mv`` later.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.grapple(bx, by, sx, sy)
    >>> print(res.summary())
    """
    bx, by, sx, sy = as_float_arrays(
        beta_exposure, beta_outcome, se_exposure, se_outcome
    )
    bx, by = harmonize_signs(bx, by)
    vx = sx ** 2
    vy = sy ** 2

    # IVW warm start
    if beta_init is None:
        w = 1.0 / vy
        denom = float(np.sum(w * bx ** 2))
        beta_init = float(np.sum(w * bx * by) / denom) if denom > 0 else 0.0

    x0 = np.array([beta_init, np.log(max(tau2_init, 1e-10))])
    res = minimize(
        _grapple_nll, x0, args=(bx, by, vx, vy),
        method="L-BFGS-B",
        options={"maxiter": 500, "gtol": 1e-8},
    )

    beta_hat = float(res.x[0])
    tau2_hat = float(np.exp(res.x[1]))
    loglik = -float(res.fun)

    # Observed Fisher info at MLE (numerical second derivative)
    h = max(abs(beta_hat) * 1e-4, 1e-6)

    def _nll_beta(b):
        return _grapple_nll(np.array([b, np.log(tau2_hat)]), bx, by, vx, vy)

    d2 = (
        _nll_beta(beta_hat + h)
        - 2 * _nll_beta(beta_hat)
        + _nll_beta(beta_hat - h)
    ) / h ** 2
    se_hat = float(np.sqrt(1.0 / d2)) if d2 > 0 else float("nan")

    z_crit = stats.norm.ppf(1 - alpha / 2)
    if np.isfinite(se_hat) and se_hat > 0:
        z = beta_hat / se_hat
        p_value = float(2.0 * stats.norm.sf(abs(z)))
        lo = beta_hat - z_crit * se_hat
        hi = beta_hat + z_crit * se_hat
    else:
        p_value = float("nan")
        lo = hi = float("nan")

    return GrappleResult(
        estimate=beta_hat,
        se=se_hat,
        ci_lower=lo,
        ci_upper=hi,
        p_value=p_value,
        tau2=tau2_hat,
        loglik=loglik,
        converged=bool(res.success),
        n_snps=len(bx),
    )
