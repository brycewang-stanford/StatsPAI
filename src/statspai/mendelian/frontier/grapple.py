"""GRAPPLE: robust profile-score MR with weak-instrument + pleiotropy robustness.

Reference
---------
Wang, J., Zhao, Q., Bowden, J., Hemani, G., Smith, G.D., Small, D.S.
& Dickhaus, T. (2021).
"Causal inference for heritable phenotypic risk factors using
heterogeneous genetic instruments." *PLoS Genetics*, 17(6), e1009575.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy import stats

from ..._result_serialize import ResultProtocolMixin
from ...exceptions import MethodIncompatibility
from ._common import as_float_arrays

__all__ = ["GrappleResult", "grapple"]


@dataclass
class GrappleResult(ResultProtocolMixin):
    """Output of :func:`grapple`.

    Attributes
    ----------
    estimate : float
        Robust profile-score estimate of the causal effect β.
    se : float
        GRAPPLE's sandwich SE.
    ci_lower, ci_upper : float
    p_value : float
    tau2 : float
        Estimated pleiotropy variance (balanced pleiotropy SD² = τ²).
    loglik : float
        Robust objective ``-sum rho(t)`` at the estimate.
    tau2_se : float
    loss : str
    converged : bool
    n_snps : int

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(8)
    >>> bx = rng.uniform(0.1, 0.5, 25)
    >>> sx = rng.uniform(0.02, 0.05, 25)
    >>> sy = rng.uniform(0.02, 0.05, 25)
    >>> by = 0.3 * bx + rng.normal(0, sy)
    >>> res = sp.grapple(bx, by, sx, sy)
    >>> type(res).__name__
    'GrappleResult'
    >>> res.n_snps
    25
    >>> bool(np.isfinite(res.estimate))
    True
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
    tau2_se: float = float("nan")
    loss: str = "tukey"

    def summary(self) -> str:
        ci = f"[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]"
        conv = "converged" if self.converged else "DID NOT CONVERGE"
        return (
            "GRAPPLE (robust profile-score MR)\n" + "=" * 62 + "\n"
            f"  n SNPs        : {self.n_snps}\n"
            f"  causal β      : {self.estimate:+.4f}   SE = {self.se:.4f}\n"
            f"  95% CI        : {ci}\n"
            f"  p-value       : {self.p_value:.4g}\n"
            f"  pleiotropy τ² : {self.tau2:.4g}\n"
            f"  log-lik       : {self.loglik:.3f}  ({conv})"
        )


def _l2_rho(r: Any, deriv: int = 0) -> Any:
    r = np.asarray(r, dtype=float)
    if deriv == 0:
        return r**2 / 2
    if deriv == 1:
        return r
    return np.ones_like(r)


def _t_values(beta: Any, tau2: Any, bx: Any, by: Any, vx: Any, vy: Any) -> Any:
    return (by - bx * beta) / np.sqrt(vx * beta**2 + vy + tau2)


def grapple(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_exposure: np.ndarray,
    se_outcome: np.ndarray,
    *,
    loss: str = "tukey",
    k: Optional[float] = None,
    alpha: float = 0.05,
    beta_init: Optional[float] = None,
    tau2_init: float = 1e-4,
    tol: float = 1e-13,
    max_iter: int = 200,
) -> GrappleResult:
    r"""GRAPPLE robust profile-score MR (single exposure).

    Solves the estimating equations of R ``GRAPPLE::grappleRobustEst``
    (Wang, Zhao, Bowden et al. 2021) for one exposure and uncorrelated
    exposure / outcome errors (``cor.mat = NULL``). With the standardised
    residual

    .. math::

       t_i(\beta, \tau^2) = \frac{\beta_{y,i} - \beta \beta_{x,i}}
       {\sqrt{s_{x,i}^2 \beta^2 + s_{y,i}^2 + \tau^2}},

    ``beta`` maximises ``-sum rho(t_i)`` at fixed ``tau2`` and ``tau2``
    solves ``sum rho(t_i) = (p - 1) E[rho(Z)]`` (``tau2 = 0`` when the
    left side is already below the right at ``tau2 = 0``). The variance is
    GRAPPLE's sandwich ``A^-1 B A^-T`` over ``(beta, tau2)``, with the
    package's loss functions (Tukey biweight scaled as
    ``1 - (1 - (r/k)^2)^3``) and moments.

    Parameters
    ----------
    beta_exposure, beta_outcome, se_exposure, se_outcome : ndarray
    loss : {'tukey', 'huber', 'l2'}, default 'tukey'
        GRAPPLE's ``loss.function`` (its default is Tukey).
    k : float, optional
        Loss tuning constant; GRAPPLE's defaults 4.685 (Tukey), 1.345
        (Huber).
    alpha : float, default 0.05
    beta_init : float, optional
        Start for ``beta``; default GRAPPLE's: the maximiser of the
        objective at ``tau2 = 0`` over a 5000-point grid on
        ``+/- 2 * quantile(|by / bx|, 0.95)``.
    tau2_init : float, default 1e-4
        Start of ``tau2`` when ``beta_init`` is given (otherwise GRAPPLE's
        start, ``tau2 = 0``, is used). The alternation re-solves ``tau2``
        first, so the start only matters for multiple roots.
    max_iter : int, default 200
        Maximum number of (tau2, beta) alternation rounds. The result's
        ``converged`` flag records whether ``tol`` was met.
    tol : float, default 1e-13
        Relative convergence tolerance of the alternation. GRAPPLE stops
        its ``tau2`` root at ``bound * eps^0.25`` (absolute) and its
        ``beta`` search at ``optim``'s default, so its reported numbers are
        a few significant digits short of the solution computed here.

    Returns
    -------
    :class:`GrappleResult`

    Notes
    -----
    Before 1.30 this function maximised a Gaussian likelihood (with the
    ``log`` variance term) jointly in ``(beta, log tau2)`` and took the SE
    from a numerical Hessian: a different estimator from GRAPPLE's robust
    profile score (on MendelianRandomization's LDL-C data, beta 2.694 vs
    GRAPPLE's 2.720 with the Tukey loss, SE 0.510 vs 0.552).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> p = 30                                   # number of instruments (SNPs)
    >>> bx = rng.normal(0.3, 0.1, p)             # SNP-exposure associations
    >>> sx = np.full(p, 0.05)                    # their standard errors
    >>> by = 0.5 * bx + rng.normal(0, 0.02, p)   # true causal effect ~0.5
    >>> sy = np.full(p, 0.05)
    >>> res = sp.grapple(bx, by, sx, sy)
    >>> round(res.estimate, 1)
    0.5
    >>> bool(res.converged)
    True
    """
    from scipy.optimize import brentq, minimize_scalar

    from .raps import _gauss_moment, _rho

    bx, by, sx, sy = as_float_arrays(
        beta_exposure, beta_outcome, se_exposure, se_outcome
    )
    if loss not in ("tukey", "huber", "l2"):
        raise MethodIncompatibility(
            f"loss must be 'tukey', 'huber' or 'l2'; got {loss!r}"
        )
    if k is None:
        k = {"tukey": 4.685, "huber": 1.345, "l2": float("nan")}[loss]
    rho = _l2_rho if loss == "l2" else _rho(loss, float(k))
    vx, vy = sx**2, sy**2
    p = len(bx)

    delta = _gauss_moment(lambda x: rho(x))
    c1 = _gauss_moment(lambda x: rho(x, deriv=1) ** 2)
    c2 = _gauss_moment(lambda x: rho(x) ** 2) - delta**2
    c4 = _gauss_moment(lambda x: rho(x, deriv=1) * x)
    c3 = c4

    def obj(b: Any, t2: Any) -> Any:  # to minimise
        return float(np.sum(rho(_t_values(b, t2, bx, by, vx, vy))))

    def tau_eq(t2: Any, b: Any) -> Any:
        return float(np.sum(rho(_t_values(b, t2, bx, by, vx, vy)))) - (p - 1) * delta

    ratio = np.abs(by / bx)
    bound_beta = 2.0 * float(np.quantile(ratio[np.isfinite(ratio)], 0.95))
    bound_tau2 = 2.0 * float(np.median(by**2))

    def solve_tau2(b: Any) -> Any:
        if tau_eq(0.0, b) < 0:
            return 0.0
        hi = bound_tau2
        while tau_eq(hi, b) > 0:
            hi *= 2.0
        return float(
            brentq(
                tau_eq, 0.0, hi, args=(b,), xtol=1e-300, rtol=4 * np.finfo(float).eps
            )
        )

    def score(
        b: Any, t2: Any
    ) -> Any:  # d/d beta of sum rho(t), up to the loss's scaling
        res = by - bx * b
        v = vx * b**2 + vy + t2
        return float(
            np.sum(rho(res / np.sqrt(v), deriv=1) * (v * bx + res * vx * b) / v**1.5)
        )

    grid = np.linspace(-bound_beta, bound_beta, 5000)
    h = grid[1] - grid[0]

    def solve_beta(t2: Any, start: Any) -> Any:
        # Global search on GRAPPLE's grid, local polish, then the root of
        # the score in a bracket around the minimiser (machine precision;
        # a minimiser of a flat objective is only good to ~sqrt(eps)).
        cands = [float(grid[int(np.argmin([obj(g, t2) for g in grid]))]), start]
        best = None
        for c in cands:
            r = minimize_scalar(
                lambda b: obj(b, t2),
                bounds=(c - h, c + h),
                method="bounded",
                options={"xatol": 1e-14},
            )
            if best is None or r.fun < best.fun:
                best = r
        b0 = float(best.x)
        w = max(abs(b0) * 1e-6, 1e-12)
        for _ in range(40):
            lo, hi = b0 - w, b0 + w
            if score(lo, t2) * score(hi, t2) < 0:
                return float(
                    brentq(
                        score,
                        lo,
                        hi,
                        args=(t2,),
                        xtol=1e-300,
                        rtol=4 * np.finfo(float).eps,
                    )
                )
            w *= 2.0
            if w > h:
                break
        return b0

    if beta_init is None:
        beta = float(grid[int(np.argmin([obj(g, 0.0) for g in grid]))])
        tau2 = 0.0
    else:
        beta = float(beta_init)
        tau2 = float(tau2_init)

    converged = False
    for _ in range(max_iter):
        tau2_new = solve_tau2(beta)
        beta_new = solve_beta(tau2_new, beta)
        step = abs(beta_new - beta) / max(abs(beta_new), 1e-10) + abs(
            tau2_new - tau2
        ) / max(abs(tau2_new), 1e-10)
        if tau2_new == 0.0 and tau2 == 0.0:
            step = abs(beta_new - beta) / max(abs(beta_new), 1e-10)
        beta, tau2 = beta_new, tau2_new
        if step <= tol:
            converged = True
            break

    if not converged:
        warnings.warn(
            f"grapple: the (tau2, beta) alternation did not reach tol={tol:g} "
            f"in max_iter={max_iter} rounds; estimates may be off the optimum.",
            RuntimeWarning,
            stacklevel=2,
        )
    V = _grapple_vcov(beta, tau2, bx, by, vx, vy, rho, c1, c2, c3, c4)
    se_hat = float(np.sqrt(V[0, 0])) if V[0, 0] > 0 else float("nan")
    tau2_se = float(np.sqrt(V[1, 1])) if tau2 > 0 and V[1, 1] > 0 else float("nan")

    z_crit = stats.norm.ppf(1 - alpha / 2)
    if np.isfinite(se_hat) and se_hat > 0:
        p_value = float(min(1.0, 2.0 * stats.norm.sf(abs(beta) / se_hat)))
        lo = beta - z_crit * se_hat
        hi = beta + z_crit * se_hat
    else:
        p_value = float("nan")
        lo = hi = float("nan")

    return GrappleResult(
        estimate=float(beta),
        se=se_hat,
        ci_lower=float(lo),
        ci_upper=float(hi),
        p_value=p_value,
        tau2=float(tau2),
        loglik=-obj(beta, tau2),
        converged=converged,
        n_snps=p,
        tau2_se=tau2_se,
        loss=loss,
    )


def _grapple_vcov(
    beta: Any,
    tau2: Any,
    bx: Any,
    by: Any,
    vx: Any,
    vy: Any,
    rho: Any,
    c1: Any,
    c2: Any,
    c3: Any,
    c4: Any,
) -> np.ndarray:
    """GRAPPLE's asymptotic variance of (beta, tau2), one exposure.

    Transcribes ``grappleRobustEst``'s block for ``r = 1`` and
    ``cor.mat = I``: ``coefs = s_x^2 beta``,
    ``u = -(v bx + res coefs) / v^(3/2)``, ``S = sum u^2``,
    ``B = diag(c1 S, p c2)``, ``A22 = -c3/2 sum 1/v``,
    ``A12 = c4 sum coefs / v^2``, ``A21 = 0`` and
    ``A11 = c4 S - (temp1 + temp)`` with ``temp1 = sum rho'(t) bx coefs /
    v^(3/2)`` and ``temp = sum res rho'(t) s_x^2 / v^(3/2)``.
    """
    p = len(bx)
    res = by - bx * beta
    v = vx * beta**2 + vy + tau2
    t = res / np.sqrt(v)
    coefs = vx * beta
    u = -(v * bx + res * coefs) / v**1.5
    S = float(u @ u)
    B = np.diag([c1 * S, p * c2])
    rdv = rho(t, deriv=1) / v**1.5
    temp = float(np.sum(res * rdv * vx))
    temp1 = float(np.sum(rdv * bx * coefs))
    A = np.array(
        [
            [c4 * S - (temp1 + temp), c4 * float(np.sum(coefs / v**2))],
            [0.0, -c3 / 2 * float(np.sum(1 / v))],
        ]
    )
    Ai = np.linalg.inv(A)
    return Ai @ B @ Ai.T
