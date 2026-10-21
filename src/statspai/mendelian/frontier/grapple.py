"""GRAPPLE: robust adjusted profile-likelihood MR (single exposure).

Implements the estimator of Wang et al. (2021), Materials and Methods
"Estimate the effects beta" and S2 Text, for one exposure and two
non-overlapping GWAS. For variant ``j`` let

.. math::

    t_j(\\beta, \\tau^2) = \\frac{\\hat\\Gamma_j - \\beta \\hat\\gamma_j}
        {\\sqrt{\\sigma_{Yj}^2 + \\beta^2 \\sigma_{Xj}^2 + \\tau^2}}

(their Equation 10). The estimate solves

* ``phi_1 = d/d beta sum_j rho(t_j) = 0`` (the robust profile likelihood
  ``l = -sum_j rho(t_j)``, Equation 11, at a stationary point), and
* ``phi_2 = sum_j rho(t_j) - (p - 1) E[rho(Z)] = 0`` with ``Z ~ N(0, 1)``
  (Equation 12 with one degree of freedom for ``beta``; ``tau^2 = 0`` when
  the left side is already negative at ``tau^2 = 0``),

with ``rho`` Tukey's biweight (default, ``c = 4.685``), Huber's loss or
the squared loss. The variance is the sandwich ``A^-1 B A^-T`` of S2 Text.

Clean-room implementation written from the paper and its S2 Text; the
reference (R ``GRAPPLE`` 0.2.2, GPL (>= 2)) was used only as a black box.
See ``docs/dev/cleanroom/grapple.md``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
from scipy import integrate, optimize, stats

from ..._result_serialize import ResultProtocolMixin
from ...exceptions import MethodIncompatibility
from ._common import as_float_arrays

_Arr = np.ndarray

__all__ = ["GrappleResult", "grapple"]

_DEFAULT_TUNING = {"tukey": 4.685, "huber": 1.345}


@dataclass
class GrappleResult(ResultProtocolMixin):
    """Output of :func:`grapple`.

    Attributes
    ----------
    estimate : float
        Robust profile-score estimate of the causal effect β.
    se : float
        Sandwich standard error of the estimate (S2 Text).
    ci_lower, ci_upper : float
    p_value : float
    tau2 : float
        Estimated pleiotropy variance τ² (balanced pleiotropy SD² = τ²).
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
        lines = [
            "GRAPPLE robust profile-likelihood MR (Wang et al. 2021)",
            "=" * 60,
            f"  loss            : {self.loss}",
            f"  n SNPs          : {self.n_snps}",
            f"  Estimate        : {self.estimate:.6g}",
            f"  SE              : {self.se:.6g}",
            f"  CI              : [{self.ci_lower:.6g}, {self.ci_upper:.6g}]",
            f"  p-value         : {self.p_value:.4g}",
            f"  tau^2           : {self.tau2:.6g} (SE {self.tau2_se:.3g})",
            f"  objective       : {self.loglik:.6g}",
            f"  converged       : {self.converged}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Loss functions and their standard-normal moments
# --------------------------------------------------------------------------


def _l2_rho(r: Any, deriv: int = 0) -> Any:
    """Squared loss ``r^2 / 2`` and its first two derivatives."""
    r = np.asarray(r, dtype=float)
    if deriv == 0:
        return 0.5 * r**2
    if deriv == 1:
        return r
    return np.ones_like(r)


def _tukey_rho(c: float) -> Callable[..., Any]:
    """Tukey's biweight ``c^2/6 [1 - (1 - (r/c)^2)^3]`` capped at ``c^2/6``."""

    def rho(r: Any, deriv: int = 0) -> Any:
        r = np.asarray(r, dtype=float)
        u2 = (r / c) ** 2
        inside = u2 <= 1.0
        w = 1.0 - u2
        if deriv == 0:
            return np.where(inside, c**2 / 6.0 * (1.0 - w**3), c**2 / 6.0)
        if deriv == 1:
            return np.where(inside, r * w**2, 0.0)
        return np.where(inside, w * (1.0 - 5.0 * u2), 0.0)

    return rho


def _huber_rho(k: float) -> Callable[..., Any]:
    """Huber's loss: ``r^2/2`` for ``|r| <= k``, ``k|r| - k^2/2`` beyond."""

    def rho(r: Any, deriv: int = 0) -> Any:
        r = np.asarray(r, dtype=float)
        inside = np.abs(r) <= k
        if deriv == 0:
            return np.where(inside, 0.5 * r**2, k * np.abs(r) - 0.5 * k**2)
        if deriv == 1:
            return np.clip(r, -k, k)
        return inside.astype(float)

    return rho


def _robust_rho(loss: str, k: Optional[float]) -> Callable[..., Any]:
    """Loss ``rho(r, deriv)`` for ``loss`` in {tukey, huber, l2}."""
    if loss == "l2":
        return _l2_rho
    tuning = _DEFAULT_TUNING[loss] if k is None else float(k)
    return _tukey_rho(tuning) if loss == "tukey" else _huber_rho(tuning)


def _normal_expectation(fn: Callable[[float], float], kinks: tuple = ()) -> float:
    """``E[fn(Z)]`` for ``Z ~ N(0, 1)`` by adaptive quadrature.

    The real line is split at ``kinks`` (where the loss changes form) so
    each piece is smooth.
    """
    pts = sorted({float(x) for x in kinks} | {-np.inf, np.inf})
    total = 0.0
    for lo, hi in zip(pts[:-1], pts[1:]):
        val, _ = integrate.quad(
            lambda z: fn(z) * stats.norm.pdf(z), lo, hi, epsabs=1e-14, epsrel=1e-13
        )
        total += val
    return float(total)


def _loss_moments(loss: str, rho: Callable[..., Any], k: Optional[float]) -> dict:
    """Standard-normal moments of the loss used by the estimator.

    ``eta = E rho(Z)``, ``score_sq = E rho'(Z)^2``, ``rho_var = Var rho(Z)``,
    ``slope = E[Z rho'(Z)] (= E rho''(Z) by Stein's identity)``.
    """
    if loss == "l2":
        return {"eta": 0.5, "score_sq": 1.0, "rho_var": 0.5, "slope": 1.0}
    c = _DEFAULT_TUNING[loss] if k is None else float(k)
    kinks = (-c, c)
    eta = _normal_expectation(lambda z: float(rho(z)), kinks)
    return {
        "eta": eta,
        "score_sq": _normal_expectation(lambda z: float(rho(z, deriv=1)) ** 2, kinks),
        "rho_var": _normal_expectation(lambda z: float(rho(z)) ** 2, kinks) - eta**2,
        "slope": _normal_expectation(lambda z: z * float(rho(z, deriv=1)), kinks),
    }


# --------------------------------------------------------------------------
# Profile statistics
# --------------------------------------------------------------------------


def _t_values(beta: Any, tau2: Any, bx: Any, by: Any, vx: Any, vy: Any) -> Any:
    """Standardized residuals ``t_j(beta, tau2)`` (Equation 10, one exposure)."""
    return (by - beta * bx) / np.sqrt(vy + beta**2 * vx + tau2)


def _dt_dbeta(
    beta: float, tau2: float, bx: _Arr, by: _Arr, vx: _Arr, vy: _Arr
) -> Tuple[_Arr, _Arr]:
    """First and second derivatives of ``t_j`` in ``beta``."""
    v = vy + beta**2 * vx + tau2
    resid = by - beta * bx
    d1 = -(v * bx + resid * beta * vx) / v**1.5
    d2 = (
        2.0 * bx * beta * vx / v**1.5
        + 3.0 * resid * beta**2 * vx**2 / v**2.5
        - resid * vx / v**1.5
    )
    return d1, d2


def _beta_score(
    beta: float,
    tau2: float,
    bx: _Arr,
    by: _Arr,
    vx: _Arr,
    vy: _Arr,
    rho: Callable[..., Any],
) -> float:
    """``phi_1``: derivative of ``sum_j rho(t_j)`` in ``beta``."""
    t = _t_values(beta, tau2, bx, by, vx, vy)
    d1, _ = _dt_dbeta(beta, tau2, bx, by, vx, vy)
    return float(np.sum(rho(t, deriv=1) * d1))


def _solve_tau2(
    beta: float,
    bx: _Arr,
    by: _Arr,
    vx: _Arr,
    vy: _Arr,
    rho: Callable[..., Any],
    eta: float,
) -> float:
    """Root in ``tau2 >= 0`` of ``sum_j rho(t_j) - (p - 1) * eta``.

    Equation 12 with one degree of freedom spent on ``beta`` (the
    reference's convention, see the clean-room spec). ``sum_j rho(t_j)``
    decreases in ``tau2``; when it is already below ``(p - 1) * eta`` at
    ``tau2 = 0`` the estimate is the boundary value 0.
    """
    target = (bx.size - 1) * eta

    def excess(tau2: float) -> float:
        return float(np.sum(rho(_t_values(beta, tau2, bx, by, vx, vy)))) - target

    if excess(0.0) <= 0.0:
        return 0.0
    hi = float(np.max(vy + beta**2 * vx))
    while excess(hi) > 0.0:
        hi *= 4.0
        if not np.isfinite(hi):
            return float("nan")
    return float(optimize.brentq(excess, 0.0, hi, xtol=1e-300, rtol=1e-15, maxiter=500))


def _solve_beta(
    beta0: float,
    tau2: float,
    bx: _Arr,
    by: _Arr,
    vx: _Arr,
    vy: _Arr,
    rho: Callable[..., Any],
    scale: float,
) -> float:
    """Local minimiser of ``sum_j rho(t_j(beta, tau2))`` nearest ``beta0``.

    A bracketing step search downhill from ``beta0`` followed by Brent's
    root finder on the score ``phi_1`` inside the bracket.
    """

    def score(b: float) -> float:
        return _beta_score(b, tau2, bx, by, vx, vy, rho)

    s0 = score(beta0)
    if s0 == 0.0:
        return float(beta0)
    direction = -np.sign(s0)  # downhill
    step = scale
    lo = beta0
    for _ in range(200):
        hi = lo + direction * step
        if np.sign(score(hi)) != np.sign(s0):
            a, b = sorted((lo, hi))
            return float(
                optimize.brentq(score, a, b, xtol=1e-300, rtol=1e-15, maxiter=500)
            )
        lo = hi
        step *= 1.6
    return float("nan")


# --------------------------------------------------------------------------
# Sandwich variance
# --------------------------------------------------------------------------


def _grapple_vcov(
    beta: Any,
    tau2: Any,
    bx: Any,
    by: Any,
    vx: Any,
    vy: Any,
    rho: Any,
    score_sq: Any,
    rho_var: Any,
    curvature: Any,
    slope: Any,
) -> np.ndarray:
    """Asymptotic covariance of ``(beta_hat, tau2_hat)`` (S2 Text).

    ``A^-1 B A^-T`` for the estimating equations ``(phi_1, phi_2)`` with,
    for one exposure, ``v_j = vy_j + beta^2 vx_j + tau2`` and
    ``d_j = dt_j / dbeta = -(v_j bx_j + (by_j - beta bx_j) beta vx_j) / v_j^1.5``:

    * ``B = diag(E[rho'(Z)^2] sum_j d_j^2, p Var[rho(Z)])`` (the
      off-diagonal has expectation 0);
    * ``A_11 = E[rho''(Z)] sum_j d_j^2 + sum_j rho'(t_j) h_j`` with
      ``h_j = -vx_j by_j / v_j^1.5``;
    * ``A_12 = E[Z rho'(Z)] sum_j beta vx_j / v_j^2``;
    * ``A_21 = 0`` (its expectation);
    * ``A_22 = -E[Z rho'(Z)] / 2 sum_j 1 / v_j``.

    ``h_j`` stands in for ``d^2 t_j / dbeta^2``: both have expectation
    ``-vx_j (vy_j + tau2) / v_j^2`` times ``E[Z rho'(Z)]`` once multiplied by
    ``rho'(t_j)`` at the true values, so either is a consistent plug-in for
    S2 Text's ``E[sum_j rho'(t_j) d^2 t_j / dbeta^2]``; ``h_j`` is the one
    whose numbers the reference reports (see the clean-room spec).

    ``score_sq`` (``E rho'(Z)^2``), ``rho_var`` (``Var rho(Z)``),
    ``curvature`` (``E rho''(Z)``) and ``slope`` (``E[Z rho'(Z)]``) are the
    standard-normal moments of the loss ``rho``.
    """
    bx, by, vx, vy = (np.asarray(a, dtype=float) for a in (bx, by, vx, vy))
    p = bx.size
    v = vy + beta**2 * vx + tau2
    t = _t_values(beta, tau2, bx, by, vx, vy)
    d1, _ = _dt_dbeta(beta, tau2, bx, by, vx, vy)
    h = -vx * by / v**1.5
    a11 = curvature * np.sum(d1**2) + np.sum(rho(t, deriv=1) * h)
    a12 = slope * np.sum(beta * vx / v**2)
    a22 = -0.5 * slope * np.sum(1.0 / v)
    A = np.array([[a11, a12], [0.0, a22]])
    B = np.diag([score_sq * np.sum(d1**2), p * rho_var])
    A_inv = np.linalg.inv(A)
    cov: np.ndarray = A_inv @ B @ A_inv.T
    return cov


# --------------------------------------------------------------------------
# Estimator
# --------------------------------------------------------------------------


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
    """GRAPPLE robust profile-likelihood MR (single exposure).

    Estimates the causal effect ``beta`` and the balanced-pleiotropy
    variance ``tau^2`` from two-sample summary statistics under
    ``Gamma_j = beta gamma_j + alpha_j``, ``alpha_j ~ N(0, tau^2)``
    (Wang et al. 2021). Both weak instruments (through the profile
    likelihood, which accounts for the exposure measurement error) and a
    few gross pleiotropic outliers (through the robust loss) are allowed.

    Parameters
    ----------
    beta_exposure, beta_outcome : array-like
        Per-variant associations with the exposure and the outcome.
    se_exposure, se_outcome : array-like
        Their standard errors (non-negative; ``se_outcome`` positive).
    loss : {"tukey", "huber", "l2"}, default "tukey"
        Robust loss ``rho``; Tukey's biweight is the paper's default.
    k : float, optional
        Loss tuning constant; defaults 4.685 (Tukey) and 1.345 (Huber),
        the reference's defaults.
        Ignored for ``"l2"``.
    alpha : float, default 0.05
        Two-sided level of the normal confidence interval.
    beta_init : float, optional
        Starting value for ``beta``. By default the squared-loss GRAPPLE
        fit (started from the inverse-variance weighted estimate) is used,
        so a robust fit picks the mode of the robust objective nearest the
        Gaussian solution.
    tau2_init : float, default 1e-4
        ``tau^2`` used in the first ``beta`` step; negative values are
        treated as 0.
    tol : float, default 1e-13
        Relative tolerance of the alternation between the ``tau^2`` root
        and the ``beta`` score root.
    max_iter : int, default 200
        Maximum number of alternation rounds; ``converged`` reports
        whether ``tol`` was met.

    Returns
    -------
    GrappleResult

    Notes
    -----
    ``beta`` solves the score equation of the robust profile likelihood
    and ``tau^2`` the moment equation ``sum_j rho(t_j) = (p - 1) E[rho(Z)]``;
    when that equation has no non-negative root the estimate is
    ``tau^2 = 0``. Standard errors are the sandwich of S2 Text (see
    :func:`_grapple_vcov`). The multi-mode diagnosis, multivariable fits
    and sample-overlap correction of the GRAPPLE framework are not
    implemented.

    Clean-room implementation; the reference (GRAPPLE 0.2.2, GPL (>= 2))
    was used only as a black box.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> bx = rng.uniform(0.1, 0.5, 30)
    >>> sx = np.full(30, 0.02)
    >>> sy = np.full(30, 0.02)
    >>> by = 0.4 * bx + rng.normal(0, sy)
    >>> r = sp.grapple(bx, by, sx, sy)
    >>> abs(r.estimate - 0.4) < 0.1
    True

    References
    ----------
    wang2021causal
    """
    if loss not in ("tukey", "huber", "l2"):
        raise MethodIncompatibility(
            f"loss must be 'tukey', 'huber' or 'l2'; got {loss!r}",
            recovery_hint="Use loss='tukey' (default), 'huber' or 'l2'.",
        )
    bx, by, sx, sy = as_float_arrays(
        beta_exposure, beta_outcome, se_exposure, se_outcome
    )
    for name, arr in (
        ("beta_exposure", bx),
        ("beta_outcome", by),
        ("se_exposure", sx),
        ("se_outcome", sy),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
    if np.any(sx < 0) or np.any(sy <= 0):
        raise ValueError(
            "se_exposure must be non-negative and se_outcome strictly positive"
        )
    if k is not None and loss != "l2" and not float(k) > 0:
        raise ValueError(f"k must be positive; got {k}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    vx, vy = sx**2, sy**2
    p = bx.size
    rho = _robust_rho(loss, k)
    mom = _loss_moments(loss, rho, k)
    eta = mom["eta"]
    # Step scale for the beta bracket search: one inverse-variance SE.
    scale = float(1.0 / np.sqrt(np.sum(bx**2 / vy)))

    def fit(
        rho_fn: Callable[..., Any],
        eta_fn: float,
        beta_start: float,
        tau2_start: float,
        iters: int,
    ) -> Tuple[float, float, bool]:
        beta, tau2 = beta_start, max(float(tau2_start), 0.0)
        beta = _solve_beta(beta, tau2, bx, by, vx, vy, rho_fn, scale)
        for it in range(iters):
            tau2_new = _solve_tau2(beta, bx, by, vx, vy, rho_fn, eta_fn)
            beta_new = _solve_beta(beta, tau2_new, bx, by, vx, vy, rho_fn, scale)
            if not (np.isfinite(beta_new) and np.isfinite(tau2_new)):
                return beta_new, tau2_new, False
            done = abs(beta_new - beta) <= tol * max(abs(beta_new), scale) and abs(
                tau2_new - tau2
            ) <= tol * max(tau2_new, 1e-300)
            beta, tau2 = beta_new, tau2_new
            if done:
                return beta, tau2, True
        return beta, tau2, False

    ivw = float(np.sum(bx * by / vy) / np.sum(bx**2 / vy))
    if beta_init is None:
        start = ivw
        if loss != "l2":
            start, _, _ = fit(_l2_rho, 0.5, ivw, tau2_init, max_iter)
            if not np.isfinite(start):
                start = ivw
    else:
        start = float(beta_init)
    beta, tau2, converged = fit(rho, eta, start, tau2_init, max_iter)
    if not converged:
        warnings.warn(
            f"grapple: the (tau2, beta) alternation did not reach tol={tol} "
            f"in max_iter={max_iter} rounds; estimates may be off the optimum.",
            RuntimeWarning,
            stacklevel=2,
        )

    V = _grapple_vcov(
        beta,
        tau2,
        bx,
        by,
        vx,
        vy,
        rho,
        mom["score_sq"],
        mom["rho_var"],
        mom["slope"],
        mom["slope"],
    )
    se = float(np.sqrt(V[0, 0])) if V[0, 0] > 0 else float("nan")
    # At the boundary tau2 = 0 the normal approximation for tau2_hat does not
    # hold, so no SE is reported for it (the beta SE keeps the full sandwich).
    tau2_se = float(np.sqrt(V[1, 1])) if tau2 > 0 and V[1, 1] > 0 else float("nan")
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    p_value = float(2.0 * stats.norm.sf(abs(beta / se))) if se > 0 else float("nan")
    objective = -float(np.sum(rho(_t_values(beta, tau2, bx, by, vx, vy))))
    return GrappleResult(
        estimate=float(beta),
        se=se,
        ci_lower=float(beta - z * se),
        ci_upper=float(beta + z * se),
        p_value=p_value,
        tau2=float(tau2),
        loglik=objective,
        converged=bool(converged),
        n_snps=int(p),
        tau2_se=tau2_se,
        loss=loss,
    )
