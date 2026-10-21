"""MR-RAPS: robust adjusted profile score for two-sample summary-data MR.

Model (Zhao, Wang, Hemani, Bowden & Small 2020, Sections 3-5)
--------------------------------------------------------------
Variant ``j`` contributes an exposure association :math:`\\hat\\gamma_j \\sim
N(\\gamma_j, \\sigma_{Xj}^2)` and an outcome association :math:`\\hat\\Gamma_j
\\sim N(\\beta\\gamma_j, \\sigma_{Yj}^2 + \\tau^2)`. Three estimators are offered.

* ``over_dispersion=False`` (``loss="l2"``): the profile-likelihood
  estimator of Section 3, :math:`\\hat\\beta = \\arg\\max_\\beta l(\\beta)` with
  :math:`l(\\beta) = -\\tfrac12\\sum_j (\\hat\\Gamma_j - \\beta\\hat\\gamma_j)^2 /
  (\\sigma_{Xj}^2\\beta^2 + \\sigma_{Yj}^2)` (eq. 3.2), standard error
  :math:`\\hat V_1^{1/2} / \\hat V_2` (Theorem 3.3).
* ``over_dispersion=True, loss="l2"``: the adjusted profile score (APS)
  estimator, the finite root of eqs. (4.2)-(4.3), sandwich variance of
  Theorem 4.2.
* ``over_dispersion=True, loss="huber" | "tukey"``: the robust adjusted
  profile score (RAPS), the root of eqs. (5.2)-(5.3) with
  :math:`t_j = (\\hat\\Gamma_j - \\beta\\hat\\gamma_j) / (\\sigma_{Xj}^2\\beta^2 +
  \\sigma_{Yj}^2 + \\tau^2)^{1/2}` and :math:`\\delta = E[Z\\rho'(Z)]`, sandwich
  variance of Theorem 5.2.

Clean-room implementation written from the paper (``zhao2020statistical``)
and the specification ``docs/dev/cleanroom/mr_raps.md``. The reference
(R ``mr.raps`` 0.4.3, GPL-3) was used only as a black box: its help pages and
its outputs on committed data (``tests/reference_parity/_fixtures``).

References
----------
zhao2020statistical
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Optional, Tuple

import numpy as np
from scipy import integrate, optimize, stats

from ..._result_serialize import ResultProtocolMixin
from ._common import as_float_arrays

_Arr = np.ndarray
_Psi = Callable[[np.ndarray], np.ndarray]

_EPS_HALF = float(np.finfo(float).eps ** 0.5)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

#: Default tuning constants (95% Gaussian efficiency in the location model,
#: Zhao et al. 2020, Section 5.3).
_DEFAULT_TUNING = {"huber": 1.345, "tukey": 4.685}

#: Variants whose ``se_exposure`` exceeds this multiple of the median are
#: dropped when ``pruning=True`` (convention checked against the reference).
_PRUNE_RATIO = 10.0

#: Gaussian moments of the squared-error loss rho(r) = r^2 / 2:
#: (delta, E[rho'^2], Var(Z rho'(Z)), E[Z^2 rho'']).
_L2_MOMENTS = (1.0, 1.0, 2.0, 1.0)


@dataclass
class MRRapsResult(ResultProtocolMixin):
    """Output of :func:`mr_raps`.

    Attributes
    ----------
    estimate : float
        Causal effect estimate.
    se : float
        Asymptotic standard error: Theorem 3.3 of Zhao et al. (2020) for the
        model without over-dispersion, the ``beta`` entry of the joint
        ``(beta, tau2)`` sandwich (Theorems 4.2 / 5.2) otherwise.
    ci_lower, ci_upper : float
        Normal confidence interval at level ``1 - alpha``.
    p_value : float
        Two-sided normal p-value for ``beta = 0``.
    tau2 : float
        Over-dispersion variance (0 for the model without it).
    tau2_se : float
        Standard error of ``tau2`` (``nan`` without over-dispersion).
    loglik_robust : float
        Objective at the estimate, ``-sum_j rho(t_j)`` (eq. 5.1, which is the
        profile log-likelihood (3.2) when ``rho(r) = r^2 / 2``).
    converged : bool
        Whether the estimating equations were solved within ``niter``
        iterations to ``tol``.
    tuning_c : float
        Loss tuning constant (``nan`` for ``loss="l2"``).
    n_snps : int
        Variants used, after pruning.
    loss : str
    over_dispersion : bool
    n_pruned : int
        Variants dropped because ``se_exposure > 10 * median(se_exposure)``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(8)
    >>> bx = rng.uniform(0.1, 0.5, 25)
    >>> sx = rng.uniform(0.02, 0.05, 25)
    >>> sy = rng.uniform(0.02, 0.05, 25)
    >>> by = 0.3 * bx + rng.normal(0, sy)
    >>> res = sp.mr_raps(bx, by, sx, sy)
    >>> type(res).__name__
    'MRRapsResult'
    >>> res.n_snps
    25
    >>> bool(np.isfinite(res.estimate))
    True
    """

    _citation_keys = ("zhao2020statistical",)

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
    tau2_se: float = float("nan")
    loss: str = "huber"
    over_dispersion: bool = True
    n_pruned: int = 0

    def summary(self) -> str:
        if self.over_dispersion:
            label = {
                "l2": "adjusted profile score",
                "huber": "robust adjusted profile score, Huber loss",
                "tukey": "robust adjusted profile score, Tukey biweight",
            }[self.loss]
        else:
            label = "profile score, no over-dispersion"
        tuning = "" if self.loss == "l2" else f" (k = {self.tuning_c:g})"
        lines = [
            f"MR-RAPS ({label}){tuning}",
            "=" * 62,
            f"  n SNPs        : {self.n_snps}"
            + (f"  ({self.n_pruned} pruned)" if self.n_pruned else ""),
            f"  estimate      : {self.estimate:+.4f}  (SE {self.se:.4f})",
            f"  CI            : [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]",
            f"  p-value       : {self.p_value:.4g}",
        ]
        if self.over_dispersion:
            lines.append(f"  tau^2         : {self.tau2:.4g}  (SE {self.tau2_se:.3g})")
        conv = "converged" if self.converged else "NOT converged"
        lines.append(f"  objective     : {self.loglik_robust:.3f}  ({conv})")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Loss functions and their Gaussian moments
# --------------------------------------------------------------------------- #


def _rho(loss: str, k: float) -> Callable[..., np.ndarray]:
    """Robust loss ``rho(r; k)`` and its first two derivatives.

    ``deriv=0`` is the loss of Zhao et al. (2020), Section 5.1:
    Huber ``r^2/2`` inside ``[-k, k]`` and ``k(|r| - k/2)`` outside; Tukey
    ``1 - (1 - (r/k)^2)^3`` inside and ``1`` outside. ``deriv=1`` and
    ``deriv=2`` return the score function and its derivative. For Tukey the
    score is the usual biweight ``r (1 - (r/k)^2)^2``, i.e. ``k^2/6`` times the
    derivative of the bounded loss above; the estimating equations and the
    sandwich variance are invariant to that factor (it cancels between the
    two Gaussian moments that enter each), and it is the scaling the
    reference's documented ``rho.tukey`` returns.
    """
    if loss == "huber":

        def rho(r: np.ndarray, deriv: int = 0) -> np.ndarray:
            r = np.asarray(r, dtype=float)
            inside = np.abs(r) <= k
            if deriv == 0:
                return np.where(inside, 0.5 * r**2, k * (np.abs(r) - 0.5 * k))
            if deriv == 1:
                return np.clip(r, -k, k)
            d2: np.ndarray = inside.astype(float)
            return d2

    elif loss == "tukey":

        def rho(r: np.ndarray, deriv: int = 0) -> np.ndarray:
            r = np.asarray(r, dtype=float)
            u2 = (r / k) ** 2
            inside = u2 <= 1.0
            if deriv == 0:
                return np.where(inside, 1.0 - (1.0 - u2) ** 3, 1.0)
            if deriv == 1:
                return np.where(inside, r * (1.0 - u2) ** 2, 0.0)
            return np.where(inside, (1.0 - u2) * (1.0 - 5.0 * u2), 0.0)

    else:
        raise ValueError(f"unknown robust loss {loss!r}")
    return rho


def _gauss_moment(f: Callable[[float], Any]) -> float:
    """``E[f(Z)]`` for ``Z ~ N(0, 1)`` by adaptive quadrature."""

    def integrand(x: float) -> float:
        return float(f(x)) * math.exp(-0.5 * x * x) / _SQRT_2PI

    # The robust scores are piecewise smooth with kinks inside [-10, 10];
    # splitting there lets quad resolve them, the tails are negligible.
    total = 0.0
    edges = (-np.inf, -10.0, -5.0, 0.0, 5.0, 10.0, np.inf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        val, _ = integrate.quad(
            integrand, lo, hi, epsabs=1e-14, epsrel=1e-12, limit=200
        )
        total += val
    return float(total)


@lru_cache(maxsize=32)
def _loss_moments(loss: str, k: float) -> Tuple[float, float, float, float]:
    """``(delta, c1, var(Z rho'(Z)), c3)`` of Theorem 5.2 for ``Z ~ N(0, 1)``."""
    if loss == "l2":
        return _L2_MOMENTS
    rho = _rho(loss, k)
    delta = _gauss_moment(lambda x: x * rho(x, 1))
    c1 = _gauss_moment(lambda x: rho(x, 1) ** 2)
    var_zpsi = _gauss_moment(lambda x: (x * rho(x, 1)) ** 2) - delta**2
    c3 = _gauss_moment(lambda x: x**2 * rho(x, 2))
    return delta, c1, var_zpsi, c3


# --------------------------------------------------------------------------- #
#  Estimating equations
# --------------------------------------------------------------------------- #


def _standardized_residuals(
    bx: np.ndarray,
    by: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    beta: float,
    tau2: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residual ``r``, variance ``v`` and ``t = r / sqrt(v)`` at ``(beta, tau2)``."""
    resid = by - beta * bx
    var = sx**2 * beta**2 + sy**2 + tau2
    return resid, var, resid / np.sqrt(var)


def _beta_score(
    bx: _Arr, by: _Arr, sx: _Arr, sy: _Arr, beta: float, tau2: float, psi: _Psi
) -> float:
    """Eq. (5.2): ``sum_j psi(t_j) u_j`` with ``u_j = -dt_j / dbeta``.

    With ``psi(t) = t`` this is the adjusted profile score (4.2), i.e. the
    derivative of the profile log-likelihood (3.2) when ``tau2 = 0``.
    """
    resid, var, t = _standardized_residuals(bx, by, sx, sy, beta, tau2)
    u = (bx * var + resid * sx**2 * beta) / var**1.5
    return float(np.sum(psi(t) * u))


def _tau2_score(
    bx: _Arr,
    by: _Arr,
    sx: _Arr,
    sy: _Arr,
    beta: float,
    tau2: float,
    psi: _Psi,
    delta: float,
) -> float:
    """Eq. (5.3): ``sum_j sx_j^2 (t_j psi(t_j) - delta) / v_j``; (4.3) for L2."""
    _, var, t = _standardized_residuals(bx, by, sx, sy, beta, tau2)
    return float(np.sum(sx**2 * (t * psi(t) - delta) / var))


def _objective(
    bx: _Arr, by: _Arr, sx: _Arr, sy: _Arr, beta: float, tau2: float, rho: _Psi
) -> float:
    """Eq. (5.1) at ``(beta, tau2)``: ``-sum_j rho(t_j)``."""
    _, _, t = _standardized_residuals(bx, by, sx, sy, beta, tau2)
    return -float(np.sum(rho(t)))


def _global_profile_max(bx: _Arr, by: _Arr, sx: _Arr, sy: _Arr) -> float:
    """Maximiser of the profile log-likelihood (3.2) over the whole real line.

    ``l(beta)`` tends to a finite limit as ``|beta| -> inf`` and may have
    more than one local maximum, so it is scanned on ``beta = tan(theta)``
    over ``theta`` in ``(-pi/2, pi/2)`` and the best cell is refined by a
    root of its derivative.
    """
    theta = np.linspace(-np.pi / 2, np.pi / 2, 4003)[1:-1]
    grid = np.tan(theta)
    resid = by[None, :] - grid[:, None] * bx[None, :]
    var = sx[None, :] ** 2 * grid[:, None] ** 2 + sy[None, :] ** 2
    loglik = -0.5 * np.sum(resid**2 / var, axis=1)
    i = int(np.argmax(loglik))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]

    def score(b: float) -> float:
        return _beta_score(bx, by, sx, sy, b, 0.0, lambda t: t)

    if score(lo) > 0 > score(hi):
        return float(
            optimize.brentq(score, lo, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps)
        )
    return float(grid[i])


def _local_beta_root(f: Callable[[float], float], beta0: float, scale: float) -> float:
    """Root of a decreasing-through-zero score nearest to ``beta0``.

    The score is the derivative of an objective in ``beta``; the root sought
    is the local maximum reached by walking uphill from ``beta0``.
    """
    f0 = f(beta0)
    if f0 == 0.0:
        return beta0
    direction = 1.0 if f0 > 0 else -1.0
    step = scale
    prev = beta0
    for _ in range(200):
        cur = beta0 + direction * step
        fc = f(cur)
        if (fc < 0) if direction > 0 else (fc > 0):
            lo, hi = (prev, cur) if direction > 0 else (cur, prev)
            return float(
                optimize.brentq(f, lo, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps)
            )
        prev = cur
        step *= 2.0
    raise RuntimeError("mr_raps: no root of the beta estimating equation found")


def _tau2_root(g: Callable[[float], float], scale: float) -> Optional[float]:
    """Smallest positive root of the ``tau2`` equation, or ``None``.

    ``g(tau2)`` tends to a negative limit as ``tau2 -> inf``; a root exists
    when ``g(0) > 0``. The upper end is found by geometric expansion from
    ``scale``.
    """
    if not g(0.0) > 0:
        return None
    lo, hi = 0.0, scale
    for _ in range(400):
        if g(hi) < 0:
            return float(
                optimize.brentq(g, lo, hi, xtol=1e-300, rtol=4 * np.finfo(float).eps)
            )
        lo, hi = hi, hi * 4.0
    return None


def _fit_overdispersed(
    bx: _Arr,
    by: _Arr,
    sx: _Arr,
    sy: _Arr,
    loss: str,
    k: float,
    beta_start: float,
    niter: int,
    tol: float,
) -> Tuple[float, float, bool, bool]:
    """Solve the (R)APS equations by alternating ``tau2`` and ``beta`` steps.

    Returns ``(beta, tau2, converged, tau2_at_zero)``. After the alternating
    iterations meet ``tol`` the joint root is polished by Newton-type steps,
    so the reported root does not depend on the stopping rule.
    """
    if loss == "l2":
        psi = lambda t: t  # noqa: E731
        delta = 1.0
    else:
        rho = _rho(loss, k)
        psi = lambda t: rho(t, 1)  # noqa: E731
        delta = _loss_moments(loss, k)[0]

    tau_scale = float(np.min(sy**2)) * 1e-6
    beta_scale = 1e-3 * max(
        abs(beta_start), float(np.median(np.abs(sy / bx))) * 1e-3, 1e-12
    )
    beta, tau2 = float(beta_start), 0.0
    at_zero = False
    converged = False
    for _ in range(int(niter)):
        root = _tau2_root(
            lambda s: _tau2_score(bx, by, sx, sy, beta, s, psi, delta), tau_scale
        )
        at_zero = root is None
        tau2_new = 0.0 if root is None else root
        beta_new = _local_beta_root(
            lambda b: _beta_score(bx, by, sx, sy, b, tau2_new, psi), beta, beta_scale
        )
        change = abs(beta_new - beta) + abs(tau2_new - tau2)
        beta, tau2 = beta_new, tau2_new
        if change < tol:
            converged = True
            break

    if converged and not at_zero:

        def joint(p: np.ndarray) -> list:
            return [
                _beta_score(bx, by, sx, sy, p[0], p[1], psi),
                _tau2_score(bx, by, sx, sy, p[0], p[1], psi, delta),
            ]

        sol = optimize.root(joint, [beta, tau2], method="hybr", tol=1e-15)
        b_new, t_new = float(sol.x[0]), float(sol.x[1])
        start_res = np.max(np.abs(joint(np.array([beta, tau2]))))
        if (
            sol.success
            and t_new >= 0
            and abs(b_new - beta) <= 1e-4 * max(abs(beta), 1e-12) + tol
            and abs(t_new - tau2) <= 1e-3 * tau2 + tol
            and np.max(np.abs(joint(sol.x))) <= start_res
        ):
            beta, tau2 = b_new, t_new
    return beta, tau2, converged, at_zero


# --------------------------------------------------------------------------- #
#  Variances
# --------------------------------------------------------------------------- #


def _simple_se(bx: _Arr, by: _Arr, sx: _Arr, sy: _Arr, beta: float) -> float:
    """Theorem 3.3: ``sqrt(V1_hat) / V2_hat`` with the unbiased plug-ins."""
    var = sx**2 * beta**2 + sy**2
    v2 = np.sum(((bx**2 - sx**2) * sy**2 + (by**2 - sy**2) * sx**2) / var**2)
    v1 = v2 + np.sum(sx**2 * sy**2 / var**2)
    return float(np.sqrt(v1) / v2) if v1 > 0 and v2 > 0 else float("nan")


def _robust_vcov(
    bx: np.ndarray,
    by: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    beta: float,
    tau2: float,
    consts: Tuple[float, float, float, float],
) -> np.ndarray:
    """Sandwich covariance of ``(beta, tau2)``, Theorems 4.2 and 5.2.

    ``consts = (delta, c1, var_zpsi, c3)`` with ``delta = E[Z psi(Z)]``,
    ``c1 = E[psi(Z)^2]``, ``var_zpsi = Var(Z psi(Z))`` (twice the paper's
    ``c2``) and ``c3 = E[Z^2 psi'(Z)]``; ``(1, 1, 2, 1)`` for the L2 loss.
    ``gamma_j^2`` and ``Gamma_j^2`` are replaced by ``bx_j^2 - sx_j^2`` and
    ``by_j^2 - sy_j^2 - tau2`` (Section 4.3).
    """
    delta, c1, var_zpsi, c3 = consts
    beta = float(beta)
    tau2 = float(tau2)
    var = sx**2 * beta**2 + sy**2 + tau2
    gamma2 = bx**2 - sx**2
    big_gamma2 = by**2 - sy**2 - tau2
    w = 1.0 / var**2
    # Score variance (V1 of Theorem 4.2 scaled by c1 and c2).
    v1_11 = np.sum(w * ((gamma2 + sx**2) * (sy**2 + tau2) + big_gamma2 * sx**2))
    v1_22 = np.sum(w * 2.0 * sx**4)
    meat = np.diag([c1 * v1_11, 0.5 * var_zpsi * v1_22])
    # Expected Jacobian (V2 of Theorem 4.2 scaled by delta and (delta + c3)/2).
    v2_11 = np.sum(w * (gamma2 * (sy**2 + tau2) + big_gamma2 * sx**2))
    v2_12 = np.sum(w * sx**2 * beta)
    v2_22 = np.sum(w * sx**2)
    bread = np.array(
        [[delta * v2_11, delta * v2_12], [0.0, 0.5 * (delta + c3) * v2_22]]
    )
    inv = np.linalg.inv(bread)
    cov: np.ndarray = inv @ meat @ inv.T
    return cov


# --------------------------------------------------------------------------- #
#  Public entry point
# --------------------------------------------------------------------------- #


def mr_raps(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_exposure: np.ndarray,
    se_outcome: np.ndarray,
    *,
    loss: Optional[str] = None,
    over_dispersion: bool = True,
    tuning_c: Optional[float] = None,
    alpha: float = 0.05,
    pruning: bool = True,
    niter: int = 20,
    tol: float = _EPS_HALF,
    beta_init: Optional[float] = None,
    tau2_init: Optional[float] = None,
) -> MRRapsResult:
    """Robust Adjusted Profile Score MR (Zhao et al. 2020).

    Profile-likelihood MR that accounts for measurement error in the
    exposure associations (weak instruments), optionally for systematic
    pleiotropy (a random-effect variance ``tau2``) and, with a robust loss,
    for a small fraction of gross pleiotropic outliers.

    Parameters
    ----------
    beta_exposure, beta_outcome : array-like
        Variant-exposure and variant-outcome associations.
    se_exposure, se_outcome : array-like
        Their standard errors (strictly positive).
    loss : {'huber', 'tukey', 'l2'}, optional
        Loss in the profile score. ``None`` means ``'huber'``.
    over_dispersion : bool, default True
        Estimate the over-dispersion variance ``tau2`` (Sections 4-5).
        ``False`` is the profile-score estimator of Section 3 and needs
        ``loss='l2'``.
    tuning_c : float, optional
        Loss tuning constant ``k``; default 1.345 (Huber) or 4.685 (Tukey).
        Ignored for ``loss='l2'``.
    alpha : float, default 0.05
        ``1 - alpha`` confidence level.
    pruning : bool, default True
        Drop variants with ``se_exposure > 10 * median(se_exposure)``.
    niter : int, default 20
        Maximum number of alternating ``tau2`` / ``beta`` iterations.
    tol : float, default ``sqrt(machine eps)``
        Convergence threshold on the change in ``(beta, tau2)``.
    beta_init, tau2_init : float, optional
        Deprecated and ignored: the over-dispersed fit starts from the
        no-over-dispersion estimate and the robust fit from the L2 fit.

    Returns
    -------
    MRRapsResult

    Raises
    ------
    ValueError
        On mismatched lengths, fewer than 3 variants, non-positive standard
        errors, an unknown ``loss``, a non-positive ``tuning_c``, or a robust
        loss without over-dispersion.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> bx = rng.uniform(0.1, 0.4, 30)
    >>> sx = np.full(30, 0.02)
    >>> sy = np.full(30, 0.02)
    >>> by = 0.5 * bx + rng.normal(0, 0.02, 30)
    >>> r = sp.mr_raps(bx, by, sx, sy, loss="tukey")
    >>> bool(abs(r.estimate - 0.5) < 0.2)
    True

    Notes
    -----
    Clean-room implementation of the paper's estimating equations; the
    reference (R ``mr.raps`` 0.4.3, GPL-3) was used only as a black box.
    The (R)APS root is solved to machine precision, so it can differ from
    the reference in the digits that the reference's own stopping rule
    leaves undetermined (see ``docs/dev/cleanroom/mr_raps.md``).

    References
    ----------
    zhao2020statistical
    """
    if beta_init is not None or tau2_init is not None:
        warnings.warn(
            "mr_raps: beta_init / tau2_init are deprecated since 1.28.0 and "
            "ignored; the fit starts from the profile-score estimate.",
            DeprecationWarning,
            stacklevel=2,
        )
    if loss is None:
        if tuning_c is not None:
            warnings.warn(
                "mr_raps: tuning_c given without loss; the default loss is "
                "'huber' since 1.28.0, so tuning_c is the Huber constant. Pass "
                "loss='tukey' to keep the Tukey biweight.",
                FutureWarning,
                stacklevel=2,
            )
        loss = "huber"
    loss = str(loss).lower()
    if loss not in ("huber", "tukey", "l2"):
        raise ValueError(f"loss must be 'huber', 'tukey' or 'l2'; got {loss!r}")
    if tuning_c is not None and not (np.isfinite(tuning_c) and tuning_c > 0):
        raise ValueError(f"tuning_c must be a positive finite number; got {tuning_c!r}")
    if not over_dispersion and loss != "l2":
        raise ValueError(
            "over_dispersion=False is only implemented for loss='l2' (the "
            "profile-score estimator of Zhao et al. 2020, Section 3)"
        )
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}")
    k = float("nan") if loss == "l2" else float(tuning_c or _DEFAULT_TUNING[loss])

    bx, by, sx, sy = as_float_arrays(
        beta_exposure, beta_outcome, se_exposure, se_outcome
    )
    if not all(np.all(np.isfinite(a)) for a in (bx, by, sx, sy)):
        raise ValueError("mr_raps: inputs must be finite")
    if np.any(sx <= 0) or np.any(sy <= 0):
        raise ValueError(
            "mr_raps: se_exposure and se_outcome must be strictly positive"
        )

    n_pruned = 0
    if pruning:
        keep = sx <= _PRUNE_RATIO * np.median(sx)
        n_pruned = int(np.sum(~keep))
        if n_pruned:
            warnings.warn(
                f"mr_raps: pruning {n_pruned} variant(s) whose se_exposure exceeds "
                f"{_PRUNE_RATIO:g} x the median (pruning=False keeps them).",
                UserWarning,
                stacklevel=2,
            )
            bx, by, sx, sy = bx[keep], by[keep], sx[keep], sy[keep]
            if bx.size < 3:
                raise ValueError("mr_raps: fewer than 3 variants left after pruning")

    beta_simple = _global_profile_max(bx, by, sx, sy)
    if not over_dispersion:
        beta, tau2, converged = beta_simple, 0.0, True
        se = _simple_se(bx, by, sx, sy, beta)
        tau2_se = float("nan")
        objective = _objective(bx, by, sx, sy, beta, 0.0, lambda t: 0.5 * t**2)
    else:
        beta, tau2, converged, at_zero = _fit_overdispersed(
            bx, by, sx, sy, "l2", k, beta_simple, niter, tol
        )
        if loss != "l2":
            beta_l2 = beta
            beta, tau2, converged_r, at_zero = _fit_overdispersed(
                bx, by, sx, sy, loss, k, beta_l2, niter, tol
            )
            converged = converged and converged_r
        if at_zero:
            warnings.warn(
                "mr_raps: the tau2 estimating equation has no positive root at "
                "the estimate; tau2 is set to 0 (no detectable over-dispersion).",
                UserWarning,
                stacklevel=2,
            )
        if not converged:
            warnings.warn(
                f"mr_raps: the estimating equations did not converge in "
                f"niter={niter} iterations; consider increasing niter.",
                UserWarning,
                stacklevel=2,
            )
        V = _robust_vcov(bx, by, sx, sy, beta, tau2, _loss_moments(loss, k))
        se = float(np.sqrt(V[0, 0])) if V[0, 0] > 0 else float("nan")
        tau2_se = float(np.sqrt(V[1, 1])) if V[1, 1] > 0 else float("nan")
        rho = (lambda t: 0.5 * t**2) if loss == "l2" else _rho(loss, k)
        objective = _objective(bx, by, sx, sy, beta, tau2, rho)

    z = stats.norm.ppf(1 - alpha / 2)
    p_value = (
        float(2 * stats.norm.sf(abs(beta / se))) if np.isfinite(se) else float("nan")
    )
    return MRRapsResult(
        estimate=float(beta),
        se=float(se),
        ci_lower=float(beta - z * se),
        ci_upper=float(beta + z * se),
        p_value=p_value,
        tau2=float(tau2),
        loglik_robust=float(objective),
        converged=bool(converged),
        tuning_c=k,
        n_snps=int(bx.size),
        tau2_se=float(tau2_se),
        loss=loss,
        over_dispersion=bool(over_dispersion),
        n_pruned=n_pruned,
    )
