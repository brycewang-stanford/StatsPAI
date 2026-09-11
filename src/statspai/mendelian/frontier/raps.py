"""MR-RAPS: Robust Adjusted Profile Score for two-sample summary-data MR.

Reference
---------
Zhao, Q., Wang, J., Hemani, G., Bowden, J. & Small, D.S. (2020).
"Statistical inference in two-sample summary-data Mendelian
randomization using robust adjusted profile score."
*The Annals of Statistics*, 48(3), 1742-1769.

A port of the authors' R package ``mr.raps`` (0.4.3, github
qingyuanzhao/mr.raps), variant for variant:

* ``over_dispersion=False, loss="l2"`` -- ``mr.raps.simple``;
* ``over_dispersion=True,  loss="l2"`` -- ``mr.raps.overdispersed``;
* ``over_dispersion=True,  loss="huber" | "tukey"`` --
  ``mr.raps.overdispersed.robust`` (the package's ``mr.raps()`` default is
  Huber with k = 1.345).

Model
-----
.. math::

   \\beta_{y,i} = \\beta \\beta_{x,i} + u_i,
   \\qquad u_i \\sim \\mathcal{N}(
   0, s_{y,i}^2 + \\beta^2 s_{x,i}^2 + \\tau^2).

The robust variant replaces the Gaussian profile log-likelihood with
:math:`-\\tfrac12\\sum \\rho(t_i)`, :math:`t_i` the standardised residual,
estimates :math:`\\tau^2` from the moment equation
:math:`\\sum s_{x,i}^2 (t_i\\rho'(t_i) - \\delta) / (\\tau^2 + s_{y,i}^2 +
s_{x,i}^2\\beta^2) = 0` with :math:`\\delta = E[Z\\rho'(Z)]`, and reports the
joint :math:`(\\beta, \\tau^2)` sandwich variance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import integrate, optimize, stats

from ..._result_serialize import ResultProtocolMixin
from ._common import as_float_arrays

_EPS_HALF = float(np.finfo(float).eps ** 0.5)


@dataclass
class MRRapsResult(ResultProtocolMixin):
    """Output of :func:`mr_raps`.

    Attributes
    ----------
    estimate : float
    se : float
        From the reference's asymptotic variance: ``sqrt(score_var / I^2)``
        for the simple model, the ``(beta, tau2)`` sandwich otherwise.
    ci_lower, ci_upper : float
    p_value : float
    tau2 : float
        Over-dispersion (0 for the simple model).
    tau2_se : float
    loglik_robust : float
        Profile objective at the estimate (robust or Gaussian).
    converged : bool
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
        ci = f"[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]"
        conv = "converged" if self.converged else "DID NOT CONVERGE"
        variant = (
            "simple"
            if not self.over_dispersion
            else f"over-dispersed, {self.loss}"
            + ("" if self.loss == "l2" else f" (k = {self.tuning_c:g})")
        )
        return (
            "MR-RAPS (Robust Adjusted Profile Score)\n" + "=" * 62 + "\n"
            f"  variant       : {variant}\n"
            f"  n SNPs        : {self.n_snps}"
            + (f"  ({self.n_pruned} pruned)" if self.n_pruned else "")
            + "\n"
            f"  causal β      : {self.estimate:+.4f}   SE = {self.se:.4f}\n"
            f"  95% CI        : {ci}\n"
            f"  p-value       : {self.p_value:.4g}\n"
            f"  pleiotropy τ² : {self.tau2:.4g}\n"
            f"  objective     : {self.loglik_robust:.3f}  ({conv})"
        )


# ── loss functions, exactly as mr.raps:::rho.huber / rho.tukey ─────────────


def _rho(loss: str, k: float) -> Callable[..., np.ndarray]:
    if loss == "huber":

        def rho(r, deriv=0):
            r = np.asarray(r, dtype=float)
            a = np.abs(r) <= k
            if deriv == 0:
                return np.where(a, r**2 / 2, k * (np.abs(r) - k / 2))
            if deriv == 1:
                return np.where(a, r, k * np.sign(r))
            return np.where(a, 1.0, 0.0)

        return rho

    def rho(r, deriv=0):  # Tukey, with the package's own scaling
        r = np.asarray(r, dtype=float)
        if deriv == 0:
            return np.minimum(1 - (1 - (r / k) ** 2) ** 3, 1.0)
        if deriv == 1:
            return r * (1 - (r / k) ** 2) ** 2 * (np.abs(r) <= k)
        t = (r / k) ** 2
        return np.where(t < 1, (1 - t) * (1 - 5 * t), 0.0)

    return rho


def _gauss_moment(f: Callable[[float], float]) -> float:
    return float(
        integrate.quad(lambda x: float(f(x)) * stats.norm.pdf(x), -np.inf, np.inf)[0]
    )


# ── the three estimators ────────────────────────────────────────────────────


def _raps_simple(bx, by, sx, sy) -> Tuple[float, float]:
    def negll(b):
        return 0.5 * np.sum((by - bx * b) ** 2 / (sy**2 + sx**2 * b**2))

    bound = float(np.quantile(np.abs(by / bx), 0.95)) * 2
    beta = _bounded_argmin(negll, bound)
    while abs(beta) > 0.95 * bound:
        bound *= 2
        beta = _bounded_argmin(negll, bound)
    d = (sy**2 + beta**2 * sx**2) ** 2
    score_var = np.sum(
        ((bx**2 - sx**2) * sy**2 + (by**2 - sy**2) * sx**2 + sx**2 * sy**2) / d
    )
    info = np.sum(((bx**2 - sx**2) * sy**2 + (by**2 - sy**2) * sx**2) / d)
    return float(beta), float(np.sqrt(score_var / info**2))


def _bounded_argmin(f, bound: float) -> float:
    """Golden-section / Brent on [-bound, bound] to R optimize()'s tolerance."""
    res = optimize.minimize_scalar(
        f, bounds=(-bound, bound), method="bounded", options={"xatol": _EPS_HALF}
    )
    return float(res.x)


def _raps_overdispersed(bx, by, sx, sy, niter: int, tol: float):
    beta, _ = _raps_simple(bx, by, sx, sy)
    tau2 = 0.0
    bound_beta = float(np.quantile(np.abs(by / bx), 0.95)) * 10
    bound_tau2 = float(np.quantile(sy**2, 0.95)) * 100

    def negll_fixbeta(t2, b):
        v = t2 + sy**2 + sx**2 * b**2
        return 0.5 * np.sum(sx**2 * np.log(v)) + 0.5 * np.sum(
            sx**2 * (by - bx * b) ** 2 / v
        )

    def negll_fixtau(b, t2):
        return 0.5 * np.sum((by - bx * b) ** 2 / (t2 + sy**2 + sx**2 * b**2))

    converged = False
    for _ in range(niter):
        b_old, t_old = beta, tau2
        tau2 = float(
            optimize.minimize_scalar(
                lambda t2: negll_fixbeta(t2, beta),
                bounds=(0.0, bound_tau2),
                method="bounded",
                options={"xatol": _EPS_HALF},
            ).x
        )
        beta = _bounded_argmin(lambda b: negll_fixtau(b, tau2), bound_beta)
        ext = 0
        while abs(beta) > 0.95 * bound_beta and ext <= niter:
            ext += 1
            bound_beta *= 2
            beta = _bounded_argmin(lambda b: negll_fixtau(b, tau2), bound_beta)
        if (
            abs(b_old - beta) / abs(beta + 1e-10)
            + abs(t_old - tau2) / abs(tau2 + 1e-10)
            <= tol
        ):
            converged = True
            break
    v = tau2 + sy**2 + sx**2 * beta**2
    score_var = np.diag(
        [
            np.sum(
                (
                    (bx**2 - sx**2) * (tau2 + sy**2)
                    + (by**2 - tau2 - sy**2) * sx**2
                    + sx**2 * (tau2 + sy**2)
                )
                / v**2
            ),
            np.sum(2 * sx**4 / v**2),
        ]
    )
    info = np.array(
        [
            [
                -np.sum(
                    ((bx**2 - sx**2) * (tau2 + sy**2) + (by**2 - tau2 - sy**2) * sx**2)
                    / v**2
                ),
                -np.sum(sx**2 * beta / v**2),
            ],
            [0.0, -np.sum(sx**2 / v**2)],
        ]
    )
    return beta, tau2, _sandwich(info, score_var), negll_fixtau(beta, tau2), converged


def _sandwich(info: np.ndarray, score_var: np.ndarray) -> np.ndarray:
    # R builds I column-wise: matrix(c(a, 0, b, d), 2, 2) = [[a, b], [0, d]].
    Ii = np.linalg.inv(info)
    return Ii @ score_var @ Ii.T


def _robust_vcov(bx, by, sx, sy, beta, tau2, consts) -> np.ndarray:
    """Sandwich covariance of (beta, tau2) for a robust loss.

    ``consts`` = (delta, c1, c2, c3), the loss's Gaussian moments. Split out
    so the formula can be checked at the reference's own estimates and
    constants, independently of how tightly either side solves for them.
    """
    delta, c1, c2, c3 = consts
    v = tau2 + sy**2 + sx**2 * beta**2
    score_var = np.diag(
        [
            c1
            * np.sum(
                (
                    (bx**2 - sx**2) * (tau2 + sy**2)
                    + (by**2 - tau2 - sy**2) * sx**2
                    + sx**2 * (tau2 + sy**2)
                )
                / v**2
            ),
            (c2 / 2) * np.sum(2 * sx**4 / v**2),
        ]
    )
    info = np.array(
        [
            [
                -delta
                * np.sum(
                    ((bx**2 - sx**2) * (tau2 + sy**2) + (by**2 - tau2 - sy**2) * sx**2)
                    / v**2
                ),
                -delta * np.sum(sx**2 * beta / v**2),
            ],
            [0.0, -(delta + c3) / 2 * np.sum(sx**2 / v**2)],
        ]
    )
    return _sandwich(info, score_var)


def _raps_robust(bx, by, sx, sy, loss: str, k: float, niter: int, tol: float):
    rho = _rho(loss, k)
    delta = _gauss_moment(lambda x: x * rho(x, 1))
    c1 = _gauss_moment(lambda x: rho(x, 1) ** 2)
    c2 = _gauss_moment(lambda x: x**2 * rho(x, 1) ** 2) - delta**2
    c3 = _gauss_moment(lambda x: x**2 * rho(x, 2))

    def neg_robust(b, t2):
        return 0.5 * np.sum(
            rho((by - bx * b) / np.sqrt(np.maximum(0.0, t2 + sy**2 + sx**2 * b**2)))
        )

    def tau_eq(t2, b):
        v = t2 + sy**2 + sx**2 * b**2
        t = (by - b * bx) / np.sqrt(np.maximum(0.0, v))
        return float(np.sum(sx**2 * (t * rho(t, 1) - delta) / v))

    beta, tau2, _, _, _ = _raps_overdispersed(bx, by, sx, sy, niter, tol)
    bound_beta = float(np.quantile(np.abs(by / bx), 0.95)) * 10
    bound_tau2 = float(np.quantile(sy**2, 0.95)) * 100

    converged = False
    for _ in range(niter):
        b_old, t_old = beta, tau2
        tau2 = _root_tau2(lambda t2: tau_eq(t2, beta), bound_tau2)
        if tau2 > bound_tau2 * 0.95:
            warnings.warn(
                "mr_raps: estimated over-dispersion seems abnormally large.",
                UserWarning,
                stacklevel=3,
            )
        res = optimize.minimize(
            lambda b: neg_robust(b[0], tau2),
            x0=[beta],
            method="L-BFGS-B",
            bounds=[(-bound_beta, bound_beta)],
            options={"ftol": 1e-15, "gtol": 1e-12},
        )
        beta = float(res.x[0])
        ext = 0
        while abs(beta) > 0.95 * bound_beta and ext <= niter:
            ext += 1
            bound_beta *= 2
            beta = _bounded_argmin(lambda b: neg_robust(b, tau2), bound_beta)
        if (
            abs(b_old - beta) / abs(beta + 1e-10)
            + abs(t_old - tau2) / abs(tau2 + 1e-10)
            <= tol
        ):
            converged = True
            break
    vcov = _robust_vcov(bx, by, sx, sy, beta, tau2, (delta, c1, c2, c3))
    return beta, tau2, vcov, neg_robust(beta, tau2), converged


def _root_tau2(f, bound: float) -> float:
    """uniroot on [0, bound] with extendInt = 'yes', floored at 0."""
    lo, hi = 0.0, bound
    flo, fhi = f(lo), f(hi)
    n = 0
    while flo * fhi > 0 and n < 60:
        hi *= 2
        fhi = f(hi)
        n += 1
    if flo * fhi > 0:
        warnings.warn(
            "mr_raps: did not find a solution for tau2; using 0.",
            UserWarning,
            stacklevel=3,
        )
        return 0.0
    root = optimize.brentq(f, lo, hi, xtol=bound * np.finfo(float).eps ** 0.25 * 1e-6)
    return max(float(root), 0.0)


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

    Profile-likelihood MR that corrects for weak instruments through the
    :math:`\\beta^2 s_x^2` term, optionally allows balanced pleiotropy as
    over-dispersion :math:`\\tau^2`, and optionally replaces the Gaussian
    loss with a robust one. A port of the authors' ``mr.raps`` package, which
    it matches variant for variant.

    Parameters
    ----------
    beta_exposure, beta_outcome : ndarray
    se_exposure, se_outcome : ndarray
    loss : {"huber", "tukey", "l2"}, optional
        ``"huber"`` / ``"tukey"`` give ``mr.raps.overdispersed.robust``;
        ``"l2"`` gives ``mr.raps.overdispersed`` (or ``mr.raps.simple`` when
        ``over_dispersion=False``). ``None`` means ``"huber"``, the
        package's ``mr.raps()`` default.
    over_dispersion : bool, default True
        ``False`` is only defined with ``loss="l2"``.
    tuning_c : float, optional
        Loss constant; defaults to 1.345 (Huber) or 4.685 (Tukey). Before
        1.28.0 this was always the Tukey constant, so passing it without
        ``loss`` warns: the loss it now applies to is Huber.
    beta_init, tau2_init
        Deprecated and ignored. The robust fit starts from the L2
        over-dispersed estimate, as ``mr.raps`` does; they will be removed
        in 1.29.
    alpha : float, default 0.05
    pruning : bool, default True
        Drop variants with ``se_exposure > 10 * median(se_exposure)``, as the
        reference does by default. A warning names how many.
    niter, tol
        Alternating-update controls, the reference's defaults.

    Returns
    -------
    :class:`MRRapsResult`

    .. versionchanged:: 1.28.0
       Rewritten as a port of ``mr.raps``. The previous implementation was a
       different estimator under this name: it minimised a Tukey loss jointly
       over ``(beta, log tau2)`` instead of solving the reference's
       ``tau2`` estimating equation, and its standard error was a
       one-dimensional sandwich that treated ``tau2`` as known. On
       ``MendelianRandomization``'s LDL-C / CHD example that put ``tau2``
       3.5x below the reference, the estimate 1.4% away and the SE 47% too
       small. The default loss also changes from Tukey (4.685) to Huber
       (1.345), the package default; pass ``loss="tukey"`` for the old
       choice of loss.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(8)
    >>> bx = rng.uniform(0.1, 0.5, 25)   # SNP-exposure associations
    >>> sx = rng.uniform(0.02, 0.05, 25)
    >>> sy = rng.uniform(0.02, 0.05, 25)
    >>> by = 0.3 * bx + rng.normal(0, sy)  # true causal beta = 0.3
    >>> res = sp.mr_raps(bx, by, sx, sy)
    >>> res.n_snps
    25
    >>> bool(np.isfinite(res.estimate))
    True
    """
    if beta_init is not None or tau2_init is not None:
        warnings.warn(
            "mr_raps: beta_init / tau2_init are ignored since 1.28.0 -- the fit "
            "starts from the L2 over-dispersed estimate, as mr.raps does -- and "
            "will be removed in 1.29.",
            DeprecationWarning,
            stacklevel=2,
        )
    if loss is None:
        loss = "huber"
        if tuning_c is not None and tuning_c > 0:
            warnings.warn(
                f"mr_raps: tuning_c={tuning_c} now applies to the default Huber "
                "loss; before 1.28.0 the only loss was Tukey. Pass loss='tukey' "
                "to keep the old loss.",
                FutureWarning,
                stacklevel=2,
            )
    if loss not in ("huber", "tukey", "l2"):
        raise ValueError(f"loss must be 'huber', 'tukey' or 'l2', got {loss!r}")
    if not over_dispersion and loss != "l2":
        raise ValueError(
            "over_dispersion=False is only defined for loss='l2' (mr.raps.simple)"
        )
    if tuning_c is None:
        tuning_c = {"huber": 1.345, "tukey": 4.685}.get(loss, float("nan"))
    elif not tuning_c > 0:
        raise ValueError(f"tuning_c must be > 0; got {tuning_c}")

    bx, by, sx, sy = as_float_arrays(
        beta_exposure, beta_outcome, se_exposure, se_outcome
    )
    n_pruned = 0
    if pruning:
        drop = sx > 10 * np.median(sx)
        n_pruned = int(drop.sum())
        if n_pruned:
            warnings.warn(
                f"mr_raps: pruning {n_pruned} variant(s) with extraordinarily large "
                "se_exposure (> 10 x median), as mr.raps does by default.",
                UserWarning,
                stacklevel=2,
            )
            keep = ~drop
            bx, by, sx, sy = bx[keep], by[keep], sx[keep], sy[keep]

    if not over_dispersion:
        beta, se = _raps_simple(bx, by, sx, sy)
        tau2, tau2_se, converged = 0.0, float("nan"), True
        obj = -0.5 * float(np.sum((by - bx * beta) ** 2 / (sy**2 + sx**2 * beta**2)))
    else:
        if loss == "l2":
            beta, tau2, V, negobj, converged = _raps_overdispersed(
                bx, by, sx, sy, niter, tol
            )
        else:
            beta, tau2, V, negobj, converged = _raps_robust(
                bx, by, sx, sy, loss, tuning_c, niter, tol
            )
        se = float(np.sqrt(V[0, 0])) if V[0, 0] > 0 else float("nan")
        tau2_se = float(np.sqrt(V[1, 1])) if V[1, 1] > 0 else float("nan")
        obj = -float(negobj)
        if not converged:
            warnings.warn(
                "mr_raps: did not converge when solving the estimating equations; "
                "consider increasing niter.",
                UserWarning,
                stacklevel=2,
            )

    z_crit = stats.norm.ppf(1 - alpha / 2)
    if np.isfinite(se) and se > 0:
        p_value = float(min(1.0, 2.0 * stats.norm.sf(abs(beta) / se)))
        lo, hi = beta - z_crit * se, beta + z_crit * se
    else:
        p_value = lo = hi = float("nan")

    return MRRapsResult(
        estimate=float(beta),
        se=float(se),
        ci_lower=float(lo),
        ci_upper=float(hi),
        p_value=p_value,
        tau2=float(tau2),
        loglik_robust=obj,
        converged=bool(converged),
        tuning_c=float(tuning_c),
        n_snps=len(bx),
        tau2_se=float(tau2_se),
        loss=loss,
        over_dispersion=bool(over_dispersion),
        n_pruned=n_pruned,
    )
