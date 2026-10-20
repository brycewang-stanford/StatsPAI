"""MR-cML: constrained maximum-likelihood Mendelian randomization.

Implements cML-BIC and cML-MA-BIC of Xue, Shen & Pan (2021) for
two-sample summary data. Each variant ``i`` contributes the Gaussian
log-likelihood (their Equation 2)

.. math::

    L = -\\frac12 \\sum_i \\Big[ \\frac{(\\hat\\beta_{Xi} - b_i)^2}{\\sigma_{Xi}^2}
        + \\frac{(\\hat\\beta_{Yi} - \\theta b_i - r_i)^2}{\\sigma_{Yi}^2} \\Big],

maximised subject to exactly ``K`` non-zero pleiotropic effects ``r_i``
(Equation 3). ``K`` is chosen by ``BIC(K) = -2 L + K log n`` (Equation 4)
or the per-``K`` estimates are averaged with BIC weights (Equation 5).

Clean-room implementation written from the paper; the reference
(R ``MendelianRandomization`` 0.10.0, ``mr_cML``, GPL-2 | GPL-3) was used
only as a black box to fix the conventions listed in
``docs/dev/cleanroom/mr_cml.md``. Data perturbation (cML-*-DP) is not
implemented.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..._result_serialize import ResultProtocolMixin
from ._common import as_float_arrays, harmonize_signs

__all__ = ["MRcMLResult", "mr_cml"]


@dataclass
class MRcMLResult(ResultProtocolMixin):
    """Output of :func:`mr_cml`.

    Attributes
    ----------
    estimate : float
        cML-BIC estimate (or the BIC model average when
        ``model_average=True``).
    se : float
        Standard error from the Fisher information at the selected K
        (or the model-averaged SE).
    ci_lower, ci_upper, p_value : float
    K_selected : int
        Number of variants flagged as invalid (pleiotropic) by BIC.
    invalid_snps : np.ndarray
        Boolean mask of the K_selected variants flagged as invalid.
    path : pd.DataFrame
        One row per K in [0, K_max] with columns
        ``[K, estimate, se, loglik, bic, converged]``.
    loglik : float
        Maximised constrained Gaussian log-likelihood at K_selected
        (Equation 2 including its normalising constant).
    n_snps : int
    n_samples : int or None
        GWAS sample size used in the BIC penalty (``None`` if not given, in
        which case the number of variants stood in and a warning was issued).
    model_average : bool
        Whether ``estimate`` / ``se`` are BIC model-averaged over K.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(8)
    >>> bx = rng.uniform(0.1, 0.5, 25)
    >>> sx = rng.uniform(0.02, 0.05, 25)
    >>> sy = rng.uniform(0.02, 0.05, 25)
    >>> by = 0.3 * bx + rng.normal(0, sy)
    >>> res = sp.mr_cml(bx, by, sx, sy, n=50_000)
    >>> type(res).__name__
    'MRcMLResult'
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
    K_selected: int
    invalid_snps: np.ndarray
    path: pd.DataFrame
    loglik: float
    n_snps: int
    n_samples: Optional[int] = None
    model_average: bool = False

    def summary(self) -> str:
        label = "MR-cML-MA-BIC" if self.model_average else "MR-cML-BIC"
        n_txt = (
            str(self.n_samples)
            if self.n_samples is not None
            else f"not given (BIC used log({self.n_snps}))"
        )
        lines = [
            f"{label} (Xue, Shen & Pan 2021)",
            "=" * 60,
            f"  n SNPs          : {self.n_snps}",
            f"  GWAS sample size: {n_txt}",
            f"  K selected (BIC): {self.K_selected}",
            f"  Estimate        : {self.estimate:.6g}",
            f"  SE              : {self.se:.6g}",
            f"  CI              : [{self.ci_lower:.6g}, {self.ci_upper:.6g}]",
            f"  p-value         : {self.p_value:.4g}",
            f"  log-likelihood  : {self.loglik:.6g}",
        ]
        if self.K_selected:
            idx = ", ".join(str(i) for i in np.flatnonzero(self.invalid_snps))
            lines.append(f"  Invalid SNPs (0-based): {idx}")
        lines.append("")
        lines.append("  K path (* = BIC-selected):")
        for row in self.path.itertuples(index=False):
            mark = "*" if int(row.K) == self.K_selected else " "
            lines.append(
                f"  {mark} K={int(row.K):3d}  est={row.estimate:+.4f}  "
                f"SE={row.se:.4f}  logL={row.loglik:.3f}  BIC={row.bic:.3f}"
            )
        return "\n".join(lines)


@dataclass
class _KFit:
    """Constrained MLE at one fixed K."""

    theta: float
    se: float
    loglik: float
    invalid: np.ndarray
    converged: bool


def _fit_fixed_k(
    bx: np.ndarray,
    by: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    K: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> _KFit:
    """Coordinate descent for Equation 3 at a fixed number K of invalid IVs.

    Steps 1-3 of the paper's Computation section, started from
    ``theta = 0`` and ``b_i = 0``:

    1. ``r_i = by_i - theta * b_i`` for the K largest
       ``(by_i - theta * b_i)^2 / vy_i``, zero otherwise;
    2. ``b_i`` from Equation 6;
    3. ``theta`` from Equation 7.

    Iteration stops when ``|theta_new - theta_old| < tol`` or after
    ``max_iter`` rounds; step 1 is then applied once more at the final
    theta before the likelihood is evaluated. The SE is
    ``sqrt([I^-1]_11)`` with ``I`` the observed information of Equation 8
    over theta and the valid ``b_i``.
    """
    m = bx.size

    def pleiotropy_step(theta: float, b: np.ndarray) -> np.ndarray:
        # Step 1: keep the K largest standardized residuals as r_i.
        resid = by - theta * b
        r = np.zeros(m)
        if K > 0:
            top = np.argsort(-(resid**2) / vy, kind="stable")[:K]
            r[top] = resid[top]
        return r

    theta = 0.0
    b = np.zeros(m)
    converged = False
    for _ in range(max_iter):
        r = pleiotropy_step(theta, b)
        # Step 2: Equation 6.
        b = (bx / vx + theta * (by - r) / vy) / (1.0 / vx + theta**2 / vy)
        # Step 3: Equation 7.
        theta_new = float(np.sum((by - r) * b / vy) / np.sum(b**2 / vy))
        step = abs(theta_new - theta)
        theta = theta_new
        if step < tol:
            converged = True
            break
    # Step 1 at the final theta: the invalid variants then fit exactly and
    # the likelihood is evaluated at a point that is coordinate-optimal in r.
    r = pleiotropy_step(theta, b)
    invalid = r != 0.0
    # Equation 2 plus the Gaussian normalising constant, so ``loglik`` is the
    # full log-density of the 2m summary statistics (BIC differences across
    # K are unaffected by the constant).
    kernel = np.sum((bx - b) ** 2 / vx + (by - theta * b - r) ** 2 / vy)
    loglik = -0.5 * float(kernel + np.sum(np.log(4.0 * np.pi**2 * vx * vy)))
    valid = ~invalid
    # Equation 8: information for (theta, b_valid); b_valid block is diagonal.
    i_tt = np.sum(b[valid] ** 2 / vy[valid])
    i_tb = (2.0 * theta * b[valid] - by[valid]) / vy[valid]
    i_bb = 1.0 / vx[valid] + theta**2 / vy[valid]
    schur = i_tt - np.sum(i_tb**2 / i_bb)
    se = float(np.sqrt(1.0 / schur)) if schur > 0 else float("nan")
    return _KFit(theta, se, loglik, invalid, converged)


def mr_cml(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_exposure: np.ndarray,
    se_outcome: np.ndarray,
    *,
    K_max: Optional[int] = None,
    alpha: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-7,
    n: Optional[int] = None,
    model_average: bool = False,
) -> MRcMLResult:
    """Constrained maximum-likelihood MR (cML-BIC / cML-MA-BIC).

    Fits the constrained MLE of Xue, Shen & Pan (2021) for every number
    of invalid instruments ``K = 0, ..., K_max``, selects ``K`` by
    ``BIC(K) = -2 L(K) + K log n`` and reports either the selected fit
    (cML-BIC) or the BIC-weighted average over ``K`` (cML-MA-BIC,
    ``model_average=True``). Robust to both correlated and uncorrelated
    pleiotropy under the plurality-valid assumption.

    Parameters
    ----------
    beta_exposure, beta_outcome : array-like
        Per-variant associations with the exposure and the outcome.
    se_exposure, se_outcome : array-like
        Their standard errors (strictly positive).
    K_max : int, optional
        Largest number of invalid instruments considered. Default
        ``n_snps - 2`` (the paper's candidate set ``{0, ..., m - 2}``,
        which keeps at least two valid instruments); must lie in
        ``[0, n_snps - 1]``.
    alpha : float, default 0.05
        Two-sided level for the normal confidence interval.
    max_iter : int, default 100
        Maximum coordinate-descent rounds per K.
    tol : float, default 1e-7
        Stop when the change in theta between rounds is below ``tol``.
    n : int, optional
        GWAS sample size for the BIC penalty; the paper recommends
        ``min(n_exposure, n_outcome)``. When omitted the number of
        variants is used and a warning is issued.
    model_average : bool, default False
        Report the BIC model average (Equation 5) instead of the
        BIC-selected fit.

    Returns
    -------
    MRcMLResult

    Notes
    -----
    Variants with negative exposure associations are sign-flipped first
    (the likelihood is invariant to flipping both associations of a
    variant). The data-perturbation variants (cML-*-DP) and random
    restarts are not implemented: the fit uses the paper's default start
    ``theta = 0, b = 0``.

    Clean-room implementation; the reference (MendelianRandomization
    0.10.0, GPL-2 | GPL-3) was used only as a black box.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> bx = rng.uniform(0.1, 0.5, 30)
    >>> sx = np.full(30, 0.02)
    >>> sy = np.full(30, 0.02)
    >>> by = 0.4 * bx + rng.normal(0, sy)
    >>> r = sp.mr_cml(bx, by, sx, sy, n=50_000)
    >>> r.K_selected <= 2
    True

    References
    ----------
    xue2021constrained
    """
    bx, by, sx, sy = as_float_arrays(
        beta_exposure, beta_outcome, se_exposure, se_outcome
    )
    m = bx.size
    for name, arr in (
        ("beta_exposure", bx),
        ("beta_outcome", by),
        ("se_exposure", sx),
        ("se_outcome", sy),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
    if np.any(sx <= 0) or np.any(sy <= 0):
        raise ValueError("se_exposure and se_outcome must be strictly positive")
    if K_max is None:
        K_max = m - 2
    if not (0 <= int(K_max) <= m - 1):
        raise ValueError(f"K_max must be in [0, {m - 1}] (n_snps - 1); got {K_max}")
    K_max = int(K_max)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if n is None:
        warnings.warn(
            "mr_cml: GWAS sample size `n` not given; the BIC penalty uses "
            f"log(n_snps) = log({m}) instead. Pass n = min(n_exposure, "
            "n_outcome) as recommended by Xue, Shen & Pan (2021).",
            UserWarning,
            stacklevel=2,
        )
        log_n = float(np.log(m))
    else:
        if not np.isfinite(n) or n <= 1:
            raise ValueError(f"n must be a sample size > 1; got {n}")
        log_n = float(np.log(n))

    bx, by = harmonize_signs(bx, by)
    vx, vy = sx**2, sy**2

    fits = [
        _fit_fixed_k(bx, by, vx, vy, K, max_iter=max_iter, tol=tol)
        for K in range(K_max + 1)
    ]
    est = np.array([f.theta for f in fits])
    ses = np.array([f.se for f in fits])
    ll = np.array([f.loglik for f in fits])
    bic = -2.0 * ll + log_n * np.arange(K_max + 1)
    path = pd.DataFrame(
        {
            "K": np.arange(K_max + 1),
            "estimate": est,
            "se": ses,
            "loglik": ll,
            "bic": bic,
            "converged": [f.converged for f in fits],
        }
    )
    k_hat = int(np.argmin(bic))
    w = np.exp(-(bic - bic.min()) / 2.0)
    w = w / w.sum()
    # Non-convergence only matters where it feeds the reported number.
    matters = w >= 1e-3 if model_average else np.arange(K_max + 1) == k_hat
    not_conv = [K for K, f in enumerate(fits) if matters[K] and not f.converged]
    if not_conv:
        warnings.warn(
            f"mr_cml: coordinate descent did not reach tol={tol} within "
            f"max_iter={max_iter} for K in {not_conv}; the reported estimate "
            "uses the last iterate (see path['converged']).",
            RuntimeWarning,
            stacklevel=2,
        )
    if model_average:
        estimate = float(np.sum(w * est))
        # Buckland et al. (1997) form used by the paper's Equation 5: the
        # weights multiply the per-K root mean squares.
        se = float(np.sum(w * np.sqrt(ses**2 + (est - estimate) ** 2)))
    else:
        estimate = float(est[k_hat])
        se = float(ses[k_hat])
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    p_value = float(2.0 * stats.norm.sf(abs(estimate / se))) if se > 0 else float("nan")
    return MRcMLResult(
        estimate=estimate,
        se=se,
        ci_lower=estimate - z * se,
        ci_upper=estimate + z * se,
        p_value=p_value,
        K_selected=k_hat,
        invalid_snps=fits[k_hat].invalid.copy(),
        path=path,
        loglik=float(ll[k_hat]),
        n_snps=m,
        n_samples=None if n is None else int(n),
        model_average=bool(model_average),
    )
