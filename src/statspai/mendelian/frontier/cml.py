"""MR-cML-BIC: constrained maximum-likelihood MR with L0-sparse pleiotropy.

Reference
---------
Xue, H., Shen, X. & Pan, W. (2021).
"Constrained maximum likelihood-based Mendelian randomization
robust to both correlated and uncorrelated pleiotropic effects."
*AJHG*, 108(7), 1251-1269.
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
        BIC-selected MR-cML-BIC point estimate.
    se : float
        Profile-likelihood SE at the selected K.
    ci_lower, ci_upper, p_value : float
    K_selected : int
        Number of SNPs flagged as invalid (pleiotropic) by BIC.
    invalid_snps : np.ndarray
        Boolean mask of the K_selected SNPs flagged as invalid.
    path : pd.DataFrame
        Full path: one row per K in [0, K_max] with columns
        ``[K, estimate, se, loglik, bic]``.
    loglik : float
        Final log-likelihood at K_selected.
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
        ci = f"[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]"
        lines = [
            "MR-cML-BIC (constrained maximum-likelihood MR)",
            "=" * 62,
            f"  n SNPs                : {self.n_snps}",
            f"  invalid (K_selected)  : {self.K_selected}",
            f"  causal β              : {self.estimate:+.4f}   " f"SE = {self.se:.4f}",
            f"  95% CI                : {ci}",
            f"  p-value               : {self.p_value:.4g}",
            f"  log-lik               : {self.loglik:.3f}",
            "",
            "K-path (BIC-selected row starred):",
        ]
        for _, row in self.path.iterrows():
            star = " *" if int(row["K"]) == self.K_selected else "  "
            lines.append(
                f"{star} K={int(row['K']):2d}  β={row['estimate']:+.4f}  "
                f"SE={row['se']:.4f}  logL={row['loglik']:8.3f}  "
                f"BIC={row['bic']:.3f}"
            )
        return "\n".join(lines)


def _fit_fixed_k(
    bx: np.ndarray,
    by: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    K: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> tuple[float, float, float, np.ndarray]:
    """MR-cML at fixed K, following ``MendelianRandomization:::cML_estimate``.

    Start from theta = 0 and mu = 0 and iterate, in this order, until
    ``|theta_new - theta| <= tol`` or ``max_iter`` sweeps:

    1. flag the K variants with the largest ``(by - theta mu)^2 / vy`` as
       invalid and set their ``r = by - theta mu`` (0 elsewhere);
    2. ``mu = (bx / vx + theta (by - r) / vy) / (1 / vx + theta^2 / vy)``;
    3. ``theta = sum((by - r) mu / vy) / sum(mu^2 / vy)``.

    For invalid variants ``mu`` is then reset to ``bx`` and ``r`` to
    ``by - theta mu``, their unconstrained maximisers. The likelihood is not
    concave in the invalid set, so the start and the update order decide
    which optimum is reached; this follows the reference so both land on the
    same one (the block-coordinate descent it replaces started from the IVW
    estimate, updated in a different order, and on the reference package's
    LDL-C example reached a different optimum at K = 6, 4.5% away).

    Returns ``(theta, se, loglik, invalid_idx)``. ``se`` is the profile
    information for theta over the valid variants
    (``MendelianRandomization:::cML_SdTheta``)::

        1 / [ sum b^2 / vy  -  sum (2 b theta - by)^2 / vy^2
                                   / (1 / vx + theta^2 / vy) ]

    The second term is the cost of estimating the nuisance exposure effects.
    Dropping it -- as this function did before 1.27.0, under a comment
    calling the result a profile-likelihood SE -- understates the standard
    error by ~12% on that example.
    """
    n = len(bx)
    theta = 0.0
    mu = np.zeros(n)
    r = np.zeros(n)
    invalid_idx = np.array([], dtype=int)
    theta_old = theta - 1.0
    it = 0
    while abs(theta_old - theta) > tol and it < max_iter:
        theta_old = theta
        it += 1
        if K > 0:
            importance = (by - theta * mu) ** 2 / vy
            invalid_idx = np.sort(np.argsort(-importance, kind="stable")[:K])
            r = np.zeros(n)
            r[invalid_idx] = (by - theta * mu)[invalid_idx]
        else:
            r = np.zeros(n)
        mu = (bx / vx + theta * (by - r) / vy) / (1.0 / vx + theta**2 / vy)
        theta = float(np.sum((by - r) * mu / vy) / np.sum(mu**2 / vy))
    if K > 0:
        mu[invalid_idx] = bx[invalid_idx]
        r[invalid_idx] = (by - theta * mu)[invalid_idx]

    neg_l = float(
        np.sum((bx - mu) ** 2 / (2 * vx))
        + np.sum((by - theta * mu - r) ** 2 / (2 * vy))
    )
    ll = -neg_l - 0.5 * float(
        np.sum(np.log(2 * np.pi * vx)) + np.sum(np.log(2 * np.pi * vy))
    )

    valid = np.ones(n, dtype=bool)
    valid[invalid_idx] = False
    info = float(
        np.sum((mu**2 / vy)[valid])
        - np.sum(
            ((2 * mu * theta - by) ** 2 / vy**2 / (1.0 / vx + theta**2 / vy))[valid]
        )
    )
    se = float(np.sqrt(1.0 / info)) if info > 0 else float("nan")
    return theta, se, ll, invalid_idx


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
    """Constrained maximum-likelihood MR (MR-cML-BIC).

    Equivalent to ``MendelianRandomization::mr_cML(..., MA = model_average,
    DP = FALSE, n = n)``. The reference's default also applies data
    perturbation (``DP = TRUE``, 200 resamples), which is not implemented
    here.

    .. versionchanged:: 1.27.0
       * The BIC penalty is ``K log(n)`` with ``n`` the GWAS **sample size**,
         which the method requires; it used the number of variants, a much
         weaker penalty (log 28 against log 17723 on the reference package's
         LDL-C example), and selected six invalid variants where the
         reference selects two. ``n`` is now a parameter; without it the old
         stand-in is used and a warning says so.
       * The standard error is the profile information, not the information
         with the nuisance exposure effects held fixed (~12% smaller).
       * The fixed-K fit follows the reference's start and update order, so
         both reach the same optimum; ``K_max`` defaults to ``n_snps - 2``
         as in the reference.
       * ``model_average=True`` gives the reference's MA-BIC estimate.

    Implements MR-cML with BIC-selected sparsity (Xue, Shen & Pan 2021,
    *AJHG* 108(7)).  The model lets each SNP have its own pleiotropy
    effect :math:`r_i` subject to :math:`\\|r\\|_0 \\le K`; the number of
    pleiotropic SNPs K is selected by BIC.

    Model
    -----
    .. math::

       \\beta_{x,i}^{obs} &= \\beta_{x,i}^{true} + e_i,
       \\quad e_i \\sim \\mathcal{N}(0, s_{x,i}^2) \\\\
       \\beta_{y,i}^{obs} &= \\beta\\,\\beta_{x,i}^{true} + r_i + \\epsilon_i,
       \\quad \\epsilon_i \\sim \\mathcal{N}(0, s_{y,i}^2) \\\\
       & \\text{with } \\|r\\|_0 \\le K.

    Jointly MLE over :math:`(\\beta, \\{\\beta_{x,i}^{true}\\}, r)` by
    block-coordinate descent; repeat for K=0..K_max and select K*
    minimising BIC = -2 logL + K log(n).

    Parameters
    ----------
    beta_exposure, beta_outcome : ndarray
    se_exposure, se_outcome : ndarray
    K_max : int, optional
        Maximum pleiotropy cardinality to try.  Defaults to ``n_snps - 2``,
        the reference's ``K_vec``.  Must be in ``[0, n_snps - 1]``.
    alpha : float, default 0.05
    max_iter, tol : int, float
        Stop the fixed-K iteration when ``|theta_new - theta| <= tol`` or
        after ``max_iter`` sweeps (the reference's rule and defaults).
    n : int, optional
        GWAS sample size for the BIC penalty. Required by the method; if
        omitted the number of variants stands in and a ``UserWarning`` is
        raised.
    model_average : bool, default False
        Average over K with weights proportional to ``exp(-BIC / 2)``; the
        standard error is ``sum w sqrt(se_K^2 + (theta_K - theta_MA)^2)``.

    Returns
    -------
    :class:`MRcMLResult`

    Notes
    -----
    When ``K_max=0`` and the DGP is free of pleiotropy MR-cML reduces
    to a measurement-error-corrected IVW (equivalent to the Bowden 2019
    attenuation-corrected IVW).  Compare with :func:`grapple` for a
    random-effects formulation of the same pleiotropy-robust target.

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
    >>> bool(np.isfinite(res.estimate))
    True
    """
    bx, by, sx, sy = as_float_arrays(
        beta_exposure, beta_outcome, se_exposure, se_outcome
    )
    bx, by = harmonize_signs(bx, by)
    vx = sx**2
    vy = sy**2
    p = len(bx)

    if K_max is None:
        K_max = max(0, p - 2)
    if not 0 <= K_max <= p - 1:
        raise ValueError(f"K_max must be in [0, n-1]; got {K_max}")
    if n is None:
        warnings.warn(
            "mr_cml: no GWAS sample size given (n=None). The BIC penalty needs "
            "it; using the number of variants instead, which penalises "
            "invalid variants far less than MR-cML-BIC as published. Pass n= "
            "to reproduce MendelianRandomization::mr_cML.",
            UserWarning,
            stacklevel=2,
        )
    log_n = float(np.log(n if n is not None else p))

    rows = []
    fits = {}
    for K in range(0, K_max + 1):
        beta_hat, se_hat, ll, invalid_idx = _fit_fixed_k(
            bx, by, vx, vy, K, max_iter=max_iter, tol=tol
        )
        rows.append(
            {
                "K": K,
                "estimate": beta_hat,
                "se": se_hat,
                "loglik": ll,
                "bic": -2.0 * ll + K * log_n,
            }
        )
        fits[K] = (beta_hat, se_hat, ll, invalid_idx)

    path_df = pd.DataFrame(rows)
    K_best = int(path_df.loc[path_df["bic"].idxmin(), "K"])
    beta_hat, se_hat, ll, invalid_idx = fits[K_best]
    if model_average:
        bic = path_df["bic"].to_numpy(float)
        w = np.exp(-0.5 * (bic - bic.min()))
        w = w / w.sum()
        th = path_df["estimate"].to_numpy(float)
        sd = path_df["se"].to_numpy(float)
        beta_hat = float(np.sum(w * th))
        se_hat = float(np.nansum(w * np.sqrt(sd**2 + (th - beta_hat) ** 2)))
        path_df["weight"] = w

    z_crit = stats.norm.ppf(1 - alpha / 2)
    if np.isfinite(se_hat) and se_hat > 0:
        z = beta_hat / se_hat
        p_value = float(2.0 * stats.norm.sf(abs(z)))
        lo = beta_hat - z_crit * se_hat
        hi = beta_hat + z_crit * se_hat
    else:
        p_value = float("nan")
        lo = hi = float("nan")

    invalid_mask = np.zeros(p, dtype=bool)
    invalid_mask[invalid_idx] = True

    return MRcMLResult(
        estimate=beta_hat,
        se=se_hat,
        ci_lower=lo,
        ci_upper=hi,
        p_value=p_value,
        K_selected=K_best,
        invalid_snps=invalid_mask,
        path=path_df,
        loglik=ll,
        n_snps=p,
        n_samples=n,
        model_average=model_average,
    )
