"""MR-cML-BIC: constrained maximum-likelihood MR with L0-sparse pleiotropy.

Reference
---------
Xue, H., Shen, X. & Pan, W. (2021).
"Constrained maximum likelihood-based Mendelian randomization
robust to both correlated and uncorrelated pleiotropic effects."
*AJHG*, 108(7), 1251-1269.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from ._common import as_float_arrays, harmonize_signs


__all__ = ["MRcMLResult", "mr_cml"]


@dataclass
class MRcMLResult:
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

    def summary(self) -> str:
        ci = f"[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]"
        lines = [
            "MR-cML-BIC (constrained maximum-likelihood MR)",
            "=" * 62,
            f"  n SNPs                : {self.n_snps}",
            f"  invalid (K_selected)  : {self.K_selected}",
            f"  causal β              : {self.estimate:+.4f}   "
            f"SE = {self.se:.4f}",
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
    bx: np.ndarray, by: np.ndarray, vx: np.ndarray, vy: np.ndarray,
    K: int, *, max_iter: int = 200, tol: float = 1e-8,
):
    """Block-coordinate-descent MR-cML at fixed K.

    Algorithm (Xue, Shen, Pan 2021 §2.3):
    - Parameters: β (scalar causal), b_x (true exposure per SNP),
      r (pleiotropy per SNP with ||r||_0 ≤ K).
    - Block update loop:
      1. Given β, r: b_x_i = (bx_i/vx_i + β (by_i - r_i)/vy_i) /
                            (1/vx_i + β² / vy_i)
      2. Given β, b_x: r_i = by_i - β * b_x_i   for the K SNPs with
         largest residual squared / vy; r_i = 0 otherwise.
      3. Given b_x, r: β = Σ b_x_i (by_i - r_i)/vy_i /
                           Σ b_x_i² / vy_i
    - Log-likelihood:
      logL = -0.5 Σ [(bx_i - b_x_i)² / vx_i + log(2π vx_i)
                     + (by_i - β b_x_i - r_i)² / vy_i + log(2π vy_i)]
    """
    n = len(bx)
    b_true = bx.copy()
    beta = float(np.sum(bx * by / vy) / max(np.sum(bx ** 2 / vy), 1e-300))
    r = np.zeros(n)
    invalid_idx = np.array([], dtype=int)

    prev_ll = -np.inf
    for _ in range(max_iter):
        # Step 1: update b_x (vector)
        num = bx / vx + beta * (by - r) / vy
        den = 1.0 / vx + beta ** 2 / vy
        b_true = num / den

        # Step 2: update r with cardinality constraint ||r||_0 ≤ K
        full_resid = by - beta * b_true
        if K > 0:
            score = full_resid ** 2 / vy
            invalid_idx = np.argpartition(-score, K - 1)[:K]
            r = np.zeros(n)
            r[invalid_idx] = full_resid[invalid_idx]
        else:
            invalid_idx = np.array([], dtype=int)
            r = np.zeros(n)

        # Step 3: update beta on the remaining (valid) SNPs
        w_num = np.sum(b_true * (by - r) / vy)
        w_den = np.sum(b_true ** 2 / vy)
        if w_den <= 1e-300:
            beta_new = beta
        else:
            beta_new = float(w_num / w_den)

        ll = -0.5 * float(
            np.sum((bx - b_true) ** 2 / vx)
            + np.sum((by - beta_new * b_true - r) ** 2 / vy)
            + np.sum(np.log(2 * np.pi * vx))
            + np.sum(np.log(2 * np.pi * vy))
        )

        if abs(ll - prev_ll) < tol:
            beta = beta_new
            break
        beta = beta_new
        prev_ll = ll

    # Profile-likelihood SE at the selected model.
    mask = np.ones(n, dtype=bool)
    if K > 0 and len(invalid_idx) > 0:
        mask[invalid_idx] = False
    info = float(np.sum(b_true[mask] ** 2 / vy[mask]))
    se = float(np.sqrt(1.0 / info)) if info > 0 else float("nan")

    return beta, se, ll, invalid_idx


def mr_cml(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_exposure: np.ndarray,
    se_outcome: np.ndarray,
    *,
    K_max: Optional[int] = None,
    alpha: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> MRcMLResult:
    """Constrained maximum-likelihood MR (MR-cML-BIC).

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
        Maximum pleiotropy cardinality to try.  Defaults to ``n_snps - 3``
        (ensures at least 3 valid SNPs).  Must be in ``[0, n_snps - 1]``.
    alpha : float, default 0.05
    max_iter, tol : inner block-CD controls.

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
    >>> res = sp.mr_cml(bx, by, sx, sy)
    >>> print(res.summary())
    """
    bx, by, sx, sy = as_float_arrays(
        beta_exposure, beta_outcome, se_exposure, se_outcome
    )
    bx, by = harmonize_signs(bx, by)
    vx = sx ** 2
    vy = sy ** 2
    n = len(bx)

    if K_max is None:
        K_max = max(0, n - 3)
    if not 0 <= K_max <= n - 1:
        raise ValueError(f"K_max must be in [0, n-1]; got {K_max}")

    rows = []
    fits = {}
    for K in range(0, K_max + 1):
        beta_hat, se_hat, ll, invalid_idx = _fit_fixed_k(
            bx, by, vx, vy, K, max_iter=max_iter, tol=tol,
        )
        bic = -2.0 * ll + K * np.log(n)
        rows.append({
            "K": K,
            "estimate": beta_hat,
            "se": se_hat,
            "loglik": ll,
            "bic": bic,
        })
        fits[K] = (beta_hat, se_hat, ll, invalid_idx)

    path_df = pd.DataFrame(rows)
    K_best = int(path_df.loc[path_df["bic"].idxmin(), "K"])
    beta_hat, se_hat, ll, invalid_idx = fits[K_best]

    z_crit = stats.norm.ppf(1 - alpha / 2)
    if np.isfinite(se_hat) and se_hat > 0:
        z = beta_hat / se_hat
        p_value = float(2.0 * stats.norm.sf(abs(z)))
        lo = beta_hat - z_crit * se_hat
        hi = beta_hat + z_crit * se_hat
    else:
        p_value = float("nan")
        lo = hi = float("nan")

    invalid_mask = np.zeros(n, dtype=bool)
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
        n_snps=n,
    )
