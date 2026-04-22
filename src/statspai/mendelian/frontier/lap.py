"""MR-Lap: sample-overlap-corrected IVW.

Reference
---------
Burgess, S., Davies, N.M. & Thompson, S.G. (2016).
"Bias due to participant overlap in two-sample Mendelian
randomization." *Genetic Epidemiology*, 40(7), 597-608.

Mounier, N. & Kutalik, Z. (2023).
"Bias correction for inverse variance weighting Mendelian
randomization." *Genetic Epidemiology*, 47(4), 314-331.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ._common import as_float_arrays, harmonize_signs, mean_f_statistic


__all__ = ["MRLapResult", "mr_lap"]


@dataclass
class MRLapResult:
    """Output of :func:`mr_lap`.

    Attributes
    ----------
    estimate : float
        Overlap-corrected causal estimate.
    se : float
        First-order SE (identical to IVW SE; overlap correction does not
        change the variance under the Burgess-Davies-Thompson 2016
        approximation).
    ci_lower, ci_upper : float
    p_value : float
    estimate_ivw : float
        The uncorrected IVW point estimate (for comparison).
    bias_correction : float
        ``estimate_ivw - estimate``, i.e. the amount subtracted from IVW.
    overlap_fraction : float
    overlap_rho : float
    f_mean : float
        Mean first-stage F-statistic across SNPs.
    n_snps : int
    """
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    estimate_ivw: float
    bias_correction: float
    overlap_fraction: float
    overlap_rho: float
    f_mean: float
    n_snps: int

    def summary(self) -> str:
        ci = f"[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]"
        lines = [
            "MR-Lap (sample-overlap-corrected IVW)",
            "=" * 62,
            f"  n SNPs          : {self.n_snps}",
            f"  overlap fraction: {self.overlap_fraction:.3f}",
            f"  overlap rho     : {self.overlap_rho:+.4f}",
            f"  mean F-stat     : {self.f_mean:.2f}"
            + ("  (weak; F<10)" if self.f_mean < 10 else ""),
            "",
            f"  IVW (naive)     : {self.estimate_ivw:+.4f}",
            f"  bias correction : {self.bias_correction:+.4f}",
            f"  MR-Lap estimate : {self.estimate:+.4f}   SE = {self.se:.4f}",
            f"  95% CI          : {ci}",
            f"  p-value         : {self.p_value:.4g}",
        ]
        return "\n".join(lines)


def mr_lap(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_exposure: np.ndarray,
    se_outcome: np.ndarray,
    *,
    overlap_fraction: float = 1.0,
    overlap_rho: float = 0.0,
    alpha: float = 0.05,
) -> MRLapResult:
    """Sample-overlap-corrected inverse-variance-weighted MR.

    Corrects the first-order bias introduced when the exposure and
    outcome GWAS share participants.  Under Burgess, Davies & Thompson
    (2016) Eq. 8, the bias in two-sample IVW satisfies

    .. math::

       E[\\hat\\beta_{IVW}] - \\beta \\approx \\frac{p \\, \\rho_{obs}}{F_{mean}}

    where :math:`p` is the overlap fraction (share of samples appearing in
    both GWAS), :math:`\\rho_{obs}` is the phenotypic correlation of
    exposure and outcome in the overlapping sub-sample, and
    :math:`F_{mean}` is the mean first-stage F-statistic of the SNPs.

    When ``overlap_fraction=0`` (clean two-sample design) the correction
    vanishes and the estimate equals standard IVW.

    Parameters
    ----------
    beta_exposure, beta_outcome : ndarray
        Per-SNP GWAS effect sizes.
    se_exposure, se_outcome : ndarray
        Per-SNP GWAS standard errors.
    overlap_fraction : float, default 1.0
        Share of the outcome sample that also appears in the exposure
        sample.  0 = clean two-sample; 1 = full overlap; in between =
        partial overlap.  Must be in [0, 1].
    overlap_rho : float, default 0.0
        Phenotypic correlation between exposure and outcome measurements
        in the overlapping sub-sample.  In practice estimated by LD-score
        regression (``ldsc --rg``).  Must be in [-1, 1].
    alpha : float, default 0.05
        Two-sided significance level for the CI and p-value.

    Returns
    -------
    :class:`MRLapResult`

    Notes
    -----
    This implements the closed-form Burgess-Davies-Thompson (2016)
    correction.  The more elaborate MR-Lap of Mounier & Kutalik (2023)
    additionally regresses bias on a non-linear function of
    ``(F_mean, overlap_fraction)`` to absorb weak-instrument curvature;
    that refinement is deferred to a future release.  When the user-
    supplied ``overlap_rho`` is obtained from LD-score regression this
    simplified correction captures most of the overlap bias on
    real-world GWAS scales (F > 30).

    The SE is kept equal to the naive-IVW SE: the correction shifts the
    point estimate only.  Monte-Carlo studies (Burgess 2016 Figs. 2-3)
    show coverage remains nominal under this conservative choice because
    the correction itself has second-order variance.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.mr_lap(bx, by, sx, sy,
    ...                 overlap_fraction=0.4, overlap_rho=0.18)
    >>> print(res.summary())
    """
    bx, by, sx, sy = as_float_arrays(
        beta_exposure, beta_outcome, se_exposure, se_outcome
    )
    bx, by = harmonize_signs(bx, by)

    if not 0.0 <= overlap_fraction <= 1.0:
        raise ValueError(
            f"overlap_fraction must be in [0, 1]; got {overlap_fraction}"
        )
    if not -1.0 <= overlap_rho <= 1.0:
        raise ValueError(
            f"overlap_rho must be in [-1, 1]; got {overlap_rho}"
        )

    # Naive IVW slope via weighted regression through the origin.
    w = 1.0 / sy ** 2
    denom = float(np.sum(w * bx ** 2))
    if denom <= 0:
        raise ValueError(
            "mr_lap: Σw·βx² is non-positive — check input signs / SEs"
        )
    b_ivw = float(np.sum(w * bx * by) / denom)
    se_ivw = float(np.sqrt(1.0 / denom))

    F_mean = mean_f_statistic(bx, sx)
    if F_mean <= 0:
        raise ValueError("mr_lap: mean F-statistic is non-positive")

    # Burgess-Davies-Thompson 2016 Eq. 8 closed-form bias approximation.
    bias = overlap_fraction * overlap_rho / F_mean
    b_corrected = b_ivw - bias

    z_crit = stats.norm.ppf(1 - alpha / 2)
    z = b_corrected / se_ivw
    p_value = 2.0 * stats.norm.sf(abs(z))

    return MRLapResult(
        estimate=b_corrected,
        se=se_ivw,
        ci_lower=b_corrected - z_crit * se_ivw,
        ci_upper=b_corrected + z_crit * se_ivw,
        p_value=float(p_value),
        estimate_ivw=b_ivw,
        bias_correction=float(bias),
        overlap_fraction=float(overlap_fraction),
        overlap_rho=float(overlap_rho),
        f_mean=F_mean,
        n_snps=len(bx),
    )
