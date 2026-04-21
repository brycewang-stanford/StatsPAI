"""
Direct and indirect standardization of rates / risks.

Given a set of stratum-specific rates (or risks) and stratum sizes in
the study population, compute the age-adjusted (or generally
covariate-adjusted) rate as a weighted average using an external
standard population.

- Direct standardization: observed stratum rates x standard weights.
- Indirect standardization: SMR = Observed / Expected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


__all__ = [
    "StandardizedRateResult",
    "SMRResult",
    "direct_standardize",
    "indirect_standardize",
]


@dataclass
class StandardizedRateResult:
    rate: float
    ci: tuple[float, float]
    stratum_rates: np.ndarray
    standard_weights: np.ndarray
    method: str = "direct"

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"Direct-standardized rate = {self.rate:.6f}  "
            f"95% CI [{lo:.6f}, {hi:.6f}]\n"
            f"  Strata: {len(self.stratum_rates)}"
        )


@dataclass
class SMRResult:
    smr: float
    ci: tuple[float, float]
    observed: float
    expected: float
    p_value: float

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"Standardized Mortality/Morbidity Ratio\n"
            f"  SMR = {self.smr:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   "
            f"p = {self.p_value:.4g}\n"
            f"  Observed = {self.observed:.2f}   Expected = {self.expected:.4f}"
        )


def direct_standardize(
    events: Sequence[float],
    population: Sequence[float],
    standard_weights: Sequence[float],
    *,
    alpha: float = 0.05,
) -> StandardizedRateResult:
    """Direct standardization of a rate.

    The standardized rate is:

        r_std = sum_k (w_k * events_k / population_k)

    where ``w_k`` are the relative weights of a standard population
    (they are normalized internally to sum to 1).

    Parameters
    ----------
    events : array-like
        Event counts in each stratum of the study population.
    population : array-like
        Denominator (person-time or population size) in each stratum.
    standard_weights : array-like
        Standard population size or proportion per stratum.  Will be
        normalized to sum to 1.
    alpha : float

    Returns
    -------
    StandardizedRateResult

    Notes
    -----
    SE is computed by the delta method on the weighted sum of stratum
    rates, treating events as Poisson.
    """
    e = np.asarray(events, dtype=float)
    p = np.asarray(population, dtype=float)
    w_raw = np.asarray(standard_weights, dtype=float)
    if not (len(e) == len(p) == len(w_raw)):
        raise ValueError("events, population, standard_weights must align.")
    if (p <= 0).any():
        raise ValueError("population must be positive in every stratum.")
    if (w_raw < 0).any():
        raise ValueError("standard_weights must be non-negative.")

    w = w_raw / w_raw.sum()
    stratum_rates = e / p
    r_std = float(np.sum(w * stratum_rates))
    var = float(np.sum(w ** 2 * e / p ** 2))
    se = np.sqrt(var)
    z = stats.norm.ppf(1 - alpha / 2)

    # Log-transform for a stable CI (Breslow-Day)
    if r_std > 0:
        log_rate = np.log(r_std)
        se_log = se / r_std
        ci = (float(np.exp(log_rate - z * se_log)),
              float(np.exp(log_rate + z * se_log)))
    else:
        ci = (0.0, float(z * se))

    return StandardizedRateResult(
        rate=r_std,
        ci=ci,
        stratum_rates=stratum_rates,
        standard_weights=w,
        method="direct",
    )


def indirect_standardize(
    observed: float,
    events_reference: Sequence[float],
    population_reference: Sequence[float],
    population_study: Sequence[float],
    *,
    alpha: float = 0.05,
) -> SMRResult:
    """Indirect standardization -> Standardized Morbidity/Mortality Ratio.

    Expected events = sum_k (rate_ref_k * pop_study_k), where
    rate_ref_k = events_reference_k / population_reference_k.
    SMR = observed / expected.

    CI uses exact Poisson (Byar's approximation / Garwood).
    """
    er = np.asarray(events_reference, dtype=float)
    pr = np.asarray(population_reference, dtype=float)
    ps = np.asarray(population_study, dtype=float)
    if not (len(er) == len(pr) == len(ps)):
        raise ValueError("Reference and study arrays must align.")
    if (pr <= 0).any():
        raise ValueError("Reference population must be positive.")

    reference_rates = er / pr
    expected = float(np.sum(reference_rates * ps))
    if expected <= 0:
        raise ValueError("Expected events = 0; SMR undefined.")
    smr = float(observed / expected)

    # Garwood exact CI for Poisson: 2*O ~ chi2(2*O) and 2*(O+1) ~ chi2(2*(O+1))
    if observed > 0:
        lo_chi = stats.chi2.ppf(alpha / 2, 2 * observed) / 2.0
        hi_chi = stats.chi2.ppf(1 - alpha / 2, 2 * (observed + 1)) / 2.0
        ci_count = (float(lo_chi), float(hi_chi))
    else:
        ci_count = (0.0, float(stats.chi2.ppf(1 - alpha / 2, 2) / 2.0))
    ci = (ci_count[0] / expected, ci_count[1] / expected)

    # Two-sided exact Poisson p-value for H0: SMR = 1  (observed ~ Poisson(expected))
    if observed == 0:
        p_one = float(np.exp(-expected))
    else:
        # Mid-p two-sided test
        lower = stats.poisson.cdf(observed, expected)
        upper = 1 - stats.poisson.cdf(observed - 1, expected)
        p_one = min(lower, upper)
    p_two = float(min(1.0, 2.0 * p_one))

    return SMRResult(
        smr=smr,
        ci=(float(ci[0]), float(ci[1])),
        observed=float(observed),
        expected=expected,
        p_value=p_two,
    )
