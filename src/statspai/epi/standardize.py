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

from .._result_serialize import ResultProtocolMixin

__all__ = [
    "StandardizedRateResult",
    "SMRResult",
    "direct_standardize",
    "indirect_standardize",
]


@dataclass
class StandardizedRateResult(ResultProtocolMixin):
    rate: float
    ci: tuple[float, float]
    stratum_rates: np.ndarray
    standard_weights: np.ndarray
    method: str = "direct"
    se: float = float("nan")
    ci_method: str = "lognormal"

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"Direct-standardized rate = {self.rate:.6f}  "
            f"95% CI [{lo:.6f}, {hi:.6f}]\n"
            f"  Strata: {len(self.stratum_rates)}"
        )


@dataclass
class SMRResult(ResultProtocolMixin):
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
    ci_method: str = "lognormal",
    variance: str = "poisson",
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
    ci_method : {"lognormal", "gamma", "normal"}, default "lognormal"
        ``"lognormal"``: ``exp(log r +/- z se / r)``. ``"gamma"``: the
        Fay-Feuer gamma interval, as R ``epitools::ageadjust.direct``.
        ``"normal"``: ``r +/- z se`` with the lower bound truncated at 0, as
        Stata ``dstdize`` (use with ``variance="binomial"``).
    variance : {"poisson", "binomial"}, default "poisson"
        ``"poisson"``: ``sum w_k^2 events_k / population_k^2`` (epitools).
        ``"binomial"``: ``sum w_k^2 r_k (1 - r_k) / population_k`` (Stata
        ``dstdize``).

    Returns
    -------
    StandardizedRateResult

    Notes
    -----
    The standard error is the square root of the chosen variance of the
    weighted sum of stratum rates; ``.se`` stores it.

    Examples
    --------
    >>> import statspai as sp
    >>> events = [30, 50, 80]          # event counts per age stratum
    >>> population = [1000, 1000, 500]
    >>> standard_weights = [4000, 4000, 2000]   # external standard population
    >>> res = sp.direct_standardize(events, population, standard_weights)
    >>> round(res.rate, 4)
    0.064
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

    if ci_method not in ("lognormal", "gamma", "normal"):
        raise ValueError("ci_method must be 'lognormal', 'gamma' or 'normal'")
    if variance not in ("poisson", "binomial"):
        raise ValueError("variance must be 'poisson' or 'binomial'")
    w = w_raw / w_raw.sum()
    stratum_rates = e / p
    r_std = float(np.sum(w * stratum_rates))
    if variance == "poisson":
        var = float(np.sum(w**2 * e / p**2))
    else:
        var = float(np.sum(w**2 * stratum_rates * (1 - stratum_rates) / p))
    se = float(np.sqrt(var))
    z = stats.norm.ppf(1 - alpha / 2)

    if ci_method == "lognormal":
        if r_std > 0:
            log_rate = np.log(r_std)
            se_log = se / r_std
            ci = (
                float(np.exp(log_rate - z * se_log)),
                float(np.exp(log_rate + z * se_log)),
            )
        else:
            ci = (0.0, float(z * se))
    elif ci_method == "gamma":
        # Fay & Feuer: gamma with the DSR's mean and variance for the lower
        # bound; mean and variance inflated by the largest w_k / pop_k for
        # the upper bound.
        wm = float(np.max(w / p))
        lo = (
            float(stats.gamma.ppf(alpha / 2, r_std**2 / var, scale=var / r_std))
            if var > 0 and r_std > 0
            else 0.0
        )
        hi = float(
            stats.gamma.isf(
                alpha / 2,
                (r_std + wm) ** 2 / (var + wm**2),
                scale=(var + wm**2) / (r_std + wm),
            )
        )
        ci = (lo, hi)
    else:
        ci = (float(max(0.0, r_std - z * se)), float(r_std + z * se))

    return StandardizedRateResult(
        rate=r_std,
        ci=ci,
        stratum_rates=stratum_rates,
        standard_weights=w,
        method="direct",
        se=se,
        ci_method=ci_method,
    )


def indirect_standardize(
    observed: float,
    events_reference: Sequence[float],
    population_reference: Sequence[float],
    population_study: Sequence[float],
    *,
    alpha: float = 0.05,
    ci_method: str = "exact",
) -> SMRResult:
    """Indirect standardization -> Standardized Morbidity/Mortality Ratio.

    Expected events = sum_k (rate_ref_k * pop_study_k), where
    rate_ref_k = events_reference_k / population_reference_k.
    SMR = observed / expected.

    ``ci_method="exact"`` (default) is the exact Poisson (Garwood)
    interval for the observed count divided by the expected count, as
    Stata ``istdize``; ``"lognormal"`` is ``SMR exp(+/- z / sqrt(O))``, as R
    ``epitools::ageadjust.indirect``. ``p_value`` is the exact two-sided
    Poisson test of SMR = 1: twice the smaller exact tail, capped at 1.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.indirect_standardize(
    ...     observed=120,
    ...     events_reference=[30, 50, 80],
    ...     population_reference=[1000, 1000, 500],
    ...     population_study=[800, 1200, 600],
    ... )
    >>> round(res.expected, 4)
    180.0
    >>> round(res.smr, 4)
    0.6667
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

    if ci_method not in ("exact", "lognormal"):
        raise ValueError("ci_method must be 'exact' or 'lognormal'")
    # Garwood exact CI for Poisson: 2*O ~ chi2(2*O) and 2*(O+1) ~ chi2(2*(O+1))
    if ci_method == "lognormal":
        if observed <= 0:
            raise ValueError("ci_method='lognormal' needs observed > 0.")
        zq = stats.norm.ppf(1 - alpha / 2)
        half = zq / np.sqrt(observed)
        ci_count = (
            float(observed * np.exp(-half)),
            float(observed * np.exp(half)),
        )
    elif observed > 0:
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
        # Exact two-sided p: twice the smaller exact tail (not mid-p)
        lower = stats.poisson.cdf(observed, expected)
        upper = stats.poisson.sf(observed - 1, expected)
        p_one = min(lower, upper)
    p_two = float(min(1.0, 2.0 * p_one))

    return SMRResult(
        smr=smr,
        ci=(float(ci[0]), float(ci[1])),
        observed=float(observed),
        expected=expected,
        p_value=p_two,
    )
