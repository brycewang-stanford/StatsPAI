"""Power and sample-size calculations for epidemiological study designs.

Complements the econometric designs in :mod:`statspai.power.power` (RCT, DiD,
RD, IV, cluster RCT, OLS) with the designs public-health and clinical
researchers reach for:

- :func:`power_two_proportions` — two-arm comparison of a binary outcome
  (cohort study / RCT with a binary endpoint).
- :func:`power_logrank` — survival / time-to-event comparison via the
  Schoenfeld (1983) number-of-events formula.
- :func:`power_case_control` — unmatched case-control study, parameterised by
  the exposure odds ratio and the control-group exposure prevalence.

Each function returns a :class:`statspai.power.power.PowerResult` and accepts
either a scalar or an array for the sample-size argument, so it composes with
power-curve plotting. Passing ``power_target`` instead of ``n`` solves for the
sample size that achieves the requested power.

References
----------
Schoenfeld, D.A. (1983). "Sample-size formula for the proportional-hazards
regression model." *Biometrics*, 39(2), 499-503. [@schoenfeld1983sample]

Fleiss, J.L., Levin, B. & Paik, M.C. (2003). *Statistical Methods for Rates
and Proportions*, 3rd ed. Wiley.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.stats import norm

from .power import PowerResult

__all__ = [
    "power_two_proportions",
    "power_logrank",
    "power_case_control",
]

ArrayLike = Union[float, int, np.ndarray]


def _as_array(value: ArrayLike) -> np.ndarray:
    return np.atleast_1d(np.asarray(value, dtype=float))


def _scalarize(arr: np.ndarray) -> Union[float, np.ndarray]:
    return float(arr.item()) if arr.size == 1 else arr


def _z_alpha(alpha: float, alternative: str) -> float:
    if alternative == "two-sided":
        return float(norm.ppf(1 - alpha / 2))
    return float(norm.ppf(1 - alpha))


def power_two_proportions(
    n: Optional[ArrayLike] = None,
    p1: float = 0.5,
    p2: float = 0.5,
    *,
    ratio: float = 1.0,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    power_target: Optional[float] = None,
) -> PowerResult:
    """Power (or sample size) to detect a difference between two proportions.

    Parameters
    ----------
    n : int, array-like, or None
        Total sample size (both groups). Pass ``None`` together with
        ``power_target`` to solve for the smallest ``n`` achieving that power.
    p1, p2 : float
        Outcome probabilities in group 1 (reference) and group 2.
    ratio : float
        Allocation ratio ``n2 / n1`` (1.0 = equal allocation).
    alpha : float
        Significance level.
    alternative : {"two-sided", "one-sided"}
        Test sidedness.
    power_target : float, optional
        Desired power; when supplied with ``n=None`` the function returns the
        required total sample size.

    Returns
    -------
    PowerResult

    Notes
    -----
    Uses the unpooled-variance normal approximation to a two-sample test of
    proportions: ``power = Phi(|p1 - p2| / se - z_alpha)`` with
    ``se = sqrt(p1(1-p1)/n1 + p2(1-p2)/n2)``.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.power_two_proportions(n=400, p1=0.5, p2=0.65)
    >>> isinstance(res, sp.PowerResult)
    True
    >>> res.design
    'two_proportions'
    >>> round(float(res.power), 4)
    0.8665
    >>> # Solve for the total sample size achieving 80% power.
    >>> int(sp.power_two_proportions(p1=0.5, p2=0.65, power_target=0.8).n)
    334
    """
    z_a = _z_alpha(alpha, alternative)
    delta = abs(p2 - p1)
    frac1 = 1.0 / (1.0 + ratio)  # share of total n in group 1

    def _power_for_n(n_total: np.ndarray) -> np.ndarray:
        n1 = n_total * frac1
        n2 = n_total * (1.0 - frac1)
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        out: np.ndarray = norm.cdf(delta / se - z_a)
        return out

    params = dict(p1=p1, p2=p2, ratio=ratio, alpha=alpha, alternative=alternative)

    if n is None:
        if power_target is None:
            raise ValueError("Provide either `n` or `power_target`.")
        if delta == 0:
            raise ValueError("p1 and p2 are equal; no detectable effect.")
        z_b = norm.ppf(power_target)
        p_bar = (p1 + ratio * p2) / (1 + ratio)
        # closed-form starting point, then refine to the integer n.
        n1_0 = (z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2) / ratio) / delta**2
        n_total = np.ceil(n1_0 * (1 + ratio))
        while float(_power_for_n(np.array([n_total]))[0]) < power_target:
            n_total += 1
        _ = p_bar  # documented but not needed for the unpooled SE
        return PowerResult(
            power_val=float(_power_for_n(np.array([n_total]))[0]),
            n=int(n_total),
            effect_size=delta,
            design="two_proportions",
            params=params,
        )

    n_arr = _as_array(n)
    pwr = _power_for_n(n_arr)
    return PowerResult(
        power_val=_scalarize(pwr),
        n=_scalarize(n_arr),
        effect_size=delta,
        design="two_proportions",
        params=params,
    )


def power_logrank(
    n: Optional[ArrayLike] = None,
    hazard_ratio: float = 0.5,
    prob_event: float = 1.0,
    *,
    ratio: float = 1.0,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    power_target: Optional[float] = None,
) -> PowerResult:
    """Power (or sample size) for a two-arm log-rank / survival comparison.

    Implements the Schoenfeld (1983) formula: power depends on the number of
    observed **events**, ``D = n * prob_event``, and the log hazard ratio.

    Parameters
    ----------
    n : int, array-like, or None
        Total sample size. ``None`` + ``power_target`` solves for ``n``.
    hazard_ratio : float
        Hazard ratio between the two arms (must be > 0, != 1).
    prob_event : float
        Probability that a randomly chosen subject is observed to have the
        event during follow-up (the overall event rate). ``D = n*prob_event``.
    ratio : float
        Allocation ratio ``n2 / n1``.
    alpha : float
        Significance level.
    alternative : {"two-sided", "one-sided"}
    power_target : float, optional
        Desired power; solve for ``n`` when supplied with ``n=None``.

    Returns
    -------
    PowerResult

    Notes
    -----
    With allocation share ``p = ratio/(1+ratio)``, the required number of
    events is ``D = (z_alpha + z_beta)^2 / (p(1-p) (ln HR)^2)`` and the power
    for a given ``D`` is ``Phi(sqrt(D p(1-p)) |ln HR| - z_alpha)``.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.power_logrank(n=300, hazard_ratio=0.6, prob_event=0.7)
    >>> res.design
    'logrank'
    >>> round(float(res.power), 4)
    0.9592
    >>> # Sample size for 80% power at a hazard ratio of 0.6.
    >>> int(sp.power_logrank(hazard_ratio=0.6, prob_event=0.7,
    ...                      power_target=0.8).n)
    172
    """
    if hazard_ratio <= 0 or hazard_ratio == 1:
        raise ValueError("hazard_ratio must be > 0 and != 1.")
    z_a = _z_alpha(alpha, alternative)
    p = ratio / (1.0 + ratio)
    log_hr = abs(np.log(hazard_ratio))
    params = dict(
        hazard_ratio=hazard_ratio,
        prob_event=prob_event,
        ratio=ratio,
        alpha=alpha,
        alternative=alternative,
    )

    def _power_for_events(d: np.ndarray) -> np.ndarray:
        out: np.ndarray = norm.cdf(np.sqrt(d * p * (1 - p)) * log_hr - z_a)
        return out

    if n is None:
        if power_target is None:
            raise ValueError("Provide either `n` or `power_target`.")
        z_b = norm.ppf(power_target)
        d_req = (z_a + z_b) ** 2 / (p * (1 - p) * log_hr**2)
        n_total = int(np.ceil(d_req / prob_event))
        return PowerResult(
            power_val=float(_power_for_events(np.array([n_total * prob_event]))[0]),
            n=n_total,
            effect_size=log_hr,
            design="logrank",
            params=dict(params, n_events=int(np.ceil(d_req))),
        )

    n_arr = _as_array(n)
    pwr = _power_for_events(n_arr * prob_event)
    return PowerResult(
        power_val=_scalarize(pwr),
        n=_scalarize(n_arr),
        effect_size=log_hr,
        design="logrank",
        params=dict(params, n_events=_scalarize(n_arr * prob_event)),
    )


def power_case_control(
    n_cases: Optional[ArrayLike] = None,
    odds_ratio: float = 2.0,
    exposure_prevalence: float = 0.3,
    *,
    ratio: float = 1.0,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    power_target: Optional[float] = None,
) -> PowerResult:
    """Power (or number of cases) for an unmatched case-control study.

    Parameters
    ----------
    n_cases : int, array-like, or None
        Number of cases. ``None`` + ``power_target`` solves for the number of
        cases.
    odds_ratio : float
        Exposure odds ratio to detect (must be > 0, != 1).
    exposure_prevalence : float
        Exposure prevalence among controls (the source-population exposure
        probability), in (0, 1).
    ratio : float
        Number of controls per case.
    alpha : float
        Significance level.
    alternative : {"two-sided", "one-sided"}
    power_target : float, optional
        Desired power; solve for the number of cases when ``n_cases=None``.

    Returns
    -------
    PowerResult

    Notes
    -----
    The control exposure prevalence ``p0`` and the odds ratio imply a case
    exposure prevalence ``p1 = (OR p0) / (1 + p0 (OR - 1))``. Power is then a
    two-proportion comparison between cases (``n_cases``) and controls
    (``ratio * n_cases``).

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.power_case_control(n_cases=200, odds_ratio=2.0,
    ...                             exposure_prevalence=0.3)
    >>> res.design
    'case_control'
    >>> round(float(res.power), 4)
    0.9213
    >>> # Number of cases for 80% power at an odds ratio of 2.0.
    >>> int(sp.power_case_control(odds_ratio=2.0, exposure_prevalence=0.3,
    ...                           power_target=0.8).n)
    138
    """
    if odds_ratio <= 0 or odds_ratio == 1:
        raise ValueError("odds_ratio must be > 0 and != 1.")
    p0 = exposure_prevalence
    if not 0 < p0 < 1:
        raise ValueError("exposure_prevalence must be in (0, 1).")
    p1 = (odds_ratio * p0) / (1 + p0 * (odds_ratio - 1))
    z_a = _z_alpha(alpha, alternative)
    delta = abs(p1 - p0)
    params = dict(
        odds_ratio=odds_ratio,
        exposure_prevalence=p0,
        case_exposure_prevalence=p1,
        ratio=ratio,
        alpha=alpha,
        alternative=alternative,
    )

    def _power_for_cases(nc: np.ndarray) -> np.ndarray:
        n_ctrl = nc * ratio
        se = np.sqrt(p1 * (1 - p1) / nc + p0 * (1 - p0) / n_ctrl)
        out: np.ndarray = norm.cdf(delta / se - z_a)
        return out

    if n_cases is None:
        if power_target is None:
            raise ValueError("Provide either `n_cases` or `power_target`.")
        z_b = norm.ppf(power_target)
        nc0 = (z_a + z_b) ** 2 * (p1 * (1 - p1) + p0 * (1 - p0) / ratio) / delta**2
        nc = np.ceil(nc0)
        while float(_power_for_cases(np.array([nc]))[0]) < power_target:
            nc += 1
        return PowerResult(
            power_val=float(_power_for_cases(np.array([nc]))[0]),
            n=int(nc),
            effect_size=delta,
            design="case_control",
            params=params,
        )

    nc_arr = _as_array(n_cases)
    pwr = _power_for_cases(nc_arr)
    return PowerResult(
        power_val=_scalarize(pwr),
        n=_scalarize(nc_arr),
        effect_size=delta,
        design="case_control",
        params=params,
    )
