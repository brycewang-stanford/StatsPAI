"""
E-value: Sensitivity analysis for unmeasured confounding.

The E-value quantifies the minimum strength of association (on the
risk ratio scale) that an unmeasured confounder would need to have
with both treatment and outcome to fully explain away an observed
treatment-outcome association.

    E-value = RR + sqrt(RR * (RR - 1))

where RR is the observed risk ratio (or transformed effect).

For a confidence interval limit:
    E-value_CI = RR_lower + sqrt(RR_lower * (RR_lower - 1))

Key interpretation: if E-value = 3.5, an unmeasured confounder would
need to be associated with both treatment and outcome by a risk ratio
of at least 3.5 (above and beyond measured confounders) to explain
away the observed effect. Higher E-values indicate more robust findings.

References
----------
VanderWeele, T. J. & Ding, P. (2017).
Sensitivity Analysis in Observational Research: Introducing the E-Value.
Annals of Internal Medicine, 167(4), 268-274. [@vanderweele2017sensitivity]

Ding, P. & VanderWeele, T. J. (2016).
Sensitivity Analysis Without Assumptions.
Epidemiology, 27(3), 368-377. [@ding2016sensitivity]
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def evalue(
    estimate: float,
    se: Optional[float] = None,
    ci: Optional[Tuple[float, float]] = None,
    measure: str = 'RR',
    rare_outcome: bool = False,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compute E-value for sensitivity to unmeasured confounding.

    Parameters
    ----------
    estimate : float
        Point estimate of the treatment effect.
        - For measure='RR': risk ratio (must be > 0).
        - For measure='OR': odds ratio (must be > 0).
        - For measure='HR': hazard ratio (must be > 0).
        - For measure='diff': risk difference (between -1 and 1).
        - For measure='SMD': standardised mean difference.
        - For measure='RD': same as 'diff'.
    se : float, optional
        Standard error of the estimate (used to compute CI if ci
        is not provided).
    ci : tuple of (float, float), optional
        Confidence interval (lower, upper). If not provided, computed
        from estimate +/- z * se.
    measure : str, default 'RR'
        Type of effect measure: 'RR', 'OR', 'HR', 'diff', 'RD', 'SMD'.
    rare_outcome : bool, default False
        If True and measure='OR', uses the rare disease approximation
        (OR approx RR). Otherwise converts OR to RR.
    alpha : float, default 0.05
        Significance level for CI if se is provided but ci is not.

    Returns
    -------
    dict
        'evalue_estimate' : float
            E-value for the point estimate.
        'evalue_ci' : float
            E-value for the CI limit closest to the null.
        'rr_estimate' : float
            The risk ratio used (after conversion if needed).
        'rr_ci' : float
            The CI limit risk ratio used.
        'measure' : str
        'interpretation' : str
            Plain-language interpretation.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.evalue(estimate=2.5, measure='RR')
    >>> print(f"E-value: {result['evalue_estimate']:.2f}")

    >>> # From a regression coefficient (SMD)
    >>> result = sp.evalue(estimate=0.5, se=0.1, measure='SMD')
    >>> print(result['interpretation'])

    >>> # Odds ratio with CI
    >>> result = sp.evalue(estimate=1.8, ci=(1.2, 2.7), measure='OR')
    """
    measure = measure.upper()
    valid_measures = {'RR', 'OR', 'HR', 'DIFF', 'RD', 'SMD'}
    if measure not in valid_measures:
        raise ValueError(f"measure must be one of {valid_measures}, got '{measure}'")

    # Compute CI if needed
    if ci is None and se is not None:
        z_crit = sp_stats.norm.ppf(1 - alpha / 2)
        ci = (estimate - z_crit * se, estimate + z_crit * se)

    # Convert to risk ratio scale
    rr = _to_rr(estimate, measure, rare_outcome)

    if ci is not None:
        ci_lower_rr = _to_rr(ci[0], measure, rare_outcome)
        ci_upper_rr = _to_rr(ci[1], measure, rare_outcome)

        # Use the CI limit closest to 1 (the null)
        if rr >= 1:
            rr_ci = ci_lower_rr
        else:
            rr_ci = ci_upper_rr
    else:
        rr_ci = None

    # Compute E-values
    ev_estimate = _compute_evalue(rr)
    ev_ci = _compute_evalue(rr_ci) if rr_ci is not None else None

    # Interpretation
    interpretation = _interpret(ev_estimate, ev_ci, measure)

    result = {
        'evalue_estimate': ev_estimate,
        'evalue_ci': ev_ci,
        'rr_estimate': rr,
        'rr_ci': rr_ci,
        'measure': measure,
        'original_estimate': estimate,
        'interpretation': interpretation,
    }

    if ci is not None:
        result['ci'] = ci

    return result


def evalue_from_result(
    result,
    measure: str = 'SMD',
    rare_outcome: bool = False,
) -> Dict[str, Any]:
    """
    Compute E-value from a StatsPAI CausalResult object.

    Parameters
    ----------
    result : CausalResult
        Result from any StatsPAI causal estimator.
    measure : str, default 'SMD'
        How to interpret the estimate. For most causal estimators
        producing ATE/ATT on continuous outcomes, 'SMD' is appropriate.
    rare_outcome : bool, default False

    Returns
    -------
    dict
        Same as evalue().

    Examples
    --------
    >>> result = sp.did(df, ...)
    >>> ev = sp.evalue_from_result(result)
    >>> print(f"E-value: {ev['evalue_estimate']:.2f}")
    """
    return evalue(
        estimate=result.estimate,
        se=result.se,
        ci=result.ci,
        measure=measure,
        rare_outcome=rare_outcome,
        alpha=result.alpha,
    )


# ======================================================================
# Internals
# ======================================================================

def _to_rr(value, measure, rare_outcome=False):
    """Convert an effect measure to the risk ratio scale."""
    if measure == 'RR' or measure == 'HR':
        rr = value
    elif measure == 'OR':
        if rare_outcome:
            rr = value  # Rare disease: OR ~ RR
        else:
            # Convert OR to RR using the square-root approximation
            # (VanderWeele & Ding 2017, supplementary)
            rr = _or_to_rr_approx(value)
    elif measure in ('DIFF', 'RD'):
        # Risk difference -> approximate RR
        # RR ~ 1 / (1 - RD) for small RD, but for E-value we use
        # the transformation: RR = (1 + sqrt(RD * (1 - p0))) / (1 - sqrt(RD * (1 - p0)))
        # Simplified: treat as approximate RR via exponential
        if abs(value) < 1:
            rr = np.exp(value * 2)  # Rough approximation
        else:
            rr = np.exp(np.sign(value) * 2)
    elif measure == 'SMD':
        # Standardised mean difference -> approximate RR
        # VanderWeele (2017): RR ~ exp(0.91 * SMD)
        rr = np.exp(0.91 * value)
    else:
        rr = value

    # E-value requires RR >= 1 (flip if protective)
    if rr <= 0:
        rr = 1e-10  # Avoid log(0)

    return float(rr)


def _or_to_rr_approx(or_val):
    """
    Convert odds ratio to approximate risk ratio.

    Uses the square-root transformation from VanderWeele & Ding (2017).
    For E-value purposes, OR^{sqrt} is a conservative bound.
    """
    if or_val <= 0:
        return 1e-10
    return float(np.sqrt(or_val))


def _compute_evalue(rr):
    """
    Compute the E-value for a given risk ratio.

    E-value = RR + sqrt(RR * (RR - 1)) for RR >= 1
    E-value = 1/RR + sqrt(1/RR * (1/RR - 1)) for RR < 1
    """
    if rr is None:
        return None

    # Ensure RR >= 1 (use reciprocal for protective effects)
    if rr < 1:
        rr = 1.0 / max(rr, 1e-10)

    if rr <= 1.0:
        return 1.0

    ev = rr + np.sqrt(rr * (rr - 1))
    return float(ev)


def _interpret(ev_est, ev_ci, measure):
    """Generate plain-language interpretation."""
    lines = []

    if ev_est is not None:
        lines.append(
            f"E-value for point estimate: {ev_est:.2f}"
        )
        if ev_est > 3:
            strength = "very robust"
        elif ev_est > 2:
            strength = "moderately robust"
        elif ev_est > 1.5:
            strength = "somewhat robust"
        else:
            strength = "potentially sensitive"

        lines.append(
            f"The observed association is {strength} to unmeasured confounding."
        )
        lines.append(
            f"An unmeasured confounder would need to be associated with both "
            f"treatment and outcome by a risk ratio of at least {ev_est:.2f} "
            f"(each, above and beyond measured confounders) to explain away "
            f"this effect."
        )

    if ev_ci is not None:
        lines.append(
            f"E-value for CI limit: {ev_ci:.2f}"
        )
        if ev_ci <= 1.0:
            lines.append(
                "The confidence interval includes the null, so the E-value "
                "for the CI limit is 1.0 (no unmeasured confounding needed "
                "to shift the CI to include the null)."
            )
        else:
            lines.append(
                f"To move the CI to include the null, an unmeasured "
                f"confounder would need associations of at least {ev_ci:.2f}."
            )

    return "\n".join(lines)
