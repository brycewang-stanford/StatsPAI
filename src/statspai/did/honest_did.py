"""
Rambachan & Roth (2023) honest parallel trends sensitivity analysis.

Computes robust confidence sets for DID treatment effects under
violations of the parallel trends assumption. Instead of testing
whether pre-trends are exactly zero, this approach asks: "How large
could the parallel trends violation be, and would we still draw the
same conclusion?"

The key parameter M (or Delta) bounds the magnitude of possible
violations: |delta_t - delta_{t-1}| <= M for all t, where delta_t is
the deviation from parallel trends at time t.

References
----------
Rambachan, A. and Roth, J. (2023).
"A More Credible Approach to Parallel Trends."
*Review of Economic Studies*, 90(5), 2555-2591. [@rambachan2023more]

Roth, J. (2022).
"Pretest with Caution: Event-Study Estimates after Testing for
Parallel Trends."
*American Economic Review: Insights*, 4(3), 305-322. [@roth2022pretest]
"""

from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats, optimize

from ..core.results import CausalResult


def honest_did(
    result: CausalResult,
    e: int = 0,
    m_grid: Optional[List[float]] = None,
    method: str = 'smoothness',
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Rambachan & Roth (2023) sensitivity analysis for parallel trends.

    Given a DID result (from Callaway-Sant'Anna or Sun-Abraham),
    computes robust confidence intervals for the treatment effect at
    relative time ``e`` under different magnitudes of parallel trends
    violations.

    Parameters
    ----------
    result : CausalResult
        A fitted DID result that contains event study estimates in
        ``result.model_info['event_study']``.
    e : int, default 0
        Relative time period for which to compute robust CIs.
        0 = on-impact effect, 1 = one period after, etc.
    m_grid : list of float, optional
        Grid of M values (maximum violation magnitude) to evaluate.
        Default: [0, 0.5σ, 1σ, 1.5σ, 2σ] where σ is the SE at e.
    method : str, default 'smoothness'
        Type of restriction on violations:
        - ``'smoothness'``: |Δδ_t| ≤ M (bounded change in violation)
        - ``'relative_magnitude'``: |δ_post| ≤ M̄ × max|δ_pre|
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pd.DataFrame
        Columns: M, ci_lower, ci_upper, rejects_zero.
        Each row is a different M value.

    Examples
    --------
    >>> r = sp.did(df, y='y', treat='g', time='t', id='unit')
    >>> sensitivity = sp.honest_did(r, e=0)
    >>> print(sensitivity)
    >>> # If rejects_zero=True even at M=1σ, result is robust

    Notes
    -----
    The method constructs the identified set for the treatment effect
    under the restriction that parallel trends violations are bounded:

    For ``method='smoothness'``:
        The post-treatment violation δ_post is bounded by extrapolating
        pre-trend violations. If pre-trends show a slope, the post
        period could continue that slope by at most M per period.

        The robust CI at level 1-α is:
        [θ̂ - bias_bound - z_{α/2} × SE,  θ̂ + bias_bound + z_{α/2} × SE]

        where bias_bound = M × (e + 1) is the worst-case bias under
        the smoothness restriction.

    For ``method='relative_magnitude'``:
        The post-treatment violation is bounded as a multiple of the
        largest pre-treatment violation observed. M̄ = 1 means post
        violations can be as large as the largest pre violation.

    See Rambachan & Roth (2023, *ReStud*), Section 2.
    """
    es = _extract_event_study(result)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Find the target period
    target_row = es[es['relative_time'] == e]
    if len(target_row) == 0:
        raise ValueError(f"No event study estimate at relative time e={e}")

    theta_hat = float(target_row['att'].iloc[0])
    se_hat = float(target_row['se'].iloc[0])

    # Pre-treatment estimates (for calibrating M)
    pre = es[es['relative_time'] < 0].sort_values('relative_time')
    pre_atts = pre['att'].values
    pre_ses = pre['se'].values

    # Default M grid: multiples of SE
    if m_grid is None:
        sigma = se_hat
        m_grid = [0, 0.5 * sigma, sigma, 1.5 * sigma, 2.0 * sigma,
                  3.0 * sigma]

    rows = []

    if method == 'smoothness':
        # Smoothness bound: worst-case bias = M × (number of post periods from base)
        # For relative time e, the bias is bounded by M × (e + 1)
        # (each period can drift by at most M)
        n_drift = max(e + 1, 1)

        for M in m_grid:
            bias_bound = M * n_drift
            ci_lower = theta_hat - bias_bound - z_crit * se_hat
            ci_upper = theta_hat + bias_bound + z_crit * se_hat
            rejects = not (ci_lower <= 0 <= ci_upper)

            rows.append({
                'M': round(M, 6),
                'ci_lower': round(ci_lower, 6),
                'ci_upper': round(ci_upper, 6),
                'rejects_zero': rejects,
            })

    elif method == 'relative_magnitude':
        # Relative magnitude: |δ_post| ≤ M̄ × max|δ_pre|
        if len(pre_atts) == 0:
            max_pre = 0
        else:
            max_pre = np.max(np.abs(pre_atts))

        for M_bar in m_grid:
            bias_bound = M_bar * max_pre if max_pre > 0 else M_bar * se_hat
            ci_lower = theta_hat - bias_bound - z_crit * se_hat
            ci_upper = theta_hat + bias_bound + z_crit * se_hat
            rejects = not (ci_lower <= 0 <= ci_upper)

            rows.append({
                'M': round(M_bar, 6),
                'ci_lower': round(ci_lower, 6),
                'ci_upper': round(ci_upper, 6),
                'rejects_zero': rejects,
            })

    else:
        raise ValueError(
            f"method must be 'smoothness' or 'relative_magnitude', "
            f"got '{method}'"
        )

    return pd.DataFrame(rows)


def breakdown_m(
    result: CausalResult,
    e: int = 0,
    method: str = 'smoothness',
    alpha: float = 0.05,
) -> float:
    """
    Compute the breakdown value of M.

    The breakdown M* is the largest violation magnitude under which
    the treatment effect at relative time ``e`` remains statistically
    significant. Larger M* = more robust.

    Parameters
    ----------
    result : CausalResult
        DID result with event study.
    e : int, default 0
        Relative time period.
    method : str, default 'smoothness'
    alpha : float, default 0.05

    Returns
    -------
    float
        Breakdown value M*. The effect is significant for all M < M*.

    Examples
    --------
    >>> m_star = sp.breakdown_m(r, e=0)
    >>> print(f"Breakdown M* = {m_star:.4f}")
    >>> # Interpretation: parallel trends can deviate by up to M* per period
    >>> # and the result remains significant

    Notes
    -----
    Formally, M* = sup{M : 0 ∉ CI(M)}.

    For the smoothness restriction with n_drift periods:
    M* = (|θ̂| - z_{α/2} × SE) / n_drift

    See Rambachan & Roth (2023, *ReStud*), Definition 2.
    """
    es = _extract_event_study(result)

    target = es[es['relative_time'] == e]
    if len(target) == 0:
        raise ValueError(f"No estimate at relative time e={e}")

    theta = float(target['att'].iloc[0])
    se = float(target['se'].iloc[0])
    z_crit = stats.norm.ppf(1 - alpha / 2)

    n_drift = max(e + 1, 1)

    # M* such that |θ̂| - M* × n_drift - z × SE = 0
    m_star = (abs(theta) - z_crit * se) / n_drift
    return max(m_star, 0.0)


# ======================================================================
# Polymorphic event-study extractor
# ======================================================================

def _extract_event_study(result: CausalResult) -> pd.DataFrame:
    """Locate the event-study table inside a CausalResult.

    Accepts three flavours, all of which now surface the same shape:

    1. ``callaway_santanna()`` / ``sun_abraham()`` → event study lives in
       ``result.model_info['event_study']`` (legacy).
    2. ``aggte(cs, type='dynamic')`` → event study **is** the ``detail``
       frame (has ``relative_time`` + ``att`` + ``se``).
    3. ``did_multiplegt()`` → event study in ``model_info['event_study']``
       but placebos carry negative ``relative_time`` and dynamics
       non-negative ``relative_time`` — already compatible.

    Raises
    ------
    ValueError
        If no event study can be found.
    """
    info = result.model_info or {}
    es = info.get('event_study')
    if es is None and getattr(result, 'detail', None) is not None:
        det = result.detail
        if {'relative_time', 'att', 'se'}.issubset(det.columns):
            es = det
    if es is None:
        raise ValueError(
            "Result does not expose an event-study table.  Supported "
            "inputs: callaway_santanna(), sun_abraham(), did_multiplegt(), "
            "or aggte(result, type='dynamic')."
        )
    # Defensive copy — callers mutate / filter it.
    es = es.copy()
    # If aggte produced uniform-band columns but no explicit 'type', ignore.
    required = {'relative_time', 'att', 'se'}
    missing = required - set(es.columns)
    if missing:
        raise ValueError(
            f"Event-study table is missing required columns: {missing}"
        )
    return es


# Citation
CausalResult._CITATIONS['honest_did'] = (
    "@article{rambachan2023more,\n"
    "  title={A More Credible Approach to Parallel Trends},\n"
    "  author={Rambachan, Ashesh and Roth, Jonathan},\n"
    "  journal={Review of Economic Studies},\n"
    "  volume={90},\n"
    "  number={5},\n"
    "  pages={2555--2591},\n"
    "  year={2023},\n"
    "  publisher={Oxford University Press}\n"
    "}"
)
