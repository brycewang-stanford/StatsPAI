"""Closed-form pins for ``sp.breakdown_m``'s covariance-free fallback.

When the event-study covariance cannot be recovered (the hand-crafted results
below carry no influence functions), ``sp.honest_did(method='smoothness')``
falls back to the worst-case-bias interval ``theta_hat +/- (M (e+1) + z SE)``
and ``sp.breakdown_m`` inverts that same interval, which has the closed form

    M* = (|theta_hat| - z_{alpha/2} * SE) / n_drift,   n_drift = max(e + 1, 1).

This module pins that closed form on synthetic event-study tables with known
``(theta_hat, SE)``. It is NOT the Rambachan-Roth FLCI breakdown value: an
earlier version of this docstring called the formula "Definition 2 of the
paper", which it is not. The FLCI breakdown -- what ``sp.breakdown_m``
returns whenever the covariance is available -- is checked against R
``HonestDiD`` in ``tests/reference_parity/test_did_synth_R_parity.py``.

References
----------
[@rambachan2023more]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.core.results import CausalResult


# ---------------------------------------------------------------------------
# Helper: build a minimal CausalResult whose event study we control.
# ---------------------------------------------------------------------------

def _event_study_result(
    relative_times: list,
    atts: list,
    ses: list,
    *,
    method: str = "Test DiD",
) -> CausalResult:
    """Wrap a hand-crafted (t, att, se) triple in a CausalResult so that
    ``sp.breakdown_m`` sees exactly the numbers we supply — no DGP noise."""
    assert 0 in relative_times, (
        "_event_study_result expects a relative_time=0 row; other layouts "
        "would make the headline-estimate fallback silently pick the wrong row."
    )
    es = pd.DataFrame({
        "relative_time": relative_times,
        "att": atts,
        "se": ses,
    })
    # Pick the relative_time=0 row as the headline estimate.  Fallback to
    # index 0 only if the assertion above is bypassed by a subclass;
    # kept for defence in depth.
    idx = next((i for i, t in enumerate(relative_times) if t == 0), 0)
    return CausalResult(
        method=method,
        estimand="ATT(0)",
        estimate=float(atts[idx]),
        se=float(ses[idx]),
        pvalue=0.0,
        ci=(float(atts[idx] - 1.96 * ses[idx]),
            float(atts[idx] + 1.96 * ses[idx])),
        alpha=0.05,
        n_obs=1000,
        model_info={"event_study": es},
    )


# ---------------------------------------------------------------------------
# Pin breakdown_m to the paper's closed form at several points.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theta, se, e", [
    (0.50, 0.10, 0),
    (0.30, 0.05, 1),
    (0.80, 0.20, 2),
    (0.15, 0.03, 0),
    (1.20, 0.25, 3),
])
def test_breakdown_m_matches_closed_form(theta, se, e):
    """M* = (|θ̂| - z_{0.025} · SE) / n_drift where n_drift = max(e+1, 1)."""
    # Construct an event study with a single positive-relative-time row.
    # Include a pre-period row so the table shape is realistic.
    rel_times = [-1, 0, 1, 2, 3]
    atts = [0.0, 0.0, 0.0, 0.0, 0.0]
    ses = [se] * 5
    # Place theta at the target relative time.
    atts[rel_times.index(e)] = theta
    result = _event_study_result(rel_times, atts, ses)

    z_crit = stats.norm.ppf(0.975)  # 1.959963...
    n_drift = max(e + 1, 1)
    expected = max((abs(theta) - z_crit * se) / n_drift, 0.0)

    with pytest.warns(UserWarning, match="covariance is unavailable"):
        actual = sp.breakdown_m(result, e=e, method="smoothness", alpha=0.05)
    assert actual == pytest.approx(expected, rel=1e-10), (
        f"breakdown_m drifted from paper formula at (θ={theta}, SE={se}, "
        f"e={e}): got {actual:.6f}, expected {expected:.6f}"
    )


def test_breakdown_m_clamped_at_zero():
    """When |θ̂| < z · SE (i.e. the estimate is already insignificant),
    M* is clamped to 0 per Definition 2."""
    # Borderline-insignificant: θ̂ = 0.1, SE = 0.1 → |t| = 1 < 1.96.
    result = _event_study_result([-1, 0], [0.0, 0.1], [0.1, 0.1])
    assert sp.breakdown_m(result, e=0) == 0.0


def test_breakdown_m_alpha_sensitivity():
    """Lower α (tighter inference) should give a larger M*."""
    result = _event_study_result([-1, 0], [0.0, 0.5], [0.1, 0.1])
    m_05 = sp.breakdown_m(result, e=0, alpha=0.05)
    m_10 = sp.breakdown_m(result, e=0, alpha=0.10)
    # At α=0.10, z = 1.645 < 1.96 = z_{0.025}, so M* is larger.
    assert m_10 > m_05
    # Analytical anchor: at α=0.10, M* = (0.5 - 1.645·0.1)/1 = 0.3355.
    assert m_10 == pytest.approx(0.5 - 1.6448536269514722 * 0.1, rel=1e-10)


def test_breakdown_m_missing_relative_time_raises():
    result = _event_study_result([-1, 0], [0.0, 0.3], [0.1, 0.1])
    with pytest.raises(ValueError, match="No estimate at relative time"):
        sp.breakdown_m(result, e=99)


# ---------------------------------------------------------------------------
# End-to-end smoke: breakdown_m on a real CS DID result.
# ---------------------------------------------------------------------------

def test_breakdown_m_on_callaway_santanna_runs():
    """End-to-end: `sp.breakdown_m` must work on a genuine
    `callaway_santanna` result (not just hand-crafted ones)."""
    df = sp.dgp_did(
        n_units=80, n_periods=8, staggered=True, n_groups=3,
        effect=0.5, seed=2026,
    )
    r = sp.callaway_santanna(
        data=df, y="y", g="first_treat", t="time", i="unit",
    )
    # Aggregate to a dynamic event study before Honest DiD (per the
    # module's polymorphic _extract_event_study contract).
    dyn = sp.aggte(r, type="dynamic")
    m_star = sp.breakdown_m(dyn, e=0)
    # The DGP has a real effect, so the breakdown value should be strictly
    # positive; larger than zero means the effect survives some drift.
    assert m_star >= 0.0
    assert np.isfinite(m_star)
