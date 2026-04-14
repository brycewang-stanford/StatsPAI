"""Tests for aggte() ↔ honest_did() integration."""
import numpy as np
import pandas as pd
import pytest

from statspai.did import (
    aggte, breakdown_m, callaway_santanna, honest_did, sun_abraham,
)


@pytest.fixture(scope='module')
def cs_result():
    rng = np.random.default_rng(42)
    rows = []
    for u in range(120):
        g = [3, 5, 7, 0][u // 30]
        ui = rng.normal(scale=0.3)
        for t in range(1, 9):
            te = max(0, t - g + 1) * 0.5 if g > 0 else 0
            rows.append({'i': u, 't': t, 'g': g,
                         'y': ui + 0.2 * t + te + rng.normal()})
    df = pd.DataFrame(rows)
    return callaway_santanna(df, y='y', g='g', t='t', i='i')


def test_honest_did_accepts_aggte_dynamic(cs_result):
    """honest_did should work on aggte(type='dynamic') output."""
    es = aggte(cs_result, type='dynamic', random_state=0, n_boot=300)
    s = honest_did(es, e=1)
    assert {'M', 'ci_lower', 'ci_upper', 'rejects_zero'} <= set(s.columns)
    assert len(s) > 0
    assert (s['ci_upper'] >= s['ci_lower']).all()


def test_breakdown_m_accepts_aggte_dynamic(cs_result):
    es = aggte(cs_result, type='dynamic', random_state=0, n_boot=300)
    m_star = breakdown_m(es, e=1)
    assert m_star > 0.0
    assert np.isfinite(m_star)


def test_honest_did_legacy_still_works(cs_result):
    """CS result with event_study in model_info still supported."""
    s = honest_did(cs_result, e=1)
    assert len(s) > 0
    assert s['rejects_zero'].iloc[0]  # M=0 rejects zero


def test_legacy_and_aggte_agree_at_point_estimate(cs_result):
    """With M=0 both paths reduce to the pointwise CI; differ only by
    bootstrap vs analytic SE."""
    s_cs = honest_did(cs_result, e=1)
    es = aggte(cs_result, type='dynamic', random_state=0, n_boot=500)
    s_es = honest_did(es, e=1)

    # M=0 row's midpoint = theta_hat in both cases.
    mid_cs = 0.5 * (s_cs['ci_lower'].iloc[0] + s_cs['ci_upper'].iloc[0])
    mid_es = 0.5 * (s_es['ci_lower'].iloc[0] + s_es['ci_upper'].iloc[0])
    assert mid_cs == pytest.approx(mid_es, rel=1e-6, abs=1e-6)


def test_honest_did_missing_event_study_raises():
    """A CausalResult without an event study should be rejected."""
    from statspai.core.results import CausalResult
    dummy = CausalResult(
        method='fake', estimand='ATT', estimate=1.0, se=0.1,
        pvalue=0.01, ci=(0.8, 1.2), alpha=0.05, n_obs=100,
    )
    with pytest.raises(ValueError, match='event-study'):
        honest_did(dummy, e=0)


def test_missing_relative_time_e_raises(cs_result):
    es = aggte(cs_result, type='dynamic', max_e=2,
               random_state=0, n_boot=200)
    # e=5 has been truncated by max_e=2
    with pytest.raises(ValueError, match='relative time'):
        honest_did(es, e=5)


def test_sun_abraham_to_honest_did(cs_result):
    """Sun-Abraham's event study also feeds honest_did."""
    df = cs_result.detail  # reuse fixture's raw data path through sa
    # Rebuild raw data from fixture inputs — easier to just rerun sa.
    rng = np.random.default_rng(42)
    rows = []
    for u in range(120):
        g = [3, 5, 7, 0][u // 30]
        ui = rng.normal(scale=0.3)
        for t in range(1, 9):
            te = max(0, t - g + 1) * 0.5 if g > 0 else 0
            rows.append({'i': u, 't': t, 'g': g,
                         'y': ui + 0.2 * t + te + rng.normal()})
    panel = pd.DataFrame(rows)
    sa = sun_abraham(panel, y='y', g='g', t='t', i='i')
    # sun_abraham attaches event_study both on detail and on model_info
    s = honest_did(sa, e=1)
    assert len(s) > 0
