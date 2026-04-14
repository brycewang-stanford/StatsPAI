"""Tests for dCDH joint placebo + avg cumulative dynamic effect (2024)."""
import numpy as np
import pandas as pd
import pytest

from statspai.did import did_multiplegt


def _make_on_off_panel(n_units=60, n_periods=8, effect=0.6, seed=0):
    """Treatment that can switch on and off (staggered but with flips)."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        switch_on = rng.integers(3, 7)
        switch_off = rng.integers(switch_on + 1, n_periods + 2)
        ui = rng.normal(scale=0.3)
        for t in range(1, n_periods + 1):
            d = int(switch_on <= t < switch_off)
            y = ui + 0.15 * t + effect * d + rng.normal()
            rows.append({'g': u, 't': t, 'd': d, 'y': y})
    return pd.DataFrame(rows)


@pytest.fixture(scope='module')
def dcdh_result():
    df = _make_on_off_panel(seed=1)
    return did_multiplegt(
        df, y='y', group='g', time='t', treatment='d',
        placebo=2, dynamic=3, n_boot=150, seed=42,
    )


def test_joint_placebo_test_shape(dcdh_result):
    jp = dcdh_result.model_info['joint_placebo_test']
    assert jp is not None
    assert 'statistic' in jp and 'df' in jp and 'pvalue' in jp
    assert jp['df'] == 2
    assert 0.0 <= jp['pvalue'] <= 1.0
    assert jp['statistic'] >= 0.0


def test_joint_placebo_absent_without_placebos():
    df = _make_on_off_panel(seed=2)
    r = did_multiplegt(df, y='y', group='g', time='t', treatment='d',
                       placebo=0, dynamic=1, n_boot=50, seed=0)
    assert r.model_info['joint_placebo_test'] is None


def test_avg_cumulative_effect_structure(dcdh_result):
    avg = dcdh_result.model_info['avg_cumulative_effect']
    assert avg is not None
    assert set(avg.keys()) >= {
        'estimate', 'se', 'ci_lower', 'ci_upper', 'pvalue', 'n_horizons',
    }
    assert avg['n_horizons'] == 4  # dynamic=3 → horizons 0..3
    assert np.isfinite(avg['estimate'])
    assert np.isfinite(avg['se']) and avg['se'] > 0
    # CI is symmetric around estimate.
    assert avg['ci_upper'] - avg['estimate'] == pytest.approx(
        avg['estimate'] - avg['ci_lower'], rel=1e-8
    )


def test_avg_cumulative_matches_mean_of_horizon_estimates(dcdh_result):
    horizons = [d['estimate'] for d in dcdh_result.model_info['dynamic']]
    avg = dcdh_result.model_info['avg_cumulative_effect']['estimate']
    assert avg == pytest.approx(float(np.mean(horizons)), rel=1e-8)


def test_power_when_true_effect_present():
    """With effect=2.0 the avg cumulative CI should exclude zero."""
    df = _make_on_off_panel(n_units=120, effect=2.0, seed=7)
    r = did_multiplegt(
        df, y='y', group='g', time='t', treatment='d',
        placebo=1, dynamic=2, n_boot=300, seed=7,
    )
    avg = r.model_info['avg_cumulative_effect']
    assert avg['pvalue'] < 0.01
    assert avg['ci_lower'] > 0
