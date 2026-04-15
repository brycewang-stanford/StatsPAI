"""Tests for the `sp.did(..., aggregation='dynamic'|...)` dispatch path."""
import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(80):
        g = [3, 5, 7, 0][u // 20]
        ui = rng.normal(scale=0.3)
        for t in range(1, 9):
            te = max(0, t - g + 1) * 0.5 if g > 0 else 0
            rows.append({'i': u, 't': t, 'g': g,
                         'y': ui + 0.2 * t + te + rng.normal()})
    return pd.DataFrame(rows)


def test_no_aggregation_returns_cs_result():
    df = _panel()
    r = sp.did(df, y='y', treat='g', time='t', id='i')
    # Without aggregation the result should carry the full (g,t) grid.
    assert len(r.detail) == 21
    assert 'aggregation' not in r.model_info


@pytest.mark.parametrize("agg,expected_rows", [
    ('simple', 1),
    ('dynamic', 11),  # relative_time -6..-1 (excl -1 base) + 0..5
    ('group', 3),
    ('calendar', 6),
])
def test_aggregation_dispatch(agg, expected_rows):
    df = _panel()
    r = sp.did(df, y='y', treat='g', time='t', id='i',
               aggregation=agg, n_boot=100, random_state=0)
    assert r.model_info['aggregation'] == agg
    assert len(r.detail) == expected_rows
    # Aggregated results carry uniform-band columns except 'simple'.
    if agg != 'simple':
        assert {'cband_lower', 'cband_upper'} <= set(r.detail.columns)


def test_invalid_aggregation_raises():
    df = _panel()
    with pytest.raises(ValueError, match='aggregation'):
        sp.did(df, y='y', treat='g', time='t', id='i', aggregation='bogus')


def test_rcs_path_through_top_level():
    """sp.did(..., panel=False) should route through the RCS branch."""
    rng = np.random.default_rng(1)
    rows = []
    for obs in range(400):
        g = int(rng.choice([3, 5, 0], p=[0.3, 0.3, 0.4]))
        t = int(rng.integers(1, 7))
        te = max(0, t - g + 1) * 0.5 if g > 0 else 0
        y = 0.2 * t + te + rng.normal()
        rows.append({'obs': obs, 't': t, 'g': g, 'y': y})
    df = pd.DataFrame(rows)
    r = sp.did(df, y='y', treat='g', time='t', id='obs',
               estimator='reg', panel=False)
    assert r.model_info['panel'] is False
    assert 'RCS' in r.model_info['estimator']


def test_aggregation_with_incompatible_method_raises():
    """Silent-ignore bug: `aggregation=` must be rejected for 2x2 / DDD."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'y': rng.normal(size=200),
        'treat': rng.integers(0, 2, size=200),
        'time': rng.integers(0, 2, size=200),
    })
    with pytest.raises(ValueError, match='Callaway'):
        sp.did(df, y='y', treat='treat', time='time',
               method='2x2', aggregation='dynamic')


def test_anticipation_rejected_for_non_cs_method():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'y': rng.normal(size=200),
        'treat': rng.integers(0, 2, size=200),
        'time': rng.integers(0, 2, size=200),
    })
    with pytest.raises(ValueError, match='anticipation'):
        sp.did(df, y='y', treat='treat', time='time',
               method='2x2', anticipation=1)
