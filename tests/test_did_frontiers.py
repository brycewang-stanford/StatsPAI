"""Smoke tests for v0.10 staggered DiD frontiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def staggered_panel():
    rng = np.random.default_rng(42)
    n_units, n_time = 20, 10
    rows = []
    for u in range(n_units):
        first_treat = 0 if u < 8 else (4 if u < 14 else 6)  # 3 cohorts
        for t in range(n_time):
            post = first_treat > 0 and t >= first_treat
            x = rng.standard_normal()
            y = (
                u * 0.1 + t * 0.2
                + (2.0 if post else 0.0)
                + 0.3 * x
                + rng.standard_normal()
            )
            rows.append({
                'unit': f'u{u}', 'time': t, 'y': y, 'x': x,
                'first_treat': first_treat,
            })
    return pd.DataFrame(rows)


def test_did_bcf(staggered_panel):
    res = sp.did_bcf(
        staggered_panel, y='y', treat='first_treat', time='time',
        id='unit', covariates=['x'], n_trees=20, seed=0,
    )
    assert hasattr(res, 'estimate')
    # True ATT ≈ 2.0
    assert -1.0 < res.estimate < 5.0
    assert res.method.startswith("DiD-BCF")
    assert 'catt_by_cohort' in res.model_info


def test_cohort_anchored(staggered_panel):
    res = sp.cohort_anchored_event_study(
        staggered_panel, y='y', treat='first_treat', time='time',
        id='unit', leads=2, lags=2,
    )
    assert hasattr(res, 'estimate')
    es = res.model_info['event_study']
    assert isinstance(es, pd.DataFrame)
    assert 'rel_time' in es.columns
    # Should produce post-treatment effects in the right ballpark
    assert -1.0 < res.estimate < 5.0


def test_design_robust(staggered_panel):
    res = sp.design_robust_event_study(
        staggered_panel, y='y', treat='first_treat', time='time',
        id='unit', leads=2, lags=2,
    )
    assert hasattr(res, 'estimate')
    es = res.model_info['event_study']
    assert isinstance(es, pd.DataFrame)
    assert 'diagnostics' in res.model_info


def test_did_misclassified_no_correction(staggered_panel):
    # pi_misclass=0, anticipation=0 → should reproduce naive ATT
    res = sp.did_misclassified(
        staggered_panel, y='y', treat='first_treat', time='time',
        id='unit', pi_misclass=0.0, anticipation_periods=0,
    )
    assert hasattr(res, 'estimate')
    assert abs(res.model_info['naive_att'] - res.estimate) < 1e-6


def test_did_misclassified_with_correction(staggered_panel):
    # pi_misclass=0.1 → corrected estimate larger in magnitude
    res = sp.did_misclassified(
        staggered_panel, y='y', treat='first_treat', time='time',
        id='unit', pi_misclass=0.1, anticipation_periods=1,
    )
    assert res.model_info['misclass_factor'] > 1.0


def test_did_misclassified_invalid_pi():
    df = pd.DataFrame({
        'y': np.random.randn(20), 'first_treat': [0]*10 + [3]*10,
        'time': list(range(10)) * 2, 'unit': [f'u{i}' for i in range(2) for _ in range(10)],
    })
    with pytest.raises(ValueError, match="pi_misclass"):
        sp.did_misclassified(df, y='y', treat='first_treat', time='time',
                              id='unit', pi_misclass=0.6)
