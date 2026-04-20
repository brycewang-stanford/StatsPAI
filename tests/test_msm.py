"""Tests for Marginal Structural Models (IPTW)."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult
from statspai.msm import stabilized_weights


def _make_panel(n=400, T=4, alpha_true=0.5, seed=7):
    """
    Time-varying treatment DGP with time-varying confounding.

        L_{t} | A_{t-1}:  post-treatment confounder drifts with lag-A
        A_t | L_t, A_{t-1}: biased toward units with higher L_t
        Y_T = alpha * sum(A_t) + 0.5*L_0 + baseline noise

    True MSM slope on cumulative A = alpha_true.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        L0 = rng.normal(0, 1)
        V = rng.normal(0, 1)  # baseline
        A_hist = []
        L_hist = [L0]
        for t in range(T):
            L_t = L_hist[-1] + (0.3 * A_hist[-1] if A_hist else 0) + rng.normal(0, 0.3)
            A_t = rng.binomial(
                1, 1 / (1 + np.exp(-(0.4 * L_t + 0.2 * V)))
            )
            A_hist.append(float(A_t))
            L_hist.append(L_t)
        # Outcome at end = linear in cumulative A + baseline
        cum_A = float(np.sum(A_hist))
        Y = alpha_true * cum_A + 0.5 * L0 + 0.3 * V + rng.normal(0, 0.5)
        for t in range(T):
            rows.append({
                'id': i,
                'time': t,
                'A': A_hist[t],
                'L_lag': L_hist[t],  # L measured *before* A_t
                'V': V,
                'Y': Y,
            })
    return pd.DataFrame(rows)


def test_msm_cumulative_slope_recovers_true():
    panel = _make_panel(n=600, T=4, alpha_true=0.5)
    result = sp.msm(
        panel, y='Y', treat='A',
        id='id', time='time',
        time_varying=['L_lag'],
        baseline=['V'],
        exposure='cumulative',
    )
    assert isinstance(result, CausalResult)
    # True cumulative-A slope = 0.5
    assert abs(result.estimate - 0.5) < 0.2


def test_stabilized_weights_shape_and_mean():
    panel = _make_panel(n=300, T=3)
    sw = stabilized_weights(
        panel, treat='A', id='id', time='time',
        time_varying=['L_lag'], baseline=['V'], treat_type='binary',
    )
    assert len(sw) == len(panel)
    # Stabilized weights should have mean ~1 under correct specification
    assert 0.5 < float(np.mean(sw)) < 2.0


def test_msm_ever_exposure():
    panel = _make_panel(n=500, T=4, alpha_true=0.5)
    result = sp.msm(
        panel, y='Y', treat='A',
        id='id', time='time',
        time_varying=['L_lag'],
        baseline=['V'],
        exposure='ever',
    )
    # Ever-treated should give a positive effect (though scale differs from slope)
    assert result.estimate > 0
    assert result.pvalue < 0.1  # loose — small DGP


def test_msm_rejects_ever_for_continuous():
    rng = np.random.default_rng(0)
    n, T = 50, 3
    rows = []
    for i in range(n):
        for t in range(T):
            rows.append({
                'id': i, 'time': t,
                'A': rng.normal(),  # continuous
                'L_lag': rng.normal(),
                'V': rng.normal(),
                'Y': rng.normal(),
            })
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match='binary'):
        sp.msm(df, y='Y', treat='A', id='id', time='time',
               time_varying=['L_lag'], baseline=['V'], exposure='ever')


def test_msm_reports_weight_diagnostics():
    panel = _make_panel(n=200, T=3)
    result = sp.msm(
        panel, y='Y', treat='A',
        id='id', time='time',
        time_varying=['L_lag'], baseline=['V'],
    )
    assert 'sw_mean' in result.model_info
    assert 'sw_max' in result.model_info
    assert 'sw_min' in result.model_info
    assert result.model_info['sw_max'] >= result.model_info['sw_min']
