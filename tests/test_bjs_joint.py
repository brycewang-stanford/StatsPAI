"""Tests for the cluster-bootstrap joint pre-trend Wald on BJS."""
import numpy as np
import pandas as pd
import pytest

from statspai.did import bjs_pretrend_joint, did_imputation


def _panel(
    n_units=80, n_periods=8, pretrend_slope=0.0, seed=0,
):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        first = int([3, 5, 7, n_periods + 1][u // (n_units // 4)])
        ui = rng.normal(scale=0.3)
        for t in range(1, n_periods + 1):
            te = max(0, t - first + 1) * 0.5 if first <= n_periods else 0
            # Optional pre-trend for the eventually-treated:
            pre = (pretrend_slope * t
                   if first <= n_periods and t < first else 0.0)
            y = ui + 0.15 * t + pre + te + rng.normal()
            rows.append({'i': u, 't': t, 'g': first, 'y': y})
    return pd.DataFrame(rows)


@pytest.fixture(scope='module')
def bjs_no_pretrend():
    df = _panel(seed=11, pretrend_slope=0.0)
    r = did_imputation(df, y='y', group='i', time='t', first_treat='g',
                       horizon=list(range(-4, 4)))
    return df, r


def test_joint_wald_returns_expected_fields(bjs_no_pretrend):
    df, r = bjs_no_pretrend
    out = bjs_pretrend_joint(
        r, data=df, y='y', group='i', time='t', first_treat='g',
        horizon=list(range(-4, 4)), n_boot=30, seed=0,
    )
    for key in ('statistic', 'df', 'pvalue', 'method', 'n_boot', 'pre_cov'):
        assert key in out
    assert out['method'] == 'cluster-bootstrap'
    assert out['statistic'] >= 0
    assert 0.0 <= out['pvalue'] <= 1.0
    # df should equal the number of pre-period horizons.
    assert out['df'] == 4  # horizons -4, -3, -2, -1


def test_joint_wald_uses_proper_covariance_matrix(bjs_no_pretrend):
    df, r = bjs_no_pretrend
    out = bjs_pretrend_joint(
        r, data=df, y='y', group='i', time='t', first_treat='g',
        horizon=list(range(-4, 4)), n_boot=40, seed=1,
    )
    cov = out['pre_cov']
    assert cov.shape == (4, 4)
    # Covariance matrix should be symmetric PSD-ish.
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)
    eigs = np.linalg.eigvalsh(cov)
    # Allow tiny negative eigenvalues from bootstrap noise.
    assert eigs.min() >= -1e-8


def test_no_event_study_raises():
    df = _panel(seed=5, pretrend_slope=0.0)
    r = did_imputation(df, y='y', group='i', time='t', first_treat='g')
    # No horizon → no event_study table → informative error.
    with pytest.raises(ValueError, match='event-study'):
        bjs_pretrend_joint(
            r, data=df, y='y', group='i', time='t', first_treat='g',
            n_boot=10, seed=0,
        )


def test_missing_cluster_column_raises(bjs_no_pretrend):
    df, r = bjs_no_pretrend
    with pytest.raises(ValueError, match='not found'):
        bjs_pretrend_joint(
            r, data=df, y='y', group='i', time='t', first_treat='g',
            cluster='nonexistent', horizon=list(range(-4, 4)),
            n_boot=5, seed=0,
        )


def test_joint_wald_detects_genuine_pretrend():
    """Large pre-trend slope → joint Wald should reject strongly."""
    df = _panel(n_units=160, pretrend_slope=0.5, seed=99)
    r = did_imputation(df, y='y', group='i', time='t', first_treat='g',
                       horizon=list(range(-4, 4)))
    out = bjs_pretrend_joint(
        r, data=df, y='y', group='i', time='t', first_treat='g',
        horizon=list(range(-4, 4)), n_boot=60, seed=99,
    )
    assert out['pvalue'] < 0.05, (
        f"expected joint Wald to reject under a 0.5-slope pretrend; "
        f"got p = {out['pvalue']}"
    )
