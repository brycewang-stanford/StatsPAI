"""
Tests for Phase 9-14: E-value, Dose-Response, Bounds, Interference,
Dynamic Treatment Regimes, Multi-valued Treatment.
"""

import pytest
import numpy as np
import pandas as pd

from statspai.diagnostics.evalue import evalue, evalue_from_result
from statspai.dose_response import dose_response, DoseResponse
from statspai.bounds import lee_bounds, manski_bounds
from statspai.interference import spillover, SpilloverEstimator
from statspai.dtr import g_estimation, GEstimation
from statspai.multi_treatment import multi_treatment, MultiTreatment
from statspai.core.results import CausalResult


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def binary_data():
    rng = np.random.default_rng(42)
    n = 1000
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n).astype(float)
    Y = 2.0 * D + X1 + rng.normal(0, 0.5, n)
    return pd.DataFrame({'y': Y, 'd': D, 'x1': X1, 'x2': X2})


@pytest.fixture
def continuous_treat_data():
    rng = np.random.default_rng(42)
    n = 1000
    X1 = rng.normal(0, 1, n)
    T = 0.5 * X1 + rng.normal(0, 1, n)  # continuous treatment
    Y = 2.0 * T - 0.3 * T ** 2 + X1 + rng.normal(0, 0.5, n)
    return pd.DataFrame({'y': Y, 't': T, 'x1': X1})


@pytest.fixture
def selection_data():
    """Data with sample selection (some outcomes missing)."""
    rng = np.random.default_rng(42)
    n = 1000
    D = rng.binomial(1, 0.5, n).astype(float)
    X1 = rng.normal(0, 1, n)
    # Selection: treated units more likely to be observed
    sel_prob = 0.8 + 0.1 * D - 0.05 * X1
    sel_prob = np.clip(sel_prob, 0.3, 0.95)
    S = rng.binomial(1, sel_prob, n).astype(float)
    Y = 2.0 * D + X1 + rng.normal(0, 0.5, n)
    Y_obs = np.where(S == 1, Y, np.nan)
    return pd.DataFrame({
        'y': Y_obs, 'd': D, 'x1': X1, 'selected': S
    })


@pytest.fixture
def cluster_data():
    """Clustered data with spillover effects."""
    rng = np.random.default_rng(42)
    rows = []
    for cluster_id in range(50):
        cluster_size = rng.integers(4, 10)
        # Cluster-level treatment assignment
        p_treat = rng.uniform(0.2, 0.8)
        for i in range(cluster_size):
            d = rng.binomial(1, p_treat)
            peer_frac = p_treat  # proxy for peer exposure
            y = 1.5 * d + 0.8 * peer_frac + rng.normal(0, 0.5)
            rows.append({
                'y': y, 'd': float(d), 'cluster': cluster_id,
                'x1': rng.normal(0, 1)
            })
    return pd.DataFrame(rows)


@pytest.fixture
def dtr_data():
    """Two-stage DTR data."""
    rng = np.random.default_rng(42)
    n = 500
    X1 = rng.normal(0, 1, n)
    A1 = rng.binomial(1, 0.5, n).astype(float)
    X2 = X1 + 0.5 * A1 + rng.normal(0, 0.5, n)
    A2 = rng.binomial(1, 0.5, n).astype(float)
    Y = 2.0 * A1 + 1.5 * A2 + X1 + X2 + rng.normal(0, 0.5, n)
    return pd.DataFrame({
        'y': Y, 'a1': A1, 'a2': A2, 'x1': X1, 'x2': X2
    })


@pytest.fixture
def multi_treat_data():
    """Data with 3-level treatment."""
    rng = np.random.default_rng(42)
    n = 1500
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    # Three treatment levels: 0, 1, 2
    D = rng.choice([0, 1, 2], size=n, p=[0.4, 0.3, 0.3])
    Y = 1.5 * (D == 1) + 3.0 * (D == 2) + X1 + rng.normal(0, 0.5, n)
    return pd.DataFrame({'y': Y, 'd': D, 'x1': X1, 'x2': X2})


# ======================================================================
# E-value Tests
# ======================================================================

class TestEvalue:

    def test_basic_rr(self):
        result = evalue(estimate=2.5, measure='RR')
        assert result['evalue_estimate'] > 1.0
        assert 'interpretation' in result

    def test_rr_with_ci(self):
        result = evalue(estimate=2.0, ci=(1.5, 2.7), measure='RR')
        assert result['evalue_ci'] is not None
        assert result['evalue_ci'] > 1.0
        assert result['evalue_ci'] <= result['evalue_estimate']

    def test_smd(self):
        result = evalue(estimate=0.5, se=0.1, measure='SMD')
        assert result['evalue_estimate'] > 1.0
        assert result['evalue_ci'] is not None

    def test_or(self):
        result = evalue(estimate=3.0, measure='OR')
        assert result['evalue_estimate'] > 1.0

    def test_protective_effect(self):
        """RR < 1 should still produce valid E-value."""
        result = evalue(estimate=0.5, measure='RR')
        assert result['evalue_estimate'] > 1.0

    def test_null_effect(self):
        """RR = 1 gives E-value = 1."""
        result = evalue(estimate=1.0, measure='RR')
        assert result['evalue_estimate'] == 1.0

    def test_from_result(self, binary_data):
        """Should work with CausalResult objects."""
        from statspai import did_2x2
        # Create a simple CausalResult
        cr = CausalResult(
            method='test', estimand='ATE',
            estimate=0.5, se=0.1, pvalue=0.01,
            ci=(0.3, 0.7), alpha=0.05, n_obs=100,
        )
        result = evalue_from_result(cr, measure='SMD')
        assert result['evalue_estimate'] > 1.0

    def test_invalid_measure_raises(self):
        with pytest.raises(ValueError, match="measure must be"):
            evalue(estimate=2.0, measure='INVALID')

    def test_import_from_statspai(self):
        import statspai as sp
        assert hasattr(sp, 'evalue')
        assert hasattr(sp, 'evalue_from_result')


# ======================================================================
# Dose-Response Tests
# ======================================================================

class TestDoseResponse:

    def test_returns_causal_result(self, continuous_treat_data):
        result = dose_response(continuous_treat_data, y='y', treat='t',
                               covariates=['x1'], n_dose_points=10,
                               n_bootstrap=30)
        assert isinstance(result, CausalResult)
        assert 'Dose-Response' in result.method

    def test_has_curve(self, continuous_treat_data):
        result = dose_response(continuous_treat_data, y='y', treat='t',
                               covariates=['x1'], n_dose_points=10,
                               n_bootstrap=30)
        assert result.detail is not None
        assert 'dose' in result.detail.columns
        assert 'response' in result.detail.columns
        assert len(result.detail) == 10

    def test_has_marginal_effect(self, continuous_treat_data):
        result = dose_response(continuous_treat_data, y='y', treat='t',
                               covariates=['x1'], n_dose_points=10,
                               n_bootstrap=30)
        assert 'avg_marginal_effect' in result.model_info

    def test_confidence_bands(self, continuous_treat_data):
        result = dose_response(continuous_treat_data, y='y', treat='t',
                               covariates=['x1'], n_dose_points=10,
                               n_bootstrap=30)
        assert 'ci_lower' in result.detail.columns
        assert 'ci_upper' in result.detail.columns
        assert np.all(result.detail['ci_lower'] <= result.detail['ci_upper'])

    def test_import(self):
        import statspai as sp
        assert hasattr(sp, 'dose_response')


# ======================================================================
# Bounds Tests
# ======================================================================

class TestLeeBounds:

    def test_returns_causal_result(self, selection_data):
        result = lee_bounds(selection_data, y='y', treat='d',
                            selection='selected', n_bootstrap=50)
        assert isinstance(result, CausalResult)
        assert 'Lee' in result.method

    def test_has_bounds(self, selection_data):
        result = lee_bounds(selection_data, y='y', treat='d',
                            selection='selected', n_bootstrap=50)
        lb = result.model_info['lower_bound']
        ub = result.model_info['upper_bound']
        assert lb <= ub

    def test_midpoint_estimate(self, selection_data):
        result = lee_bounds(selection_data, y='y', treat='d',
                            selection='selected', n_bootstrap=50)
        lb = result.model_info['lower_bound']
        ub = result.model_info['upper_bound']
        assert abs(result.estimate - (lb + ub) / 2) < 1e-10


class TestManskiBounds:

    def test_returns_causal_result(self, binary_data):
        result = manski_bounds(binary_data, y='y', treat='d',
                               n_bootstrap=50)
        assert isinstance(result, CausalResult)
        assert 'Manski' in result.method

    def test_has_bounds(self, binary_data):
        result = manski_bounds(binary_data, y='y', treat='d',
                               n_bootstrap=50)
        lb = result.model_info['lower_bound']
        ub = result.model_info['upper_bound']
        assert lb <= ub

    def test_mtr_tighter(self, binary_data):
        r_none = manski_bounds(binary_data, y='y', treat='d',
                               assumption='none', n_bootstrap=30)
        r_mtr = manski_bounds(binary_data, y='y', treat='d',
                              assumption='mtr', n_bootstrap=30)
        # MTR should give tighter (or equal) bounds
        w_none = r_none.model_info['bound_width']
        w_mtr = r_mtr.model_info['bound_width']
        assert w_mtr <= w_none + 1e-6

    def test_import(self):
        import statspai as sp
        assert hasattr(sp, 'lee_bounds')
        assert hasattr(sp, 'manski_bounds')


# ======================================================================
# Interference Tests
# ======================================================================

class TestSpillover:

    def test_returns_causal_result(self, cluster_data):
        result = spillover(cluster_data, y='y', treat='d',
                           cluster='cluster', n_bootstrap=50)
        assert isinstance(result, CausalResult)
        assert 'Spillover' in result.method

    def test_decomposes_effects(self, cluster_data):
        result = spillover(cluster_data, y='y', treat='d',
                           cluster='cluster', n_bootstrap=50)
        assert 'direct_effect' in result.model_info
        assert 'spillover_effect' in result.model_info
        assert 'total_effect' in result.model_info

    def test_detail_has_three_effects(self, cluster_data):
        result = spillover(cluster_data, y='y', treat='d',
                           cluster='cluster', n_bootstrap=50)
        assert len(result.detail) == 3
        types = set(result.detail['effect_type'])
        assert types == {'Direct', 'Spillover', 'Total'}

    def test_exposure_functions(self, cluster_data):
        for fn in ['fraction', 'any', 'count']:
            result = spillover(cluster_data, y='y', treat='d',
                               cluster='cluster', exposure_fn=fn,
                               n_bootstrap=30)
            assert isinstance(result, CausalResult)

    def test_import(self):
        import statspai as sp
        assert hasattr(sp, 'spillover')


# ======================================================================
# DTR Tests
# ======================================================================

class TestGEstimation:

    def test_returns_causal_result(self, dtr_data):
        result = g_estimation(
            dtr_data, y='y',
            treatments=['a1', 'a2'],
            covariates_by_stage=[['x1'], ['x1', 'x2']],
            n_bootstrap=50,
        )
        assert isinstance(result, CausalResult)
        assert 'G-Estimation' in result.method

    def test_has_stage_estimates(self, dtr_data):
        result = g_estimation(
            dtr_data, y='y',
            treatments=['a1', 'a2'],
            covariates_by_stage=[['x1'], ['x1', 'x2']],
            n_bootstrap=50,
        )
        assert result.detail is not None
        assert len(result.detail) == 2
        assert 'blip_estimate' in result.detail.columns

    def test_positive_blips(self, dtr_data):
        """Both treatment stages have positive true effects."""
        result = g_estimation(
            dtr_data, y='y',
            treatments=['a1', 'a2'],
            covariates_by_stage=[['x1'], ['x1', 'x2']],
            n_bootstrap=50,
        )
        psis = result.model_info['psi_estimates']
        # Both should be positive (true effects are 2.0 and 1.5)
        assert psis[0] > 0
        assert psis[1] > 0

    def test_mismatched_stages_raises(self, dtr_data):
        with pytest.raises(ValueError, match="entries"):
            g_estimation(
                dtr_data, y='y',
                treatments=['a1', 'a2'],
                covariates_by_stage=[['x1']],  # only 1, need 2
            )

    def test_import(self):
        import statspai as sp
        assert hasattr(sp, 'g_estimation')


# ======================================================================
# Multi-valued Treatment Tests
# ======================================================================

class TestMultiTreatment:

    def test_returns_causal_result(self, multi_treat_data):
        result = multi_treatment(multi_treat_data, y='y', treat='d',
                                 covariates=['x1', 'x2'],
                                 n_bootstrap=50)
        assert isinstance(result, CausalResult)
        assert 'Multi-valued' in result.method

    def test_pairwise_effects(self, multi_treat_data):
        result = multi_treatment(multi_treat_data, y='y', treat='d',
                                 covariates=['x1', 'x2'],
                                 n_bootstrap=50)
        # 3 levels, reference=0 -> 2 contrasts
        assert len(result.detail) == 2
        assert 'treatment' in result.detail.columns

    def test_effect_ordering(self, multi_treat_data):
        """Effect of D=2 should be larger than D=1 (true: 3.0 vs 1.5)."""
        result = multi_treatment(multi_treat_data, y='y', treat='d',
                                 covariates=['x1', 'x2'],
                                 n_bootstrap=50)
        eff_1 = result.detail.loc[result.detail['treatment'] == 1, 'estimate'].values[0]
        eff_2 = result.detail.loc[result.detail['treatment'] == 2, 'estimate'].values[0]
        assert eff_2 > eff_1

    def test_custom_reference(self, multi_treat_data):
        result = multi_treatment(multi_treat_data, y='y', treat='d',
                                 covariates=['x1', 'x2'],
                                 reference=1, n_bootstrap=50)
        # Reference=1 -> contrasts are 0 vs 1 and 2 vs 1
        assert len(result.detail) == 2
        assert all(result.detail['reference'] == 1)

    def test_two_levels(self):
        """Should work with exactly 2 levels (like binary)."""
        rng = np.random.default_rng(42)
        n = 300
        df = pd.DataFrame({
            'y': rng.normal(0, 1, n),
            'd': rng.choice([0, 1], size=n),
            'x1': rng.normal(0, 1, n),
        })
        result = multi_treatment(df, y='y', treat='d',
                                 covariates=['x1'], n_bootstrap=30)
        assert isinstance(result, CausalResult)

    def test_import(self):
        import statspai as sp
        assert hasattr(sp, 'multi_treatment')
