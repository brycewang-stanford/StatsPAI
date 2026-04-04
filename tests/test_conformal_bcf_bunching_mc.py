"""
Tests for Conformal Causal, BCF, Bunching, and Matrix Completion.
"""

import pytest
import numpy as np
import pandas as pd

from statspai.conformal_causal import conformal_cate, ConformalCATE
from statspai.bcf import bcf, BayesianCausalForest
from statspai.bunching import bunching, BunchingEstimator
from statspai.matrix_completion import mc_panel, MCPanel
from statspai.core.results import CausalResult


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def treatment_data():
    rng = np.random.default_rng(42)
    n = 1000
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n).astype(float)
    Y = 2.0 * D + X1 + rng.normal(0, 0.5, n)
    return pd.DataFrame({'y': Y, 'd': D, 'x1': X1, 'x2': X2})


@pytest.fixture
def bunching_data():
    """Income data with bunching at threshold 50000."""
    rng = np.random.default_rng(42)
    n = 5000
    income = rng.normal(50000, 15000, n)
    # Add bunching: some people near threshold bunch to just below
    near_thresh = np.abs(income - 50000) < 3000
    income[near_thresh] = income[near_thresh] - np.abs(
        rng.normal(0, 500, near_thresh.sum())
    )
    return pd.DataFrame({'income': income})


@pytest.fixture
def panel_data():
    """Simple panel data with treatment."""
    rng = np.random.default_rng(42)
    units = 20
    periods = 10
    treat_period = 7

    rows = []
    for i in range(units):
        unit_fe = rng.normal(0, 1)
        treated_unit = i < 5  # first 5 units are treated
        for t in range(periods):
            time_fe = 0.5 * t
            is_treated = 1 if (treated_unit and t >= treat_period) else 0
            y = unit_fe + time_fe + 3.0 * is_treated + rng.normal(0, 0.5)
            rows.append({
                'unit': i, 'time': t,
                'y': y, 'treat': is_treated
            })
    return pd.DataFrame(rows)


# ======================================================================
# Conformal CATE Tests
# ======================================================================

class TestConformalCATE:

    def test_returns_causal_result(self, treatment_data):
        result = conformal_cate(treatment_data, y='y', treat='d',
                                covariates=['x1', 'x2'],
                                calib_fraction=0.3)
        assert isinstance(result, CausalResult)
        assert 'Conformal' in result.method

    def test_has_intervals(self, treatment_data):
        result = conformal_cate(treatment_data, y='y', treat='d',
                                covariates=['x1', 'x2'],
                                calib_fraction=0.3)
        assert 'cate_lower' in result.model_info
        assert 'cate_upper' in result.model_info
        lower = result.model_info['cate_lower']
        upper = result.model_info['cate_upper']
        assert len(lower) == len(treatment_data)
        assert np.all(lower <= upper)

    def test_interval_width_positive(self, treatment_data):
        result = conformal_cate(treatment_data, y='y', treat='d',
                                covariates=['x1', 'x2'],
                                calib_fraction=0.3)
        assert result.model_info['interval_width'] > 0

    def test_predict_new_data(self, treatment_data):
        est = ConformalCATE(data=treatment_data, y='y', treat='d',
                           covariates=['x1', 'x2'], calib_fraction=0.3)
        est.fit()
        X_new = np.random.randn(5, 2)
        out = est.predict(X_new)
        assert 'cate' in out
        assert 'lower' in out
        assert 'upper' in out
        assert len(out['cate']) == 5

    def test_citation(self, treatment_data):
        result = conformal_cate(treatment_data, y='y', treat='d',
                                covariates=['x1', 'x2'],
                                calib_fraction=0.3)
        assert 'lei2021' in result.cite()


# ======================================================================
# BCF Tests
# ======================================================================

class TestBCF:

    def test_returns_causal_result(self, treatment_data):
        result = bcf(treatment_data, y='y', treat='d',
                     covariates=['x1', 'x2'],
                     n_bootstrap=30, n_trees_mu=50, n_trees_tau=20)
        assert isinstance(result, CausalResult)
        assert 'BCF' in result.method

    def test_has_cate_with_uncertainty(self, treatment_data):
        result = bcf(treatment_data, y='y', treat='d',
                     covariates=['x1', 'x2'],
                     n_bootstrap=30, n_trees_mu=50, n_trees_tau=20)
        assert 'cate' in result.model_info
        assert 'cate_sd' in result.model_info
        assert 'cate_lower' in result.model_info
        assert 'cate_upper' in result.model_info
        assert len(result.model_info['cate']) == len(treatment_data)

    def test_effect_recovery(self, treatment_data):
        result = bcf(treatment_data, y='y', treat='d',
                     covariates=['x1', 'x2'],
                     n_bootstrap=50, n_trees_mu=100, n_trees_tau=30)
        assert abs(result.estimate - 2.0) < 1.5

    def test_effect_method(self, treatment_data):
        est = BayesianCausalForest(
            data=treatment_data, y='y', treat='d',
            covariates=['x1', 'x2'],
            n_bootstrap=20, n_trees_mu=50, n_trees_tau=20
        )
        est.fit()
        cate = est.effect()
        assert len(cate) == len(treatment_data)

    def test_citation(self, treatment_data):
        result = bcf(treatment_data, y='y', treat='d',
                     covariates=['x1', 'x2'],
                     n_bootstrap=20, n_trees_mu=50, n_trees_tau=20)
        assert 'hahn2020' in result.cite()


# ======================================================================
# Bunching Tests
# ======================================================================

class TestBunching:

    def test_returns_causal_result(self, bunching_data):
        result = bunching(bunching_data, running_var='income',
                          threshold=50000)
        assert isinstance(result, CausalResult)
        assert 'Bunching' in result.method

    def test_has_detail_histogram(self, bunching_data):
        result = bunching(bunching_data, running_var='income',
                          threshold=50000)
        assert result.detail is not None
        assert 'observed' in result.detail.columns
        assert 'counterfactual' in result.detail.columns

    def test_elasticity_with_dt(self, bunching_data):
        result = bunching(bunching_data, running_var='income',
                          threshold=50000, dt=0.10)
        assert 'elasticity' in result.model_info

    def test_kink_vs_notch(self, bunching_data):
        r_kink = bunching(bunching_data, running_var='income',
                          threshold=50000, dt=0.10, design='kink')
        r_notch = bunching(bunching_data, running_var='income',
                           threshold=50000, dt=0.10, design='notch')
        assert 'Kink' in r_kink.method
        assert 'Notch' in r_notch.method

    def test_custom_bunching_region(self, bunching_data):
        result = bunching(bunching_data, running_var='income',
                          threshold=50000,
                          bunch_region=(48000, 52000))
        assert isinstance(result, CausalResult)

    def test_missing_column_raises(self, bunching_data):
        with pytest.raises(ValueError, match="not found"):
            bunching(bunching_data, running_var='salary',
                     threshold=50000)

    def test_citation(self, bunching_data):
        result = bunching(bunching_data, running_var='income',
                          threshold=50000)
        assert 'kleven2013' in result.cite()


# ======================================================================
# Matrix Completion Tests
# ======================================================================

class TestMCPanel:

    def test_returns_causal_result(self, panel_data):
        result = mc_panel(panel_data, y='y', unit='unit',
                          time='time', treat='treat',
                          n_bootstrap=20)
        assert isinstance(result, CausalResult)
        assert 'Matrix Completion' in result.method

    def test_effect_positive(self, panel_data):
        """True ATT is 3.0, should be reasonably close."""
        result = mc_panel(panel_data, y='y', unit='unit',
                          time='time', treat='treat',
                          n_bootstrap=30)
        assert abs(result.estimate - 3.0) < 2.0

    def test_has_detail_per_unit(self, panel_data):
        result = mc_panel(panel_data, y='y', unit='unit',
                          time='time', treat='treat',
                          n_bootstrap=20)
        assert result.detail is not None
        assert 'unit' in result.detail.columns
        assert 'att' in result.detail.columns

    def test_effective_rank(self, panel_data):
        result = mc_panel(panel_data, y='y', unit='unit',
                          time='time', treat='treat',
                          n_bootstrap=20)
        assert result.model_info['effective_rank'] >= 1

    def test_max_rank_constraint(self, panel_data):
        result = mc_panel(panel_data, y='y', unit='unit',
                          time='time', treat='treat',
                          max_rank=2, n_bootstrap=20)
        assert result.model_info['effective_rank'] <= 2

    def test_missing_column_raises(self, panel_data):
        with pytest.raises(ValueError, match="Columns not found"):
            mc_panel(panel_data, y='y', unit='unit',
                     time='time', treat='nonexistent',
                     n_bootstrap=20)

    def test_citation(self, panel_data):
        result = mc_panel(panel_data, y='y', unit='unit',
                          time='time', treat='treat',
                          n_bootstrap=20)
        assert 'athey2021' in result.cite()


# ======================================================================
# Import Tests
# ======================================================================

class TestImports:
    def test_all_imports(self):
        import statspai as sp
        assert hasattr(sp, 'conformal_cate')
        assert hasattr(sp, 'ConformalCATE')
        assert hasattr(sp, 'bcf')
        assert hasattr(sp, 'BayesianCausalForest')
        assert hasattr(sp, 'bunching')
        assert hasattr(sp, 'BunchingEstimator')
        assert hasattr(sp, 'mc_panel')
        assert hasattr(sp, 'MCPanel')
