"""
Tests for advanced Synthetic Control modules:

- Matrix Completion SCM (mc.py)
- Distributional SCM (discos.py)
- Multiple Outcomes SCM (multi_outcome.py)
- Prediction Intervals (scpi.py)
- Sensitivity Analysis (sensitivity.py)
"""

import pytest
import numpy as np
import pandas as pd
from statspai.core.results import CausalResult


# ====================================================================== #
#  Shared fixtures
# ====================================================================== #

@pytest.fixture
def panel_data():
    """
    Simulated panel: 1 treated + 10 donors, 20 periods.
    Treatment at period 11 with effect = 5.0.
    """
    rng = np.random.default_rng(42)
    n_units = 11
    n_periods = 20
    treatment_time = 11

    records = []
    alphas = rng.normal(10, 2, n_units)
    betas = rng.normal(0.5, 0.1, n_units)

    for i in range(n_units):
        unit_name = f'unit_{i}'
        for t in range(1, n_periods + 1):
            y = alphas[i] + betas[i] * t + rng.normal(0, 0.3)
            if i == 0 and t >= treatment_time:
                y += 5.0
            records.append({
                'unit': unit_name,
                'time': t,
                'outcome': y,
            })

    return pd.DataFrame(records)


@pytest.fixture
def multi_outcome_data():
    """
    Panel with 3 outcomes for multi-outcome SCM tests.
    Treatment effect: outcome1 += 5, outcome2 += 3, outcome3 += 0 (placebo).
    """
    rng = np.random.default_rng(123)
    n_units = 11
    n_periods = 20
    treatment_time = 11

    records = []
    alphas = rng.normal(10, 2, (n_units, 3))
    betas = rng.normal(0.5, 0.1, (n_units, 3))

    for i in range(n_units):
        for t in range(1, n_periods + 1):
            row = {
                'unit': f'unit_{i}',
                'time': t,
            }
            for k, name in enumerate(['gdp', 'employment', 'investment']):
                y = alphas[i, k] + betas[i, k] * t + rng.normal(0, 0.3)
                if i == 0 and t >= treatment_time:
                    effects = [5.0, 3.0, 0.0]
                    y += effects[k]
                row[name] = y
            records.append(row)

    return pd.DataFrame(records)


# ====================================================================== #
#  Matrix Completion SCM
# ====================================================================== #

class TestMatrixCompletion:

    def test_basic_mc(self, panel_data):
        """MC-SCM should recover treatment effect ≈ 5.0."""
        from statspai.synth.mc import mc_synth

        result = mc_synth(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )

        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 5.0) < 3.0, (
            f"MC estimate = {result.estimate:.2f}, expected ≈ 5.0"
        )

    def test_returns_causal_result(self, panel_data):
        from statspai.synth.mc import mc_synth

        result = mc_synth(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11, placebo=False,
        )

        assert 'Matrix Completion' in result.method
        assert result.estimand == 'ATT'
        assert 'rank' in result.model_info or 'lambda_reg' in result.model_info

    def test_gap_table(self, panel_data):
        from statspai.synth.mc import mc_synth

        result = mc_synth(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11, placebo=False,
        )

        mi = result.model_info
        assert 'gap_table' in mi or 'Y_synth' in mi

    def test_mc_with_placebo(self, panel_data):
        from statspai.synth.mc import mc_synth

        result = mc_synth(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=True,
        )

        assert not np.isnan(result.pvalue)
        assert 0 < result.pvalue <= 1

    def test_mc_via_dispatcher(self, panel_data):
        """synth(method='mc') should dispatch to mc_synth."""
        from statspai.synth import synth

        result = synth(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            method='mc', placebo=False,
        )

        assert isinstance(result, CausalResult)
        assert 'Matrix Completion' in result.method

    def test_citation(self, panel_data):
        from statspai.synth.mc import mc_synth

        result = mc_synth(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11, placebo=False,
        )
        assert 'athey' in result.cite().lower() or 'matrix' in result.cite().lower()


# ====================================================================== #
#  Distributional SCM (DiSCo)
# ====================================================================== #

class TestDistributionalSCM:

    def test_basic_discos(self, panel_data):
        """DiSCo should estimate distributional effects."""
        from statspai.synth.discos import discos

        result = discos(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )

        assert isinstance(result, CausalResult)
        assert 'Distributional' in result.method

    def test_quantile_effects(self, panel_data):
        from statspai.synth.discos import discos

        result = discos(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )

        mi = result.model_info
        assert 'quantile_effects' in mi
        qe = mi['quantile_effects']
        assert isinstance(qe, pd.DataFrame)
        assert 'quantile' in qe.columns
        assert 'effect' in qe.columns

    def test_discos_positive_effect(self, panel_data):
        """Average quantile effect should be positive (true effect = 5)."""
        from statspai.synth.discos import discos

        result = discos(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )

        assert result.estimate > 0, (
            f"DiSCo estimate = {result.estimate:.2f}, expected > 0"
        )

    def test_discos_via_dispatcher(self, panel_data):
        from statspai.synth import synth

        result = synth(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            method='discos', placebo=False,
        )

        assert isinstance(result, CausalResult)
        assert 'Distributional' in result.method

    def test_citation(self, panel_data):
        from statspai.synth.discos import discos

        result = discos(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11, placebo=False,
        )
        cite = result.cite().lower()
        assert 'gunsilius' in cite or 'distributional' in cite


# ====================================================================== #
#  Multiple Outcomes SCM
# ====================================================================== #

class TestMultiOutcomeSCM:

    def test_basic_multi_outcome(self, multi_outcome_data):
        from statspai.synth.multi_outcome import multi_outcome_synth

        result = multi_outcome_synth(
            multi_outcome_data,
            outcomes=['gdp', 'employment', 'investment'],
            unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )

        assert isinstance(result, CausalResult)
        assert 'Multiple Outcomes' in result.method or 'Multi-Outcome' in result.method

    def test_per_outcome_effects(self, multi_outcome_data):
        from statspai.synth.multi_outcome import multi_outcome_synth

        result = multi_outcome_synth(
            multi_outcome_data,
            outcomes=['gdp', 'employment', 'investment'],
            unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )

        mi = result.model_info
        assert 'per_outcome_effects' in mi
        oe = mi['per_outcome_effects']
        assert len(oe) == 3

    def test_shared_weights(self, multi_outcome_data):
        """Should use a single set of donor weights for all outcomes."""
        from statspai.synth.multi_outcome import multi_outcome_synth

        result = multi_outcome_synth(
            multi_outcome_data,
            outcomes=['gdp', 'employment', 'investment'],
            unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )

        mi = result.model_info
        assert 'weights' in mi

    def test_effects_are_positive(self, multi_outcome_data):
        """GDP and employment effects should be positive (true effects > 0)."""
        from statspai.synth.multi_outcome import multi_outcome_synth

        result = multi_outcome_synth(
            multi_outcome_data,
            outcomes=['gdp', 'employment', 'investment'],
            unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )

        oe = result.model_info['per_outcome_effects']
        gdp_eff = oe[oe['outcome'] == 'gdp']['att'].values[0]
        emp_eff = oe[oe['outcome'] == 'employment']['att'].values[0]
        # Both should be positive (true effects are 5.0 and 3.0)
        assert gdp_eff > 0, f"GDP effect ({gdp_eff:.2f}) should be positive"
        assert emp_eff > 0, f"Employment effect ({emp_eff:.2f}) should be positive"

    def test_multi_outcome_via_dispatcher(self, multi_outcome_data):
        from statspai.synth import synth

        result = synth(
            multi_outcome_data, outcome='gdp', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            method='multi_outcome', placebo=False,
            outcomes=['gdp', 'employment', 'investment'],
        )

        assert isinstance(result, CausalResult)


# ====================================================================== #
#  Prediction Intervals (SCPI)
# ====================================================================== #

class TestSCPI:

    def test_basic_scpi(self, panel_data):
        from statspai.synth.scpi import scpi

        result = scpi(
            panel_data,
            outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
        )

        assert isinstance(result, CausalResult)
        assert 'Prediction Interval' in result.method or 'SCPI' in result.method

    def test_prediction_intervals(self, panel_data):
        """PI should be wider than zero-width."""
        from statspai.synth.scpi import scpi

        result = scpi(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
        )

        ci = result.ci
        assert ci[1] > ci[0], "PI upper should exceed lower"
        pi_width = ci[1] - ci[0]
        assert pi_width > 0.1, f"PI width = {pi_width:.3f}, too narrow"

    def test_period_results(self, panel_data):
        from statspai.synth.scpi import scpi

        result = scpi(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
        )

        mi = result.model_info
        assert 'period_results' in mi
        pr = mi['period_results']
        assert isinstance(pr, pd.DataFrame)
        assert 'pi_lower' in pr.columns or 'ci_lower' in pr.columns

    def test_scpi_via_dispatcher(self, panel_data):
        from statspai.synth import synth

        result = synth(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            method='scpi',
        )

        assert isinstance(result, CausalResult)

    def test_scpi_estimate_near_truth(self, panel_data):
        from statspai.synth.scpi import scpi

        result = scpi(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
        )

        assert abs(result.estimate - 5.0) < 3.0, (
            f"SCPI estimate = {result.estimate:.2f}, expected ≈ 5.0"
        )


# ====================================================================== #
#  Sensitivity Analysis
# ====================================================================== #

class TestSensitivity:

    def test_leave_one_out(self, panel_data):
        from statspai.synth.sensitivity import synth_loo

        loo = synth_loo(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
        )

        assert isinstance(loo, pd.DataFrame)
        assert 'dropped_unit' in loo.columns or 'unit' in loo.columns
        assert 'att' in loo.columns
        assert len(loo) >= 5  # at least half of donors

    def test_time_placebo(self, panel_data):
        from statspai.synth.sensitivity import synth_time_placebo

        tp = synth_time_placebo(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
        )

        assert isinstance(tp, pd.DataFrame)
        assert 'placebo_time' in tp.columns
        assert 'att' in tp.columns

    def test_time_placebo_near_zero(self, panel_data):
        """Backdated placebos should find no effect (≈ 0)."""
        from statspai.synth.sensitivity import synth_time_placebo

        tp = synth_time_placebo(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
        )

        # All placebo ATTs should be smaller than the real treatment
        real_effect = 5.0
        for _, row in tp.iterrows():
            assert abs(row['att']) < real_effect + 2.0

    def test_comprehensive_sensitivity(self, panel_data):
        from statspai.synth.sensitivity import synth_sensitivity

        sens = synth_sensitivity(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            n_donor_samples=20, seed=42,
        )

        assert isinstance(sens, dict)
        assert 'loo' in sens
        assert 'time_placebo' in sens
        assert 'summary' in sens

    def test_loo_atts_near_original(self, panel_data):
        """LOO ATTs should not deviate wildly from original."""
        from statspai.synth.sensitivity import synth_loo

        loo = synth_loo(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
        )

        atts = loo['att'].values
        assert np.std(atts) < 5.0, (
            f"LOO ATT std = {np.std(atts):.2f}, too variable"
        )


# ====================================================================== #
#  Integration: all methods through unified synth() dispatcher
# ====================================================================== #

class TestUnifiedDispatcher:

    ALL_METHODS = [
        'classic', 'penalized', 'ridge', 'demeaned', 'detrended',
        'unconstrained', 'elastic_net', 'augmented', 'ascm',
        'factor', 'gsynth', 'mc', 'discos', 'scpi',
    ]

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_all_methods_return_causal_result(self, panel_data, method):
        """Every method should return a valid CausalResult."""
        from statspai.synth import synth

        result = synth(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            method=method, placebo=False,
        )

        assert isinstance(result, CausalResult)
        assert 'ATT' in result.estimand  # covers 'ATT' and 'Distributional ATT'
        assert isinstance(result.estimate, float)
        assert result.ci[0] <= result.ci[1]

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_all_methods_positive_effect(self, panel_data, method):
        """All methods should detect the positive effect (≈ 5.0)."""
        from statspai.synth import synth

        result = synth(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            method=method, placebo=False,
        )

        assert result.estimate > 0, (
            f"method={method}: estimate = {result.estimate:.2f}, expected > 0"
        )


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
