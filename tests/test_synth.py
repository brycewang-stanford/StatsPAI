"""
Tests for Synthetic Control Method.

Uses simulated panel data with known treatment effects.
"""

import pytest
import numpy as np
import pandas as pd
from statspai.synth import synth, SyntheticControl
from statspai.core.results import CausalResult


@pytest.fixture
def panel_data():
    """
    Simulated panel: 1 treated unit + 10 donors, 20 periods.
    Treatment at period 11 with effect = 5.0.

    DGP: Y_it = alpha_i + beta_i * t + eps_it
          Treated unit gets +5.0 after t >= 11.
    """
    rng = np.random.default_rng(42)
    n_units = 11  # 1 treated + 10 donors
    n_periods = 20
    treatment_time = 11

    records = []
    alphas = rng.normal(10, 2, n_units)
    betas = rng.normal(0.5, 0.1, n_units)

    for i in range(n_units):
        unit_name = f'unit_{i}'
        for t in range(1, n_periods + 1):
            y = alphas[i] + betas[i] * t + rng.normal(0, 0.3)
            # Add treatment effect to unit_0
            if i == 0 and t >= treatment_time:
                y += 5.0
            records.append({
                'unit': unit_name,
                'time': t,
                'outcome': y,
            })

    return pd.DataFrame(records)


@pytest.fixture
def panel_no_effect():
    """Panel with no treatment effect (placebo check)."""
    rng = np.random.default_rng(99)
    n_units = 8
    n_periods = 15

    records = []
    for i in range(n_units):
        alpha = rng.normal(5, 1)
        for t in range(1, n_periods + 1):
            y = alpha + 0.3 * t + rng.normal(0, 0.2)
            records.append({'unit': f'u{i}', 'time': t, 'outcome': y})

    return pd.DataFrame(records)


class TestSyntheticControl:

    def test_basic_synth(self, panel_data):
        """Should recover treatment effect ≈ 5.0"""
        result = synth(
            panel_data,
            outcome='outcome',
            unit='unit',
            time='time',
            treated_unit='unit_0',
            treatment_time=11,
            placebo=False,
        )

        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 5.0) < 2.0, (
            f"SCM estimate = {result.estimate:.2f}, expected ≈ 5.0"
        )

    def test_returns_causal_result(self, panel_data):
        """Result should have correct structure."""
        result = synth(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11, placebo=False,
        )

        assert result.method == 'Synthetic Control Method'
        assert result.estimand == 'ATT'
        assert result.se > 0
        assert result.ci[0] < result.estimate < result.ci[1]

    def test_weights_sum_to_one(self, panel_data):
        """Donor weights should sum to 1."""
        model = SyntheticControl(
            data=panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
        )
        result = model.fit(placebo=False)

        weight_df = result.model_info['weights']
        assert abs(weight_df['weight'].sum() - 1.0) < 0.01

    def test_weights_non_negative(self, panel_data):
        """All donor weights should be non-negative."""
        result = synth(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11, placebo=False,
        )

        weights = result.model_info['weights']
        assert (weights['weight'] >= -1e-6).all()

    def test_pre_treatment_fit(self, panel_data):
        """Pre-treatment RMSE should be small."""
        result = synth(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11, placebo=False,
        )

        pre_rmse = result.model_info['pre_treatment_rmse']
        assert pre_rmse < 2.0, f"Pre-treatment RMSE = {pre_rmse:.3f}, too large"

    def test_gap_table(self, panel_data):
        """Gap table should have correct structure."""
        result = synth(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11, placebo=False,
        )

        gap = result.model_info['gap_table']
        assert 'time' in gap.columns
        assert 'treated' in gap.columns
        assert 'synthetic' in gap.columns
        assert 'gap' in gap.columns
        assert len(gap) == 20

    def test_placebo_inference(self, panel_data):
        """Placebo inference should produce a p-value."""
        result = synth(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=True,
        )

        assert not np.isnan(result.pvalue)
        assert 0 < result.pvalue <= 1
        assert 'n_placebos' in result.model_info

    def test_no_effect_panel(self, panel_no_effect):
        """With no effect, estimate should be near zero."""
        result = synth(
            panel_no_effect, outcome='outcome', unit='unit', time='time',
            treated_unit='u0', treatment_time=8, placebo=False,
        )

        assert abs(result.estimate) < 2.0, (
            f"Null effect estimate = {result.estimate:.2f}, should be ≈ 0"
        )

    def test_penalized_scm(self, panel_data):
        """Penalized SCM should run without error."""
        result = synth(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            penalization=0.1, placebo=False,
        )

        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 5.0) < 3.0

    def test_model_info(self, panel_data):
        """Model info should contain expected fields."""
        result = synth(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11, placebo=False,
        )

        info = result.model_info
        assert 'n_donors' in info
        assert info['n_donors'] == 10
        assert 'n_pre_periods' in info
        assert info['n_pre_periods'] == 10
        assert info['treatment_time'] == 11

    def test_summary(self, panel_data):
        """Summary should contain SCM-specific info."""
        result = synth(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11, placebo=False,
        )
        s = result.summary()
        assert 'Synthetic Control' in s

    def test_citation(self, panel_data):
        """Should return Abadie et al. citation."""
        result = synth(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11, placebo=False,
        )
        assert 'abadie' in result.cite().lower()

    # --- Error handling ---

    def test_missing_column(self, panel_data):
        with pytest.raises(ValueError, match="not found"):
            synth(panel_data, outcome='nonexistent', unit='unit',
                  time='time', treated_unit='unit_0', treatment_time=11)

    def test_missing_treated_unit(self, panel_data):
        with pytest.raises(ValueError, match="not found"):
            synth(panel_data, outcome='outcome', unit='unit',
                  time='time', treated_unit='nonexistent', treatment_time=11)

    def test_too_few_pre_periods(self, panel_data):
        """Should raise error with < 2 pre-treatment periods."""
        with pytest.raises(ValueError, match="pre-treatment"):
            synth(panel_data, outcome='outcome', unit='unit',
                  time='time', treated_unit='unit_0', treatment_time=2)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
