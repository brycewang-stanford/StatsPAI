"""
Tests for Rambachan-Roth honest DID and Synthetic DID.
"""

import pytest
import numpy as np
import pandas as pd

from statspai.did import callaway_santanna, honest_did, breakdown_m
from statspai.synth.sdid import (
    sdid, synthdid_estimate, sc_estimate, did_estimate,
    synthdid_placebo, synthdid_plot, synthdid_units_plot,
    synthdid_rmse_plot, california_prop99,
)
from statspai.core.results import CausalResult


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def staggered_panel():
    """Same DGP as other DID tests: known positive effects."""
    rng = np.random.default_rng(42)
    n_units = 300
    unit_fe = rng.normal(0, 1, n_units)
    g_assign = np.zeros(n_units, dtype=int)
    g_assign[:100] = 4
    g_assign[100:200] = 6
    rows = []
    for u in range(n_units):
        for t in range(1, 9):
            te = 1.0 * (t - g_assign[u] + 1) if g_assign[u] > 0 and t >= g_assign[u] else 0
            rows.append({'unit': u, 'time': t, 'y': unit_fe[u] + 0.5 * t + te + rng.normal(0, 0.5), 'g': g_assign[u]})
    return pd.DataFrame(rows)


@pytest.fixture
def scm_panel():
    """Panel for SDID: 1 treated unit, 20 controls, 10 pre + 5 post."""
    rng = np.random.default_rng(42)
    n_units = 21
    T = 15
    treat_time = 11
    unit_fe = rng.normal(0, 1, n_units)
    rows = []
    for u in range(n_units):
        for t in range(1, T + 1):
            te = 3.0 if u == 0 and t >= treat_time else 0
            y = unit_fe[u] + 0.3 * t + te + rng.normal(0, 0.5)
            rows.append({'unit': u, 'time': t, 'y': y})
    return pd.DataFrame(rows), treat_time


# ======================================================================
# Rambachan-Roth tests
# ======================================================================

class TestHonestDID:
    """Tests for Rambachan & Roth (2023) sensitivity analysis."""

    def test_basic_run(self, staggered_panel):
        """Should return a DataFrame with expected columns."""
        r = callaway_santanna(staggered_panel, y='y', g='g', t='time', i='unit')
        result = honest_did(r, e=0)
        assert isinstance(result, pd.DataFrame)
        assert 'M' in result.columns
        assert 'ci_lower' in result.columns
        assert 'ci_upper' in result.columns
        assert 'rejects_zero' in result.columns

    def test_m_zero_matches_original(self, staggered_panel):
        """At M=0 (exact parallel trends), CI should match standard CI."""
        r = callaway_santanna(staggered_panel, y='y', g='g', t='time', i='unit')
        result = honest_did(r, e=0, m_grid=[0])
        es = r.model_info['event_study']
        target = es[es['relative_time'] == 0].iloc[0]
        # At M=0, honest CI = standard CI
        assert abs(result.iloc[0]['ci_lower'] - target['ci_lower']) < 0.01
        assert abs(result.iloc[0]['ci_upper'] - target['ci_upper']) < 0.01

    def test_ci_widens_with_m(self, staggered_panel):
        """CIs should widen as M increases."""
        r = callaway_santanna(staggered_panel, y='y', g='g', t='time', i='unit')
        result = honest_did(r, e=0)
        widths = result['ci_upper'] - result['ci_lower']
        # Each row should have wider CI than previous
        assert all(widths.iloc[i] <= widths.iloc[i + 1] + 1e-10
                   for i in range(len(widths) - 1))

    def test_rejects_at_small_m(self, staggered_panel):
        """With strong effect, should reject zero at small M."""
        r = callaway_santanna(staggered_panel, y='y', g='g', t='time', i='unit')
        result = honest_did(r, e=0, m_grid=[0])
        assert result.iloc[0]['rejects_zero'] is True or result.iloc[0]['rejects_zero'] == True

    def test_relative_magnitude_method(self, staggered_panel):
        """Relative magnitude method should work."""
        r = callaway_santanna(staggered_panel, y='y', g='g', t='time', i='unit')
        result = honest_did(r, e=0, method='relative_magnitude')
        assert len(result) > 0

    def test_breakdown_m(self, staggered_panel):
        """Breakdown M should be positive when effect is significant."""
        r = callaway_santanna(staggered_panel, y='y', g='g', t='time', i='unit')
        m_star = breakdown_m(r, e=0)
        assert m_star > 0

    def test_different_e(self, staggered_panel):
        """Should work for different relative times."""
        r = callaway_santanna(staggered_panel, y='y', g='g', t='time', i='unit')
        r0 = honest_did(r, e=0)
        r1 = honest_did(r, e=1)
        assert len(r0) > 0
        assert len(r1) > 0


# ======================================================================
# Synthetic DID tests
# ======================================================================

class TestSyntheticDID:
    """Tests for Arkhangelsky et al. (2021) SDID."""

    def test_basic_run(self, scm_panel):
        panel, treat_time = scm_panel
        result = sdid(panel, y='y', unit='unit', time='time',
                      treat_unit=0, treat_time=treat_time,
                      n_reps=50, seed=42)
        assert isinstance(result, CausalResult)
        assert 'Synthetic' in result.method

    def test_positive_effect(self, scm_panel):
        """True effect = 3.0, estimate should be positive."""
        panel, treat_time = scm_panel
        result = sdid(panel, y='y', unit='unit', time='time',
                      treat_unit=0, treat_time=treat_time,
                      n_reps=50, seed=42)
        assert result.estimate > 0

    def test_effect_magnitude(self, scm_panel):
        """Estimate should be in reasonable range of true value 3.0."""
        panel, treat_time = scm_panel
        result = sdid(panel, y='y', unit='unit', time='time',
                      treat_unit=0, treat_time=treat_time,
                      n_reps=100, seed=42)
        assert abs(result.estimate - 3.0) < 2.0

    def test_unit_weights(self, scm_panel):
        """Unit weights should sum to ~1 and be non-negative."""
        panel, treat_time = scm_panel
        result = sdid(panel, y='y', unit='unit', time='time',
                      treat_unit=0, treat_time=treat_time,
                      n_reps=50, seed=42)
        w = result.model_info['unit_weights']
        assert abs(w['weight'].sum() - 1.0) < 0.01
        assert (w['weight'] >= -0.01).all()

    def test_time_weights(self, scm_panel):
        """Time weights should sum to ~1 and be non-negative."""
        panel, treat_time = scm_panel
        result = sdid(panel, y='y', unit='unit', time='time',
                      treat_unit=0, treat_time=treat_time,
                      n_reps=50, seed=42)
        lam = result.model_info['time_weights']
        assert abs(lam.sum() - 1.0) < 0.01
        assert (lam >= -0.01).all()

    def test_ci_reasonable(self, scm_panel):
        """CI should be in a reasonable range around estimate."""
        panel, treat_time = scm_panel
        result = sdid(panel, y='y', unit='unit', time='time',
                      treat_unit=0, treat_time=treat_time,
                      n_reps=200, seed=42)
        # CI should be finite and centered around estimate
        assert result.ci[0] < result.estimate < result.ci[1]
        assert result.se > 0

    def test_cite(self, scm_panel):
        panel, treat_time = scm_panel
        result = sdid(panel, y='y', unit='unit', time='time',
                      treat_unit=0, treat_time=treat_time,
                      n_reps=20, seed=42)
        assert 'arkhangelsky' in result.cite().lower()

    def test_summary(self, scm_panel):
        panel, treat_time = scm_panel
        result = sdid(panel, y='y', unit='unit', time='time',
                      treat_unit=0, treat_time=treat_time,
                      n_reps=20, seed=42)
        s = result.summary()
        assert 'ATT' in s


class TestSynthdidMethods:
    """Test all three estimator methods: sdid, sc, did."""

    def test_sc_method(self, scm_panel):
        panel, treat_time = scm_panel
        result = sdid(panel, y='y', unit='unit', time='time',
                      treat_unit=0, treat_time=treat_time,
                      method='sc', n_reps=30, seed=42)
        assert result.model_info['estimator'] == 'sc'
        assert result.estimate > 0

    def test_did_method(self, scm_panel):
        panel, treat_time = scm_panel
        result = sdid(panel, y='y', unit='unit', time='time',
                      treat_unit=0, treat_time=treat_time,
                      method='did', n_reps=30, seed=42)
        assert result.model_info['estimator'] == 'did'
        # DID should also detect positive effect
        assert result.estimate > 0

    def test_r_style_aliases(self, scm_panel):
        panel, treat_time = scm_panel
        r1 = synthdid_estimate(panel, y='y', unit='unit', time='time',
                               treat_unit=0, treat_time=treat_time,
                               n_reps=20, seed=42)
        r2 = sc_estimate(panel, y='y', unit='unit', time='time',
                         treat_unit=0, treat_time=treat_time,
                         n_reps=20, seed=42)
        r3 = did_estimate(panel, y='y', unit='unit', time='time',
                          treat_unit=0, treat_time=treat_time,
                          n_reps=20, seed=42)
        assert r1.model_info['estimator'] == 'sdid'
        assert r2.model_info['estimator'] == 'sc'
        assert r3.model_info['estimator'] == 'did'

    def test_three_methods_different(self, scm_panel):
        """Three methods should give different estimates."""
        panel, treat_time = scm_panel
        results = {}
        for m in ['sdid', 'sc', 'did']:
            results[m] = sdid(panel, y='y', unit='unit', time='time',
                              treat_unit=0, treat_time=treat_time,
                              method=m, n_reps=20, seed=42)
        # Not all three should be exactly the same
        ests = [results[m].estimate for m in ['sdid', 'sc', 'did']]
        assert len(set(round(e, 6) for e in ests)) > 1


class TestSEMethods:
    """Test all three SE methods: placebo, bootstrap, jackknife."""

    def test_placebo_se(self, scm_panel):
        panel, treat_time = scm_panel
        result = sdid(panel, y='y', unit='unit', time='time',
                      treat_unit=0, treat_time=treat_time,
                      se_method='placebo', seed=42)
        assert result.se > 0
        assert result.model_info['se_method'] == 'placebo'

    def test_bootstrap_se(self, scm_panel):
        panel, treat_time = scm_panel
        result = sdid(panel, y='y', unit='unit', time='time',
                      treat_unit=0, treat_time=treat_time,
                      se_method='bootstrap', n_reps=50, seed=42)
        assert result.se > 0
        assert result.model_info['se_method'] == 'bootstrap'

    def test_jackknife_se(self, scm_panel):
        panel, treat_time = scm_panel
        result = sdid(panel, y='y', unit='unit', time='time',
                      treat_unit=0, treat_time=treat_time,
                      se_method='jackknife', seed=42)
        assert result.se > 0
        assert result.model_info['se_method'] == 'jackknife'


class TestCaliforniaProp99:
    """Test the built-in example dataset."""

    def test_dataset_shape(self):
        df = california_prop99()
        assert isinstance(df, pd.DataFrame)
        assert 'state' in df.columns
        assert 'year' in df.columns
        assert 'packspercapita' in df.columns
        # 39 states × 31 years
        assert len(df) == 39 * 31

    def test_dataset_balanced(self):
        df = california_prop99()
        counts = df.groupby('state').size()
        assert counts.nunique() == 1  # all states have same number of years

    def test_dataset_with_sdid(self):
        df = california_prop99()
        result = sdid(df, y='packspercapita', unit='state', time='year',
                      treat_unit='California', treat_time=1989,
                      n_reps=30, seed=42)
        # Should detect negative effect (tobacco control reduces smoking)
        assert result.estimate < 0


class TestPlots:
    """Test plotting functions (just check they don't error)."""

    @pytest.fixture
    def sdid_result(self, scm_panel):
        panel, treat_time = scm_panel
        return sdid(panel, y='y', unit='unit', time='time',
                    treat_unit=0, treat_time=treat_time,
                    n_reps=20, seed=42)

    def test_synthdid_plot(self, sdid_result):
        pytest.importorskip('matplotlib')
        fig, ax = synthdid_plot(sdid_result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_units_plot(self, sdid_result):
        pytest.importorskip('matplotlib')
        fig, ax = synthdid_units_plot(sdid_result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_rmse_plot(self, sdid_result):
        pytest.importorskip('matplotlib')
        fig, ax = synthdid_rmse_plot(sdid_result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlaceboAnalysis:
    """Test synthdid_placebo."""

    def test_placebo_runs(self, scm_panel):
        panel, treat_time = scm_panel
        result = synthdid_placebo(
            panel, y='y', unit='unit', time='time',
            treat_unit=0, treat_time=treat_time,
            n_reps=10, seed=42,
        )
        assert isinstance(result, pd.DataFrame)
        assert 'unit' in result.columns
        assert 'estimate' in result.columns
        assert len(result) > 0


class TestModelInfo:
    """Test model_info fields for downstream use."""

    def test_trajectories_stored(self, scm_panel):
        panel, treat_time = scm_panel
        result = sdid(panel, y='y', unit='unit', time='time',
                      treat_unit=0, treat_time=treat_time,
                      n_reps=20, seed=42)
        mi = result.model_info
        assert 'Y_obs' in mi
        assert 'Y_synth' in mi
        assert len(mi['Y_obs']) == len(mi['all_times'])
        assert len(mi['Y_synth']) == len(mi['all_times'])

    def test_all_fields_present(self, scm_panel):
        panel, treat_time = scm_panel
        result = sdid(panel, y='y', unit='unit', time='time',
                      treat_unit=0, treat_time=treat_time,
                      n_reps=20, seed=42)
        mi = result.model_info
        for key in ['estimator', 'n_treated', 'n_control',
                    'T_pre', 'T_post', 'treat_time',
                    'unit_weights', 'time_weights',
                    'pre_times', 'post_times']:
            assert key in mi, f"Missing key: {key}"


# ======================================================================
# Integration
# ======================================================================

class TestIntegration:
    def test_import(self):
        import statspai as sp
        assert hasattr(sp, 'honest_did')
        assert hasattr(sp, 'breakdown_m')
        assert hasattr(sp, 'sdid')
        assert hasattr(sp, 'synthdid_estimate')
        assert hasattr(sp, 'sc_estimate')
        assert hasattr(sp, 'did_estimate')
        assert hasattr(sp, 'california_prop99')
        assert hasattr(sp, 'synthdid_plot')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
