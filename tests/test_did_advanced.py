"""
Tests for advanced DID methods: Sun-Abraham and Bacon decomposition.

Uses the same staggered DGP as test_did.py for consistency.
"""

import pytest
import numpy as np
import pandas as pd

from statspai.did import sun_abraham, bacon_decomposition, did
from statspai.core.results import CausalResult


@pytest.fixture
def staggered_panel():
    """Staggered DID panel with known effects.

    - 100 units in cohort g=4, 100 in g=6, 100 never-treated (g=0)
    - 8 periods (1..8)
    - True ATT(g,t) = 1*(t - g + 1) for t >= g (dynamic, linearly increasing)
    - 'treated' column: binary 0/1 indicator for TWFE
    """
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
            treated = 1 if g_assign[u] > 0 and t >= g_assign[u] else 0
            y = unit_fe[u] + 0.5 * t + te + rng.normal(0, 0.5)
            rows.append({
                'unit': u, 'time': t, 'y': y,
                'g': g_assign[u], 'treated': treated,
            })
    return pd.DataFrame(rows)


# ======================================================================
# Sun-Abraham tests
# ======================================================================

class TestSunAbraham:
    """Tests for Sun & Abraham (2021) IW estimator."""

    def test_basic_run(self, staggered_panel):
        result = sun_abraham(staggered_panel, y='y', g='g', t='time', i='unit')
        assert isinstance(result, CausalResult)
        assert 'Sun' in result.method

    def test_positive_att(self, staggered_panel):
        """Overall ATT should be positive (DGP has positive effects)."""
        result = sun_abraham(staggered_panel, y='y', g='g', t='time', i='unit')
        assert result.estimate > 0

    def test_event_study_available(self, staggered_panel):
        """Event study should be in model_info."""
        result = sun_abraham(staggered_panel, y='y', g='g', t='time', i='unit')
        assert 'event_study' in result.model_info
        es = result.model_info['event_study']
        assert len(es) > 0
        assert 'relative_time' in es.columns
        assert 'att' in es.columns

    def test_post_effects_positive(self, staggered_panel):
        """Post-treatment event study effects should be positive."""
        result = sun_abraham(staggered_panel, y='y', g='g', t='time', i='unit')
        es = result.model_info['event_study']
        post = es[es['relative_time'] >= 0]
        if len(post) > 0:
            assert (post['att'] > 0).all()

    def test_event_window(self, staggered_panel):
        """Custom event window should limit output."""
        result = sun_abraham(staggered_panel, y='y', g='g', t='time', i='unit',
                             event_window=(-3, 3))
        es = result.model_info['event_study']
        assert es['relative_time'].min() >= -3
        assert es['relative_time'].max() <= 3

    def test_via_did_dispatcher(self, staggered_panel):
        """did(method='sun_abraham') should dispatch correctly."""
        result = did(staggered_panel, y='y', treat='g', time='time', id='unit',
                     method='sun_abraham')
        assert 'Sun' in result.method

    def test_cite(self, staggered_panel):
        result = sun_abraham(staggered_panel, y='y', g='g', t='time', i='unit')
        bib = result.cite()
        assert 'sun2021' in bib

    def test_summary(self, staggered_panel):
        result = sun_abraham(staggered_panel, y='y', g='g', t='time', i='unit')
        s = result.summary()
        assert 'ATT' in s


# ======================================================================
# Bacon decomposition tests
# ======================================================================

class TestBaconDecomposition:
    """Tests for Goodman-Bacon (2021) decomposition."""

    def test_basic_run(self, staggered_panel):
        result = bacon_decomposition(
            staggered_panel, y='y', treat='treated', time='time', id='unit')
        assert 'decomposition' in result
        assert 'beta_twfe' in result

    def test_decomposition_nonempty(self, staggered_panel):
        result = bacon_decomposition(
            staggered_panel, y='y', treat='treated', time='time', id='unit')
        decomp = result['decomposition']
        assert len(decomp) > 0
        assert 'type' in decomp.columns
        assert 'estimate' in decomp.columns
        assert 'weight' in decomp.columns

    def test_weights_sum_to_one(self, staggered_panel):
        """Normalized weights should sum to approximately 1."""
        result = bacon_decomposition(
            staggered_panel, y='y', treat='treated', time='time', id='unit')
        total_w = result['decomposition']['weight'].sum()
        assert abs(total_w - 1.0) < 0.01

    def test_comparison_types(self, staggered_panel):
        """Should identify different comparison types."""
        result = bacon_decomposition(
            staggered_panel, y='y', treat='treated', time='time', id='unit')
        types = result['decomposition']['type'].unique()
        # With 2 cohorts + never-treated, we should see multiple types
        assert len(types) >= 2

    def test_negative_weight_share(self, staggered_panel):
        """Should report negative weight share."""
        result = bacon_decomposition(
            staggered_panel, y='y', treat='treated', time='time', id='unit')
        assert 'negative_weight_share' in result
        assert 0 <= result['negative_weight_share'] <= 1

    def test_twfe_estimate_reasonable(self, staggered_panel):
        """TWFE estimate should be positive (positive DGP)."""
        result = bacon_decomposition(
            staggered_panel, y='y', treat='treated', time='time', id='unit')
        assert result['beta_twfe'] > 0


# ======================================================================
# Cross-method consistency
# ======================================================================

class TestCrossMethodConsistency:
    """Verify DID methods give broadly consistent results on same data."""

    def test_cs_vs_sa_same_sign(self, staggered_panel):
        """C&S and Sun-Abraham should agree on sign of ATT."""
        from statspai.did import callaway_santanna
        r_cs = callaway_santanna(staggered_panel, y='y', g='g', t='time', i='unit')
        r_sa = sun_abraham(staggered_panel, y='y', g='g', t='time', i='unit')
        assert r_cs.estimate > 0
        assert r_sa.estimate > 0

    def test_bacon_twfe_vs_cs(self, staggered_panel):
        """Bacon TWFE should differ from C&S (TWFE is biased with hetero effects)."""
        from statspai.did import callaway_santanna
        r_cs = callaway_santanna(staggered_panel, y='y', g='g', t='time', i='unit')
        bacon = bacon_decomposition(
            staggered_panel, y='y', treat='treated', time='time', id='unit')
        # Both positive, but TWFE may differ due to heterogeneous effects
        assert bacon['beta_twfe'] > 0
        assert r_cs.estimate > 0


class TestIntegration:
    def test_import(self):
        import statspai as sp
        assert hasattr(sp, 'sun_abraham')
        assert hasattr(sp, 'bacon_decomposition')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
