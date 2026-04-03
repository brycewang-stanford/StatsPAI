"""
Tests for DID module: 2x2 DID and Callaway-Sant'Anna staggered DID.

All tests use simulated data with known true treatment effects
to validate correctness.
"""

import pytest
import numpy as np
import pandas as pd

from statspai.did import did, did_2x2, callaway_santanna
from statspai.core.results import CausalResult


# ======================================================================
# Fixtures: simulated data with known truth
# ======================================================================

@pytest.fixture
def data_2x2():
    """Classic 2x2 DID data. True ATT = 5.0."""
    rng = np.random.default_rng(42)
    n = 2000
    treat = rng.integers(0, 2, n)
    post = rng.integers(0, 2, n)
    # Y = 1 + 2*treat + 3*post + 5*treat*post + ε
    y = 1 + 2 * treat + 3 * post + 5 * treat * post + rng.normal(0, 1, n)
    return pd.DataFrame({
        'y': y, 'treat': treat, 'post': post,
        'x1': rng.normal(0, 1, n),
        'cluster_id': rng.integers(0, 20, n),
    })


@pytest.fixture
def data_staggered():
    """Staggered DID panel. True ATT(g,t) = 1*(t-g+1) for t>=g.

    - 100 units in cohort g=4 (treated at period 4)
    - 100 units in cohort g=6 (treated at period 6)
    - 100 units never-treated (g=0)
    - 8 time periods (1..8)
    """
    rng = np.random.default_rng(42)
    n_units = 300
    n_periods = 8
    unit_fe = rng.normal(0, 1, n_units)

    g_assign = np.zeros(n_units, dtype=int)
    g_assign[:100] = 4
    g_assign[100:200] = 6

    rows = []
    for unit in range(n_units):
        for t in range(1, n_periods + 1):
            time_effect = 0.5 * t
            te = 0.0
            if g_assign[unit] > 0 and t >= g_assign[unit]:
                te = 1.0 * (t - g_assign[unit] + 1)
            y_val = unit_fe[unit] + time_effect + te + rng.normal(0, 0.5)
            rows.append({
                'unit': unit,
                'time': t,
                'y': y_val,
                'g': g_assign[unit],
                'x1': rng.normal(0, 1),
            })

    return pd.DataFrame(rows)


# ======================================================================
# 2x2 DID tests
# ======================================================================

class TestDID2x2:
    """Tests for 2×2 DID estimator."""

    def test_basic_att(self, data_2x2):
        """ATT should be close to true value of 5.0."""
        result = did_2x2(data_2x2, y='y', treat='treat', time='post')
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 5.0) < 0.5

    def test_returns_causal_result(self, data_2x2):
        """Result should be a CausalResult with expected attributes."""
        result = did_2x2(data_2x2, y='y', treat='treat', time='post')
        assert result.method == 'Difference-in-Differences (2x2)'
        assert result.estimand == 'ATT'
        assert result.se > 0
        assert result.ci[0] < result.estimate < result.ci[1]
        assert 0 <= result.pvalue <= 1
        assert result.n_obs == len(data_2x2)

    def test_summary_output(self, data_2x2):
        """Summary should be a non-empty string with key info."""
        result = did_2x2(data_2x2, y='y', treat='treat', time='post')
        summary = result.summary()
        assert isinstance(summary, str)
        assert 'ATT' in summary
        assert '2x2' in summary

    def test_robust_vs_nonrobust(self, data_2x2):
        """Robust and non-robust SEs should differ."""
        r1 = did_2x2(data_2x2, y='y', treat='treat', time='post', robust=True)
        r2 = did_2x2(data_2x2, y='y', treat='treat', time='post', robust=False)
        assert abs(r1.estimate - r2.estimate) < 1e-10  # point estimates same
        assert r1.se != r2.se  # SEs differ

    def test_cluster_se(self, data_2x2):
        """Clustered SEs should run without error."""
        result = did_2x2(data_2x2, y='y', treat='treat', time='post',
                         cluster='cluster_id')
        assert result.se > 0

    def test_with_covariates(self, data_2x2):
        """Adding covariates should not break estimation."""
        result = did_2x2(data_2x2, y='y', treat='treat', time='post',
                         covariates=['x1'])
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 5.0) < 0.5

    def test_detail_table(self, data_2x2):
        """Detail should contain all coefficient info."""
        result = did_2x2(data_2x2, y='y', treat='treat', time='post')
        assert result.detail is not None
        assert 'variable' in result.detail.columns
        assert 'coefficient' in result.detail.columns
        assert len(result.detail) >= 4  # const, treat, post, interaction

    def test_to_latex(self, data_2x2):
        """LaTeX export should produce valid LaTeX."""
        result = did_2x2(data_2x2, y='y', treat='treat', time='post')
        latex = result.to_latex()
        assert '\\begin{table}' in latex
        assert '\\end{table}' in latex
        assert 'treatxpost' in latex  # DID interaction coefficient

    def test_cite(self, data_2x2):
        """Citation should return BibTeX."""
        result = did_2x2(data_2x2, y='y', treat='treat', time='post')
        bib = result.cite()
        assert '@book' in bib or '@article' in bib

    def test_unified_did_dispatches_2x2(self, data_2x2):
        """did() with binary treatment and no id should dispatch to 2x2."""
        result = did(data_2x2, y='y', treat='treat', time='post')
        assert '2x2' in result.method

    def test_invalid_treat_values(self, data_2x2):
        """Non-binary treatment without id should raise error."""
        df = data_2x2.copy()
        df['treat'] = np.random.choice([0, 1, 2], len(df))
        with pytest.raises(ValueError):
            did(df, y='y', treat='treat', time='post')


# ======================================================================
# Callaway-Sant'Anna tests
# ======================================================================

class TestCallawayStantanna:
    """Tests for staggered DID (Callaway & Sant'Anna 2021)."""

    def test_basic_estimation(self, data_staggered):
        """Should estimate without errors and return CausalResult."""
        result = callaway_santanna(
            data_staggered, y='y', g='g', t='time', i='unit'
        )
        assert isinstance(result, CausalResult)
        assert result.estimate > 0  # treatment effect is positive in DGP

    def test_att_magnitude(self, data_staggered):
        """Aggregated ATT should be in reasonable range.

        True simple-average post-treatment ATT ≈ 2.625.
        """
        result = callaway_santanna(
            data_staggered, y='y', g='g', t='time', i='unit'
        )
        assert 1.0 < result.estimate < 5.0

    def test_group_time_effects(self, data_staggered):
        """Should produce group-time ATT estimates."""
        result = callaway_santanna(
            data_staggered, y='y', g='g', t='time', i='unit'
        )
        assert result.detail is not None
        assert 'group' in result.detail.columns
        assert 'time' in result.detail.columns
        assert 'att' in result.detail.columns
        assert 'se' in result.detail.columns
        assert len(result.detail) > 0

    def test_event_study(self, data_staggered):
        """Event study estimates should be available."""
        result = callaway_santanna(
            data_staggered, y='y', g='g', t='time', i='unit'
        )
        assert 'event_study' in result.model_info
        es = result.model_info['event_study']
        assert 'relative_time' in es.columns
        assert 'att' in es.columns

        # Post-treatment effects should be positive
        post_es = es[es['relative_time'] >= 0]
        assert (post_es['att'] > 0).all()

    def test_event_study_increasing(self, data_staggered):
        """Event study coefficients should increase (dynamic effect in DGP)."""
        result = callaway_santanna(
            data_staggered, y='y', g='g', t='time', i='unit'
        )
        es = result.model_info['event_study']
        post = es[es['relative_time'] >= 0].sort_values('relative_time')
        if len(post) > 1:
            atts = post['att'].values
            # Roughly increasing
            assert atts[-1] > atts[0]

    def test_pretrend_test(self, data_staggered):
        """Pre-trend test should not reject (parallel trends hold in DGP)."""
        result = callaway_santanna(
            data_staggered, y='y', g='g', t='time', i='unit'
        )
        pt = result.pretrend_test()
        assert 'pvalue' in pt
        # Should fail to reject (DGP has parallel trends).
        # Use a very loose threshold since this is stochastic with n=300.
        assert pt['pvalue'] > 0.001

    def test_pre_treatment_near_zero(self, data_staggered):
        """Pre-treatment ATT(g,t) should be close to zero."""
        result = callaway_santanna(
            data_staggered, y='y', g='g', t='time', i='unit'
        )
        pre = result.detail[result.detail['relative_time'] < 0]
        if len(pre) > 0:
            assert abs(pre['att'].mean()) < 0.5

    def test_estimator_dr(self, data_staggered):
        """DR estimator should work."""
        result = callaway_santanna(
            data_staggered, y='y', g='g', t='time', i='unit',
            estimator='dr'
        )
        assert result.estimate > 0

    def test_estimator_ipw(self, data_staggered):
        """IPW estimator should work."""
        result = callaway_santanna(
            data_staggered, y='y', g='g', t='time', i='unit',
            estimator='ipw'
        )
        assert result.estimate > 0

    def test_estimator_reg(self, data_staggered):
        """REG estimator should work."""
        result = callaway_santanna(
            data_staggered, y='y', g='g', t='time', i='unit',
            estimator='reg'
        )
        assert result.estimate > 0

    def test_not_yet_treated(self, data_staggered):
        """Not-yet-treated comparison group should work."""
        result = callaway_santanna(
            data_staggered, y='y', g='g', t='time', i='unit',
            control_group='notyettreated'
        )
        assert isinstance(result, CausalResult)
        assert result.estimate > 0

    def test_with_covariates(self, data_staggered):
        """Conditional DID with covariates should work."""
        result = callaway_santanna(
            data_staggered, y='y', g='g', t='time', i='unit',
            x=['x1']
        )
        assert isinstance(result, CausalResult)
        assert result.estimate > 0

    def test_summary_output(self, data_staggered):
        """Summary should contain key information."""
        result = callaway_santanna(
            data_staggered, y='y', g='g', t='time', i='unit'
        )
        summary = result.summary()
        assert "Callaway" in summary
        assert "ATT" in summary
        assert "Event Study" in summary

    def test_to_latex(self, data_staggered):
        """LaTeX export should work."""
        result = callaway_santanna(
            data_staggered, y='y', g='g', t='time', i='unit'
        )
        latex = result.to_latex()
        assert '\\begin{table}' in latex

    def test_cite(self, data_staggered):
        """Citation should return Callaway & Sant'Anna BibTeX."""
        result = callaway_santanna(
            data_staggered, y='y', g='g', t='time', i='unit'
        )
        bib = result.cite()
        assert 'callaway2021' in bib

    def test_confidence_intervals(self, data_staggered):
        """CIs should contain the point estimate."""
        result = callaway_santanna(
            data_staggered, y='y', g='g', t='time', i='unit'
        )
        assert result.ci[0] < result.estimate < result.ci[1]

        # Group-time CIs
        for _, row in result.detail.iterrows():
            assert row['ci_lower'] <= row['att'] <= row['ci_upper']

    def test_unified_did_dispatches_cs(self, data_staggered):
        """did() with id provided should dispatch to C&S."""
        result = did(data_staggered, y='y', treat='g', time='time', id='unit')
        assert "Callaway" in result.method

    def test_repr(self, data_staggered):
        """__repr__ should work."""
        result = callaway_santanna(
            data_staggered, y='y', g='g', t='time', i='unit'
        )
        r = repr(result)
        assert 'CausalResult' in r
        assert 'ATT' in r

    def test_backward_compat_params(self, data_staggered):
        """params/std_errors properties should work for outreg2 compat."""
        result = callaway_santanna(
            data_staggered, y='y', g='g', t='time', i='unit'
        )
        assert 'ATT' in result.params.index
        assert result.params['ATT'] == result.estimate
        assert result.std_errors['ATT'] == result.se


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
