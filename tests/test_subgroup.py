"""Tests for subgroup_analysis()."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 600
    education = rng.normal(12, 3, n)
    experience = rng.normal(10, 5, n).clip(0)
    female = rng.binomial(1, 0.5, n)
    region = rng.choice(['East', 'West', 'South'], n)
    # True effect: 1.5 for male, 1.0 for female (heterogeneity)
    effect = np.where(female, 1.0, 1.5)
    wage = (5 + effect * education + 0.8 * experience
            + rng.normal(0, 3, n))
    return pd.DataFrame({
        'wage': wage,
        'education': education,
        'experience': experience,
        'female': female,
        'region': region,
    })


class TestSubgroupBasic:

    def test_single_grouping(self, sample_data):
        import statspai as sp
        result = sp.subgroup_analysis(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
            by={'Gender': 'female'},
        )
        assert len(result.results_df) == 2  # female=0, female=1
        assert result.overall_estimate is not None

    def test_multiple_groupings(self, sample_data):
        import statspai as sp
        result = sp.subgroup_analysis(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
            by={'Gender': 'female', 'Region': 'region'},
        )
        # 2 gender groups + 3 region groups = 5
        assert len(result.results_df) == 5

    def test_het_test_present(self, sample_data):
        import statspai as sp
        result = sp.subgroup_analysis(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
            by={'Gender': 'female'},
        )
        assert 'Gender' in result.het_tests
        ht = result.het_tests['Gender']
        assert 'chi2' in ht
        assert 'pvalue' in ht
        assert 'df' in ht

    def test_detects_heterogeneity(self, sample_data):
        """True heterogeneity by gender should be detected."""
        import statspai as sp
        result = sp.subgroup_analysis(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
            by={'Gender': 'female'},
        )
        # With n=600 and effect diff of 0.5, should detect het
        ht = result.het_tests['Gender']
        assert ht['pvalue'] < 0.10  # at least marginally significant

    def test_subgroup_estimates_reasonable(self, sample_data):
        """Male effect ~1.5, female effect ~1.0."""
        import statspai as sp
        result = sp.subgroup_analysis(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
            by={'Gender': 'female'},
        )
        df = result.results_df
        male_est = df[df['group_val'] == '0']['estimate'].iloc[0]
        female_est = df[df['group_val'] == '1']['estimate'].iloc[0]
        assert abs(male_est - 1.5) < 0.4
        assert abs(female_est - 1.0) < 0.4
        assert male_est > female_est  # direction correct


class TestSubgroupOutputs:

    def test_summary(self, sample_data):
        import statspai as sp
        result = sp.subgroup_analysis(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
            by={'Gender': 'female'},
        )
        s = result.summary()
        assert 'Subgroup Heterogeneity' in s
        assert 'Heterogeneity test' in s

    def test_to_latex(self, sample_data):
        import statspai as sp
        result = sp.subgroup_analysis(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
            by={'Gender': 'female'},
        )
        latex = result.to_latex()
        assert r'\begin{table}' in latex
        assert 'Heterogeneity' in latex

    def test_plot(self, sample_data):
        import statspai as sp
        import matplotlib
        matplotlib.use('Agg')

        result = sp.subgroup_analysis(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
            by={'Gender': 'female', 'Region': 'region'},
        )
        fig, ax = result.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_results_df_columns(self, sample_data):
        import statspai as sp
        result = sp.subgroup_analysis(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
            by={'Gender': 'female'},
        )
        expected = {
            'group_var', 'group_val', 'estimate', 'se',
            'ci_lower', 'ci_upper', 'pvalue', 'nobs', 'label',
        }
        assert expected.issubset(set(result.results_df.columns))


class TestSubgroupEdgeCases:

    def test_bad_column_raises(self, sample_data):
        import statspai as sp
        with pytest.raises(ValueError, match="not found"):
            sp.subgroup_analysis(
                data=sample_data,
                formula="wage ~ education + experience",
                x='education',
                by={'Bad': 'nonexistent'},
            )

    def test_group_var_in_controls(self, sample_data):
        """If group var is also a control, it should be removed for subgroup."""
        import statspai as sp
        result = sp.subgroup_analysis(
            data=sample_data,
            formula="wage ~ education + experience + female",
            x='education',
            by={'Gender': 'female'},
        )
        assert len(result.results_df) == 2
