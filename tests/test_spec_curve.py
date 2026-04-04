"""Tests for specification curve analysis."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """Generate a realistic cross-section dataset."""
    rng = np.random.default_rng(42)
    n = 500
    education = rng.normal(12, 3, n)
    experience = rng.normal(10, 5, n).clip(0)
    female = rng.binomial(1, 0.5, n)
    ability = rng.normal(0, 1, n)
    wage = (5 + 1.5 * education + 0.8 * experience
            - 0.5 * female + 2 * ability + rng.normal(0, 3, n))
    region = rng.choice(['East', 'West', 'South', 'North'], n)
    return pd.DataFrame({
        'wage': wage,
        'education': education,
        'experience': experience,
        'female': female,
        'ability': ability,
        'region': region,
    })


class TestSpecCurveBasic:
    """Basic functionality tests."""

    def test_minimal_call(self, sample_data):
        """spec_curve works with just y, x, and data."""
        import statspai as sp
        result = sp.spec_curve(data=sample_data, y='wage', x='education')
        assert result.n_specs >= 1
        assert result.x == 'education'
        assert result.y == 'wage'

    def test_multiple_control_sets(self, sample_data):
        """Multiple control specifications are enumerated."""
        import statspai as sp
        result = sp.spec_curve(
            data=sample_data,
            y='wage',
            x='education',
            controls=[
                [],
                ['experience'],
                ['experience', 'female'],
            ],
        )
        # 3 control sets × 2 SE types (default) = 6 specs
        assert result.n_specs == 6

    def test_se_types(self, sample_data):
        """Different SE types produce different specifications."""
        import statspai as sp
        result = sp.spec_curve(
            data=sample_data,
            y='wage',
            x='education',
            controls=[['experience']],
            se_types=['nonrobust', 'hc1'],
        )
        assert result.n_specs == 2
        ses = result.results_df['se_type'].unique()
        assert set(ses) == {'nonrobust', 'hc1'}

    def test_subsets(self, sample_data):
        """Subsample specifications work."""
        import statspai as sp
        result = sp.spec_curve(
            data=sample_data,
            y='wage',
            x='education',
            controls=[['experience']],
            se_types=['hc1'],
            subsets={
                'Full': None,
                'Male': sample_data['female'] == 0,
                'Female': sample_data['female'] == 1,
            },
        )
        assert result.n_specs == 3
        assert set(result.results_df['subset']) == {'Full', 'Male', 'Female'}

    def test_cluster_se(self, sample_data):
        """Clustered SE specification is auto-added."""
        import statspai as sp
        result = sp.spec_curve(
            data=sample_data,
            y='wage',
            x='education',
            controls=[['experience']],
            se_types=['hc1'],
            cluster_var='region',
        )
        # hc1 + cluster = 2
        assert result.n_specs == 2
        assert 'cluster' in result.results_df['se_type'].values

    def test_y_transforms(self, sample_data):
        """Outcome transformations expand the grid."""
        import statspai as sp
        df = sample_data.copy()
        df['wage'] = df['wage'].clip(lower=1)  # ensure log is valid
        result = sp.spec_curve(
            data=df,
            y='wage',
            x='education',
            controls=[['experience']],
            se_types=['hc1'],
            y_transforms={
                'Level': None,
                'Log': np.log,
            },
        )
        assert result.n_specs == 2
        transforms = result.results_df['y_transform'].unique()
        assert set(transforms) == {'Level', 'Log'}


class TestSpecCurveResults:
    """Test result properties and methods."""

    def test_result_columns(self, sample_data):
        """results_df has all expected columns."""
        import statspai as sp
        result = sp.spec_curve(
            data=sample_data, y='wage', x='education',
            controls=[[], ['experience']],
        )
        expected = {
            'spec_id', 'estimate', 'se', 'ci_lower', 'ci_upper',
            'pvalue', 'tstat', 'nobs', 'r_squared', 'controls',
            'se_type', 'subset', 'model', 'y_transform', 'significant',
        }
        assert expected.issubset(set(result.results_df.columns))

    def test_share_significant(self, sample_data):
        """share_significant is between 0 and 1."""
        import statspai as sp
        result = sp.spec_curve(
            data=sample_data, y='wage', x='education',
            controls=[[], ['experience'], ['experience', 'female']],
        )
        assert 0 <= result.share_significant <= 1

    def test_share_positive(self, sample_data):
        """Education effect should be positive in most specs."""
        import statspai as sp
        result = sp.spec_curve(
            data=sample_data, y='wage', x='education',
            controls=[[], ['experience'], ['experience', 'female']],
        )
        # True effect is 1.5, should be positive in all specs
        assert result.share_positive == 1.0

    def test_median_estimate(self, sample_data):
        """Median estimate is reasonable."""
        import statspai as sp
        result = sp.spec_curve(
            data=sample_data, y='wage', x='education',
            controls=[[], ['experience'], ['experience', 'female']],
        )
        # True effect is 1.5
        assert 1.0 < result.median_estimate < 2.5

    def test_summary_string(self, sample_data):
        """summary() returns a non-empty string."""
        import statspai as sp
        result = sp.spec_curve(
            data=sample_data, y='wage', x='education',
            controls=[[], ['experience']],
        )
        s = result.summary()
        assert isinstance(s, str)
        assert 'Specification Curve Analysis' in s
        assert 'education' in s

    def test_to_latex(self, sample_data):
        """to_latex() produces valid LaTeX."""
        import statspai as sp
        result = sp.spec_curve(
            data=sample_data, y='wage', x='education',
            controls=[[], ['experience']],
        )
        latex = result.to_latex()
        assert r'\begin{table}' in latex
        assert 'education' in latex

    def test_cite(self, sample_data):
        """cite() returns Simonsohn et al. BibTeX."""
        import statspai as sp
        result = sp.spec_curve(
            data=sample_data, y='wage', x='education',
        )
        bib = result.cite()
        assert 'simonsohn2020specification' in bib
        assert 'Nature Human Behaviour' in bib

    def test_to_dataframe(self, sample_data):
        """to_dataframe() returns a copy."""
        import statspai as sp
        result = sp.spec_curve(
            data=sample_data, y='wage', x='education',
            controls=[[], ['experience']],
        )
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == result.n_specs


class TestSpecCurvePlot:
    """Test plotting (non-interactive)."""

    def test_plot_returns_figure(self, sample_data):
        """plot() returns (fig, axes)."""
        import statspai as sp
        import matplotlib
        matplotlib.use('Agg')

        result = sp.spec_curve(
            data=sample_data, y='wage', x='education',
            controls=[[], ['experience'], ['experience', 'female']],
        )
        fig, axes = result.plot()
        assert fig is not None
        assert len(axes) == 2
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_custom_title(self, sample_data):
        """plot() accepts custom title."""
        import statspai as sp
        import matplotlib
        matplotlib.use('Agg')

        result = sp.spec_curve(
            data=sample_data, y='wage', x='education',
            controls=[[], ['experience']],
        )
        fig, axes = result.plot(title="My Custom Title")
        assert axes[0].get_title() == "My Custom Title"
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestSpecCurveEdgeCases:
    """Edge cases and error handling."""

    def test_all_significant(self, sample_data):
        """Strong effect should be significant in all specs."""
        import statspai as sp
        result = sp.spec_curve(
            data=sample_data, y='wage', x='education',
            controls=[[], ['experience'], ['experience', 'female']],
            se_types=['nonrobust'],
        )
        # True effect 1.5 with n=500 should always be significant
        assert result.share_significant == 1.0

    def test_combinatorial_explosion(self, sample_data):
        """Many combinations run correctly."""
        import statspai as sp
        result = sp.spec_curve(
            data=sample_data,
            y='wage',
            x='education',
            controls=[
                [],
                ['experience'],
                ['female'],
                ['experience', 'female'],
            ],
            se_types=['nonrobust', 'hc1'],
            subsets={
                'Full': None,
                'Male': sample_data['female'] == 0,
            },
        )
        # 4 controls × 2 SE × 2 subsets = 16, but 'female' control
        # on Male-only subset is constant → singular, so 4 specs dropped
        assert result.n_specs == 12

    def test_empty_controls(self, sample_data):
        """Empty control list uses no controls."""
        import statspai as sp
        result = sp.spec_curve(
            data=sample_data, y='wage', x='education',
            controls=[[]],
            se_types=['hc1'],
        )
        assert result.n_specs == 1
        assert result.results_df['controls'].iloc[0] == '(none)'

    def test_bad_variable_raises(self, sample_data):
        """Non-existent variable raises error."""
        import statspai as sp
        with pytest.raises((KeyError, ValueError)):
            sp.spec_curve(
                data=sample_data, y='nonexistent', x='education',
            )
