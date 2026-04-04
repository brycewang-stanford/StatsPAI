"""Tests for robustness_report()."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 500
    education = rng.normal(12, 3, n)
    experience = rng.normal(10, 5, n).clip(0)
    female = rng.binomial(1, 0.5, n)
    age = rng.normal(35, 8, n).clip(18)
    region = rng.choice(['East', 'West', 'South', 'North'], n)
    wage = (5 + 1.5 * education + 0.8 * experience
            - 0.5 * female + rng.normal(0, 3, n))
    return pd.DataFrame({
        'wage': wage,
        'education': education,
        'experience': experience,
        'female': female,
        'age': age,
        'region': region,
    })


class TestRobustnessReportBasic:

    def test_minimal(self, sample_data):
        import statspai as sp
        report = sp.robustness_report(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
        )
        assert report.n_checks >= 2  # baseline + OLS SE
        assert abs(report.baseline_estimate - 1.5) < 0.5

    def test_with_cluster(self, sample_data):
        import statspai as sp
        report = sp.robustness_report(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
            cluster_var='region',
        )
        checks = report.results_df['check'].tolist()
        assert any('Clustered' in c for c in checks)

    def test_extra_controls(self, sample_data):
        import statspai as sp
        report = sp.robustness_report(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
            extra_controls=['female', 'age'],
        )
        checks = report.results_df['check'].tolist()
        assert '+ female' in checks
        assert '+ age' in checks

    def test_drop_controls(self, sample_data):
        import statspai as sp
        report = sp.robustness_report(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
            drop_controls=['experience'],
        )
        checks = report.results_df['check'].tolist()
        assert '- experience' in checks

    def test_winsorize(self, sample_data):
        import statspai as sp
        report = sp.robustness_report(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
            winsor_levels=[0.01, 0.05],
        )
        checks = report.results_df['check'].tolist()
        assert any('Winsorize' in c for c in checks)
        assert report.n_checks >= 4  # baseline + OLS + 2 winsor

    def test_trim(self, sample_data):
        import statspai as sp
        report = sp.robustness_report(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
            trim_pct=0.01,
        )
        checks = report.results_df['check'].tolist()
        assert any('Trim' in c for c in checks)

    def test_subsets(self, sample_data):
        import statspai as sp
        report = sp.robustness_report(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
            subsets={
                'Male': sample_data['female'] == 0,
                'Female': sample_data['female'] == 1,
            },
        )
        checks = report.results_df['check'].tolist()
        assert 'Sub: Male' in checks
        assert 'Sub: Female' in checks


class TestRobustnessReportOutputs:

    def test_summary(self, sample_data):
        import statspai as sp
        report = sp.robustness_report(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
        )
        s = report.summary()
        assert 'Robustness Report' in s
        assert 'Baseline' in s
        assert 'Stability Assessment' in s

    def test_to_latex(self, sample_data):
        import statspai as sp
        report = sp.robustness_report(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
        )
        latex = report.to_latex()
        assert r'\begin{table}' in latex

    def test_plot(self, sample_data):
        import statspai as sp
        import matplotlib
        matplotlib.use('Agg')

        report = sp.robustness_report(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
            extra_controls=['female'],
        )
        fig, ax = report.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_all_estimates_positive(self, sample_data):
        """True effect is 1.5, should be positive in all checks."""
        import statspai as sp
        report = sp.robustness_report(
            data=sample_data,
            formula="wage ~ education + experience",
            x='education',
            extra_controls=['female', 'age'],
            winsor_levels=[0.01, 0.05],
            trim_pct=0.02,
        )
        assert (report.results_df['estimate'] > 0).all()
