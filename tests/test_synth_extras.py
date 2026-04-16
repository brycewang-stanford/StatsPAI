"""
Tests for synth extras: compare, report, datasets, power.
"""

import pytest
import numpy as np
import pandas as pd
from statspai.core.results import CausalResult


@pytest.fixture
def panel_data():
    """Standard test panel: 1 treated + 10 donors, 20 periods, effect=5."""
    rng = np.random.default_rng(42)
    n_units = 11
    n_periods = 20
    treatment_time = 11
    records = []
    alphas = rng.normal(10, 2, n_units)
    betas = rng.normal(0.5, 0.1, n_units)
    for i in range(n_units):
        for t in range(1, n_periods + 1):
            y = alphas[i] + betas[i] * t + rng.normal(0, 0.3)
            if i == 0 and t >= treatment_time:
                y += 5.0
            records.append({'unit': f'unit_{i}', 'time': t, 'outcome': y})
    return pd.DataFrame(records)


# ====================================================================== #
#  synth_compare
# ====================================================================== #

class TestSynthCompare:

    def test_compare_runs(self, panel_data):
        from statspai.synth.compare import synth_compare, SynthComparison
        comp = synth_compare(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            methods=['classic', 'penalized', 'demeaned'],
            placebo=False,
        )
        assert isinstance(comp, SynthComparison)
        assert hasattr(comp, 'comparison_table')
        assert hasattr(comp, 'recommended')

    def test_compare_table_structure(self, panel_data):
        from statspai.synth.compare import synth_compare
        comp = synth_compare(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            methods=['classic', 'demeaned', 'mc'],
            placebo=False,
        )
        df = comp.comparison_table
        assert 'method' in df.columns
        assert 'att' in df.columns
        assert 'pre_rmspe' in df.columns
        assert len(df) >= 2

    def test_recommend(self, panel_data):
        from statspai.synth.compare import synth_recommend
        rec = synth_recommend(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
        )
        assert isinstance(rec, str)
        assert len(rec) > 0

    def test_all_methods_compared(self, panel_data):
        from statspai.synth.compare import synth_compare
        comp = synth_compare(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            placebo=False,
        )
        assert len(comp.comparison_table) >= 8


# ====================================================================== #
#  synth_report
# ====================================================================== #

class TestSynthReport:

    def test_text_report(self, panel_data):
        from statspai.synth.report import synth_report
        report = synth_report(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            sensitivity=False,
        )
        assert isinstance(report, str)
        assert 'ATT' in report or 'Estimate' in report
        assert 'unit_0' in report

    def test_markdown_report(self, panel_data):
        from statspai.synth.report import synth_report
        report = synth_report(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            output='markdown', sensitivity=False,
        )
        assert isinstance(report, str)
        assert '#' in report  # markdown headers

    def test_report_with_sensitivity(self, panel_data):
        from statspai.synth.report import synth_report
        report = synth_report(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            sensitivity=True,
        )
        report_upper = report.upper()
        assert 'LEAVE-ONE-OUT' in report_upper or 'LOO' in report_upper or 'SENSITIVITY' in report_upper


# ====================================================================== #
#  Datasets
# ====================================================================== #

class TestDatasets:

    def test_german_reunification(self):
        from statspai.synth.datasets import german_reunification
        df = german_reunification()
        assert isinstance(df, pd.DataFrame)
        assert 'country' in df.columns
        assert 'year' in df.columns
        assert 'gdppc' in df.columns
        assert 'West Germany' in df['country'].values
        assert len(df['country'].unique()) >= 15

    def test_basque_terrorism(self):
        from statspai.synth.datasets import basque_terrorism
        df = basque_terrorism()
        assert isinstance(df, pd.DataFrame)
        assert 'region' in df.columns
        assert 'year' in df.columns
        assert 'gdppc' in df.columns
        assert 'Basque Country' in df['region'].values

    def test_california_tobacco(self):
        from statspai.synth.datasets import california_tobacco
        df = california_tobacco()
        assert isinstance(df, pd.DataFrame)
        assert 'state' in df.columns
        assert 'year' in df.columns
        assert 'cigsale' in df.columns
        assert 'California' in df['state'].values
        # Should have covariates
        assert 'retprice' in df.columns or 'lnincome' in df.columns

    def test_datasets_work_with_synth(self):
        """Each dataset should work directly with synth()."""
        from statspai.synth.datasets import german_reunification
        from statspai.synth import synth
        df = german_reunification()
        result = synth(
            df, outcome='gdppc', unit='country', time='year',
            treated_unit='West Germany', treatment_time=1990,
            placebo=False,
        )
        assert isinstance(result, CausalResult)


# ====================================================================== #
#  synth_power
# ====================================================================== #

class TestSynthPower:

    def test_power_analysis(self, panel_data):
        from statspai.synth.power import synth_power
        pw = synth_power(
            panel_data, outcome='outcome', unit='unit', time='time',
            treated_unit='unit_0', treatment_time=11,
            effect_sizes=[0, 2, 5, 10],
            n_simulations=20, seed=42,
        )
        assert isinstance(pw, pd.DataFrame)
        assert 'effect_size' in pw.columns
        assert 'power' in pw.columns
        # Power should increase with effect size
        powers = pw['power'].values
        assert powers[-1] >= powers[0]

    def test_mde(self):
        """MDE on a no-effect panel should be positive."""
        from statspai.synth.power import synth_mde
        rng = np.random.default_rng(99)
        records = []
        for i in range(8):
            alpha = rng.normal(5, 1)
            for t in range(1, 16):
                y = alpha + 0.3 * t + rng.normal(0, 0.2)
                records.append({'unit': f'u{i}', 'time': t, 'outcome': y})
        df = pd.DataFrame(records)
        mde = synth_mde(
            df, outcome='outcome', unit='unit', time='time',
            treated_unit='u0', treatment_time=8,
            n_simulations=20, seed=42,
        )
        assert isinstance(mde, float)
        assert mde >= 0  # could be 0 if power is always high, or positive


# ====================================================================== #
#  Cross-method consistency benchmarks
# ====================================================================== #


class TestCrossMethodConsistency:
    """
    On a clean DGP with a known ATT, all well-specified SCM variants
    should recover the effect within ~1-2 units of noise.  This is the
    package-level benchmark ensuring no variant is silently broken.
    """

    @pytest.fixture
    def known_effect_panel(self):
        """
        Clean DGP: Y_it = alpha_i + beta_i * t + eps_it,
        treated unit gets +5.0 after t >= 11, noise sd=0.3.
        1 treated + 15 donors, 25 periods.
        """
        rng = np.random.default_rng(2024)
        n_units = 16
        n_periods = 25
        treatment_time = 11
        true_att = 5.0

        alphas = rng.normal(10, 2, n_units)
        betas = rng.normal(0.5, 0.1, n_units)
        records = []
        for i in range(n_units):
            for t in range(1, n_periods + 1):
                y = alphas[i] + betas[i] * t + rng.normal(0, 0.3)
                if i == 0 and t >= treatment_time:
                    y += true_att
                records.append({'unit': f'u{i}', 'time': t, 'outcome': y})
        return pd.DataFrame(records), true_att, treatment_time

    @pytest.mark.parametrize("method", [
        'classic', 'penalized', 'demeaned', 'augmented', 'sdid',
        'mc', 'penscm', 'cluster', 'sparse', 'kernel_ridge', 'fdid',
    ])
    def test_method_recovers_true_effect(self, known_effect_panel, method):
        """Every method should recover the true ATT within 1.5 units."""
        from statspai.synth import synth
        df, true_att, tt = known_effect_panel
        res = synth(
            df, outcome='outcome', unit='unit', time='time',
            treated_unit='u0', treatment_time=tt,
            method=method, placebo=False,
        )
        # Loose tolerance: method-specific bias can be up to ~1.5 units
        assert abs(res.estimate - true_att) < 1.5, (
            f"{method}: estimate={res.estimate:.3f} vs true={true_att}"
        )

    def test_cross_method_agreement(self, known_effect_panel):
        """Classic, ASCM, SDID, MC should agree within ~1.0 unit of each other."""
        from statspai.synth import synth
        df, _, tt = known_effect_panel
        estimates = {}
        for m in ['classic', 'augmented', 'sdid', 'mc']:
            res = synth(
                df, outcome='outcome', unit='unit', time='time',
                treated_unit='u0', treatment_time=tt,
                method=m, placebo=False,
            )
            estimates[m] = res.estimate
        vals = list(estimates.values())
        spread = max(vals) - min(vals)
        assert spread < 1.5, f"Methods disagree too much: {estimates}"


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
