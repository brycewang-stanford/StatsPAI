"""
Tests for Cinelli-Hazlett sensemakr and all new modules.
"""

import pytest
import numpy as np
import pandas as pd

from statspai.diagnostics.sensemakr import sensemakr


@pytest.fixture
def confounded_data():
    """Data with confounding. education → wage, but ability confounds."""
    rng = np.random.default_rng(42)
    n = 2000
    ability = rng.normal(0, 1, n)
    education = 12 + 0.5 * ability + rng.normal(0, 2, n)
    experience = rng.normal(10, 3, n)
    wage = 5 + 1.5 * education + 0.8 * experience + 2 * ability + rng.normal(0, 2, n)
    return pd.DataFrame({
        'wage': wage, 'education': education,
        'experience': experience, 'ability': ability,
    })


class TestSensemakr:
    def test_basic_run(self, confounded_data):
        result = sensemakr(confounded_data, y='wage', treat='education',
                           controls=['experience'])
        assert 'rv_q' in result
        assert 'rv_qa' in result
        assert 'interpretation' in result

    def test_rv_positive(self, confounded_data):
        """RV should be positive when effect is significant."""
        result = sensemakr(confounded_data, y='wage', treat='education',
                           controls=['experience'])
        assert result['rv_q'] > 0
        assert result['rv_qa'] > 0

    def test_rv_bounded(self, confounded_data):
        """RV should be between 0 and 1."""
        result = sensemakr(confounded_data, y='wage', treat='education',
                           controls=['experience'])
        assert 0 <= result['rv_q'] <= 1
        assert 0 <= result['rv_qa'] <= 1

    def test_benchmark_table(self, confounded_data):
        result = sensemakr(confounded_data, y='wage', treat='education',
                           controls=['experience'],
                           benchmark=['experience'])
        assert isinstance(result['benchmark_table'], pd.DataFrame)

    def test_with_multiple_controls(self, confounded_data):
        # Include ability as a control (should increase R² and reduce RV)
        result = sensemakr(confounded_data, y='wage', treat='education',
                           controls=['experience', 'ability'])
        assert 'rv_q' in result

    def test_robustness_label(self, confounded_data):
        result = sensemakr(confounded_data, y='wage', treat='education',
                           controls=['experience'])
        assert result['robustness'] in ('ROBUST', 'MODERATELY ROBUST', 'FRAGILE')

    def test_interpretation_string(self, confounded_data):
        result = sensemakr(confounded_data, y='wage', treat='education',
                           controls=['experience'])
        assert 'RV' in result['interpretation']


class TestIntegration:
    def test_import(self):
        import statspai as sp
        assert hasattr(sp, 'sensemakr')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
