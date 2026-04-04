"""
Tests for Entropy Balancing (Hainmueller 2012).
"""

import pytest
import numpy as np
import pandas as pd

from statspai.matching.ebalance import ebalance
from statspai.core.results import CausalResult


@pytest.fixture
def confounded_data():
    """Observational data with selection on observables. True ATT = 3."""
    rng = np.random.default_rng(42)
    n = 2000
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    # Selection: more likely treated if x1 > 0
    p = 1 / (1 + np.exp(-(0.8 * x1 + 0.4 * x2)))
    d = rng.binomial(1, p, n)
    y = 1 + x1 + 0.5 * x2 + 3.0 * d + rng.normal(0, 1, n)
    return pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})


class TestEbalance:
    def test_basic_run(self, confounded_data):
        result = ebalance(confounded_data, y='y', treat='d',
                          covariates=['x1', 'x2'])
        assert isinstance(result, CausalResult)
        assert 'Entropy' in result.method

    def test_att_positive(self, confounded_data):
        """ATT should be positive (true = 3)."""
        result = ebalance(confounded_data, y='y', treat='d',
                          covariates=['x1', 'x2'])
        assert result.estimate > 0

    def test_att_magnitude(self, confounded_data):
        """ATT should be close to 3.0."""
        result = ebalance(confounded_data, y='y', treat='d',
                          covariates=['x1', 'x2'])
        assert abs(result.estimate - 3.0) < 1.0

    def test_balance_improved(self, confounded_data):
        """SMD after balancing should be smaller than before."""
        result = ebalance(confounded_data, y='y', treat='d',
                          covariates=['x1', 'x2'])
        balance = result.model_info['balance']
        for _, row in balance.iterrows():
            assert abs(row['smd_after']) < abs(row['smd_before']) + 0.01

    def test_near_exact_balance(self, confounded_data):
        """After entropy balancing, SMD should be near zero."""
        result = ebalance(confounded_data, y='y', treat='d',
                          covariates=['x1', 'x2'])
        balance = result.model_info['balance']
        assert (balance['smd_after'].abs() < 0.05).all()

    def test_weights_sum_to_one(self, confounded_data):
        result = ebalance(confounded_data, y='y', treat='d',
                          covariates=['x1', 'x2'])
        w = result.model_info['weights']
        assert abs(np.sum(w) - 1.0) < 0.001

    def test_weights_positive(self, confounded_data):
        result = ebalance(confounded_data, y='y', treat='d',
                          covariates=['x1', 'x2'])
        w = result.model_info['weights']
        assert (w >= 0).all()

    def test_second_moments(self, confounded_data):
        """Balancing on variance (moments=2)."""
        result = ebalance(confounded_data, y='y', treat='d',
                          covariates=['x1', 'x2'], moments=2)
        assert isinstance(result, CausalResult)

    def test_effective_sample_size(self, confounded_data):
        result = ebalance(confounded_data, y='y', treat='d',
                          covariates=['x1', 'x2'])
        ess = result.model_info['eff_sample_size']
        assert ess > 1  # should be reasonable
        n_c = result.model_info['n_control']
        assert ess <= n_c  # can't exceed total controls

    def test_cite(self, confounded_data):
        result = ebalance(confounded_data, y='y', treat='d',
                          covariates=['x1', 'x2'])
        assert 'hainmueller' in result.cite().lower()

    def test_summary(self, confounded_data):
        result = ebalance(confounded_data, y='y', treat='d',
                          covariates=['x1', 'x2'])
        s = result.summary()
        assert 'ATT' in s


class TestIntegration:
    def test_import(self):
        import statspai as sp
        assert hasattr(sp, 'ebalance')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
