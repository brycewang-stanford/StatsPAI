"""
Tests for Matching module (PSM, Mahalanobis, CEM).

Uses simulated data with known treatment effects and selection bias.
"""

import pytest
import numpy as np
import pandas as pd
from statspai.matching import match, MatchEstimator
from statspai.core.results import CausalResult


@pytest.fixture
def selection_bias_data():
    """
    DGP with selection on observables:
        X1, X2 ~ Normal
        Treatment: P(T=1) depends on X1, X2 (selection bias)
        Y = 1 + 2*T + 3*X1 + X2 + eps  (true ATT = 2.0)
    """
    rng = np.random.default_rng(42)
    n = 2000

    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.5, n)

    # Selection: higher X1/X2 → more likely treated
    logit = -0.5 + 0.8 * X1 + 0.5 * X2
    prob = 1 / (1 + np.exp(-logit))
    T = rng.binomial(1, prob, n)

    # Outcome with constant treatment effect
    Y = 1 + 2 * T + 3 * X1 + X2 + eps

    return pd.DataFrame({
        'y': Y, 'treat': T, 'x1': X1, 'x2': X2,
        'group': rng.choice(['A', 'B', 'C'], n),
    })


@pytest.fixture
def simple_data():
    """Simple balanced data for basic testing."""
    rng = np.random.default_rng(123)
    n = 500

    X = rng.normal(0, 1, n)
    T = (X > 0).astype(int)
    Y = 1 + 3 * T + 2 * X + rng.normal(0, 0.5, n)

    return pd.DataFrame({'y': Y, 'treat': T, 'x': X})


class TestPSM:

    def test_basic_psm(self, selection_bias_data):
        """PSM should recover ATT ≈ 2.0"""
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'], method='psm',
        )

        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.0, (
            f"PSM ATT = {result.estimate:.2f}, expected ≈ 2.0"
        )

    def test_psm_corrects_naive_bias(self, selection_bias_data):
        """PSM should be closer to 2.0 than naive difference in means."""
        df = selection_bias_data
        naive = df[df['treat'] == 1]['y'].mean() - df[df['treat'] == 0]['y'].mean()

        result = match(df, y='y', treat='treat',
                       covariates=['x1', 'x2'], method='psm')

        assert abs(result.estimate - 2.0) < abs(naive - 2.0), (
            f"PSM ({result.estimate:.2f}) should be closer to 2.0 than "
            f"naive ({naive:.2f})"
        )

    def test_psm_significance(self, selection_bias_data):
        """Effect should be significant."""
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'], method='psm',
        )
        assert result.pvalue < 0.05

    def test_psm_ci(self, selection_bias_data):
        """CI should contain true value."""
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'], method='psm', alpha=0.05,
        )
        assert result.ci[0] < 2.0 < result.ci[1]


class TestMahalanobis:

    def test_basic_mahalanobis(self, selection_bias_data):
        """Mahalanobis matching should recover ATT ≈ 2.0"""
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'], method='mahalanobis',
        )

        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.5


class TestCEM:

    def test_basic_cem(self, selection_bias_data):
        """CEM should recover ATT approximately."""
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'], method='cem',
        )

        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 2.0


class TestMatchGeneral:

    def test_balance_table(self, selection_bias_data):
        """Should produce balance diagnostics."""
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'], method='psm',
        )

        balance = result.model_info['balance']
        assert isinstance(balance, pd.DataFrame)
        assert 'variable' in balance.columns
        assert 'smd' in balance.columns
        assert len(balance) >= 2  # at least x1, x2

    def test_model_info(self, selection_bias_data):
        """Model info should contain matching metadata."""
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'], method='psm',
        )

        info = result.model_info
        assert 'n_treated' in info
        assert 'n_control' in info
        assert info['method'] == 'PSM'

    def test_summary(self, selection_bias_data):
        """Summary should be informative."""
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'], method='psm',
        )
        s = result.summary()
        assert 'Matching' in s
        assert 'ATT' in s

    def test_citation(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'], method='psm',
        )
        assert 'abadie' in result.cite().lower()

    def test_repr(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'], method='psm',
        )
        assert 'CausalResult' in repr(result)

    # --- Error handling ---

    def test_missing_column(self, selection_bias_data):
        with pytest.raises(ValueError, match="not found"):
            match(selection_bias_data, y='nonexistent', treat='treat',
                  covariates=['x1'], method='psm')

    def test_invalid_method(self, selection_bias_data):
        with pytest.raises(ValueError, match="method must be"):
            match(selection_bias_data, y='y', treat='treat',
                  covariates=['x1'], method='invalid')

    def test_invalid_estimand(self, selection_bias_data):
        with pytest.raises(ValueError, match="estimand must be"):
            match(selection_bias_data, y='y', treat='treat',
                  covariates=['x1'], estimand='INVALID')

    def test_non_binary_treatment(self):
        df = pd.DataFrame({
            'y': [1, 2, 3], 'treat': [0, 1, 2], 'x': [1, 2, 3],
        })
        with pytest.raises(ValueError, match="binary"):
            match(df, y='y', treat='treat', covariates=['x'])


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
