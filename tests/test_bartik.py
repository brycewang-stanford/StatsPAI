"""
Tests for Shift-Share (Bartik) IV module.
"""

import pytest
import numpy as np
import pandas as pd
from statspai.bartik import bartik, BartikIV
from statspai.core.results import EconometricResults


@pytest.fixture
def bartik_data():
    """
    Simulated Bartik data:
    - 100 regions, 5 industries
    - Shares sum to ~1 per region
    - National shocks drive endogenous variable
    - True coefficient on endogenous var = 2.0
    """
    rng = np.random.default_rng(42)
    n_regions = 100
    n_industries = 5

    # Industry shares (Dirichlet → sums to 1)
    shares = rng.dirichlet(np.ones(n_industries), size=n_regions)
    share_df = pd.DataFrame(
        shares,
        columns=[f'ind_{k}' for k in range(n_industries)],
    )

    # National shocks
    true_shocks = rng.normal(0.05, 0.02, n_industries)
    shock_series = pd.Series(true_shocks, index=share_df.columns)

    # Bartik instrument
    B = shares @ true_shocks

    # Endogenous variable (correlated with B + noise)
    eps_endog = rng.normal(0, 0.01, n_regions)
    X_endog = B + eps_endog

    # Outcome
    eps_y = rng.normal(0, 0.02, n_regions)
    Y = 1.0 + 2.0 * X_endog + eps_y

    data = pd.DataFrame({
        'y': Y,
        'x_endog': X_endog,
        'control': rng.normal(0, 1, n_regions),
    })

    return data, share_df, shock_series


class TestBartikBasic:

    def test_basic_bartik(self, bartik_data):
        """Should recover coefficient ≈ 2.0"""
        data, shares, shocks = bartik_data
        result = bartik(data, y='y', endog='x_endog',
                        shares=shares, shocks=shocks)

        assert isinstance(result, EconometricResults)
        assert abs(result.params['x_endog'] - 2.0) < 0.5, (
            f"Bartik IV coef = {result.params['x_endog']:.2f}, expected ≈ 2.0"
        )

    def test_with_covariates(self, bartik_data):
        """Should work with controls."""
        data, shares, shocks = bartik_data
        result = bartik(data, y='y', endog='x_endog',
                        shares=shares, shocks=shocks,
                        covariates=['control'])
        assert result is not None
        assert 'control' in result.params.index

    def test_first_stage_f(self, bartik_data):
        """First-stage F should be large (strong instrument)."""
        data, shares, shocks = bartik_data
        result = bartik(data, y='y', endog='x_endog',
                        shares=shares, shocks=shocks)

        f = result.diagnostics['First-stage F']
        assert f > 10, f"First-stage F = {f:.1f}, should be > 10"


class TestBartikDiagnostics:

    def test_rotemberg_weights(self, bartik_data):
        """Rotemberg weights should be available."""
        data, shares, shocks = bartik_data
        model = BartikIV(data=data, y='y', endog='x_endog',
                         shares=shares, shocks=shocks)
        model.fit()

        rot = model.rotemberg_weights
        assert isinstance(rot, pd.DataFrame)
        assert 'industry' in rot.columns
        assert 'weight' in rot.columns
        assert len(rot) == 5

    def test_n_industries(self, bartik_data):
        data, shares, shocks = bartik_data
        result = bartik(data, y='y', endog='x_endog',
                        shares=shares, shocks=shocks)
        assert result.diagnostics['N industries'] == 5


class TestBartikOutput:

    def test_summary(self, bartik_data):
        data, shares, shocks = bartik_data
        result = bartik(data, y='y', endog='x_endog',
                        shares=shares, shocks=shocks)
        s = result.summary()
        assert 'Bartik' in s
        assert 'First-stage F' in s

    def test_repr(self, bartik_data):
        data, shares, shocks = bartik_data
        result = bartik(data, y='y', endog='x_endog',
                        shares=shares, shocks=shocks)
        assert 'Bartik' in repr(result)


class TestBartikErrors:

    def test_missing_column(self, bartik_data):
        data, shares, shocks = bartik_data
        with pytest.raises(ValueError, match="not found"):
            bartik(data, y='nonexistent', endog='x_endog',
                   shares=shares, shocks=shocks)

    def test_mismatched_rows(self, bartik_data):
        data, shares, shocks = bartik_data
        with pytest.raises(ValueError, match="rows"):
            bartik(data.iloc[:50], y='y', endog='x_endog',
                   shares=shares, shocks=shocks)

    def test_no_common_industries(self, bartik_data):
        data, shares, shocks = bartik_data
        bad_shocks = pd.Series([0.1, 0.2], index=['foo', 'bar'])
        with pytest.raises(ValueError, match="No common"):
            bartik(data, y='y', endog='x_endog',
                   shares=shares, shocks=bad_shocks)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
