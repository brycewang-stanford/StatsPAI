"""
Tests for Causal Mediation Analysis.
"""

import pytest
import numpy as np
import pandas as pd
from statspai.mediation import mediate, MediationAnalysis
from statspai.core.results import CausalResult


@pytest.fixture
def mediation_data():
    """
    DGP with known mediation structure:
        T → M: M = 0.5 + 1.0*T + X + e_m
        M → Y: Y = 1.0 + 0.5*T + 2.0*M + X + e_y

    ACME = a1 * b2 = 1.0 * 2.0 = 2.0
    ADE  = b1 = 0.5
    Total = 2.5
    Prop mediated = 2.0 / 2.5 = 0.8
    """
    rng = np.random.default_rng(42)
    n = 2000

    X = rng.normal(0, 1, n)
    T = rng.binomial(1, 0.5, n).astype(float)
    e_m = rng.normal(0, 0.3, n)
    e_y = rng.normal(0, 0.3, n)

    M = 0.5 + 1.0 * T + X + e_m
    Y = 1.0 + 0.5 * T + 2.0 * M + X + e_y

    return pd.DataFrame({'y': Y, 'treat': T, 'mediator': M, 'x': X})


@pytest.fixture
def no_mediation_data():
    """No mediation: T does not affect M."""
    rng = np.random.default_rng(99)
    n = 1000

    T = rng.binomial(1, 0.5, n).astype(float)
    M = rng.normal(0, 1, n)  # M independent of T
    Y = 3.0 * T + 0.5 * M + rng.normal(0, 0.5, n)

    return pd.DataFrame({'y': Y, 'treat': T, 'mediator': M})


class TestMediationBasic:

    def test_acme_estimate(self, mediation_data):
        """ACME should be ≈ 2.0"""
        result = mediate(
            mediation_data, y='y', treat='treat',
            mediator='mediator', covariates=['x'], n_boot=500,
        )

        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 0.5, (
            f"ACME = {result.estimate:.2f}, expected ≈ 2.0"
        )

    def test_ade(self, mediation_data):
        """ADE should be ≈ 0.5"""
        result = mediate(
            mediation_data, y='y', treat='treat',
            mediator='mediator', covariates=['x'], n_boot=500,
        )
        ade = result.model_info['ade']
        assert abs(ade - 0.5) < 0.5, f"ADE = {ade:.2f}, expected ≈ 0.5"

    def test_total_effect(self, mediation_data):
        """Total = ACME + ADE ≈ 2.5"""
        result = mediate(
            mediation_data, y='y', treat='treat',
            mediator='mediator', covariates=['x'], n_boot=500,
        )
        total = result.model_info['total_effect']
        assert abs(total - 2.5) < 0.5

    def test_prop_mediated(self, mediation_data):
        """Proportion mediated ≈ 0.8"""
        result = mediate(
            mediation_data, y='y', treat='treat',
            mediator='mediator', covariates=['x'], n_boot=500,
        )
        prop = result.model_info['prop_mediated']
        assert abs(prop - 0.8) < 0.2

    def test_no_mediation(self, no_mediation_data):
        """When T doesn't affect M, ACME should be near zero."""
        result = mediate(
            no_mediation_data, y='y', treat='treat',
            mediator='mediator', n_boot=500,
        )
        assert abs(result.estimate) < 0.5

    def test_significance(self, mediation_data):
        result = mediate(
            mediation_data, y='y', treat='treat',
            mediator='mediator', covariates=['x'], n_boot=500,
        )
        assert result.pvalue < 0.05


class TestMediationOutput:

    def test_detail_table(self, mediation_data):
        result = mediate(
            mediation_data, y='y', treat='treat',
            mediator='mediator', n_boot=200,
        )
        detail = result.detail
        assert len(detail) == 4
        effects = detail['effect'].tolist()
        assert 'ACME (indirect)' in effects
        assert 'ADE (direct)' in effects
        assert 'Total Effect' in effects

    def test_summary(self, mediation_data):
        result = mediate(
            mediation_data, y='y', treat='treat',
            mediator='mediator', n_boot=200,
        )
        s = result.summary()
        assert 'Mediation' in s

    def test_citation(self, mediation_data):
        result = mediate(
            mediation_data, y='y', treat='treat',
            mediator='mediator', n_boot=200,
        )
        assert 'imai' in result.cite().lower()

    def test_missing_column(self, mediation_data):
        with pytest.raises(ValueError, match="not found"):
            mediate(mediation_data, y='nonexistent', treat='treat',
                    mediator='mediator')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
