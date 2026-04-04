"""
Tests for inference module: Wild Cluster Bootstrap and AIPW.
"""

import pytest
import numpy as np
import pandas as pd

from statspai.inference import wild_cluster_bootstrap, aipw
from statspai.core.results import CausalResult


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def clustered_data():
    """Panel data with 20 clusters. True β_treatment = 2.0."""
    rng = np.random.default_rng(42)
    G = 20  # clusters
    n_per = 50  # obs per cluster
    rows = []
    for g in range(G):
        cluster_effect = rng.normal(0, 2)
        for _ in range(n_per):
            x = rng.normal(0, 1)
            y = 1 + 2.0 * x + cluster_effect + rng.normal(0, 1)
            rows.append({'y': y, 'x': x, 'cluster': g})
    return pd.DataFrame(rows)


@pytest.fixture
def few_clusters_data():
    """Data with only 8 clusters — where bootstrap matters most."""
    rng = np.random.default_rng(42)
    G = 8
    n_per = 100
    rows = []
    for g in range(G):
        ce = rng.normal(0, 3)
        for _ in range(n_per):
            x = rng.normal(0, 1)
            y = 1 + 2.0 * x + ce + rng.normal(0, 1)
            rows.append({'y': y, 'x': x, 'cluster': g})
    return pd.DataFrame(rows)


@pytest.fixture
def aipw_data():
    """Observational data with confounding. True ATE = 3.0."""
    rng = np.random.default_rng(42)
    n = 2000
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    # Confounded treatment
    p_treat = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
    d = rng.binomial(1, p_treat, n)
    # Outcome with treatment effect = 3
    y = 1 + 0.5 * x1 + 0.8 * x2 + 3.0 * d + rng.normal(0, 1, n)
    return pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})


# ======================================================================
# Wild Cluster Bootstrap
# ======================================================================

class TestWildClusterBootstrap:
    """Tests for Wild Cluster Bootstrap (Cameron et al. 2008)."""

    def test_basic_run(self, clustered_data):
        """Should run and return expected keys."""
        result = wild_cluster_bootstrap(
            clustered_data, y='y', x=['x'], cluster='cluster',
            test_var='x', n_boot=299, seed=42,
        )
        assert 'p_boot' in result
        assert 'ci_boot' in result
        assert 'beta_hat' in result
        assert 'n_clusters' in result
        assert result['n_clusters'] == 20

    def test_significant_effect(self, clustered_data):
        """True β=2.0 should be significant."""
        result = wild_cluster_bootstrap(
            clustered_data, y='y', x=['x'], cluster='cluster',
            test_var='x', n_boot=499, seed=42,
        )
        assert result['p_boot'] < 0.05
        assert abs(result['beta_hat'] - 2.0) < 0.5

    def test_ci_contains_true(self, clustered_data):
        """Bootstrap CI should contain true value 2.0."""
        result = wild_cluster_bootstrap(
            clustered_data, y='y', x=['x'], cluster='cluster',
            test_var='x', n_boot=499, seed=42,
        )
        assert result['ci_boot'][0] < 2.0 < result['ci_boot'][1]

    def test_few_clusters_warning(self, few_clusters_data):
        """Should warn when G < 6 with Rademacher."""
        # 8 clusters is above 6, so no warning here
        # Let's test with a subset
        tiny = few_clusters_data[few_clusters_data['cluster'] < 5].copy()
        with pytest.warns(UserWarning, match="clusters"):
            wild_cluster_bootstrap(
                tiny, y='y', x=['x'], cluster='cluster',
                n_boot=99, seed=42,
            )

    def test_webb_weights(self, few_clusters_data):
        """Webb weights should work for few clusters."""
        result = wild_cluster_bootstrap(
            few_clusters_data, y='y', x=['x'], cluster='cluster',
            weight_type='webb', n_boot=299, seed=42,
        )
        assert result['weight_type'] == 'webb'
        assert result['p_boot'] < 0.05

    def test_mammen_weights(self, clustered_data):
        """Mammen weights should work."""
        result = wild_cluster_bootstrap(
            clustered_data, y='y', x=['x'], cluster='cluster',
            weight_type='mammen', n_boot=199, seed=42,
        )
        assert result['weight_type'] == 'mammen'

    def test_null_hypothesis(self, clustered_data):
        """Testing H0: β=2 should NOT reject (true value)."""
        result = wild_cluster_bootstrap(
            clustered_data, y='y', x=['x'], cluster='cluster',
            test_var='x', h0=2.0, n_boot=499, seed=42,
        )
        assert result['p_boot'] > 0.05

    def test_recommendation_string(self, clustered_data):
        result = wild_cluster_bootstrap(
            clustered_data, y='y', x=['x'], cluster='cluster',
            n_boot=99, seed=42,
        )
        assert isinstance(result['recommendation'], str)


# ======================================================================
# AIPW
# ======================================================================

class TestAIPW:
    """Tests for AIPW (Doubly Robust) estimator."""

    def test_basic_run(self, aipw_data):
        """Should run and return CausalResult."""
        result = aipw(aipw_data, y='y', treat='d',
                       covariates=['x1', 'x2'], seed=42)
        assert isinstance(result, CausalResult)
        assert result.estimand == 'ATE'

    def test_ate_magnitude(self, aipw_data):
        """ATE should be close to true value 3.0."""
        result = aipw(aipw_data, y='y', treat='d',
                       covariates=['x1', 'x2'], seed=42)
        assert abs(result.estimate - 3.0) < 0.5

    def test_ci_contains_true(self, aipw_data):
        """CI should contain true ATE = 3.0."""
        result = aipw(aipw_data, y='y', treat='d',
                       covariates=['x1', 'x2'], seed=42)
        assert result.ci[0] < 3.0 < result.ci[1]

    def test_att_estimand(self, aipw_data):
        """ATT estimation should work."""
        result = aipw(aipw_data, y='y', treat='d',
                       covariates=['x1', 'x2'], estimand='ATT', seed=42)
        assert result.estimand == 'ATT'
        assert result.estimate > 0

    def test_summary(self, aipw_data):
        result = aipw(aipw_data, y='y', treat='d',
                       covariates=['x1', 'x2'], seed=42)
        s = result.summary()
        assert 'AIPW' in s or 'Doubly Robust' in s

    def test_cite(self, aipw_data):
        result = aipw(aipw_data, y='y', treat='d',
                       covariates=['x1', 'x2'], seed=42)
        bib = result.cite()
        assert 'glynn' in bib.lower() or 'aipw' in bib.lower()

    def test_cross_fitting(self, aipw_data):
        """Different n_folds should give similar results."""
        r3 = aipw(aipw_data, y='y', treat='d',
                   covariates=['x1', 'x2'], n_folds=3, seed=42)
        r10 = aipw(aipw_data, y='y', treat='d',
                    covariates=['x1', 'x2'], n_folds=10, seed=42)
        assert abs(r3.estimate - r10.estimate) < 0.5


# ======================================================================
# Integration
# ======================================================================

class TestIntegration:
    def test_import(self):
        import statspai as sp
        assert hasattr(sp, 'wild_cluster_bootstrap')
        assert hasattr(sp, 'aipw')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
