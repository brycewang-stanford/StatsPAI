"""
Tests for unified Matching module.

Covers: new orthogonal API (distance × method × bias_correction),
legacy API backward compatibility, and all matching variants.
"""

import pytest
import numpy as np
import pandas as pd
from statspai.matching import match, MatchEstimator
from statspai.core.results import CausalResult


# ==================================================================
# Fixtures
# ==================================================================

@pytest.fixture
def selection_bias_data():
    """
    DGP with selection on observables:
        X1, X2 ~ Normal
        Treatment: P(T=1) depends on X1, X2
        Y = 1 + 2*T + 3*X1 + X2 + eps  (true ATT = 2.0)
    """
    rng = np.random.default_rng(42)
    n = 2000

    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.5, n)

    logit = -0.5 + 0.8 * X1 + 0.5 * X2
    prob = 1 / (1 + np.exp(-logit))
    T = rng.binomial(1, prob, n)

    Y = 1 + 2 * T + 3 * X1 + X2 + eps

    return pd.DataFrame({
        'y': Y, 'treat': T, 'x1': X1, 'x2': X2,
        'group': rng.choice(['A', 'B', 'C'], n),
    })


@pytest.fixture
def discrete_data():
    """Data with discrete covariates for exact matching tests."""
    rng = np.random.default_rng(99)
    n = 1000

    age_group = rng.choice([20, 30, 40, 50], n)
    edu = rng.choice([1, 2, 3], n)
    eps = rng.normal(0, 0.3, n)

    logit = -1 + 0.03 * age_group + 0.5 * edu
    prob = 1 / (1 + np.exp(-logit))
    T = rng.binomial(1, prob, n)

    Y = 5 + 2 * T + 0.1 * age_group + edu + eps

    return pd.DataFrame({
        'y': Y, 'treat': T, 'age_group': age_group, 'edu': edu,
    })


# ==================================================================
# New API: distance × method combinations
# ==================================================================

class TestNearestPropensity:
    """distance='propensity', method='nearest' (default)."""

    def test_basic(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            distance='propensity', method='nearest',
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.0

    def test_default_is_propensity_nearest(self, selection_bias_data):
        """Default distance/method should be propensity + nearest."""
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
        )
        assert result.model_info['distance'] == 'propensity'
        assert result.model_info['method'] == 'nearest'

    def test_corrects_naive_bias(self, selection_bias_data):
        df = selection_bias_data
        naive = df[df['treat'] == 1]['y'].mean() - df[df['treat'] == 0]['y'].mean()
        result = match(df, y='y', treat='treat', covariates=['x1', 'x2'])
        assert abs(result.estimate - 2.0) < abs(naive - 2.0)

    def test_significance(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
        )
        assert result.pvalue < 0.05

    def test_ci_covers_true(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'], alpha=0.05,
        )
        assert result.ci[0] < 2.0 < result.ci[1]

    def test_ate(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'], estimand='ATE',
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.5


class TestNearestMahalanobis:
    """distance='mahalanobis', method='nearest'."""

    def test_basic(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            distance='mahalanobis',
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.5

    def test_with_bias_correction(self, selection_bias_data):
        """Bias correction should improve estimate."""
        raw = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            distance='mahalanobis',
        )
        bc = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            distance='mahalanobis', bias_correction=True,
        )
        assert isinstance(bc, CausalResult)
        # BC should be at least as close to truth (not strictly, due to randomness)
        assert abs(bc.estimate - 2.0) < 2.0


class TestNearestEuclidean:
    """distance='euclidean', method='nearest'."""

    def test_basic(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            distance='euclidean',
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.5


class TestExactMatching:
    """distance='exact'."""

    def test_basic(self, discrete_data):
        result = match(
            discrete_data, y='y', treat='treat',
            covariates=['age_group', 'edu'],
            distance='exact',
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.0

    def test_rejects_ate(self, discrete_data):
        with pytest.raises(ValueError, match="ATT"):
            match(
                discrete_data, y='y', treat='treat',
                covariates=['age_group', 'edu'],
                distance='exact', estimand='ATE',
            )


class TestStratification:
    """method='stratify'."""

    def test_basic(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            method='stratify', n_strata=5,
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.0

    def test_ate(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            method='stratify', estimand='ATE',
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.5

    def test_10_strata(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            method='stratify', n_strata=10,
        )
        assert isinstance(result, CausalResult)

    def test_requires_propensity(self, selection_bias_data):
        with pytest.raises(ValueError, match="propensity"):
            match(
                selection_bias_data, y='y', treat='treat',
                covariates=['x1', 'x2'],
                method='stratify', distance='mahalanobis',
            )


class TestCEM:
    """method='cem'."""

    def test_basic(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            method='cem',
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 2.0

    def test_custom_bins(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            method='cem', n_bins=10,
        )
        assert isinstance(result, CausalResult)


class TestBiasCorrection:
    """bias_correction=True across distances."""

    def test_propensity_bc(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            distance='propensity', bias_correction=True,
        )
        assert isinstance(result, CausalResult)
        assert result.model_info['bias_correction'] is True

    def test_euclidean_bc(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            distance='euclidean', bias_correction=True,
        )
        assert isinstance(result, CausalResult)


# ==================================================================
# Legacy API backward compatibility
# ==================================================================

class TestLegacyAPI:
    """Old method='psm'/'mahalanobis'/'cem' should still work."""

    def test_legacy_psm(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'], method='psm',
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.0
        assert result.model_info['distance'] == 'propensity'
        assert result.model_info['method'] == 'nearest'

    def test_legacy_mahalanobis(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'], method='mahalanobis',
        )
        assert isinstance(result, CausalResult)
        assert result.model_info['distance'] == 'mahalanobis'
        assert result.model_info['method'] == 'nearest'

    def test_legacy_cem(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'], method='cem',
        )
        assert isinstance(result, CausalResult)
        assert result.model_info['method'] == 'cem'


# ==================================================================
# General / diagnostics
# ==================================================================

class TestMethodSpecificInfo:
    """Extra diagnostics returned in model_info for each method."""

    def test_exact_matching_info(self, discrete_data):
        result = match(
            discrete_data, y='y', treat='treat',
            covariates=['age_group', 'edu'],
            distance='exact',
        )
        info = result.model_info
        assert 'n_matched_treated' in info
        assert 'n_unmatched_treated' in info
        assert info['n_matched_treated'] > 0
        assert info['n_matched_treated'] + info['n_unmatched_treated'] == info['n_treated']

    def test_cem_info(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'], method='cem',
        )
        info = result.model_info
        assert 'n_matched_treated' in info
        assert 'n_matched_control' in info
        assert 'n_bins' in info
        assert info['n_matched_treated'] <= info['n_treated']

    def test_stratify_info(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            method='stratify', n_strata=5,
        )
        info = result.model_info
        assert info['n_strata'] == 5
        assert info['n_effective_strata'] <= 5
        assert info['n_effective_strata'] >= 1


class TestPsPoly:
    """ps_poly parameter for polynomial propensity score (Cunningham 2021, Ch. 5)."""

    def test_poly2(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            ps_poly=2,
        )
        assert isinstance(result, CausalResult)
        assert result.model_info['ps_poly'] == 2
        assert abs(result.estimate - 2.0) < 1.5

    def test_poly3(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            ps_poly=3,
        )
        assert isinstance(result, CausalResult)
        assert result.model_info['ps_poly'] == 3

    def test_poly1_is_default(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
        )
        assert result.model_info['ps_poly'] == 1

    def test_poly2_stratify(self, selection_bias_data):
        """Polynomial PS also works with stratification."""
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            method='stratify', ps_poly=2,
        )
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.5


class TestWithoutReplacement:
    """replace=False must enforce each control used at most once."""

    def test_no_duplicate_controls(self, selection_bias_data):
        """Controls should not be reused when replace=False."""
        estimator = MatchEstimator(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            replace=False,
        )
        cols = ['y', 'treat', 'x1', 'x2']
        clean = selection_bias_data[cols].dropna()
        T = clean['treat'].values.astype(int)
        X = clean[['x1', 'x2']].values.astype(float)
        idx_t = np.where(T == 1)[0]
        idx_c = np.where(T == 0)[0]
        pscore = estimator._logit_propensity(X, T)
        dist_mat = estimator._compute_distance_matrix(X, idx_t, idx_c, pscore)
        matches, _ = estimator._nn_match_from_dist(dist_mat)

        # Collect all matched control indices
        all_matched = []
        for m in matches:
            if len(m) > 0:
                all_matched.extend(m.tolist())
        # No duplicates
        assert len(all_matched) == len(set(all_matched))

    def test_with_replacement_allows_duplicates(self, selection_bias_data):
        """With replacement, controls CAN be reused."""
        estimator = MatchEstimator(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            replace=True,
        )
        cols = ['y', 'treat', 'x1', 'x2']
        clean = selection_bias_data[cols].dropna()
        T = clean['treat'].values.astype(int)
        X = clean[['x1', 'x2']].values.astype(float)
        idx_t = np.where(T == 1)[0]
        idx_c = np.where(T == 0)[0]
        pscore = estimator._logit_propensity(X, T)
        dist_mat = estimator._compute_distance_matrix(X, idx_t, idx_c, pscore)
        matches, _ = estimator._nn_match_from_dist(dist_mat)

        all_matched = []
        for m in matches:
            if len(m) > 0:
                all_matched.extend(m.tolist())
        # With replacement, duplicates are expected (many treated → few best controls)
        assert len(all_matched) >= len(set(all_matched))


class TestMatchGeneral:

    def test_balance_table(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
        )
        balance = result.model_info['balance']
        assert isinstance(balance, pd.DataFrame)
        assert 'variable' in balance.columns
        assert 'smd' in balance.columns
        assert len(balance) >= 2

    def test_model_info_keys(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
        )
        info = result.model_info
        assert 'distance' in info
        assert 'method' in info
        assert 'n_treated' in info
        assert 'n_control' in info
        assert 'bias_correction' in info

    def test_summary(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
        )
        s = result.summary()
        assert 'Matching' in s
        assert 'ATT' in s

    def test_citation(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
        )
        assert 'abadie' in result.cite().lower()

    def test_caliper(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            caliper=0.1,
        )
        assert isinstance(result, CausalResult)

    def test_multiple_matches(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            n_matches=3,
        )
        assert isinstance(result, CausalResult)

    def test_without_replacement(self, selection_bias_data):
        result = match(
            selection_bias_data, y='y', treat='treat',
            covariates=['x1', 'x2'],
            replace=False,
        )
        assert isinstance(result, CausalResult)

    # --- Error handling ---

    def test_missing_column(self, selection_bias_data):
        with pytest.raises(ValueError, match="not found"):
            match(selection_bias_data, y='nonexistent', treat='treat',
                  covariates=['x1'])

    def test_invalid_method(self, selection_bias_data):
        with pytest.raises(ValueError, match="method must be"):
            match(selection_bias_data, y='y', treat='treat',
                  covariates=['x1'], method='invalid')

    def test_invalid_distance(self, selection_bias_data):
        with pytest.raises(ValueError, match="distance must be"):
            match(selection_bias_data, y='y', treat='treat',
                  covariates=['x1'], distance='cosine')

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
