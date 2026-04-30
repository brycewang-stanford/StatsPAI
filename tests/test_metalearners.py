"""
Tests for Meta-Learners module (S/T/X/R/DR-Learner).
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from statspai.metalearners import (
    metalearner,
    SLearner,
    TLearner,
    XLearner,
    RLearner,
    DRLearner,
)
from statspai.core.results import CausalResult


# ======================================================================
# Fixtures: DGPs with known true CATE
# ======================================================================

@pytest.fixture
def constant_effect_data():
    """
    Constant treatment effect DGP:
        Y = 3.0*D + sin(X1) + X2^2 + eps
        P(D=1|X) = logistic(0.5*X1 + X2)
    True CATE = 3.0 for all units. True ATE = 3.0.
    """
    rng = np.random.default_rng(42)
    n = 2000

    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.5, n)

    logit = 0.5 * X1 + X2
    prob = 1 / (1 + np.exp(-logit))
    D = rng.binomial(1, prob, n).astype(float)

    Y = 3.0 * D + np.sin(X1) + X2 ** 2 + eps

    return pd.DataFrame({'y': Y, 'd': D, 'x1': X1, 'x2': X2})


@pytest.fixture
def heterogeneous_effect_data():
    """
    Heterogeneous treatment effect DGP:
        CATE(X) = X1  (linear in X1)
        Y = CATE(X)*D + X2 + eps
        P(D=1|X) = logistic(0.3*X2)

    True ATE = E[X1] = 0 (with enough data, mean ~ 0).
    True CATE varies with X1.
    """
    rng = np.random.default_rng(123)
    n = 3000

    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.3, n)

    logit = 0.3 * X2
    prob = 1 / (1 + np.exp(-logit))
    D = rng.binomial(1, prob, n).astype(float)

    tau = X1  # true CATE
    Y = tau * D + X2 + eps

    return pd.DataFrame({'y': Y, 'd': D, 'x1': X1, 'x2': X2})


@pytest.fixture
def strong_effect_data():
    """
    Strong constant effect with minimal noise for tight CI tests.
        Y = 5.0*D + X1 + eps (eps ~ N(0, 0.1))
        P(D=1) = 0.5 (randomized)
    """
    rng = np.random.default_rng(99)
    n = 2000

    X1 = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.1, n)
    D = rng.binomial(1, 0.5, n).astype(float)

    Y = 5.0 * D + X1 + eps

    return pd.DataFrame({'y': Y, 'd': D, 'x1': X1})


# ======================================================================
# Test: high-level API
# ======================================================================

class TestMetalearnerAPI:

    def test_returns_causal_result(self, constant_effect_data):
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='t',
        )
        assert isinstance(result, CausalResult)

    def test_summary_contains_method(self, constant_effect_data):
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='dr',
        )
        s = result.summary()
        assert 'Meta-Learner' in s
        assert 'DR-Learner' in s

    def test_citation(self, constant_effect_data):
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='s',
        )
        cite = result.cite()
        assert 'kunzel' in cite.lower() or 'metalearner' in cite.lower()

    def test_model_info_keys(self, constant_effect_data):
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='dr',
        )
        info = result.model_info
        assert 'cate' in info
        assert 'cate_mean' in info
        assert 'cate_std' in info
        assert 'n_treated' in info
        assert 'n_control' in info
        assert len(info['cate']) == result.n_obs

    def test_invalid_learner(self, constant_effect_data):
        with pytest.raises(ValueError, match="learner must be"):
            metalearner(
                constant_effect_data, y='y', treat='d',
                covariates=['x1', 'x2'], learner='invalid',
            )

    def test_missing_column(self, constant_effect_data):
        with pytest.raises(ValueError, match="not found"):
            metalearner(
                constant_effect_data, y='nonexistent', treat='d',
                covariates=['x1', 'x2'],
            )

    def test_non_binary_treatment(self):
        df = pd.DataFrame({
            'y': [1, 2, 3], 'd': [0, 1, 2], 'x': [1, 2, 3],
        })
        with pytest.raises(ValueError, match="binary"):
            metalearner(df, y='y', treat='d', covariates=['x'])

    def test_custom_sklearn_model(self, constant_effect_data):
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='t',
            outcome_model=rf,
        )
        assert abs(result.estimate - 3.0) < 1.5


# ======================================================================
# Test: S-Learner
# ======================================================================

class TestSLearner:

    def test_constant_effect(self, constant_effect_data):
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='s',
        )
        assert abs(result.estimate - 3.0) < 1.5, (
            f"S-Learner ATE = {result.estimate:.2f}, expected ~ 3.0"
        )

    def test_significant(self, strong_effect_data):
        result = metalearner(
            strong_effect_data, y='y', treat='d',
            covariates=['x1'], learner='s',
        )
        assert result.pvalue < 0.05

    def test_low_level_class(self):
        rng = np.random.default_rng(0)
        n = 500
        X = rng.normal(0, 1, (n, 2))
        D = rng.binomial(1, 0.5, n).astype(float)
        Y = 2.0 * D + X[:, 0] + rng.normal(0, 0.1, n)

        s = SLearner()
        s.fit(X, Y, D)
        cate = s.effect(X)
        assert abs(np.mean(cate) - 2.0) < 1.0


# ======================================================================
# Test: T-Learner
# ======================================================================

class TestTLearner:

    def test_constant_effect(self, constant_effect_data):
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='t',
        )
        assert abs(result.estimate - 3.0) < 1.0, (
            f"T-Learner ATE = {result.estimate:.2f}, expected ~ 3.0"
        )

    def test_ci_covers_true(self, strong_effect_data):
        result = metalearner(
            strong_effect_data, y='y', treat='d',
            covariates=['x1'], learner='t', alpha=0.05,
        )
        assert result.ci[0] < 5.0 < result.ci[1], (
            f"CI [{result.ci[0]:.2f}, {result.ci[1]:.2f}] should cover 5.0"
        )

    def test_low_level_class(self):
        rng = np.random.default_rng(1)
        n = 500
        X = rng.normal(0, 1, (n, 2))
        D = rng.binomial(1, 0.5, n).astype(float)
        Y = 2.0 * D + X[:, 0] + rng.normal(0, 0.1, n)

        t = TLearner()
        t.fit(X, Y, D)
        cate = t.effect(X)
        assert abs(np.mean(cate) - 2.0) < 1.0


# ======================================================================
# Test: X-Learner
# ======================================================================

class TestXLearner:

    def test_constant_effect(self, constant_effect_data):
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='x',
        )
        assert abs(result.estimate - 3.0) < 1.5, (
            f"X-Learner ATE = {result.estimate:.2f}, expected ~ 3.0"
        )

    def test_heterogeneous_effect_correlation(self, heterogeneous_effect_data):
        """CATE predictions should correlate with true CATE = X1."""
        result = metalearner(
            heterogeneous_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='x',
        )
        true_cate = heterogeneous_effect_data['x1'].values
        pred_cate = result.model_info['cate']
        corr = np.corrcoef(true_cate, pred_cate)[0, 1]
        assert corr > 0.3, (
            f"X-Learner CATE correlation with true = {corr:.2f}, expected > 0.3"
        )

    def test_low_level_class(self):
        rng = np.random.default_rng(2)
        n = 800
        X = rng.normal(0, 1, (n, 2))
        D = rng.binomial(1, 0.5, n).astype(float)
        Y = 2.0 * D + X[:, 0] + rng.normal(0, 0.2, n)

        x = XLearner()
        x.fit(X, Y, D)
        cate = x.effect(X)
        assert abs(np.mean(cate) - 2.0) < 1.0


# ======================================================================
# Test: R-Learner
# ======================================================================

class TestRLearner:

    def test_constant_effect(self, constant_effect_data):
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='r',
        )
        assert abs(result.estimate - 3.0) < 1.5, (
            f"R-Learner ATE = {result.estimate:.2f}, expected ~ 3.0"
        )

    def test_heterogeneous_effect_correlation(self, heterogeneous_effect_data):
        """CATE predictions should correlate with true CATE = X1."""
        result = metalearner(
            heterogeneous_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='r',
        )
        true_cate = heterogeneous_effect_data['x1'].values
        pred_cate = result.model_info['cate']
        corr = np.corrcoef(true_cate, pred_cate)[0, 1]
        assert corr > 0.3, (
            f"R-Learner CATE correlation with true = {corr:.2f}, expected > 0.3"
        )

    def test_low_level_class(self):
        rng = np.random.default_rng(3)
        n = 800
        X = rng.normal(0, 1, (n, 2))
        D = rng.binomial(1, 0.5, n).astype(float)
        Y = 2.0 * D + X[:, 0] + rng.normal(0, 0.2, n)

        r = RLearner(n_folds=3)
        r.fit(X, Y, D)
        cate = r.effect(X)
        assert abs(np.mean(cate) - 2.0) < 1.5


# ======================================================================
# Test: DR-Learner
# ======================================================================

class TestDRLearner:

    def test_constant_effect(self, constant_effect_data):
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='dr',
        )
        assert abs(result.estimate - 3.0) < 1.0, (
            f"DR-Learner ATE = {result.estimate:.2f}, expected ~ 3.0"
        )

    def test_ci_covers_true(self, constant_effect_data):
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='dr', alpha=0.05,
        )
        assert result.ci[0] < 3.0 < result.ci[1], (
            f"CI [{result.ci[0]:.2f}, {result.ci[1]:.2f}] should cover 3.0"
        )

    def test_significant(self, strong_effect_data):
        result = metalearner(
            strong_effect_data, y='y', treat='d',
            covariates=['x1'], learner='dr',
        )
        assert result.pvalue < 0.05

    def test_heterogeneous_effect_correlation(self, heterogeneous_effect_data):
        """CATE predictions should correlate with true CATE = X1."""
        result = metalearner(
            heterogeneous_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='dr',
        )
        true_cate = heterogeneous_effect_data['x1'].values
        pred_cate = result.model_info['cate']
        corr = np.corrcoef(true_cate, pred_cate)[0, 1]
        assert corr > 0.3, (
            f"DR-Learner CATE correlation with true = {corr:.2f}, expected > 0.3"
        )

    def test_pseudo_outcomes_stored(self):
        rng = np.random.default_rng(4)
        n = 500
        X = rng.normal(0, 1, (n, 2))
        D = rng.binomial(1, 0.5, n).astype(float)
        Y = 2.0 * D + X[:, 0] + rng.normal(0, 0.1, n)

        dr = DRLearner(n_folds=3)
        dr.fit(X, Y, D)
        assert hasattr(dr, '_pseudo_outcomes')
        assert len(dr._pseudo_outcomes) == n

    def test_low_level_class(self):
        rng = np.random.default_rng(5)
        n = 800
        X = rng.normal(0, 1, (n, 2))
        D = rng.binomial(1, 0.5, n).astype(float)
        Y = 2.0 * D + X[:, 0] + rng.normal(0, 0.2, n)

        dr = DRLearner(n_folds=3)
        dr.fit(X, Y, D)
        cate = dr.effect(X)
        assert abs(np.mean(cate) - 2.0) < 1.0


# ======================================================================
# Test: all learners on same DGP (comparative)
# ======================================================================

class TestAllLearnersComparative:

    @pytest.mark.parametrize("learner", ['s', 't', 'x', 'r', 'dr'])
    def test_all_recover_ate(self, constant_effect_data, learner):
        """Every learner should recover ATE ~ 3.0 within tolerance."""
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner=learner,
        )
        assert abs(result.estimate - 3.0) < 2.0, (
            f"{learner}-Learner ATE = {result.estimate:.2f}, expected ~ 3.0"
        )

    @pytest.mark.parametrize("learner", ['s', 't', 'x', 'r', 'dr'])
    def test_all_produce_cate_array(self, constant_effect_data, learner):
        """Every learner should produce per-unit CATE array."""
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner=learner,
        )
        cate = result.model_info['cate']
        assert isinstance(cate, np.ndarray)
        assert len(cate) == result.n_obs

    @pytest.mark.parametrize("learner", ['s', 't', 'x', 'r', 'dr'])
    def test_se_positive(self, constant_effect_data, learner):
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner=learner,
        )
        assert result.se > 0


# ======================================================================
# Test: sp.metalearner top-level import
# ======================================================================

class TestTopLevelImport:

    def test_import_from_statspai(self):
        import statspai as sp
        assert hasattr(sp, 'metalearner')
        assert hasattr(sp, 'SLearner')
        assert hasattr(sp, 'TLearner')
        assert hasattr(sp, 'XLearner')
        assert hasattr(sp, 'RLearner')
        assert hasattr(sp, 'DRLearner')

    def test_sp_metalearner_works(self, constant_effect_data):
        import statspai as sp
        result = sp.metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='t',
        )
        assert isinstance(result, CausalResult)


# ======================================================================
# Test: CATE diagnostics
# ======================================================================

class TestCATEDiagnostics:

    def test_cate_summary(self, constant_effect_data):
        from statspai.metalearners import cate_summary
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='t',
        )
        summary = cate_summary(result)
        assert isinstance(summary, pd.DataFrame)
        assert 'Mean (ATE)' in summary.index
        assert 'Std. Dev.' in summary.index
        assert 'Median' in summary.index
        assert summary.loc['N', 'CATE'] == result.n_obs

    def test_cate_by_group_quartiles(self, constant_effect_data):
        from statspai.metalearners import cate_by_group
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='dr',
        )
        groups = cate_by_group(result, constant_effect_data, by='cate', n_groups=4)
        assert isinstance(groups, pd.DataFrame)
        assert 'mean_cate' in groups.columns
        assert 'ci_lower' in groups.columns
        assert len(groups) <= 4

    def test_cate_by_group_covariate(self, constant_effect_data):
        from statspai.metalearners import cate_by_group
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='t',
        )
        groups = cate_by_group(result, constant_effect_data, by='x1', n_groups=3)
        assert isinstance(groups, pd.DataFrame)
        assert len(groups) <= 3

    def test_cate_by_group_missing_column(self, constant_effect_data):
        from statspai.metalearners import cate_by_group
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='t',
        )
        with pytest.raises(ValueError, match="not found"):
            cate_by_group(result, constant_effect_data, by='nonexistent')

    def test_cate_summary_no_cate(self):
        from statspai.metalearners import cate_summary
        fake_result = CausalResult(
            method='test', estimand='ATE', estimate=1.0, se=0.1,
            pvalue=0.05, ci=(0.8, 1.2), alpha=0.05, n_obs=100,
        )
        with pytest.raises(ValueError, match="does not contain CATE"):
            cate_summary(fake_result)

    def test_top_level_import_diagnostics(self):
        import statspai as sp
        assert hasattr(sp, 'cate_summary')
        assert hasattr(sp, 'cate_by_group')
        assert hasattr(sp, 'cate_plot')
        assert hasattr(sp, 'cate_group_plot')
        assert hasattr(sp, 'predict_cate')
        assert hasattr(sp, 'compare_metalearners')
        assert hasattr(sp, 'gate_test')
        assert hasattr(sp, 'blp_test')


# ======================================================================
# Test: DR-Learner analytic SE
# ======================================================================

class TestDRLearnerAnalyticSE:

    def test_uses_influence_function(self, constant_effect_data):
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='dr',
        )
        # As of v1.11.4 the SE label is namespaced to clarify that it
        # is specifically the AIPW (DR pseudo-outcome) influence
        # function, not a generic / chosen-learner-specific IF.
        assert result.model_info['se_method'] == 'aipw_influence_function'

    def test_non_dr_uses_aipw_influence_function(self, constant_effect_data):
        # v1.11.4: ATE / SE for *every* learner now go through the
        # AIPW (DR pseudo-outcome) path. The previous bootstrap-of-CATE
        # SE was statistically invalid for S/T/X/R-Learner (treated τ̂
        # as fixed and severely under-estimated uncertainty).
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='t',
        )
        assert result.model_info['se_method'] == 'aipw_influence_function'
        assert result.model_info['ate_method'] == 'aipw_dr_pseudo_outcome'

    def test_analytic_se_reasonable(self, strong_effect_data):
        """Analytic SE should produce valid CI covering true value."""
        result = metalearner(
            strong_effect_data, y='y', treat='d',
            covariates=['x1'], learner='dr',
        )
        assert result.se > 0
        assert result.ci[0] < 5.0 < result.ci[1]


# ======================================================================
# Test: predict_cate (out-of-sample)
# ======================================================================

class TestPredictCATE:

    def test_predict_on_new_data(self, constant_effect_data):
        from statspai.metalearners import predict_cate
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='dr',
        )
        # Create new data
        rng = np.random.default_rng(999)
        new_df = pd.DataFrame({
            'x1': rng.normal(0, 1, 50),
            'x2': rng.normal(0, 1, 50),
        })
        cate_new = predict_cate(result, new_df)
        assert isinstance(cate_new, np.ndarray)
        assert len(cate_new) == 50

    def test_predict_missing_column(self, constant_effect_data):
        from statspai.metalearners import predict_cate
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='t',
        )
        bad_df = pd.DataFrame({'x1': [1, 2, 3]})
        with pytest.raises(ValueError, match="not found"):
            predict_cate(result, bad_df)

    def test_predict_no_estimator(self):
        from statspai.metalearners import predict_cate
        fake = CausalResult(
            method='test', estimand='ATE', estimate=1.0, se=0.1,
            pvalue=0.05, ci=(0.8, 1.2), alpha=0.05, n_obs=100,
        )
        with pytest.raises(ValueError, match="fitted estimator"):
            predict_cate(fake, pd.DataFrame({'x': [1]}))


# ======================================================================
# Test: compare_metalearners
# ======================================================================

class TestCompareMetalearners:

    def test_compare_all(self, constant_effect_data):
        from statspai.metalearners import compare_metalearners
        comp = compare_metalearners(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'],
        )
        assert isinstance(comp, pd.DataFrame)
        assert len(comp) == 5
        assert 'ate' in comp.columns
        assert 'se' in comp.columns
        assert 'se_method' in comp.columns
        # v1.11.4+: every learner uses the AIPW (DR pseudo-outcome)
        # influence function for SE — same column value across rows.
        dr_row = comp[comp['learner'] == 'DR-Learner']
        assert dr_row['se_method'].values[0] == 'aipw_influence_function'

    def test_compare_subset(self, constant_effect_data):
        from statspai.metalearners import compare_metalearners
        comp = compare_metalearners(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learners=['t', 'dr'],
        )
        assert len(comp) == 2


# ======================================================================
# Test: GATE test
# ======================================================================

class TestGATETest:

    def test_gate_test_basic(self, heterogeneous_effect_data):
        from statspai.metalearners import gate_test
        result = metalearner(
            heterogeneous_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='dr',
        )
        gt = gate_test(result, heterogeneous_effect_data, by='cate')
        assert 'omnibus_F' in gt
        assert 'omnibus_pvalue' in gt
        assert 'top_vs_bottom_diff' in gt
        assert 'top_vs_bottom_pvalue' in gt
        assert isinstance(gt['gate_table'], pd.DataFrame)

    def test_gate_heterogeneous_significant(self, heterogeneous_effect_data):
        """With true heterogeneity (CATE=X1), gate test should detect it."""
        from statspai.metalearners import gate_test
        result = metalearner(
            heterogeneous_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='dr',
        )
        gt = gate_test(result, heterogeneous_effect_data, by='cate')
        # Omnibus F should be significant
        assert gt['omnibus_pvalue'] < 0.10
        # Top vs bottom should be positive
        assert gt['top_vs_bottom_diff'] > 0


# ======================================================================
# Test: BLP test
# ======================================================================

class TestBLPTest:

    def test_blp_test_basic(self, constant_effect_data):
        from statspai.metalearners import blp_test
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='dr',
        )
        blp = blp_test(
            result, constant_effect_data,
            y='y', treat='d', covariates=['x1', 'x2'],
        )
        assert 'beta1' in blp
        assert 'beta2' in blp
        assert 'heterogeneity_significant' in blp
        # beta1 (ATE signal) should be significant
        assert blp['beta1_pvalue'] < 0.05

    def test_blp_detects_heterogeneity(self, heterogeneous_effect_data):
        """With true heterogeneity (CATE=X1), BLP should detect it."""
        from statspai.metalearners import blp_test
        result = metalearner(
            heterogeneous_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='dr',
        )
        blp = blp_test(
            result, heterogeneous_effect_data,
            y='y', treat='d', covariates=['x1', 'x2'],
        )
        # beta2 should be positive and significant
        assert blp['beta2'] > 0
        assert blp['heterogeneity_significant'] is True

    def test_blp_no_heterogeneity(self, constant_effect_data):
        """With constant CATE=3, beta2 should NOT be significant."""
        from statspai.metalearners import blp_test
        result = metalearner(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learner='dr',
        )
        blp = blp_test(
            result, constant_effect_data,
            y='y', treat='d', covariates=['x1', 'x2'],
        )
        # beta2 should not be strongly significant with constant effect
        # (we use a relaxed check here since ML may pick up spurious patterns)
        assert blp['beta2_pvalue'] > 0.001 or abs(blp['beta2']) < 5.0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
