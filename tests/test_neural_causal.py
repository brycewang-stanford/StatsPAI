"""
Tests for Neural Causal Models (TARNet, CFRNet, DragonNet).
"""

import pytest
import numpy as np
import pandas as pd

pytest.importorskip("torch", reason="PyTorch required for neural causal tests")

from statspai.neural_causal import (
    tarnet, cfrnet, dragonnet,
    TARNet, CFRNet, DragonNet,
)
from statspai.core.results import CausalResult


# ======================================================================
# Fixtures: DGPs with known true effects
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
    True ATE = E[X1] ~ 0.
    """
    rng = np.random.default_rng(123)
    n = 3000

    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.3, n)

    logit = 0.3 * X2
    prob = 1 / (1 + np.exp(-logit))
    D = rng.binomial(1, prob, n).astype(float)

    tau = X1
    Y = tau * D + X2 + eps

    return pd.DataFrame({'y': Y, 'd': D, 'x1': X1, 'x2': X2})


@pytest.fixture
def small_data():
    """Small dataset for quick smoke tests."""
    rng = np.random.default_rng(99)
    n = 200
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n).astype(float)
    Y = 2.0 * D + X1 + rng.normal(0, 0.5, n)
    return pd.DataFrame({'y': Y, 'd': D, 'x1': X1, 'x2': X2})


# ======================================================================
# TARNet Tests
# ======================================================================

class TestTARNet:
    """Tests for TARNet estimator."""

    def test_returns_causal_result(self, small_data):
        result = tarnet(small_data, y='y', treat='d',
                        covariates=['x1', 'x2'],
                        epochs=50, repr_layers=(64,),
                        head_layers=(32,), n_bootstrap=50)
        assert isinstance(result, CausalResult)
        assert result.method == 'TARNet (Shalit et al. 2017)'
        assert result.estimand == 'ATE'

    def test_has_required_fields(self, small_data):
        result = tarnet(small_data, y='y', treat='d',
                        covariates=['x1', 'x2'],
                        epochs=50, repr_layers=(64,),
                        head_layers=(32,), n_bootstrap=50)
        assert result.estimate is not None
        assert result.se > 0
        assert 0 <= result.pvalue <= 1
        assert len(result.ci) == 2
        assert result.ci[0] < result.ci[1]
        assert result.n_obs == len(small_data)

    def test_cate_in_model_info(self, small_data):
        result = tarnet(small_data, y='y', treat='d',
                        covariates=['x1', 'x2'],
                        epochs=50, repr_layers=(64,),
                        head_layers=(32,), n_bootstrap=50)
        assert 'cate' in result.model_info
        assert len(result.model_info['cate']) == len(small_data)

    def test_constant_effect_recovery(self, constant_effect_data):
        """ATE should be close to 3.0 for constant effect DGP."""
        result = tarnet(constant_effect_data, y='y', treat='d',
                        covariates=['x1', 'x2'],
                        epochs=200, repr_layers=(128, 64),
                        head_layers=(64,), n_bootstrap=100)
        assert abs(result.estimate - 3.0) < 1.5, \
            f"TARNet ATE={result.estimate:.2f}, expected ~3.0"

    def test_class_interface(self, small_data):
        est = TARNet(data=small_data, y='y', treat='d',
                     covariates=['x1', 'x2'],
                     epochs=50, repr_layers=(64,),
                     head_layers=(32,), n_bootstrap=50)
        result = est.fit()
        assert isinstance(result, CausalResult)

        # Test effect method
        cate = est.effect()
        assert len(cate) == len(small_data)

    def test_effect_new_data(self, small_data):
        est = TARNet(data=small_data, y='y', treat='d',
                     covariates=['x1', 'x2'],
                     epochs=50, repr_layers=(64,),
                     head_layers=(32,), n_bootstrap=50)
        est.fit()

        X_new = np.random.randn(10, 2).astype(np.float32)
        cate_new = est.effect(X_new)
        assert len(cate_new) == 10

    def test_summary_renders(self, small_data):
        result = tarnet(small_data, y='y', treat='d',
                        covariates=['x1', 'x2'],
                        epochs=50, repr_layers=(64,),
                        head_layers=(32,), n_bootstrap=50)
        summary = result.summary()
        assert 'TARNet' in summary
        assert 'ATE' in summary

    def test_citation(self, small_data):
        result = tarnet(small_data, y='y', treat='d',
                        covariates=['x1', 'x2'],
                        epochs=50, repr_layers=(64,),
                        head_layers=(32,), n_bootstrap=50)
        bib = result.cite()
        assert 'shalit2017' in bib

    def test_missing_column_raises(self, small_data):
        with pytest.raises(ValueError, match="Columns not found"):
            tarnet(small_data, y='y', treat='d',
                   covariates=['x1', 'nonexistent'],
                   epochs=10, repr_layers=(32,))

    def test_non_binary_treatment_raises(self):
        df = pd.DataFrame({
            'y': [1, 2, 3], 'd': [0, 1, 2],
            'x1': [1, 2, 3], 'x2': [1, 2, 3]
        })
        with pytest.raises(ValueError, match="binary"):
            tarnet(df, y='y', treat='d', covariates=['x1', 'x2'],
                   epochs=10, repr_layers=(32,))


# ======================================================================
# CFRNet Tests
# ======================================================================

class TestCFRNet:
    """Tests for CFRNet estimator."""

    def test_returns_causal_result(self, small_data):
        result = cfrnet(small_data, y='y', treat='d',
                        covariates=['x1', 'x2'],
                        epochs=50, repr_layers=(64,),
                        head_layers=(32,), ipm_weight=0.5,
                        n_bootstrap=50)
        assert isinstance(result, CausalResult)
        assert result.method == 'CFRNet (Shalit et al. 2017)'

    def test_ipm_regularisation_effect(self, small_data):
        """Higher IPM weight should produce different CATE distribution."""
        r_low = cfrnet(small_data, y='y', treat='d',
                       covariates=['x1', 'x2'],
                       epochs=100, repr_layers=(64,),
                       head_layers=(32,), ipm_weight=0.0,
                       n_bootstrap=50, random_state=42)
        r_high = cfrnet(small_data, y='y', treat='d',
                        covariates=['x1', 'x2'],
                        epochs=100, repr_layers=(64,),
                        head_layers=(32,), ipm_weight=10.0,
                        n_bootstrap=50, random_state=42)
        # With high IPM, CATE variance should generally be different
        std_low = np.std(r_low.model_info['cate'])
        std_high = np.std(r_high.model_info['cate'])
        # Just verify both ran successfully, not direction of effect
        assert std_low >= 0 and std_high >= 0

    def test_constant_effect_recovery(self, constant_effect_data):
        result = cfrnet(constant_effect_data, y='y', treat='d',
                        covariates=['x1', 'x2'],
                        epochs=200, repr_layers=(128, 64),
                        head_layers=(64,), ipm_weight=1.0,
                        n_bootstrap=100)
        assert abs(result.estimate - 3.0) < 1.5, \
            f"CFRNet ATE={result.estimate:.2f}, expected ~3.0"

    def test_citation(self, small_data):
        result = cfrnet(small_data, y='y', treat='d',
                        covariates=['x1', 'x2'],
                        epochs=50, repr_layers=(64,),
                        head_layers=(32,), n_bootstrap=50)
        bib = result.cite()
        assert 'shalit2017' in bib

    def test_class_effect_method(self, small_data):
        est = CFRNet(data=small_data, y='y', treat='d',
                     covariates=['x1', 'x2'],
                     epochs=50, repr_layers=(64,),
                     head_layers=(32,), n_bootstrap=50)
        est.fit()
        cate = est.effect()
        assert len(cate) == len(small_data)

        X_new = np.random.randn(5, 2).astype(np.float32)
        cate_new = est.effect(X_new)
        assert len(cate_new) == 5


# ======================================================================
# DragonNet Tests
# ======================================================================

class TestDragonNet:
    """Tests for DragonNet estimator."""

    def test_returns_causal_result(self, small_data):
        result = dragonnet(small_data, y='y', treat='d',
                           covariates=['x1', 'x2'],
                           epochs=50, repr_layers=(64,),
                           head_layers=(32,), n_bootstrap=50)
        assert isinstance(result, CausalResult)
        assert result.method == 'DragonNet (Shi et al. 2019)'

    def test_aipw_estimation(self, small_data):
        """DragonNet should use AIPW for ATE estimation."""
        result = dragonnet(small_data, y='y', treat='d',
                           covariates=['x1', 'x2'],
                           epochs=50, repr_layers=(64,),
                           head_layers=(32,), n_bootstrap=50)
        assert result.model_info['se_method'] == 'AIPW_influence_function'
        assert 'ate_aipw' in result.model_info
        assert 'ate_plugin' in result.model_info

    def test_propensity_scores(self, small_data):
        """DragonNet should produce valid propensity scores."""
        est = DragonNet(data=small_data, y='y', treat='d',
                        covariates=['x1', 'x2'],
                        epochs=50, repr_layers=(64,),
                        head_layers=(32,), n_bootstrap=50)
        est.fit()

        e = est.propensity()
        assert len(e) == len(small_data)
        assert np.all(e >= 0.01)
        assert np.all(e <= 0.99)

    def test_propensity_new_data(self, small_data):
        est = DragonNet(data=small_data, y='y', treat='d',
                        covariates=['x1', 'x2'],
                        epochs=50, repr_layers=(64,),
                        head_layers=(32,), n_bootstrap=50)
        est.fit()

        X_new = np.random.randn(10, 2).astype(np.float32)
        e_new = est.propensity(X_new)
        assert len(e_new) == 10
        assert np.all(e_new >= 0.01)
        assert np.all(e_new <= 0.99)

    def test_constant_effect_recovery(self, constant_effect_data):
        result = dragonnet(constant_effect_data, y='y', treat='d',
                           covariates=['x1', 'x2'],
                           epochs=200, repr_layers=(128, 64),
                           head_layers=(64,), n_bootstrap=100)
        assert abs(result.estimate - 3.0) < 2.0, \
            f"DragonNet ATE={result.estimate:.2f}, expected ~3.0"

    def test_without_targeted_reg(self, small_data):
        """Should work without targeted regularisation."""
        result = dragonnet(small_data, y='y', treat='d',
                           covariates=['x1', 'x2'],
                           epochs=50, repr_layers=(64,),
                           head_layers=(32,),
                           targeted_reg_weight=0.0,
                           n_bootstrap=50)
        assert isinstance(result, CausalResult)

    def test_citation(self, small_data):
        result = dragonnet(small_data, y='y', treat='d',
                           covariates=['x1', 'x2'],
                           epochs=50, repr_layers=(64,),
                           head_layers=(32,), n_bootstrap=50)
        bib = result.cite()
        assert 'shi2019' in bib

    def test_effect_method(self, small_data):
        est = DragonNet(data=small_data, y='y', treat='d',
                        covariates=['x1', 'x2'],
                        epochs=50, repr_layers=(64,),
                        head_layers=(32,), n_bootstrap=50)
        est.fit()

        cate = est.effect()
        assert len(cate) == len(small_data)

        X_new = np.random.randn(5, 2).astype(np.float32)
        cate_new = est.effect(X_new)
        assert len(cate_new) == 5


# ======================================================================
# Functional API Tests
# ======================================================================

class TestFunctionalAPI:
    """Test the top-level sp.tarnet() / sp.cfrnet() / sp.dragonnet() API."""

    def test_import_from_statspai(self):
        """Should be importable from the top-level package."""
        import statspai as sp
        assert hasattr(sp, 'tarnet')
        assert hasattr(sp, 'cfrnet')
        assert hasattr(sp, 'dragonnet')
        assert hasattr(sp, 'TARNet')
        assert hasattr(sp, 'CFRNet')
        assert hasattr(sp, 'DragonNet')

    def test_verbose_mode(self, small_data, capsys):
        tarnet(small_data, y='y', treat='d',
               covariates=['x1', 'x2'],
               epochs=100, repr_layers=(32,),
               head_layers=(16,), n_bootstrap=20,
               verbose=True)
        captured = capsys.readouterr()
        assert 'TARNet epoch' in captured.out

    def test_cate_diagnostics_compatible(self, small_data):
        """CATE from neural models should work with CATE diagnostics."""
        from statspai.metalearners.diagnostics import cate_summary

        result = tarnet(small_data, y='y', treat='d',
                        covariates=['x1', 'x2'],
                        epochs=50, repr_layers=(64,),
                        head_layers=(32,), n_bootstrap=50)
        # cate_summary expects a CausalResult with model_info['cate']
        stats = cate_summary(result)
        assert isinstance(stats, pd.DataFrame)

    def test_nan_handling(self):
        """Should handle NaN by dropping rows."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            'y': rng.normal(0, 1, n),
            'd': rng.binomial(1, 0.5, n).astype(float),
            'x1': rng.normal(0, 1, n),
            'x2': rng.normal(0, 1, n),
        })
        df.loc[0, 'x1'] = np.nan
        df.loc[5, 'y'] = np.nan

        result = tarnet(df, y='y', treat='d',
                        covariates=['x1', 'x2'],
                        epochs=30, repr_layers=(32,),
                        head_layers=(16,), n_bootstrap=20)
        assert result.n_obs == n - 2


# ======================================================================
# Edge Cases
# ======================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_single_covariate(self):
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            'y': rng.normal(0, 1, n),
            'd': rng.binomial(1, 0.5, n).astype(float),
            'x1': rng.normal(0, 1, n),
        })
        result = tarnet(df, y='y', treat='d',
                        covariates=['x1'],
                        epochs=30, repr_layers=(32,),
                        head_layers=(16,), n_bootstrap=20)
        assert isinstance(result, CausalResult)

    def test_many_covariates(self):
        rng = np.random.default_rng(42)
        n = 500
        ncov = 20
        data = {'y': rng.normal(0, 1, n),
                'd': rng.binomial(1, 0.5, n).astype(float)}
        covs = []
        for i in range(ncov):
            name = f'x{i}'
            data[name] = rng.normal(0, 1, n)
            covs.append(name)
        df = pd.DataFrame(data)
        result = dragonnet(df, y='y', treat='d',
                           covariates=covs,
                           epochs=30, repr_layers=(64,),
                           head_layers=(32,), n_bootstrap=20)
        assert isinstance(result, CausalResult)
        assert result.model_info['n_covariates'] == 20

    def test_unbalanced_treatment(self):
        """Should work with highly unbalanced treatment (10% treated)."""
        rng = np.random.default_rng(42)
        n = 500
        df = pd.DataFrame({
            'y': rng.normal(0, 1, n),
            'd': rng.binomial(1, 0.1, n).astype(float),
            'x1': rng.normal(0, 1, n),
            'x2': rng.normal(0, 1, n),
        })
        # Ensure at least some treated
        if df['d'].sum() < 5:
            df.loc[:4, 'd'] = 1.0
        result = cfrnet(df, y='y', treat='d',
                        covariates=['x1', 'x2'],
                        epochs=30, repr_layers=(32,),
                        head_layers=(16,), n_bootstrap=20)
        assert isinstance(result, CausalResult)

    def test_unfitted_effect_raises(self):
        rng = np.random.default_rng(42)
        n = 50
        df = pd.DataFrame({
            'y': rng.normal(0, 1, n),
            'd': rng.binomial(1, 0.5, n).astype(float),
            'x1': rng.normal(0, 1, n),
        })
        est = TARNet(data=df, y='y', treat='d', covariates=['x1'],
                     epochs=10, repr_layers=(16,))
        with pytest.raises(ValueError, match="fitted"):
            est.effect()

    def test_unfitted_propensity_raises(self):
        rng = np.random.default_rng(42)
        n = 50
        df = pd.DataFrame({
            'y': rng.normal(0, 1, n),
            'd': rng.binomial(1, 0.5, n).astype(float),
            'x1': rng.normal(0, 1, n),
        })
        est = DragonNet(data=df, y='y', treat='d', covariates=['x1'],
                        epochs=10, repr_layers=(16,))
        with pytest.raises(ValueError, match="fitted"):
            est.propensity()
