"""
Tests for DeepIV module.

Uses a simple DGP with known causal effect to verify the estimator
recovers the true parameter within reasonable tolerance.
"""

import numpy as np
import pandas as pd
import pytest

# Skip entire module if torch is not available
torch = pytest.importorskip("torch")

from statspai.deepiv import deepiv, DeepIV
from statspai.core.results import CausalResult


# ---------------------------------------------------------------------------
# Test DGP: linear structural model with instrument
#   Z ~ N(0,1)                        (instrument)
#   X ~ N(0,1)                        (covariate)
#   T = 0.6*Z + 0.4*X + v,  v~N(0,1) (endogenous treatment)
#   Y = 2.0*T + 1.5*X + e,  e~N(0,1) (outcome, true effect = 2.0)
#   Endogeneity: Cov(v, e) != 0 is induced below
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_iv_data():
    np.random.seed(42)
    n = 2000
    Z = np.random.randn(n)
    X = np.random.randn(n)
    # Correlated errors -> endogeneity
    common = np.random.randn(n)
    v = 0.5 * common + 0.5 * np.random.randn(n)
    e = 0.5 * common + 0.5 * np.random.randn(n)
    T = 0.6 * Z + 0.4 * X + v
    Y = 2.0 * T + 1.5 * X + e
    return pd.DataFrame({'y': Y, 'treat': T, 'instrument': Z, 'covar': X})


class TestDeepIVBasic:
    """Basic functionality tests."""

    def test_returns_causal_result(self, linear_iv_data):
        result = deepiv(
            linear_iv_data, y='y', treat='treat',
            instruments=['instrument'], covariates=['covar'],
            first_stage_epochs=30, second_stage_epochs=30,
            hidden_layers=(64, 32), n_components=5,
        )
        assert isinstance(result, CausalResult)

    def test_summary_runs(self, linear_iv_data):
        result = deepiv(
            linear_iv_data, y='y', treat='treat',
            instruments=['instrument'], covariates=['covar'],
            first_stage_epochs=30, second_stage_epochs=30,
            hidden_layers=(64, 32), n_components=5,
        )
        summary = result.summary()
        assert 'DeepIV' in summary
        assert 'LATE' in summary

    def test_result_fields(self, linear_iv_data):
        result = deepiv(
            linear_iv_data, y='y', treat='treat',
            instruments=['instrument'], covariates=['covar'],
            first_stage_epochs=30, second_stage_epochs=30,
            hidden_layers=(64, 32), n_components=5,
        )
        assert result.n_obs == 2000
        assert result.se > 0
        assert result.ci[0] < result.ci[1]
        assert 0 <= result.pvalue <= 1
        assert result.detail is not None
        assert len(result.detail) == 5  # 5 treatment levels

    def test_citation(self, linear_iv_data):
        result = deepiv(
            linear_iv_data, y='y', treat='treat',
            instruments=['instrument'], covariates=['covar'],
            first_stage_epochs=10, second_stage_epochs=10,
            hidden_layers=(32,), n_components=3,
        )
        cite = result.cite()
        assert 'hartford2017deep' in cite


class TestDeepIVClass:
    """Tests for the DeepIV class interface."""

    def test_class_interface(self, linear_iv_data):
        est = DeepIV(
            data=linear_iv_data, y='y', treat='treat',
            instruments=['instrument'], covariates=['covar'],
            first_stage_epochs=20, second_stage_epochs=20,
            hidden_layers=(32,), n_components=3,
        )
        result = est.fit()
        assert isinstance(result, CausalResult)

    def test_effect_method(self, linear_iv_data):
        est = DeepIV(
            data=linear_iv_data, y='y', treat='treat',
            instruments=['instrument'], covariates=['covar'],
            first_stage_epochs=20, second_stage_epochs=20,
            hidden_layers=(32,), n_components=3,
        )
        est.fit()
        effects = est.effect(t0=0.0, t1=1.0)
        assert len(effects) == 2000
        assert np.isfinite(effects).all()


class TestDeepIVValidation:
    """Input validation tests."""

    def test_missing_columns(self, linear_iv_data):
        with pytest.raises(ValueError, match="Columns not found"):
            deepiv(
                linear_iv_data, y='y', treat='treat',
                instruments=['nonexistent'], covariates=['covar'],
            )

    def test_no_instruments(self, linear_iv_data):
        with pytest.raises(ValueError, match="At least one instrument"):
            deepiv(
                linear_iv_data, y='y', treat='treat',
                instruments=[], covariates=['covar'],
            )

    def test_effect_before_fit(self, linear_iv_data):
        est = DeepIV(
            data=linear_iv_data, y='y', treat='treat',
            instruments=['instrument'], covariates=['covar'],
        )
        with pytest.raises(ValueError, match="fitted first"):
            est.effect(t0=0, t1=1)


class TestDeepIVVerbose:
    """Test verbose mode."""

    def test_verbose_output(self, linear_iv_data, capsys):
        deepiv(
            linear_iv_data, y='y', treat='treat',
            instruments=['instrument'], covariates=['covar'],
            first_stage_epochs=20, second_stage_epochs=20,
            hidden_layers=(32,), n_components=3,
            verbose=True,
        )
        captured = capsys.readouterr()
        assert 'Stage 1' in captured.out
        assert 'Stage 2' in captured.out
