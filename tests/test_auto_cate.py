"""Tests for ``sp.auto_cate`` — one-line multi-learner CATE race.

Spec: docs/superpowers/specs/2026-04-20-v094-auto-cate-strict-id-design.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import statspai as sp
from statspai.core.results import CausalResult
from statspai.metalearners import auto_cate, AutoCATEResult


# ---------------------------------------------------------------------------
# DGPs with known true CATE
# ---------------------------------------------------------------------------


@pytest.fixture
def constant_effect_data():
    """Constant CATE = 3.0."""
    rng = np.random.default_rng(42)
    n = 1200
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
    """CATE(x) = 1 + 2*x1 — strong heterogeneity."""
    rng = np.random.default_rng(7)
    n = 1200
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.5, n)
    prob = 1 / (1 + np.exp(-(0.3 * X1 + 0.2 * X2)))
    D = rng.binomial(1, prob, n).astype(float)
    tau = 1.0 + 2.0 * X1
    Y = tau * D + X1 - 0.5 * X2 + eps
    return pd.DataFrame({'y': Y, 'd': D, 'x1': X1, 'x2': X2})


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_auto_cate_basic_api(constant_effect_data):
    result = auto_cate(
        constant_effect_data, y='y', treat='d',
        covariates=['x1', 'x2'], learners=('s', 't'),
        n_folds=3,
    )
    assert isinstance(result, AutoCATEResult)
    assert hasattr(result, 'leaderboard')
    assert hasattr(result, 'best_learner')
    assert hasattr(result, 'best_result')
    assert hasattr(result, 'results')
    assert hasattr(result, 'agreement')
    assert hasattr(result, 'selection_rule')
    assert hasattr(result, 'n_obs')


def test_auto_cate_leaderboard_shape(constant_effect_data):
    result = auto_cate(
        constant_effect_data, y='y', treat='d',
        covariates=['x1', 'x2'], learners=('s', 't', 'x'),
        n_folds=3,
    )
    assert len(result.leaderboard) == 3
    for col in ['learner', 'ate', 'se', 'ci_lower', 'ci_upper',
                'r_loss', 'blp_beta1', 'blp_beta2']:
        assert col in result.leaderboard.columns


def test_auto_cate_recovers_ate_on_constant_dgp(constant_effect_data):
    result = auto_cate(
        constant_effect_data, y='y', treat='d',
        covariates=['x1', 'x2'], learners=('t', 'x', 'dr'),
        n_folds=3,
    )
    # Best learner's ATE within 4 SE of 3.0
    ate = result.best_result.estimate
    se = result.best_result.se
    assert abs(ate - 3.0) < 4 * max(se, 0.05), (
        f"ATE {ate:.3f} far from true 3.0 (se={se:.3f})"
    )


def test_auto_cate_positive_on_positive_dgp(constant_effect_data):
    result = auto_cate(
        constant_effect_data, y='y', treat='d',
        covariates=['x1', 'x2'], learners=('s', 't', 'x', 'r', 'dr'),
        n_folds=3,
    )
    assert (result.leaderboard['ate'] > 0).all(), (
        f"Not all ATEs positive: {result.leaderboard[['learner', 'ate']]}"
    )


def test_auto_cate_learner_subset(constant_effect_data):
    result = auto_cate(
        constant_effect_data, y='y', treat='d',
        covariates=['x1', 'x2'], learners=('t', 'dr'),
        n_folds=3,
    )
    assert set(result.results.keys()) == {'t', 'dr'}


def test_auto_cate_invalid_learner_raises(constant_effect_data):
    with pytest.raises(ValueError, match='bogus'):
        auto_cate(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'], learners=('bogus',),
            n_folds=3,
        )


def test_auto_cate_selection_rule_nonempty(constant_effect_data):
    result = auto_cate(
        constant_effect_data, y='y', treat='d',
        covariates=['x1', 'x2'], learners=('t', 'dr'),
        n_folds=3,
    )
    assert isinstance(result.selection_rule, str)
    assert len(result.selection_rule) > 10


def test_auto_cate_agreement_matrix(constant_effect_data):
    result = auto_cate(
        constant_effect_data, y='y', treat='d',
        covariates=['x1', 'x2'], learners=('t', 'x', 'dr'),
        n_folds=3,
    )
    agr = result.agreement
    assert agr.shape == (3, 3)
    # Diagonal is 1.0
    diag = np.diag(agr.values)
    np.testing.assert_allclose(diag, 1.0, atol=1e-10)


def test_auto_cate_best_result_is_causal_result(constant_effect_data):
    result = auto_cate(
        constant_effect_data, y='y', treat='d',
        covariates=['x1', 'x2'], learners=('t', 'dr'),
        n_folds=3,
    )
    assert isinstance(result.best_result, CausalResult)
    tidy = result.best_result.tidy()
    assert isinstance(tidy, pd.DataFrame)
    assert len(tidy) >= 1
    glance = result.best_result.glance()
    assert isinstance(glance, pd.DataFrame)
    assert len(glance) == 1


def test_auto_cate_custom_models(constant_effect_data):
    rf_outcome = RandomForestRegressor(n_estimators=50, random_state=0)
    result = auto_cate(
        constant_effect_data, y='y', treat='d',
        covariates=['x1', 'x2'], learners=('t',),
        outcome_model=rf_outcome,
        n_folds=3,
    )
    assert 't' in result.results


def test_auto_cate_summary_mentions_winner(constant_effect_data):
    result = auto_cate(
        constant_effect_data, y='y', treat='d',
        covariates=['x1', 'x2'], learners=('t', 'dr'),
        n_folds=3,
    )
    summary = result.summary()
    assert isinstance(summary, str)
    assert len(summary) > 50
    assert result.best_learner in summary


def test_auto_cate_at_sp_top_level():
    assert sp.auto_cate is auto_cate
    assert sp.AutoCATEResult is AutoCATEResult


def test_auto_cate_heterogeneous_picks_nontrivial_cate(heterogeneous_effect_data):
    """With real heterogeneity, the best learner should have non-trivial CATE std."""
    result = auto_cate(
        heterogeneous_effect_data, y='y', treat='d',
        covariates=['x1', 'x2'], learners=('t', 'x', 'dr'),
        n_folds=3,
    )
    cate = result.best_result.model_info['cate']
    # True CATE std is 2.0 (since CATE = 1 + 2*x1 and std(x1)=1)
    # Learner should pick up at least some of that
    assert np.std(cate) > 0.5
