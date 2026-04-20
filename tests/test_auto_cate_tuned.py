"""Tests for ``sp.auto_cate_tuned`` — Optuna-tuned CATE learner race."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.metalearners import AutoCATEResult, auto_cate_tuned

optuna = pytest.importorskip(
    "optuna",
    reason="Optuna is an optional dependency; skip tuner tests if missing.",
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def constant_effect_data():
    """Constant CATE = 3.0."""
    rng = np.random.default_rng(42)
    n = 800
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    D = rng.binomial(1, 1 / (1 + np.exp(-(0.3 * X1 + 0.2 * X2))), n).astype(float)
    Y = 3.0 * D + X1 ** 2 + rng.normal(0, 0.5, n)
    return pd.DataFrame({'y': Y, 'd': D, 'x1': X1, 'x2': X2})


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def test_auto_cate_tuned_top_level_export():
    assert sp.auto_cate_tuned is auto_cate_tuned


def test_auto_cate_tuned_returns_auto_cate_result(constant_effect_data):
    r = auto_cate_tuned(
        constant_effect_data, y='y', treat='d', covariates=['x1', 'x2'],
        learners=('t', 'dr'), n_trials=5, n_folds=3, random_state=1,
    )
    assert isinstance(r, AutoCATEResult)
    assert 'tuned_params' in r.best_result.model_info
    assert 'n_trials' in r.best_result.model_info
    assert r.best_result.model_info['n_trials'] <= 5


def test_auto_cate_tuned_records_best_r_loss(constant_effect_data):
    r = auto_cate_tuned(
        constant_effect_data, y='y', treat='d', covariates=['x1', 'x2'],
        learners=('t',), n_trials=3, n_folds=3, random_state=1,
    )
    info = r.best_result.model_info
    assert 'best_r_loss_nuisance' in info
    assert np.isfinite(info['best_r_loss_nuisance'])


def test_auto_cate_tuned_selection_rule_mentions_trials(constant_effect_data):
    r = auto_cate_tuned(
        constant_effect_data, y='y', treat='d', covariates=['x1', 'x2'],
        learners=('t',), n_trials=3, n_folds=3, random_state=1,
    )
    assert 'Optuna' in r.selection_rule


def test_auto_cate_tuned_recovers_ate(constant_effect_data):
    r = auto_cate_tuned(
        constant_effect_data, y='y', treat='d', covariates=['x1', 'x2'],
        learners=('t', 'dr'), n_trials=5, n_folds=3, random_state=1,
    )
    # Best learner's ATE within 4 SE of 3.0 (constant effect)
    ate = r.best_result.estimate
    se = max(r.best_result.se, 0.05)
    assert abs(ate - 3.0) < 4 * se, (
        f"ATE {ate:.3f} far from true 3.0 (se={r.best_result.se:.3f})"
    )


# ---------------------------------------------------------------------------
# Custom search space
# ---------------------------------------------------------------------------

def test_auto_cate_tuned_custom_search_space(constant_effect_data):
    tiny_space = {
        'outcome_n_estimators': [50],
        'outcome_max_depth': [3],
        'outcome_learning_rate': [0.05],
        'outcome_subsample': [1.0],
        'propensity_n_estimators': [50],
        'propensity_max_depth': [3],
        'propensity_learning_rate': [0.05],
    }
    r = auto_cate_tuned(
        constant_effect_data, y='y', treat='d', covariates=['x1', 'x2'],
        learners=('t',), n_trials=2, n_folds=3,
        search_space=tiny_space, random_state=1,
    )
    tp = r.best_result.model_info['tuned_params']
    # With a singleton space the only possible trial is our choice
    assert tp['outcome_n_estimators'] == 50
    assert tp['propensity_learning_rate'] == 0.05


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_auto_cate_tuned_invalid_treatment_raises():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'y': rng.normal(size=100),
        'd': rng.choice([0.0, 1.0, 2.0], size=100),
        'x1': rng.normal(size=100),
    })
    with pytest.raises(ValueError, match='binary'):
        auto_cate_tuned(df, y='y', treat='d', covariates=['x1'],
                        learners=('t',), n_trials=2, n_folds=3)


# ---------------------------------------------------------------------------
# Per-learner tuner (v0.9.6)
# ---------------------------------------------------------------------------

def test_auto_cate_tuned_invalid_tune_mode_raises(constant_effect_data):
    with pytest.raises(ValueError, match='tune'):
        auto_cate_tuned(
            constant_effect_data, y='y', treat='d',
            covariates=['x1', 'x2'],
            learners=('t',), tune='bogus', n_trials=2, n_folds=3,
        )


def test_auto_cate_tuned_per_learner_mode(constant_effect_data):
    r = auto_cate_tuned(
        constant_effect_data, y='y', treat='d', covariates=['x1', 'x2'],
        learners=('t', 'dr'), tune='per_learner',
        n_trials_per_learner=3, n_folds=3, random_state=1,
    )
    info = r.best_result.model_info
    assert 'per_learner_params' in info
    assert 'per_learner_r_loss' in info
    assert set(info['per_learner_params'].keys()) == {'t', 'dr'}
    assert 'best_per_learner_code' in info


def test_auto_cate_tuned_per_learner_no_nuisance_params(constant_effect_data):
    """When tune='per_learner', nuisance metadata should NOT be set."""
    r = auto_cate_tuned(
        constant_effect_data, y='y', treat='d', covariates=['x1', 'x2'],
        learners=('t',), tune='per_learner',
        n_trials_per_learner=2, n_folds=3, random_state=1,
    )
    info = r.best_result.model_info
    assert 'tuned_params' not in info
    assert info['tune_mode'] == 'per_learner'


def test_auto_cate_tuned_both_mode(constant_effect_data):
    r = auto_cate_tuned(
        constant_effect_data, y='y', treat='d', covariates=['x1', 'x2'],
        learners=('t', 'dr'), tune='both',
        n_trials=3, n_trials_per_learner=2, n_folds=3, random_state=1,
    )
    info = r.best_result.model_info
    assert info['tune_mode'] == 'both'
    assert 'tuned_params' in info
    assert 'per_learner_params' in info
    assert set(info['per_learner_params'].keys()) == {'t', 'dr'}


def test_auto_cate_tuned_per_learner_selection_rule(constant_effect_data):
    r = auto_cate_tuned(
        constant_effect_data, y='y', treat='d', covariates=['x1', 'x2'],
        learners=('t',), tune='per_learner',
        n_trials_per_learner=2, n_folds=3, random_state=1,
    )
    assert 'per-learner' in r.selection_rule
