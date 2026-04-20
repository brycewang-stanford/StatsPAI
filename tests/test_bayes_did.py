"""Tests for ``sp.bayes_did`` — Bayesian difference-in-differences.

Spec: docs/superpowers/specs/2026-04-20-v095-bayes-optuna-rust-spike.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.bayes import BayesianCausalResult, bayes_did

pymc = pytest.importorskip(
    "pymc",
    reason="PyMC is an optional dependency; skip Bayesian tests if missing.",
)


# ---------------------------------------------------------------------------
# Fixtures: canonical DGPs with a known ATT
# ---------------------------------------------------------------------------

@pytest.fixture
def did_2x2_data():
    """30 units x 2 periods; true ATT = 1.5."""
    rng = np.random.default_rng(101)
    rows = []
    for u in range(60):
        treated = u < 30
        for t in range(2):
            post = (t == 1)
            noise = rng.normal(0, 0.5)
            y = (1.0
                 + 0.5 * treated
                 + 0.3 * post
                 + 1.5 * (treated and post)
                 + noise)
            rows.append({'y': y, 'treat': int(treated),
                         'post': int(post), 'unit': u, 'time': t})
    return pd.DataFrame(rows)


@pytest.fixture
def did_panel_data():
    """50 units x 4 periods; true ATT = -0.8 (negative effect)."""
    rng = np.random.default_rng(102)
    rows = []
    for u in range(50):
        treated = u < 25
        alpha = rng.normal(0, 0.8)  # unit random effect
        for t in range(4):
            post = int(t >= 2)
            noise = rng.normal(0, 0.4)
            did = treated * post
            y = 1.0 + alpha + 0.4 * post + (-0.8) * did + noise
            rows.append({'y': y, 'treat': int(treated), 'post': int(post),
                         'unit': u, 'time': t})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# API + integration
# ---------------------------------------------------------------------------

def test_bayes_did_returns_bayesian_causal_result(did_2x2_data):
    r = bayes_did(did_2x2_data, y='y', treat='treat', post='post',
                  draws=300, tune=300, chains=2, progressbar=False)
    assert isinstance(r, BayesianCausalResult)
    assert r.estimand == 'ATT'
    assert r.method.startswith('Bayesian DID')
    assert r.n_obs == len(did_2x2_data)


def test_bayes_did_top_level_export():
    assert sp.bayes_did is bayes_did
    assert sp.BayesianCausalResult is BayesianCausalResult


def test_bayes_did_tidy_glance(did_2x2_data):
    r = bayes_did(did_2x2_data, y='y', treat='treat', post='post',
                  draws=300, tune=300, chains=2, progressbar=False)
    tidy = r.tidy()
    assert isinstance(tidy, pd.DataFrame)
    assert len(tidy) == 1
    for col in ['term', 'estimate', 'std_error',
                'conf_low', 'conf_high', 'prob_positive']:
        assert col in tidy.columns

    glance = r.glance()
    assert isinstance(glance, pd.DataFrame)
    assert len(glance) == 1
    for col in ['method', 'nobs', 'rhat', 'ess', 'chains', 'draws']:
        assert col in glance.columns


def test_bayes_did_summary_is_string(did_2x2_data):
    r = bayes_did(did_2x2_data, y='y', treat='treat', post='post',
                  draws=300, tune=300, chains=2, progressbar=False)
    s = r.summary()
    assert isinstance(s, str)
    assert 'Bayesian DID' in s
    assert 'R-hat' in s


# ---------------------------------------------------------------------------
# Recovery (statistical correctness)
# ---------------------------------------------------------------------------

def test_bayes_did_2x2_hdi_covers_true_att(did_2x2_data):
    r = bayes_did(did_2x2_data, y='y', treat='treat', post='post',
                  draws=600, tune=600, chains=2, progressbar=False,
                  random_state=7)
    assert r.hdi_lower < 1.5 < r.hdi_upper, (
        f"True ATT 1.5 not covered by 95% HDI "
        f"[{r.hdi_lower:.3f}, {r.hdi_upper:.3f}]; "
        f"posterior mean {r.posterior_mean:.3f}."
    )


def test_bayes_did_panel_hdi_covers_true_att(did_panel_data):
    r = bayes_did(did_panel_data, y='y', treat='treat', post='post',
                  unit='unit', draws=600, tune=600, chains=2,
                  progressbar=False, random_state=8)
    true_att = -0.8
    assert r.hdi_lower < true_att < r.hdi_upper, (
        f"True ATT {true_att} not covered by 95% HDI "
        f"[{r.hdi_lower:.3f}, {r.hdi_upper:.3f}]; "
        f"posterior mean {r.posterior_mean:.3f}."
    )
    assert r.model_info['use_unit_re'] is True


def test_bayes_did_prob_positive_strong_on_strong_effect(did_2x2_data):
    r = bayes_did(did_2x2_data, y='y', treat='treat', post='post',
                  draws=600, tune=600, chains=2, progressbar=False,
                  random_state=9)
    # True ATT = 1.5 > 0 with tight noise => posterior > 0 should
    # dominate.
    assert r.prob_positive > 0.95


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_bayes_did_missing_column_raises(did_2x2_data):
    with pytest.raises(ValueError, match="not found"):
        bayes_did(did_2x2_data, y='nope', treat='treat', post='post',
                  draws=50, tune=50, chains=1, progressbar=False)


def test_bayes_did_non_binary_treat_raises(did_2x2_data):
    df = did_2x2_data.copy()
    df['treat'] = df['treat'].astype(float) * 2.0  # 0/2 coding
    with pytest.raises(ValueError, match="binary"):
        bayes_did(df, y='y', treat='treat', post='post',
                  draws=50, tune=50, chains=1, progressbar=False)


def test_bayes_did_rope_populated_when_supplied(did_2x2_data):
    r = bayes_did(did_2x2_data, y='y', treat='treat', post='post',
                  rope=(-0.1, 0.1),
                  draws=300, tune=300, chains=2, progressbar=False)
    assert r.prob_rope is not None
    assert 0.0 <= r.prob_rope <= 1.0
