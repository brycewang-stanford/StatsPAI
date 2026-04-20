"""Tests for front-door adjustment estimator."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult


@pytest.fixture
def front_door_dgp():
    """
    Front-door DGP:
        U -> D, U -> Y       (unobserved confounder)
        D -> M, M -> Y       (mediator fully transmits D's effect)

    True ATE = E[Y|do(D=1)] - E[Y|do(D=0)]. With:
        M = 0.8*D + noise_M
        Y = 1.5*M + noise_Y
    the total causal effect of D on Y is 0.8 * 1.5 = 1.2.
    """
    rng = np.random.default_rng(42)
    n = 2500
    U = rng.normal(0, 1, n)
    # Strong U->D confounding
    prob_D = 1 / (1 + np.exp(-(1.5 * U)))
    D = rng.binomial(1, prob_D, n).astype(float)
    # D -> M (no direct U -> M, that's the front-door assumption)
    M = 0.8 * D + rng.normal(0, 0.3, n)
    # Y depends on M and U (but NOT directly on D)
    Y = 1.5 * M + 0.7 * U + rng.normal(0, 0.3, n)
    return pd.DataFrame({'y': Y, 'd': D, 'm': M, 'u': U})


def test_front_door_recovers_true_ate(front_door_dgp):
    """Front-door should recover ATE ≈ 1.2 even though U is unobserved."""
    result = sp.front_door(
        front_door_dgp, y='y', treat='d', mediator='m',
        mediator_type='continuous', n_boot=150, n_mc=100, seed=0,
    )
    assert isinstance(result, CausalResult)
    assert abs(result.estimate - 1.2) < 0.2, (
        f"Front-door estimate {result.estimate:.3f}, expected ≈ 1.2"
    )


def test_front_door_beats_naive_when_confounded(front_door_dgp):
    """Naive OLS would be biased by U; front-door closes the gap."""
    df = front_door_dgp
    # Naive Y ~ D regression (no adjustment)
    naive = np.polyfit(df['d'], df['y'], 1)[0]
    fd = sp.front_door(df, y='y', treat='d', mediator='m',
                       mediator_type='continuous',
                       n_boot=100, n_mc=100, seed=0)
    # Front-door estimate should be closer to 1.2 than naive
    assert abs(fd.estimate - 1.2) < abs(naive - 1.2)


def test_front_door_binary_mediator():
    rng = np.random.default_rng(1)
    n = 2000
    U = rng.normal(0, 1, n)
    D = rng.binomial(1, 1 / (1 + np.exp(-1.2 * U)), n).astype(float)
    # P(M=1|D) — binary mediator
    prob_M = np.where(D == 1, 0.8, 0.2)
    M = rng.binomial(1, prob_M, n).astype(float)
    Y = 2.0 * M + 0.5 * U + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'm': M})

    # ATE = (0.8 - 0.2) * 2.0 = 1.2
    result = sp.front_door(df, y='y', treat='d', mediator='m',
                           mediator_type='binary', n_boot=100, seed=0)
    assert abs(result.estimate - 1.2) < 0.25


def test_front_door_rejects_nonbinary_treatment():
    df = pd.DataFrame({
        'y': [1.0, 2.0, 3.0],
        'd': [0.5, 1.0, 1.5],
        'm': [0.1, 0.2, 0.3],
    })
    with pytest.raises(ValueError, match='binary D'):
        sp.front_door(df, y='y', treat='d', mediator='m')
