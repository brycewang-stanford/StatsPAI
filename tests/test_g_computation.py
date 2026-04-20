"""Tests for g-computation estimator."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult


@pytest.fixture
def binary_dgp():
    """
    Binary-treatment DGP with linear outcome.
        P(D=1|X) = logistic(0.5*X1 + X2)
        Y = 2*D + X1 + 0.5*X2 + eps
    True ATE = 2.0.
    """
    rng = np.random.default_rng(42)
    n = 1500
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    prob = 1 / (1 + np.exp(-(0.5 * X1 + X2)))
    D = rng.binomial(1, prob, n).astype(float)
    Y = 2.0 * D + X1 + 0.5 * X2 + rng.normal(0, 0.5, n)
    return pd.DataFrame({'y': Y, 'd': D, 'x1': X1, 'x2': X2})


def test_g_computation_ate(binary_dgp):
    result = sp.g_computation(
        binary_dgp, y='y', treat='d', covariates=['x1', 'x2'],
        n_boot=200, seed=0,
    )
    assert isinstance(result, CausalResult)
    assert abs(result.estimate - 2.0) < 0.25
    assert result.pvalue < 0.05
    assert result.ci[0] < 2.0 < result.ci[1]


def test_g_computation_att(binary_dgp):
    result = sp.g_computation(
        binary_dgp, y='y', treat='d', covariates=['x1', 'x2'],
        estimand='ATT', n_boot=150, seed=0,
    )
    # Linear outcome + additive effect => ATE = ATT numerically
    assert abs(result.estimate - 2.0) < 0.3
    assert result.estimand == 'ATT'


def test_g_computation_dose_response():
    rng = np.random.default_rng(7)
    n = 1200
    X = rng.normal(0, 1, n)
    D = rng.uniform(0, 10, n)
    Y = 0.3 * D + 0.8 * X + rng.normal(0, 0.5, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'x': X})

    result = sp.g_computation(
        df, y='y', treat='d', covariates=['x'],
        estimand='dose_response',
        treat_values=[0, 2, 5, 8],
        n_boot=100, seed=0,
    )
    curve = result.detail
    assert len(curve) == 4
    # Slope between dose=0 and dose=8 should be ≈ 8 * 0.3 = 2.4
    slope = curve.loc[3, 'estimate'] - curve.loc[0, 'estimate']
    assert abs(slope - 2.4) < 0.3


def test_g_computation_rejects_nonbinary_ate():
    df = pd.DataFrame({'y': [1.0, 2.0, 3.0], 'd': [0.5, 1.5, 2.5], 'x': [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match='binary'):
        sp.g_computation(df, y='y', treat='d', covariates=['x'])


def test_g_computation_dose_response_requires_values():
    df = pd.DataFrame({'y': [1.0, 2.0], 'd': [0.5, 1.5], 'x': [1.0, 2.0]})
    with pytest.raises(ValueError, match='treat_values'):
        sp.g_computation(df, y='y', treat='d', covariates=['x'],
                         estimand='dose_response')
