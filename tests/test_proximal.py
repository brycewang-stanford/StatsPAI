"""Tests for Proximal Causal Inference (linear 2SLS bridge)."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult


@pytest.fixture
def proximal_dgp():
    """
    DGP with unmeasured U; Z is a pre-treatment proxy, W is a
    post-treatment proxy.

        U ~ N(0, 1)
        D = 0.8*U + noise_D         (U -> D)
        Z = 0.9*U + noise_Z         (U -> Z)   [treatment proxy]
        W = 0.9*U + noise_W         (U -> W)   [outcome proxy]
        Y = 1.5*D + 0.7*U + noise_Y

    Naive OLS on Y ~ D is biased by U. Proximal uses (Z, W) to
    untangle U's contribution; linear bridge gives γ_D = 1.5.
    """
    rng = np.random.default_rng(42)
    n = 3000
    U = rng.normal(0, 1, n)
    D = 0.8 * U + rng.normal(0, 0.5, n)
    Z = 0.9 * U + rng.normal(0, 0.3, n)
    W = 0.9 * U + rng.normal(0, 0.3, n)
    Y = 1.5 * D + 0.7 * U + rng.normal(0, 0.3, n)
    return pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'w': W, 'u': U})


def test_proximal_recovers_true_ate(proximal_dgp):
    result = sp.proximal(
        proximal_dgp, y='y', treat='d',
        proxy_z=['z'], proxy_w=['w'],
    )
    assert isinstance(result, CausalResult)
    assert abs(result.estimate - 1.5) < 0.15, (
        f"Proximal estimate {result.estimate:.3f}, expected ≈ 1.5"
    )


def test_proximal_beats_naive(proximal_dgp):
    df = proximal_dgp
    # Naive OLS of Y on D — biased upward by U
    x_design = np.column_stack([np.ones(len(df)), df['d'].values])
    beta_ols = np.linalg.lstsq(x_design, df['y'].values, rcond=None)[0]
    naive = beta_ols[1]

    prox = sp.proximal(df, y='y', treat='d',
                       proxy_z=['z'], proxy_w=['w']).estimate
    # Proximal should be strictly closer to 1.5 than naive OLS
    assert abs(prox - 1.5) < abs(naive - 1.5)


def test_proximal_order_condition_check():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        'y': rng.normal(0, 1, n),
        'd': rng.normal(0, 1, n),
        'z1': rng.normal(0, 1, n),
        'w1': rng.normal(0, 1, n),
        'w2': rng.normal(0, 1, n),
    })
    with pytest.raises(ValueError, match='Order condition'):
        sp.proximal(df, y='y', treat='d',
                    proxy_z=['z1'], proxy_w=['w1', 'w2'])


def test_proximal_with_covariates(proximal_dgp):
    """Adding an unrelated covariate should leave ATE ≈ unchanged."""
    df = proximal_dgp.copy()
    rng = np.random.default_rng(1)
    df['x'] = rng.normal(0, 1, len(df))
    result = sp.proximal(
        df, y='y', treat='d',
        proxy_z=['z'], proxy_w=['w'], covariates=['x'],
    )
    assert abs(result.estimate - 1.5) < 0.2


def test_proximal_bootstrap_se(proximal_dgp):
    result = sp.proximal(
        proximal_dgp, y='y', treat='d',
        proxy_z=['z'], proxy_w=['w'],
        n_boot=100, seed=0,
    )
    assert result.model_info['se_method'] == 'bootstrap'
    assert result.model_info['n_boot'] == 100
    assert result.se > 0


def test_proximal_first_stage_F_reported(proximal_dgp):
    result = sp.proximal(proximal_dgp, y='y', treat='d',
                         proxy_z=['z'], proxy_w=['w'])
    assert result.model_info['first_stage_F'] is not None
    # With strong proxies, F should be large
    assert result.model_info['first_stage_F'] > 10
