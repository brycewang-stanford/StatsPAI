"""Tests for front_door integrate_by='marginal' vs 'conditional'."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def fd_with_covariates():
    """Continuous-M front-door DGP with a baseline covariate."""
    rng = np.random.default_rng(42)
    n = 2000
    U = rng.normal(0, 1, n)
    X = rng.normal(0, 1, n)
    prob_D = 1 / (1 + np.exp(-(1.0 * U + 0.3 * X)))
    D = rng.binomial(1, prob_D, n).astype(float)
    M = 0.8 * D + 0.2 * X + rng.normal(0, 0.3, n)
    Y = 1.5 * M + 0.7 * U + 0.4 * X + rng.normal(0, 0.3, n)
    return pd.DataFrame({'y': Y, 'd': D, 'm': M, 'x': X})


def test_marginal_and_conditional_both_recover_true(fd_with_covariates):
    truth = 1.2  # 0.8 * 1.5
    r_m = sp.front_door(
        fd_with_covariates, y='y', treat='d', mediator='m',
        covariates=['x'], integrate_by='marginal',
        n_boot=80, n_mc=100, seed=0,
    )
    r_c = sp.front_door(
        fd_with_covariates, y='y', treat='d', mediator='m',
        covariates=['x'], integrate_by='conditional',
        n_boot=80, n_mc=100, seed=0,
    )
    assert abs(r_m.estimate - truth) < 0.25
    assert abs(r_c.estimate - truth) < 0.25


def test_integrate_by_recorded_in_model_info(fd_with_covariates):
    r = sp.front_door(
        fd_with_covariates, y='y', treat='d', mediator='m',
        covariates=['x'], integrate_by='conditional',
        n_boot=40, n_mc=50, seed=0,
    )
    assert r.model_info['integrate_by'] == 'conditional'


def test_integrate_by_rejects_bad_value(fd_with_covariates):
    with pytest.raises(ValueError, match='integrate_by'):
        sp.front_door(
            fd_with_covariates, y='y', treat='d', mediator='m',
            covariates=['x'], integrate_by='nonsense',
        )
