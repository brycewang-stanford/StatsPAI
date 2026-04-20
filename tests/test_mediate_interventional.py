"""Tests for interventional mediation effects."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult


@pytest.fixture
def interventional_dgp():
    """
    DGP with treatment-induced mediator-outcome confounder.

        D -> L  (post-treatment)
        L -> M, L -> Y
        D -> M, M -> Y, D -> Y
    """
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n).astype(float)
    # Treatment-induced confounder
    L = 0.5 * D + 0.4 * X + rng.normal(0, 0.3, n)
    # Mediator depends on D, L, X
    M = 0.6 * D + 0.3 * L + 0.2 * X + rng.normal(0, 0.3, n)
    # Outcome depends on D, M, L, X
    Y = 0.4 * D + 0.8 * M + 0.3 * L + 0.5 * X + rng.normal(0, 0.3, n)
    return pd.DataFrame({'y': Y, 'd': D, 'm': M, 'l': L, 'x': X})


def test_interventional_returns_result(interventional_dgp):
    result = sp.mediate_interventional(
        interventional_dgp, y='y', treat='d', mediator='m',
        covariates=['x'], tv_confounders=['l'],
        n_mc=150, n_boot=80, seed=0,
    )
    assert isinstance(result, CausalResult)
    assert result.estimand == 'IIE'


def test_interventional_detail_rows(interventional_dgp):
    result = sp.mediate_interventional(
        interventional_dgp, y='y', treat='d', mediator='m',
        covariates=['x'], tv_confounders=['l'],
        n_mc=100, n_boot=60, seed=0,
    )
    detail = result.detail
    assert list(detail['effect']) == [
        'IIE (interventional indirect)',
        'IDE (interventional direct)',
        'Total',
    ]
    # IIE + IDE should ≈ Total (definitional)
    iie = detail.loc[0, 'estimate']
    ide = detail.loc[1, 'estimate']
    total = detail.loc[2, 'estimate']
    assert abs((iie + ide) - total) < 0.05


def test_interventional_total_effect_sign(interventional_dgp):
    """With positive DGP, total effect should be positive."""
    result = sp.mediate_interventional(
        interventional_dgp, y='y', treat='d', mediator='m',
        covariates=['x'], tv_confounders=['l'],
        n_mc=100, n_boot=60, seed=0,
    )
    total = result.detail.loc[2, 'estimate']
    assert total > 0


def test_interventional_rejects_nonbinary_d():
    df = pd.DataFrame({
        'y': [1.0, 2.0, 3.0],
        'd': [0.5, 1.5, 2.0],
        'm': [0.1, 0.2, 0.3],
    })
    with pytest.raises(ValueError, match='binary'):
        sp.mediate_interventional(df, y='y', treat='d', mediator='m')
