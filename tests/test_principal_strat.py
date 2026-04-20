"""Tests for Principal Stratification."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.principal_strat import PrincipalStratResult


@pytest.fixture
def air_dgp():
    """
    AIR-style monotonicity DGP.
    - 60% compliers (S(0)=0, S(1)=1)
    - 20% always-takers (S(0)=S(1)=1)
    - 20% never-takers (S(0)=S(1)=0)
    No defiers.
    Complier PCE (LATE) = 2.0.
    """
    rng = np.random.default_rng(42)
    n = 3000
    # Random treatment assignment
    D = rng.binomial(1, 0.5, n).astype(float)
    # Compliance types
    u = rng.uniform(0, 1, n)
    types = np.where(
        u < 0.6, 'C',
        np.where(u < 0.8, 'A', 'N'),
    )
    # Realize S = S(d): complier takes D, always-taker =1, never-taker =0
    S = np.where(types == 'A', 1.0,
                 np.where(types == 'N', 0.0, D))
    # Y depends on stratum and realized D; LATE = 2.0 on compliers
    Y = np.where(
        types == 'C', 2.0 * D + rng.normal(0, 0.3, n),
        np.where(
            types == 'A', 5.0 + rng.normal(0, 0.3, n),
            0.0 + rng.normal(0, 0.3, n),
        ),
    )
    return pd.DataFrame({'y': Y, 'd': D, 's': S, 'type': types})


def test_principal_strat_monotonicity_late(air_dgp):
    result = sp.principal_strat(
        air_dgp, y='y', treat='d', strata='s',
        method='monotonicity', n_boot=200, seed=0,
    )
    assert isinstance(result, PrincipalStratResult)
    # Complier LATE row
    late_row = result.effects.iloc[0]
    assert abs(late_row['estimate'] - 2.0) < 0.2


def test_principal_strat_stratum_proportions(air_dgp):
    result = sp.principal_strat(
        air_dgp, y='y', treat='d', strata='s',
        method='monotonicity', n_boot=50, seed=0,
    )
    # True proportions: A=0.2, C=0.6, N=0.2
    props = result.strata_proportions
    assert abs(props['complier'] - 0.6) < 0.05
    assert abs(props['always-taker / always-survivor'] - 0.2) < 0.05
    assert abs(props['never-taker / never-survivor'] - 0.2) < 0.05


def test_principal_strat_sace_bounds_valid(air_dgp):
    """SACE bounds should be a valid interval (lower <= upper)."""
    result = sp.principal_strat(
        air_dgp, y='y', treat='d', strata='s',
        method='monotonicity', n_boot=50, seed=0,
    )
    lo = result.bounds.loc[0, 'estimate']
    hi = result.bounds.loc[1, 'estimate']
    assert lo <= hi


def test_principal_score_method_with_covariates():
    """
    Principal score needs covariates that predict stratum membership —
    otherwise e_s(X) is flat and the estimator collapses to the biased
    unweighted cell mean. Build a DGP where X1, X2 carry stratum signal.
    """
    rng = np.random.default_rng(42)
    n = 4000
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    # Stratum type depends on X1, X2 (ordered logit-like over {N, C, A})
    score = 0.8 * X1 + 0.5 * X2 + rng.normal(0, 0.5, n)
    # Thresholds chosen so marginal proportions ≈ 0.2 N, 0.6 C, 0.2 A
    lo, hi = np.quantile(score, [0.2, 0.8])
    types = np.where(score < lo, 'N',
                     np.where(score < hi, 'C', 'A'))
    D = rng.binomial(1, 0.5, n).astype(float)
    S = np.where(types == 'A', 1.0,
                 np.where(types == 'N', 0.0, D))
    Y = np.where(
        types == 'C', 2.0 * D + rng.normal(0, 0.3, n),
        np.where(
            types == 'A', 5.0 + rng.normal(0, 0.3, n),
            0.0 + rng.normal(0, 0.3, n),
        ),
    )
    df = pd.DataFrame({'y': Y, 'd': D, 's': S, 'x1': X1, 'x2': X2})

    result = sp.principal_strat(
        df, y='y', treat='d', strata='s',
        method='principal_score', covariates=['x1', 'x2'],
        n_boot=100, seed=0,
    )
    complier_row = result.effects[result.effects['stratum'] == 'Complier PCE'].iloc[0]
    # With informative X, principal-score recovers LATE ≈ 2.0 within 0.5.
    # Tolerance is loose because principal ignorability is an assumption —
    # we measure approach, not exactness.
    assert abs(complier_row['estimate'] - 2.0) < 0.5


def test_principal_score_requires_covariates(air_dgp):
    with pytest.raises(ValueError, match='covariate'):
        sp.principal_strat(
            air_dgp, y='y', treat='d', strata='s',
            method='principal_score',
        )


def test_principal_strat_rejects_bad_method(air_dgp):
    with pytest.raises(ValueError, match='method'):
        sp.principal_strat(
            air_dgp, y='y', treat='d', strata='s',
            method='wrong_method',
        )


def test_sace_helper(air_dgp):
    result = sp.survivor_average_causal_effect(
        air_dgp, y='y', treat='d', survival='s',
        n_boot=100, seed=0,
    )
    assert 'sace_lower' in result.model_info
    assert 'sace_upper' in result.model_info
    assert result.model_info['sace_lower'] <= result.model_info['sace_upper']
