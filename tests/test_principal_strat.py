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


# --------------------------------------------------------------------------- #
#  Step G: encouragement-design / instrument= path (Wald / AIR LATE)
# --------------------------------------------------------------------------- #

@pytest.fixture
def encouragement_dgp():
    """Z = encouragement, D = uptake, S = post-treatment stratum, Y = outcome.

    - 60% compliers w.r.t. Z (D switches with Z; D(0)=0, D(1)=1)
    - 30% never-takers (D=0 always)
    - 10% always-takers (D=1 always)
    No defiers (monotonicity holds).
    True LATE on Y = 1.5; true LATE on S = 0.4.
    """
    rng = np.random.default_rng(7)
    n = 4000
    Z = rng.binomial(1, 0.5, n).astype(float)
    u = rng.uniform(0, 1, n)
    types = np.where(u < 0.6, 'C', np.where(u < 0.9, 'N', 'A'))
    D = np.zeros(n)
    D[(types == 'C') & (Z == 1)] = 1.0
    D[types == 'A'] = 1.0
    # S: complier S = 0.4*D + noise (so τ_S ≈ 0.4 on D)
    base_s = 0.3 + 0.4 * D
    S = (rng.uniform(0, 1, n) < base_s).astype(float)
    # Y: complier τ_Y = 1.5 on D
    Y = 0.5 * D + 1.0 * D * (types == 'C').astype(float) + rng.normal(0, 0.4, n)
    # Adjust so the marginal complier LATE matches the target — easier
    # to express directly: Y = 1.5*D for compliers + noise.
    Y = np.where(
        types == 'C',
        1.5 * D + rng.normal(0, 0.5, n),
        rng.normal(0, 0.5, n),
    )
    return pd.DataFrame({'y': Y, 'd': D, 's': S, 'z': Z})


def test_instrument_path_returns_two_wald_lates(encouragement_dgp):
    """``principal_strat(instrument=...)`` reports both τ_Y and τ_S."""
    result = sp.principal_strat(
        encouragement_dgp, y='y', treat='d', strata='s',
        instrument='z', n_boot=100, seed=0,
    )
    assert isinstance(result, PrincipalStratResult)
    assert 'instrument_air' in result.method
    # Two LATE rows in effects: outcome and stratum
    strata = list(result.effects['stratum'])
    assert any('LATE on Y' in s for s in strata)
    assert any('LATE on S' in s for s in strata)


def test_instrument_path_recovers_true_complier_share(encouragement_dgp):
    """π_C(Z) ≈ 0.6 in the encouragement DGP."""
    result = sp.principal_strat(
        encouragement_dgp, y='y', treat='d', strata='s',
        instrument='z', n_boot=50, seed=0,
    )
    pi_c = result.strata_proportions['complier (w.r.t. Z)']
    # Tolerance accommodates sampling noise in n=4000.
    assert 0.50 <= pi_c <= 0.70, f"complier share off: {pi_c}"


def test_instrument_path_recovers_true_late_on_y(encouragement_dgp):
    """Wald LATE on Y ≈ 1.5 in the encouragement DGP."""
    result = sp.principal_strat(
        encouragement_dgp, y='y', treat='d', strata='s',
        instrument='z', n_boot=50, seed=0,
    )
    tau_y = result.effects.loc[
        result.effects['stratum'].str.contains('LATE on Y'), 'estimate'
    ].iloc[0]
    assert 1.0 <= tau_y <= 2.0, f"τ_Y off: {tau_y}"


def test_instrument_path_rejects_nonbinary_z(encouragement_dgp):
    df = encouragement_dgp.copy()
    df['z'] = df['z'] + 0.5  # not binary anymore
    with pytest.raises(ValueError, match="instrument must be binary"):
        sp.principal_strat(
            df, y='y', treat='d', strata='s',
            instrument='z', n_boot=10, seed=0,
        )


def test_instrument_path_warns_on_weak_first_stage(encouragement_dgp):
    """Z that does not predict D triggers the weak-IV warning."""
    df = encouragement_dgp.copy()
    # Deterministic independence: each Z cell has the same D mean, so the
    # first stage is exactly zero rather than sample-noise dependent.
    df['z'] = np.resize([0.0, 1.0], len(df))
    df['d'] = np.resize([0.0, 0.0, 1.0, 1.0], len(df))
    with pytest.warns(RuntimeWarning, match="Weak first stage"):
        sp.principal_strat(
            df, y='y', treat='d', strata='s',
            instrument='z', n_boot=20, seed=0,
        )


def test_instrument_path_warns_on_negative_first_stage(encouragement_dgp):
    """A reversed instrument must not be mislabeled as weak compliance."""
    df = encouragement_dgp.copy()
    df['z'] = 1.0 - df['z']

    with pytest.warns(RuntimeWarning, match="Negative first stage"):
        result = sp.principal_strat(
            df, y='y', treat='d', strata='s',
            instrument='z', n_boot=20, seed=0,
        )

    assert result.strata_proportions[
        'first_stage (D|Z=1 - D|Z=0)'
    ] < 0
    late_y = result.effects.loc[
        result.effects['stratum'].str.contains('LATE on Y')
    ].iloc[0]
    assert np.isnan(late_y['estimate'])
    assert np.isnan(late_y['ci_lower'])
