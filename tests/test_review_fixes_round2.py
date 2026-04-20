"""
Regression tests for the second-pass post-review fixes.

Covers:
* IRM subgroup fit threshold (mirror of IIVM fix).
* Principal score monotonicity-violation warning.
* Proximal bridge='linear' scaffold and NotImplementedError for others.
* front_door model_info['integrate_by_effective'] records what ran.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.dml import DoubleMLIRM


# ---------------------------------------------------------------------
# IRM subgroup fit threshold mirroring IIVM
# ---------------------------------------------------------------------

def test_irm_has_min_subgroup_fit_constant():
    assert hasattr(DoubleMLIRM, '_MIN_SUBGROUP_FIT')
    assert DoubleMLIRM._MIN_SUBGROUP_FIT >= 5


def test_irm_raises_when_treatment_is_constant():
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(0, 1, n)
    D = np.ones(n)  # no variation
    Y = 2.0 + X + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'x': X})
    with pytest.raises(ValueError, match='variation in D'):
        sp.dml(df, y='y', treat='d', covariates=['x'], model='irm')


def test_irm_tiny_subgroup_does_not_blow_up():
    """
    Very unbalanced D with a small sample exercises the subgroup-fit
    fallback. Previously the GBM ran on a few training rows and
    produced noisy g1/g0; now subgroups below the threshold fall back
    to the subgroup mean. Estimator should stay finite.
    """
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(0, 1, n)
    # Rare treatment — ~5% treated
    D = (rng.uniform(0, 1, n) < 0.05).astype(float)
    # At least one treated/one control required
    D[0] = 1.0
    D[1] = 0.0
    Y = 1.5 * D + 0.5 * X + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'x': X})
    r = sp.dml(df, y='y', treat='d', covariates=['x'], model='irm',
               n_folds=2)
    assert np.isfinite(r.estimate)
    assert np.isfinite(r.se) and r.se > 0


# ---------------------------------------------------------------------
# Principal score monotonicity violation warning
# ---------------------------------------------------------------------

def test_principal_score_warns_on_monotonicity_violation():
    """
    DGP where S responds OPPOSITE to D for a subset of units (i.e.
    defiers dominate). The fitted p10(x) > p11(x) for those X values,
    raw e_complier goes negative, and we expect a RuntimeWarning.
    """
    rng = np.random.default_rng(7)
    n = 3000
    X = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n).astype(float)
    # For rows with X > 0, S = 1 - D (defiers); for X <= 0, S = D (compliers).
    S = np.where(X > 0, 1 - D, D).astype(float)
    Y = 0.5 * D + 0.2 * X + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 's': S, 'x': X})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sp.principal_strat(
            df, y='y', treat='d', strata='s',
            method='principal_score', covariates=['x'],
            n_boot=20, seed=0,
        )
    # A monotonicity warning should fire
    assert any('monotonicity' in str(w.message).lower() for w in caught), (
        "Expected a monotonicity warning for a defier-heavy DGP."
    )
    # And model_info records the violation fraction
    assert result.model_info['mono_violation_frac'] > 0.05


def test_principal_score_no_warning_when_monotonicity_holds():
    rng = np.random.default_rng(42)
    n = 2000
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    score = 0.8 * X1 + 0.5 * X2 + rng.normal(0, 0.5, n)
    lo, hi = np.quantile(score, [0.2, 0.8])
    types = np.where(score < lo, 'N',
                     np.where(score < hi, 'C', 'A'))
    D = rng.binomial(1, 0.5, n).astype(float)
    S = np.where(types == 'A', 1.0,
                 np.where(types == 'N', 0.0, D)).astype(float)
    Y = rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 's': S, 'x1': X1, 'x2': X2})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sp.principal_strat(
            df, y='y', treat='d', strata='s',
            method='principal_score', covariates=['x1', 'x2'],
            n_boot=20, seed=0,
        )
    mono_warnings = [w for w in caught
                     if 'monotonicity' in str(w.message).lower()]
    assert len(mono_warnings) == 0
    assert result.model_info['mono_violation_frac'] <= 0.05


# ---------------------------------------------------------------------
# Proximal bridge kwarg scaffold
# ---------------------------------------------------------------------

def test_proximal_rejects_unimplemented_bridge():
    rng = np.random.default_rng(0)
    n = 500
    U = rng.normal(0, 1, n)
    D = 0.8 * U + rng.normal(0, 0.5, n)
    Z = 0.9 * U + rng.normal(0, 0.3, n)
    W = 0.9 * U + rng.normal(0, 0.3, n)
    Y = 1.5 * D + 0.7 * U + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'w': W})
    with pytest.raises(NotImplementedError, match='bridge'):
        sp.proximal(df, y='y', treat='d',
                    proxy_z=['z'], proxy_w=['w'],
                    bridge='kernel')


def test_proximal_records_bridge_in_model_info():
    rng = np.random.default_rng(0)
    n = 800
    U = rng.normal(0, 1, n)
    D = 0.8 * U + rng.normal(0, 0.5, n)
    Z = 0.9 * U + rng.normal(0, 0.3, n)
    W = 0.9 * U + rng.normal(0, 0.3, n)
    Y = 1.5 * D + 0.7 * U + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'w': W})
    r = sp.proximal(df, y='y', treat='d',
                    proxy_z=['z'], proxy_w=['w'])
    assert r.model_info['bridge'] == 'linear'


# ---------------------------------------------------------------------
# front_door integrate_by_effective audit trail
# ---------------------------------------------------------------------

def test_front_door_effective_equals_user_request_with_covariates():
    rng = np.random.default_rng(42)
    n = 800
    U = rng.normal(0, 1, n)
    X = rng.normal(0, 1, n)
    D = rng.binomial(1, 1 / (1 + np.exp(-(U + 0.3 * X))), n).astype(float)
    M = 0.7 * D + 0.2 * X + rng.normal(0, 0.3, n)
    Y = 1.2 * M + 0.5 * U + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'm': M, 'x': X})

    r = sp.front_door(df, y='y', treat='d', mediator='m', covariates=['x'],
                      integrate_by='marginal', n_boot=30, n_mc=50, seed=0)
    assert r.model_info['integrate_by'] == 'marginal'
    assert r.model_info['integrate_by_effective'] == 'marginal'


def test_front_door_effective_annotates_no_covariates():
    rng = np.random.default_rng(42)
    n = 800
    U = rng.normal(0, 1, n)
    D = rng.binomial(1, 1 / (1 + np.exp(-U)), n).astype(float)
    M = 0.7 * D + rng.normal(0, 0.3, n)
    Y = 1.2 * M + 0.5 * U + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'm': M})

    r = sp.front_door(df, y='y', treat='d', mediator='m',
                      integrate_by='marginal', n_boot=30, n_mc=50, seed=0)
    # Without covariates the two integrate_by paths coincide; the
    # effective label should reflect that.
    assert r.model_info['integrate_by'] == 'marginal'
    assert 'conditional' in r.model_info['integrate_by_effective'].lower()


def test_front_door_effective_for_binary_mediator():
    rng = np.random.default_rng(42)
    n = 1000
    U = rng.normal(0, 1, n)
    D = rng.binomial(1, 1 / (1 + np.exp(-U)), n).astype(float)
    prob_M = np.where(D == 1, 0.8, 0.2)
    M = rng.binomial(1, prob_M, n).astype(float)
    Y = 2.0 * M + 0.5 * U + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'm': M})
    r = sp.front_door(df, y='y', treat='d', mediator='m',
                      mediator_type='binary',
                      integrate_by='marginal', n_boot=30, seed=0)
    assert 'binary' in r.model_info['integrate_by_effective'].lower()
