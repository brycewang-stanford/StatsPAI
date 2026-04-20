"""
Contract tests for the "deferred-with-escape-hatch" items documented
in docs/ROADMAP.md.

Each test locks in the **user-facing contract** that lets callers work
around the deferred feature — so future refactors that silently break
the escape hatch get caught.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult


# ---------------------------------------------------------------------
# Escape hatch 1 — Proximal bridge='kernel' / 'sieve' raise, stay documented
# ---------------------------------------------------------------------

@pytest.mark.parametrize('bridge', ['kernel', 'sieve'])
def test_proximal_kernel_and_sieve_bridges_raise(bridge):
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
                    bridge=bridge)


# ---------------------------------------------------------------------
# Escape hatch 2 — PLIV multi-instrument points at scalar_iv_projection
# ---------------------------------------------------------------------

def test_pliv_multi_instrument_error_mentions_scalar_iv_projection():
    df = pd.DataFrame({
        'y':  np.arange(50, dtype=float) / 50,
        'd':  np.arange(50, dtype=float) * 0.2,
        'z1': np.linspace(0, 1, 50),
        'z2': np.linspace(1, 0, 50),
        'x':  np.linspace(-1, 1, 50),
    })
    with pytest.raises(ValueError) as exc_info:
        sp.dml(df, y='y', treat='d', covariates=['x'],
               model='pliv', instrument=['z1', 'z2'])
    msg = str(exc_info.value)
    assert 'scalar_iv_projection' in msg, (
        "Multi-instrument PLIV error should point users to the "
        "documented escape hatch sp.scalar_iv_projection."
    )


def test_scalar_iv_projection_end_to_end_with_pliv():
    """
    Full round-trip: a DGP with two instruments, use scalar_iv_projection
    to build a scalar index, feed it to PLIV, and recover the treatment
    effect.
    """
    rng = np.random.default_rng(42)
    n = 3000
    X = rng.normal(0, 1, n)
    U = rng.normal(0, 1, n)
    Z1 = rng.normal(0, 1, n)
    Z2 = rng.normal(0, 1, n)
    # Endogenous D: depends on Z1, Z2, X, and unobserved U
    D = 0.5 * Z1 + 0.5 * Z2 + 0.3 * X + 0.7 * U + rng.normal(0, 0.3, n)
    Y = 1.2 * D + 0.5 * X + 0.7 * U + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z1': Z1, 'z2': Z2, 'x': X})

    # Use the documented escape hatch
    df_aug = sp.scalar_iv_projection(
        df, treat='d', instruments=['z1', 'z2'], covariates=['x'],
    )
    assert 'd_iv_hat' in df_aug.columns
    r = sp.dml(df_aug, y='y', treat='d', covariates=['x'],
               model='pliv', instrument='d_iv_hat')
    assert isinstance(r, CausalResult)
    assert abs(r.estimate - 1.2) < 0.25


def test_scalar_iv_projection_returns_series_option():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        'd': rng.normal(0, 1, n),
        'z1': rng.normal(0, 1, n),
        'z2': rng.normal(0, 1, n),
    })
    s = sp.scalar_iv_projection(
        df, treat='d', instruments=['z1', 'z2'],
        return_column=True,
    )
    assert isinstance(s, pd.Series)
    assert len(s) == n


def test_scalar_iv_projection_rejects_empty_instruments():
    df = pd.DataFrame({'d': [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match='non-empty'):
        sp.scalar_iv_projection(df, treat='d', instruments=[])


# ---------------------------------------------------------------------
# Escape hatch 3 — MSM trim_per_period flag
# ---------------------------------------------------------------------

def _msm_panel(seed=0, n_units=200, T=3, weight_shock=True):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        V = rng.normal()
        L_t = rng.normal()
        A_prev = 0.0
        for t in range(T):
            # Strong confounder signal to generate some extreme per-period
            # weights, so trimming at either stage has measurable effect.
            L_t = L_t + 0.5 * A_prev + rng.normal(0, 0.3)
            mult = 3.0 if weight_shock else 1.0
            A = float(rng.binomial(1, 1 / (1 + np.exp(-mult * L_t))))
            A_prev = A
            rows.append({'id': i, 'time': t, 'A': A, 'L_lag': L_t, 'V': V})
    panel = pd.DataFrame(rows)
    panel['Y'] = (
        0.4 * panel.groupby('id')['A'].cumsum()
        + panel['V'] * 0.2
        + rng.normal(0, 0.3, len(panel))
    )
    return panel


def test_msm_trim_per_period_records_flag():
    panel = _msm_panel(seed=7)
    r = sp.msm(panel, y='Y', treat='A', id='id', time='time',
               time_varying=['L_lag'], baseline=['V'],
               trim_per_period=True)
    assert r.model_info['trim_per_period'] is True


def test_msm_trim_per_period_default_matches_old_behaviour():
    panel = _msm_panel(seed=7)
    r_default = sp.msm(panel, y='Y', treat='A', id='id', time='time',
                       time_varying=['L_lag'], baseline=['V'])
    assert r_default.model_info['trim_per_period'] is False


def test_msm_trim_per_period_reduces_extreme_weights():
    """
    Per-period trimming should keep the max stabilised weight smaller
    than post-cumulative trimming on a DGP with weight blow-up.
    """
    panel = _msm_panel(seed=42)
    r_post = sp.msm(panel, y='Y', treat='A', id='id', time='time',
                    time_varying=['L_lag'], baseline=['V'],
                    trim=0.02, trim_per_period=False)
    r_pre = sp.msm(panel, y='Y', treat='A', id='id', time='time',
                   time_varying=['L_lag'], baseline=['V'],
                   trim=0.02, trim_per_period=True)
    # Per-period trimming produces a max weight ≤ post-cum trimming
    assert r_pre.model_info['sw_max'] <= r_post.model_info['sw_max'] + 1e-6


# ---------------------------------------------------------------------
# Escape hatch 4 — front_door continuous D error message points to g_computation
# ---------------------------------------------------------------------

def test_front_door_continuous_D_error_points_to_g_computation():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        'y': rng.normal(0, 1, n),
        'd': rng.normal(0, 1, n),  # continuous — not supported
        'm': rng.normal(0, 1, n),
    })
    with pytest.raises(ValueError) as exc_info:
        sp.front_door(df, y='y', treat='d', mediator='m')
    msg = str(exc_info.value)
    assert 'g_computation' in msg
    assert 'dose_response' in msg


# ---------------------------------------------------------------------
# Escape hatch 5 — mediate_interventional pvalue_method kwarg
# ---------------------------------------------------------------------

def _med_dgp(seed=42):
    rng = np.random.default_rng(seed)
    n = 2000
    X = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n).astype(float)
    L = 0.3 * D + rng.normal(0, 0.3, n)
    M = 0.5 * D + 0.3 * L + rng.normal(0, 0.3, n)
    Y = 0.4 * D + 0.8 * M + 0.3 * L + rng.normal(0, 0.3, n)
    return pd.DataFrame({'y': Y, 'd': D, 'm': M, 'l': L, 'x': X})


def test_mediate_interventional_default_is_bootstrap_sign():
    df = _med_dgp()
    r = sp.mediate_interventional(
        df, y='y', treat='d', mediator='m',
        covariates=['x'], tv_confounders=['l'],
        n_boot=60, n_mc=80, seed=0,
    )
    assert r.model_info['pvalue_method'] == 'bootstrap_sign'


def test_mediate_interventional_wald_pvalue():
    df = _med_dgp()
    r_sign = sp.mediate_interventional(
        df, y='y', treat='d', mediator='m',
        covariates=['x'], tv_confounders=['l'],
        n_boot=60, n_mc=80, seed=0,
        pvalue_method='bootstrap_sign',
    )
    r_wald = sp.mediate_interventional(
        df, y='y', treat='d', mediator='m',
        covariates=['x'], tv_confounders=['l'],
        n_boot=60, n_mc=80, seed=0,
        pvalue_method='wald',
    )
    # Same point estimate; different p-value convention
    assert r_wald.model_info['pvalue_method'] == 'wald'
    assert r_sign.model_info['pvalue_method'] == 'bootstrap_sign'
    assert abs(r_sign.estimate - r_wald.estimate) < 1e-10
    # Wald p-value should be a valid probability
    assert 0 <= r_wald.pvalue <= 1


def test_mediate_interventional_rejects_bad_pvalue_method():
    df = _med_dgp()
    with pytest.raises(ValueError, match='pvalue_method'):
        sp.mediate_interventional(
            df, y='y', treat='d', mediator='m',
            pvalue_method='garbage',
            n_boot=10,
        )
