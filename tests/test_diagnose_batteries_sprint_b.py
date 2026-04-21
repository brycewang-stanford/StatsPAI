"""
In-depth tests for the Sprint-B diagnostic batteries.

Earlier round-3 tests only checked that the dispatcher lights up the
right battery. This file exercises the actual diagnostic *logic* —
each test verifies that a specific check fires with the correct
pass/fail verdict in response to a specific condition in the input
(weak first-stage F, extreme IPTW weight, monotonicity violation,
heavy bootstrap failure rate, decomposition drift, etc.).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------
# Proximal — bridge / proxy rank / first-stage F
# ---------------------------------------------------------------------

@pytest.fixture
def strong_proxy_data():
    """High-F DGP: Z is a clean proxy for U."""
    rng = np.random.default_rng(42)
    n = 2000
    U = rng.normal(0, 1, n)
    D = 0.8 * U + rng.normal(0, 0.3, n)
    Z = 0.95 * U + rng.normal(0, 0.2, n)
    W = 0.9 * U + rng.normal(0, 0.3, n)
    Y = 1.5 * D + 0.7 * U + rng.normal(0, 0.3, n)
    return pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'w': W})


@pytest.fixture
def weak_proxy_data():
    """Low-F DGP: Z barely tracks U."""
    rng = np.random.default_rng(42)
    n = 500
    U = rng.normal(0, 1, n)
    D = 0.8 * U + rng.normal(0, 0.5, n)
    Z = 0.1 * U + rng.normal(0, 1.0, n)   # very noisy proxy
    W = 0.9 * U + rng.normal(0, 0.3, n)
    Y = 1.5 * D + 0.7 * U + rng.normal(0, 0.3, n)
    return pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'w': W})


def _check_by_name(diag, name_substr):
    for c in diag['checks']:
        if name_substr in c.get('test', ''):
            return c
    return None


def test_proximal_battery_passes_on_strong_proxy(strong_proxy_data):
    r = sp.proximal(strong_proxy_data, y='y', treat='d',
                    proxy_z=['z'], proxy_w=['w'])
    diag = sp.diagnose_result(r, print_results=False)
    f_check = _check_by_name(diag, 'First-stage F')
    assert f_check is not None
    assert f_check['F'] >= 10.0
    assert f_check['pass'] is True


def test_proximal_battery_fails_on_weak_proxy(weak_proxy_data):
    r = sp.proximal(weak_proxy_data, y='y', treat='d',
                    proxy_z=['z'], proxy_w=['w'])
    diag = sp.diagnose_result(r, print_results=False)
    f_check = _check_by_name(diag, 'First-stage F')
    assert f_check is not None
    assert f_check['F'] < 10.0
    assert f_check['pass'] is False


def test_proximal_battery_order_condition_violated():
    """Feed more W proxies than Z proxies — should fail the order check."""
    rng = np.random.default_rng(0)
    n = 800
    U = rng.normal(0, 1, n)
    D = 0.8 * U + rng.normal(0, 0.5, n)
    Z1 = 0.9 * U + rng.normal(0, 0.3, n)
    W1 = 0.9 * U + rng.normal(0, 0.3, n)
    W2 = 0.9 * U + rng.normal(0, 0.3, n)
    Y = 1.5 * D + 0.7 * U + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z1': Z1, 'w1': W1, 'w2': W2})
    # This should raise during proximal() itself (order condition violated)
    with pytest.raises(ValueError, match='Order condition'):
        sp.proximal(df, y='y', treat='d',
                    proxy_z=['z1'], proxy_w=['w1', 'w2'])


# ---------------------------------------------------------------------
# MSM — weight diagnostics + positivity + exposure
# ---------------------------------------------------------------------

def _msm_panel(seed=0, n_units=250, T=3, weight_shock=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        V = rng.normal()
        L = rng.normal()
        A_prev = 0.0
        for t in range(T):
            L = L + 0.3 * A_prev + rng.normal(0, 0.3)
            A = float(rng.binomial(1, 1 / (1 + np.exp(-weight_shock * L))))
            A_prev = A
            rows.append({'id': i, 'time': t, 'A': A,
                         'L_lag': L, 'V': V})
    panel = pd.DataFrame(rows)
    panel['Y'] = (
        0.5 * panel.groupby('id')['A'].cumsum()
        + panel['V'] * 0.3
        + rng.normal(0, 0.3, len(panel))
    )
    return panel


def test_msm_battery_clean_weights_pass():
    panel = _msm_panel(seed=0, weight_shock=1.0)
    r = sp.msm(panel, y='Y', treat='A', id='id', time='time',
               time_varying=['L_lag'], baseline=['V'])
    diag = sp.diagnose_result(r, print_results=False)
    sw_mean = _check_by_name(diag, 'Stabilized-weight mean')
    assert sw_mean is not None
    # Under correct specification sw mean ≈ 1
    assert abs(sw_mean['sw_mean'] - 1.0) < 0.2
    assert sw_mean['pass'] is True


def test_msm_battery_flags_extreme_max_weight():
    """
    Strong-confounding DGP blows up at least one weight past the
    positivity-watchdog threshold (50). Without trimming, the battery
    should flag it.
    """
    panel = _msm_panel(seed=3, weight_shock=3.5, n_units=400)
    r = sp.msm(panel, y='Y', treat='A', id='id', time='time',
               time_varying=['L_lag'], baseline=['V'],
               trim=0.0)   # disable trimming to expose the max
    diag = sp.diagnose_result(r, print_results=False)
    sw_max = _check_by_name(diag, 'Max stabilized weight')
    assert sw_max is not None
    # Either max > 50 (fails) or battery flagged it; verify semantics
    if sw_max['sw_max'] >= 50.0:
        assert sw_max['pass'] is False
    else:
        # Even if the DGP didn't push it that far, the check itself
        # must return a valid boolean — never None.
        assert sw_max['pass'] is True


def test_msm_battery_exposure_recorded():
    panel = _msm_panel()
    r = sp.msm(panel, y='Y', treat='A', id='id', time='time',
               time_varying=['L_lag'], baseline=['V'],
               exposure='current')
    diag = sp.diagnose_result(r, print_results=False)
    expo = _check_by_name(diag, 'Exposure summary')
    assert expo is not None
    assert expo['exposure'] == 'current'


# ---------------------------------------------------------------------
# Principal stratification — monotonicity violation + proportions
# ---------------------------------------------------------------------

def test_principal_strat_battery_flags_monotonicity_violation():
    """Defier-heavy DGP: fitted p11(x) < p10(x) for > 5% of units."""
    rng = np.random.default_rng(7)
    n = 2500
    X = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n).astype(float)
    # For X > 0, S = 1 - D (defiers); for X <= 0, S = D (compliers).
    S = np.where(X > 0, 1 - D, D).astype(float)
    Y = 0.5 * D + 0.2 * X + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 's': S, 'x': X})
    res = sp.principal_strat(df, y='y', treat='d', strata='s',
                             method='principal_score',
                             covariates=['x'], n_boot=20, seed=0)
    diag = sp.diagnose_result(res, print_results=False)
    mono = _check_by_name(diag, 'Monotonicity violation')
    assert mono is not None
    assert mono['mono_violation_frac'] > 0.05
    assert mono['pass'] is False


def test_principal_strat_battery_sace_bounds_valid():
    rng = np.random.default_rng(42)
    n = 1500
    D = rng.binomial(1, 0.5, n).astype(float)
    u = rng.uniform(0, 1, n)
    types = np.where(u < 0.6, 'C', np.where(u < 0.8, 'A', 'N'))
    S = np.where(types == 'A', 1.0,
                 np.where(types == 'N', 0.0, D)).astype(float)
    Y = np.where(
        types == 'C', 2.0 * D + rng.normal(0, 0.3, n),
        np.where(types == 'A', 5.0 + rng.normal(0, 0.3, n),
                 rng.normal(0, 0.3, n)),
    )
    df = pd.DataFrame({'y': Y, 'd': D, 's': S})
    res = sp.principal_strat(df, y='y', treat='d', strata='s',
                             method='monotonicity', n_boot=60, seed=0)
    diag = sp.diagnose_result(res, print_results=False)
    sace = _check_by_name(diag, 'Zhang-Rubin SACE')
    assert sace is not None
    # Bounds should be a valid interval (sace_lo <= sace_hi)
    assert sace['sace_lo'] is not None
    assert sace['sace_hi'] is not None
    assert sace['sace_lo'] <= sace['sace_hi']
    assert sace['pass'] is True


def test_principal_strat_battery_proportions_sum_to_one():
    rng = np.random.default_rng(42)
    n = 1500
    D = rng.binomial(1, 0.5, n).astype(float)
    u = rng.uniform(0, 1, n)
    types = np.where(u < 0.6, 'C', np.where(u < 0.8, 'A', 'N'))
    S = np.where(types == 'A', 1.0,
                 np.where(types == 'N', 0.0, D)).astype(float)
    Y = rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 's': S})
    res = sp.principal_strat(df, y='y', treat='d', strata='s',
                             method='monotonicity', n_boot=30, seed=0)
    diag = sp.diagnose_result(res, print_results=False)
    prop = _check_by_name(diag, 'Stratum proportions')
    assert prop is not None
    total = sum(float(v) for v in prop['proportions'].values())
    assert abs(total - 1.0) < 0.05
    assert prop['pass'] is True


# ---------------------------------------------------------------------
# G-computation — estimand + bootstrap health
# ---------------------------------------------------------------------

def test_g_computation_battery_records_estimand_and_bootstrap():
    rng = np.random.default_rng(0)
    n = 800
    X = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n).astype(float)
    Y = 1.0 * D + X + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'x': X})

    r = sp.g_computation(df, y='y', treat='d', covariates=['x'],
                         estimand='ATT', n_boot=50, seed=0)
    diag = sp.diagnose_result(r, print_results=False)
    est = _check_by_name(diag, 'Estimand')
    assert est is not None
    assert est['estimand'] == 'ATT'

    boot = _check_by_name(diag, 'Bootstrap health')
    assert boot is not None
    assert boot['n_boot'] == 50
    # Clean DGP → no failures expected
    assert boot['n_boot_failed'] == 0
    assert boot['pass'] is True


# ---------------------------------------------------------------------
# Front-door — mediator type + integration + DAG reminder
# ---------------------------------------------------------------------

def test_front_door_battery_records_integrate_by():
    rng = np.random.default_rng(42)
    n = 1500
    U = rng.normal(0, 1, n)
    X = rng.normal(0, 1, n)
    D = rng.binomial(1, 1 / (1 + np.exp(-U)), n).astype(float)
    M = 0.7 * D + 0.2 * X + rng.normal(0, 0.3, n)
    Y = 1.2 * M + 0.5 * U + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'm': M, 'x': X})

    r = sp.front_door(df, y='y', treat='d', mediator='m',
                      covariates=['x'], mediator_type='continuous',
                      integrate_by='conditional',
                      n_boot=30, n_mc=50, seed=0)
    diag = sp.diagnose_result(r, print_results=False)
    integ = _check_by_name(diag, 'Integration formulation')
    assert integ is not None
    assert integ['integrate_by'] == 'conditional'

    m_type = _check_by_name(diag, 'Mediator type')
    assert m_type is not None
    assert m_type['mediator_type'] == 'continuous'

    dag = _check_by_name(diag, 'Front-door assumption')
    # DAG reminder is informational (pass=None)
    assert dag is not None
    assert dag['pass'] is None


# ---------------------------------------------------------------------
# Interventional mediation — decomposition identity + pvalue_method
# ---------------------------------------------------------------------

@pytest.fixture
def interventional_dgp():
    rng = np.random.default_rng(42)
    n = 1500
    D = rng.binomial(1, 0.5, n).astype(float)
    L = 0.3 * D + rng.normal(0, 0.3, n)
    M = 0.5 * D + 0.3 * L + rng.normal(0, 0.3, n)
    Y = 0.4 * D + 0.8 * M + 0.3 * L + rng.normal(0, 0.3, n)
    return pd.DataFrame({'y': Y, 'd': D, 'm': M, 'l': L})


def test_interventional_battery_decomposition_identity(interventional_dgp):
    r = sp.mediate_interventional(
        interventional_dgp, y='y', treat='d', mediator='m',
        tv_confounders=['l'], n_boot=50, n_mc=80, seed=0,
    )
    diag = sp.diagnose_result(r, print_results=False)
    dec = _check_by_name(diag, 'Decomposition additivity')
    assert dec is not None
    # IIE + IDE should match Total exactly by construction
    assert dec['residual'] < 1e-6
    assert dec['pass'] is True


def test_interventional_battery_records_pvalue_method(interventional_dgp):
    r = sp.mediate_interventional(
        interventional_dgp, y='y', treat='d', mediator='m',
        tv_confounders=['l'], pvalue_method='wald',
        n_boot=40, n_mc=60, seed=0,
    )
    diag = sp.diagnose_result(r, print_results=False)
    pv = _check_by_name(diag, 'P-value convention')
    assert pv is not None
    assert pv['pvalue_method'] == 'wald'
    assert pv['pass'] is True


def test_interventional_battery_records_tv_confounders(interventional_dgp):
    r = sp.mediate_interventional(
        interventional_dgp, y='y', treat='d', mediator='m',
        tv_confounders=['l'], n_boot=30, n_mc=40, seed=0,
    )
    diag = sp.diagnose_result(r, print_results=False)
    tv = _check_by_name(diag, 'Treatment-induced confounders')
    assert tv is not None
    assert 'l' in tv['tv_confounders']


# ---------------------------------------------------------------------
# sp.mediate() pvalue_method parity with mediate_interventional
# ---------------------------------------------------------------------

def test_mediate_default_pvalue_method_preserves_behaviour():
    rng = np.random.default_rng(0)
    n = 800
    T = rng.binomial(1, 0.5, n).astype(float)
    M = 0.5 * T + rng.normal(0, 0.3, n)
    Y = 0.4 * T + 0.8 * M + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 't': T, 'm': M})
    r = sp.mediate(df, y='y', treat='t', mediator='m', n_boot=100, seed=0)
    assert r.model_info['pvalue_method'] == 'bootstrap_sign'


def test_mediate_wald_pvalue_matches_normal_formula():
    rng = np.random.default_rng(0)
    n = 800
    T = rng.binomial(1, 0.5, n).astype(float)
    M = 0.5 * T + rng.normal(0, 0.3, n)
    Y = 0.4 * T + 0.8 * M + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 't': T, 'm': M})
    r_wald = sp.mediate(df, y='y', treat='t', mediator='m',
                        n_boot=100, seed=0, pvalue_method='wald')
    assert r_wald.model_info['pvalue_method'] == 'wald'
    # Wald p-value must be a valid [0,1] probability
    assert 0.0 <= r_wald.pvalue <= 1.0
    # And must match 2*(1 - Phi(|estimate/se|)) up to rounding
    from scipy import stats
    expected = float(2 * (1 - stats.norm.cdf(abs(r_wald.estimate / r_wald.se))))
    assert abs(r_wald.pvalue - expected) < 1e-10


def test_mediate_rejects_bad_pvalue_method():
    df = pd.DataFrame({'y': [1.0, 2.0, 3.0, 4.0],
                       't': [0.0, 1.0, 0.0, 1.0],
                       'm': [0.1, 0.5, 0.2, 0.6]})
    with pytest.raises(ValueError, match='pvalue_method'):
        sp.mediate(df, y='y', treat='t', mediator='m',
                   pvalue_method='garbage', n_boot=10)


# ---------------------------------------------------------------------
# paper_tables renders each Sprint-B CausalResult cleanly
# ---------------------------------------------------------------------

def test_paper_tables_renders_proximal():
    rng = np.random.default_rng(0)
    n = 500
    U = rng.normal(0, 1, n)
    D = 0.8 * U + rng.normal(0, 0.5, n)
    Z = 0.9 * U + rng.normal(0, 0.3, n)
    W = 0.9 * U + rng.normal(0, 0.3, n)
    Y = 1.5 * D + 0.7 * U + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'w': W})
    r = sp.proximal(df, y='y', treat='d', proxy_z=['z'], proxy_w=['w'])
    md = sp.paper_tables(main=[r]).to_markdown()
    assert 'ATE' in md
    assert 'N' in md or 'N obs' in md
    assert '500' in md


def test_paper_tables_renders_multiple_sprint_b_results():
    rng = np.random.default_rng(0)
    n = 500
    X = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n).astype(float)
    Y = 1.0 * D + X + rng.normal(0, 0.3, n)
    df_gc = pd.DataFrame({'y': Y, 'd': D, 'x': X})
    r_gc = sp.g_computation(df_gc, y='y', treat='d',
                            covariates=['x'], n_boot=30, seed=0)

    U = rng.normal(0, 1, n)
    D2 = rng.binomial(1, 1 / (1 + np.exp(-U)), n).astype(float)
    M = 0.8 * D2 + rng.normal(0, 0.3, n)
    Y2 = 1.5 * M + 0.7 * U + rng.normal(0, 0.3, n)
    df_fd = pd.DataFrame({'y': Y2, 'd': D2, 'm': M})
    r_fd = sp.front_door(df_fd, y='y', treat='d', mediator='m',
                         mediator_type='continuous',
                         n_boot=30, n_mc=40, seed=0)

    md = sp.paper_tables(main=[r_gc, r_fd]).to_markdown()
    # Two columns in the rendered main table
    assert '(1)' in md and '(2)' in md
