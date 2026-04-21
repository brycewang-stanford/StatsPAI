"""
Round-4 Smart-tool extensions:

1. compare_estimators(method_hints={...}) routes proximal / msm /
   principal_strat / mediate / mediate_interventional / front_door
   using per-method kwargs, with the ROADMAP §6 collision rule
   (hint wins + UserWarning on conflict).

2. sensitivity_dashboard gains Sprint-B-aware dimensions:
   'first_stage_f' (proximal), 'trim_sweep' (msm),
   'monotonicity' (principal_strat).

3. sp.g_computation rejects non-binary D with a COMPACT message
   (truncated value list), not a 400-float dump.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------
# C) g_computation compact error message
# ---------------------------------------------------------------------

def test_g_computation_binary_error_is_compact():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        'y': rng.normal(0, 1, n),
        'd': rng.normal(0, 1, n),   # continuous
        'x': rng.normal(0, 1, n),
    })
    with pytest.raises(ValueError) as exc_info:
        sp.g_computation(df, y='y', treat='d', covariates=['x'], n_boot=5)
    msg = str(exc_info.value)
    # Old behaviour dumped all 400 unique floats into the error
    # (~20 KB). The fixed version truncates to 5 values + an
    # "(N more)" suffix. Keep the cap generous enough for the suffix
    # but tight enough to catch the regression.
    assert len(msg) < 400, (
        f'error message is {len(msg)} chars; expected <400. Got: {msg}'
    )
    # Still actionable
    assert 'binary' in msg.lower()
    assert 'dose_response' in msg
    # Preview truncation suffix
    assert 'more' in msg


# ---------------------------------------------------------------------
# B) sensitivity_dashboard Sprint-B-aware dimensions
# ---------------------------------------------------------------------

@pytest.fixture
def proximal_dgp():
    rng = np.random.default_rng(0)
    n = 500
    U = rng.normal(0, 1, n)
    D = 0.8 * U + rng.normal(0, 0.5, n)
    Z = 0.9 * U + rng.normal(0, 0.3, n)
    W = 0.9 * U + rng.normal(0, 0.3, n)
    Y = 1.5 * D + 0.7 * U + rng.normal(0, 0.3, n)
    return pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'w': W})


def test_sensitivity_dashboard_proximal_first_stage_f_dimension(proximal_dgp):
    r = sp.proximal(proximal_dgp, y='y', treat='d',
                    proxy_z=['z'], proxy_w=['w'])
    dash = sp.sensitivity_dashboard(
        r, data=proximal_dgp, verbose=False,
        dimensions=['first_stage_f'],
    )
    dims = [d['dimension'] for d in dash.dimensions]
    assert 'Proximal first-stage F (weak-IV)' in dims
    # Strong proxies → should be 'stable'
    fs_dim = next(d for d in dash.dimensions
                  if d['dimension'] == 'Proximal first-stage F (weak-IV)')
    assert fs_dim['stable'] is True


def test_sensitivity_dashboard_proximal_weak_F_flagged_unstable():
    rng = np.random.default_rng(1)
    n = 500
    U = rng.normal(0, 1, n)
    D = 0.8 * U + rng.normal(0, 0.5, n)
    Z = 0.1 * U + rng.normal(0, 1.0, n)   # WEAK proxy
    W = 0.9 * U + rng.normal(0, 0.3, n)
    Y = 1.5 * D + 0.7 * U + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'w': W})
    r = sp.proximal(df, y='y', treat='d', proxy_z=['z'], proxy_w=['w'])
    dash = sp.sensitivity_dashboard(
        r, data=df, verbose=False, dimensions=['first_stage_f'],
    )
    fs_dim = next(d for d in dash.dimensions
                  if 'first-stage F' in d['dimension'])
    assert fs_dim['stable'] is False


def test_sensitivity_dashboard_msm_weight_dimension():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(100):
        V = rng.normal()
        L = rng.normal()
        A_prev = 0.0
        for t in range(3):
            L = L + 0.3 * A_prev + rng.normal(0, 0.3)
            A = float(rng.binomial(1, 1 / (1 + np.exp(-0.4 * L))))
            A_prev = A
            rows.append({'id': i, 'time': t, 'A': A, 'L_lag': L, 'V': V})
    panel = pd.DataFrame(rows)
    panel['Y'] = (0.5 * panel.groupby('id')['A'].cumsum()
                  + 0.3 * panel['V']
                  + rng.normal(0, 0.3, len(panel)))
    r = sp.msm(panel, y='Y', treat='A', id='id', time='time',
               time_varying=['L_lag'], baseline=['V'])
    dash = sp.sensitivity_dashboard(
        r, data=panel, verbose=False,
        dimensions=['trim_sweep'],
    )
    dims = [d['dimension'] for d in dash.dimensions]
    assert any('MSM weight stability' in d for d in dims)


def test_sensitivity_dashboard_default_dimensions_auto_expand_for_sprint_b(proximal_dgp):
    """
    Review fix (round 2): when the caller does NOT pass dimensions=,
    the default list is expanded with the Sprint-B token matching the
    result type so the dashboard actually shows the proximal F / MSM
    weight / monotonicity rows without requiring the caller to know
    about those tokens.
    """
    r = sp.proximal(proximal_dgp, y='y', treat='d',
                    proxy_z=['z'], proxy_w=['w'])
    # NO explicit dimensions= argument — the fix must auto-append.
    dash = sp.sensitivity_dashboard(r, data=proximal_dgp, verbose=False)
    dims = [d['dimension'] for d in dash.dimensions]
    assert any('first-stage F' in d for d in dims), (
        f'Expected the Proximal first-stage F dimension to auto-'
        f'populate; got dims: {dims}'
    )


def test_sensitivity_dashboard_principal_strat_monotonicity_dimension():
    rng = np.random.default_rng(7)
    n = 2000
    X = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n).astype(float)
    # 40 % defiers — violation fraction must exceed 5 % tolerance
    S = np.where(X > 0, 1 - D, D).astype(float)
    Y = 0.5 * D + 0.2 * X + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 's': S, 'x': X})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = sp.principal_strat(
            df, y='y', treat='d', strata='s',
            method='principal_score', covariates=['x'],
            n_boot=20, seed=0,
        )
    dash = sp.sensitivity_dashboard(
        r, data=df, verbose=False, dimensions=['monotonicity'],
    )
    dims = [d for d in dash.dimensions
            if 'monotonicity violation' in d['dimension'].lower()]
    assert len(dims) == 1
    # Violation fraction is high → unstable
    assert dims[0]['stable'] is False


# ---------------------------------------------------------------------
# A) compare_estimators(method_hints={...})
# ---------------------------------------------------------------------

def test_compare_estimators_routes_proximal_via_method_hints(proximal_dgp):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        comp = sp.compare_estimators(
            data=proximal_dgp, y='y', treatment='d',
            methods=['proximal'],
            covariates=[],
            method_hints={'proximal': {
                'proxy_z': ['z'],
                'proxy_w': ['w'],
            }},
        )
    assert 'Proximal' in comp.results
    r = comp.results['Proximal']
    assert abs(r.estimate - 1.5) < 0.3


def test_compare_estimators_proximal_without_hint_is_skipped(proximal_dgp):
    """Missing required hint → UserWarning + method dropped, rest still runs."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        comp = sp.compare_estimators(
            data=proximal_dgp, y='y', treatment='d',
            methods=['ols', 'proximal'],   # proximal w/o hint
            covariates=['z', 'w'],
            # NO method_hints → the proximal branch must raise inside
            # its try/except and be recorded as a warning
        )
    # OLS must still be there
    assert 'OLS' in comp.results
    # Proximal must NOT be there
    assert 'Proximal' not in comp.results
    # Warning mentions proximal
    proximal_warns = [w for w in caught
                      if 'proximal' in str(w.message).lower()]
    assert len(proximal_warns) >= 1


def test_compare_estimators_hint_wins_on_covariate_conflict():
    """
    When shared covariates and hint covariates differ, the hint wins
    AND a UserWarning fires. This test pins BOTH halves of the
    contract — not just the warning (which could fire while the code
    still uses the wrong value). We verify the hint propagated by
    running the same call with TWO different hint covariate sets and
    asserting the estimates diverge.
    """
    rng = np.random.default_rng(0)
    n = 500
    U = rng.normal(0, 1, n)
    D = 0.8 * U + rng.normal(0, 0.5, n)
    Z = 0.9 * U + rng.normal(0, 0.3, n)
    W = 0.9 * U + rng.normal(0, 0.3, n)
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    Y = 1.5 * D + 0.7 * U + 0.3 * X1 + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'w': W,
                       'x1': X1, 'x2': X2})

    # Half 1: conflict-warning check
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        comp_hint_x1 = sp.compare_estimators(
            data=df, y='y', treatment='d',
            methods=['proximal'],
            covariates=['x1', 'x2', 'z', 'w'],
            method_hints={'proximal': {
                'proxy_z': ['z'], 'proxy_w': ['w'],
                'covariates': ['x1'],   # hint subset
            }},
        )
    conflict = [w for w in caught
                if 'overrides the shared' in str(w.message)]
    assert len(conflict) >= 1, (
        f'Expected an override warning, got warnings: '
        f'{[str(w.message)[:80] for w in caught]}'
    )

    # Half 2: hint value actually used — rerun with a DIFFERENT hint
    # and verify the estimate changes. If the hint is discarded and the
    # shared ['x1','x2'] list is used instead, both runs would produce
    # the same estimate.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        comp_hint_x1x2 = sp.compare_estimators(
            data=df, y='y', treatment='d',
            methods=['proximal'],
            covariates=['x1', 'x2', 'z', 'w'],
            method_hints={'proximal': {
                'proxy_z': ['z'], 'proxy_w': ['w'],
                'covariates': ['x1', 'x2'],   # different hint
            }},
        )
    est_x1 = comp_hint_x1.results['Proximal'].estimate
    est_x1x2 = comp_hint_x1x2.results['Proximal'].estimate
    assert abs(est_x1 - est_x1x2) > 1e-6, (
        f'Hint did NOT propagate — estimates identical '
        f'(covariates=[x1]: {est_x1:.6f}, '
        f'covariates=[x1,x2]: {est_x1x2:.6f}). '
        f'This would be silent-fallback behaviour.'
    )


def test_compare_estimators_msm_route():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(100):
        V = rng.normal()
        L = rng.normal()
        A_prev = 0.0
        for t in range(3):
            L = L + 0.3 * A_prev + rng.normal(0, 0.3)
            A = float(rng.binomial(1, 1 / (1 + np.exp(-0.4 * L))))
            A_prev = A
            rows.append({'id': i, 'time': t, 'A': A, 'L_lag': L, 'V': V})
    panel = pd.DataFrame(rows)
    panel['Y'] = (0.5 * panel.groupby('id')['A'].cumsum()
                  + 0.3 * panel['V']
                  + rng.normal(0, 0.3, len(panel)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        comp = sp.compare_estimators(
            data=panel, y='Y', treatment='A',
            methods=['msm'],
            covariates=['V'],
            id='id', time='time',
            method_hints={'msm': {'time_varying': ['L_lag']}},
        )
    assert 'MSM' in comp.results


def test_compare_estimators_principal_strat_route():
    rng = np.random.default_rng(42)
    n = 1500
    D = rng.binomial(1, 0.5, n).astype(float)
    u = rng.uniform(0, 1, n)
    types = np.where(u < 0.6, 'C',
                     np.where(u < 0.8, 'A', 'N'))
    S = np.where(types == 'A', 1.0,
                 np.where(types == 'N', 0.0, D)).astype(float)
    Y = np.where(types == 'C', 2.0 * D + rng.normal(0, 0.3, n),
                 np.where(types == 'A', 5.0 + rng.normal(0, 0.3, n),
                          rng.normal(0, 0.3, n)))
    df = pd.DataFrame({'y': Y, 'd': D, 's': S})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        comp = sp.compare_estimators(
            data=df, y='y', treatment='d',
            methods=['principal_strat'],
            covariates=[],
            method_hints={'principal_strat': {'strata': 's'}},
        )
    assert 'Principal Strat' in comp.results


def test_compare_estimators_mediate_interventional_route():
    rng = np.random.default_rng(0)
    n = 800
    D = rng.binomial(1, 0.5, n).astype(float)
    L = 0.3 * D + rng.normal(0, 0.3, n)
    M = 0.5 * D + 0.3 * L + rng.normal(0, 0.3, n)
    Y = 0.4 * D + 0.8 * M + 0.3 * L + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'm': M, 'l': L})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        comp = sp.compare_estimators(
            data=df, y='y', treatment='d',
            methods=['mediate_interventional'],
            covariates=[],
            method_hints={'mediate_interventional': {
                'mediator': 'm',
                'tv_confounders': ['l'],
            }},
        )
    assert 'Mediation (interventional)' in comp.results
