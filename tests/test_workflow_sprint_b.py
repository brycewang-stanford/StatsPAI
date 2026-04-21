"""
Contract tests for the Sprint-B ecosystem integration:

1. sp.recommend() now surfaces msm / proximal / principal_strat /
   mediate_interventional / front_door / g_computation when the
   matching hint kwargs are supplied.
2. sp.causal() workflow accepts the new hints and routes to the
   right Sprint-B estimator.
3. sp.diagnose_result() dispatches to proximal / msm /
   principal_strat / g_computation / front_door /
   mediation_interventional batteries.
4. docs/index.md no longer links to the dead guides/did.md page.
5. sp.modelsummary accepts CausalResult from new modules.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def proximal_dgp():
    rng = np.random.default_rng(42)
    n = 1500
    U = rng.normal(0, 1, n)
    D = 0.8 * U + rng.normal(0, 0.5, n)
    Z = 0.9 * U + rng.normal(0, 0.3, n)
    W = 0.9 * U + rng.normal(0, 0.3, n)
    Y = 1.5 * D + 0.7 * U + rng.normal(0, 0.3, n)
    return pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'w': W})


@pytest.fixture
def msm_panel():
    rng = np.random.default_rng(7)
    rows = []
    for i in range(150):
        V = rng.normal()
        L = rng.normal()
        A_prev = 0.0
        for t in range(3):
            L = L + 0.3 * A_prev + rng.normal(0, 0.3)
            A = float(rng.binomial(1, 1 / (1 + np.exp(-0.4 * L))))
            A_prev = A
            rows.append({'id': i, 'time': t, 'A': A, 'L_lag': L, 'V': V})
    panel = pd.DataFrame(rows)
    panel['Y'] = (
        0.5 * panel.groupby('id')['A'].cumsum()
        + panel['V'] * 0.3
        + rng.normal(0, 0.3, len(panel))
    )
    return panel


@pytest.fixture
def principal_strat_dgp():
    rng = np.random.default_rng(42)
    n = 2000
    D = rng.binomial(1, 0.5, n).astype(float)
    u = rng.uniform(0, 1, n)
    types = np.where(u < 0.6, 'C', np.where(u < 0.8, 'A', 'N'))
    S = np.where(types == 'A', 1.0, np.where(types == 'N', 0.0, D)).astype(float)
    Y = np.where(
        types == 'C', 2.0 * D + rng.normal(0, 0.3, n),
        np.where(types == 'A', 5.0 + rng.normal(0, 0.3, n),
                 rng.normal(0, 0.3, n)),
    )
    return pd.DataFrame({'y': Y, 'd': D, 's': S})


# ---------------------------------------------------------------------
# 1. sp.recommend() now includes Sprint-B candidates
# ---------------------------------------------------------------------

def test_recommend_surfaces_proximal_when_proxies_supplied(proximal_dgp):
    rec = sp.recommend(
        data=proximal_dgp, y='y', treatment='d',
        proxy_z=['z'], proxy_w=['w'],
    )
    methods = [r['function'] for r in rec.recommendations]
    assert 'proximal' in methods


def test_recommend_surfaces_msm_for_panel_tv_confounders(msm_panel):
    rec = sp.recommend(
        data=msm_panel, y='Y', treatment='A',
        id='id', time='time',
        tv_confounders=['L_lag'],
    )
    methods = [r['function'] for r in rec.recommendations]
    assert 'msm' in methods


def test_recommend_surfaces_principal_strat(principal_strat_dgp):
    rec = sp.recommend(
        data=principal_strat_dgp, y='y', treatment='d',
        post_treat_strata='s',
    )
    methods = [r['function'] for r in rec.recommendations]
    assert 'principal_strat' in methods


def test_recommend_surfaces_mediation_family(proximal_dgp):
    # Re-purpose the DGP — pretend W is the mediator
    rec = sp.recommend(
        data=proximal_dgp, y='y', treatment='d',
        mediator='w',
    )
    methods = [r['function'] for r in rec.recommendations]
    # Both natural and front-door should be offered
    assert 'mediate' in methods
    assert 'front_door' in methods


def test_recommend_adds_g_computation_for_observational(proximal_dgp):
    rec = sp.recommend(
        data=proximal_dgp, y='y', treatment='d',
        design='observational',
    )
    methods = [r['function'] for r in rec.recommendations]
    assert 'g_computation' in methods


# ---------------------------------------------------------------------
# 2. sp.causal() workflow routes Sprint-B hints correctly
# ---------------------------------------------------------------------

def test_causal_workflow_routes_to_proximal(proximal_dgp):
    w = sp.causal(
        data=proximal_dgp, y='y', treatment='d',
        proxy_z=['z'], proxy_w=['w'],
        design='observational',
    )
    assert 'Proximal' in (w.result.method or '')
    assert abs(w.result.estimate - 1.5) < 0.3


def test_causal_workflow_routes_to_msm(msm_panel):
    w = sp.causal(
        data=msm_panel, y='Y', treatment='A',
        id='id', time='time',
        tv_confounders=['L_lag'],
        design='panel',
    )
    assert 'Marginal Structural' in (w.result.method or '')


def test_causal_workflow_routes_to_principal_strat(principal_strat_dgp):
    w = sp.causal(
        data=principal_strat_dgp, y='y', treatment='d',
        post_treat_strata='s',
        design='observational',
    )
    # PrincipalStratResult is a dataclass, not CausalResult — the
    # workflow stores it on .result regardless.
    from statspai.principal_strat import PrincipalStratResult
    assert isinstance(w.result, PrincipalStratResult)


def test_causal_workflow_hint_fields_stored(proximal_dgp):
    w = sp.causal(
        data=proximal_dgp, y='y', treatment='d',
        proxy_z=['z'], proxy_w=['w'],
        design='observational',
        auto_run=False,
    )
    assert w.proxy_z == ['z']
    assert w.proxy_w == ['w']
    assert w.mediator is None
    assert w.tv_confounders is None


# ---------------------------------------------------------------------
# 3. sp.diagnose_result() dispatches to Sprint-B batteries
# ---------------------------------------------------------------------

def test_diagnose_result_detects_proximal(proximal_dgp):
    r = sp.proximal(
        proximal_dgp, y='y', treat='d',
        proxy_z=['z'], proxy_w=['w'],
    )
    diag = sp.diagnose_result(r, print_results=False)
    assert diag['method_type'] == 'proximal'
    check_names = [c.get('test', '') for c in diag['checks']]
    assert any('Bridge' in n for n in check_names)
    assert any('Proximal ATE' in n for n in check_names)


def test_diagnose_result_detects_msm(msm_panel):
    r = sp.msm(
        msm_panel, y='Y', treat='A',
        id='id', time='time',
        time_varying=['L_lag'], baseline=['V'],
    )
    diag = sp.diagnose_result(r, print_results=False)
    assert diag['method_type'] == 'msm'
    check_names = [c.get('test', '') for c in diag['checks']]
    assert any('Stabilized-weight mean' in n for n in check_names)


def test_diagnose_result_detects_principal_strat(principal_strat_dgp):
    r = sp.principal_strat(
        principal_strat_dgp, y='y', treat='d', strata='s',
        method='monotonicity', n_boot=50, seed=0,
    )
    diag = sp.diagnose_result(r, print_results=False)
    assert diag['method_type'] == 'principal_strat'
    check_names = [c.get('test', '') for c in diag['checks']]
    assert any('Zhang-Rubin SACE' in n for n in check_names)


def test_diagnose_result_detects_g_computation():
    rng = np.random.default_rng(0)
    n = 500
    X = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n).astype(float)
    Y = 1.0 * D + X + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'x': X})
    r = sp.g_computation(df, y='y', treat='d', covariates=['x'],
                         n_boot=30, seed=0)
    diag = sp.diagnose_result(r, print_results=False)
    assert diag['method_type'] == 'g_computation'
    check_names = [c.get('test', '') for c in diag['checks']]
    assert any('Estimand' in n for n in check_names)
    assert any('Bootstrap health' in n for n in check_names)


def test_diagnose_result_detects_front_door(proximal_dgp):
    df = proximal_dgp.copy()
    df['d'] = (df['d'] > df['d'].median()).astype(float)   # binarise
    r = sp.front_door(df, y='y', treat='d', mediator='w',
                      mediator_type='continuous',
                      n_boot=30, n_mc=40, seed=0)
    diag = sp.diagnose_result(r, print_results=False)
    assert diag['method_type'] == 'front_door'


def test_diagnose_result_detects_mediation_interventional():
    rng = np.random.default_rng(0)
    n = 500
    D = rng.binomial(1, 0.5, n).astype(float)
    L = 0.3 * D + rng.normal(0, 0.3, n)
    M = 0.5 * D + 0.3 * L + rng.normal(0, 0.3, n)
    Y = 0.4 * D + 0.8 * M + 0.3 * L + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'm': M, 'l': L})
    r = sp.mediate_interventional(
        df, y='y', treat='d', mediator='m', tv_confounders=['l'],
        n_boot=40, n_mc=60, seed=0,
    )
    diag = sp.diagnose_result(r, print_results=False)
    assert diag['method_type'] == 'mediation_interventional'
    check_names = [c.get('test', '') for c in diag['checks']]
    assert any('Decomposition additivity' in n for n in check_names)


# ---------------------------------------------------------------------
# 4. docs/index.md no longer links to the dead guides/did.md
# ---------------------------------------------------------------------

def test_index_md_has_no_stale_did_guide_link():
    root = Path(__file__).resolve().parent.parent
    idx = (root / 'docs' / 'index.md').read_text()
    assert 'guides/did.md' not in idx, (
        'docs/index.md still contains a link to the non-existent '
        'guides/did.md; update it to a real guide path '
        '(e.g. guides/callaway_santanna.md).'
    )


# ---------------------------------------------------------------------
# 5. sp.modelsummary works with Sprint-B CausalResult objects
# ---------------------------------------------------------------------

def test_modelsummary_works_on_proximal_result(proximal_dgp):
    r = sp.proximal(proximal_dgp, y='y', treat='d',
                    proxy_z=['z'], proxy_w=['w'])
    out = sp.modelsummary(r, output='markdown')
    assert isinstance(out, str)
    # Point estimate should appear in the table to at least 2 dps
    assert f'{r.estimate:.2f}'[:3] in out or f'{r.estimate:.3f}'[:4] in out
