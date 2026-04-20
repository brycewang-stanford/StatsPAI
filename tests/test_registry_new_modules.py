"""
Registry + integration tests for Sprint-B modules.

Pins behaviours discovered in the post-sprint audit:
* All new public functions are listed by sp.list_functions().
* sp.describe_function(name) returns a populated spec for each.
* sp.search_functions() surfaces each by its primary tag.
* The dml registry entry reflects the IIVM addition (model enum).
* CausalResult.tidy() / .glance() work on new-module outputs.
* Class wrappers (MarginalStructuralModel, ProximalCausalInference) fit.
* Integer-typed treatment columns don't break the estimators.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult


# ---------------------------------------------------------------------
# Registry coverage
# ---------------------------------------------------------------------

NEW_FUNCTIONS = (
    'g_computation',
    'front_door',
    'msm',
    'mediate_interventional',
    'proximal',
    'principal_strat',
)


@pytest.mark.parametrize('fn_name', NEW_FUNCTIONS)
def test_registry_lists_new_function(fn_name):
    assert fn_name in sp.list_functions(), (
        f'{fn_name} missing from sp.list_functions()'
    )


@pytest.mark.parametrize('fn_name', NEW_FUNCTIONS)
def test_registry_describe_function_populated(fn_name):
    spec = sp.describe_function(fn_name)
    assert spec['name'] == fn_name
    assert spec['category'] == 'causal'
    assert len(spec['description']) > 50
    assert len(spec['params']) >= 3
    # Every param must have a non-empty description (we pay for LLM discoverability here)
    for p in spec['params']:
        assert p['description'] or p['name'] == 'data', (
            f'{fn_name} param {p["name"]!r} missing description'
        )


def test_dml_registry_enum_includes_iivm():
    """After the sprint the dml model= enum must list iivm."""
    spec = sp.describe_function('dml')
    model_param = next(p for p in spec['params'] if p['name'] == 'model')
    assert 'iivm' in model_param['enum']
    assert {'plr', 'irm', 'pliv', 'iivm'}.issubset(set(model_param['enum']))


def test_registry_search_picks_up_new_modules():
    """Tag-based search should locate each new module by a salient keyword."""
    assert 'proximal' in [f['name'] for f in sp.search_functions('bridge')]
    assert 'msm' in [f['name'] for f in sp.search_functions('iptw')]
    assert 'front_door' in [f['name'] for f in sp.search_functions('pearl')]
    assert 'principal_strat' in [f['name'] for f in sp.search_functions('sace')]


# ---------------------------------------------------------------------
# CausalResult integration (tidy / glance)
# ---------------------------------------------------------------------

@pytest.fixture
def simple_binary_data():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n).astype(float)
    Y = 1.0 * D + 0.5 * X + rng.normal(0, 0.3, n)
    return pd.DataFrame({'y': Y, 'd': D, 'x': X})


def test_g_computation_tidy_glance(simple_binary_data):
    r = sp.g_computation(simple_binary_data, y='y', treat='d',
                         covariates=['x'], n_boot=20, seed=0)
    tidy = r.tidy()
    glance = r.glance()
    # Consistent broom-style columns
    for col in ['term', 'estimate', 'std_error', 'statistic',
                'p_value', 'conf_low', 'conf_high']:
        assert col in tidy.columns
    assert glance.shape[0] == 1


def test_proximal_tidy(simple_binary_data):
    rng = np.random.default_rng(7)
    n = 1000
    U = rng.normal(0, 1, n)
    D = 0.8 * U + rng.normal(0, 0.5, n)
    Z = 0.9 * U + rng.normal(0, 0.3, n)
    W = 0.9 * U + rng.normal(0, 0.3, n)
    Y = 1.5 * D + 0.7 * U + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'w': W})
    r = sp.proximal(df, y='y', treat='d', proxy_z=['z'], proxy_w=['w'])
    tidy = r.tidy()
    assert tidy.shape[0] >= 1
    assert not np.isnan(tidy.iloc[0]['estimate'])


# ---------------------------------------------------------------------
# Custom learner for g_computation
# ---------------------------------------------------------------------

def test_g_computation_custom_sklearn_learner(simple_binary_data):
    from sklearn.ensemble import GradientBoostingRegressor
    r = sp.g_computation(
        simple_binary_data, y='y', treat='d', covariates=['x'],
        ml_Q=GradientBoostingRegressor(n_estimators=30, random_state=0),
        n_boot=20, seed=0,
    )
    assert r.model_info['ml_Q'] == 'GradientBoostingRegressor'
    assert abs(r.estimate - 1.0) < 0.3


# ---------------------------------------------------------------------
# Class wrappers
# ---------------------------------------------------------------------

def test_marginal_structural_model_class():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
        for t in range(3):
            rows.append({
                'id': i, 'time': t,
                'A': rng.binomial(1, 0.4),
                'L': rng.normal(), 'V': rng.normal(),
            })
    panel = pd.DataFrame(rows)
    panel['Y'] = (
        0.5 * panel.groupby('id')['A'].cumsum()
        + panel['V'] * 0.3
        + rng.normal(0, 0.3, len(panel))
    )
    obj = sp.MarginalStructuralModel(
        y='Y', treat='A', id='id', time='time',
        time_varying=['L'], baseline=['V'],
    ).fit(panel)
    assert obj.result_ is not None
    assert np.isfinite(obj.result_.estimate)


def test_proximal_causal_inference_class():
    rng = np.random.default_rng(7)
    n = 1000
    U = rng.normal(0, 1, n)
    D = 0.8 * U + rng.normal(0, 0.5, n)
    Z = 0.9 * U + rng.normal(0, 0.3, n)
    W = 0.9 * U + rng.normal(0, 0.3, n)
    Y = 1.5 * D + 0.7 * U + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'w': W})
    obj = sp.ProximalCausalInference(
        y='y', treat='d', proxy_z=['z'], proxy_w=['w'],
    ).fit(df)
    assert obj.result_ is not None
    assert abs(obj.result_.estimate - 1.5) < 0.3


# ---------------------------------------------------------------------
# Integer-type treatment column
# ---------------------------------------------------------------------

def test_g_computation_accepts_integer_treatment(simple_binary_data):
    df = simple_binary_data.copy()
    df['d'] = df['d'].astype(int)
    r = sp.g_computation(df, y='y', treat='d', covariates=['x'],
                         n_boot=10, seed=0)
    assert np.isfinite(r.estimate)


def test_iivm_accepts_integer_treatment_and_instrument():
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.normal(0, 1, n)
    Z = rng.binomial(1, 0.5, n)            # int
    u = rng.uniform(0, 1, n)
    D = np.where(u < 0.7, Z, 1)            # int
    Y = 1.5 * D + 0.5 * X + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D.astype(int), 'z': Z.astype(int), 'x': X})
    r = sp.dml(df, y='y', treat='d', covariates=['x'],
               model='iivm', instrument='z')
    assert np.isfinite(r.estimate)
    assert abs(r.estimate - 1.5) < 0.3
