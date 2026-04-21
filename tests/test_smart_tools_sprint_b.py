"""
Smart-tool integration tests for Sprint-B causal modules.

Covers the second-round coverage fixes:

1. ``sp.pub_ready`` now accepts a single ``CausalResult`` (was only
   list/tuple).
2. ``sp.assumption_audit`` recognises Sprint-B methods and fires the
   method-specific assumption checks (proximal, msm, principal_strat,
   g_computation, front_door, interventional/natural mediation).
3. ``sp.assumption_audit`` does NOT leak panel / OLS / IV checks into
   Sprint-B results via keyword-substring collisions
   (e.g. "fe"/"re"/"iv" inside "Proximal Causal Inference (linear 2SLS)").
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def proximal_result():
    rng = np.random.default_rng(0)
    n = 600
    U = rng.normal(0, 1, n)
    D = 0.8 * U + rng.normal(0, 0.5, n)
    Z = 0.9 * U + rng.normal(0, 0.3, n)
    W = 0.9 * U + rng.normal(0, 0.3, n)
    Y = 1.5 * D + 0.7 * U + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'w': W})
    return sp.proximal(df, y='y', treat='d', proxy_z=['z'], proxy_w=['w'])


@pytest.fixture
def msm_result():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(150):
        V = rng.normal()
        L = rng.normal()
        A_prev = 0.0
        for t in range(3):
            L = L + 0.3 * A_prev + rng.normal(0, 0.3)
            A = float(rng.binomial(1, 1 / (1 + np.exp(-0.4 * L))))
            A_prev = A
            rows.append({'id': i, 'time': t, 'A': A,
                         'L_lag': L, 'V': V})
    panel = pd.DataFrame(rows)
    panel['Y'] = (
        0.5 * panel.groupby('id')['A'].cumsum()
        + panel['V'] * 0.3
        + rng.normal(0, 0.3, len(panel))
    )
    return sp.msm(panel, y='Y', treat='A', id='id', time='time',
                  time_varying=['L_lag'], baseline=['V'])


# ---------------------------------------------------------------------
# sp.pub_ready accepts single result (bug fix)
# ---------------------------------------------------------------------

def test_pub_ready_accepts_single_causal_result(proximal_result):
    """Pre-fix, this raised TypeError: 'CausalResult' not iterable."""
    pub = sp.pub_ready(proximal_result)
    assert pub is not None
    assert 0 <= pub.score <= 100


def test_pub_ready_single_and_list_agree(proximal_result):
    """
    Passing ``r`` vs ``[r]`` must produce the same report. The fix
    auto-wraps a scalar result; this contract test pins that
    equivalence.
    """
    pub_single = sp.pub_ready(proximal_result)
    pub_list = sp.pub_ready([proximal_result])
    assert pub_single.score == pub_list.score
    assert pub_single.present == pub_list.present
    assert pub_single.missing == pub_list.missing


def test_pub_ready_none_still_works():
    """Backward-compat: sp.pub_ready() without results still returns a report."""
    pub = sp.pub_ready()
    assert pub is not None


# ---------------------------------------------------------------------
# sp.assumption_audit: Sprint-B method recognition
# ---------------------------------------------------------------------

def test_assumption_audit_proximal_fires_proxy_checks(proximal_result):
    audit = sp.assumption_audit(proximal_result, verbose=False)
    names = {c.assumption for c in audit.checks}
    # All three proxy-specific assumptions must be present
    assert 'Outcome bridge functional form' in names
    assert 'Proxy order condition (k_z ≥ k_w)' in names
    assert 'Proxy relevance (first-stage F)' in names


def test_assumption_audit_proximal_does_not_leak_panel_or_ols_checks(proximal_result):
    """
    Regression: 'Proximal Causal Inference' contains 're', 'fe', 'iv',
    'linear' as substrings. The pre-fix code used naive substring
    matching and ran panel/OLS/IV checks on proximal results. Word-
    boundary matching + Sprint-B-first dispatch must prevent leakage.
    """
    audit = sp.assumption_audit(proximal_result, verbose=False)
    names = {c.assumption for c in audit.checks}
    # These assumption names are Panel-/OLS-/IV-specific and must NOT
    # appear on a proximal audit. The names are taken verbatim from the
    # corresponding _audit_panel / _audit_linear / _audit_iv functions
    # in src/statspai/smart/assumptions.py — not guesses.
    forbidden = {
        'FE vs RE specification',
        'No serial correlation',
        'Homoskedasticity',            # OLS
        'No multicollinearity (VIF)',  # OLS
        'Instrument relevance',        # IV (correct verbatim name)
        'Instrument exogeneity',       # IV
    }
    leaked = names & forbidden
    assert not leaked, (
        f'Panel/OLS/IV checks leaked into proximal audit: {leaked}'
    )


def test_assumption_audit_msm_fires_weight_checks(msm_result):
    audit = sp.assumption_audit(msm_result, verbose=False)
    names = {c.assumption for c in audit.checks}
    assert 'Stabilized-weight mean centred at 1' in names
    assert 'Positivity (no extreme weights)' in names
    assert 'Sequential exchangeability (identification)' in names


def test_assumption_audit_principal_strat_monotonicity_method():
    rng = np.random.default_rng(42)
    n = 1500
    D = rng.binomial(1, 0.5, n).astype(float)
    u = rng.uniform(0, 1, n)
    types = np.where(u < 0.6, 'C', np.where(u < 0.8, 'A', 'N'))
    S = np.where(types == 'A', 1.0,
                 np.where(types == 'N', 0.0, D)).astype(float)
    Y = rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 's': S})
    r = sp.principal_strat(df, y='y', treat='d', strata='s',
                           method='monotonicity', n_boot=30, seed=0)
    audit = sp.assumption_audit(r, verbose=False)
    names = {c.assumption for c in audit.checks}
    # Under 'monotonicity' method, only the proportions check fires
    # (monotonicity-violation check is principal_score-specific).
    assert 'Stratum proportions sum to 1' in names


def test_assumption_audit_principal_strat_principal_score_fires_mono_and_pi():
    rng = np.random.default_rng(42)
    n = 1500
    X1, X2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    score = 0.8 * X1 + 0.5 * X2 + rng.normal(0, 0.5, n)
    lo, hi = np.quantile(score, [0.2, 0.8])
    types = np.where(score < lo, 'N',
                     np.where(score < hi, 'C', 'A'))
    D = rng.binomial(1, 0.5, n).astype(float)
    S = np.where(types == 'A', 1.0,
                 np.where(types == 'N', 0.0, D)).astype(float)
    Y = rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 's': S, 'x1': X1, 'x2': X2})
    r = sp.principal_strat(df, y='y', treat='d', strata='s',
                           method='principal_score',
                           covariates=['x1', 'x2'], n_boot=30, seed=0)
    audit = sp.assumption_audit(r, verbose=False)
    names = {c.assumption for c in audit.checks}
    assert 'Monotonicity S(1) ≥ S(0)' in names
    assert 'Stratum proportions sum to 1' in names
    assert 'Principal ignorability (Y(d) ⊥ stratum | X, D=d)' in names


def test_assumption_audit_g_computation_flags_outcome_model():
    rng = np.random.default_rng(0)
    n = 500
    X = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n).astype(float)
    Y = 1.0 * D + X + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'x': X})
    r = sp.g_computation(df, y='y', treat='d', covariates=['x'],
                         n_boot=30, seed=0)
    audit = sp.assumption_audit(r, verbose=False)
    names = {c.assumption for c in audit.checks}
    assert 'Outcome model correctly specified' in names
    assert 'Bootstrap inference well-behaved' in names


def test_assumption_audit_front_door_design_reminders():
    rng = np.random.default_rng(42)
    n = 1000
    U = rng.normal(0, 1, n)
    D = rng.binomial(1, 1 / (1 + np.exp(-U)), n).astype(float)
    M = 0.8 * D + rng.normal(0, 0.3, n)
    Y = 1.5 * M + 0.7 * U + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'm': M})
    r = sp.front_door(df, y='y', treat='d', mediator='m',
                      mediator_type='continuous',
                      n_boot=30, n_mc=50, seed=0)
    audit = sp.assumption_audit(r, verbose=False)
    names = {c.assumption for c in audit.checks}
    assert 'No direct D→Y path (full mediation)' in names
    assert 'No unmeasured M-Y confounder' in names


def test_assumption_audit_mediation_interventional_decomp_identity():
    rng = np.random.default_rng(0)
    n = 800
    D = rng.binomial(1, 0.5, n).astype(float)
    L = 0.3 * D + rng.normal(0, 0.3, n)
    M = 0.5 * D + 0.3 * L + rng.normal(0, 0.3, n)
    Y = 0.4 * D + 0.8 * M + 0.3 * L + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'm': M, 'l': L})
    r = sp.mediate_interventional(
        df, y='y', treat='d', mediator='m', tv_confounders=['l'],
        n_boot=30, n_mc=40, seed=0,
    )
    audit = sp.assumption_audit(r, verbose=False)
    names = {c.assumption for c in audit.checks}
    assert 'Decomposition identity IIE + IDE = Total' in names
    assert 'Treatment-induced confounders listed' in names


def test_assumption_audit_mediation_natural_cross_world_reminder():
    rng = np.random.default_rng(0)
    n = 500
    T = rng.binomial(1, 0.5, n).astype(float)
    M = 0.5 * T + rng.normal(0, 0.3, n)
    Y = 0.4 * T + 0.8 * M + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 't': T, 'm': M})
    r = sp.mediate(df, y='y', treat='t', mediator='m',
                   n_boot=30, seed=0)
    audit = sp.assumption_audit(r, verbose=False)
    names = {c.assumption for c in audit.checks}
    assert 'Cross-world independence (natural effects)' in names


# ---------------------------------------------------------------------
# Review fixes (round 2) — pin the 3 MAJOR + 3 MINOR issues the
# code-review agent raised so they can't silently regress.
# ---------------------------------------------------------------------

def test_audit_mediation_interventional_no_vacuous_pass_when_keys_missing():
    """
    Round-2 review fix: if the result's ``model_info`` is missing any of
    (iie, ide, total_effect), the decomposition-identity check must
    emit an explicit ``passed=None`` rather than silently skip and
    grade the audit ``A`` with zero checks.
    """
    from statspai.core.results import CausalResult
    # Construct a fake interventional result with NO iie/ide/total_effect keys
    fake = CausalResult(
        method='Interventional Mediation Analysis (synthetic)',
        estimand='IIE',
        estimate=0.5, se=0.1, pvalue=0.0, ci=(0.3, 0.7),
        alpha=0.05, n_obs=500,
        model_info={},   # ← deliberately empty
    )
    audit = sp.assumption_audit(fake, verbose=False)
    names = {c.assumption for c in audit.checks}
    assert 'Decomposition identity IIE + IDE = Total' in names
    # And the check must be explicit about what went wrong
    check = next(c for c in audit.checks
                 if c.assumption == 'Decomposition identity IIE + IDE = Total')
    assert check.passed is None
    assert 'keys not found' in check.test_name.lower()


def test_audit_dispatch_no_double_firing_on_mixed_mediation_labels():
    """
    Round-2 review fix: a method string containing BOTH
    "interventional mediation" AND "causal mediation" must fire the
    interventional branch only, not both.
    """
    from statspai.core.results import CausalResult
    fake = CausalResult(
        # Carry BOTH keywords so the dispatcher could double-fire
        # without the sprint_b_matched guard.
        method='Interventional Mediation (causal mediation variant)',
        estimand='IIE',
        estimate=0.5, se=0.1, pvalue=0.0, ci=(0.3, 0.7),
        alpha=0.05, n_obs=500,
        model_info={'iie': 0.3, 'ide': 0.2, 'total_effect': 0.5},
    )
    audit = sp.assumption_audit(fake, verbose=False)
    names = [c.assumption for c in audit.checks]
    # Interventional-specific check must fire
    assert 'Decomposition identity IIE + IDE = Total' in names
    # Natural-specific check must NOT fire even though "causal mediation"
    # is present — the sprint_b_matched guard prevents double-dispatch.
    assert 'Cross-world independence (natural effects)' not in names


def test_audit_mediation_natural_bootstrap_failure_rate():
    """
    Round-2 review fix: natural-mediation bootstrap health check now
    tracks failure rate rather than reporting an unconditional pass.
    """
    from statspai.core.results import CausalResult
    # Fake result with 50/100 bootstrap replications failed
    fake = CausalResult(
        method='Causal Mediation Analysis',
        estimand='ACME',
        estimate=0.3, se=0.1, pvalue=0.0, ci=(0.1, 0.5),
        alpha=0.05, n_obs=500,
        model_info={'n_boot': 100, 'n_boot_failed': 50},
    )
    audit = sp.assumption_audit(fake, verbose=False)
    boot = next((c for c in audit.checks
                 if c.assumption == 'Bootstrap inference well-behaved'),
                None)
    assert boot is not None
    assert boot.passed is False, (
        'high bootstrap failure rate (50%) must fail the health check'
    )


def test_pub_ready_handles_numpy_array_of_results(proximal_result):
    """
    Round-2 review fix: passing a numpy array of results iterates the
    array elements, not wraps the array as a single-element list.
    """
    arr = np.array([proximal_result, proximal_result], dtype=object)
    pub = sp.pub_ready(arr)
    # With 2 results, main_results should be marked present — same as
    # passing a plain list.
    pub_list = sp.pub_ready([proximal_result, proximal_result])
    assert pub.score == pub_list.score
