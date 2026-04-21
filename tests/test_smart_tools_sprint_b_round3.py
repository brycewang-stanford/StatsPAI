"""
Round-3 Smart-tool coverage for Sprint-B: compare_estimators registers
g_computation, sensitivity_dashboard reports the correct method label
for Sprint-B results, and ROADMAP §6 documents the compare_estimators
gap for the remaining hint-driven estimators.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------
# compare_estimators — g_computation is now a first-class option
# ---------------------------------------------------------------------

@pytest.fixture
def binary_dgp():
    rng = np.random.default_rng(0)
    n = 500
    X = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n).astype(float)
    Y = 1.0 * D + 0.5 * X + rng.normal(0, 0.3, n)
    return pd.DataFrame({'y': Y, 'd': D, 'x': X})


def test_compare_estimators_accepts_g_computation(binary_dgp):
    comp = sp.compare_estimators(
        data=binary_dgp, y='y', treatment='d',
        methods=['ols', 'g_computation'],
        covariates=['x'],
    )
    # Both methods ran (agreement diagnostics populated)
    assert hasattr(comp, 'summary')
    txt = comp.summary()
    assert 'G-computation' in txt
    assert 'OLS' in txt
    # Sign agreement 100 % on a linear, binary-D, positive-effect DGP.
    assert 'Sign agreement:        100%' in txt


def test_compare_estimators_g_computation_recovers_truth(binary_dgp):
    """Both OLS and g_computation should hug the true effect ≈ 1.0."""
    comp = sp.compare_estimators(
        data=binary_dgp, y='y', treatment='d',
        methods=['g_computation'],
        covariates=['x'],
    )
    r = comp.results['G-computation']
    assert abs(r.estimate - 1.0) < 0.15


def test_compare_estimators_unsupported_method_does_not_crash(binary_dgp):
    """
    Sprint-B methods with extra required args (proximal, msm, ...) are
    documented as NOT supported. Requesting one must not silently
    drop the supported methods too.

    ``compare_estimators`` wraps each method in its own try/except +
    ``warnings.warn`` (see src/statspai/smart/compare.py), so the call
    itself MUST NOT raise to the caller. The contract we're pinning:
    the supported method (OLS) still appears in ``.results`` even
    when an unsupported method ('proximal') sits beside it in the
    ``methods=`` list.
    """
    import warnings
    with warnings.catch_warnings():
        # compare_estimators emits UserWarning for the unsupported
        # branch; suppress it so the test output is clean while still
        # exercising the real path.
        warnings.simplefilter('ignore')
        comp = sp.compare_estimators(
            data=binary_dgp, y='y', treatment='d',
            methods=['ols', 'proximal'],   # 'proximal' unsupported
            covariates=['x'],
        )
    # OLS must still be present regardless of what the unsupported
    # method did. No permissive try/except — a true regression
    # (OLS silently disappearing) should fail this assertion.
    assert 'OLS' in comp.results
    # 'proximal' must NOT appear (never dispatched)
    assert 'Proximal' not in comp.results


# ---------------------------------------------------------------------
# sensitivity_dashboard — method label now reads from .method first
# ---------------------------------------------------------------------

def test_sensitivity_dashboard_reports_proximal_method_label():
    rng = np.random.default_rng(0)
    n = 500
    U = rng.normal(0, 1, n)
    D = 0.8 * U + rng.normal(0, 0.5, n)
    Z = 0.9 * U + rng.normal(0, 0.3, n)
    W = 0.9 * U + rng.normal(0, 0.3, n)
    Y = 1.5 * D + 0.7 * U + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'w': W})
    r = sp.proximal(df, y='y', treat='d', proxy_z=['z'], proxy_w=['w'])
    dash = sp.sensitivity_dashboard(r, data=df, verbose=False,
                                    dimensions=['sample'])
    assert 'Proximal' in dash.method, (
        f"Expected 'Proximal' in dash.method, got {dash.method!r}"
    )


def test_sensitivity_dashboard_reports_msm_method_label():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(100):
        for t in range(3):
            rows.append({
                'id': i, 'time': t,
                'A': rng.binomial(1, 0.5),
                'L': rng.normal(),
                'V': rng.normal(),
            })
    panel = pd.DataFrame(rows)
    panel['Y'] = (0.5 * panel.groupby('id')['A'].cumsum()
                  + 0.3 * panel['V']
                  + rng.normal(0, 0.3, len(panel)))
    r = sp.msm(panel, y='Y', treat='A', id='id', time='time',
               time_varying=['L'], baseline=['V'])
    dash = sp.sensitivity_dashboard(r, data=panel, verbose=False,
                                    dimensions=['sample'])
    assert 'Marginal Structural' in dash.method, (
        f"Expected 'Marginal Structural' in label, got {dash.method!r}"
    )


def test_sensitivity_dashboard_legacy_econometric_result_label():
    """Backward-compat: EconometricResults path still reads model_type."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        'y': rng.normal(0, 1, n),
        'x': rng.normal(0, 1, n),
    })
    r = sp.regress('y ~ x', data=df)
    dash = sp.sensitivity_dashboard(r, data=df, verbose=False,
                                    dimensions=['sample'])
    # Not Unknown — the model_info['model_type'] fallback still works
    assert dash.method != 'Unknown'


# ---------------------------------------------------------------------
# ROADMAP §6 documents the compare_estimators Sprint-B gap
# ---------------------------------------------------------------------

def test_compare_estimators_g_computation_binary_guard_compact():
    """
    Review-fix: the g_computation branch emits a CLEAN warning on
    non-binary D (≤ 500 chars) rather than dumping the full list of
    unique D values (which for n=400 continuous D was ~20 KB).
    """
    import warnings
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        'y': rng.normal(0, 1, n),
        'd': rng.normal(0, 1, n),  # continuous — not binary
        'x': rng.normal(0, 1, n),
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        comp = sp.compare_estimators(
            data=df, y='y', treatment='d',
            methods=['g_computation'],
            covariates=['x'],
        )
    # g_computation branch should have emitted a compact warning.
    g_warns = [w for w in caught if 'g_computation' in str(w.message)]
    assert len(g_warns) >= 1, "expected a g_computation warning"
    msg = str(g_warns[0].message)
    # Compact: the warning fits on a few lines, not a 20 KB value dump
    assert len(msg) < 500, (
        f"g_computation warning is {len(msg)} chars; expected <500. "
        f"First 200 chars: {msg[:200]}"
    )
    # Useful: it actually mentions the issue and points to the fix
    assert 'binary' in msg.lower()


def test_roadmap_documents_compare_estimators_sprint_b_gap():
    root = Path(__file__).resolve().parent.parent
    roadmap = (root / 'docs' / 'ROADMAP.md').read_text()
    # Section 6 heading present
    assert 'compare_estimators' in roadmap
    # The four hint-driven methods are called out
    for name in ('proximal', 'msm', 'principal_strat', 'mediator'):
        assert name in roadmap.lower()
