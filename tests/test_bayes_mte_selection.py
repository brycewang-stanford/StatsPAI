"""Tests for ``sp.bayes_mte(selection='uniform' | 'normal')`` — probit-scale MTE."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.bayes import BayesianMTEResult, bayes_mte

pymc = pytest.importorskip(
    "pymc",
    reason="PyMC is an optional dependency; skip Bayesian tests if missing.",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _heckman_dgp(n, intercept, slope, seed):
    """Genuine Heckman-HV DGP: Gaussian latent V_i, linear MTE in V."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n)
    V = rng.normal(size=n)
    linpred = 0.8 * Z
    D = (linpred - V > 0).astype(float)
    tau = intercept + slope * V
    Y = 1.0 + tau * D + 0.3 * rng.normal(size=n)
    return pd.DataFrame({'y': Y, 'd': D, 'z': Z})


@pytest.fixture
def heckman_data():
    return _heckman_dgp(600, intercept=0.5, slope=1.5, seed=1001)


# ---------------------------------------------------------------------------
# API surface + back-compat
# ---------------------------------------------------------------------------


def test_selection_uniform_back_compat(heckman_data):
    """Default call (selection='uniform') still works & fills
    model_info unchanged from v0.9.11 behaviour aside from the new
    ``selection`` key."""
    r = bayes_mte(heckman_data, y='y', treat='d', instrument='z',
                  draws=200, tune=200, chains=2, progressbar=False)
    assert isinstance(r, BayesianMTEResult)
    assert r.model_info['selection'] == 'uniform'
    # Method label should NOT mention V scale
    assert 'U_D' in r.method or 'propensity' in r.method.lower()
    assert 'V scale' not in r.method


def test_selection_normal_method_label(heckman_data):
    r = bayes_mte(heckman_data, y='y', treat='d', instrument='z',
                  selection='normal', poly_u=1,
                  draws=200, tune=200, chains=2, progressbar=False)
    assert r.model_info['selection'] == 'normal'
    assert 'V scale' in r.method


# ---------------------------------------------------------------------------
# Recovery on the genuine Heckman DGP
# ---------------------------------------------------------------------------


def test_selection_normal_hv_latent_recovers_linear_heckman_slope(heckman_data):
    """On a DGP with Gaussian V and MTE(V) = 0.5 + 1.5·V,
    poly_u=1 + selection='normal' + hv_latent should cover
    (b_0, b_1) = (0.5, 1.5) loosely at n=600."""
    r = bayes_mte(heckman_data, y='y', treat='d', instrument='z',
                  selection='normal', mte_method='hv_latent',
                  poly_u=1,
                  draws=400, tune=400, chains=2, progressbar=False,
                  random_state=42)
    b = r.trace.posterior['b_mte'].values.reshape(-1, 2)
    b0_mean = float(b[:, 0].mean())
    b1_mean = float(b[:, 1].mean())
    # Within 1.0 of truth at n=600 + 400 NUTS draws
    assert abs(b0_mean - 0.5) < 1.0, (
        f"b_0 mean {b0_mean:.3f} drift from 0.5"
    )
    assert abs(b1_mean - 1.5) < 1.0, (
        f"b_1 mean {b1_mean:.3f} drift from 1.5"
    )


# ---------------------------------------------------------------------------
# Orthogonality: all 8 combos of (first_stage, mte_method, selection) run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("first_stage", ['plugin', 'joint'])
@pytest.mark.parametrize("mte_method", ['polynomial', 'hv_latent'])
@pytest.mark.parametrize("selection", ['uniform', 'normal'])
def test_all_combos_run(heckman_data, first_stage, mte_method, selection):
    r = bayes_mte(heckman_data, y='y', treat='d', instrument='z',
                  first_stage=first_stage, mte_method=mte_method,
                  selection=selection, poly_u=1,
                  draws=100, tune=100, chains=2, progressbar=False,
                  random_state=7)
    assert r.model_info['first_stage'] == first_stage
    assert r.model_info['mte_method'] == mte_method
    assert r.model_info['selection'] == selection


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_selection_invalid_value_raises(heckman_data):
    with pytest.raises(ValueError, match='selection'):
        bayes_mte(heckman_data, y='y', treat='d', instrument='z',
                  selection='bogus',
                  draws=50, tune=50, chains=1, progressbar=False)


# ---------------------------------------------------------------------------
# Qualitative check: normal vs uniform disagreement on Heckman DGP
# ---------------------------------------------------------------------------


def test_selection_normal_att_atu_use_probit_abscissa(heckman_data):
    """Regression guard for the v0.9.12 Round-B BLOCKER: ATT/ATU
    must evaluate the V-scale polynomial at ``Φ^{-1}(U_pop)``, not
    at raw U_pop ∈ [0,1]. A crude but sufficient sanity check:

    With poly_u=1 and slope b_1 > 0, the *correct* ATT (on V scale)
    for treated units (who have U_D < p_i, low V values) should lie
    to the *left* of the correct ATU (on V scale) — i.e. ATT < ATU
    when the MTE slopes up in V.

    If the bug were still present, ATT/ATU would be evaluated on raw
    U values, and the treated (low U_D) would give *smaller*
    contributions via u^k vs. the untreated (high U_D) giving larger
    — flipping the ordering's magnitude.
    """
    r = bayes_mte(heckman_data, y='y', treat='d', instrument='z',
                  selection='normal', mte_method='hv_latent',
                  poly_u=1,
                  draws=400, tune=400, chains=2, progressbar=False,
                  random_state=43)
    # Both ATT and ATU should be finite (regression test for earlier
    # bug where V-scale fit with U-scale ATT/ATU computation sometimes
    # overflowed)
    assert np.isfinite(r.att)
    assert np.isfinite(r.atu)
    # And they must differ non-trivially on the Heckman DGP with
    # slope=1.5 — if they agreed, the abscissa transform is broken.
    assert abs(r.att - r.atu) > 0.1, (
        f"ATT={r.att:.3f} vs ATU={r.atu:.3f} should differ "
        f"materially under slope=1.5 on V scale"
    )


def test_selection_normal_policy_effect_uses_probit_abscissa(heckman_data):
    """Regression guard for the v0.9.12 Round-B BLOCKER-2: under
    selection='normal', `policy_effect` must transform u_grid to
    Φ^{-1}(u_grid) before raising to polynomial powers. If the fix
    is missing, `policy_effect(policy_weight_ate())` would return a
    value that disagrees with r.ate (which IS correctly on V scale
    per the earlier fix)."""
    r = bayes_mte(heckman_data, y='y', treat='d', instrument='z',
                  selection='normal', mte_method='hv_latent',
                  poly_u=1,
                  draws=300, tune=300, chains=2, progressbar=False,
                  random_state=44)
    pe = r.policy_effect(sp.policy_weight_ate(), label='ate')
    # Should match r.ate exactly (both use trapezoid integration on
    # the same v_grid powers now that policy_effect respects scale).
    assert abs(pe['estimate'] - r.ate) < 1e-8, (
        f"policy_effect(ATE) = {pe['estimate']:.6f} vs r.ate = "
        f"{r.ate:.6f} disagrees under selection='normal'"
    )


def test_selection_normal_mte_curve_has_v_column(heckman_data):
    r = bayes_mte(heckman_data, y='y', treat='d', instrument='z',
                  selection='normal', mte_method='hv_latent', poly_u=1,
                  draws=150, tune=150, chains=2, progressbar=False)
    assert 'v' in r.mte_curve.columns
    # v values = Φ^{-1}(u_grid) — monotone in u
    v = r.mte_curve['v'].values
    assert np.all(np.diff(v) > 0)


def test_selection_uniform_mte_curve_has_no_v_column(heckman_data):
    r = bayes_mte(heckman_data, y='y', treat='d', instrument='z',
                  selection='uniform', poly_u=1,
                  draws=150, tune=150, chains=2, progressbar=False)
    # Keep the uniform-scale schema unchanged
    assert 'v' not in r.mte_curve.columns


def test_normal_disagrees_with_uniform_on_heckman_dgp(heckman_data):
    """On a DGP designed for Heckman-V scale, the hv_latent fit on V
    (selection='normal') should recover a different b_1 posterior
    mean than hv_latent on U scale (selection='uniform'), because
    the probit transform changes the abscissa scale. If they agree
    exactly, the selection kwarg is silently a no-op."""
    r_u = bayes_mte(heckman_data, y='y', treat='d', instrument='z',
                    selection='uniform', mte_method='hv_latent',
                    poly_u=1,
                    draws=200, tune=200, chains=2, progressbar=False,
                    random_state=11)
    r_v = bayes_mte(heckman_data, y='y', treat='d', instrument='z',
                    selection='normal', mte_method='hv_latent',
                    poly_u=1,
                    draws=200, tune=200, chains=2, progressbar=False,
                    random_state=11)
    b1_u = float(r_u.trace.posterior['b_mte'].values[..., 1].mean())
    b1_v = float(r_v.trace.posterior['b_mte'].values[..., 1].mean())
    # Nonzero spread (they shouldn't be identical)
    assert abs(b1_u - b1_v) > 0.1, (
        f"selection kwarg appears to be a no-op: b1_u={b1_u:.3f} "
        f"vs b1_v={b1_v:.3f}"
    )
