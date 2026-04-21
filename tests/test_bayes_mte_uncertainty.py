"""Tests for v0.9.13 ATT/ATU uncertainty on BayesianMTEResult."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pymc = pytest.importorskip(
    "pymc",
    reason="PyMC is an optional dependency; skip Bayesian tests if missing.",
)

from statspai.bayes import bayes_mte


def _hv_dgp(n, slope, seed):
    """Reuse the Heckman DGP: tau(U) = 0 + slope·U."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n)
    U_D = rng.uniform(0, 1, size=n)
    p_true = 1.0 / (1.0 + np.exp(-0.8 * Z))
    D = (p_true > U_D).astype(float)
    tau = slope * U_D
    Y = 1.0 + tau * D + 0.3 * rng.normal(size=n)
    return pd.DataFrame({'y': Y, 'd': D, 'z': Z})


def test_att_atu_sd_fields_populated():
    """After a typical fit, both ATT and ATU SDs should be finite."""
    df = _hv_dgp(400, slope=1.5, seed=11)
    r = bayes_mte(df, y='y', treat='d', instrument='z',
                  mte_method='hv_latent', poly_u=1,
                  draws=250, tune=250, chains=2, progressbar=False)
    assert np.isfinite(r.att_sd)
    assert np.isfinite(r.atu_sd)
    assert r.att_sd > 0
    assert r.atu_sd > 0


def test_att_atu_hdi_brackets_posterior_mean():
    """HDI endpoints must straddle the posterior mean."""
    df = _hv_dgp(400, slope=1.5, seed=12)
    r = bayes_mte(df, y='y', treat='d', instrument='z',
                  mte_method='hv_latent', poly_u=1,
                  draws=250, tune=250, chains=2, progressbar=False)
    assert r.att_hdi_lower <= r.att <= r.att_hdi_upper, (
        f"ATT HDI [{r.att_hdi_lower:.3f}, {r.att_hdi_upper:.3f}] "
        f"does not bracket posterior mean {r.att:.3f}"
    )
    assert r.atu_hdi_lower <= r.atu <= r.atu_hdi_upper, (
        f"ATU HDI does not bracket posterior mean"
    )


def test_ate_uncertainty_via_posterior_sd_not_new_field():
    """`posterior_sd` already covers the primary ATE estimand — we
    don't duplicate it into an ``ate_sd`` attribute."""
    df = _hv_dgp(300, slope=1.0, seed=13)
    r = bayes_mte(df, y='y', treat='d', instrument='z', poly_u=1,
                  draws=200, tune=200, chains=2, progressbar=False)
    # posterior_sd maps to the ATE posterior SD
    assert r.posterior_sd > 0
    # No redundant `ate_sd` field
    assert not hasattr(r, 'ate_sd')


def test_bayes_mte_att_atu_both_finite_on_realistic_dgp():
    """Smoke: on a realistic DGP with positive counts on both sides,
    all four ATT/ATU finite fields should populate.

    The ``_integrated_effect`` helper has a defensive
    ``if U_population.size == 0`` branch returning ``(nan,)*4`` — but
    it is unreachable from ``bayes_mte`` because ``_logit_propensity``
    enforces a 2-class requirement upstream (sklearn's
    LogisticRegression raises before we ever reach the integrator).
    So we don't attempt to exercise the NaN path end-to-end; it's a
    guardrail for future refactors that might call the helper
    directly."""
    df = _hv_dgp(300, slope=1.0, seed=17)
    r = bayes_mte(df, y='y', treat='d', instrument='z', poly_u=1,
                  draws=150, tune=150, chains=2, progressbar=False)
    assert np.isfinite(r.att)
    assert np.isfinite(r.atu)
    assert np.isfinite(r.att_sd)
    assert np.isfinite(r.atu_sd)
