"""Tests for ``sp.bayes_mte(mte_method='bivariate_normal')`` —
textbook Heckman-Vytlacil trivariate-normal MTE model
(U_0, U_1, V) ~ N(0, Σ) with D = 1{Z'π > V}.

The model identifies MTE(v) = β_D + (σ_1V - σ_0V)·v on the probit
V-scale and requires selection='normal' + first_stage='joint'. It
differs from mte_method='polynomial' + mte_method='hv_latent' by
ADDING inverse-Mills-ratio correction terms to the structural
equation; the trade-off is fewer free parameters (2 covariances
vs poly_u+1 polynomial coefficients) and tighter posteriors at the
price of a stricter structural assumption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.bayes import BayesianMTEResult, bayes_mte

pymc = pytest.importorskip(
    "pymc",
    reason="PyMC is an optional dependency; skip Bayesian tests if missing.",
)


def _trivariate_normal_dgp(n, sigma_0V, sigma_1V, beta_D, seed):
    """Full HV DGP: (U_0, U_1, V) ~ N(0, Σ); D = 1{Z'π > V}.

    ``β_D = μ_1 - μ_0`` is the MTE intercept on V=0.
    ``σ_1V - σ_0V`` is the MTE slope in v.
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n)

    # Covariance of (U_0, U_1, V). Variances fixed at 1 for V (selection
    # scale) and diagonal variances of 1 on U_0, U_1 for simplicity. Only
    # the σ_0V and σ_1V off-diagonals matter for identification.
    cov = np.array([
        [1.0,       0.0,       sigma_0V],
        [0.0,       1.0,       sigma_1V],
        [sigma_0V,  sigma_1V,  1.0     ],
    ])
    errs = rng.multivariate_normal(mean=[0, 0, 0], cov=cov, size=n)
    U_0, U_1, V = errs[:, 0], errs[:, 1], errs[:, 2]

    mu_0 = 1.0
    mu_1 = mu_0 + beta_D
    Y_0 = mu_0 + U_0
    Y_1 = mu_1 + U_1

    # Selection: D = 1{Z'π > V}. Use a moderately strong instrument.
    pi_Z = 0.9
    D = (pi_Z * Z > V).astype(float)
    Y = D * Y_1 + (1.0 - D) * Y_0
    return pd.DataFrame({'y': Y, 'd': D, 'z': Z})


@pytest.fixture
def hv_data():
    # MTE(v) = 0.8 + (1.0 - (-0.5))·v = 0.8 + 1.5·v
    return _trivariate_normal_dgp(
        n=800, sigma_0V=-0.5, sigma_1V=1.0, beta_D=0.8, seed=1234,
    )


# ---------------------------------------------------------------------------
# API + input validation
# ---------------------------------------------------------------------------


def test_bivariate_normal_requires_normal_selection(hv_data):
    with pytest.raises(ValueError, match="selection='normal'"):
        bayes_mte(hv_data, y='y', treat='d', instrument='z',
                  mte_method='bivariate_normal',
                  selection='uniform', first_stage='joint',
                  draws=50, tune=50, chains=1, progressbar=False)


def test_bivariate_normal_requires_joint_first_stage(hv_data):
    with pytest.raises(ValueError, match="first_stage='joint'"):
        bayes_mte(hv_data, y='y', treat='d', instrument='z',
                  mte_method='bivariate_normal',
                  selection='normal', first_stage='plugin',
                  draws=50, tune=50, chains=1, progressbar=False)


def test_bivariate_normal_overrides_poly_u(hv_data):
    with pytest.warns(UserWarning, match='linear in V'):
        r = bayes_mte(hv_data, y='y', treat='d', instrument='z',
                      mte_method='bivariate_normal',
                      selection='normal', first_stage='joint',
                      poly_u=3,
                      draws=50, tune=50, chains=1, progressbar=False)
    # The overridden poly_u must be recorded as 1 in model_info so
    # downstream code and users reading the result see the effective
    # polynomial order, not the ignored one.
    assert r.model_info['poly_u'] == 1


# ---------------------------------------------------------------------------
# Fit + structural coefficients in posterior
# ---------------------------------------------------------------------------


def test_bivariate_normal_exposes_b_mte_shape_2(hv_data):
    r = bayes_mte(hv_data, y='y', treat='d', instrument='z',
                  mte_method='bivariate_normal',
                  selection='normal', first_stage='joint',
                  draws=200, tune=200, chains=2, progressbar=False,
                  random_state=0)
    assert isinstance(r, BayesianMTEResult)
    # b_mte is a Deterministic of shape (2,) = [β_D, σ_1V - σ_0V]
    b = r.trace.posterior['b_mte'].values
    assert b.shape[-1] == 2


def test_bivariate_normal_method_label_mentions_trivariate(hv_data):
    r = bayes_mte(hv_data, y='y', treat='d', instrument='z',
                  mte_method='bivariate_normal',
                  selection='normal', first_stage='joint',
                  draws=100, tune=100, chains=2, progressbar=False,
                  random_state=1)
    assert 'trivariate-normal' in r.method
    assert r.model_info['mte_method'] == 'bivariate_normal'


def test_bivariate_normal_structural_params_in_posterior(hv_data):
    r = bayes_mte(hv_data, y='y', treat='d', instrument='z',
                  mte_method='bivariate_normal',
                  selection='normal', first_stage='joint',
                  draws=100, tune=100, chains=2, progressbar=False,
                  random_state=2)
    for name in ('beta_D', 'sigma_0V', 'sigma_1V'):
        assert name in r.trace.posterior


# ---------------------------------------------------------------------------
# Recovery on the genuine HV trivariate-normal DGP
# ---------------------------------------------------------------------------


def test_bivariate_normal_recovers_slope_on_hv_dgp(hv_data):
    """On a DGP with σ_0V = -0.5, σ_1V = 1.0, β_D = 0.8, the
    posterior on ``b_mte`` should cover (0.8, 1.5) loosely at
    n=800 + 400 NUTS draws."""
    r = bayes_mte(hv_data, y='y', treat='d', instrument='z',
                  mte_method='bivariate_normal',
                  selection='normal', first_stage='joint',
                  draws=400, tune=400, chains=2, progressbar=False,
                  random_state=42)
    b = r.trace.posterior['b_mte'].values.reshape(-1, 2)
    b0_mean = float(b[:, 0].mean())  # β_D
    b1_mean = float(b[:, 1].mean())  # σ_1V - σ_0V
    # Within 0.8 of the truth — the model identifies these cleanly on
    # the HV DGP, and the trivariate-normal structural assumption
    # matches the DGP exactly so the posterior concentrates fast.
    assert abs(b0_mean - 0.8) < 0.8, (
        f"β_D posterior mean {b0_mean:.3f} drift from 0.8"
    )
    assert abs(b1_mean - 1.5) < 0.8, (
        f"(σ_1V - σ_0V) posterior mean {b1_mean:.3f} drift from 1.5"
    )
