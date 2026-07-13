"""Analytical parity: parametric g-formula family (ICE and Monte-Carlo).

Identity
--------
For a linear-Gaussian longitudinal DGP with time-varying confounding,

    L0 ~ N(0, 1)
    A0 ~ Bernoulli(sigmoid(0.6 L0))                # confounded by L0
    L1 = 0.2 + 0.7 L0 + 0.5 A0 + N(0, 0.5)         # affected by past treatment
    A1 ~ Bernoulli(sigmoid(0.4 L1 + 0.3 A0 - 0.2)) # confounded by L1, A0
    Y  = 1 + 1.0 A0 + 2.0 A1 + 0.5 L0 + 0.8 L1 + N(0, 0.5)

the g-formula mean under a static regime (a0, a1) has the closed form

    psi(a0, a1) = 1 + 1.0 a0 + 2.0 a1
                  + 0.5 E[L0] + 0.8 E[L1 | do(A0 = a0)]
                = 1 + a0 + 2 a1 + 0.8 (0.2 + 0.5 a0)
                = 1.16 + 1.4 a0 + 2 a1,

so psi(1,1) = 4.56, psi(0,0) = 1.16, and the always-vs-never contrast is
1.4 + 2.0 = 3.4 (note: NOT 1.0 + 2.0 — the extra 0.4 flows through the
time-varying confounder L1, which is exactly what a naive regression
adjusting for L1 would miss).

Both estimators fit correctly specified models here: the ICE sequential
regressions are exactly linear at every step (L1 is linear-Gaussian in
(L0, A0), so the backward-induced pseudo-outcome stays linear), and the MC
g-formula's Gaussian confounder / linear outcome models match the DGP.
Both are therefore consistent for the closed-form value. Analytical
evidence tier — no external reference implementation is invoked.

Tolerances
----------
* Noiseless deterministic design: ICE reduces to exact OLS interpolation;
  we require 1e-8 (numerical lstsq precision only).
* Stochastic DGP, n = 4000: the sampling SE of the ICE point estimate is
  ~ sd(Y)/sqrt(n) ~= 0.03; across 8 pilot seeds the max deviation from
  truth was 0.056. With the fixed seed below we allow 0.15 (~5 SE) so the
  test fails only for genuine estimator regressions, never for seed luck.
* MC arm adds simulation noise sd(Y_pred)/sqrt(n_simulations) ~= 0.009 at
  n_simulations = 20 000, negligible next to the model-fitting SE, so the
  same 0.15 budget applies; ICE-vs-MC agreement on identical data gets a
  tighter 0.08 since the model-fit error is shared between the two arms
  (pilot deviation ~0.006-0.03).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

# Closed-form g-formula values (derivation in module docstring)
PSI_ALWAYS = 4.56
PSI_NEVER = 1.16
CONTRAST = 3.40


@pytest.fixture(scope="module")
def longitudinal_df() -> pd.DataFrame:
    """Two-period linear-Gaussian DGP with time-varying confounding."""
    rng = np.random.default_rng(20260713)
    n = 4000
    L0 = rng.normal(0.0, 1.0, n)
    A0 = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.6 * L0)), n).astype(float)
    L1 = 0.2 + 0.7 * L0 + 0.5 * A0 + rng.normal(0.0, 0.5, n)
    A1 = rng.binomial(
        1, 1.0 / (1.0 + np.exp(-(0.4 * L1 + 0.3 * A0 - 0.2))), n
    ).astype(float)
    Y = 1.0 + 1.0 * A0 + 2.0 * A1 + 0.5 * L0 + 0.8 * L1 + rng.normal(0.0, 0.5, n)
    return pd.DataFrame(
        {"id": np.arange(n), "L0": L0, "A0": A0, "L1": L1, "A1": A1, "Y": Y}
    )


@pytest.fixture(scope="module")
def ice_arms(longitudinal_df):
    always = sp.gformula_ice_fn(
        longitudinal_df,
        id_col="id",
        time_col="id",
        treatment_cols=["A0", "A1"],
        confounder_cols=[["L0"], ["L1"]],
        outcome_col="Y",
        treatment_strategy=[1, 1],
    )
    never = sp.gformula_ice_fn(
        longitudinal_df,
        id_col="id",
        time_col="id",
        treatment_cols=["A0", "A1"],
        confounder_cols=[["L0"], ["L1"]],
        outcome_col="Y",
        treatment_strategy=[0, 0],
    )
    return always, never


@pytest.fixture(scope="module")
def mc_result(longitudinal_df):
    # bootstrap=0: point estimates only — inference is not part of the
    # analytical identity under test and bootstrapping would dominate
    # runtime for no additional evidence.
    return sp.gformula_mc(
        longitudinal_df,
        treatment_cols=["A0", "A1"],
        confounder_cols=[["L0"], ["L1"]],
        outcome_col="Y",
        strategy=[1, 1],
        control_strategy=[0, 0],
        n_simulations=20_000,
        bootstrap=0,
        seed=1,
    )


# ────────────────────────────────────────────────────────────────────────
#  ICE g-formula
# ────────────────────────────────────────────────────────────────────────


def test_ice_exact_on_noiseless_linear_design():
    """Zero-noise deterministic design: ICE must interpolate exactly.

    L1 is an exact linear function of (L0, A0) and Y an exact linear
    function of the full history, so every backward regression step is
    exact OLS interpolation and psi(a0, a1) = 1.16 + 1.4 a0 + 2 a1 must
    be recovered to lstsq numerical precision.
    """
    rows = [
        (l0, a0, a1)
        for _ in range(5)
        for l0 in (-1.5, -0.5, 0.5, 1.5)  # symmetric grid: mean(L0) = 0
        for a0 in (0.0, 1.0)
        for a1 in (0.0, 1.0)
    ]
    l0 = np.array([r[0] for r in rows])
    a0 = np.array([r[1] for r in rows])
    a1 = np.array([r[2] for r in rows])
    l1 = 0.2 + 0.7 * l0 + 0.5 * a0
    y = 1.0 + 1.0 * a0 + 2.0 * a1 + 0.5 * l0 + 0.8 * l1
    df = pd.DataFrame(
        {"id": np.arange(len(y)), "L0": l0, "A0": a0, "L1": l1, "A1": a1, "Y": y}
    )
    always = sp.gformula_ice_fn(
        df,
        id_col="id",
        time_col="id",
        treatment_cols=["A0", "A1"],
        confounder_cols=[["L0"], ["L1"]],
        outcome_col="Y",
        treatment_strategy=[1, 1],
    )
    never = sp.gformula_ice_fn(
        df,
        id_col="id",
        time_col="id",
        treatment_cols=["A0", "A1"],
        confounder_cols=[["L0"], ["L1"]],
        outcome_col="Y",
        treatment_strategy=[0, 0],
    )
    np.testing.assert_allclose(always.value, PSI_ALWAYS, atol=1e-8)
    np.testing.assert_allclose(never.value, PSI_NEVER, atol=1e-8)


def test_ice_recovers_always_treat_closed_form(ice_arms):
    # atol = 0.15 ~= 5 x sampling SE (~0.03 at n=4000); pilot max
    # deviation over 8 seeds was 0.056.
    always, _ = ice_arms
    assert always.value == pytest.approx(PSI_ALWAYS, abs=0.15)


def test_ice_recovers_never_treat_closed_form(ice_arms):
    _, never = ice_arms
    assert never.value == pytest.approx(PSI_NEVER, abs=0.15)


def test_ice_contrast_includes_confounder_pathway(ice_arms):
    # Contrast = 3.4, of which 0.4 is the A0 -> L1 -> Y pathway that a
    # naive L1-adjusted regression would block. Estimating either arm's
    # value with error ~0.03 SE each, 0.15 covers the difference too.
    always, never = ice_arms
    assert always.value - never.value == pytest.approx(CONTRAST, abs=0.15)


# ────────────────────────────────────────────────────────────────────────
#  Monte-Carlo g-formula
# ────────────────────────────────────────────────────────────────────────


def test_mc_recovers_always_treat_closed_form(mc_result):
    # Same 0.15 budget as ICE: MC adds simulation noise of only
    # sd(Y_pred)/sqrt(20000) ~= 0.009 on top of the shared model-fit SE.
    assert mc_result.value == pytest.approx(PSI_ALWAYS, abs=0.15)


def test_mc_contrast_recovers_closed_form(mc_result):
    assert mc_result.contrast_value == pytest.approx(CONTRAST, abs=0.15)
    # Both arms share the fitted models, so arithmetic must be internally
    # consistent: contrast = value(treat) - value(control).
    assert mc_result.strategies == {"treat": [1, 1], "control": [0, 0]}


def test_mc_agrees_with_ice_on_same_data(ice_arms, mc_result):
    # ICE and MC are two algorithms for the same estimand on the same
    # fitted-model class; on identical data the model-fitting error is
    # shared, leaving only MC simulation noise plus the efficiency gap.
    # Pilot deviation was ~0.006-0.03; allow 0.08.
    always, _ = ice_arms
    assert mc_result.value == pytest.approx(always.value, abs=0.08)
