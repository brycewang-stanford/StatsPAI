"""Analytical parity: sp.multi_treatment AIPW recovery of known arm effects.

Under unconfoundedness, the Cattaneo (2010) AIPW estimator of a multi-valued
treatment identifies E[Y(k)] for every arm k via

    E[Y(k)] = E[ mu_k(X) + 1{D=k} * (Y - mu_k(X)) / e_k(X) ],

so each contrast E[Y(k)] - E[Y(ref)] equals the arm-specific ATE. We simulate
a 3-arm DGP with *confounded* assignment: a multinomial-logit treatment whose
utilities load on x1 (u1 = +0.6*x1, u2 = -0.6*x1 + 0.4*x2) while the outcome
also loads on x1 and x2:

    Y = tau_1*1{D=1} + tau_2*1{D=2} + 1.0*x1 + 0.5*x2 + N(0,1).

The confounding is strong enough that naive arm-mean differences are biased
by roughly +0.45 (arm 1) and -0.45 (arm 2); AIPW with the correctly specified
multinomial-logit GPS and a GBM outcome model must undo that bias and recover
tau_1 = 1.0 and tau_2 = 2.5 within abs=0.25 at n=2500 (observed error across
seeds <= 0.11; bootstrap SE ~ 0.06, so 0.25 is > 2x the worst seed error yet
far tighter than the naive bias). A second DGP includes a genuine null arm
(tau_1 = 0) whose 95% CI must cover 0. Analytical evidence tier: known-DGP
recovery plus internal identities (n_arms - 1 contrasts, potential-outcome
ordering), no external reference implementation.

The full call is deterministic (fixed rng seed + fixed random_state for the
bootstrap), so the stochastic margins above only need to hold for the seeds
baked in here. n_bootstrap=30 keeps the two module-scoped fits ~12s total.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

N = 2500
NBOOT = 30  # bootstrap only feeds SE/CI width; 30 replicates suffice here.
TAU1, TAU2 = 1.0, 2.5  # arm effects vs reference arm 0 (effects DGP)
NULL_TAU2 = 2.0  # nonzero arm in the null-arm DGP (its arm 1 has tau = 0)


def _three_arm_confounded_dgp(seed: int, tau1: float, tau2: float, n: int = N):
    """3-arm multinomial-logit assignment confounded with the outcome."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    # Softmax over utilities (0, 0.6*x1, -0.6*x1 + 0.4*x2): x1 pushes units
    # into arm 1 and out of arm 2, and x1 also raises Y -> confounded.
    util = np.column_stack([np.zeros(n), 0.6 * x1, -0.6 * x1 + 0.4 * x2])
    probs = np.exp(util)
    probs /= probs.sum(axis=1, keepdims=True)
    draws = rng.random(n)
    treat = (draws[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
    y = (
        tau1 * (treat == 1)
        + tau2 * (treat == 2)
        + 1.0 * x1
        + 0.5 * x2
        + rng.normal(0, 1, n)
    )
    return pd.DataFrame({"y": y, "T": treat, "x1": x1, "x2": x2})


@pytest.fixture(scope="module")
def effects_case():
    """Confounded DGP with known nonzero effects tau_1=1.0, tau_2=2.5."""
    df = _three_arm_confounded_dgp(seed=0, tau1=TAU1, tau2=TAU2)
    res = sp.multi_treatment(
        df,
        y="y",
        treat="T",
        covariates=["x1", "x2"],
        reference=0,
        n_bootstrap=NBOOT,
        random_state=42,
    )
    return df, res


@pytest.fixture(scope="module")
def null_arm_case():
    """Confounded DGP where arm 1 has exactly zero effect (tau_1=0)."""
    df = _three_arm_confounded_dgp(seed=11, tau1=0.0, tau2=NULL_TAU2)
    res = sp.multi_treatment(
        df,
        y="y",
        treat="T",
        covariates=["x1", "x2"],
        reference=0,
        n_bootstrap=NBOOT,
        random_state=42,
    )
    return df, res


def test_recovers_each_pairwise_effect(effects_case):
    _, res = effects_case
    eff = res.detail.set_index("treatment")["estimate"]
    # abs=0.25: > 2x the worst observed seed error (0.11) and ~4 bootstrap
    # SEs, yet well below the ~0.45 naive confounding bias, so a pass proves
    # genuine bias correction rather than a loose window.
    assert eff[1] == pytest.approx(TAU1, abs=0.25)
    assert eff[2] == pytest.approx(TAU2, abs=0.25)


def test_corrects_naive_confounding_bias(effects_case):
    df, res = effects_case
    naive = df.groupby("T")["y"].mean()
    naive_1 = float(naive[1] - naive[0])
    naive_2 = float(naive[2] - naive[0])
    # The DGP tilts high-x1 units into arm 1 and out of arm 2, so the naive
    # contrasts are biased up (arm 1) and down (arm 2) by ~0.45.
    assert naive_1 > TAU1 + 0.25
    assert naive_2 < TAU2 - 0.25
    # AIPW must land strictly closer to the truth than the naive contrast.
    eff = res.detail.set_index("treatment")["estimate"]
    assert abs(eff[1] - TAU1) < abs(naive_1 - TAU1)
    assert abs(eff[2] - TAU2) < abs(naive_2 - TAU2)


def test_null_arm_ci_covers_zero(null_arm_case):
    _, res = null_arm_case
    row = res.detail.set_index("treatment").loc[1]
    assert row["ci_lower"] <= 0.0 <= row["ci_upper"]
    assert row["estimate"] == pytest.approx(0.0, abs=0.2)
    # The genuinely nonzero arm is still recovered in the same fit.
    eff2 = res.detail.set_index("treatment").loc[2, "estimate"]
    assert eff2 == pytest.approx(NULL_TAU2, abs=0.25)


def test_number_of_contrasts_is_arms_minus_one(effects_case):
    _, res = effects_case
    assert res.model_info["n_levels"] == 3
    assert len(res.detail) == res.model_info["n_levels"] - 1
    assert res.model_info["reference"] == 0
    assert 0 not in set(res.detail["treatment"])
    assert res.model_info["treatment_levels"] == [0, 1, 2]


def test_potential_outcome_means_ordering_matches_dgp(effects_case):
    _, res = effects_case
    po = res.model_info["potential_outcomes"]
    # tau_2 > tau_1 > 0 and E[x]=0 => E[Y(2)] > E[Y(1)] > E[Y(0)].
    assert po[2] > po[1] > po[0]
    # E[Y(0)] = 0 in this DGP (covariates and noise are mean-zero).
    assert po[0] == pytest.approx(0.0, abs=0.2)
    # Contrasts in detail must equal potential-outcome differences exactly.
    eff = res.detail.set_index("treatment")["estimate"]
    assert eff[1] == pytest.approx(po[1] - po[0], abs=1e-10)
    assert eff[2] == pytest.approx(po[2] - po[0], abs=1e-10)


def test_inference_columns_are_coherent(effects_case):
    _, res = effects_case
    det = res.detail
    assert np.all(det["se"] > 0)
    assert np.all(det["ci_lower"] < det["estimate"])
    assert np.all(det["estimate"] < det["ci_upper"])
    assert np.all((det["pvalue"] >= 0) & (det["pvalue"] <= 1))
    # Headline estimate mirrors the first contrast row.
    assert res.estimate == pytest.approx(float(det.iloc[0]["estimate"]))
    assert res.n_obs == N
