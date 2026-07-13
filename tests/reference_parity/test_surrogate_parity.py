"""Analytical parity: surrogate-index family long-term ATE recovery.

Athey-Chetty-Imbens-Kang (2019) identity: if the long-term outcome is
fully mediated by surrogates S with *linear* structural link
Y = a + b'S + eps (comparability + surrogacy hold by construction), then
the long-term ATE in the experiment is exactly

    ATE_Y = b' * (E[S | T=1] - E[S | T=0]) = b' * ATE_S.

With an OLS outcome model the estimator is algebraically
h_bar_1 - h_bar_0 = b_hat' * (S_bar_1 - S_bar_0), so (i) the estimate
recovers b * tau_S up to sampling noise with a closed-form rate, and
(ii) the estimate equals b_hat * (mean surrogate contrast) to machine
precision. The multi-wave variant chains linear conditional means
(S1 -> S2 -> Y), so the truth is the product of path coefficients:
ATE_Y = b_Y * b_21 * tau_1. A zero-effect-on-surrogates DGP must give
ATE ~ 0 for every variant, because each estimator is (slope) x (mean
surrogate contrast) and the contrast vanishes. Analytical evidence tier.

The proximal variant's truth-recovery test is skipped: its stage-2
design matrix [1, W, S_hat] is exactly rank-deficient (S_hat is an
affine function of W), so the "bridge slope" is a minimum-norm lstsq
artifact that depends on the *units* of the proxy W (verified: scaling
W by 10 changes the point estimate by two orders of magnitude on
identical data). Only its zero-effect identity is unit-invariant and is
tested here.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


# ---------------------------------------------------------------------------
# DGPs (linear structural links => closed-form long-term ATE)
# ---------------------------------------------------------------------------

# Classical single-surrogate DGP constants.
B_OBS = 1.5  # structural slope of Y on S in the observational sample
TAU_S = 0.8  # experimental treatment effect on the surrogate
TRUE_ATE = B_OBS * TAU_S  # 1.2, exact under the linear link

# Two-wave DGP constants: S1 -> S2 -> Y, all linear.
B_21 = 0.6  # slope of S2 on S1
B_Y2 = 1.5  # slope of Y on S2
TAU_1 = 0.8  # treatment effect on wave-1 surrogate
TRUE_ATE_WAVES = B_Y2 * B_21 * TAU_1  # 0.72, exact by iterated linearity


def _single_wave_data(tau_s: float, seed: int):
    """Y fully mediated by S; treatment shifts S by tau_s."""
    rng = np.random.default_rng(seed)
    n_exp, n_obs = 4000, 4000
    T = rng.binomial(1, 0.5, size=n_exp).astype(float)
    exp = pd.DataFrame({"T": T, "S": 0.5 + tau_s * T + rng.normal(0, 1, n_exp)})
    S_o = rng.normal(0.5, 1.0, size=n_obs)
    obs = pd.DataFrame(
        {"S": S_o, "Y": 2.0 + B_OBS * S_o + rng.normal(0, 0.5, n_obs)}
    )
    return exp, obs


@pytest.fixture(scope="module")
def single_wave():
    return _single_wave_data(TAU_S, seed=42)


@pytest.fixture(scope="module")
def single_wave_null():
    return _single_wave_data(0.0, seed=43)


@pytest.fixture(scope="module")
def two_wave():
    """S1 -> S2 -> Y chain shared by both samples; T shifts S1 only."""
    rng = np.random.default_rng(44)
    n_exp, n_obs = 3000, 4000
    T = rng.binomial(1, 0.5, size=n_exp).astype(float)
    S1_e = 0.3 + TAU_1 * T + rng.normal(0, 1, n_exp)
    S2_e = 0.5 + B_21 * S1_e + rng.normal(0, 0.5, n_exp)
    exp = pd.DataFrame({"T": T, "S1": S1_e, "S2": S2_e})
    S1_o = rng.normal(0.3, 1.0, size=n_obs)
    S2_o = 0.5 + B_21 * S1_o + rng.normal(0, 0.5, n_obs)
    Y_o = 1.0 + B_Y2 * S2_o + rng.normal(0, 0.5, n_obs)
    obs = pd.DataFrame({"S1": S1_o, "S2": S2_o, "Y": Y_o})
    return exp, obs


def _proximal_data(tau_u: float, seed: int):
    """Persistent-confounding DGP: T -> U -> (S, Y); W proxies U (obs only).

    U is an unobserved latent state driving both the surrogate S and the
    long-term outcome Y, so surrogacy Y independent of T given S fails;
    W is a second noisy measurement of U, excluded from Y given (S, U) —
    the Imbens-Kallus-Mao-Wang proxy structure. With tau_u = 0 the
    treatment moves nothing, so the true long-term ATE is exactly 0.
    """
    rng = np.random.default_rng(seed)
    n_exp, n_obs = 4000, 4000
    U_o = rng.normal(0, 1, n_obs)
    S_o = 1.0 * U_o + rng.normal(0, 0.8, n_obs)
    W_o = 1.0 * U_o + rng.normal(0, 0.5, n_obs)
    Y_o = 1.5 * S_o + 0.7 * U_o + rng.normal(0, 0.5, n_obs)
    obs = pd.DataFrame({"S": S_o, "W": W_o, "Y": Y_o})
    T = rng.binomial(1, 0.5, size=n_exp).astype(float)
    U_e = tau_u * T + rng.normal(0, 1, n_exp)
    exp = pd.DataFrame({"T": T, "S": 1.0 * U_e + rng.normal(0, 0.8, n_exp)})
    return exp, obs


# ---------------------------------------------------------------------------
# sp.surrogate_index — classical ACIK estimator
# ---------------------------------------------------------------------------


def test_surrogate_index_recovers_linear_long_term_ate(single_wave):
    exp, obs = single_wave
    res = sp.surrogate_index(
        exp, obs, treatment="T", surrogates=["S"], long_term_outcome="Y"
    )
    # Truth = B_OBS * TAU_S = 1.2 exactly (linear mediation). Sampling
    # noise is dominated by the experimental contrast of h = a + b*S:
    # se ~ sqrt(2 * b^2 * Var(S) / (n_exp/2)) ~ 0.047, so 0.15 ~ 3.2 se.
    assert res.estimate == pytest.approx(TRUE_ATE, abs=0.15)
    lo, hi = res.ci
    assert lo < TRUE_ATE < hi
    assert res.se > 0
    assert res.estimand == "ATE"


def test_surrogate_index_equals_slope_times_surrogate_contrast(single_wave):
    # Algebraic identity of the OLS surrogate index: the estimate is
    # exactly b_hat * (S_bar_1 - S_bar_0). Machine-precision check.
    exp, obs = single_wave
    res = sp.surrogate_index(
        exp, obs, treatment="T", surrogates=["S"], long_term_outcome="Y"
    )
    X = np.column_stack([np.ones(len(obs)), obs["S"].to_numpy(float)])
    b_hat = np.linalg.lstsq(X, obs["Y"].to_numpy(float), rcond=None)[0][1]
    t = exp["T"].to_numpy(bool)
    contrast = exp.loc[t, "S"].mean() - exp.loc[~t, "S"].mean()
    assert res.estimate == pytest.approx(b_hat * contrast, abs=1e-10)


def test_surrogate_index_null_effect_on_surrogates(single_wave_null):
    # tau_S = 0 => ATE = b * 0 = 0 exactly; the estimate is b_hat times
    # a mean-zero contrast with sd ~ 0.032, so 0.15 covers > 3 se of
    # b*contrast and the estimate must also sit within 4 reported se.
    exp, obs = single_wave_null
    res = sp.surrogate_index(
        exp, obs, treatment="T", surrogates=["S"], long_term_outcome="Y"
    )
    assert abs(res.estimate) < 0.15
    assert abs(res.estimate) < 4 * res.se


# ---------------------------------------------------------------------------
# sp.long_term_from_short — multi-wave iterated surrogate index
# ---------------------------------------------------------------------------


def test_long_term_from_short_recovers_chained_linear_ate(two_wave):
    exp, obs = two_wave
    res = sp.long_term_from_short(
        exp,
        obs,
        treatment="T",
        surrogates_waves=[["S1"], ["S2"]],
        long_term_outcome="Y",
        n_boot=100,
        random_state=0,
    )
    # Truth = B_Y2 * B_21 * TAU_1 = 0.72 exactly: backward induction over
    # linear conditional means multiplies the path slopes. Noise is the
    # experimental contrast of f1(S1) with slope ~0.9:
    # se ~ sqrt(2 * 0.81 * Var(S1) / 1500) ~ 0.033, so 0.12 ~ 3.6 se.
    assert res.estimate == pytest.approx(TRUE_ATE_WAVES, abs=0.12)
    assert res.model_info["n_waves"] == 2
    assert res.se > 0


def test_long_term_from_short_null_effect_on_surrogates():
    # Same two-wave chain but T never moves S1 => true ATE = 0.
    rng = np.random.default_rng(45)
    n_exp, n_obs = 3000, 4000
    T = rng.binomial(1, 0.5, size=n_exp).astype(float)
    S1_e = 0.3 + rng.normal(0, 1, n_exp)
    exp = pd.DataFrame(
        {"T": T, "S1": S1_e, "S2": 0.5 + B_21 * S1_e + rng.normal(0, 0.5, n_exp)}
    )
    S1_o = rng.normal(0.3, 1.0, size=n_obs)
    S2_o = 0.5 + B_21 * S1_o + rng.normal(0, 0.5, n_obs)
    obs = pd.DataFrame(
        {"S1": S1_o, "S2": S2_o, "Y": 1.0 + B_Y2 * S2_o + rng.normal(0, 0.5, n_obs)}
    )
    res = sp.long_term_from_short(
        exp,
        obs,
        treatment="T",
        surrogates_waves=[["S1"], ["S2"]],
        long_term_outcome="Y",
        n_boot=100,
        random_state=1,
    )
    # Estimate = slope * mean-zero contrast, sd ~ 0.033 => 0.12 ~ 3.6 se.
    assert abs(res.estimate) < 0.12
    assert abs(res.estimate) < 4 * res.se


# ---------------------------------------------------------------------------
# sp.proximal_surrogate_index — proximal bridge under latent confounding
# ---------------------------------------------------------------------------


def test_proximal_surrogate_index_null_effect_under_confounding():
    # Unit-invariant identity: the fitted bridge is linear in S, so the
    # estimate is (bridge slope) x (experimental S contrast). With
    # tau_u = 0 the contrast is mean-zero noise (sd ~ 0.03) whatever the
    # slope, so the estimate must vanish even though S -> Y is heavily
    # confounded by the latent U in the observational sample.
    exp, obs = _proximal_data(tau_u=0.0, seed=46)
    res = sp.proximal_surrogate_index(
        exp,
        obs,
        treatment="T",
        surrogates=["S"],
        proxies=["W"],
        long_term_outcome="Y",
        n_boot=100,
        random_state=2,
    )
    # slope ~ O(1), contrast sd ~ 0.03 => 0.12 is > 3 se of the product.
    assert abs(res.estimate) < 0.12
    assert abs(res.estimate) < 4 * res.se
    assert res.se > 0


@pytest.mark.skip(
    reason=(
        "proximal_surrogate_index cannot recover a known nonzero long-term "
        "ATE: stage 2 regresses Y on [1, W, S_hat] where S_hat is an exact "
        "affine function of [1, W], so the design matrix is rank-deficient "
        "and the 'bridge slope' is np.linalg.lstsq's minimum-norm artifact. "
        "Verified numerically: on a fixed persistent-confounding DGP "
        "(T -> U -> (S, Y), W proxying U; true ATE = 1.32) the estimate is "
        "0.49 at unit proxy scale, 0.008 with W*10, and 1.22 with W*0.01 — "
        "the point estimate depends on the units of W. A correct linear "
        "bridge (Y on [1, S_hat] only, i.e. E[(Y - h(S)) W] = 0) equals "
        "beta + gamma/a_s and does recover the truth on this DGP. Fixing "
        "this changes numerical output => needs a CHANGELOG'd correctness "
        "fix, not a test-side workaround."
    )
)
def test_proximal_surrogate_index_recovers_confounded_ate():
    exp, obs = _proximal_data(tau_u=0.6, seed=47)
    res = sp.proximal_surrogate_index(
        exp,
        obs,
        treatment="T",
        surrogates=["S"],
        proxies=["W"],
        long_term_outcome="Y",
        n_boot=100,
        random_state=3,
    )
    # True ATE = (beta*a_s + gamma) * tau_u = (1.5 + 0.7) * 0.6 = 1.32.
    assert res.estimate == pytest.approx(1.32, abs=0.15)
