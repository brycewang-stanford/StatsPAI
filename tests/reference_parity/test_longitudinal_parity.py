"""Analytical parity: sp.longitudinal_analyze / sp.longitudinal_contrast
and the treatment-regime DSL (sp.regime / sp.always_treat / sp.never_treat).

Two deterministic DGPs with known population parameters:

1. **Randomized, subject-constant treatment (IPW path).** Each subject i
   draws A_i ~ Bernoulli(0.5) once and keeps A_it = A_i for all K = 3
   periods; the end-of-study outcome is

       Y_i = beta0 + tau * sum_t A_it + eps_i
           = beta0 + K*tau * A_i + eps_i,   eps_i ~ N(0, sigma^2).

   With no confounding, E[Y(always_treat)] = beta0 + K*tau and
   E[Y(never_treat)] = beta0, so the always-vs-never contrast is exactly
   K*tau = 2.4. Because ``analyze`` routes to the baseline-IPW estimator
   when no time-varying confounders are supplied (weights are constant
   under randomization), the estimator reduces to arm means of Y and must
   recover these values up to sampling noise.

2. **Sequential (time-varying) confounding (g-formula ICE path).** A
   two-period linear DGP where L_1 is affected by A_0 and confounds A_1:

       L_0 ~ N(0,1);  A_0 | L_0 ~ Bern(expit(1.2 L_0))
       L_1 = 0.5 L_0 + 0.6 A_0 + e_1
       A_1 | L_1 ~ Bern(expit(1.2 L_1))
       Y   = 1.0 + 0.5 A_0 + 0.7 A_1 + 0.4 L_0 + 0.8 L_1 + eps.

   Substituting the L_1 structural equation gives the closed-form truth
   E[Y(a0, a1)] = 1.0 + (0.5 + 0.8*0.6) a0 + 0.7 a1, hence

       E[Y(1,1)] - E[Y(0,0)] = 0.5 + 0.7 + 0.8*0.6 = 1.68.

   The ICE g-formula fits linear sequential regressions, which are
   correctly specified for this DGP, so it identifies the truth exactly;
   a naive unadjusted comparison of the concordant always/never arms is
   biased upward by roughly +1.1 (verified below), so passing is genuine
   evidence of confounding adjustment, not a smoke check.

Regime identity checks verify that sp.regime / sp.always_treat /
sp.never_treat materialize exactly the intended per-period treatment
vectors. Analytical evidence tier.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

# --------------------------------------------------------------------------- #
#  DGP 1 constants (randomized -> IPW path)
# --------------------------------------------------------------------------- #
N_IPW = 2000
K_IPW = 3
TAU = 0.8
BETA0 = 2.0
SIGMA = 0.5
# Estimate = arm mean of Y (constant weights under randomization), so the
# Monte-Carlo SE of each regime mean is sigma/sqrt(n/2) ~= 0.016 and of the
# contrast ~= 0.023 (MC over 20 seeds gave sd 0.025). Tolerances are set at
# ~4-5x those SEs so a correct estimator passes with wide margin while a
# missing-period or wrong-arm bug (error >= tau = 0.8) fails decisively.
ATOL_IPW_MEAN = 0.08
ATOL_IPW_DIFF = 0.10


@pytest.fixture(scope="module")
def ipw_panel():
    rng = np.random.default_rng(12345)
    a = rng.binomial(1, 0.5, N_IPW).astype(float)
    y = BETA0 + TAU * K_IPW * a + rng.normal(0.0, SIGMA, N_IPW)
    rows = []
    for i in range(N_IPW):
        for t in range(K_IPW):
            rows.append({"id": i, "time": t, "A": a[i], "Y": y[i]})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def ipw_fits(ipw_panel):
    always = sp.longitudinal_analyze(
        ipw_panel,
        id="id",
        time="time",
        treatment="A",
        outcome="Y",
        regime=sp.always_treat(K=K_IPW),
    )
    never = sp.longitudinal_analyze(
        ipw_panel,
        id="id",
        time="time",
        treatment="A",
        outcome="Y",
        regime=sp.never_treat(K=K_IPW),
    )
    return always, never


def test_analyze_dispatches_ipw_and_reports_panel_shape(ipw_fits):
    always, never = ipw_fits
    assert always.method == "ipw"
    assert always.regime_name == "always_treat"
    assert never.regime_name == "never_treat"
    assert always.n == N_IPW
    assert always.n_periods == K_IPW


def test_analyze_always_treat_recovers_beta0_plus_K_tau(ipw_fits):
    always, _ = ipw_fits
    # Population value E[Y(always)] = beta0 + K*tau = 2.0 + 3*0.8 = 4.4.
    assert always.estimate == pytest.approx(BETA0 + K_IPW * TAU, abs=ATOL_IPW_MEAN)


def test_analyze_never_treat_recovers_beta0(ipw_fits):
    _, never = ipw_fits
    # Population value E[Y(never)] = beta0 = 2.0.
    assert never.estimate == pytest.approx(BETA0, abs=ATOL_IPW_MEAN)


def test_analyze_always_minus_never_recovers_K_tau(ipw_fits):
    always, never = ipw_fits
    # Cumulative effect of K periods of treatment: K*tau = 2.4.
    assert always.estimate - never.estimate == pytest.approx(
        K_IPW * TAU, abs=ATOL_IPW_DIFF
    )


def test_contrast_recovers_K_tau_and_matches_analyze(ipw_panel, ipw_fits):
    always, never = ipw_fits
    res = sp.longitudinal_contrast(
        ipw_panel,
        id="id",
        time="time",
        treatment="A",
        outcome="Y",
        regime_a=sp.always_treat(K=K_IPW),
        regime_b=sp.never_treat(K=K_IPW),
    )
    assert res["regime_a"] == "always_treat"
    assert res["regime_b"] == "never_treat"
    # Population contrast K*tau = 2.4 (same tolerance rationale as above).
    assert res["contrast"] == pytest.approx(K_IPW * TAU, abs=ATOL_IPW_DIFF)
    # Internal identities (documented plug-in difference + delta-method SE)
    # hold to machine precision on the same deterministic data.
    assert res["contrast"] == pytest.approx(
        always.estimate - never.estimate, abs=1e-12
    )
    assert res["se"] == pytest.approx(
        float(np.sqrt(always.se**2 + never.se**2)), abs=1e-12
    )
    lo, hi = res["ci"]
    assert lo < K_IPW * TAU < hi


# --------------------------------------------------------------------------- #
#  DGP 2 constants (sequential confounding -> g-formula ICE path)
# --------------------------------------------------------------------------- #
N_GF = 800
GF_BETA0 = 1.0
GF_TAU0 = 0.5
GF_TAU1 = 0.7
GF_BL0 = 0.4
GF_BL1 = 0.8
GF_AL = 0.5  # L1 <- L0
GF_AA = 0.6  # L1 <- A0
# Closed-form: E[Y(a0,a1)] = beta0 + (tau0 + bL1*aA)*a0 + tau1*a1.
GF_TRUTH_ALWAYS = GF_BETA0 + (GF_TAU0 + GF_BL1 * GF_AA) + GF_TAU1  # 2.68
GF_TRUTH_NEVER = GF_BETA0  # 1.00
GF_TRUTH_DIFF = GF_TRUTH_ALWAYS - GF_TRUTH_NEVER  # 1.68
# Monte-Carlo over 30 seeds: sd(E[Y(1,1)]) ~= 0.036, sd(E[Y(0,0)]) ~= 0.045,
# sd(diff) ~= 0.062, all with negligible bias. Tolerances at ~3.5-4x MC SE;
# the naive confounded comparison is off by ~+1.1, far outside them.
ATOL_GF_MEAN = 0.15
ATOL_GF_DIFF = 0.25


@pytest.fixture(scope="module")
def gf_panel():
    rng = np.random.default_rng(6789)
    l0 = rng.normal(0.0, 1.0, N_GF)
    a0 = (rng.random(N_GF) < 1.0 / (1.0 + np.exp(-1.2 * l0))).astype(float)
    l1 = GF_AL * l0 + GF_AA * a0 + rng.normal(0.0, 0.5, N_GF)
    a1 = (rng.random(N_GF) < 1.0 / (1.0 + np.exp(-1.2 * l1))).astype(float)
    y = (
        GF_BETA0
        + GF_TAU0 * a0
        + GF_TAU1 * a1
        + GF_BL0 * l0
        + GF_BL1 * l1
        + rng.normal(0.0, 0.5, N_GF)
    )
    rows = []
    for i in range(N_GF):
        rows.append({"id": i, "time": 0, "A": a0[i], "L": l0[i], "Y": y[i]})
        rows.append({"id": i, "time": 1, "A": a1[i], "L": l1[i], "Y": y[i]})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def gf_fits(gf_panel):
    # time_varying + static regime -> auto-dispatch to the g-formula ICE path.
    always = sp.longitudinal_analyze(
        gf_panel,
        id="id",
        time="time",
        treatment="A",
        outcome="Y",
        time_varying=["L"],
        regime=sp.always_treat(K=2),
    )
    never = sp.longitudinal_analyze(
        gf_panel,
        id="id",
        time="time",
        treatment="A",
        outcome="Y",
        time_varying=["L"],
        regime=sp.never_treat(K=2),
    )
    return always, never


def test_gformula_recovers_regime_means_under_tv_confounding(gf_fits):
    always, never = gf_fits
    assert always.method == "g-formula"
    assert never.method == "g-formula"
    # Linear sequential regressions are correctly specified for this DGP,
    # so ICE identifies E[Y(1,1)] = 2.68 and E[Y(0,0)] = 1.00 exactly.
    assert always.estimate == pytest.approx(GF_TRUTH_ALWAYS, abs=ATOL_GF_MEAN)
    assert never.estimate == pytest.approx(GF_TRUTH_NEVER, abs=ATOL_GF_MEAN)


def test_gformula_contrast_recovers_truth_where_naive_is_biased(gf_panel, gf_fits):
    always, never = gf_fits
    est = always.estimate - never.estimate
    assert est == pytest.approx(GF_TRUTH_DIFF, abs=ATOL_GF_DIFF)

    # Naive unadjusted comparison of concordant arms E[Y | A0=A1=1] -
    # E[Y | A0=A1=0] is confounded through L0/L1 (treated units have
    # systematically higher L, which raises Y). Its population bias is
    # ~+1.1 here; requiring bias > 0.6 (>> 4x its MC sd ~0.09, and >
    # 2x ATOL_GF_DIFF) proves the g-formula pass is not vacuous.
    last = (
        gf_panel.sort_values(["id", "time"])
        .groupby("id")
        .agg(A0=("A", "first"), A1=("A", "last"), Y=("Y", "last"))
    )
    naive = float(
        last.loc[(last.A0 == 1) & (last.A1 == 1), "Y"].mean()
        - last.loc[(last.A0 == 0) & (last.A1 == 0), "Y"].mean()
    )
    assert naive - GF_TRUTH_DIFF > 0.6
    assert abs(est - GF_TRUTH_DIFF) < abs(naive - GF_TRUTH_DIFF)


# --------------------------------------------------------------------------- #
#  Regime DSL identities
# --------------------------------------------------------------------------- #


def test_always_and_never_regime_vectors_are_exact():
    always = sp.always_treat(K=K_IPW)
    never = sp.never_treat(K=K_IPW)
    assert always.kind == "static" and never.kind == "static"
    assert always.rule == [1.0] * K_IPW
    assert never.rule == [0.0] * K_IPW
    # Per-period difference is exactly 1, so under the linear DGP above the
    # implied always-vs-never contrast is exactly K*tau — the value asserted
    # in the recovery tests.
    for t in range(K_IPW + 2):  # +2 checks clamping beyond horizon
        assert always.treatment({}, t) - never.treatment({}, t) == 1.0


def test_regime_static_sequence_evaluates_per_period():
    reg = sp.regime([1, 0, 1])
    assert reg.kind == "static"
    assert [reg.treatment({}, t) for t in range(3)] == [1.0, 0.0, 1.0]
    # Beyond the horizon the last value is held (documented clamping).
    assert reg.treatment({}, 5) == 1.0


def test_dynamic_regime_string_reproduces_threshold_rule():
    reg = sp.regime("if L < 0 then 1 else 0")
    assert reg.kind == "dynamic"
    rng = np.random.default_rng(202607)
    hist = pd.DataFrame({"L": rng.normal(0.0, 1.0, 200)})
    produced = np.array([reg.treatment(row, 0) for row in hist.to_dict("records")])
    expected = (hist["L"].to_numpy() < 0).astype(int)
    # Deterministic rule evaluation: must match the intended treatment
    # vector element-for-element (exact identity, no tolerance).
    np.testing.assert_array_equal(produced, expected)
