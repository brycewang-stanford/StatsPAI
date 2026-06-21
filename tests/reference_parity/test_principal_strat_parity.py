"""Reference parity: principal stratification family.

Estimators
----------
``sp.principal_strat`` (monotonicity / AIR-Wald-LATE / principal-score
dispatcher) and ``sp.survivor_average_causal_effect`` (Zhang-Rubin sharp
SACE bounds).  Each previously had only a smoke test; this file is their
first numerical guarantee — every assert pins an estimate to a hand-set
truth or an exact algebraic identity, never to mere finiteness.

Setting (principal stratification under monotonicity)
-----------------------------------------------------
Binary treatment ``D``, binary post-treatment variable ``S`` (compliance
/ survival), outcome ``Y``.  Units are classified by the latent pair
``(S(0), S(1))`` into always-takers/-survivors ``(1, 1)``, compliers
``(0, 1)``, never-takers/-survivors ``(0, 0)`` — defiers ``(1, 0)`` are
ruled out by AIR monotonicity ``S(1) >= S(0)``.  The observable mixtures
``P(S=1|D=1) = pi_always + pi_complier`` and ``P(S=1|D=0) = pi_always``
point-identify the three stratum shares and the complier PCE; the
always-survivor SACE is partially identified by Zhang-Rubin bounds.

When a randomly assigned binary instrument ``Z`` is supplied, the
function switches to the Angrist-Imbens-Rubin (1996) encouragement
design: ``D`` becomes the endogenous uptake and the Wald estimators
recover the LATEs among ``Z``-compliers,
``tau_Y = (E[Y|Z=1]-E[Y|Z=0]) / (P(D=1|Z=1)-P(D=1|Z=0))``.

Anchors
-------
A. **Known-DGP IV (AIR) recovery + complier-share recovery** (recovery).
   A monotonicity (no-defiers) encouragement DGP with hand-set strata
   shares (always-takers 0.2, never-takers 0.3, compliers 0.5) and a
   hand-set complier LATE ``TRUE_LATE = 2.0``.  ``sp.principal_strat``
   with ``instrument=`` recovers ``TRUE_LATE`` within 4 of its bootstrap
   SE (probed z ~0.03) AND recovers the complier share 0.5 from the
   first stage ``P(D=1|Z=1)-P(D=1|Z=0)`` within 4 two-proportion sigma
   (probed z ~0.7).  Non-tautological: a +20% bias on ``tau_Y``
   (-> 2.4) lands ~8 sigma out.
B. **Closed-form saturated collapse** (closed_form).  (i) The
   monotonicity complier LATE equals the hand-computed Wald-mixture
   ``(mu_11*p11 - mu_01*p10)/(p11-p10)`` of the *sample* cell means to
   machine precision (probed |diff| = 0.0) — an exact algebraic identity
   of the estimator.  (ii) Perfect compliance ``S == D`` forces
   ``pi_complier = 1``, ``pi_always = pi_never = 0`` and collapses the
   LATE to ``E[Y | D=1, S=1] = mean(Y | D=1)`` (probed |diff| = 0.0).
C. **Naive-bias contrast on the SACE** (naive_bias).  A survivor-bias /
   truncation-by-death DGP: treatment additionally rescues a "protected"
   stratum whose outcomes sit *inside* the always-survivor outcome
   range, so the (D=1, S=1) cell is contaminated.  The naive survivor
   comparison ``E[Y|D=1,S=1]-E[Y|D=0,S=1]`` is > 4 two-proportion sigma
   biased away from the true ``SACE = 1.0`` (probed z ~-8), while the
   Zhang-Rubin sharp bounds returned by
   ``sp.survivor_average_causal_effect`` strictly bracket the truth
   (probed [-0.15, 1.71] ∋ 1.0).  Assert BOTH.
D. **Cross-method / internal consistency** (consistency).  (i) The SACE
   ``estimate`` equals the midpoint of the bounds endpoints exactly
   (principal_strat.py:894; probed |diff| = 0.0), and ``model_info``'s
   ``sace_lower <= sace_upper``.  (ii) The three monotonicity stratum
   proportions sum to 1 exactly and recover the hand-set 0.4/0.3/0.3
   shares (complier within 4 two-proportion sigma, probed z ~1.0).
E. **Orientation / determinism** (orientation).  The positive-LATE IV
   DGP yields a positive ``tau_Y``; the SACE point bounds carry no
   bootstrap noise (principal_strat.py:325-348 are deterministic given
   the sample), so two calls with the *same* seed produce bitwise-equal
   endpoints — and the bounds are an interval, ``lower <= upper``.

Implementation facts the anchors rely on (cited file:line)
----------------------------------------------------------
- ``src/statspai/principal_strat/principal_strat.py:293-310`` —
  monotonicity ``pi_complier = max(P(S=1|D=1)-P(S=1|D=0), 0)``,
  ``pi_always = P(S=1|D=0)``, ``pi_never = 1-P(S=1|D=1)``, and the
  complier LATE ``tau_c = (mu_11*p11 - mu_01*p10)/pi_complier`` — the
  exact formula anchor B-i reproduces by hand.
- ``principal_strat.py:415-419`` — ``strata_proportions`` keys are
  ``'always-taker / always-survivor'`` / ``'complier'`` /
  ``'never-taker / never-survivor'``; the LATE is the single row of
  ``effects`` (``'Complier (LATE)'``).
- ``principal_strat.py:251-257, 499-510`` — supplying ``instrument``
  routes to ``_fit_instrument_air``; ``effects`` row 0 is the Wald LATE
  on ``Y`` (``tau_Y``), and ``strata_proportions['first_stage ...']``
  is ``P(D=1|Z=1)-P(D=1|Z=0)``.
- ``principal_strat.py:325-348`` — Zhang-Rubin bounds are a sorted
  q-slice (``q = pi_always/P(S=1|D=1)``) of the (D=1, S=1) outcomes
  minus ``mu_01``; the point endpoints use NO bootstrap, hence the
  determinism in anchor E.
- ``principal_strat.py:888-918`` — ``survivor_average_causal_effect``
  reads ``bounds.loc[0/1, 'estimate']`` as ``sace_lower``/``sace_upper``
  and sets the ``CausalResult.estimate`` to their midpoint (anchor D-i).

References (bib keys verified present in paper.bib via grep)
------------------------------------------------------------
- Frangakis, C.E. & Rubin, D.B. (2002), "Principal Stratification in
  Causal Inference", *Biometrics* 58(1), 21-29. [@frangakis2002principal]
- Zhang, J.L. & Rubin, D.B. (2003), "Estimation of Causal Effects via
  Principal Stratification When Some Outcomes Are Truncated by 'Death'",
  *J. Educ. Behav. Stat.* 28(4), 353-368. [@zhang2003estimation]
- Angrist, J.D., Imbens, G.W. & Rubin, D.B. (1996), "Identification of
  Causal Effects Using Instrumental Variables", *JASA* 91(434), 444-455.
  [@angrist1996identification]
- Ding, P. & Lu, J. (2017), "Principal stratification analysis using
  principal scores", *JRSS-B* 79(3), 757-777. [@ding2017principal]
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# Hand-set truths shared across DGPs (stored as module constants so a
# verifier can see exactly what each assert is pinned to).
TRUE_LATE = 2.0  # complier Wald LATE on Y in the encouragement DGP
TRUE_SACE = 1.0  # E[Y(1)-Y(0) | always-survivor] in the survivor DGP


def _within_n_se(est, truth, se, n_sigma=4.0):
    return abs(est - truth) <= n_sigma * se


# ---------------------------------------------------------------------------
# Deterministic DGP builders (every draw seeded via default_rng).
# ---------------------------------------------------------------------------


def _make_encouragement_dgp(seed, n=8000):
    """Monotonicity (no-defiers) IV DGP with KNOWN shares + complier LATE.

    Latent strata w.r.t. the instrument Z (random encouragement):
        always-takers (D=1 regardless of Z)   share 0.2
        never-takers  (D=0 regardless of Z)   share 0.3
        compliers     (D = Z)                 share 0.5
    Exclusion holds (Z enters Y only through D); the complier Wald LATE
    on Y is the hand-set ``TRUE_LATE``.  S is a monotone-in-D survival
    column (only needed so the call has a ``strata`` argument on the
    instrument path; identification there comes from Z, not S).
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n)
    # 0.2 always-takers, 0.3 never-takers, 0.5 compliers.
    typ = np.where(u < 0.2, "AT", np.where(u < 0.5, "NT", "CO"))
    Z = rng.binomial(1, 0.5, n).astype(float)
    D = np.where(typ == "AT", 1.0, np.where(typ == "NT", 0.0, Z))
    # Stratum-level intercepts (constant in D -> exclusion preserved);
    # the only D-driven term is TRUE_LATE * D.
    level = np.where(typ == "AT", 0.5, np.where(typ == "NT", -0.5, 0.0))
    Y = 1.0 + level + TRUE_LATE * D + rng.normal(size=n)
    S = (rng.uniform(size=n) < 0.3 + 0.4 * D).astype(float)
    return pd.DataFrame({"y": Y, "d": D, "s": S, "z": Z})


def _make_monotone_strata_dgp(seed, n=10000):
    """No-instrument monotonicity DGP with KNOWN stratum shares.

    always-survivors 0.3 (S=1 both arms), compliers 0.4 (S=D),
    never-survivors 0.3 (S=0 both arms).  Used for the closed-form Wald
    collapse (anchor B-i) and the stratum-share recovery (anchor D-ii).
    """
    rng = np.random.default_rng(seed)
    D = rng.binomial(1, 0.5, n).astype(float)
    u = rng.uniform(size=n)
    typ = np.where(u < 0.3, "AS", np.where(u < 0.7, "CO", "NS"))
    S = np.where(typ == "AS", 1.0, np.where(typ == "CO", D, 0.0))
    Y = 2.0 + 1.0 * S + 0.5 * D + rng.normal(size=n)
    return pd.DataFrame({"y": Y, "d": D, "s": S})


def _make_perfect_compliance_dgp(seed, n=6000):
    """S == D exactly -> everyone is a complier (pi_complier = 1).

    The complier LATE then collapses to ``E[Y | D=1, S=1]`` because the
    (D=0, S=1) cell is empty (mu_01 falls back to 0 and P(S=1|D=0)=0,
    so the second term of the Wald numerator vanishes).
    """
    rng = np.random.default_rng(seed)
    D = rng.binomial(1, 0.5, n).astype(float)
    S = D.copy()  # S(0)=0, S(1)=1 for everyone
    Y = 1.0 + 3.0 * D + rng.normal(size=n)
    return pd.DataFrame({"y": Y, "d": D, "s": S})


def _make_survivor_bias_dgp(seed, n=20000):
    """Truncation-by-death DGP where the naive survivor diff is biased.

    Strata (w.r.t. D's effect on survival S):
        always-survivors AS  share 0.5  (S=1 both arms)
        protected        PR  share 0.3  (S(0)=0, S(1)=1)  -> S = D
        doomed           DO  share 0.2  (S=0 both arms)
    Always-survivor outcomes: Y(0) ~ N(5, 1.5), Y(1) ~ N(6, 1.5), so the
    true SACE = E[Y(1)-Y(0) | AS] = ``TRUE_SACE`` = 1.0.  The protected
    stratum is observed ONLY under D=1 (it survives only when treated)
    with Y ~ N(5.5, 1.5) — sitting INSIDE the always-survivor Y(1)
    spread, so it contaminates the (D=1, S=1) cell and the naive
    survivor comparison is biased.  The wide overlapping spread lets the
    Zhang-Rubin q-slice still bracket the truth.
    """
    rng = np.random.default_rng(seed)
    D = rng.binomial(1, 0.5, n).astype(float)
    u = rng.uniform(size=n)
    typ = np.where(u < 0.5, "AS", np.where(u < 0.8, "PR", "DO"))
    S = np.where(typ == "AS", 1.0, np.where(typ == "PR", D, 0.0))
    y_as0 = 5.0 + rng.normal(scale=1.5, size=n)
    y_as1 = (5.0 + TRUE_SACE) + rng.normal(scale=1.5, size=n)  # mean 6
    y_pr1 = 5.5 + rng.normal(scale=1.5, size=n)
    Y = np.where(
        typ == "AS",
        np.where(D == 1, y_as1, y_as0),
        np.where(typ == "PR", y_pr1, 0.0 + rng.normal(scale=0.5, size=n)),
    )
    return pd.DataFrame({"y": Y, "d": D, "s": S})


# ---------------------------------------------------------------------------
# Module fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def encouragement_data():
    return _make_encouragement_dgp(20260614)


@pytest.fixture(scope="module")
def monotone_strata_data():
    return _make_monotone_strata_dgp(55)


@pytest.fixture(scope="module")
def perfect_compliance_data():
    return _make_perfect_compliance_dgp(7)


@pytest.fixture(scope="module")
def survivor_bias_data():
    return _make_survivor_bias_dgp(2024)


# ---------------------------------------------------------------------------
# A. Known-DGP IV (AIR) recovery + complier-share recovery.
# ---------------------------------------------------------------------------


class TestEncouragementRecovery:
    """sp.principal_strat(instrument=...) recovers the AIR/Wald LATE."""

    def test_wald_late_recovers_truth(self, encouragement_data):
        df = encouragement_data
        r = sp.principal_strat(
            df,
            y="y",
            treat="d",
            strata="s",
            instrument="z",
            n_boot=200,
            seed=1,
        )
        tau_y = float(r.effects.iloc[0]["estimate"])  # Wald LATE on Y
        se_y = float(r.effects.iloc[0]["se"])
        # 4-sigma recovery (REFERENCES.md convention; false-failure
        # 6.3e-5).  Probed z ~0.03.  A +20% bias (2.4) lands ~8 sigma
        # out and fails — this is NOT a finiteness check.
        assert _within_n_se(tau_y, TRUE_LATE, se_y, n_sigma=4.0), (
            f"AIR Wald LATE {tau_y:.4f} (SE {se_y:.4f}) misses truth "
            f"{TRUE_LATE} by {abs(tau_y - TRUE_LATE) / se_y:.1f} sigma."
        )

    def test_complier_share_recovers(self, encouragement_data):
        """First stage P(D=1|Z=1)-P(D=1|Z=0) recovers the 0.5 complier share.

        The hand-set complier share w.r.t. Z is 0.5.  The reported first
        stage must land within 4 two-proportion sigma of 0.5 (probed
        z ~0.7).  SE is hand-rolled (no statspai) so the band is
        independent of the estimator's own bootstrap.
        """
        df = encouragement_data
        r = sp.principal_strat(
            df,
            y="y",
            treat="d",
            strata="s",
            instrument="z",
            n_boot=50,
            seed=1,
        )
        first_stage = r.strata_proportions["first_stage (D|Z=1 - D|Z=0)"]
        # Hand-rolled two-proportion SE of P(D=1|Z=1)-P(D=1|Z=0).
        D = df["d"].values
        Z = df["z"].values
        p1 = D[Z == 1].mean()
        p0 = D[Z == 0].mean()
        n1 = int((Z == 1).sum())
        n0 = int((Z == 0).sum())
        se_fs = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
        assert abs(first_stage - 0.5) <= 4.0 * se_fs, (
            f"complier share (first stage) {first_stage:.4f} misses the "
            f"hand-set 0.5 by {abs(first_stage - 0.5) / se_fs:.1f} sigma."
        )

    def test_positive_late_yields_positive_estimate(self, encouragement_data):
        """Orientation: a strictly positive-LATE DGP -> positive tau_Y."""
        df = encouragement_data
        r = sp.principal_strat(
            df,
            y="y",
            treat="d",
            strata="s",
            instrument="z",
            n_boot=20,
            seed=1,
        )
        assert float(r.effects.iloc[0]["estimate"]) > 0


# ---------------------------------------------------------------------------
# B. Closed-form saturated collapse (machine precision).
# ---------------------------------------------------------------------------


class TestClosedFormCollapse:
    """The monotonicity LATE equals an exact hand-computed cell-mean form."""

    TOL = 1e-9

    def test_late_equals_wald_mixture_of_cell_means(self, monotone_strata_data):
        """tau_c == (mu_11*p11 - mu_01*p10)/(p11-p10) of the sample cells.

        principal_strat.py:309-310 computes exactly this ratio of sample
        cell means; recomputing it with raw numpy must agree in exact
        arithmetic.  Tolerance 1e-9: the only slack is float
        associativity in the mean/ratio (~1e-15); probed |diff| = 0.0,
        so 1e-9 has >5 orders of headroom while staying tighter than the
        1e-8 machine-collapse bar.  This pins the estimate to a specific
        hand-computed scalar — a 20% bias breaks it by construction.
        """
        df = monotone_strata_data
        r = sp.principal_strat(
            df,
            y="y",
            treat="d",
            strata="s",
            method="monotonicity",
            n_boot=20,
            seed=1,
        )
        est = float(r.effects.iloc[0]["estimate"])

        Y = df["y"].values
        D = df["d"].values
        S = df["s"].values
        p11 = S[D == 1].mean()
        p10 = S[D == 0].mean()
        mu_11 = Y[(D == 1) & (S == 1)].mean()
        mu_01 = Y[(D == 0) & (S == 1)].mean()
        tau_hand = (mu_11 * p11 - mu_01 * p10) / (p11 - p10)

        assert abs(est - tau_hand) < self.TOL, (
            f"monotonicity LATE {est!r} != hand Wald-mixture {tau_hand!r} "
            f"(|diff|={abs(est - tau_hand):.2e}); the closed-form cell-mean "
            f"identity is broken."
        )

    def test_perfect_compliance_collapses_to_treated_survivor_mean(
        self, perfect_compliance_data
    ):
        """S == D -> pi_complier = 1 and LATE = E[Y | D=1, S=1].

        With S == D every unit is a complier (pi_always = pi_never = 0),
        P(S=1|D=0) = 0 so the second Wald term drops, and the LATE
        reduces to mu_11 = mean(Y | D=1, S=1) = mean(Y | D=1).
        Tolerance 1e-9: exact-identity collapse (probed |diff| = 0.0).
        The stratum proportions must hit the degenerate (1, 0, 0) point
        exactly — a separate non-tautological pin.
        """
        df = perfect_compliance_data
        r = sp.principal_strat(
            df,
            y="y",
            treat="d",
            strata="s",
            method="monotonicity",
            n_boot=20,
            seed=1,
        )
        props = r.strata_proportions
        assert props["complier"] == pytest.approx(1.0, abs=1e-12)
        assert props["always-taker / always-survivor"] == pytest.approx(0.0, abs=1e-12)
        assert props["never-taker / never-survivor"] == pytest.approx(0.0, abs=1e-12)

        est = float(r.effects.iloc[0]["estimate"])
        Y = df["y"].values
        D = df["d"].values
        treated_survivor_mean = float(Y[D == 1].mean())  # == mean(Y|D=1,S=1)
        assert abs(est - treated_survivor_mean) < self.TOL, (
            f"perfect-compliance LATE {est!r} != E[Y|D=1,S=1] "
            f"{treated_survivor_mean!r} "
            f"(|diff|={abs(est - treated_survivor_mean):.2e})."
        )


# ---------------------------------------------------------------------------
# C. Naive-bias contrast on the SACE.
# ---------------------------------------------------------------------------


class TestSACENaiveBias:
    """Naive survivor diff is biased; Zhang-Rubin bounds bracket the truth."""

    def test_naive_biased_bounds_bracket_truth(self, survivor_bias_data):
        df = survivor_bias_data
        Y = df["y"].values
        D = df["d"].values
        S = df["s"].values

        # Naive survivor comparison E[Y|D=1,S=1]-E[Y|D=0,S=1] with a
        # hand-rolled two-proportion-style (two-sample) SE — no statspai.
        y11 = Y[(D == 1) & (S == 1)]
        y01 = Y[(D == 0) & (S == 1)]
        naive = float(y11.mean() - y01.mean())
        se_naive = float(
            np.sqrt(y11.var(ddof=1) / len(y11) + y01.var(ddof=1) / len(y01))
        )
        # The naive survivor estimator is provably biased (>4 sigma off
        # the true SACE; probed z ~-8) because the (D=1, S=1) cell mixes
        # always-survivors with the protected stratum.  If this fails the
        # survivor selection vanished and the contrast is vacuous.
        assert abs(naive - TRUE_SACE) > 4.0 * se_naive, (
            f"naive survivor diff {naive:.4f} should be > 4 sigma from "
            f"truth {TRUE_SACE} (SE {se_naive:.4f}) — survivor selection "
            f"vanished from the DGP."
        )

        res = sp.survivor_average_causal_effect(
            df,
            y="y",
            treat="d",
            survival="s",
            n_boot=100,
            seed=1,
        )
        lo = res.model_info["sace_lower"]
        hi = res.model_info["sace_upper"]
        # The Zhang-Rubin sharp bounds must STRICTLY bracket the hand-set
        # truth (probed [-0.15, 1.71] ∋ 1.0).  This is the recovery half
        # of the contrast: the partial-identification region captures the
        # truth that the naive point estimate missed.  Injecting +20% into
        # either endpoint can push the interval off 1.0 and fail.
        assert lo < TRUE_SACE < hi, (
            f"Zhang-Rubin bounds [{lo:.4f}, {hi:.4f}] do not bracket the "
            f"true SACE {TRUE_SACE}."
        )
        # And the biased naive point lies strictly inside the bounds too
        # (the bounds are sharp, so they cannot exclude an observed-data
        # consistent value).
        assert lo <= naive <= hi


# ---------------------------------------------------------------------------
# D. Cross-method / internal consistency.
# ---------------------------------------------------------------------------


class TestInternalConsistency:
    """Identities tying the SACE wrapper to the monotonicity result."""

    def test_sace_estimate_is_bounds_midpoint(self, survivor_bias_data):
        """CausalResult.estimate == (sace_lower + sace_upper) / 2 exactly.

        principal_strat.py:894 defines the SACE point estimate as the
        midpoint of the two bound endpoints; recomputing the midpoint
        from the publicly reported endpoints must agree to machine
        precision (probed |diff| = 0.0).  Also asserts lower <= upper
        (a valid interval).  Tolerance 1e-12 = pure float roundoff.
        """
        df = survivor_bias_data
        res = sp.survivor_average_causal_effect(
            df,
            y="y",
            treat="d",
            survival="s",
            n_boot=50,
            seed=3,
        )
        lo = res.model_info["sace_lower"]
        hi = res.model_info["sace_upper"]
        assert lo <= hi, f"SACE bounds inverted: lower {lo:.4f} > upper {hi:.4f}."
        midpoint = (lo + hi) / 2.0
        assert abs(res.estimate - midpoint) < 1e-12, (
            f"SACE estimate {res.estimate!r} != bounds midpoint "
            f"{midpoint!r} (|diff|={abs(res.estimate - midpoint):.2e})."
        )
        assert res.estimand == "SACE"

    def test_strata_proportions_sum_to_one_and_recover_shares(
        self, monotone_strata_data
    ):
        """Shares sum to 1 exactly and recover the hand-set 0.4/0.3/0.3.

        pi_complier = P(S=1|D=1)-P(S=1|D=0), pi_always = P(S=1|D=0),
        pi_never = 1-P(S=1|D=1); these telescope to 1 identically (probed
        |sum-1| = 0).  The complier share must recover the hand-set 0.4
        within 4 two-proportion sigma (probed z ~1.0).  SE hand-rolled.
        """
        df = monotone_strata_data
        r = sp.principal_strat(
            df,
            y="y",
            treat="d",
            strata="s",
            method="monotonicity",
            n_boot=20,
            seed=1,
        )
        props = r.strata_proportions
        pc = props["complier"]
        pa = props["always-taker / always-survivor"]
        pn = props["never-taker / never-survivor"]
        # Exact telescoping identity.
        assert (
            abs((pc + pa + pn) - 1.0) < 1e-12
        ), f"stratum proportions sum to {pc + pa + pn!r}, not 1."
        # Complier-share recovery (true 0.4); hand-rolled two-proportion SE.
        D = df["d"].values
        S = df["s"].values
        p11 = S[D == 1].mean()
        p10 = S[D == 0].mean()
        n1 = int((D == 1).sum())
        n0 = int((D == 0).sum())
        se_pc = np.sqrt(p11 * (1 - p11) / n1 + p10 * (1 - p10) / n0)
        assert abs(pc - 0.4) <= 4.0 * se_pc, (
            f"complier share {pc:.4f} misses the hand-set 0.4 by "
            f"{abs(pc - 0.4) / se_pc:.1f} sigma."
        )


# ---------------------------------------------------------------------------
# E. Determinism of the SACE point bounds.
# ---------------------------------------------------------------------------


class TestDeterminism:
    """The Zhang-Rubin point endpoints carry no bootstrap noise."""

    def test_same_seed_gives_identical_bounds(self, survivor_bias_data):
        """Two same-seed calls -> bitwise-equal SACE endpoints.

        principal_strat.py:325-348 computes the bound endpoints from the
        sorted (D=1, S=1) outcomes deterministically — the bootstrap only
        feeds the bound SEs, never the point endpoints — so two calls
        with the same seed must return identical ``sace_lower`` /
        ``sace_upper`` / ``estimate`` (probed all equal).  Pins
        reproducibility, the discovery-suite analogue of seed-stability.
        """
        df = survivor_bias_data
        kw = dict(y="y", treat="d", survival="s", n_boot=50, seed=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1 = sp.survivor_average_causal_effect(df, **kw)
            r2 = sp.survivor_average_causal_effect(df, **kw)
        assert r1.model_info["sace_lower"] == r2.model_info["sace_lower"]
        assert r1.model_info["sace_upper"] == r2.model_info["sace_upper"]
        assert r1.estimate == r2.estimate
