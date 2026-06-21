"""Reference parity: ``sp.tmle`` (Targeted Maximum Likelihood Estimation).

Anchors
-------
A. **Saturated nonparametric collapse** — one binary covariate with a
   saturated (unpenalised) logistic propensity model.  In that case the
   TMLE plug-in equals the hand-computed stratified ATE *exactly*
   (van der Laan-Rubin 2006 targeted-MLE collapse; derivation in
   ``TestSaturatedCollapse``).  Machine-grade tolerance 1e-9.
B. **EIF zero** — the defining targeting property: the empirical mean
   of the efficient influence curve at the TMLE estimate is ~0.  The
   nuisances are reconstructed *independently* with plain sklearn fits
   (identical hyperparameters, deterministic lbfgs) and the publicly
   reported ``model_info['epsilon']``; no statspai private state is
   touched.
C. **Frozen-R parity** — base-R ``stats::glm`` TMLE
   (``_fixtures/_generate_tmle.R``) on the frozen
   ``_fixtures/tmle_data.csv``; psi/SE/epsilon pinned to 1e-9 / 1e-8.
D. **Cross-estimator** — TMLE vs ``sp.aipw`` and vs
   ``sp.g_computation`` on the same confounded DGP, within
   4*sqrt(SE_A^2 + SE_B^2) (suite convention, REFERENCES.md).
E. **Recovery + bias contrast** — naive difference-in-means is > 4
   sigma biased while TMLE recovers truth within 4 sigma; continuous
   and binary (risk-difference) outcome variants.
F. **Propensity-bounds invariance** — on a well-overlapped DGP the
   estimate is invariant to ``propensity_bounds``.
G. **SE sanity** — the influence-curve SE is within 3x of the
   empirical SD over 30 Monte-Carlo replications.

Implementation facts the anchors rely on (cited file:line)
-----------------------------------------------------------
- ``src/statspai/tmle/tmle.py:237-251`` — the outcome model is a
  *single* SuperLearner fit on the stacked design ``[A, W]`` (not
  per-arm Q fitting); counterfactual predictions set the A column
  to 1/0.
- ``src/statspai/tmle/super_learner.py:229-234, 259-270`` — SuperLearner
  *refits every learner on the full sample* and ``predict`` uses those
  full-data fits.  Predictions are therefore NOT cross-fitted
  (no CV-TMLE); ``n_folds`` only drives the ensemble-weight CV
  (super_learner.py:141-201) and with a single-learner library the
  simplex weight is identically [1.0], so ``n_folds`` cannot affect
  the estimate (it must still be >= 2 for sklearn's KFold).
- ``src/statspai/tmle/tmle.py:220`` — binary outcome detected via
  unique values ⊆ {0, 1}; ``tmle.py:223-229, 334-338`` — continuous Y
  is affinely mapped to [0,1] and back (exact bijection), so all
  identities below carry to the original scale.
- ``src/statspai/tmle/tmle.py:254-257`` — Q̄ clipped to
  [1e-5, 1-1e-5]; ``super_learner.py:277-279`` — classification
  ensemble predictions clipped to [1e-6, 1-1e-6]; ``tmle.py:267-269``
  — g clipped to ``propensity_bounds`` (default (0.025, 0.975)),
  raw-clip counts exposed in
  ``model_info['propensity_diagnostics']`` (tmle.py:275-287, 387).
- ``src/statspai/tmle/tmle.py:308-311`` — ATE clever covariate
  H_A = A/g - (1-A)/(1-g), H_1 = 1/g, H_0 = -1/(1-g).
- ``src/statspai/tmle/tmle.py:416-441`` — epsilon solves the
  one-dimensional fluctuation score sum(H*(Y - expit(logit Q̄ + eps*H)))
  by Newton-Raphson with stopping rule |delta| < 1e-8.
- ``src/statspai/tmle/tmle.py:344-353, 368`` — psi = mean(Q*1 - Q*0);
  EIF = (Q*1 - Q*0) + A(Y-Q*A)/g - (1-A)(Y-Q*A)/(1-g) - psi;
  SE = std(EIF, ddof=1)/sqrt(n).
- ``src/statspai/tmle/tmle.py:380-396`` — ``model_info`` publicly
  exposes ``epsilon``, ``Q_star_1_mean``, ``Q_star_0_mean``.

Fixture lifecycle
-----------------
``_fixtures/tmle_data.csv`` is the deterministic seed=7321 DGP from
``_fixtures/_generate_tmle_data.py``.  The frozen base-R reference
``_fixtures/tmle_R.json`` is produced by
``Rscript tests/reference_parity/_fixtures/_generate_tmle.R``
(stats::glm only — no CRAN packages).  Re-run the generators only when
the DGP itself changes; otherwise the fixtures are the contract.

References (bib keys verified in paper.bib)
-------------------------------------------
- van der Laan & Rubin (2006), Targeted Maximum Likelihood Learning,
  *Int. J. Biostatistics* 2(1). [@vanderlaan2006targeted]
- van der Laan & Rose (2011), *Targeted Learning*, Springer.
  [@vanderlaan2011targeted]
- Robins, Rotnitzky & Zhao (1994), JASA 89(427) — AIPW / efficient
  influence function. [@robins1994estimation]
- Glynn & Quinn (2010), *Political Analysis* 18(1) — AIPW estimator
  cross-checked in anchor D. [@glynn2010introduction]
- Robins (1986), *Mathematical Modelling* 7 — g-computation
  cross-checked in anchor D. [@robins1986new]
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression, LogisticRegression

import statspai as sp

_FIXTURE_DIR = pathlib.Path(__file__).parent / "_fixtures"


def _within_n_se(est, truth, se, n_sigma=4.0):
    return abs(est - truth) <= n_sigma * se


def _logreg():
    """Unpenalised logistic MLE — deterministic given the data.

    penalty=None makes sklearn match the exact (unregularised) MLE that
    base-R stats::glm computes; lbfgs with tol=1e-12 is deterministic,
    so two fits on the same data are bit-identical (needed by the
    independent EIF reconstruction in anchor B).
    """
    return LogisticRegression(penalty=None, solver="lbfgs", tol=1e-12, max_iter=10000)


# ---------------------------------------------------------------------------
# Module fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def saturated_binary_data():
    """One binary covariate, large balanced cells, binary outcome.

    w ~ Bernoulli(0.5); P(A=1|w) = 0.35 + 0.30 w  (g in {0.35, 0.65},
    far inside the default (0.025, 0.975) truncation so no clipping);
    P(Y=1|A,w) = 0.20 + 0.30 A + 0.20 w.  n = 4000 keeps every one of
    the 4 (A, w) cells > 600 observations.
    """
    rng = np.random.default_rng(11)
    n = 4000
    w = rng.binomial(1, 0.5, n)
    a = rng.binomial(1, 0.35 + 0.30 * w)
    yb = rng.binomial(1, 0.20 + 0.30 * a + 0.20 * w)
    df = pd.DataFrame(
        {"y": yb.astype(float), "a": a.astype(float), "w": w.astype(float)}
    )
    return df


@pytest.fixture(scope="module")
def tmle_fixture_data():
    """Frozen seed=7321 DGP — the same CSV the base-R generator reads."""
    return pd.read_csv(_FIXTURE_DIR / "tmle_data.csv")


@pytest.fixture(scope="module")
def tmle_fixture_truth(tmle_fixture_data):
    """Sample-average risk difference, recomputed from the CSV columns.

    Hand-rolled from the DGP's potential-outcome probabilities
    (see _generate_tmle_data.py):
        truth = mean[ expit(-0.4 + 0.8 + 0.9 w1 + 0.5 w2)
                      - expit(-0.4 + 0.9 w1 + 0.5 w2) ]
    No statspai code involved.
    """
    w1 = tmle_fixture_data["w1"].values
    w2 = tmle_fixture_data["w2"].values
    return float(
        np.mean(
            expit(-0.4 + 0.8 + 0.9 * w1 + 0.5 * w2) - expit(-0.4 + 0.9 * w1 + 0.5 * w2)
        )
    )


@pytest.fixture(scope="module")
def r_reference():
    """Frozen base-R TMLE output; skip (gracefully) if absent."""
    path = _FIXTURE_DIR / "tmle_R.json"
    if not path.exists():
        pytest.skip("tmle_R.json missing — run _generate_tmle.R")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def tmle_on_fixture(tmle_fixture_data):
    """sp.tmle on the frozen CSV with single-learner glm-equivalent
    libraries — shared by the frozen-R, EIF-zero and binary-recovery
    anchors (the call is deterministic, so sharing is sound)."""
    return sp.tmle(
        tmle_fixture_data,
        y="y",
        treat="a",
        covariates=["w1", "w2"],
        outcome_library=[_logreg()],
        propensity_library=[_logreg()],
        n_folds=2,  # irrelevant to the estimate with a 1-learner library
        # (super_learner.py:229-234 refits on full data);
        # 2 is the sklearn KFold minimum.
    )


# ---------------------------------------------------------------------------
# A) Saturated nonparametric collapse (closed form)
# ---------------------------------------------------------------------------


class TestSaturatedCollapse:
    """TMLE = stratified estimator when the propensity model is saturated.

    Derivation of exactness (van der Laan-Rubin 2006 collapse):

    1. With one binary covariate w, the logistic model g(w) =
       expit(b0 + b1 w) is *saturated* (2 parameters, 2 strata), so the
       unpenalised MLE reproduces the within-stratum treated shares
       exactly: g(w) = n_{1w} / n_w  (up to lbfgs tol=1e-12).
    2. For ANY outcome fit Q that is constant within (A, w) cells —
       true for every prediction function evaluated at the same input —
       the AIPW functional with this saturated g collapses cell by cell:
       sum over stratum w of [Q1(w)-Q0(w)] + n_w(Ȳ_{1w}-Q1(w))
       - n_w(Ȳ_{0w}-Q0(w)) = n_w(Ȳ_{1w}-Ȳ_{0w}) — the Q terms cancel
       exactly, leaving the stratified estimator.
    3. The TMLE targeting step (tmle.py:416-441) solves
       mean(H_A (Y - Q*_A)) = 0, so psi_TMLE = mean(Q*1-Q*0)
       = AIPW(Q*, g) = stratified ATE, up to the Newton stopping rule
       |delta| < 1e-8 (quadratic convergence makes the realised score
       residual ~1e-13 or smaller).

    Because SuperLearner predictions are full-data refits (no
    cross-fitting — super_learner.py:229-234), the collapse is exact:
    calibrated over 5 seeds (11-15, this DGP), the worst observed
    |TMLE - stratified| was 7.1e-13.  Tolerance 1e-9 leaves >1000x
    slack yet is 10x tighter than the 1e-8 machine-collapse bar.
    """

    TOL = 1e-9

    def test_tmle_equals_stratified_ate(self, saturated_binary_data):
        df = saturated_binary_data
        n = len(df)

        # Hand-computed stratified ATE with pandas (independent of
        # statspai): sum_w (n_w/n) * (Ȳ_{1w} - Ȳ_{0w}).
        strat = 0.0
        for wv in (0.0, 1.0):
            sub = df[df["w"] == wv]
            strat += (len(sub) / n) * (
                sub.loc[sub["a"] == 1, "y"].mean() - sub.loc[sub["a"] == 0, "y"].mean()
            )

        r = sp.tmle(
            df,
            y="y",
            treat="a",
            covariates=["w"],
            outcome_library=[_logreg()],
            propensity_library=[_logreg()],
            n_folds=2,
        )
        assert abs(r.estimate - strat) < self.TOL, (
            f"saturated TMLE {r.estimate:.12f} != stratified ATE "
            f"{strat:.12f} (|diff|={abs(r.estimate - strat):.2e}); "
            f"the van der Laan-Rubin collapse is an algebraic identity "
            f"here — any drift beyond {self.TOL} is a numerical bug."
        )

    def test_plug_in_identity(self, saturated_binary_data):
        """estimate == mean(Q*1) - mean(Q*0) — the plug-in form.

        tmle.py:345 computes psi = mean(Q*1 - Q*0) and tmle.py:390-391
        stores the two means; the difference of means equals the mean
        of differences up to float associativity (~1e-16), so 1e-12 is
        a pure machine-identity tolerance.
        """
        r = sp.tmle(
            saturated_binary_data,
            y="y",
            treat="a",
            covariates=["w"],
            outcome_library=[_logreg()],
            propensity_library=[_logreg()],
            n_folds=2,
        )
        plug_in = r.model_info["Q_star_1_mean"] - r.model_info["Q_star_0_mean"]
        assert abs(r.estimate - plug_in) < 1e-12, (
            f"TMLE estimate {r.estimate:.15f} is not the plug-in "
            f"mean(Q*1)-mean(Q*0)={plug_in:.15f} — the targeting "
            f"plug-in property is broken."
        )


# ---------------------------------------------------------------------------
# B) EIF zero (targeting property), via independent reconstruction
# ---------------------------------------------------------------------------


class TestEIFZero:
    """Mean of the efficient influence curve at the TMLE estimate is ~0.

    The CausalResult does NOT expose per-unit nuisance fits (only
    summary scalars in model_info), so instead of reaching into private
    state we reconstruct the nuisances *independently*: the single-
    learner libraries make SuperLearner predictions equal to one plain
    sklearn fit on the full data (super_learner.py:229-234 + trivial
    simplex weight [1.0]), and LogisticRegression with fixed
    hyperparameters is deterministic, so a second fit on the same data
    is bit-identical.  Only the publicly reported
    ``model_info['epsilon']`` links the reconstruction to the result.
    """

    def _reconstruct(self, df, eps):
        Y = df["y"].values
        A = df["a"].values
        W = df[["w1", "w2"]].values
        n = len(Y)
        AW = np.column_stack([A, W])

        q_fit = _logreg().fit(AW, Y)
        g_fit = _logreg().fit(W, A)

        # Replicate the documented clipping chain: SuperLearner
        # classification predict clips to [1e-6, 1-1e-6]
        # (super_learner.py:277-279), then tmle clips Q̄ to
        # [1e-5, 1-1e-5] (tmle.py:254-257) — composition = clip at
        # 1e-5; g additionally truncated to the default
        # propensity_bounds (0.025, 0.975) (tmle.py:267-269).  All
        # clips are no-ops on this DGP (probabilities interior) but we
        # replicate them so the reconstruction is the exact algorithm.
        def clip_q(p):
            return np.clip(np.clip(p, 1e-6, 1 - 1e-6), 1e-5, 1 - 1e-5)

        Qb_A = clip_q(q_fit.predict_proba(AW)[:, 1])
        Qb_1 = clip_q(q_fit.predict_proba(np.column_stack([np.ones(n), W]))[:, 1])
        Qb_0 = clip_q(q_fit.predict_proba(np.column_stack([np.zeros(n), W]))[:, 1])
        g = np.clip(np.clip(g_fit.predict_proba(W)[:, 1], 1e-6, 1 - 1e-6), 0.025, 0.975)

        H_A = A / g - (1 - A) / (1 - g)  # tmle.py:308-311
        Qs_A = expit(logit(Qb_A) + eps * H_A)
        Qs_1 = expit(logit(Qb_1) + eps / g)
        Qs_0 = expit(logit(Qb_0) - eps / (1 - g))
        return Y, A, g, H_A, Qs_A, Qs_1, Qs_0

    def test_mean_eif_is_zero_at_tmle(self, tmle_fixture_data, tmle_on_fixture):
        r = tmle_on_fixture
        eps = r.model_info["epsilon"]
        Y, A, g, H_A, Qs_A, Qs_1, Qs_0 = self._reconstruct(tmle_fixture_data, eps)
        n = len(Y)

        # Reconstruction fidelity: the plug-in from OUR nuisances must
        # equal the reported estimate.  Bit-identical sklearn refits
        # imply equality to float roundoff; observed 0.0, tolerance
        # 1e-10 guards against platform-level BLAS noise.
        psi_mine = float(np.mean(Qs_1 - Qs_0))
        assert abs(psi_mine - r.estimate) < 1e-10, (
            f"independent nuisance reconstruction gives plug-in "
            f"{psi_mine:.12f} != reported {r.estimate:.12f} — the "
            f"reconstruction no longer mirrors the algorithm."
        )

        # Defining targeting property: mean(EIF) = mean(H_A(Y - Q*_A))
        # (the plug-in terms cancel psi exactly).  Newton stops at
        # |delta| < 1e-8 (tmle.py:440), i.e. |score|/n <=
        # 1e-8 * mean(H^2 p(1-p)) ≈ 9.3e-9 on this DGP — so 1e-8 is the
        # stopping-rule-derived bound; quadratic convergence makes the
        # realised value ~1e-17.
        eif = (
            (Qs_1 - Qs_0)
            + A * (Y - Qs_A) / g
            - (1 - A) * (Y - Qs_A) / (1 - g)
            - r.estimate
        )
        assert abs(float(np.mean(eif))) < 1e-8, (
            f"mean(EIF) = {np.mean(eif):.3e} at the TMLE estimate — "
            f"targeting (the defining TMLE property) failed."
        )

        # SE identity: reported SE must be sd(EIF, ddof=1)/sqrt(n)
        # (tmle.py:368).  Same float ops on bit-identical inputs;
        # observed 0.0, tolerance 1e-10.
        se_mine = float(np.std(eif, ddof=1) / np.sqrt(n))
        assert abs(se_mine - r.se) < 1e-10, (
            f"IC-based SE {r.se:.12f} != independent reconstruction " f"{se_mine:.12f}."
        )


# ---------------------------------------------------------------------------
# C) Frozen base-R parity (stats::glm TMLE)
# ---------------------------------------------------------------------------


class TestFrozenRParity:
    """sp.tmle vs a base-R (stats::glm) TMLE on the frozen CSV.

    Both sides fit the identical unpenalised logistic MLEs (sklearn
    lbfgs tol=1e-12 vs R IRLS epsilon=1e-12) and solve the same 1-D
    fluctuation MLE, so agreement is limited only by optimiser
    tolerance.  Observed: |Δpsi| = 5.6e-12, |Δse| = 1.3e-12,
    |Δepsilon| = 5.8e-11.  Tolerances 1e-9 (psi, se) / 1e-8 (epsilon)
    leave ~100x slack while staying machine-grade.
    """

    def test_psi_matches_R(self, tmle_on_fixture, r_reference):
        r_psi = r_reference["ate"]["psi"]
        assert abs(tmle_on_fixture.estimate - r_psi) < 1e-9, (
            f"psi: Python {tmle_on_fixture.estimate:.12f} vs base-R "
            f"{r_psi:.12f} (|diff|={abs(tmle_on_fixture.estimate - r_psi):.2e})"
        )

    def test_se_matches_R(self, tmle_on_fixture, r_reference):
        r_se = r_reference["ate"]["se"]
        assert abs(tmle_on_fixture.se - r_se) < 1e-9, (
            f"EIF SE: Python {tmle_on_fixture.se:.12f} vs base-R " f"{r_se:.12f}"
        )

    def test_epsilon_matches_R(self, tmle_on_fixture, r_reference):
        r_eps = r_reference["ate"]["epsilon"]
        py_eps = tmle_on_fixture.model_info["epsilon"]
        assert abs(py_eps - r_eps) < 1e-8, (
            f"fluctuation epsilon: Python {py_eps:.6e} vs base-R "
            f"{r_eps:.6e} — both solve the same offset-logistic MLE."
        )

    def test_fixture_csv_intact(self, tmle_fixture_data):
        """Guard that the CSV fixture wasn't accidentally mutated."""
        assert len(tmle_fixture_data) == 2000
        assert list(tmle_fixture_data.columns) == ["y", "a", "w1", "w2"]
        assert set(np.unique(tmle_fixture_data["a"])) == {0.0, 1.0}
        assert set(np.unique(tmle_fixture_data["y"])) == {0.0, 1.0}

    def test_fixture_R_meta_present(self, r_reference):
        """Guard that the R fixture has reproducibility metadata."""
        assert "meta" in r_reference
        assert r_reference["meta"]["n"] == 2000
        assert r_reference["meta"]["propensity_bounds"] == [0.025, 0.975]
        # No truncation on this DGP — if this flips, the DGP changed.
        assert r_reference["ate"]["n_g_truncated"] == 0


# ---------------------------------------------------------------------------
# D) Cross-estimator parity (combined SE)
# ---------------------------------------------------------------------------


class TestCrossEstimatorParity:
    """TMLE vs AIPW vs g-computation on the CIA DGP (truth = 2.0).

    matching_cia_data (conftest.py) has a linear outcome in (d, X) and
    a logistic propensity in X — so LinearRegression Q and logistic g
    are both correctly specified and all three doubly-robust /
    plug-in estimators target the same ATE.  sp.aipw uses statsmodels
    nuisances with cross-fitting (inference/aipw.py:113-137) and
    sp.g_computation uses statsmodels OLS with bootstrap SE
    (inference/g_computation.py:162-199) — genuinely independent code
    paths from sp.tmle's sklearn SuperLearner.
    """

    @pytest.fixture(scope="class")
    def tmle_cia(self, matching_cia_data):
        return sp.tmle(
            matching_cia_data,
            y="y",
            treat="d",
            covariates=["X1", "X2", "X3"],
            outcome_library=[LinearRegression()],
            propensity_library=[_logreg()],
            n_folds=2,
        )

    def test_tmle_vs_aipw(self, matching_cia_data, tmle_cia):
        r_a = sp.aipw(
            matching_cia_data,
            y="y",
            treat="d",
            covariates=["X1", "X2", "X3"],
            n_folds=5,
            seed=123,
        )
        # Suite convention: |A - B| <= 4*sqrt(SE_A^2 + SE_B^2)
        # (REFERENCES.md, cross-estimator parity).  Observed
        # |diff| = 0.0006 vs allowed 0.18.
        comb = 4.0 * np.sqrt(tmle_cia.se**2 + r_a.se**2)
        assert abs(tmle_cia.estimate - r_a.estimate) <= comb, (
            f"TMLE {tmle_cia.estimate:.4f} vs AIPW {r_a.estimate:.4f} "
            f"diverge beyond 4*combined-SE ({comb:.4f})."
        )

    def test_tmle_vs_g_computation(self, matching_cia_data, tmle_cia):
        # n_boot=200 (not the 500 default) keeps runtime low; the
        # bootstrap SE enters only the tolerance, where 200 reps give
        # ~5% SE-of-SE — negligible against the 4-sigma band.
        r_g = sp.g_computation(
            matching_cia_data,
            y="y",
            treat="d",
            covariates=["X1", "X2", "X3"],
            n_boot=200,
            seed=123,
        )
        comb = 4.0 * np.sqrt(tmle_cia.se**2 + r_g.se**2)
        assert abs(tmle_cia.estimate - r_g.estimate) <= comb, (
            f"TMLE {tmle_cia.estimate:.4f} vs g-computation "
            f"{r_g.estimate:.4f} diverge beyond 4*combined-SE "
            f"({comb:.4f})."
        )


# ---------------------------------------------------------------------------
# E) Recovery + bias contrast (4-sigma)
# ---------------------------------------------------------------------------


class TestRecoveryAndBiasContrast:
    """Naive diff-in-means is > 4 sigma biased; TMLE recovers truth."""

    def test_continuous_naive_is_biased(self, matching_cia_data):
        """Hand-computed naive contrast on the confounded CIA DGP.

        Welch SE from plain pandas/numpy.  Observed bias z ≈ 16 —
        comfortably past the 4-sigma bar this suite uses to declare an
        estimator biased.
        """
        df = matching_cia_data
        truth = df.attrs["true_effect"]
        y1 = df.loc[df["d"] == 1, "y"]
        y0 = df.loc[df["d"] == 0, "y"]
        naive = y1.mean() - y0.mean()
        se_naive = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
        assert abs(naive - truth) > 4.0 * se_naive, (
            f"naive {naive:.3f} should be > 4 sigma from truth {truth} "
            f"(SE {se_naive:.4f}) — if not, the DGP lost its "
            f"confounding and the recovery test below is vacuous."
        )

    def test_continuous_tmle_recovers(self, matching_cia_data):
        truth = matching_cia_data.attrs["true_effect"]
        r = sp.tmle(
            matching_cia_data,
            y="y",
            treat="d",
            covariates=["X1", "X2", "X3"],
            outcome_library=[LinearRegression()],
            propensity_library=[_logreg()],
            n_folds=2,
        )
        # 4-sigma recovery (REFERENCES.md convention; false-failure
        # probability 6.3e-5).  Observed z ≈ 1.2.
        assert _within_n_se(r.estimate, truth, r.se, n_sigma=4.0), (
            f"TMLE {r.estimate:.4f} (SE {r.se:.4f}) misses truth "
            f"{truth} by {abs(r.estimate - truth) / r.se:.1f} sigma."
        )

    def test_binary_naive_is_biased(self, tmle_fixture_data, tmle_fixture_truth):
        """Binary outcome: naive risk difference vs the known SATE.

        Two-proportion SE, hand-computed.  Observed bias z ≈ 6.0 on
        the frozen draw.
        """
        df = tmle_fixture_data
        p1 = df.loc[df["a"] == 1, "y"].mean()
        p0 = df.loc[df["a"] == 0, "y"].mean()
        n1 = int((df["a"] == 1).sum())
        n0 = int((df["a"] == 0).sum())
        naive = p1 - p0
        se_naive = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
        assert abs(naive - tmle_fixture_truth) > 4.0 * se_naive, (
            f"naive RD {naive:.4f} should be > 4 sigma from truth "
            f"{tmle_fixture_truth:.4f} (SE {se_naive:.4f})."
        )

    def test_binary_tmle_recovers(self, tmle_on_fixture, tmle_fixture_truth):
        # 4-sigma recovery; the logistic Q model matches the DGP's
        # outcome link exactly, so TMLE is unbiased here.  Observed
        # z ≈ 1.4.
        r = tmle_on_fixture
        assert _within_n_se(r.estimate, tmle_fixture_truth, r.se, n_sigma=4.0), (
            f"TMLE RD {r.estimate:.4f} (SE {r.se:.4f}) misses truth "
            f"{tmle_fixture_truth:.4f} by "
            f"{abs(r.estimate - tmle_fixture_truth) / r.se:.1f} sigma."
        )

    def test_sign_correctness(self, tmle_on_fixture):
        """Positive-effect DGP must produce a positive estimate
        (suite-standard sign smoke test, REFERENCES.md)."""
        assert tmle_on_fixture.estimate > 0


# ---------------------------------------------------------------------------
# F) Propensity-bounds invariance
# ---------------------------------------------------------------------------


class TestPropensityBoundsInvariance:
    """On a well-overlapped DGP, propensity truncation is inert."""

    def test_bounds_invariance(self, matching_cia_data):
        kwargs = dict(
            y="y",
            treat="d",
            covariates=["X1", "X2", "X3"],
            outcome_library=[LinearRegression()],
            propensity_library=[_logreg()],
            n_folds=2,
        )
        r_wide = sp.tmle(matching_cia_data, propensity_bounds=(0.025, 0.975), **kwargs)
        r_narrow = sp.tmle(
            matching_cia_data, propensity_bounds=(0.001, 0.999), **kwargs
        )

        # Headline criterion: < 4-sigma movement (combined SE).
        comb = 4.0 * np.sqrt(r_wide.se**2 + r_narrow.se**2)
        diff = abs(r_wide.estimate - r_narrow.estimate)
        assert diff <= comb, (
            f"propensity_bounds moved the estimate by {diff:.4f} "
            f"(> 4*combined-SE {comb:.4f}) despite good overlap."
        )

        # Stronger: estimated propensities live in [0.06, 0.93] on this
        # DGP, so NEITHER bound clips (verified via the public clip
        # diagnostics, tmle.py:275-287).  When no score is clipped the
        # two runs perform identical arithmetic, so the estimates are
        # machine-equal (observed diff = 0.0; 1e-12 guards roundoff).
        def n_clipped(r):
            d = r.model_info["propensity_diagnostics"]
            return d["n_clipped_below"] + d["n_clipped_above"]

        if n_clipped(r_wide) == 0 and n_clipped(r_narrow) == 0:
            assert diff < 1e-12, (
                f"no propensity was clipped under either bound yet the "
                f"estimates differ by {diff:.2e} — truncation leaked "
                f"into the arithmetic."
            )


# ---------------------------------------------------------------------------
# G) SE sanity: IC-based SE vs Monte-Carlo SD
# ---------------------------------------------------------------------------


class TestSESanity:
    """Influence-curve SE within 3x of the empirical SD (30 MC reps).

    n=500 per rep, single-learner libraries; with 30 reps the SD
    estimate has ~13% relative noise, so the 3x band has enormous
    slack (observed ratio 0.95).  Catches order-of-magnitude SE bugs
    (wrong n scaling, missing ddof, variance-of-IF mistakes).
    """

    N_REPS = 30
    N = 500
    TRUE_ATE = 1.0

    def test_ic_se_within_3x_of_mc_sd(self):
        ests, ses = [], []
        for rep in range(self.N_REPS):
            rng = np.random.default_rng(9000 + rep)
            x1 = rng.normal(size=self.N)
            x2 = rng.normal(size=self.N)
            a = rng.binomial(1, expit(0.3 * x1 - 0.3 * x2))
            y = (
                1.0
                + self.TRUE_ATE * a
                + 0.8 * x1
                + 0.5 * x2
                + rng.normal(scale=1.0, size=self.N)
            )
            df = pd.DataFrame({"y": y, "a": a.astype(float), "x1": x1, "x2": x2})
            r = sp.tmle(
                df,
                y="y",
                treat="a",
                covariates=["x1", "x2"],
                outcome_library=[LinearRegression()],
                propensity_library=[_logreg()],
                n_folds=2,
            )
            ests.append(r.estimate)
            ses.append(r.se)

        emp_sd = float(np.std(ests, ddof=1))
        mean_se = float(np.mean(ses))
        assert emp_sd / 3.0 <= mean_se <= 3.0 * emp_sd, (
            f"IC-based SE (mean {mean_se:.4f}) is not within 3x of the "
            f"Monte-Carlo SD ({emp_sd:.4f}) over {self.N_REPS} reps — "
            f"the influence-function variance is miscalibrated."
        )

        # MC-mean recovery: SE of the mean over 30 reps is
        # emp_sd/sqrt(30); 4-sigma band per suite convention.
        mc_mean = float(np.mean(ests))
        assert abs(mc_mean - self.TRUE_ATE) <= 4.0 * emp_sd / np.sqrt(self.N_REPS), (
            f"MC mean {mc_mean:.4f} drifted from truth "
            f"{self.TRUE_ATE} — systematic bias across replications."
        )
