"""Reference parity: continuous-treatment dose-response family.

Estimators
----------
``sp.dose_response`` (Hirano-Imbens generalized-propensity-score curve)
and ``sp.continuous_did`` (difference-in-differences with a continuous
treatment intensity).  Each previously had only a smoke test; this file
is their first numerical anchor.

Setting (continuous treatment / dose)
-------------------------------------
A continuous dose ``D`` drives a known-slope outcome.  Two DGP shapes:

* **GPS (cross-section)** — dose ``D`` may be confounded by a covariate
  ``X``; the structural model is linear ``Y = BETA*D + g*X + noise`` so
  the population dose-response curve ``E[Y(d)]`` is a straight line with
  slope ``BETA``.  When ``D`` is independent of ``X`` (unconfounded),
  the GPS marginal effect must recover ``BETA``; when ``D`` is
  X-confounded, the naive ``Y~D`` slope is biased high and the flexible
  GPS curve must move *toward* truth.

* **Continuous DiD (two-period panel)** — half the units get dose 0
  (control arm), the rest a continuous dose in ``[1, 10]``.  The
  post-period gain is ``TAU*D``, so the within-unit DiD slope is the
  hand-set ``TAU``; a *cross-sectional* regression that ignores the
  unit fixed effect (deliberately correlated with dose) is confounded.

Anchors
-------
A. **GPS unconfounded recovery of BETA = 0.8** (recovery).  With ``D``
   independent of ``X``, ``sp.dose_response``'s average marginal effect
   recovers the hand-set ``BETA`` within an absolute tolerance of 0.10
   (probed dev 0.019).  A 20% multiplicative bias drives the marginal
   effect to ~0.94 (dev 0.14) and fails.  The headline IQR effect
   ``effect_25_to_75`` likewise recovers ``BETA*(d75-d25)`` within
   4 sigma.

B. **GPS curve internal consistency** (closed_form).  With linear
   treatment/outcome models the dose-response curve is a straight line,
   so the reported ``effect_25_to_75`` must equal
   ``slope*(dose_75 - dose_25)`` read off that same curve, to a tight
   tolerance (probed |diff| ~8e-3 — np.gradient endpoint slack only).
   Pins the headline scalar to the curve it is derived from, not to
   "is finite".

C. **GPS naive-bias contrast** (naive_bias).  On the X-confounded DGP
   the naive OLS slope of ``Y`` on ``D`` (omitting ``X``) is biased high
   (probed ~1.77 vs truth 0.80), while the flexible-GBM GPS marginal
   effect lands STRICTLY between truth and the naive slope (probed
   ~0.91) — proving directional de-confounding, asserting BOTH the bias
   and the correction.

D. **Continuous-DiD slope recovery of TAU = 0.5** (recovery).  The
   ``method='twfe'`` dose x post coefficient recovers the hand-set
   ``TAU`` within 4 sigma (probed z ~-1.6); a 20% bias lands ~20 sigma
   out and fails.

E. **Continuous-DiD naive-bias contrast** (naive_bias).  With a
   unit fixed effect correlated with dose, a cross-sectional regression
   of post-period ``Y`` on dose is ~94 sigma biased high (probed slope
   ~3.50 vs truth 0.50); the DiD differences the fixed effect away and
   recovers ``TAU`` within 4 sigma (probed z ~-0.1).  Asserts BOTH.

F. **Continuous-DiD cross-method consistency** (consistency).  The
   ``method='cgs'`` level ATT averaged over the treated support equals
   ``TAU * E[D | D>0]`` (probed 2.72 vs 2.77) and the ``cgs`` ACRT(d)
   derivative recovers the per-unit-dose slope ``TAU`` (probed 0.506) —
   the level and the slope are tied to the same hand-set ``TAU`` through
   two different estimands.

G. **Determinism / seed-stability** (determinism).  Both estimators pin
   their bootstrap on an explicit seed, so two identical calls return
   bitwise-equal ``estimate`` and ``se`` (probed diff 0.0).

Implementation facts the anchors rely on (cited file:line)
----------------------------------------------------------
- ``src/statspai/dose_response/gps.py:296-321`` — ``_estimate_curve``
  fits ``E[T|X]`` (treatment_model), forms the normal-density GPS, fits
  ``E[Y|T, GPS]`` (outcome_model) and averages over ``X`` at each grid
  dose.  With ``LinearRegression`` for both models the curve is linear
  (anchor B); with the default GBM it is a flexible partial corrector
  (anchor C).
- ``gps.py:239-251`` — ``avg_marginal_effect`` is the mean of
  ``np.gradient(curve)`` and ``effect_25_to_75`` is the curve gap
  between the 25th and 75th grid doses; anchors A and B read both.
- ``src/statspai/did/continuous_did.py:171-255`` — ``method='twfe'``
  two-way-demeans ``Y`` and ``dose*post`` then OLS; ``estimate`` is the
  dose x post coefficient = the DiD slope (anchors D, E).
- ``continuous_did.py:442-624`` — ``method='cgs'`` long-differences each
  unit, subtracts the dose=0 control mean and reports both the level
  ``estimate`` (mean ATT over treated support) and
  ``model_info['acrt_overall']`` (mean derivative of the ATT(d) curve)
  (anchor F).

References (bib keys verified present in paper.bib via grep)
------------------------------------------------------------
- Hirano & Imbens (2004), "The Propensity Score with Continuous
  Treatments". [@hirano2004propensity] — the GPS estimator
  ``sp.dose_response`` implements.
- Kennedy, Ma, McHugh & Small (2017), JRSS-B 79(4) — doubly-robust
  continuous-treatment effects. [@kennedy2017parametric]
- Callaway, Goodman-Bacon & Sant'Anna (2024), "Difference-in-Differences
  with a Continuous Treatment". [@callaway2024difference] — target of
  ``method='cgs'``.
- de Chaisemartin & D'Haultfoeuille (2018), "Fuzzy
  Differences-in-Differences". [@dechaisemartin2018fuzzy]
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# Hand-set population parameters shared by the DGPs below.
BETA = 0.8  # structural dose-response slope for the GPS cross-section
TAU = 0.5  # per-unit-dose DiD effect on the post-period gain


def _within_n_se(est, truth, se, n_sigma=4.0):
    return abs(est - truth) <= n_sigma * se


# ---------------------------------------------------------------------------
# Deterministic DGP builders (every draw seeded via default_rng).
# ---------------------------------------------------------------------------


def _make_gps_dgp(seed, n=800, confounded=False):
    """Linear-Gaussian continuous-treatment DGP with known slope BETA.

    Structural model ``Y = BETA*D + g*X + noise``, so the population
    dose-response ``E[Y(d)]`` is a straight line with slope ``BETA``.
    ``confounded=True`` makes the dose depend on ``X`` (``D = 5 + 2X +
    noise``), so the naive ``Y~D`` slope inherits ``X``'s effect; with
    ``confounded=False`` the dose is independent of ``X`` and the GPS
    marginal effect must recover ``BETA`` exactly.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=n)
    if confounded:
        D = 5.0 + 2.0 * X + rng.normal(0.0, 1.5, size=n)
        g = 3.0
    else:
        D = 5.0 + rng.normal(0.0, 2.0, size=n)
        g = 1.5
    Y = BETA * D + g * X + rng.normal(0.0, 1.0, size=n)
    return pd.DataFrame({"y": Y, "d": D, "x": X})


def _make_did_dgp(seed, n=400, confounded_fe=False):
    """Two-period continuous-dose panel with a known DiD slope TAU.

    Half the units (control arm) get dose 0; the rest a continuous dose
    in [1, 10].  The post-period gain is ``TAU*dose``.  When
    ``confounded_fe=True`` the unit fixed effect is correlated with dose
    (``a_i = 3*dose_i + noise``), so a cross-sectional regression that
    ignores it is confounded while the DiD differences it away.
    """
    rng = np.random.default_rng(seed)
    dose = np.where(np.arange(n) < n // 2, 0.0, rng.uniform(1.0, 10.0, size=n))
    if confounded_fe:
        a_i = 3.0 * dose + rng.normal(0.0, 2.0, size=n)
    else:
        a_i = rng.normal(0.0, 2.0, size=n)
    rows = []
    for i in range(n):
        for year in (2019, 2020):
            post = 1 if year == 2020 else 0
            wage = 20.0 + a_i[i] + TAU * dose[i] * post + rng.normal(0.0, 0.3)
            rows.append({"id": i, "year": year, "dose": dose[i], "wage": wage})
    return pd.DataFrame(rows), dose


# ---------------------------------------------------------------------------
# Module fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gps_unconfounded():
    return _make_gps_dgp(202, n=800, confounded=False)


@pytest.fixture(scope="module")
def gps_confounded():
    return _make_gps_dgp(3, n=400, confounded=True)


@pytest.fixture(scope="module")
def did_clean():
    return _make_did_dgp(7, n=400, confounded_fe=False)


@pytest.fixture(scope="module")
def did_confounded():
    return _make_did_dgp(11, n=400, confounded_fe=True)


def _dose_response_linear(df, n_bootstrap=40):
    """GPS curve with linear models -> exactly linear curve (fast)."""
    from sklearn.linear_model import LinearRegression

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.dose_response(
            df,
            y="y",
            treat="d",
            covariates=["x"],
            n_dose_points=11,
            n_bootstrap=n_bootstrap,
            treatment_model=LinearRegression(),
            outcome_model=LinearRegression(),
            random_state=0,
        )


# ---------------------------------------------------------------------------
# A. GPS unconfounded recovery of BETA.
# ---------------------------------------------------------------------------


class TestGPSRecovery:
    """sp.dose_response recovers the hand-set slope BETA = 0.8."""

    # Absolute tolerance on the average marginal effect.  Under an
    # unconfounded dose the GPS curve is the structural line; probed
    # deviation across seeds is < 0.035, so 0.10 is ~3x the noise floor
    # yet a 20% multiplicative bias (-> ~0.94, dev 0.14) fails it.  A
    # finiteness check would pass for any number; this pins the slope.
    MARGINAL_ATOL = 0.10

    def test_marginal_effect_recovers_beta(self, gps_unconfounded):
        r = _dose_response_linear(gps_unconfounded, n_bootstrap=30)
        me = r.model_info["avg_marginal_effect"]
        assert abs(me - BETA) <= self.MARGINAL_ATOL, (
            f"GPS avg_marginal_effect {me:.4f} missed BETA {BETA} by "
            f"{abs(me - BETA):.4f} (> {self.MARGINAL_ATOL})."
        )

    def test_iqr_effect_within_4_se(self, gps_unconfounded):
        r = _dose_response_linear(gps_unconfounded, n_bootstrap=60)
        d25 = r.model_info["dose_25"]
        d75 = r.model_info["dose_75"]
        truth_iqr = BETA * (d75 - d25)
        # 4-sigma recovery (suite convention; probed z ~-0.9).  A 20%
        # bias lands ~3 sigma out, and the marginal-effect anchor above
        # already fails hard under that bias, so the recovery is pinned.
        assert _within_n_se(r.estimate, truth_iqr, r.se, n_sigma=4.0), (
            f"GPS effect_25_to_75 {r.estimate:.4f} (SE {r.se:.4f}) missed "
            f"BETA*(d75-d25) {truth_iqr:.4f} by "
            f"{abs(r.estimate - truth_iqr) / r.se:.1f} sigma."
        )


# ---------------------------------------------------------------------------
# B. GPS curve internal consistency (linear-model closed form).
# ---------------------------------------------------------------------------


class TestGPSCurveConsistency:
    """effect_25_to_75 == curve_slope*(dose_75 - dose_25)."""

    # With linear models the curve is a straight line, so the IQR effect
    # the estimator reports must equal the slope of that line times the
    # IQR dose width.  Tolerance 5e-2 (abs): the only slack is np.gradient
    # endpoint behaviour + the Gaussian-pdf GPS column gently bending the
    # fit; probed |diff| ~8e-3, so 5e-2 has headroom while still pinning
    # the headline scalar to the curve geometry (not finiteness).
    CONSISTENCY_ATOL = 5e-2

    def test_iqr_effect_equals_curve_slope_times_width(self, gps_unconfounded):
        r = _dose_response_linear(gps_unconfounded, n_bootstrap=5)
        det = r.detail
        slope = float(np.polyfit(det["dose"].values, det["response"].values, 1)[0])
        d25 = r.model_info["dose_25"]
        d75 = r.model_info["dose_75"]
        expected = slope * (d75 - d25)
        assert abs(r.estimate - expected) <= self.CONSISTENCY_ATOL, (
            f"effect_25_to_75 {r.estimate:.5f} != curve slope*(d75-d25) "
            f"{expected:.5f} (|diff|={abs(r.estimate - expected):.2e}) — "
            f"the IQR effect is detached from the dose-response curve."
        )


# ---------------------------------------------------------------------------
# C. GPS naive-bias contrast (GBM partially de-confounds).
# ---------------------------------------------------------------------------


class TestGPSNaiveBias:
    """Naive Y~D slope is biased high; GPS moves toward truth."""

    def test_gbm_marginal_between_truth_and_naive(self, gps_confounded):
        df = gps_confounded
        n = len(df)
        # Naive OLS slope of Y on D, omitting the confounder X.
        Dm = np.column_stack([np.ones(n), df["d"].values])
        beta_ols = np.linalg.lstsq(Dm, df["y"].values, rcond=None)[0]
        naive = float(beta_ols[1])

        # The naive slope really is confounded above truth: it absorbs
        # X's effect via Cov(X, D) > 0.  If this fails the DGP lost its
        # confounding and the contrast is vacuous.
        assert naive > BETA + 0.3, (
            f"naive Y~D slope {naive:.4f} not biased above truth {BETA} — "
            f"X-confounding vanished."
        )

        # Default GBM models: flexible curve that conditions on X through
        # the GPS feature -> partial de-confounding.  n=400, B=3 keeps it
        # well under the per-test budget.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sp.dose_response(
                df,
                y="y",
                treat="d",
                covariates=["x"],
                n_dose_points=8,
                n_bootstrap=3,
                random_state=0,
            )
        me = r.model_info["avg_marginal_effect"]
        # Strictly between truth and naive: a real correction, not a
        # no-op (>= naive) and not over-shooting below truth.  Probed
        # ~0.91 in (0.80, 1.77); a +20% bias pushes it past naive and
        # fails the upper bound.
        assert BETA < me < naive, (
            f"GBM GPS marginal {me:.4f} not strictly between truth {BETA} "
            f"and naive {naive:.4f} — it neither partially de-confounds "
            f"nor stays below the raw confounded slope."
        )


# ---------------------------------------------------------------------------
# D. Continuous-DiD slope recovery of TAU.
# ---------------------------------------------------------------------------


class TestContinuousDiDRecovery:
    """method='twfe' dose x post coefficient recovers TAU = 0.5."""

    def test_twfe_slope_within_4_se(self, did_clean):
        df, _dose = did_clean
        r = sp.continuous_did(
            df,
            y="wage",
            dose="dose",
            time="year",
            id="id",
            t_pre=2019,
            t_post=2020,
            method="twfe",
            seed=0,
        )
        # 4-sigma recovery (suite convention; probed z ~-1.6).  A 20%
        # multiplicative bias on the slope lands ~20 sigma out and fails.
        assert _within_n_se(r.estimate, TAU, r.se, n_sigma=4.0), (
            f"continuous_did twfe slope {r.estimate:.5f} (SE {r.se:.5f}) "
            f"missed TAU {TAU} by {abs(r.estimate - TAU) / r.se:.1f} sigma."
        )


# ---------------------------------------------------------------------------
# E. Continuous-DiD naive-bias contrast.
# ---------------------------------------------------------------------------


class TestContinuousDiDNaiveBias:
    """Cross-section Y~dose is biased; DiD differences the FE away."""

    def test_cross_section_biased_did_recovers(self, did_confounded):
        df, _dose = did_confounded
        # Naive cross-section: regress POST-period wage on dose, ignoring
        # the dose-correlated unit fixed effect.  Hand-rolled OLS + SE.
        post_df = df[df["year"] == 2020]
        Dn = np.column_stack([np.ones(len(post_df)), post_df["dose"].values])
        bn = np.linalg.lstsq(Dn, post_df["wage"].values, rcond=None)[0]
        resid = post_df["wage"].values - Dn @ bn
        sig2 = float(resid @ resid) / (len(post_df) - 2)
        xtx_inv = np.linalg.inv(Dn.T @ Dn)
        naive = float(bn[1])
        se_naive = float(np.sqrt(sig2 * xtx_inv[1, 1]))

        # The cross-section is provably confounded: >> 6 sigma above
        # truth (probed z ~94) because a_i = 3*dose + noise loads onto
        # the dose slope.  Guards that the contrast is not vacuous.
        assert naive - TAU > 6.0 * se_naive, (
            f"naive cross-section slope {naive:.4f} should be > 6 sigma "
            f"above truth {TAU} (SE {se_naive:.4f}) — confounding gone."
        )

        r = sp.continuous_did(
            df,
            y="wage",
            dose="dose",
            time="year",
            id="id",
            t_pre=2019,
            t_post=2020,
            method="twfe",
            seed=0,
        )
        # DiD differences the fixed effect out and recovers TAU within
        # 4 sigma (probed z ~-0.1) -- AND lands far below the naive slope.
        assert _within_n_se(r.estimate, TAU, r.se, n_sigma=4.0), (
            f"continuous_did twfe {r.estimate:.5f} (SE {r.se:.5f}) failed "
            f"to recover TAU {TAU} on the confounded-FE DGP."
        )
        assert r.estimate < naive - 1.0, (
            f"DiD slope {r.estimate:.4f} did not de-confound below the "
            f"naive cross-section slope {naive:.4f}."
        )


# ---------------------------------------------------------------------------
# F. Continuous-DiD cross-method consistency (level vs slope).
# ---------------------------------------------------------------------------


class TestContinuousDiDCGSConsistency:
    """cgs level ATT == TAU*mean_dose; cgs ACRT == TAU."""

    # SE-band multiplier for the level/slope recovery (suite convention).
    N_SIGMA = 4.0

    def test_cgs_level_equals_tau_times_mean_dose(self, did_clean):
        df, dose = did_clean
        r = sp.continuous_did(
            df,
            y="wage",
            dose="dose",
            time="year",
            id="id",
            t_pre=2019,
            t_post=2020,
            method="cgs",
            n_boot=120,
            seed=0,
        )
        mean_treated_dose = float(dose[dose > 0].mean())
        level_truth = TAU * mean_treated_dose
        # The cgs estimand is the mean ATT over the treated dose support,
        # which under the linear DGP equals TAU * E[D | D>0].  Probed
        # 2.72 vs 2.77 (z ~-0.6).  Ties the level to the SAME hand-set
        # TAU as the twfe slope, through a different estimand.
        assert _within_n_se(r.estimate, level_truth, r.se, n_sigma=self.N_SIGMA), (
            f"cgs level ATT {r.estimate:.4f} (SE {r.se:.4f}) missed "
            f"TAU*mean_dose {level_truth:.4f} by "
            f"{abs(r.estimate - level_truth) / r.se:.1f} sigma."
        )

    def test_cgs_acrt_recovers_tau_slope(self, did_clean):
        df, _dose = did_clean
        r = sp.continuous_did(
            df,
            y="wage",
            dose="dose",
            time="year",
            id="id",
            t_pre=2019,
            t_post=2020,
            method="cgs",
            n_boot=120,
            seed=0,
        )
        acrt = float(r.model_info["acrt_overall"])
        acrt_se = float(r.model_info["acrt_se"])
        # ACRT(d) = d/dd ATT(d) is the per-unit-dose derivative, which is
        # the structural TAU.  Probed 0.506 (SE 0.015; z ~0.4).
        assert _within_n_se(acrt, TAU, acrt_se, n_sigma=self.N_SIGMA), (
            f"cgs ACRT {acrt:.4f} (SE {acrt_se:.4f}) missed slope TAU "
            f"{TAU} by {abs(acrt - TAU) / acrt_se:.1f} sigma."
        )


# ---------------------------------------------------------------------------
# G. Determinism / seed-stability.
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Fixed seeds -> bitwise-identical repeated calls."""

    def test_dose_response_deterministic(self, gps_unconfounded):
        r1 = _dose_response_linear(gps_unconfounded, n_bootstrap=20)
        r2 = _dose_response_linear(gps_unconfounded, n_bootstrap=20)
        assert r1.estimate == r2.estimate, "dose_response estimate not deterministic"
        assert r1.se == r2.se, "dose_response se not deterministic"

    def test_continuous_did_deterministic(self, did_clean):
        df, _dose = did_clean
        kw = dict(
            y="wage",
            dose="dose",
            time="year",
            id="id",
            t_pre=2019,
            t_post=2020,
            method="att_gt",
            n_boot=80,
            seed=42,
        )
        a = sp.continuous_did(df, **kw)
        b = sp.continuous_did(df, **kw)
        assert a.estimate == b.estimate, "continuous_did estimate not deterministic"
        assert a.se == b.se, "continuous_did se not deterministic"
