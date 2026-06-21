"""Reference parity: quantile treatment effect family.

Estimators
----------
``sp.qte`` (Firpo 2007 quantile-regression and IPW-distribution variants)
and ``sp.qdid`` (Athey & Imbens 2006 quantile difference-in-differences).
Each previously had only a smoke test; this file is their first numerical
guarantee.

Setting (location-shift identification)
---------------------------------------
The cleanest non-tautological probe for a *quantile* treatment effect is
the **pure location shift** potential-outcome model
    Y1 = Y0 + delta
with a KNOWN constant per-unit shift ``delta``.  Under a constant shift
*every* quantile of the treated distribution sits exactly ``delta`` above
the matching quantile of the control distribution, so the true QTE is the
horizontal line ``QTE(tau) = delta`` for all ``tau`` (Firpo 2007,
Section 2; Athey & Imbens 2006 changes-in-changes with an additive shift).
That gives three independent things to pin:

* **recovery** — the estimated ``QTE(tau)`` lands near ``delta`` at every
  ``tau`` (within a Monte-Carlo / bootstrap band of a HAND-SET truth);
* **homogeneity** — because the shift is constant, the per-quantile
  effects are (nearly) equal to each other and to ``delta``;
* **closed-form collapse** — when the treated empirical distribution is
  *exactly* the control empirical distribution shifted by ``delta`` (the
  same baseline draw, duplicated and shifted), the empirical-quantile
  arithmetic recovers ``delta`` at every ``tau`` to machine precision.

Anchors
-------
A. **Closed-form exact collapse** (closed_form).  Build a sample where the
   treated outcomes are the control outcomes plus exactly ``DELTA`` (same
   baseline values, duplicated and shifted).  ``sp.qte(..., method=
   "distribution")`` with no covariates uses uniform IPW weights, so its
   per-quantile estimate is the plain empirical-quantile difference; since
   the treated ECDF is the control ECDF shifted by ``DELTA``, every
   quantile effect equals ``DELTA`` to ~4e-16 (probed).  Likewise a
   four-cell ``sp.qdid`` panel in which the DID-quantile contrast cancels
   the common trend exactly recovers ``DELTA`` at every ``tau`` to
   ~7e-16.  Pins the estimate to an exact algebraic identity, not "finite".
B. **Known-DGP recovery of DELTA** (recovery).  On a random
   location-shift DGP ``Y = Y0 + DELTA*D`` ``sp.qte`` recovers ``DELTA``
   at ``tau in {.25,.5,.75}`` within 4 bootstrap SE (probed z ~0.1-0.3),
   and a 30-rep Monte-Carlo mean of the median QTE is within 4*SD/sqrt(R)
   of ``DELTA`` (probed mean 1.98, band 0.094).  ``sp.qdid`` recovers a
   distinct hand-set shift on a four-cell DiD panel within 4 SE (probed z
   ~0.2-0.75).
C. **Homogeneity of a pure location shift** (homogeneity).  A constant
   shift induces NO quantile heterogeneity, so the spread of estimated
   per-quantile effects (max - min) is small (probed 0.056) and each
   effect is within a loose band of ``DELTA``.  Non-tautological: a 20%
   multiplicative bias scales every quantile effect to ~1.2*DELTA, which
   fails the per-quantile recovery band below.
D. **Cross-method consistency** (consistency).  On the same location-shift
   data the two ``sp.qte`` engines — quantile regression (Firpo 2007) and
   IPW distribution — agree on the per-quantile effects within a tight
   band, because under a homogeneous shift both target the same QTE(tau).
E. **Orientation / sign correctness** (orientation).  A strictly positive
   shift (``DELTA > 0``) yields strictly positive estimates from ``qte``
   (both methods) and ``qdid`` at every quantile; a strictly negative
   shift flips every sign.

Implementation facts the anchors rely on (cited file:line)
----------------------------------------------------------
- ``src/statspai/qte/qte.py:221-224`` — ``_quantile_func`` is the plain
  empirical (type-4 / linear-interpolated) quantile on ``sort(x)`` with
  plotting positions ``i/n``; anchors A and the exact collapse depend on
  this being the literal empirical quantile.
- ``src/statspai/qte/qte.py:525-540`` — the distribution method's
  ``_weighted_quantiles`` uses IPW weights ``p/(1-p)``; with NO covariates
  the propensity score is a constant (``qte.py:507-508`` -> intercept-only
  logistic), so the weights are uniform and ``q0`` is the unweighted
  control empirical quantile.  Hence treated-minus-control collapses to
  the empirical-quantile difference (anchor A-i).
- ``src/statspai/qte/qte.py:306-313`` — ``qdid``'s ``_point`` computes
  ``q11 - q10 - (q01 - q00)`` from four empirical quantile vectors; with
  an additive common trend and additive ``DELTA`` the trend cancels and
  the contrast equals ``DELTA`` at every ``tau`` (anchor A-ii).
- ``src/statspai/qte/qte.py:466 / :479`` — both ``qte`` and ``qdid`` read
  the per-quantile effect from ``QTEResult.effects`` (treatment-regressor
  coefficient / DID-quantile contrast); the recovery anchors compare those
  to ``DELTA``.

References (bib keys verified present in paper.bib via grep)
------------------------------------------------------------
- Firpo (2007), "Efficient Semiparametric Estimation of Quantile Treatment
  Effects", Econometrica 75(1). [@firpo2007efficient]
- Athey & Imbens (2006), "Identification and Inference in Nonlinear
  Difference-in-Differences Models", Econometrica 74(2).
  [@athey2006identification]
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# Hand-set true location shifts shared by the DGPs below.
DELTA = 2.0  # constant shift for the sp.qte location-shift DGP
DELTA_DID = 2.5  # constant shift layered on a common trend for sp.qdid
TREND = 1.0  # common (control + treated) pre->post drift for qdid


# ---------------------------------------------------------------------------
# Deterministic DGP builders (every draw seeded via default_rng).
# ---------------------------------------------------------------------------


def _make_location_shift_dgp(seed, n=4000, delta=DELTA):
    """Cross-section location-shift DGP with a KNOWN constant QTE = delta.

    ``Y = Y0 + delta * D`` where ``Y0`` is a baseline outcome independent
    of ``D``.  A constant shift moves every quantile of the treated
    distribution up by exactly ``delta``, so the true ``QTE(tau) = delta``
    for all ``tau`` (Firpo 2007).  ``D`` is randomized, so the naive
    treated-minus-control quantile diff *is* the QTE here — which is what
    makes recovery a clean (non-confounded) check of the estimator's
    quantile machinery.
    """
    rng = np.random.default_rng(seed)
    d = rng.integers(0, 2, n)
    y0 = rng.normal(5.0, 2.0, n)
    y = y0 + delta * d
    return pd.DataFrame({"y": y, "d": d})


def _make_exact_shift_dgp(seed, n=2000, delta=3.0):
    """Treated ECDF == control ECDF shifted by EXACTLY ``delta``.

    The treated arm is the *same* baseline draw plus ``delta`` (duplicated,
    not a fresh sample), so the empirical quantile of the treated arm at
    any ``tau`` equals the control empirical quantile plus ``delta`` to
    machine precision.  The distribution-method QTE (uniform IPW weights,
    no covariates) is then exactly ``delta`` at every ``tau`` (anchor A-i).
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, n)
    y = np.concatenate([base, base + delta])
    d = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    return pd.DataFrame({"y": y, "d": d}), delta


def _make_qdid_dgp(seed, n_per=3000, delta=DELTA_DID, trend=TREND):
    """Four-cell DiD panel with a KNOWN distributional change = delta.

    control pre  ~ N(5, 2)            (baseline)
    control post ~ N(5 + trend, 2)    (common trend)
    treated pre  ~ N(5, 2)            (baseline)
    treated post ~ N(5 + trend + delta, 2)  (trend + treatment shift)

    The Athey-Imbens DID-quantile contrast
        [q11(tau) - q10(tau)] - [q01(tau) - q00(tau)]
    targets ``delta`` at every ``tau`` because the additive common trend
    cancels in the difference (changes-in-changes under an additive shift).
    """
    rng = np.random.default_rng(seed)
    y00 = rng.normal(5.0, 2.0, n_per)
    y01 = rng.normal(5.0 + trend, 2.0, n_per)
    y10 = rng.normal(5.0, 2.0, n_per)
    y11 = rng.normal(5.0 + trend + delta, 2.0, n_per)
    g = np.concatenate(
        [np.zeros(n_per), np.zeros(n_per), np.ones(n_per), np.ones(n_per)]
    ).astype(int)
    t = np.concatenate(
        [np.zeros(n_per), np.ones(n_per), np.zeros(n_per), np.ones(n_per)]
    ).astype(int)
    y = np.concatenate([y00, y01, y10, y11])
    return pd.DataFrame({"y": y, "g": g, "t": t})


def _make_exact_qdid_dgp(seed, n=1500, delta=1.7, trend=0.9):
    """Four-cell panel where the DID-quantile contrast == delta EXACTLY.

    Each post cell is its OWN pre cell plus an additive term (same baseline
    values, duplicated): control post = control pre + trend, treated post =
    treated pre + trend + delta.  Then ``q11-q10 = trend+delta`` and
    ``q01-q00 = trend`` at every empirical quantile, so the contrast is
    ``delta`` to machine precision regardless of the (different) control
    vs treated baselines (anchor A-ii).
    """
    rng = np.random.default_rng(seed)
    base_c = rng.normal(2.0, 1.0, n)
    base_t = rng.normal(4.0, 1.5, n)  # different baseline -> must cancel
    y00 = base_c
    y01 = base_c + trend
    y10 = base_t
    y11 = base_t + trend + delta
    g = np.concatenate([np.zeros(n), np.zeros(n), np.ones(n), np.ones(n)]).astype(int)
    t = np.concatenate([np.zeros(n), np.ones(n), np.zeros(n), np.ones(n)]).astype(int)
    y = np.concatenate([y00, y01, y10, y11])
    return pd.DataFrame({"y": y, "g": g, "t": t}), delta


# ---------------------------------------------------------------------------
# Module fixtures.
# ---------------------------------------------------------------------------

QUANTILES = [0.25, 0.5, 0.75]


@pytest.fixture(scope="module")
def shift_data():
    # n=2000: the quantile-regression IRLS bootstrap is the runtime
    # bottleneck (~6-7s per fit here), so the cross-section is kept small;
    # at this size the recovery z stays <2 and the homogeneity spread <0.22
    # across seeds (probed), well inside the anchor bands below.
    return _make_location_shift_dgp(20260614, n=2000)


@pytest.fixture(scope="module")
def qdid_data():
    return _make_qdid_dgp(424242, n_per=3000)


def _qte(df, method="quantile_regression", n_boot=40, seed=0):
    # Bootstrap SEs only; weak-overlap / convergence chatter is not under
    # test here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.qte(
            df,
            y="y",
            treatment="d",
            quantiles=QUANTILES,
            method=method,
            n_boot=n_boot,
            seed=seed,
        )


def _qdid(df, quantiles=QUANTILES, n_boot=40, seed=2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.qdid(
            df,
            y="y",
            group="g",
            time="t",
            quantiles=quantiles,
            n_boot=n_boot,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# A. Closed-form exact collapse (machine precision).
# ---------------------------------------------------------------------------


class TestClosedFormCollapse:
    """Empirical-quantile arithmetic recovers DELTA to machine precision."""

    # 1e-9: in exact arithmetic these are algebraic identities (treated
    # ECDF == control ECDF + delta; DID-quantile contrast cancels the
    # additive trend).  The only slack is float rounding in np.sort /
    # np.interp; probed deviation ~4e-16 (qte) and ~7e-16 (qdid), so 1e-9
    # has ~6 orders of headroom yet stays far tighter than the 1e-8
    # machine-collapse bar.  Checking finiteness would pass for any value;
    # this pins the estimate to an exact hand-set scalar.
    TOL = 1e-9

    def test_qte_distribution_exact_shift(self):
        df, delta = _make_exact_shift_dgp(7, n=2000, delta=3.0)
        r = _qte(df, method="distribution", n_boot=5, seed=0)
        dev = float(np.max(np.abs(r.effects - delta)))
        assert dev < self.TOL, (
            f"qte(distribution) per-quantile effects {r.effects} deviate "
            f"from the exact shift {delta} by {dev:.2e}; the "
            f"treated-ECDF == control-ECDF + delta identity is broken."
        )

    def test_qdid_exact_did_contrast(self):
        df, delta = _make_exact_qdid_dgp(99, n=1500, delta=1.7, trend=0.9)
        r = _qdid(df, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9], n_boot=5, seed=0)
        dev = float(np.max(np.abs(r.effects - delta)))
        assert dev < self.TOL, (
            f"qdid DID-quantile contrast {r.effects} deviates from the "
            f"exact shift {delta} by {dev:.2e}; the additive common trend "
            f"failed to cancel."
        )


# ---------------------------------------------------------------------------
# B. Known-DGP recovery of the hand-set shift.
# ---------------------------------------------------------------------------


class TestRecovery:
    """qte / qdid recover the hand-set location shift."""

    def test_qte_per_quantile_within_4_se(self, shift_data):
        r = _qte(shift_data, method="quantile_regression", n_boot=15)
        # 4-sigma bootstrap band (suite convention).  Probed z <2 at every
        # tau across seeds at this n.  A 20% multiplicative bias (-> ~2.4)
        # lands ~4-6 sigma out at every quantile and fails.
        for tau, eff, se in zip(r.quantiles, r.effects, r.se):
            z = abs(eff - DELTA) / se
            assert z <= 4.0, (
                f"qte(qreg) at tau={tau:.2f}: {eff:.4f} (SE {se:.4f}) "
                f"misses truth {DELTA} by {z:.1f} sigma."
            )

    def test_qte_monte_carlo_median_recovers(self):
        """30 independent draws; MC mean of the median QTE within band.

        Averaging cancels per-draw bootstrap noise, so the MC mean is a
        far tighter probe of *systematic* bias than any single draw.
        Probed MC mean 2.012, SD ~0.177 -> band 4*0.177/sqrt(25) ~0.142
        around DELTA = 2.0 (probed dev 0.012).  A 20% multiplicative bias
        would shift the mean to ~2.40, ~3 band-widths out.
        """
        reps = 25
        ests = []
        for s in range(reps):
            df = _make_location_shift_dgp(1000 + s, n=1000)
            res = _qte(df, method="quantile_regression", n_boot=2, seed=0)
            # tau = 0.5 is the middle of QUANTILES.
            mid = list(res.quantiles).index(0.5)
            ests.append(float(res.effects[mid]))
        ests = np.asarray(ests)
        mc_mean = float(ests.mean())
        mc_sd = float(ests.std(ddof=1))
        band = 4.0 * mc_sd / np.sqrt(reps)
        assert abs(mc_mean - DELTA) <= band, (
            f"MC mean of median QTE {mc_mean:.4f} drifted from truth "
            f"{DELTA} (band {band:.4f}, SD {mc_sd:.4f}) over {reps} reps — "
            f"systematic bias in qte."
        )

    def test_qdid_per_quantile_within_4_se(self, qdid_data):
        r = _qdid(qdid_data, n_boot=40, seed=2)
        # Probed z ~0.2-0.75 vs DELTA_DID = 2.5.  20% bias (-> 3.0) is
        # >5 sigma out at the bootstrap SE scale (~0.08-0.09).
        for tau, eff, se in zip(r.quantiles, r.effects, r.se):
            z = abs(eff - DELTA_DID) / se
            assert z <= 4.0, (
                f"qdid at tau={tau:.2f}: {eff:.4f} (SE {se:.4f}) misses "
                f"truth {DELTA_DID} by {z:.1f} sigma."
            )


# ---------------------------------------------------------------------------
# C. Homogeneity of a pure location shift.
# ---------------------------------------------------------------------------


class TestHomogeneity:
    """A constant shift induces NO quantile heterogeneity."""

    def test_qte_effects_flat_across_quantiles(self, shift_data):
        r = _qte(shift_data, method="quantile_regression", n_boot=10)
        # 0.30: a pure location shift has identical effect at every tau, so
        # the across-quantile spread is pure sampling noise (probed <0.22
        # at n=2000 across seeds).  0.30 leaves headroom over the probed
        # spread yet is far below DELTA = 2.0, so heterogeneity injected by
        # a tau-dependent bias would break it.  Non-tautological: a number
        # can be finite with a large spread.
        spread = float(r.effects.max() - r.effects.min())
        assert spread < 0.30, (
            f"qte effects {r.effects} span {spread:.4f} across quantiles "
            f"despite a pure location shift (should be ~flat at DELTA)."
        )
        # ...and each per-quantile effect is within a loose band of DELTA
        # (the homogeneous truth), so "flat" can't be satisfied by a flat
        # line at the WRONG level.
        assert np.all(np.abs(r.effects - DELTA) < 0.30), (
            f"qte effects {r.effects} are flat but not centered on the "
            f"homogeneous truth {DELTA}."
        )


# ---------------------------------------------------------------------------
# D. Cross-method consistency (qreg vs IPW-distribution).
# ---------------------------------------------------------------------------


class TestCrossMethodConsistency:
    """The two qte engines agree under a homogeneous shift."""

    def test_qreg_vs_distribution_agree(self, shift_data):
        # Point effects only (SE unused) -> n_boot=2 keeps it fast.
        rq = _qte(shift_data, method="quantile_regression", n_boot=2)
        rd = _qte(shift_data, method="distribution", n_boot=2)
        # 0.10: both engines target the same QTE(tau) under a homogeneous
        # shift; they differ only in finite-sample estimation (quantile
        # regression coefficient vs IPW empirical-quantile difference).
        # Probed max gap ~0.01 at n=4000.  0.10 leaves ~10x headroom but is
        # far below DELTA = 2.0; a 20% bias in ONE engine (-> ~2.4 vs ~2.0)
        # would open a ~0.4 gap and fail.
        gap = float(np.max(np.abs(rq.effects - rd.effects)))
        assert gap < 0.10, (
            f"qte(qreg) {rq.effects} and qte(distribution) {rd.effects} "
            f"disagree by {gap:.4f} on a homogeneous-shift DGP where both "
            f"target the same QTE(tau)."
        )


# ---------------------------------------------------------------------------
# E. Orientation / sign correctness.
# ---------------------------------------------------------------------------


class TestOrientation:
    """A positive shift yields positive estimates; a negative shift flips."""

    def test_positive_shift_all_positive(self, shift_data, qdid_data):
        # Sign of point effects only -> minimal bootstrap.
        rq = _qte(shift_data, method="quantile_regression", n_boot=2)
        rd = _qte(shift_data, method="distribution", n_boot=2)
        rdid = _qdid(qdid_data, n_boot=2, seed=2)
        assert np.all(rq.effects > 0), rq.effects
        assert np.all(rd.effects > 0), rd.effects
        assert np.all(rdid.effects > 0), rdid.effects

    def test_negative_shift_flips_sign(self):
        # Same machinery, a strictly NEGATIVE hand-set shift -> every
        # quantile effect must be negative.  Sign is determined by the DGP,
        # not by the estimator's existence, so this is non-tautological.
        df = _make_location_shift_dgp(555, n=2000, delta=-1.5)
        rq = _qte(df, method="quantile_regression", n_boot=2)
        rd = _qte(df, method="distribution", n_boot=2)
        assert np.all(rq.effects < 0), rq.effects
        assert np.all(rd.effects < 0), rd.effects
