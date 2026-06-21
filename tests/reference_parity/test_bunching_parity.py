"""Reference parity: ``sp.bunching`` and ``sp.general_bunching``.

These bunching estimators previously had only smoke tests; this file is
their first numerical anchor.  The bunching method (Saez 2010 / Chetty
et al. 2011 / Kleven-Waseem 2013) recovers behavioural responses to a
policy threshold by (1) binning the running variable, (2) fitting a
*counterfactual* polynomial to the density EXCLUDING a bunching region,
and (3) reading off the excess mass = observed - counterfactual in that
region.  Every anchor below pins the estimate to a HAND-COMPUTED truth
(a planted integer excess or a planted elasticity), never to finiteness.

Anchors
-------
A. **Closed-form excess-mass integral (machine collapse).** Build the
   histogram DETERMINISTICALLY so the per-bin counts follow a polynomial
   I control (flat, then linear), place EXACTLY ``EXCESS`` extra points
   in one in-region bin, and assert ``model_info['excess_mass_raw']``
   equals the integer I planted and ``counterfactual_at_threshold``
   equals the polynomial intercept I set.  Because polyfit reproduces a
   low-order polynomial from exact integer counts, this is an algebraic
   identity (observed |diff| = 0.0, tol 1e-9).  Same idea pins
   ``sp.general_bunching``'s elasticity to the hand value
   ``EXCESS / (n * f_at * bandwidth^2)`` (observed ~5e-17, tol 1e-9).
B. **Known-DGP recovery within ~10%.** A smooth uniform base plus
   ``N_EXTRA`` planted bunchers landing inside the default bunch region:
   ``excess_mass_raw`` recovers ``N_EXTRA`` within 10% (probed max rel
   error 4.9% over 6 seeds).  Independently, ``general_bunching``
   recovers a clearly nonzero elasticity at |z| ~25.
C. **Null: smooth density, NO notch -> ~zero excess.** With NO planted
   bunchers the normalised excess mass / elasticity is statistically
   indistinguishable from zero (within 4 SE; probed |z| < 1).  This is
   the necessary contrast to anchor B — a method that fabricated mass
   would fail it.
D. **Internal-consistency identity.** ``sp.bunching``'s reported
   estimate (normalised B) equals ``excess_mass_raw /
   counterfactual_at_threshold`` exactly (tol 1e-9) — the definition the
   summary/CI are built on.  ``general_bunching``'s naive (order-2) and
   bias-corrected elasticities coincide on a flat counterfactual where
   higher-order terms vanish (tol 1e-9).

Implementation facts relied on (cited file:line)
------------------------------------------------
- ``src/statspai/bunching/bunching.py:233-244`` — counterfactual is
  ``np.polyfit`` on the OUT-of-bunch bins (x normalised by bin_width),
  ``excess = sum(counts[in]) - sum(cf[in])``,
  ``cf_at_threshold = polyval(coeffs, 0)``, and the reported estimate is
  ``B_normalised = excess / cf_at_threshold``.
- ``src/statspai/bunching/bunching.py:211-213`` — default bunch region
  is ``[threshold - 2*bin_width, threshold + 2*bin_width]``.
- ``src/statspai/bunching/bunching.py:298-307`` — ``model_info`` exposes
  ``excess_mass_raw``, ``excess_mass_normalised``,
  ``counterfactual_at_threshold`` and (when ``dt`` given) ``elasticity``.
- ``src/statspai/bunching/general.py:126-143`` — ``_elasticity`` measures
  ``excess = sum(counts[excluded] - cf[excluded])`` over bins within
  ``bin_width`` of the cutoff, ``f_at = mean(counts[fit])/(n*bin_width)``,
  and ``eps = excess / (n * f_at * bandwidth^2)``; ``bin_width`` defaults
  to ``bandwidth/25``.  ``naive`` uses order 2, ``bias_corrected`` uses
  ``polynomial_order``.

References (bib keys grep-confirmed in paper.bib)
-------------------------------------------------
- Kleven, H.J. & Waseem, M. (2013). Using Notches to Uncover
  Optimization Frictions and Structural Elasticities. *QJE* 128(2),
  669-723. [@kleven2013using]
- Chetty, R., Friedman, J.N., Olsen, T. & Pistaferri, L. (2011).
  Adjustment Costs, Firm Responses, and Micro vs. Macro Labor Supply
  Elasticities. *QJE* 126(2), 749-804. [@chetty2011adjustment]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# ---------------------------------------------------------------------------
# Deterministic-histogram builders (independent of statspai internals)
# ---------------------------------------------------------------------------


def _make_bunching_deterministic(
    threshold, bin_width, n_bins, a, b, excess, excess_bin_center
):
    """Place points so each bin's count is exactly ``a + b*center``.

    Bins are centred at ``threshold + bin_width*(k + 0.5)`` for
    ``k in [-n_bins, n_bins)`` (mirrors bunching.py:201-204).  A point
    placed AT a bin centre lands deterministically in that bin, so the
    histogram counts are exactly the rounded ``a + b*center`` heights
    plus the planted ``excess`` in one in-region bin.  Returns the data
    array, the bin centres and the integer counts so the test can do its
    OWN hand integral.
    """
    centers = threshold + bin_width * (np.arange(-n_bins, n_bins) + 0.5)
    counts = (a + b * centers).round().astype(int)
    pts = []
    for c, k in zip(centers, counts):
        pts.extend([c] * int(k))
    pts.extend([excess_bin_center] * excess)
    return np.asarray(pts, dtype=float), centers, counts


# ---------------------------------------------------------------------------
# A. Closed-form excess-mass integral (machine collapse)
# ---------------------------------------------------------------------------


class TestBunchingClosedForm:
    """Deterministic counts -> excess mass is the planted integer.

    Tolerance 1e-9: with integer per-bin counts and a polynomial of
    order >= the true degree, np.polyfit reproduces the counterfactual
    exactly, so ``excess = sum(counts[in]) - sum(cf[in])`` equals the
    planted integer in exact arithmetic.  Observed |diff| = 0.0 for the
    excess and the intercept; 1e-9 leaves machine headroom while staying
    far under any 20%-bias perturbation.
    """

    TOL = 1e-9

    def test_flat_counterfactual_excess_is_exact(self):
        # Flat density: counts = CONST per bin -> cf_at_threshold = CONST.
        threshold, bin_width, n_bins = 0.0, 1.0, 10
        CONST, EXCESS = 50, 37
        data, _, _ = _make_bunching_deterministic(
            threshold,
            bin_width,
            n_bins,
            a=CONST,
            b=0.0,
            excess=EXCESS,
            excess_bin_center=0.5,  # in-region bin
        )
        r = sp.bunching(
            pd.DataFrame({"z": data}),
            running_var="z",
            threshold=threshold,
            bin_width=bin_width,
            n_bins=n_bins,
            poly_order=3,
            n_bootstrap=5,
        )
        mi = r.model_info
        assert abs(mi["excess_mass_raw"] - EXCESS) < self.TOL, (
            f"flat-cf raw excess {mi['excess_mass_raw']!r} != planted "
            f"{EXCESS} (cf reproduces a constant exactly)"
        )
        assert abs(mi["counterfactual_at_threshold"] - CONST) < self.TOL, (
            f"cf_at_threshold {mi['counterfactual_at_threshold']!r} != "
            f"flat height {CONST}"
        )
        # Normalised estimate is EXCESS / CONST by definition.
        assert abs(r.estimate - EXCESS / CONST) < self.TOL

    def test_linear_counterfactual_integral_is_exact(self):
        """Non-tautological hand integral against a LINEAR counterfactual.

        Counts follow ``a + b*center`` with b != 0, so the test computes
        the in-region counterfactual integral by hand (sum over the four
        in-region bin centres) and asserts the estimator reproduces
        ``observed_in - cf_integral`` exactly.  This pins the integral,
        not just a constant.
        """
        threshold, bin_width, n_bins = 0.0, 1.0, 10
        a, b, EXCESS = 60.0, 2.0, 23
        data, centers, counts = _make_bunching_deterministic(
            threshold,
            bin_width,
            n_bins,
            a=a,
            b=b,
            excess=EXCESS,
            excess_bin_center=-0.5,
        )
        r = sp.bunching(
            pd.DataFrame({"z": data}),
            running_var="z",
            threshold=threshold,
            bin_width=bin_width,
            n_bins=n_bins,
            poly_order=2,
            n_bootstrap=5,
        )

        # Hand integral over the default region [thr-2bw, thr+2bw]:
        # in-region bin centres are -1.5, -0.5, 0.5, 1.5.
        in_mask = (centers >= threshold - 2 * bin_width) & (
            centers <= threshold + 2 * bin_width
        )
        cf_integral_hand = float(np.sum(a + b * centers[in_mask]))
        observed_in = float(np.sum(counts[in_mask])) + EXCESS
        expected_excess = observed_in - cf_integral_hand

        assert abs(r.model_info["excess_mass_raw"] - expected_excess) < self.TOL, (
            f"linear-cf raw excess {r.model_info['excess_mass_raw']!r} "
            f"!= hand integral {expected_excess!r}"
        )
        assert abs(r.model_info["counterfactual_at_threshold"] - a) < self.TOL, (
            f"cf_at_threshold {r.model_info['counterfactual_at_threshold']!r}"
            f" != linear intercept {a}"
        )


class TestGeneralBunchingClosedForm:
    """general_bunching elasticity = excess / (n * f_at * bandwidth^2).

    Tolerance 1e-9: a deterministic flat density makes the polynomial
    counterfactual exact, so the elasticity equals the hand-computed
    value to machine precision (observed ~5e-17).
    """

    TOL = 1e-9

    def test_elasticity_matches_hand_formula(self):
        cutoff, bandwidth = 0.0, 1.0
        bin_width = bandwidth / 25.0  # general.py default
        bins = np.arange(cutoff - bandwidth, cutoff + bandwidth + bin_width, bin_width)
        centers = 0.5 * (bins[:-1] + bins[1:])
        excluded = (centers > cutoff - bin_width) & (centers < cutoff + bin_width)

        CONST, EXCESS = 40, 30
        pts = []
        for c in centers:
            pts.extend([c] * CONST)
        # Plant all excess into a single excluded bin.
        pts.extend([float(centers[excluded][0])] * EXCESS)
        data = np.asarray(pts, dtype=float)
        n = len(data)

        res = sp.general_bunching(
            pd.DataFrame({"r": data}),
            running="r",
            cutoff=cutoff,
            bandwidth=bandwidth,
            polynomial_order=3,
            n_boot=5,
            seed=0,
        )

        # Hand: flat cf height CONST -> f_at = CONST / (n * bin_width);
        # eps = EXCESS / (n * f_at * bandwidth^2).
        f_at = CONST / (n * bin_width)
        eps_hand = EXCESS / (n * f_at * bandwidth**2)
        assert abs(res.bias_corrected_elasticity - eps_hand) < self.TOL, (
            f"corrected eps {res.bias_corrected_elasticity!r} != hand "
            f"formula {eps_hand!r}"
        )

    def test_naive_equals_corrected_on_flat_density(self):
        """Order-2 (naive) == order-k (corrected) when cf is flat.

        On a flat counterfactual the higher-order polynomial terms are
        identically zero, so the Saez first-order and bias-corrected
        elasticities coincide (tol 1e-9; observed ~3e-17).  An
        internal-consistency identity, not finiteness.
        """
        cutoff, bandwidth = 0.0, 1.0
        bin_width = bandwidth / 25.0
        bins = np.arange(cutoff - bandwidth, cutoff + bandwidth + bin_width, bin_width)
        centers = 0.5 * (bins[:-1] + bins[1:])
        excluded = (centers > cutoff - bin_width) & (centers < cutoff + bin_width)
        CONST, EXCESS = 40, 30
        pts = []
        for c in centers:
            pts.extend([c] * CONST)
        pts.extend([float(centers[excluded][0])] * EXCESS)
        res = sp.general_bunching(
            pd.DataFrame({"r": np.asarray(pts, float)}),
            running="r",
            cutoff=cutoff,
            bandwidth=bandwidth,
            polynomial_order=4,
            n_boot=5,
            seed=0,
        )
        assert abs(res.naive_elasticity - res.bias_corrected_elasticity) < self.TOL, (
            f"naive {res.naive_elasticity!r} != corrected "
            f"{res.bias_corrected_elasticity!r} on a flat counterfactual"
        )


# ---------------------------------------------------------------------------
# B. Known-DGP recovery within ~10%
# ---------------------------------------------------------------------------


class TestRecovery:
    """Planted excess mass / elasticity recovered within tolerance."""

    # Smooth uniform base over [60, 140]; threshold 100, default bin
    # width ~1, default region [98, 102].  Planted bunchers in [98, 100]
    # all land inside the region.
    THRESHOLD = 100.0
    N_BASE = 60_000
    N_EXTRA = 2000

    def _dgp(self, seed):
        rng = np.random.default_rng(seed)
        base = rng.uniform(60.0, 140.0, self.N_BASE)
        bunchers = rng.uniform(98.0, 100.0, self.N_EXTRA)
        return pd.DataFrame({"z": np.concatenate([base, bunchers])})

    def test_raw_excess_recovers_planted_within_10pct(self):
        # rtol 0.10: the counterfactual polynomial must interpolate the
        # uniform shoulders and subtract the base mass the bunchers
        # displace.  Probed max relative error 4.9% over 6 seeds, so a
        # 10% band recovers the truth while a 20% estimate bias (-> rel
        # error ~0.20) breaks it.
        r = sp.bunching(
            self._dgp(2024),
            running_var="z",
            threshold=self.THRESHOLD,
            n_bins=40,
            poly_order=2,
            dt=0.10,
            n_bootstrap=30,
        )
        raw = r.model_info["excess_mass_raw"]
        rel_err = abs(raw - self.N_EXTRA) / self.N_EXTRA
        assert rel_err < 0.10, (
            f"raw excess {raw:.1f} vs planted {self.N_EXTRA} "
            f"(rel error {rel_err:.3f} >= 0.10)"
        )

    def test_elasticity_reported_when_dt_given(self):
        """dt -> a finite elasticity keyed in model_info (bunching.py:248).

        Pinned to a positive, finite value (the planted excess is
        positive), not merely present — a sign flip or NaN would fail.
        """
        r = sp.bunching(
            self._dgp(2024),
            running_var="z",
            threshold=self.THRESHOLD,
            n_bins=40,
            poly_order=2,
            dt=0.10,
            n_bootstrap=10,
        )
        assert "elasticity" in r.model_info
        eps = r.model_info["elasticity"]
        assert np.isfinite(eps) and eps > 0.0, (
            f"kink elasticity {eps!r} should be finite and positive "
            f"for a below-threshold mass"
        )

    def test_general_bunching_detects_excess(self):
        """general_bunching recovers a strongly nonzero elasticity.

        With a smooth uniform base + planted near-cutoff mass the
        corrected elasticity sits many SE from zero (probed |z| ~25), so
        it is a genuine recovery, not noise.
        """
        rng = np.random.default_rng(55)
        base = rng.uniform(-1.0, 1.0, 40_000)
        bunchers = rng.uniform(-0.04, 0.04, 1200)  # within bin_width=0.04
        df = pd.DataFrame({"r": np.concatenate([base, bunchers])})
        res = sp.general_bunching(
            df,
            running="r",
            cutoff=0.0,
            bandwidth=1.0,
            polynomial_order=4,
            n_boot=30,
            seed=7,
        )
        z = abs(res.bias_corrected_elasticity) / res.se
        assert z > 4.0, (
            f"corrected eps {res.bias_corrected_elasticity:.4f} only "
            f"{z:.1f} SE from zero — excess not detected"
        )
        assert res.bias_corrected_elasticity > 0.0


# ---------------------------------------------------------------------------
# C. Null: smooth density, NO notch -> ~zero excess
# ---------------------------------------------------------------------------


class TestNull:
    """A smooth density with no bunching yields ~zero excess mass.

    This is the required contrast to the recovery anchors: a method that
    fabricated mass on a smooth density would fail here.
    """

    def test_bunching_null_excess_near_zero(self):
        rng = np.random.default_rng(7)
        # Smooth uniform, no planted notch.
        df = pd.DataFrame({"z": rng.uniform(60.0, 140.0, 40_000)})
        r = sp.bunching(
            df,
            running_var="z",
            threshold=100.0,
            n_bins=40,
            poly_order=2,
            n_bootstrap=40,
        )
        # Normalised excess within 4 SE of zero (suite recovery band;
        # probed |z| < 1).  se>0 here so the band is informative.
        assert r.se > 0.0
        assert abs(r.estimate) <= 4.0 * r.se, (
            f"smooth-density excess {r.estimate:.4f} is {abs(r.estimate)/r.se:.1f}"
            f" SE from zero — false bunching on a notch-free density"
        )

    def test_general_bunching_null_elasticity_near_zero(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({"r": rng.uniform(-1.0, 1.0, 40_000)})
        res = sp.general_bunching(
            df,
            running="r",
            cutoff=0.0,
            bandwidth=1.0,
            polynomial_order=4,
            n_boot=40,
            seed=7,
        )
        z = abs(res.bias_corrected_elasticity) / res.se
        assert z <= 4.0, (
            f"smooth-density corrected eps "
            f"{res.bias_corrected_elasticity:.4f} is {z:.1f} SE from "
            f"zero — false bunching"
        )


# ---------------------------------------------------------------------------
# D. Internal-consistency identity
# ---------------------------------------------------------------------------


class TestInternalConsistency:
    """estimate == raw excess / counterfactual_at_threshold (bunching.py:242)."""

    TOL = 1e-9

    def test_normalised_equals_raw_over_cf(self):
        rng = np.random.default_rng(99)
        base = rng.uniform(60.0, 140.0, 30_000)
        bunchers = rng.uniform(98.0, 100.0, 1500)
        df = pd.DataFrame({"z": np.concatenate([base, bunchers])})
        r = sp.bunching(
            df,
            running_var="z",
            threshold=100.0,
            n_bins=40,
            poly_order=3,
            n_bootstrap=10,
        )
        mi = r.model_info
        cf0 = mi["counterfactual_at_threshold"]
        assert cf0 > 0.0  # guard the division is well-posed
        expected = mi["excess_mass_raw"] / cf0
        assert abs(r.estimate - expected) < self.TOL, (
            f"reported B {r.estimate!r} != raw/cf {expected!r} — the "
            f"normalisation identity (bunching.py:242) is broken"
        )
        # And the public mirror field agrees.
        assert abs(mi["excess_mass_normalised"] - r.estimate) < self.TOL
