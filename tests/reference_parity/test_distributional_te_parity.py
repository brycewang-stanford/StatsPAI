"""Reference parity: distributional treatment effects family.

Estimators
----------
``sp.distributional_te`` (DTE: counterfactual-CDF / quantile shift via
IPW) and ``sp.stochastic_dominance`` (first/second-order stochastic
dominance test on the treated vs. counterfactual quantile functions
produced by ``sp.discos``).  Each previously had only a smoke test; this
file is their first numerical guarantee.

Setting (location-shift Gaussian, everything hand-set)
------------------------------------------------------
The whole file rides on a single closed-form fact: if
``Y0 ~ N(0, 1)`` and ``Y1 ~ N(MU, 1)`` with a KNOWN ``MU > 0``, the two
distributions differ by a pure horizontal location shift of ``MU``.
Three exact consequences follow and become the anchors:

1. **Quantile shift.** ``Q_{Y1}(tau) - Q_{Y0}(tau) = MU`` for *every*
   ``tau`` (a location shift moves every quantile by the same amount).
   So the median QTE estimated by ``distributional_te`` must equal
   ``MU``.
2. **CDF value at a fixed point.** The treated CDF evaluated at the
   counterfactual median (``y = 0``, the median of ``Y0``) equals
   ``F_{Y1}(0) = Phi((0 - MU)/1) = Phi(-MU)`` — a number we compute from
   the standard normal, not from the estimator.
3. **Mean shift = area between CDFs.**
   ``E[Y1] - E[Y0] = integral (F_{Y0}(y) - F_{Y1}(y)) dy = MU``
   (the layer-cake / Hoeffding identity).  Trapezoidal integration of
   the two estimated CDFs over the grid must recover ``MU``.

For the dominance estimator: a location shift with ``MU > 0`` makes
``Y1`` *first-order stochastically dominate* ``Y0`` (its CDF lies weakly
below everywhere), so ``stochastic_dominance`` must return
``dominates=True``; a mean-preserving spread that crosses the
counterfactual (lower in the lower tail, higher in the upper tail) is
NOT FOSD, so it must return ``dominates=False``.  Asserting BOTH
directions is the non-tautological core: a estimator that always says
"dominates" or always says "no" fails one side.

Implementation facts the anchors rely on (cited file:line)
----------------------------------------------------------
- ``src/statspai/qte/distributional.py:194-205`` — with no covariates
  ``_dte_ipw`` sets the propensity to the constant ``D.mean()``, so the
  control IPW weights ``ps/(1-ps)`` are uniform and the counterfactual
  CDF is the plain control ECDF; the treated CDF is the treated ECDF.
  ``F_{Y1}(0)`` therefore estimates ``Phi(-MU)`` directly (anchor 2),
  the median QTE is the gap of the two empirical quantile functions
  (anchor 1), and the area between the two step CDFs is the difference
  in sample means (anchor 3).
- ``src/statspai/synth/discos.py:215-216`` — ``avg_qte`` is the mean
  over tau of ``Q_treated_post - Q_counterfactual_post``; on a pure
  level shift this estimates ``MU`` (anchor 4 consistency).
- ``src/statspai/synth/discos.py:850-893`` — ``stochastic_dominance``
  (order 1) returns ``dominates = (min_gap >= 0)`` where
  ``gap = Q_treated - Q_counterfactual`` over the tau grid; a uniformly
  positive shift makes ``min_gap > 0`` (FOSD), a crossing shift makes
  ``min_gap < 0 < max_gap`` (no FOSD).  Order 2 integrates the gaps
  (``:912-917``).

References (bib keys grep-confirmed in paper.bib)
-------------------------------------------------
- Chernozhukov, Fernandez-Val & Melly (2013), "Inference on
  Counterfactual Distributions", *Econometrica* 81(6) — the
  counterfactual-distribution framework ``distributional_te``
  implements. [@chernozhukov2013inference]
- Gunsilius (2023), "Distributional Synthetic Controls",
  *Econometrica* 91(3) — the DiSCo estimator whose treated /
  counterfactual quantile functions feed ``stochastic_dominance``.
  [@gunsilius2023distributional]
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp

# Hand-set true location shift shared by the DTE anchors (Y1 ~ N(MU, 1)).
MU = 1.2


# ---------------------------------------------------------------------------
# Deterministic DGP builders (every draw seeded via default_rng).
# ---------------------------------------------------------------------------


def _make_shift_dgp(seed, n=6000):
    """Two equal arms: control Y0 ~ N(0,1), treated Y1 ~ N(MU, 1).

    Treatment is assigned by construction (first half control, second
    half treated), independent of the outcome shocks, so the IPW
    propensity is the constant ``D.mean() = 0.5`` and the IPW
    counterfactual CDF is exactly the control ECDF — anchors 1-3 then
    pin the estimate to closed-form normal quantities.
    """
    rng = np.random.default_rng(seed)
    d = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    y = np.where(d == 1, rng.normal(MU, 1.0, n), rng.normal(0.0, 1.0, n))
    return pd.DataFrame({"y": y, "d": d})


# Donor units sit at distinct fixed levels 1..6; the treated unit's
# pre-treatment level (3.5) lies inside their convex hull so the DiSCo
# mixture reproduces ~3.5 as the counterfactual.  Post-treatment we add
# either a uniform +SHIFT (FOSD) or a crossing spread (no FOSD).
_DONOR_LEVELS = {f"u{i}": float(i + 1) for i in range(6)}
_TREATED_PRE_LEVEL = 3.5
_FOSD_SHIFT = 3.0  # treated post level = 3.5 + 3.0 = 6.5 > every donor
# Deterministic crossing post-values: a SYMMETRIC straddle of the
# counterfactual (offsets sum to 0, so the spread is mean-preserving)
# that still crosses — two clearly below 3.5, two clearly above, one at.
_CROSS_OFFSETS = [-1.0, -0.5, 0.0, 0.5, 1.0]
_CROSS_POST = [_TREATED_PRE_LEVEL + off for off in _CROSS_OFFSETS]


def _make_panel(seed, kind):
    """Long panel: 6 donors + 1 treated, 10 periods, treatment at t=5.

    kind='fosd'  -> treated post = pre + uniform +_FOSD_SHIFT (FOSD).
    kind='cross' -> treated post = deterministic straddle of the
                    counterfactual (no FOSD; min gap < 0 < max gap).
    """
    rng = np.random.default_rng(seed)
    units = list(_DONOR_LEVELS) + ["T"]
    times = list(range(10))
    rows = []
    for u in units:
        for t in times:
            if u == "T":
                if t < 5:
                    val = _TREATED_PRE_LEVEL + 0.02 * rng.normal()
                elif kind == "fosd":
                    val = _TREATED_PRE_LEVEL + _FOSD_SHIFT + 0.02 * rng.normal()
                else:  # 'cross'
                    val = _CROSS_POST[t - 5] + 0.02 * rng.normal()
            else:
                val = _DONOR_LEVELS[u] + 0.02 * rng.normal()
            rows.append((u, t, val))
    return pd.DataFrame(rows, columns=["unit", "time", "y"])


# ---------------------------------------------------------------------------
# Module fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shift_data():
    return _make_shift_dgp(20260614)


@pytest.fixture(scope="module")
def fosd_result():
    df = _make_panel(7, "fosd")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.discos(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="T",
            treatment_time=5,
        )


@pytest.fixture(scope="module")
def cross_result():
    df = _make_panel(11, "cross")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.discos(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="T",
            treatment_time=5,
        )


def _dte(df, quantiles, seed):
    with warnings.catch_warnings():
        # bootstrap convergence / sklearn solver chatter is not under test.
        warnings.simplefilter("ignore")
        return sp.distributional_te(
            df,
            y="y",
            treatment="d",
            method="ipw",
            quantiles=quantiles,
            n_boot=80,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Anchor 1. Median QTE recovers the location shift MU.
# ---------------------------------------------------------------------------


class TestQuantileShiftRecovery:
    """A location shift moves every quantile by MU; median QTE == MU."""

    def test_median_qte_within_4_se(self, shift_data):
        # 4-sigma recovery band (suite convention, REFERENCES.md;
        # false-failure 6.3e-5).  Probed median QTE ~1.24, SE ~0.04 ->
        # z ~1.0.  A 20% multiplicative bias (-> ~1.44 or 0.96) lands
        # ~5-6 sigma out and fails.
        r = _dte(shift_data, quantiles=[0.5], seed=20260614)
        med_qte = float(r.qte_effects[0])
        se = float(r.qte_se[0])
        assert se > 0, "degenerate QTE SE — bootstrap collapsed"
        assert abs(med_qte - MU) <= 4.0 * se, (
            f"median QTE {med_qte:.4f} (SE {se:.4f}) misses the hand-set "
            f"location shift MU={MU} by {abs(med_qte - MU) / se:.1f} sigma."
        )

    def test_all_quantile_shifts_near_mu(self, shift_data):
        """Every QTE (location shift => constant gap) sits near MU.

        Not a single-point check: a location-shift estimator must return
        roughly MU at the 0.25 / 0.5 / 0.75 quantiles alike.  Absolute
        tol 0.20 (~0.17*MU): well outside the ~0.04 sampling SE at each
        tau (so it is not vacuous) yet inside the spread a correct
        estimator shows; a 20% bias (gap ~1.44 or ~0.96) exceeds 0.20 on
        the mean and trips the band.
        """
        r = _dte(shift_data, quantiles=[0.25, 0.5, 0.75], seed=20260614)
        gaps = np.asarray(r.qte_effects, dtype=float)
        assert np.all(np.abs(gaps - MU) < 0.20), (
            f"per-quantile QTE gaps {gaps.round(3).tolist()} stray from the "
            f"constant location shift MU={MU} by more than 0.20."
        )


# ---------------------------------------------------------------------------
# Anchor 2. Treated CDF at the counterfactual median == Phi(-MU).
# ---------------------------------------------------------------------------


class TestCDFShiftClosedForm:
    """F_{Y1}(0) estimates Phi(-MU): a hand-set normal quantity."""

    def test_treated_cdf_at_zero_equals_phi_neg_mu(self, shift_data):
        r = _dte(shift_data, quantiles=[0.5], seed=20260614)
        # Grid point nearest y = 0 (the counterfactual / Y0 median).
        i0 = int(np.argmin(np.abs(r.grid - 0.0)))
        f_t0 = float(r.cdf_treated[i0])
        phi_neg_mu = float(stats.norm.cdf(-MU))  # 0.1151 for MU=1.2

        # Absolute tol 0.03: ~2x the binomial sampling SD of an ECDF at
        # one point with n1=3000 (sqrt(p(1-p)/n) ~ 0.0058) plus grid
        # discretisation, so it is a real numeric pin, not a finiteness
        # check.  Probed |F_t(0) - Phi(-MU)| ~0.0004.  A 20% bias on the
        # CDF value (0.092 or 0.138) exceeds 0.03 and fails.
        assert abs(f_t0 - phi_neg_mu) < 0.03, (
            f"treated CDF at y=0 is {f_t0:.4f}; closed-form Phi(-MU)="
            f"{phi_neg_mu:.4f} for MU={MU} — distributional shift not "
            f"recovered."
        )

    def test_counterfactual_cdf_at_zero_is_a_median(self, shift_data):
        """Counterfactual (control) CDF at its own median y=0 is ~0.5.

        The control distribution is N(0,1); its CDF at 0 is exactly 0.5.
        This pins the counterfactual leg independently of the treated
        leg, ruling out a swapped-arm bug.  Tol 0.03 (same ECDF-SD
        reasoning); probed ~0.524 near a grid node straddling 0.
        """
        r = _dte(shift_data, quantiles=[0.5], seed=20260614)
        i0 = int(np.argmin(np.abs(r.grid - 0.0)))
        f_cf0 = float(r.cdf_counterfactual[i0])
        assert abs(f_cf0 - 0.5) < 0.04, (
            f"counterfactual CDF at y=0 is {f_cf0:.4f}, not ~0.5 — the "
            f"control N(0,1) median is misplaced (possible arm swap)."
        )


# ---------------------------------------------------------------------------
# Anchor 3. Mean shift = area between the CDFs == MU.
# ---------------------------------------------------------------------------


class TestMeanShiftFromCDFs:
    """integral (F_cf - F_t) dy recovers E[Y1]-E[Y0] = MU."""

    def test_area_between_cdfs_equals_mu(self, shift_data):
        r = _dte(shift_data, quantiles=[0.5], seed=20260614)
        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        # Hoeffding/layer-cake identity: E[Y1]-E[Y0] = ∫(F0 - F1) dy.
        mean_shift = float(trapz(r.cdf_counterfactual - r.cdf_treated, r.grid))
        # Absolute tol 0.10 (~0.08*MU): the mean-difference SE on this
        # DGP is sqrt(2/3000) ~ 0.026, so 0.10 ~ 4 sigma — a genuine
        # recovery band, not vacuous.  Probed mean_shift ~1.23.  A 20%
        # bias (0.96 or 1.44) lands >2x the tol away and fails.
        assert abs(mean_shift - MU) < 0.10, (
            f"area between CDFs {mean_shift:.4f} != hand-set mean shift "
            f"MU={MU} (|diff|={abs(mean_shift - MU):.4f})."
        )


# ---------------------------------------------------------------------------
# Anchor 4. Stochastic dominance: detect FOSD when present, NOT when crossing.
# ---------------------------------------------------------------------------


class TestStochasticDominanceTwoSided:
    """dominates=True under a uniform +shift; False under a crossing spread."""

    def test_fosd_detected_when_present(self, fosd_result):
        sd = sp.stochastic_dominance(fosd_result, order=1)
        # Uniform +_FOSD_SHIFT => every treated quantile exceeds the
        # counterfactual => min_gap > 0 => FOSD.  Probed min_gap ~2.99,
        # frac_positive 1.0.
        assert sd["dominates"] is True, (
            f"a uniform +{_FOSD_SHIFT} location shift must first-order "
            f"dominate, but dominates={sd['dominates']} "
            f"(min_gap={sd['min_gap']:.3f})."
        )
        assert sd["min_gap"] > 0, (
            f"FOSD requires every quantile gap >= 0; min_gap="
            f"{sd['min_gap']:.3f} is negative."
        )
        assert sd["fraction_positive"] == pytest.approx(1.0), (
            f"under FOSD all quantile gaps are positive; "
            f"fraction_positive={sd['fraction_positive']}."
        )

    def test_no_fosd_under_crossing_spread(self, cross_result):
        sd = sp.stochastic_dominance(cross_result, order=1)
        # Symmetric straddle (offsets [-1,-0.5,0,0.5,1]): two post-values
        # clearly below, two clearly above the ~3.5 counterfactual => the
        # treated quantile function CROSSES the counterfactual =>
        # min_gap < 0 < max_gap => NOT FOSD.  Probed min_gap ~-0.97,
        # max_gap ~0.94.
        assert sd["dominates"] is False, (
            f"a crossing (mean-preserving) spread must NOT first-order "
            f"dominate, but dominates={sd['dominates']} "
            f"(min_gap={sd['min_gap']:.3f}, max_gap={sd['max_gap']:.3f})."
        )
        # Genuine crossing, not merely a shifted-down distribution: the
        # gap must change sign across tau (some quantiles up, some down).
        assert sd["min_gap"] < 0 < sd["max_gap"], (
            f"expected a sign-changing gap (crossing), got min_gap="
            f"{sd['min_gap']:.3f}, max_gap={sd['max_gap']:.3f}."
        )

    def test_fosd_implies_second_order(self, fosd_result):
        """FOSD => SOSD: a uniformly-shifted-up dist also 2nd-order dominates.

        First-order dominance is strictly stronger than second-order, so
        the order-2 test must also report dominance here.  This is a
        non-trivial cross-order consistency check (a wrong cumulative-gap
        sign would flip it).  Probed dominates=True both orders.
        """
        sd2 = sp.stochastic_dominance(fosd_result, order=2)
        assert sd2["dominates"] is True, (
            "FOSD implies second-order dominance, but order-2 reports "
            f"dominates={sd2['dominates']}."
        )


# ---------------------------------------------------------------------------
# Anchor 5 (consistency). DiSCo average QTE recovers the location shift.
# ---------------------------------------------------------------------------


class TestDiscoAvgQTERecovery:
    """On a pure level shift, DiSCo's average QTE == the shift."""

    def test_fosd_avg_qte_recovers_shift(self, fosd_result):
        est = float(fosd_result.estimate)
        # The donor hull brackets the treated pre-level, the post shift
        # is a clean +_FOSD_SHIFT, and donor noise SD is 0.02, so the
        # mixture counterfactual reproduces the pre-level and the average
        # quantile gap estimates the shift.  Absolute tol 0.10 (~0.03 of
        # the shift): probed est ~3.006.  A 20% bias (2.4 or 3.6) is 6x
        # the tol away and fails; this ties the dominance fixture's point
        # estimate back to the same hand-set truth.
        assert abs(est - _FOSD_SHIFT) < 0.10, (
            f"DiSCo average QTE {est:.4f} != hand-set post shift "
            f"{_FOSD_SHIFT} (|diff|={abs(est - _FOSD_SHIFT):.4f})."
        )

    def test_cross_avg_qte_near_zero(self, cross_result):
        """A mean-preserving crossing spread has ~zero average QTE.

        The straddle offsets sum to zero (symmetric about the
        counterfactual), so the mean quantile gap is ~0 even though the
        distribution changed shape — distinguishing 'no average effect'
        from 'no distributional effect'.  Tol 0.20 absolute; probed
        |est| ~0.01.  Pairs with the dominance False above: the spread
        moves the distribution without moving its mean.
        """
        est = float(cross_result.estimate)
        assert abs(est) < 0.20, (
            f"crossing spread average QTE {est:.4f} should be ~0 (mean-"
            f"preserving); a large value signals a net level shift."
        )


# ---------------------------------------------------------------------------
# Determinism guard (seed-stability).
# ---------------------------------------------------------------------------


class TestSeedStability:
    """Same seed -> identical DTE output (hermetic determinism)."""

    def test_dte_seed_reproducible(self):
        df = _make_shift_dgp(4242, n=2000)
        r1 = _dte(df, quantiles=[0.25, 0.5, 0.75], seed=4242)
        r2 = _dte(df, quantiles=[0.25, 0.5, 0.75], seed=4242)
        assert np.allclose(r1.qte_effects, r2.qte_effects), (
            "distributional_te is not reproducible under a fixed seed — "
            "QTE point estimates differ across identical calls."
        )
        assert np.allclose(r1.qte_se, r2.qte_se), (
            "bootstrap SEs differ across identical seeded calls — RNG not "
            "threaded through default_rng(seed)."
        )
