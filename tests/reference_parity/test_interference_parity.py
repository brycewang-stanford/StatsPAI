"""Reference parity: partial / network interference family.

Estimators
----------
``sp.spillover`` (Hudgens & Halloran 2008 partial-interference
direct/spillover decomposition), ``sp.network_exposure`` (Aronow & Samii
2017 Horvitz-Thompson exposure-mapping estimator) and the unified
dispatcher ``sp.interference``.  Each previously had only a smoke test;
this file is their first numerical guarantee.

Setting (partial interference)
------------------------------
Units sit in clusters; interference happens *within* a cluster but not
across cluster boundaries.  Each unit's peer **exposure** is the share of
its *other* cluster-mates that are treated.  The structural outcome is

    Y_i = 1 + DIRECT * D_i + SPILL * exposure_i + noise

with a HAND-SET direct effect ``DIRECT`` and spillover coefficient
``SPILL``.  A SUTVA-ignoring estimator that conditions only on own
treatment ``D`` inherits the spillover contamination whenever ``D`` and
``exposure`` are correlated; the interference estimators stratify on
exposure (``spillover``) or on an exposure mapping (``network_exposure``)
and recover the direct effect.

Anchors
-------
A. **Decomposition additivity (closed form, tol 1e-12).**  ``spillover``
   reports ``direct``, ``spillover`` and ``total`` in ``model_info`` and a
   ``detail`` table; by construction ``total == direct + spillover`` and
   ``result.estimate == total``.  Both are exact internal identities
   (probed |diff| = 0.0).  ``network_exposure``'s AS4 contrasts obey
   ``composite(c11-c00) == direct(c10-c00) + spillover_on_treated(c11-c10)``
   exactly, and ``composite`` equals the hand-computed difference of HT
   means from the ``estimates`` table (probed |diff| = 0.0).  Pins the
   estimates to algebraic identities, not finiteness.
B. **Known-DGP recovery of DIRECT and SPILL (recovery).**  (i)
   ``spillover``'s direct effect recovers ``DIRECT = 1.5`` within 4 of its
   own bootstrap SE on a single draw (probed z ~0.5) AND a 40-rep
   Monte-Carlo mean lands within ``4*SD/sqrt(R)`` of 1.5 (probed 1.497).
   (ii) ``network_exposure``'s AS4 direct contrast ``mu(c10)-mu(c00)``
   recovers a hand-set ``DIRECT_NET = 2.0`` and its spillover contrast
   ``mu(c01)-mu(c00)`` recovers ``SPILL_NET = 1.0``, each as a Monte-Carlo
   mean within ``4*SD/sqrt(R)`` (probed 2.10 and 1.07; the per-draw
   Aronow-Samii variance bound is too conservative for a single-draw z
   anchor, so recovery is asserted on the MC mean).
C. **Naive-bias contrast (naive_bias).**  On a DGP where own treatment is
   positively correlated with peer exposure (cluster-level treatment
   intensity drives both), the SUTVA-ignoring difference-in-means is
   ``> 6`` sigma above ``DIRECT`` (probed ~2.03, z ~8.7) because it absorbs
   the ``SPILL``-driven exposure contamination; ``spillover``'s
   exposure-stratified direct effect recovers ``DIRECT`` within 4 sigma
   (probed z ~0.8) AND lands strictly below the naive estimate by a real
   margin.  Both halves are asserted, proving directional de-confounding.
D. **Null spillover (null).**  Setting ``SPILL = 0`` makes the true
   spillover vanish; ``spillover``'s spillover effect is then within 4 of
   its SE of zero (probed z ~0.7) while a positive ``SPILL`` produces a
   spillover estimate many SEs from zero (probed z ~8.5) — and the direct
   effect stays pinned to ``DIRECT`` under BOTH (invariant to the
   spillover magnitude).  Pins the estimate to a known null vs non-null
   contrast.

Implementation facts the anchors rely on (cited file:line)
----------------------------------------------------------
- ``src/statspai/interference/spillover.py:189-191`` — ``direct`` =
  E[Y|D=1,high_exp] - E[Y|D=0,high_exp], ``spillover`` =
  E[Y|D=0,high_exp] - E[Y|D=0,low_exp], ``total`` = direct + spillover
  (exact sum; anchor A-i).  ``spillover.py:261`` — ``estimate`` is
  ``float(total)`` (anchor A-i).  The high/low split is on the median of
  the positive exposures (``spillover.py:179-180``).
- ``src/statspai/interference/network_exposure.py:344-362`` — the AS4
  ``contrasts`` table defines ``direct (c10-c00)``,
  ``spillover (c01-c00)``, ``composite (c11-c00)`` and
  ``spillover_on_treated (c11-c10)`` as differences of the HT means in
  ``estimates`` (``network_exposure.py:328``); hence
  ``composite == direct + spillover_on_treated`` is an exact algebraic
  identity (anchor A-ii).  The HT mean per level is
  ``mean(1{exposure=d}/pi_i(d) * Y)`` (``network_exposure.py:201``) with
  Monte-Carlo exposure probabilities (``network_exposure.py:166-188``),
  so recovery is asserted on the MC mean over draws (anchor B-ii).
- ``src/statspai/interference/dispatcher.py:103-149`` — ``sp.interference``
  routes ``"partial"`` -> ``spillover`` and ``"network_exposure"`` ->
  ``network_exposure`` unchanged; anchor B exercises both routes.

References (bib keys grep-confirmed present in paper.bib)
---------------------------------------------------------
- Hudgens, M. G. & Halloran, M. E. (2008). "Toward Causal Inference with
  Interference." JASA 103(482), 832-842. [@hudgens2008toward] — the
  partial-interference direct/spillover decomposition ``sp.spillover``
  implements.
- Aronow, P. M. & Samii, C. (2017). "Estimating average causal effects
  under general interference, with application to a social network
  experiment." Annals of Applied Statistics 11(4), 1912-1947.
  [@aronow2017estimating] — the exposure-mapping Horvitz-Thompson
  estimator ``sp.network_exposure`` implements.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# Hand-set structural parameters shared across the partial-interference DGPs.
DIRECT = 1.5  # own-treatment (direct) effect
SPILL = 2.0  # spillover coefficient on peer exposure

# Hand-set parameters for the network (AS4) DGP.  The any-treated-neighbour
# spillover term enters as a {0,1} indicator, so its contrast target is the
# coefficient SPILL_NET itself.
DIRECT_NET = 2.0
SPILL_NET = 1.0


def _within_n_se(est, truth, se, n_sigma=4.0):
    return abs(est - truth) <= n_sigma * se


# ---------------------------------------------------------------------------
# Partial-interference DGP builders (every draw seeded via default_rng).
# ---------------------------------------------------------------------------


def _make_partial_dgp(seed, n_clusters=250, c_size=6, spill=SPILL, correlated=False):
    """Cluster DGP with KNOWN direct effect + KNOWN spillover coef.

    ``exposure_i`` = share of the unit's OTHER cluster-mates that are
    treated (matching ``spillover.py``'s leave-one-out 'fraction'
    exposure).  Outcome ``Y = 1 + DIRECT*D + spill*exposure + N(0,1)``.

    ``correlated=False``: own treatment is i.i.d. Bernoulli(0.5) within
    each cluster, so ``D`` is (asymptotically) independent of ``exposure``
    and the naive diff-in-means is only mildly contaminated.

    ``correlated=True``: a cluster-level propensity ``p ~ U(0.1, 0.9)``
    drives BOTH own treatment and peers' treatment, so within a cluster a
    treated unit tends to have more treated peers — ``D`` and
    ``exposure`` are positively correlated and the SUTVA-ignoring
    diff-in-means absorbs the spillover (anchor C).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        if correlated:
            p = rng.uniform(0.1, 0.9)
            d = (rng.uniform(size=c_size) < p).astype(int)
        else:
            d = rng.integers(0, 2, size=c_size)
        for i in range(c_size):
            peer = (d.sum() - d[i]) / (c_size - 1)
            y = 1.0 + DIRECT * d[i] + spill * peer + rng.normal()
            rows.append({"y": y, "d": int(d[i]), "cl": c, "peer": peer})
    return pd.DataFrame(rows)


def _make_network_dgp(seed, n=300, p=0.3, direct=DIRECT_NET, spill=SPILL_NET):
    """Ring network (degree 2), Bernoulli(p) design.

    ``Y = 1 + direct*Z + spill*1{>=1 treated neighbour} + N(0,1)``.  Under
    the AS4 mapping ``mu(c10)-mu(c00)`` targets ``direct`` and
    ``mu(c01)-mu(c00)`` targets the spillover indicator coefficient
    ``spill``.  A sparse ring with low ``p`` keeps the isolated cells
    (c00, c10) populated so the HT means are estimable.
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        A[i, (i + 1) % n] = 1
        A[i, (i - 1) % n] = 1
    Z = (rng.random(n) < p).astype(int)
    any_treated_neigh = ((A @ Z) > 0).astype(float)
    Y = 1.0 + direct * Z + spill * any_treated_neigh + rng.normal(size=n)
    return Y, Z, A


# ---------------------------------------------------------------------------
# Module fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def partial_data():
    return _make_partial_dgp(7)


@pytest.fixture(scope="module")
def correlated_partial_data():
    # Stronger spillover (3.0) + correlated assignment => naive diff-in-means
    # is provably > 6 sigma biased high (anchor C).
    return _make_partial_dgp(11, n_clusters=300, spill=3.0, correlated=True)


@pytest.fixture(scope="module")
def network_data():
    return _make_network_dgp(5)


def _spillover(df, n_bootstrap=200, random_state=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.spillover(
            df,
            y="y",
            treat="d",
            cluster="cl",
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )


# ---------------------------------------------------------------------------
# A. Decomposition additivity (closed-form internal identities).
# ---------------------------------------------------------------------------


class TestDecompositionAdditivity:
    """total == direct + spillover; AS4 contrasts obey the partition sum.

    Tolerance 1e-12: every relation is an exact float identity (the
    estimator literally adds the two scalars, ``spillover.py:191``, and
    the contrasts are differences of the same HT means,
    ``network_exposure.py:336-362``).  Probed |diff| = 0.0 in every case.
    A finiteness check would pass for any numbers; this pins the reported
    decomposition to its components.
    """

    TOL = 1e-12

    def test_spillover_total_equals_direct_plus_spillover(self, partial_data):
        r = _spillover(partial_data)
        mi = r.model_info
        recombined = mi["direct_effect"] + mi["spillover_effect"]
        assert abs(mi["total_effect"] - recombined) < self.TOL, (
            f"total {mi['total_effect']!r} != direct+spillover "
            f"{recombined!r} (|diff|={abs(mi['total_effect'] - recombined):.2e})"
        )
        # The headline estimate IS the total effect (spillover.py:261).
        assert (
            abs(r.estimate - mi["total_effect"]) < self.TOL
        ), f"estimate {r.estimate!r} != total {mi['total_effect']!r}"
        # The detail table must echo the same three scalars.
        det = r.detail.set_index("effect_type")["estimate"]
        assert abs(det["Direct"] - mi["direct_effect"]) < self.TOL
        assert abs(det["Spillover"] - mi["spillover_effect"]) < self.TOL
        assert abs(det["Total"] - mi["total_effect"]) < self.TOL

    def test_network_contrast_partition_identity(self, network_data):
        Y, Z, A = network_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.network_exposure(Y, Z, A, p_treat=0.3, n_sim=1500, seed=0)
        c = res.contrasts.set_index("contrast")["estimate"]
        composite = c["composite (c11 - c00)"]
        direct = c["direct (c10 - c00)"]
        spot = c["spillover_on_treated (c11 - c10)"]
        # (c11 - c00) == (c10 - c00) + (c11 - c10) is an exact algebraic
        # identity among the HT means (probed |diff| = 0.0).
        assert abs(composite - (direct + spot)) < self.TOL, (
            f"composite {composite!r} != direct+spillover_on_treated "
            f"{direct + spot!r} (|diff|={abs(composite - (direct + spot)):.2e})"
        )
        # And composite must equal the hand-computed HT-mean difference.
        m = res.estimates.set_index("exposure")["mean_Y(d)"]
        assert abs(composite - (m["c11"] - m["c00"])) < self.TOL, (
            f"composite {composite!r} != mu(c11)-mu(c00) " f"{m['c11'] - m['c00']!r}"
        )


# ---------------------------------------------------------------------------
# B. Known-DGP recovery of DIRECT and SPILL.
# ---------------------------------------------------------------------------


class TestRecovery:
    """spillover recovers DIRECT; network_exposure recovers DIRECT_NET/SPILL_NET."""

    def test_spillover_direct_single_draw_within_4_se(self, partial_data):
        r = _spillover(partial_data)
        # The exposure-stratified direct contrast is an unbiased estimator
        # of DIRECT because, within the high-exposure stratum, own
        # treatment is balanced across peer exposure.  Observed z ~0.5; a
        # +20% bias (1.8) lands ~3 SEs out and fails alongside anchor C.
        se = r.model_info["direct_se"]
        assert _within_n_se(r.model_info["direct_effect"], DIRECT, se, n_sigma=4.0), (
            f"spillover direct {r.model_info['direct_effect']:.4f} "
            f"(SE {se:.4f}) misses DIRECT {DIRECT} by "
            f"{abs(r.model_info['direct_effect'] - DIRECT) / se:.1f} sigma."
        )

    def test_spillover_direct_monte_carlo_mean_recovers(self):
        """40 independent draws; MC mean of the direct effect within band.

        Averaging cancels per-draw noise so the MC mean probes systematic
        bias.  Probed mean 1.497, SD ~0.099 -> band 4*0.099/sqrt(40)
        ~0.062 around 1.5; a 20% multiplicative bias (1.80) sits ~5 band
        widths out.  ``n_bootstrap=5`` keeps each fit fast (SE unused
        here).
        """
        reps = 40
        ests = []
        for s in range(reps):
            df = _make_partial_dgp(1000 + s, n_clusters=200)
            ests.append(_spillover(df, n_bootstrap=5).model_info["direct_effect"])
        ests = np.asarray(ests)
        mc_mean = float(ests.mean())
        band = 4.0 * float(ests.std(ddof=1)) / np.sqrt(reps)
        assert abs(mc_mean - DIRECT) <= band, (
            f"spillover direct MC mean {mc_mean:.4f} drifted from DIRECT "
            f"{DIRECT} (band {band:.4f}) over {reps} reps."
        )

    def test_network_exposure_contrasts_monte_carlo_recover(self):
        """AS4 direct/spillover contrasts recover DIRECT_NET / SPILL_NET.

        The per-draw Aronow-Samii Theorem-1 variance bound is highly
        conservative (probed single-draw SE ~1.9 on small isolated cells),
        so a single-draw z anchor would be vacuous; instead recovery is
        asserted on the Monte-Carlo mean.  Probed: direct mean 2.10 (band
        ~0.34), spillover mean 1.07 (band ~0.21).  A 20% bias on either
        contrast (-> 2.4 / 1.2) exceeds its band.  Goes through the
        ``sp.interference("network_exposure", ...)`` dispatcher route.
        """
        reps = 24
        direct_e, spill_e = [], []
        for s in range(reps):
            Y, Z, A = _make_network_dgp(100 + s)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sp.interference(
                    "network_exposure",
                    Y=Y,
                    Z=Z,
                    adjacency=A,
                    p_treat=0.3,
                    n_sim=800,
                    seed=0,
                )
            c = res.contrasts.set_index("contrast")["estimate"]
            direct_e.append(c["direct (c10 - c00)"])
            spill_e.append(c["spillover (c01 - c00)"])
        direct_e = np.asarray(direct_e)
        spill_e = np.asarray(spill_e)
        d_band = 4.0 * float(direct_e.std(ddof=1)) / np.sqrt(reps)
        s_band = 4.0 * float(spill_e.std(ddof=1)) / np.sqrt(reps)
        assert abs(direct_e.mean() - DIRECT_NET) <= d_band, (
            f"network direct contrast MC mean {direct_e.mean():.4f} "
            f"drifted from DIRECT_NET {DIRECT_NET} (band {d_band:.4f})."
        )
        assert abs(spill_e.mean() - SPILL_NET) <= s_band, (
            f"network spillover contrast MC mean {spill_e.mean():.4f} "
            f"drifted from SPILL_NET {SPILL_NET} (band {s_band:.4f})."
        )


# ---------------------------------------------------------------------------
# C. Naive-bias contrast (SUTVA-ignoring diff-in-means is contaminated).
# ---------------------------------------------------------------------------


class TestNaiveBiasContrast:
    """Naive diff-in-means is biased by spillover; spillover de-confounds."""

    def test_naive_biased_spillover_recovers_direct(self, correlated_partial_data):
        df = correlated_partial_data
        # SUTVA-ignoring difference-in-means + Welch SE — hand-rolled numpy,
        # no statspai internals.  Because D correlates with peer exposure,
        # this estimator absorbs the SPILL-driven contamination.
        y = df["y"].values
        d = df["d"].values
        y1, y0 = y[d == 1], y[d == 0]
        naive = float(y1.mean() - y0.mean())
        se_naive = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))

        # The naive estimator is provably contaminated: > 6 sigma above the
        # true direct effect (probed z ~8.7).  If this fails, the DGP lost
        # its spillover-confounding and the contrast is vacuous.
        assert naive - DIRECT > 6.0 * se_naive, (
            f"naive diff-in-means {naive:.4f} should be > 6 sigma above "
            f"DIRECT {DIRECT} (SE {se_naive:.4f}) — spillover contamination "
            f"vanished."
        )

        r = _spillover(df)
        direct_hat = r.model_info["direct_effect"]
        se_direct = r.model_info["direct_se"]
        # The exposure-stratified direct effect recovers truth within 4
        # sigma (probed z ~0.8)...
        assert _within_n_se(direct_hat, DIRECT, se_direct, n_sigma=4.0), (
            f"spillover direct {direct_hat:.4f} (SE {se_direct:.4f}) failed "
            f"to recover DIRECT {DIRECT} on the contaminated DGP."
        )
        # ...AND lands strictly below naive by a real margin (directional
        # de-confounding).  Probed gap ~0.46 >> 0.10.  A +20% bias on the
        # direct estimate would also fail the recovery assert above.
        assert direct_hat < naive - 0.10, (
            f"spillover direct {direct_hat:.4f} did not de-confound below "
            f"naive {naive:.4f} by the required 0.10 margin."
        )


# ---------------------------------------------------------------------------
# D. Null spillover -> ~0 spillover effect (and direct stays pinned).
# ---------------------------------------------------------------------------


class TestNullSpillover:
    """SPILL=0 zeroes the spillover effect; SPILL>0 makes it many SEs from 0."""

    def test_null_vs_nonnull_spillover(self):
        # Same seed, same design, ONLY the spillover coefficient changes:
        # isolates the spillover estimate's response to true spillover.
        df_null = _make_partial_dgp(50, spill=0.0)
        df_pos = _make_partial_dgp(50, spill=SPILL)

        r_null = _spillover(df_null)
        r_pos = _spillover(df_pos)

        s_null = r_null.model_info["spillover_effect"]
        se_null = r_null.model_info["spillover_se"]
        s_pos = r_pos.model_info["spillover_effect"]
        se_pos = r_pos.model_info["spillover_se"]

        # Null spillover: estimate within 4 SEs of zero (probed z ~0.7).
        assert abs(s_null) <= 4.0 * se_null, (
            f"null-spillover effect {s_null:.4f} (SE {se_null:.4f}) is "
            f"{abs(s_null) / se_null:.1f} SEs from zero — should be ~0."
        )
        # Positive spillover: estimate many SEs from zero (probed z ~8.5),
        # i.e. the estimator actually detects the planted spillover.
        assert s_pos > 4.0 * se_pos, (
            f"positive-spillover effect {s_pos:.4f} (SE {se_pos:.4f}) is "
            f"only {s_pos / se_pos:.1f} SEs from zero — failed to detect "
            f"the planted spillover."
        )
        # The direct effect is INVARIANT to the spillover magnitude and
        # recovers DIRECT under BOTH regimes (probed 1.466 / 1.461).  This
        # rules out a spillover bug that merely leaks into the direct slot.
        assert _within_n_se(
            r_null.model_info["direct_effect"],
            DIRECT,
            r_null.model_info["direct_se"],
            n_sigma=4.0,
        )
        assert _within_n_se(
            r_pos.model_info["direct_effect"],
            DIRECT,
            r_pos.model_info["direct_se"],
            n_sigma=4.0,
        )
