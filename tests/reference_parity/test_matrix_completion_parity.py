"""Reference parity: matrix-completion causal panel family.

Estimators
----------
``sp.matrix_completion`` (article alias, ``d`` -> ``treat``) and
``sp.mc_panel`` — both route to ``MCPanel.fit`` (soft-imputed
nuclear-norm completion, Athey et al. 2021).  Each previously had only a
smoke test; this file is their first numerical anchor.

Setting
-------
``MCPanel`` pivots the long panel into an ``(N, T)`` outcome matrix
``Y`` and a treatment mask ``W``.  It fits a low-rank ``L`` to the
CONTROL cells (``W == 0``) by soft-impute — repeatedly SVD-ing
``Z = where(control, Y, L)`` and soft-thresholding the singular values
by ``lambda_reg`` (with an optional hard ``max_rank`` cap) — then
reports ATT ``= mean(Y - L)`` over the treated cells
(``mc_panel.py:251-263``).  ``model_info`` exposes the imputed
``completed_matrix`` (``L``), ``effective_rank`` (``# s_i > 1e-6``,
``mc_panel.py:310-311``) and the per-cell ``treatment_effects_matrix``.

Two regimes exercise that machinery:

* **Pure completion.**  Take a KNOWN rank-2 matrix
  ``M = a⊗b + c⊗e`` (probed exact rank 2; singular values
  ``18.6, 15.6, 0, ...``), observe ``Y = M + noise`` and HOLD OUT ~30%
  of cells by labelling them ``treat == 1``.  The held-out cells are
  invisible to the ``L`` fit (only control cells enter ``Omega``), so
  ``completed_matrix`` is the estimator's reconstruction of ``M`` on the
  masked cells — exactly the matrix-completion recovery problem.
* **Causal panel.**  A staggered treated/control panel whose control
  surface is a known rank-2 trend ``M``; treated post-cells get a
  hand-set additive effect ``TAU``.  The true counterfactual is ``M``,
  so the ATT must recover ``TAU`` while a trend-ignoring naive contrast
  is biased.

Anchors
-------
A. **Known-rank-2 recovery within the noise floor** (recovery).  With
   ``max_rank=2`` and a small ``lambda``, the relative Frobenius error
   ``||L - M||_F / ||M||_F`` on the held-out (masked) cells is below
   the 0.05 noise-floor bar (probed ~0.017 at noise sd 0.02).  Pins the
   reconstruction to the HAND-SET ``M``, not to finiteness.
B. **Noiseless near-exact collapse** (closed_form).  With ``Y = M``
   exactly (no noise), ``max_rank=2`` and ``lambda=1e-4``, the relative
   Frobenius error on the masked cells collapses to ~1e-5 — a
   near-closed-form recovery (rtol 1e-2 with ~3 orders of headroom).
C. **Singular-value gap / rank-2 recovery** (consistency).  Run with
   ``lambda`` ONLY (no ``max_rank`` cap), so the rank must EMERGE from
   the data: at the converged fixed point the data matrix's 3rd
   singular value is a genuine nonzero (probed 0.78, vs raw zero-filled
   5.13) that the ``lambda=1.0`` soft-threshold drives to exactly 0,
   leaving ``effective_rank == 2`` and ``s[2]/s[1] == 0``.  A solver
   that failed to converge / threshold would report rank > 2 — this is
   not a tautology of ``max_rank``.
D. **Causal ATT recovery + naive-bias contrast** (recovery + naive_bias).
   On the staggered panel ``sp.mc_panel`` recovers the hand-set
   ``TAU = 3.0`` within 4 of its own bootstrap SE (probed z ~0.2),
   AND a trend-ignoring naive pre/post contrast on the treated units is
   biased high by > 1.5 (probed naive ~6.7 vs truth 3.0); we assert
   BOTH — the method de-confounds the low-rank trend the naive contrast
   absorbs.  ``sp.matrix_completion`` (the ``d``-named alias) returns
   the SAME estimate, pinning the alias mapping.
E. **Determinism / seed-stability** (consistency).  ``random_state``
   pins the bootstrap RNG, so two identical calls return bitwise-equal
   ``estimate`` and ``se`` (probed diff 0.0).

Implementation facts the anchors rely on (cited file:line)
----------------------------------------------------------
- ``mc_panel.py:234-235`` — ``Omega = (W == 0) & ~isnan(Y)``: only
  CONTROL cells fit ``L``; treated/masked cells are held out (anchors
  A-C exploit this to turn ``treat`` into a completion mask).
- ``mc_panel.py:341-378`` ``_soft_impute`` — ``Z = where(Omega, Y, L)``,
  SVD, ``s_thresh = max(s - lambda, 0)``, optional ``max_rank`` cap,
  reconstruct ``U * s_thresh @ Vt``; the lambda-only run (C) leaves the
  rank to emerge from thresholding.
- ``mc_panel.py:259-263`` — ``tau = Y - L`` on treated cells,
  ``att = mean(tau)`` (anchor D recovery target).
- ``mc_panel.py:310-322`` — ``effective_rank`` and ``completed_matrix``
  exposed via ``model_info`` (anchors A-C read these).
- ``mc_panel.py:266-284`` — bootstrap reseeded by ``random_state``
  (``np.random.RandomState``), making the SE deterministic (anchor E).
- ``_article_aliases.py:547-596`` — ``sp.matrix_completion(d=...)``
  forwards to ``mc_panel(treat=d, ...)`` with no numerical code of its
  own (anchor D alias check).

References (bib key verified present in paper.bib via grep)
-----------------------------------------------------------
- Athey, Bayati, Doudchenko, Imbens & Khosravi (2021), "Matrix
  Completion Methods for Causal Panel Data Models", JASA 116(536),
  1716-1730. [@athey2021matrix]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# Hand-set truths shared across the file.
TAU = 3.0  # additive treatment effect on treated cells (causal anchor D)


# ---------------------------------------------------------------------------
# DGP builders (every draw seeded via default_rng).
# ---------------------------------------------------------------------------


def _rank2_completion_dgp(seed, noise_sd, N=25, T=20, mask_frac=0.30):
    """KNOWN rank-2 matrix M = a⊗b + c⊗e; ~``mask_frac`` cells held out.

    Returns the long panel (``treat`` == the held-out mask), the true
    low-rank ``M`` and the boolean ``mask``.  Held-out cells are
    invisible to the ``L`` fit (mc_panel.py:234), so ``completed_matrix``
    reconstructs ``M`` on those cells — the matrix-completion problem.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=N)
    b = rng.normal(size=T)
    c = rng.normal(size=N)
    e = rng.normal(size=T)
    M = np.outer(a, b) + np.outer(c, e)  # exact rank 2 by construction
    Y = M + (0.0 if noise_sd == 0 else rng.normal(scale=noise_sd, size=(N, T)))
    mask = rng.uniform(size=(N, T)) < mask_frac
    rows = [
        {"unit": i, "time": t, "y": float(Y[i, t]), "d": int(mask[i, t])}
        for i in range(N)
        for t in range(T)
    ]
    return pd.DataFrame(rows), M, mask


def _causal_panel_dgp(seed, N=30, T=12, T0=8, n_treated=6, noise_sd=0.05):
    """Staggered panel: rank-2 control trend + hand-set additive ``TAU``.

    The control surface ``M = level + loading * ramp`` is rank 2 with a
    strong, heterogeneous, DETERMINISTIC time ramp (so a trend-ignoring
    naive pre/post contrast is reliably biased).  The last ``n_treated``
    units are treated from period ``T0`` on, with ``Y += TAU`` there;
    their counterfactual is exactly ``M`` -> ATT == TAU.
    """
    rng = np.random.default_rng(seed)
    level = rng.normal(size=N)
    loading = rng.uniform(0.5, 2.0, size=N)
    ramp = np.linspace(0.0, 5.0, T)
    M = level[:, None] + loading[:, None] * ramp[None, :]  # rank 2
    treated_units = np.arange(N) >= (N - n_treated)
    W = np.zeros((N, T), dtype=int)
    for i in range(N):
        if treated_units[i]:
            W[i, T0:] = 1
    Y = M + rng.normal(scale=noise_sd, size=(N, T))
    Y[W == 1] += TAU
    rows = [
        {"unit": i, "time": t, "y": float(Y[i, t]), "d": int(W[i, t])}
        for i in range(N)
        for t in range(T)
    ]
    return pd.DataFrame(rows), Y, W, treated_units, T0


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def noisy_completion():
    # noise sd 0.02; probed rel-Fro on masked cells ~0.017, eff_rank 2.
    return _rank2_completion_dgp(seed=20260614, noise_sd=0.02)


@pytest.fixture(scope="module")
def noiseless_completion():
    return _rank2_completion_dgp(seed=11, noise_sd=0.0)


@pytest.fixture(scope="module")
def causal_panel():
    return _causal_panel_dgp(seed=2027)


def _masked_rel_fro(L, M, mask):
    """Relative Frobenius error of L vs M restricted to masked cells."""
    return float(np.linalg.norm((L - M)[mask]) / np.linalg.norm(M[mask]))


# ---------------------------------------------------------------------------
# A. Known-rank-2 recovery within the noise floor.
# ---------------------------------------------------------------------------


class TestRecoveryFrobenius:
    """completed_matrix reconstructs the HAND-SET rank-2 M on held-out cells."""

    # 0.05: the irreducible reconstruction error is set by the noise
    # (sd 0.02) leaking into the fitted L; probed rel-Fro ~0.017, so 0.05
    # has ~3x headroom yet a +20% scaling of L (-> ~0.19) blows past it.
    NOISE_FLOOR = 0.05

    def test_masked_frobenius_within_noise_floor(self, noisy_completion):
        df, M, mask = noisy_completion
        r = sp.matrix_completion(
            df,
            y="y",
            d="d",
            unit="unit",
            time="time",
            max_rank=2,
            lambda_reg=0.05,
            n_bootstrap=3,
            random_state=0,
        )
        L = r.model_info["completed_matrix"]
        rel = _masked_rel_fro(L, M, mask)
        assert rel < self.NOISE_FLOOR, (
            f"matrix completion rel-Fro on held-out cells {rel:.4f} "
            f">= noise floor {self.NOISE_FLOOR}; the rank-2 reconstruction "
            f"drifted from the hand-set M."
        )

    def test_mask_fraction_is_genuine(self, noisy_completion):
        """Guard: the held-out fraction really is ~30% (recovery is
        non-trivial only if a real chunk of cells is hidden)."""
        df, _M, _mask = noisy_completion
        frac = float(df["d"].mean())
        assert 0.20 < frac < 0.40, f"mask fraction {frac:.3f} off design"


# ---------------------------------------------------------------------------
# B. Noiseless near-exact collapse.
# ---------------------------------------------------------------------------


class TestNoiselessCollapse:
    """Y == M exactly -> near-exact low-rank recovery (rtol ~1e-2)."""

    # rtol 1e-2: noiseless rank-2 completion with a tiny lambda is a
    # (near) closed-form fixed point; probed rel-Fro ~1.2e-5, ~3 orders
    # under 1e-2.  A 20% bias (rel-Fro -> ~0.19) fails by ~19x.
    RTOL = 1e-2

    def test_noiseless_recovery_near_exact(self, noiseless_completion):
        df, M, mask = noiseless_completion
        r = sp.matrix_completion(
            df,
            y="y",
            d="d",
            unit="unit",
            time="time",
            max_rank=2,
            lambda_reg=1e-4,
            n_bootstrap=3,
            random_state=0,
            max_iter=5000,
            tol=1e-10,
        )
        L = r.model_info["completed_matrix"]
        rel = _masked_rel_fro(L, M, mask)
        assert rel < self.RTOL, (
            f"noiseless completion rel-Fro {rel:.2e} >= rtol {self.RTOL}; "
            f"expected near-exact recovery of the rank-2 M."
        )


# ---------------------------------------------------------------------------
# C. Singular-value gap / rank-2 recovery (non-tautological: no max_rank).
# ---------------------------------------------------------------------------


class TestSingularValueGap:
    """lambda ALONE recovers rank 2: a genuine 3rd singular value is
    thresholded to zero (not forced by a max_rank cap)."""

    def test_lambda_only_recovers_rank_two(self, noisy_completion):
        """No max_rank: rank must emerge from soft-thresholding.

        At the converged fixed point the imputed data matrix's 3rd
        singular value is a real nonzero (~0.78, far from machine zero;
        the raw zero-filled data has it at ~5.1), which lambda=1.0 drives
        to exactly 0.  So effective_rank == 2 is a recovery FACT, not a
        consequence of a rank cap.  s[2]/s[1] of the reconstruction is
        then exactly 0 (the soft-thresholded singular values past index 1
        are zeroed): tolerance 1e-12 (bitwise zero from np.maximum).
        """
        df, _M, _mask = noisy_completion
        r = sp.matrix_completion(
            df,
            y="y",
            d="d",
            unit="unit",
            time="time",
            lambda_reg=1.0,
            n_bootstrap=3,
            random_state=0,
            max_iter=2000,
        )
        assert r.model_info["effective_rank"] == 2, (
            f"lambda-only soft-impute recovered rank "
            f"{r.model_info['effective_rank']}, expected 2 (the noise "
            f"singular values should be thresholded away)."
        )
        L = r.model_info["completed_matrix"]
        s = np.linalg.svd(L, compute_uv=False)
        # Two leading singular values survive thresholding...
        assert s[1] > 1.0, f"second singular value {s[1]:.3f} collapsed"
        # ...and the third is exactly zeroed -> a hard rank-2 gap.
        assert s[2] / s[1] < 1e-12, (
            f"singular-value gap broken: s[2]/s[1] = {s[2] / s[1]:.2e} "
            f"(expected exactly 0 after soft-thresholding)."
        )


# ---------------------------------------------------------------------------
# D. Causal ATT recovery + naive-bias contrast.
# ---------------------------------------------------------------------------


class TestCausalRecoveryAndNaiveBias:
    """mc_panel recovers TAU; a trend-ignoring naive contrast is biased."""

    def test_att_recovers_and_naive_is_biased(self, causal_panel):
        df, Y, W, treated_units, T0 = causal_panel

        # --- naive pre/post contrast on the treated units (hand-rolled,
        #     no statspai): mean(treated post) - mean(treated pre).  It
        #     absorbs the steep low-rank ramp and is biased high.
        naive = float(Y[W == 1].mean() - Y[treated_units][:, :T0].mean())
        # Sanity: the naive contrast really is biased away from TAU
        # (probed ~6.7 vs 3.0); if not, the trend vanished and the
        # contrast is vacuous.
        assert naive - TAU > 1.5, (
            f"naive pre/post contrast {naive:.3f} not biased above TAU "
            f"{TAU} by the required 1.5 — DGP lost its trend confounding."
        )

        # --- mc_panel recovers the hand-set TAU within 4 bootstrap SE.
        r = sp.mc_panel(
            df,
            y="y",
            unit="unit",
            time="time",
            treat="d",
            max_rank=2,
            lambda_reg=0.1,
            n_bootstrap=50,
            random_state=0,
        )
        assert r.estimand == "ATT"
        z = abs(r.estimate - TAU) / r.se
        # 4-sigma recovery (suite convention); probed z ~0.2.  A +20%
        # bias (3.6) lands ~15 sigma out.
        assert z <= 4.0, (
            f"mc_panel ATT {r.estimate:.4f} (SE {r.se:.4f}) misses TAU "
            f"{TAU} by {z:.1f} sigma."
        )
        # AND it lands strictly below the biased naive contrast by a real
        # margin (directional de-confounding, not mere execution).
        assert r.estimate < naive - 1.0, (
            f"mc_panel {r.estimate:.4f} did not de-confound below naive "
            f"{naive:.4f} by the required 1.0 margin."
        )

    def test_alias_matches_mc_panel(self, causal_panel):
        """sp.matrix_completion(d=...) == sp.mc_panel(treat=...) exactly.

        The alias only renames d -> treat (_article_aliases.py:547-596),
        adding no numerical code, so on identical inputs and random_state
        the two point estimates are bitwise-equal.  Tolerance 1e-12.
        """
        df, _Y, _W, _tu, _T0 = causal_panel
        r_alias = sp.matrix_completion(
            df,
            y="y",
            d="d",
            unit="unit",
            time="time",
            max_rank=2,
            lambda_reg=0.1,
            n_bootstrap=20,
            random_state=0,
        )
        r_panel = sp.mc_panel(
            df,
            y="y",
            unit="unit",
            time="time",
            treat="d",
            max_rank=2,
            lambda_reg=0.1,
            n_bootstrap=20,
            random_state=0,
        )
        assert abs(r_alias.estimate - r_panel.estimate) < 1e-12, (
            f"alias estimate {r_alias.estimate!r} != mc_panel "
            f"{r_panel.estimate!r}; the d->treat wrapper changed the "
            f"numerics."
        )


# ---------------------------------------------------------------------------
# E. Determinism / seed-stability.
# ---------------------------------------------------------------------------


class TestDeterminism:
    """random_state pins the bootstrap -> identical estimate AND se."""

    def test_repeated_call_is_bitwise_stable(self, causal_panel):
        df, _Y, _W, _tu, _T0 = causal_panel
        kw = dict(
            y="y",
            unit="unit",
            time="time",
            treat="d",
            max_rank=2,
            lambda_reg=0.1,
            n_bootstrap=30,
            random_state=0,
        )
        r1 = sp.mc_panel(df, **kw)
        r2 = sp.mc_panel(df, **kw)
        assert (
            abs(r1.estimate - r2.estimate) < 1e-12
        ), f"non-deterministic estimate: {r1.estimate!r} vs {r2.estimate!r}"
        # SE is bootstrap-derived; random_state must pin it too.
        assert abs(r1.se - r2.se) < 1e-12, (
            f"non-deterministic SE: {r1.se!r} vs {r2.se!r} — random_state "
            f"failed to seed the bootstrap."
        )
