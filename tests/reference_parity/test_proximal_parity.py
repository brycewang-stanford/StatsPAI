"""Reference parity: proximal causal inference family.

Estimators
----------
``sp.proximal`` (linear-bridge 2SLS), ``sp.fortified_pci`` (doubly-robust
PCI), ``sp.bidirectional_pci`` (outcome + treatment bridge).  Each
previously had only a smoke test; this file is their first numerical
guarantee.

Setting (proximal causal inference)
-----------------------------------
A back-door confounder ``U`` is *unmeasured*.  We observe two proxies of
``U``: a treatment-side proxy ``Z`` (``Z ⊥ Y | D, U, X``) and an
outcome-side proxy ``W`` (``W ⊥ D | U, X``).  With a **linear** outcome
bridge ``h(W, D, X) = γ0 + γ_D D + γ_W W + γ_X X`` the ATE ``γ_D`` is
identified by a 2SLS of ``Y`` on ``(D, W, X)`` instrumented by
``(D, Z, X)`` (Cui et al. 2024 linear case).  The naive OLS of ``Y`` on
``(D, X)`` omits ``U`` and is therefore confounded.

Anchors
-------
A. **Closed-form just-identified collapse** (closed_form).  With a
   single ``Z``, single ``W`` and no covariates the 2SLS system is
   square (instruments ``[1, D, Z]``, regressors ``[1, D, W]``), so the
   just-identified IV solution is exactly ``(Z'X)^{-1} Z'Y``.
   ``sp.proximal``'s ``estimate`` (coefficient on ``D``) must equal that
   hand-computed entry to machine precision (probed |diff| ~2e-16).
   Pins the estimate to an exact algebraic identity.
B. **Known-DGP recovery of GAMMA_D = 1.5** (recovery).  On the
   linear-Gaussian proximal DGP (correctly specified linear bridge),
   ``sp.proximal`` recovers ``GAMMA_D`` within 4 of its own reported SE
   on a single draw (probed z ~1.1), AND a 40-rep Monte-Carlo mean is
   within 4*SD/sqrt(R) of truth (probed MC mean 1.5004).
C. **Naive-bias contrast** (naive_bias).  The naive OLS slope of ``Y``
   on ``(D, X)`` is > 6 sigma biased high (probed ~1.98 vs truth 1.50,
   z ~30) because it omits ``U``; ``sp.proximal`` recovers truth within
   4 sigma AND lands strictly below the naive estimate by a margin,
   proving directional de-confounding rather than mere execution.
D. **Family internal consistency** (consistency).  (i) Continuous ``D``
   forces ``bidirectional_pci``'s Z-IPW logistic step to fail (logistic
   regression rejects a non-binary target), so it falls back to the
   outcome-bridge-only estimate and equals ``sp.proximal`` exactly
   (``treatment_bridge_fallback`` True; probed diff 0.0).  (ii) Binary
   ``D`` activates the Z-IPW (fallback False); then both
   ``bidirectional_pci`` and ``fortified_pci`` land strictly between the
   biased naive difference-in-means and the truth, while ``sp.proximal``
   recovers truth — pinning the partial correctors between "does
   nothing" and "fully corrects".
E. **Orientation / sign correctness** (orientation).  Strictly
   positive-effect DGPs (continuous and binary ``D``) yield positive
   estimates from all three estimators.

Implementation facts the anchors rely on (cited file:line)
----------------------------------------------------------
- ``src/statspai/proximal/p2sls.py:172-178`` — exogenous block is
  ``[const, D, X]``, excluded instrument is ``Z``, endogenous regressor
  is ``W``; ``proximal.py:202`` — ``estimate = beta[1]`` (the coefficient
  on ``D``, sitting right after the constant).
- ``src/statspai/proximal/p2sls.py:344-368`` — ``_linear_iv_fit``
  projects regressors onto the instruments (``X_hat = Zmat @ Pi``) then
  OLS ``Y`` on ``X_hat``.  When ``instruments`` and ``regressors`` have
  the same width (just-identified) this is algebraically the IV
  estimator ``(Z'X)^{-1} Z'Y`` — the identity anchor A relies on.
- ``src/statspai/proximal/bidirectional.py:113-137`` — the treatment
  bridge fits ``LogisticRegression`` on ``(Z, X)`` against ``D``; a
  continuous ``D`` raises inside sklearn, the ``except`` sets
  ``tau_treatment = tau_outcome`` and ``treatment_bridge_fallback =
  True`` (anchor D-i).  The final estimate is
  ``0.5*tau_outcome + 0.5*tau_treatment`` (line 137), which under
  fallback collapses to ``tau_outcome`` — the same 2SLS bridge
  ``sp.proximal`` computes (``bidirectional.py:104-109``).
- ``src/statspai/proximal/fortified.py:120-161`` — fortified estimate is
  ``0.5*tau_bridge + 0.5*tau_outcome`` where ``tau_outcome`` regresses
  ``Y`` on ``(D, Z, X)``; on a confounded binary-D DGP this is a partial
  corrector (anchor D-ii).

References (bib keys verified present in paper.bib via grep)
------------------------------------------------------------
- Tchetgen Tchetgen, Ying, Cui, Shi & Miao (2020), "An Introduction to
  Proximal Causal Learning", arXiv:2009.10982. [@tchetgen2020introduction]
- Miao, Geng & Tchetgen Tchetgen (2018), Biometrika 105(4).
  [@miao2018identifying]
- Cui, Pu, Shi, Miao & Tchetgen Tchetgen (2024), JASA 119(546) — the
  linear-bridge 2SLS this estimator implements. [@cui2024semiparametric]
- Yu, Shi & Tchetgen Tchetgen (2025), arXiv:2506.13152 — fortified PCI.
  [@yu2025fortified]
- Min, Zhang & Luo (2025), arXiv:2507.13965 — bidirectional PCI.
  [@min2025regression]
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# Hand-set true treatment effect shared by every DGP below.
GAMMA_D = 1.5


def _within_n_se(est, truth, se, n_sigma=4.0):
    return abs(est - truth) <= n_sigma * se


# ---------------------------------------------------------------------------
# Deterministic DGP builders (every draw seeded via default_rng).
# ---------------------------------------------------------------------------


def _make_proximal_dgp(seed, n=4000, binary_d=False):
    """Linear-Gaussian proximal DGP with a known ATE = GAMMA_D.

    U is the unmeasured confounder.  Z = U + noise (treatment-side
    proxy), W = U + noise (outcome-side proxy); both satisfy the PCI
    proxy independencies by construction (Z, W are noisy copies of U,
    conditionally independent of the outcome / treatment shocks).  X is
    a measured covariate.  The outcome
        Y = GAMMA_D * D + 1.0 * U + 0.3 * X + noise
    means OLS of Y on (D, X) inherits U's confounding, while the
    proximal 2SLS using Z as an instrument for W removes it.
    """
    rng = np.random.default_rng(seed)
    U = rng.normal(size=n)
    Z = U + rng.normal(scale=1.0, size=n)
    W = U + rng.normal(scale=1.0, size=n)
    X = rng.normal(size=n)
    if binary_d:
        lin = 0.8 * U + 0.5 * X
        D = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-lin))).astype(float)
    else:
        D = 0.8 * U + 0.5 * X + rng.normal(scale=1.0, size=n)
    Y = GAMMA_D * D + 1.0 * U + 0.3 * X + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"y": Y, "d": D, "z": Z, "w": W, "x": X})


def _make_just_identified_dgp(seed, n=2000):
    """Single Z, single W, NO covariates -> square 2SLS system.

    With instruments [const, D, Z] and regressors [const, D, W] both
    3-wide, the proximal 2SLS reduces to the just-identified IV
    estimator (Z'X)^{-1} Z'Y, which anchor A computes by hand.
    """
    rng = np.random.default_rng(seed)
    U = rng.normal(size=n)
    Z = U + rng.normal(scale=1.0, size=n)
    W = U + rng.normal(scale=1.0, size=n)
    D = 0.8 * U + rng.normal(scale=1.0, size=n)
    Y = GAMMA_D * D + 1.0 * U + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"y": Y, "d": D, "z": Z, "w": W})


# ---------------------------------------------------------------------------
# Module fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def just_identified_data():
    return _make_just_identified_dgp(123)


@pytest.fixture(scope="module")
def continuous_d_data():
    return _make_proximal_dgp(2026, n=4000, binary_d=False)


@pytest.fixture(scope="module")
def binary_d_data():
    return _make_proximal_dgp(777, n=4000, binary_d=True)


def _proximal(df, covariates=None):
    # n_boot=0 -> closed-form 2SLS sandwich SE (deterministic, fast).
    with warnings.catch_warnings():
        # weak-instrument / first-stage F warnings are not under test here.
        warnings.simplefilter("ignore")
        return sp.proximal(
            df, y="y", treat="d", proxy_z=["z"], proxy_w=["w"], covariates=covariates
        )


# ---------------------------------------------------------------------------
# A. Closed-form just-identified collapse (machine precision).
# ---------------------------------------------------------------------------


class TestClosedFormCollapse:
    """proximal == hand-computed (Z'X)^{-1} Z'Y coefficient on D.

    Tolerance 1e-9: in exact arithmetic this is an algebraic identity
    (square 2SLS == just-identified IV).  The only slack is the
    difference between ``np.linalg.solve`` (anchor) and the pinv-based
    projection inside ``_linear_iv_fit`` (p2sls.py:368) on a
    well-conditioned 3x3 system; probed |diff| ~2e-16, so 1e-9 has ~7
    orders of headroom yet stays tighter than the 1e-8 machine-collapse
    bar.  An anchor that merely checked finiteness would pass for any
    number; this one pins the estimate to a specific hand-computed
    scalar.
    """

    TOL = 1e-9

    def test_proximal_equals_just_identified_iv(self, just_identified_data):
        df = just_identified_data
        r = _proximal(df)

        n = len(df)
        const = np.ones((n, 1))
        D = df["d"].values.reshape(-1, 1)
        Z = df["z"].values.reshape(-1, 1)
        W = df["w"].values.reshape(-1, 1)
        Y = df["y"].values
        Zmat = np.hstack([const, D, Z])  # instruments [1, D, Z]
        Xmat = np.hstack([const, D, W])  # regressors  [1, D, W]
        # Just-identified IV: beta = (Z'X)^{-1} Z'Y; coefficient on D is
        # at index 1 (right after the constant) — same slot p2sls.py:202
        # reads.
        beta_iv = np.linalg.solve(Zmat.T @ Xmat, Zmat.T @ Y)
        expected = float(beta_iv[1])

        assert abs(r.estimate - expected) < self.TOL, (
            f"proximal {r.estimate!r} != just-identified IV coef on D "
            f"{expected!r} (|diff|={abs(r.estimate - expected):.2e}); "
            f"the square-2SLS == (Z'X)^-1 Z'Y identity is broken."
        )


# ---------------------------------------------------------------------------
# B. Known-DGP recovery of GAMMA_D.
# ---------------------------------------------------------------------------


class TestRecovery:
    """proximal recovers the hand-set GAMMA_D = 1.5."""

    def test_single_draw_within_4_se(self, continuous_d_data):
        r = _proximal(continuous_d_data, covariates=["x"])
        # 4-sigma recovery (suite convention, REFERENCES.md; false-failure
        # 6.3e-5).  Observed z ~1.1.  A 20% bias (1.8) lands ~11 sigma
        # out and fails.
        assert _within_n_se(r.estimate, GAMMA_D, r.se, n_sigma=4.0), (
            f"proximal {r.estimate:.4f} (SE {r.se:.4f}) misses truth "
            f"{GAMMA_D} by {abs(r.estimate - GAMMA_D) / r.se:.1f} sigma."
        )

    def test_monte_carlo_mean_recovers(self):
        """40 independent draws; MC mean within 4*SD/sqrt(R) of truth.

        Averaging cancels the per-draw sampling noise, so the MC mean is
        a far tighter probe of *systematic* bias than any single draw.
        Probed MC mean 1.5004, SD ~0.039 -> band 4*0.039/sqrt(40) ~0.025
        around 1.50.  A 20% multiplicative bias would shift the mean to
        ~1.80, ~49 band-widths out.
        """
        reps = 40
        ests = []
        for s in range(reps):
            df = _make_proximal_dgp(10_000 + s, n=2000, binary_d=False)
            ests.append(_proximal(df, covariates=["x"]).estimate)
        ests = np.asarray(ests)
        mc_mean = float(ests.mean())
        mc_sd = float(ests.std(ddof=1))
        band = 4.0 * mc_sd / np.sqrt(reps)
        assert abs(mc_mean - GAMMA_D) <= band, (
            f"MC mean {mc_mean:.4f} drifted from truth {GAMMA_D} "
            f"(band {band:.4f}, SD {mc_sd:.4f}) over {reps} reps — "
            f"systematic bias in proximal."
        )


# ---------------------------------------------------------------------------
# C. Naive-bias contrast (proximal corrects the confounding).
# ---------------------------------------------------------------------------


class TestNaiveBiasContrast:
    """Naive OLS-on-D is biased high; proximal de-confounds."""

    def test_naive_biased_proximal_corrects(self, continuous_d_data):
        df = continuous_d_data
        # Naive OLS of Y on [1, D, X] — hand-rolled numpy, no statspai.
        Dx = np.column_stack([np.ones(len(df)), df["d"].values, df["x"].values])
        beta_ols = np.linalg.lstsq(Dx, df["y"].values, rcond=None)[0]
        resid = df["y"].values - Dx @ beta_ols
        sig2 = float(resid @ resid) / (len(df) - Dx.shape[1])
        xtx_inv = np.linalg.inv(Dx.T @ Dx)
        naive = float(beta_ols[1])
        se_naive = float(np.sqrt(sig2 * xtx_inv[1, 1]))

        # The naive estimator is provably confounded: > 6 sigma above
        # truth (probed z ~30).  If this fails the DGP lost its
        # confounding and the contrast is vacuous.
        assert naive - GAMMA_D > 6.0 * se_naive, (
            f"naive OLS {naive:.4f} should be > 6 sigma above truth "
            f"{GAMMA_D} (SE {se_naive:.4f}) — confounding vanished."
        )

        r = _proximal(df, covariates=["x"])
        # proximal recovers truth within 4 sigma...
        assert _within_n_se(r.estimate, GAMMA_D, r.se, n_sigma=4.0), (
            f"proximal {r.estimate:.4f} (SE {r.se:.4f}) failed to recover "
            f"truth {GAMMA_D}."
        )
        # ...AND lands strictly below naive by a real margin (directional
        # de-confounding).  Probed gap ~0.51 >> 0.10.  A +20% bias on
        # proximal (-> ~1.76) would erase this gap (naive ~1.98) while
        # also failing the recovery assert above.
        assert r.estimate < naive - 0.10, (
            f"proximal {r.estimate:.4f} did not de-confound below naive "
            f"{naive:.4f} by the required 0.10 margin."
        )


# ---------------------------------------------------------------------------
# D. Family internal consistency.
# ---------------------------------------------------------------------------


class TestFamilyConsistency:
    """Cross-estimator identities within the proximal family."""

    def test_continuous_d_bidirectional_collapses_to_proximal(self, continuous_d_data):
        """Continuous D -> Z-IPW logistic fails -> bidir == proximal.

        ``LogisticRegression`` rejects a non-binary target, so
        ``bidirectional_pci``'s treatment bridge falls back to the
        outcome bridge only (bidirectional.py:123-134), and the
        0.5/0.5 combination collapses to the 2SLS bridge that
        ``sp.proximal`` computes.  Tolerance 1e-9: identical 2SLS
        arithmetic on the same data; probed diff exactly 0.0.  The
        documented ``treatment_bridge_fallback`` flag must be True
        (guards that the collapse happened for the *right* reason, not a
        coincidental numeric match).
        """
        df = continuous_d_data
        rp = _proximal(df, covariates=["x"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rb = sp.bidirectional_pci(
                df,
                y="y",
                treat="d",
                proxy_z=["z"],
                proxy_w=["w"],
                covariates=["x"],
                n_boot=20,
                seed=0,
            )
        assert (
            rb.model_info["treatment_bridge_fallback"] is True
        ), "expected the Z-IPW logistic step to fall back on continuous D"
        assert abs(rb.estimate - rp.estimate) < 1e-9, (
            f"bidirectional {rb.estimate!r} should collapse to proximal "
            f"{rp.estimate!r} under fallback "
            f"(|diff|={abs(rb.estimate - rp.estimate):.2e})."
        )

    def test_binary_d_partial_correctors_between_naive_and_truth(self, binary_d_data):
        """Binary D -> Z-IPW active; bidir & fortified sit in (truth, naive).

        With binary D the logistic Z-IPW runs (fallback False).  On this
        confounded DGP the naive difference-in-means is biased high
        (probed 2.32 vs truth 1.50).  ``sp.proximal`` recovers truth
        within 4 sigma, while ``bidirectional_pci`` and ``fortified_pci``
        — each a 0.5 average of a confounded and a corrected term — must
        land STRICTLY between truth and the naive bias (probed 1.75 and
        1.75).  This is non-tautological: a number could be finite yet
        outside (truth, naive); injecting +20% into either would push it
        past the naive value and fail the upper bound.
        """
        df = binary_d_data
        y1 = df.loc[df["d"] == 1, "y"].mean()
        y0 = df.loc[df["d"] == 0, "y"].mean()
        naive = float(y1 - y0)
        # Sanity: the naive contrast really is biased high here.
        assert naive > GAMMA_D + 0.3, (
            f"binary-D naive diff-in-means {naive:.4f} not biased above "
            f"truth {GAMMA_D} — DGP lost its confounding."
        )

        rp = _proximal(df, covariates=["x"])
        assert _within_n_se(rp.estimate, GAMMA_D, rp.se, n_sigma=4.0), (
            f"proximal {rp.estimate:.4f} (SE {rp.se:.4f}) failed to "
            f"recover truth {GAMMA_D} on binary-D DGP."
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rb = sp.bidirectional_pci(
                df,
                y="y",
                treat="d",
                proxy_z=["z"],
                proxy_w=["w"],
                covariates=["x"],
                n_boot=20,
                seed=0,
            )
            rf = sp.fortified_pci(
                df,
                y="y",
                treat="d",
                proxy_z=["z"],
                proxy_w=["w"],
                covariates=["x"],
                n_boot=20,
                seed=0,
            )
        # Z-IPW actually engaged (fallback NOT taken).
        assert (
            rb.model_info["treatment_bridge_fallback"] is False
        ), "expected the Z-IPW logistic step to engage on binary D"
        # Both partial correctors strictly inside the (truth, naive) band.
        for name, est in [("bidirectional", rb.estimate), ("fortified", rf.estimate)]:
            assert GAMMA_D < est < naive, (
                f"{name} {est:.4f} not strictly between truth {GAMMA_D} "
                f"and naive {naive:.4f} — it neither partially corrects "
                f"nor stays below the raw confounded contrast."
            )


# ---------------------------------------------------------------------------
# E. Orientation / sign correctness.
# ---------------------------------------------------------------------------


class TestOrientation:
    """Positive-effect DGPs (GAMMA_D = 1.5 > 0) yield positive estimates."""

    def test_continuous_d_all_positive(self, continuous_d_data):
        df = continuous_d_data
        rp = _proximal(df, covariates=["x"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf = sp.fortified_pci(
                df,
                y="y",
                treat="d",
                proxy_z=["z"],
                proxy_w=["w"],
                covariates=["x"],
                n_boot=20,
                seed=0,
            )
            rb = sp.bidirectional_pci(
                df,
                y="y",
                treat="d",
                proxy_z=["z"],
                proxy_w=["w"],
                covariates=["x"],
                n_boot=20,
                seed=0,
            )
        assert rp.estimate > 0
        assert rf.estimate > 0
        assert rb.estimate > 0

    def test_binary_d_all_positive(self, binary_d_data):
        df = binary_d_data
        rp = _proximal(df, covariates=["x"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf = sp.fortified_pci(
                df,
                y="y",
                treat="d",
                proxy_z=["z"],
                proxy_w=["w"],
                covariates=["x"],
                n_boot=20,
                seed=0,
            )
            rb = sp.bidirectional_pci(
                df,
                y="y",
                treat="d",
                proxy_z=["z"],
                proxy_w=["w"],
                covariates=["x"],
                n_boot=20,
                seed=0,
            )
        assert rp.estimate > 0
        assert rf.estimate > 0
        assert rb.estimate > 0
