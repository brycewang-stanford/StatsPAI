"""Reference parity: dynamic-panel GMM (Arellano-Bond) family.

Estimators
----------
``sp.xtabond`` (Arellano-Bond / Blundell-Bond first-differenced GMM) and
``sp.gmm`` (Hansen general moment-condition GMM).  Both previously had
only smoke tests; this file is their first numerical guarantee on the
canonical dynamic-panel identification problem.

Setting (dynamic panel data)
----------------------------
The data-generating process is the textbook Arellano-Bond panel

    y_{it} = RHO * y_{i,t-1} + BETA * x_{it} + alpha_i + e_{it},

with a unit fixed effect ``alpha_i`` correlated with the lagged dependent
variable through the dynamics, a strictly-exogenous regressor ``x_{it}``,
and i.i.d. ``e_{it}``.  The structural parameters are HAND-SET to
``RHO = 0.5`` and ``BETA = 1.0`` (stored as module constants); a long
burn-in renders the panel approximately stationary before the ``T``
retained periods.  Because ``y_{i,t-1}`` is mechanically correlated with
``alpha_i``, pooled OLS is inconsistent and — crucially — the
within-group (LSDV / fixed-effects) estimator is biased for ``RHO`` by an
O(1/T) amount that does NOT vanish in ``N`` (Nickell 1981; the "Nickell
bias").  First-differencing removes ``alpha_i``; lagged *levels*
``y_{i,t-2}, y_{i,t-3}, ...`` are valid GMM instruments for the
differenced lagged dependent variable, which is exactly what ``xtabond``
exploits.

Anchors (4 independent, non-tautological)
-----------------------------------------
A. **Known-DGP recovery of RHO and BETA** (recovery).  Over a Monte-Carlo
   bank of independent panels the mean ``xtabond`` ``rho_hat`` recovers
   ``RHO = 0.5`` within ``4 * SD / sqrt(reps)`` (probed MC mean 0.504,
   essentially unbiased; a +20% estimate bias shifts the mean to ~0.60,
   ~3.3 band-widths out → fails), AND on a single larger draw the
   ``x``-coefficient recovers ``BETA = 1.0`` within 4 of its own reported
   SE (probed z ~1.2; a +20% bias on BETA lands ~8 sigma out).
B. **Nickell-bias contrast** (naive_bias).  The within-group (LSDV)
   estimator of ``RHO`` is provably biased DOWN here (small ``T``): across
   an 8-seed bank its rho is asserted to sit *strictly below* truth by a
   real margin (probed ~0.33, max 0.357 << 0.5), while ``xtabond`` (a) is
   asserted to recover ``RHO`` within 4 sigma and (b) to land strictly
   ABOVE the within-group estimate.  Both halves are asserted, so the
   contrast proves directional de-biasing, not mere execution.
C. **Cross-method consistency: sp.gmm == sp.xtabond** (consistency).  The
   Arellano-Bond first-differenced moment system (regressors
   ``[Δy_{t-1}, Δx]``, block-diagonal lagged-level instruments plus the
   ``Δx`` standard instrument, one-step weight ``A = (Σ_i Z_i' H Z_i)^-1``
   with the MA(1) ``H``) is rebuilt by hand and fed to the *generic*
   ``sp.gmm`` one-step minimiser with that same ``W = A``.  Its
   ``(rho, beta)`` must equal ``xtabond``'s coefficient table to
   ``rtol/atol = 1e-5`` (probed max|diff| ~3e-7, limited only by the BFGS
   ``gtol=1e-8`` stopping rule vs xtabond's closed-form solve).  Pins the
   two estimators to the *same* moment-minimisation arithmetic.
D. **Orientation / sign correctness** (orientation).  The DGP has
   ``RHO > 0`` and ``BETA > 0``; ``xtabond`` returns a positive lagged-Y
   coefficient and a positive ``x`` coefficient.

Implementation facts the anchors rely on (cited file:line)
----------------------------------------------------------
- ``src/statspai/gmm/arellano_bond.py:257`` — equation periods are
  ``p = n_ylags+1 .. T-1`` (so ``T=6`` ⇒ ``p ∈ {2,3,4,5}``);
  ``:264-270`` — instrument columns keyed ``(p, s)`` for every level lag
  ``s <= p-2`` (block-diagonal); ``:316-317`` — ``Δx`` enters as its own
  standard instrument column; ``:368-369`` — one-step weight is
  ``A = (Σ_i Z_i' H Z_i)^-1`` with ``H = _ab_H`` the MA(1) structure
  (2 on the diagonal, -1 on consecutive off-diagonals).
- ``src/statspai/gmm/arellano_bond.py:489-497`` — ``detail`` rows are the
  lagged-Y coefficient first (``L1.y``), then the exogenous regressors;
  ``:521-533`` — ``estimate`` is ``rho`` (``beta[0]``).  Anchor C reads
  ``detail['coefficient']`` to compare the *full* ``(rho, beta)`` vector.
- ``src/statspai/gmm/general_gmm.py:144-150`` — with ``method='onestep'``
  and a supplied ``W`` the estimator minimises ``ḡ(θ)' W ḡ(θ)`` once over
  the user moments.  Feeding *unit-aggregated* AB moments
  ``g_unit(θ) = Z_i'(Δy_i - W_i θ)`` with ``W = A`` reproduces the AB
  one-step point estimate (the ``1/n`` averaging in ``g_bar`` is an
  overall scalar that cancels in the argmin).

References
----------
- Arellano & Bond (1991), Review of Economic Studies 58(2) — the
  first-differenced dynamic-panel GMM estimator ``xtabond`` implements.
  [@arellano1991some]
- Blundell & Bond (1998), Journal of Econometrics 87(1) — system-GMM
  companion. [@blundell1998initial]
- Roodman (2009), Stata Journal 9(1) — ``xtabond2`` practitioner guide.
  [@roodman2009xtabond]
- Hansen (1982), Econometrica 50(4) — generalized method of moments, the
  framework ``sp.gmm`` implements. [@hansen1982large]
- Nickell (1981), "Biases in Dynamic Models with Fixed Effects",
  Econometrica 49(6), 1417-1426 — the within-group small-T bias the
  Nickell-contrast anchor exploits.  (No bib key in paper.bib; named
  without a citation key per the suite convention.)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.gmm.arellano_bond import _ab_H

# Hand-set structural parameters of the Arellano-Bond DGP.
RHO = 0.5
BETA = 1.0


# ---------------------------------------------------------------------------
# Deterministic DGP builder (every draw seeded via default_rng).
# ---------------------------------------------------------------------------


def _make_ab_panel(seed, N=200, T=6, burn=20):
    """Arellano-Bond dynamic panel with known RHO, BETA.

        y_{it} = RHO*y_{i,t-1} + BETA*x_{it} + alpha_i + e_{it}

    ``alpha_i ~ N(0,1)`` is the unit fixed effect (correlated with the
    lagged y through the dynamics, which is what makes pooled OLS and the
    within estimator inconsistent).  ``x_{it}`` is strictly exogenous
    i.i.d. N(0,1); ``e_{it}`` is i.i.d. N(0,1).  A ``burn``-period
    warm-up brings each unit near its stationary distribution before the
    ``T`` retained periods so the initial-condition transient does not
    contaminate the recovered parameters.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N):
        alpha = rng.normal(0.0, 1.0)
        # start near the stationary mean alpha/(1-RHO) plus a shock
        y = alpha / (1.0 - RHO) + rng.normal(0.0, 1.0)
        for _ in range(burn):
            x = rng.normal(0.0, 1.0)
            y = RHO * y + BETA * x + alpha + rng.normal(0.0, 1.0)
        for t in range(T):
            x = rng.normal(0.0, 1.0)
            y = RHO * y + BETA * x + alpha + rng.normal(0.0, 1.0)
            rows.append({"id": i, "time": t, "y": y, "x": x})
    return pd.DataFrame(rows)


def _xtabond(df, **kw):
    """xtabond wrapper that silences the unrelated diagnostic warnings."""
    with warnings.catch_warnings():
        # robust-SE / weak-instrument chatter is not under test here.
        warnings.simplefilter("ignore")
        return sp.xtabond(df, y="y", x=["x"], id="id", time="time", **kw)


def _xtabond_coefs(df, **kw):
    """Return (rho_hat, beta_hat) from the xtabond coefficient table."""
    r = _xtabond(df, **kw)
    coefs = r.detail["coefficient"].values
    return float(coefs[0]), float(coefs[1])


def _within_group_rho(df):
    """LSDV (within-group) estimate of RHO — the Nickell-biased baseline.

    Hand-rolled with numpy: lag y within unit, drop the first period,
    within-transform (y, ylag, x) by subtracting unit means, then OLS.
    Uses no statspai estimator so the contrast in anchor B is genuinely
    against an independent baseline.
    """
    d = df.sort_values(["id", "time"]).copy()
    d["ylag"] = d.groupby("id")["y"].shift(1)
    d = d.dropna().copy()
    for c in ["y", "ylag", "x"]:
        d[c + "_w"] = d[c] - d.groupby("id")[c].transform("mean")
    Xw = np.column_stack([d["ylag_w"].values, d["x_w"].values])
    Yw = d["y_w"].values
    beta_w = np.linalg.lstsq(Xw, Yw, rcond=None)[0]
    return float(beta_w[0])


def _gmm_replicate_ab(df, T):
    """Rebuild the AB difference-GMM one-step system and solve via sp.gmm.

    Mirrors ``arellano_bond.xtabond`` exactly for ``lags=1`` and a single
    exogenous regressor ``x``: equation periods ``p = 2 .. T-1``,
    block-diagonal instrument columns ``(p, s)`` for every level lag
    ``s <= p-2``, plus one ``Δx`` standard-instrument column; the one-step
    weight is ``A = (Σ_i Z_i' H Z_i)^-1`` with the MA(1) ``_ab_H``.  The
    per-*unit* aggregated moments ``Z_i'(Δy_i - W_i θ)`` are handed to the
    generic ``sp.gmm`` one-step minimiser with ``W = A``.
    """
    df = df.sort_values(["id", "time"])
    units = df["id"].unique()

    eq_positions = list(range(2, T))  # p = 2 .. T-1 for lags=1
    ycols = []
    for p in eq_positions:
        for s in range(0, p - 1):  # s <= p-2  ->  lag = p-s >= 2
            ycols.append((p, s))
    ycol_pos = {key: j for j, key in enumerate(ycols)}
    nyc = len(ycols)
    m = nyc + 1  # + one column for the Δx standard instrument

    W_rows, Z_rows, dY, row_unit, row_p = [], [], [], [], []
    for ui, uid in enumerate(units):
        g = df[df["id"] == uid]
        yp = {t: v for t, v in zip(g["time"], g["y"])}
        xp = {t: v for t, v in zip(g["time"], g["x"])}
        for p in eq_positions:
            if any(q not in yp for q in (p, p - 1, p - 2)):
                continue
            dY.append(yp[p] - yp[p - 1])
            W_rows.append([yp[p - 1] - yp[p - 2], xp[p] - xp[p - 1]])
            zrow = np.zeros(m)
            for s in range(0, p - 1):
                zrow[ycol_pos[(p, s)]] = yp[s]
            zrow[nyc] = xp[p] - xp[p - 1]
            Z_rows.append(zrow)
            row_unit.append(ui)
            row_p.append(p)

    W = np.asarray(W_rows)
    Z = np.asarray(Z_rows)
    dYv = np.asarray(dY)
    row_unit = np.asarray(row_unit)
    row_p = np.asarray(row_p)
    unit_rows = [np.where(row_unit == ui)[0] for ui in range(len(units))]
    unit_rows = [r for r in unit_rows if r.size > 0]

    ZHZ = np.zeros((m, m))
    for r in unit_rows:
        Zi = Z[r]
        ZHZ += Zi.T @ _ab_H(row_p[r]) @ Zi
    A = np.linalg.inv(ZHZ)

    def moment_fn(theta, data):
        res = dYv - W @ theta
        out = np.zeros((len(unit_rows), m))
        for j, r in enumerate(unit_rows):
            out[j] = Z[r].T @ res[r]
        return out

    gdata = pd.DataFrame({"_unit": np.arange(len(unit_rows))})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rg = sp.gmm(
            moment_fn,
            theta0=np.zeros(2),
            data=gdata,
            W=A,
            method="onestep",
            param_names=["rho", "beta"],
        )
    return float(rg.params["rho"]), float(rg.params["beta"])


# ---------------------------------------------------------------------------
# Module fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ab_panel():
    """Single canonical panel, N=200 T=6 (the design-hint size)."""
    return _make_ab_panel(0, N=200, T=6)


@pytest.fixture(scope="module")
def ab_panel_large():
    """Larger draw (N=400 T=7) used for the tight single-draw BETA band."""
    return _make_ab_panel(7, N=400, T=7)


# ---------------------------------------------------------------------------
# A. Known-DGP recovery of RHO and BETA.
# ---------------------------------------------------------------------------


class TestRecovery:
    """xtabond recovers the hand-set RHO = 0.5 and BETA = 1.0."""

    def test_monte_carlo_rho_recovers(self):
        """20 independent panels; MC mean rho within 4*SD/sqrt(reps) of RHO.

        Averaging over independent draws cancels per-draw sampling noise,
        so the MC mean is a sharp probe of *systematic* bias.  At
        N=250, T=7 the AB-GMM rho is essentially unbiased (probed MC mean
        0.504, SD ~0.035, band 4*0.035/sqrt(20) ~0.032 around 0.50,
        |mean-0.5| ~0.004).  A +20% multiplicative estimate bias shifts
        every fit so the MC mean lands near 0.605, ~3.3 band-widths out,
        and the assert fails — non-tautological.
        """
        reps = 20
        rhos = np.array(
            [
                _xtabond_coefs(_make_ab_panel(2000 + s, N=250, T=7))[0]
                for s in range(reps)
            ]
        )
        mc_mean = float(rhos.mean())
        mc_sd = float(rhos.std(ddof=1))
        band = 4.0 * mc_sd / np.sqrt(reps)
        assert abs(mc_mean - RHO) <= band, (
            f"AB-GMM MC mean rho {mc_mean:.4f} drifted from truth {RHO} "
            f"(band {band:.4f}, SD {mc_sd:.4f}) over {reps} reps — "
            f"systematic bias in xtabond's rho."
        )

    def test_single_draw_beta_within_4_se(self, ab_panel_large):
        """x-coefficient recovers BETA = 1.0 within 4 of its reported SE.

        The exogenous-regressor coefficient is recovered with a tight SE
        (probed beta_hat ~0.966, SE ~0.026, z ~1.3 on this draw); a 4-sigma
        band keeps the false-failure rate at the suite's 6.3e-5 while a
        +20% bias on BETA (→1.2) lands ~8 sigma out and fails.
        """
        r = _xtabond(ab_panel_large)
        beta_hat = float(r.detail["coefficient"].values[1])
        beta_se = float(r.detail["se"].values[1])
        assert abs(beta_hat - BETA) <= 4.0 * beta_se, (
            f"xtabond beta_hat {beta_hat:.4f} (SE {beta_se:.4f}) misses "
            f"truth {BETA} by {abs(beta_hat - BETA) / beta_se:.1f} sigma."
        )


# ---------------------------------------------------------------------------
# B. Nickell-bias contrast (within-group biased; AB-GMM de-biases).
# ---------------------------------------------------------------------------


class TestNickellBiasContrast:
    """Within-group rho is biased DOWN (small T); AB-GMM recovers RHO."""

    def test_within_group_biased_low_abgmm_recovers(self, ab_panel):
        """Assert BOTH: LSDV rho < truth by a margin AND xtabond ≈ truth.

        With T=6 the Nickell (1981) bias drags the within-group rho well
        below RHO=0.5 (probed ~0.33 here, max 0.357 across an 8-seed bank;
        the bias is O(1/T) and negative for positive rho).  AB-GMM removes
        it.  The contrast is non-tautological because it pins *two*
        independent facts: (i) the biased baseline sits strictly below
        truth, (ii) AB-GMM both recovers truth within 4 sigma and lands
        strictly above the biased baseline.  A +20% UP injection on the AB
        estimate would still beat the baseline (part ii-b) but would
        violate the 4-sigma recovery (part ii-a, probed 0.52→0.63), so the
        anchor fails under injection.
        """
        # (i) within-group rho is biased strictly below truth — verified
        # across an 8-seed bank so this is not a one-draw fluke.
        within_rhos = np.array(
            [_within_group_rho(_make_ab_panel(s, N=200, T=6)) for s in range(8)]
        )
        assert within_rhos.max() < RHO - 0.10, (
            f"within-group rho bank max {within_rhos.max():.4f} is not "
            f"strictly below truth {RHO} by the required 0.10 margin — "
            f"the Nickell bias vanished and the contrast is vacuous."
        )

        # (ii-a) AB-GMM recovers RHO within 4 sigma on the canonical panel.
        r = _xtabond(ab_panel)
        rho_hat = float(r.detail["coefficient"].values[0])
        rho_se = float(r.detail["se"].values[0])
        assert abs(rho_hat - RHO) <= 4.0 * rho_se, (
            f"AB-GMM rho_hat {rho_hat:.4f} (SE {rho_se:.4f}) failed to "
            f"recover truth {RHO} ({abs(rho_hat - RHO) / rho_se:.1f} sigma)."
        )

        # (ii-b) AB-GMM lands strictly above the within-group estimate on
        # the *same* panel (directional de-biasing, not coincidence).
        within_same = _within_group_rho(ab_panel)
        assert rho_hat > within_same + 0.10, (
            f"AB-GMM rho {rho_hat:.4f} did not de-bias above the "
            f"within-group rho {within_same:.4f} by the 0.10 margin."
        )


# ---------------------------------------------------------------------------
# C. Cross-method consistency: sp.gmm reproduces sp.xtabond.
# ---------------------------------------------------------------------------


class TestCrossMethodConsistency:
    """Generic sp.gmm one-step on the AB moments == sp.xtabond coefs."""

    # rtol/atol 1e-5: in exact arithmetic this is an identity (same
    # moment system, same weight A, same argmin).  The only slack is the
    # BFGS gtol=1e-8 stopping rule in sp.gmm vs xtabond's closed-form
    # linear solve; probed max|diff| ~3e-7 over 3 seeds, so 1e-5 has ~30x
    # headroom while staying far tighter than any economically meaningful
    # coefficient difference.
    RTOL = 1e-5
    ATOL = 1e-5

    def test_gmm_equals_xtabond_rho_and_beta(self, ab_panel):
        ab_rho, ab_beta = _xtabond_coefs(ab_panel)
        g_rho, g_beta = _gmm_replicate_ab(ab_panel, T=6)
        np.testing.assert_allclose(
            [g_rho, g_beta],
            [ab_rho, ab_beta],
            rtol=self.RTOL,
            atol=self.ATOL,
            err_msg=(
                f"sp.gmm one-step (rho={g_rho:.6f}, beta={g_beta:.6f}) does "
                f"not match sp.xtabond (rho={ab_rho:.6f}, beta={ab_beta:.6f}) "
                f"on the same AB moment system — the two GMM paths diverged."
            ),
        )

    def test_gmm_equals_xtabond_multiple_seeds(self):
        """Identity holds across independent panels, not one lucky draw."""
        for s in range(3):
            df = _make_ab_panel(s, N=200, T=6)
            ab_rho, ab_beta = _xtabond_coefs(df)
            g_rho, g_beta = _gmm_replicate_ab(df, T=6)
            np.testing.assert_allclose(
                [g_rho, g_beta],
                [ab_rho, ab_beta],
                rtol=self.RTOL,
                atol=self.ATOL,
                err_msg=f"sp.gmm != sp.xtabond on seed {s}.",
            )


# ---------------------------------------------------------------------------
# D. Orientation / sign correctness.
# ---------------------------------------------------------------------------


class TestOrientation:
    """RHO > 0 and BETA > 0 ⇒ positive lagged-Y and x coefficients."""

    def test_both_coefficients_positive(self, ab_panel):
        rho_hat, beta_hat = _xtabond_coefs(ab_panel)
        assert rho_hat > 0.0, f"expected positive rho, got {rho_hat:.4f}"
        assert beta_hat > 0.0, f"expected positive beta, got {beta_hat:.4f}"
