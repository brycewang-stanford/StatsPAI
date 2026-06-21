"""IPW reference parity tests for ``sp.ipw``.

Anchors
-------
A. **Saturated-propensity closed form** — with a single binary covariate
   the logistic propensity model (intercept + x = 2 parameters, 2 cells)
   is saturated, so the MLE fitted probabilities equal the empirical
   cell shares exactly.  The Hajek (and Horvitz-Thompson) ATE/ATT/ATC
   then collapse to hand-computable stratified means, which we compute
   with raw pandas/numpy arithmetic (no statspai internals).
B. **Randomized collapse** — with treatment independent of covariates,
   IPW must agree with the simple difference in means within combined
   sampling error.
C. **Frozen base-R fixture** — ``stats::glm(t ~ x1 + x2, binomial)`` +
   hand-rolled Hajek weighted means, pinned to machine precision.
D. **CIA recovery** — on the shared ``matching_cia_data`` DGP (true
   homogeneous effect 2.0; Imbens-Wooldridge 2009), IPW must recover
   the truth within 4 sigma.  (``test_matching_parity.py`` covers
   match/ebalance/cbps/overlap_weights but NOT ``sp.ipw``.)
E. **Trim semantics** — ``trim`` winsorizes propensities via
   ``np.clip`` (it does not drop units), so on a well-overlapped DGP
   ``trim=0.01`` is a bitwise no-op, while on a poor-overlap DGP it
   caps the maximum inverse-propensity weight at exactly ``1/trim``.
F. **Bootstrap SE calibration** — the bootstrap SE must track the
   Monte-Carlo SD of the point estimate across independent DGP draws.

Implementation facts relied on (src/statspai/inference/ipw.py)
---------------------------------------------------------------
- ipw.py:202-233 ``_estimate_propensity``: statsmodels GLM Binomial
  with an intercept (``sm.add_constant``), i.e. UNPENALIZED logit MLE;
  the sklearn fallback also sets ``penalty=None``.  Hence base R
  ``stats::glm(family=binomial)`` is the same estimator (no L2
  mismatch), and a saturated model returns exact cell shares.
- ipw.py:123-124: ``trim`` clips ``pscore`` into ``[trim, 1-trim]``
  (winsorize, not drop).
- ipw.py:236-274 ``_compute_weights``:
    ATE: w1 = T/p,        w0 = (1-T)/(1-p)
    ATT: w1 = T,          w0 = (1-T)*p/(1-p)
    ATC: w1 = T*(1-p)/p,  w0 = (1-T)
  ``normalize=True`` (default) -> Hajek: each weight vector divided by
  its own sum; ``normalize=False`` -> divided by n (Horvitz-Thompson).
- ipw.py:130: estimate = sum(w1*Y) - sum(w0*Y).
- ipw.py:133-143: bootstrap SE re-fits the propensity on each resample
  (seeded via ``np.random.RandomState(seed)``).
- ipw.py:152-164: ``model_info`` reports ``pscore_min``/``pscore_max``
  AFTER trimming (pscore is reassigned at ipw.py:124 first).

References
----------
- Horvitz, D.G. and Thompson, D.J. (1952). A Generalization of Sampling
  Without Replacement From a Finite Universe. *JASA*, 47(260), 663-685.
  [@horvitz1952generalization]
- Hirano, K., Imbens, G.W. and Ridder, G. (2003). Efficient Estimation
  of Average Treatment Effects Using the Estimated Propensity Score.
  *Econometrica*, 71(4), 1161-1189. [@hirano2003efficient]
- Crump, R.K., Hotz, V.J., Imbens, G.W. and Mitnik, O.A. (2009).
  Dealing with Limited Overlap in Estimation of Average Treatment
  Effects. *Biometrika*, 96(1), 187-199. [@crump2009dealing]
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIXTURE_DIR = pathlib.Path(__file__).parent / "_fixtures"


def _within_n_se(est, truth, se, n_sigma=4.0):
    return abs(est - truth) <= n_sigma * se


# ---------------------------------------------------------------------------
# Shared DGPs (module-scoped; deterministic seeds)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def saturated_data():
    """One binary covariate -> saturated logistic propensity.

    P(T=1|x=0)=0.30, P(T=1|x=1)=0.65; effect is HETEROGENEOUS
    (2.0 + 0.5*x) so the ATE / ATT / ATC closed forms differ — a
    swapped-estimand bug cannot slip through.
    """
    rng = np.random.default_rng(20260601)
    n = 4000
    x = rng.binomial(1, 0.45, n)
    p = np.where(x == 1, 0.65, 0.30)
    t = (rng.uniform(size=n) < p).astype(int)
    y = 1.0 + 1.5 * x + 2.0 * t + 0.5 * t * x + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"y": y, "t": t, "x": x})


@pytest.fixture(scope="module")
def saturated_closed_forms(saturated_data):
    """Hand-computed stratified means — INDEPENDENT of statspai.

    Saturation argument: the logit score equations
    sum_i (T_i - p_i) = 0 and sum_i x_i (T_i - p_i) = 0 force the
    fitted propensity in each x-cell to equal the empirical treated
    share p_hat_x = n_{1x}/n_x.  Substituting into the Hajek weights:

      ATE treated term: sum_x sum_{i in (1,x)} y_i / p_hat_x, with
      total weight sum_x n_{1x}/p_hat_x = sum_x n_x = n
        => sum_x (n_x/n) * ybar_{1x}           (and same for control)
      => ATE = sum_x (n_x/n)   * (ybar_{1x} - ybar_{0x})
         ATT = sum_x (n_{1x}/n_1) * (ybar_{1x} - ybar_{0x})
         ATC = sum_x (n_{0x}/n_0) * (ybar_{1x} - ybar_{0x})

    Because the normalizing sums equal n exactly, Horvitz-Thompson
    (divide-by-n, normalize=False) coincides with Hajek here.
    """
    df = saturated_data
    n = len(df)
    cells = {}
    for xv in (0, 1):
        sub = df[df["x"] == xv]
        cells[xv] = dict(
            nx=len(sub),
            n1=int((sub["t"] == 1).sum()),
            n0=int((sub["t"] == 0).sum()),
            y1=sub.loc[sub["t"] == 1, "y"].mean(),
            y0=sub.loc[sub["t"] == 0, "y"].mean(),
        )
    n1 = cells[0]["n1"] + cells[1]["n1"]
    n0 = cells[0]["n0"] + cells[1]["n0"]
    gap = {v: cells[v]["y1"] - cells[v]["y0"] for v in (0, 1)}
    return {
        "ATE": sum(cells[v]["nx"] / n * gap[v] for v in (0, 1)),
        "ATT": sum(cells[v]["n1"] / n1 * gap[v] for v in (0, 1)),
        "ATC": sum(cells[v]["n0"] / n0 * gap[v] for v in (0, 1)),
    }


@pytest.fixture(scope="module")
def randomized_data():
    """Treatment assigned Bernoulli(0.5) INDEPENDENT of (x1, x2).

    True effect 1.2.  Because T is independent of the covariates, the
    population propensity is constant and IPW must collapse to the
    difference in means up to estimated-propensity sampling noise.
    """
    rng = np.random.default_rng(20260602)
    n = 2500
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.5, n)
    t = rng.binomial(1, 0.5, n)
    y = 0.5 + 0.8 * x1 - 0.4 * x2 + 1.2 * t + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})
    df.attrs["true_effect"] = 1.2
    return df


@pytest.fixture(scope="module")
def overlap_good_data():
    """Well-overlapped DGP: ps = expit(0.4*x), x~N(0,1) -> ps in ~[0.15, 0.88].

    Estimated propensities stay far inside (0.01, 0.99), so trim=0.01
    is provably a no-op (np.clip leaves every value unchanged).
    """
    rng = np.random.default_rng(20260603)
    n = 1500
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-0.4 * x))
    t = (rng.uniform(size=n) < p).astype(int)
    y = 1.0 + 0.9 * x + 1.0 * t + rng.normal(scale=0.8, size=n)
    return pd.DataFrame({"y": y, "t": t, "x": x})


@pytest.fixture(scope="module")
def overlap_poor_data():
    """Poor-overlap DGP: ps = expit(3*x) -> ~6% of units have ps < 0.01.

    Raw ATE weights explode (1/ps up to ~1e5); trimming at 0.01 must
    cap the maximum inverse-propensity weight at exactly 100.
    """
    rng = np.random.default_rng(20260604)
    n = 1500
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-3.0 * x))
    t = (rng.uniform(size=n) < p).astype(int)
    y = 1.0 + 0.9 * x + 1.0 * t + rng.normal(scale=0.8, size=n)
    return pd.DataFrame({"y": y, "t": t, "x": x})


@pytest.fixture(scope="module")
def ipw_data():
    """Load the shared seed=20260612 DGP — same csv base R consumed."""
    return pd.read_csv(_FIXTURE_DIR / "ipw_data.csv")


@pytest.fixture(scope="module")
def ipw_r_reference():
    """Load the frozen base-R reference (skip gracefully if absent)."""
    path = _FIXTURE_DIR / "ipw_R.json"
    if not path.exists():
        pytest.skip(
            "ipw_R.json missing — run "
            "Rscript tests/reference_parity/_fixtures/_generate_ipw.R"
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# A. Saturated-propensity closed form (machine-collapse anchor)
# ---------------------------------------------------------------------------


class TestSaturatedClosedForm:
    """sp.ipw on a saturated propensity must equal stratified means.

    Tolerance 1e-9: the identity is exact in exact arithmetic (see
    ``saturated_closed_forms`` derivation); the only slack is the GLM
    IRLS deviance tolerance (~1e-8) on the fitted cell shares.
    Observed slack ~4e-13, so 1e-9 has >3 orders of headroom while
    staying tighter than the 1e-8 machine-collapse bar.
    """

    TOL = 1e-9

    @pytest.mark.parametrize("estimand", ["ATE", "ATT", "ATC"])
    def test_hajek_matches_stratified_means(
        self, saturated_data, saturated_closed_forms, estimand
    ):
        expected = saturated_closed_forms[estimand]
        # n_bootstrap=10: SE unused here; keeps the anchor fast.
        r = sp.ipw(
            saturated_data,
            y="y",
            treat="t",
            covariates=["x"],
            estimand=estimand,
            normalize=True,
            n_bootstrap=10,
            seed=2,
        )
        assert abs(r.estimate - expected) < self.TOL, (
            f"Hajek {estimand}: sp.ipw {r.estimate!r} vs closed form "
            f"{expected!r} (|diff|={abs(r.estimate - expected):.2e})"
        )

    def test_ht_equals_hajek_under_saturation(
        self, saturated_data, saturated_closed_forms
    ):
        """normalize=False (HT, /n) == Hajek when weights sum to n exactly.

        Under saturation sum_i T_i/p_hat_i = sum_x n_x = n, so the HT
        divisor n equals the Hajek divisor — divergence beyond float
        noise means the normalization branch (ipw.py:263-272) broke.
        Tolerance 1e-9 as above (observed ~8e-13).
        """
        expected = saturated_closed_forms["ATE"]
        r_ht = sp.ipw(
            saturated_data,
            y="y",
            treat="t",
            covariates=["x"],
            estimand="ATE",
            normalize=False,
            n_bootstrap=10,
            seed=2,
        )
        assert abs(r_ht.estimate - expected) < self.TOL, (
            f"HT ATE: {r_ht.estimate!r} vs closed form {expected!r} "
            f"(|diff|={abs(r_ht.estimate - expected):.2e})"
        )


# ---------------------------------------------------------------------------
# B. Randomized collapse to difference in means
# ---------------------------------------------------------------------------


class TestRandomizedCollapse:
    """With T independent of X, IPW ≈ difference in means."""

    def test_ipw_agrees_with_diff_in_means(self, randomized_data):
        df = randomized_data
        y, t = df["y"].values, df["t"].values
        # Difference in means + Welch SE — hand-rolled numpy only.
        y1, y0 = y[t == 1], y[t == 0]
        dim = y1.mean() - y0.mean()
        se_dim = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))

        r = sp.ipw(
            df,
            y="y",
            treat="t",
            covariates=["x1", "x2"],
            estimand="ATE",
            n_bootstrap=200,
            seed=7,
        )

        # 4 * combined SE (REFERENCES.md cross-estimator convention);
        # conservative since both estimators share the same sample.
        # Observed |diff| ~0.031 vs band ~0.27.
        band = 4.0 * np.sqrt(r.se**2 + se_dim**2)
        assert abs(r.estimate - dim) <= band, (
            f"IPW {r.estimate:.4f} vs diff-in-means {dim:.4f} " f"(band {band:.4f})"
        )


# ---------------------------------------------------------------------------
# C. Frozen base-R fixture (stats::glm + Hajek means)
# ---------------------------------------------------------------------------


class TestFrozenRParity:
    """sp.ipw vs base R glm(t ~ x1 + x2, binomial) Hajek means.

    Valid anchor because sp.ipw's propensity is the UNPENALIZED logit
    MLE with intercept (statsmodels GLM Binomial, ipw.py:202-212; the
    sklearn fallback also uses penalty=None, ipw.py:217-229) — exactly
    what base R's stats::glm computes.  No L2 mismatch to worry about.

    Tolerance 1e-9: both IRLS solvers iterate deviance to ~1e-8 on a
    well-conditioned n=800 logit; observed |diff| is ~4e-16 (ATE) and
    ~2e-15 (ATT), so 1e-9 has ~6 orders of headroom yet would still
    catch any change in the weighting convention or model spec.
    """

    TOL = 1e-9

    def test_hajek_ate_matches_R(self, ipw_data, ipw_r_reference):
        r = sp.ipw(
            ipw_data,
            y="y",
            treat="t",
            covariates=["x1", "x2"],
            estimand="ATE",
            normalize=True,
            trim=0.0,
            n_bootstrap=10,
            seed=1,
        )
        r_val = ipw_r_reference["hajek_ate"]
        assert abs(r.estimate - r_val) < self.TOL, (
            f"Hajek ATE drifted from base R: Python={r.estimate!r}, "
            f"R={r_val!r} (|diff|={abs(r.estimate - r_val):.2e})"
        )

    def test_hajek_att_matches_R(self, ipw_data, ipw_r_reference):
        r = sp.ipw(
            ipw_data,
            y="y",
            treat="t",
            covariates=["x1", "x2"],
            estimand="ATT",
            normalize=True,
            trim=0.0,
            n_bootstrap=10,
            seed=1,
        )
        r_val = ipw_r_reference["hajek_att"]
        assert abs(r.estimate - r_val) < self.TOL, (
            f"Hajek ATT drifted from base R: Python={r.estimate!r}, "
            f"R={r_val!r} (|diff|={abs(r.estimate - r_val):.2e})"
        )

    def test_fixture_csv_intact(self, ipw_data):
        """Guard that the CSV fixture wasn't accidentally mutated."""
        assert len(ipw_data) == 800
        assert list(ipw_data.columns) == ["y", "t", "x1", "x2"]
        assert int(ipw_data["t"].sum()) == 331

    def test_fixture_R_meta_present(self, ipw_r_reference):
        """Guard that the R fixture records its provenance."""
        meta = ipw_r_reference["meta"]
        assert meta["n"] == 800
        assert meta["n_treated"] == 331
        assert "glm" in meta["formula"]


# ---------------------------------------------------------------------------
# D. CIA recovery on the shared matching_cia_data fixture
# ---------------------------------------------------------------------------


class TestCIARecovery:
    """IPW on matching_cia_data must recover the true effect = 2.0.

    test_matching_parity.py covers sp.match / sp.ebalance / sp.cbps /
    sp.overlap_weights on this fixture but NOT sp.ipw — this is the
    missing anchored coverage.  Effect is homogeneous, so ATE = ATT.
    """

    COVARIATES = ["X1", "X2", "X3"]

    def test_ipw_att_recovers(self, matching_cia_data):
        truth = matching_cia_data.attrs["true_effect"]
        r = sp.ipw(
            matching_cia_data,
            y="y",
            treat="d",
            covariates=self.COVARIATES,
            estimand="ATT",
            n_bootstrap=200,
            seed=42,
        )
        # 4-sigma recovery (REFERENCES.md convention). Observed |z|~0.6.
        assert _within_n_se(
            r.estimate, truth, r.se, n_sigma=4.0
        ), f"IPW ATT: {r.estimate:.4f} vs truth {truth} (SE {r.se:.4f})"

    def test_ipw_ate_recovers(self, matching_cia_data):
        truth = matching_cia_data.attrs["true_effect"]
        r = sp.ipw(
            matching_cia_data,
            y="y",
            treat="d",
            covariates=self.COVARIATES,
            estimand="ATE",
            n_bootstrap=200,
            seed=42,
        )
        # 4-sigma recovery. Observed |z|~1.5.
        assert _within_n_se(
            r.estimate, truth, r.se, n_sigma=4.0
        ), f"IPW ATE: {r.estimate:.4f} vs truth {truth} (SE {r.se:.4f})"


# ---------------------------------------------------------------------------
# E. Trim semantics
# ---------------------------------------------------------------------------


class TestTrimSemantics:
    """trim winsorizes propensities (np.clip at ipw.py:123-124)."""

    def test_trim_noop_on_well_overlapped_dgp(self, overlap_good_data):
        """trim=0.01 with all ps inside (0.01, 0.99) is a bitwise no-op.

        np.clip(ps, 0.01, 0.99) returns ps unchanged when
        min(ps) > 0.01 and max(ps) < 0.99 (guarded below), so the two
        point estimates are computed from identical weight arrays.
        Tolerance 1e-12: exact-identity anchor; observed |diff| = 0.0.
        This trivially also satisfies the spec's < 4-sigma movement.
        """
        r0 = sp.ipw(
            overlap_good_data,
            y="y",
            treat="t",
            covariates=["x"],
            estimand="ATE",
            trim=0.0,
            n_bootstrap=100,
            seed=3,
        )
        r1 = sp.ipw(
            overlap_good_data,
            y="y",
            treat="t",
            covariates=["x"],
            estimand="ATE",
            trim=0.01,
            n_bootstrap=100,
            seed=3,
        )
        # Guard: the no-op argument requires genuine overlap.
        # (model_info reports post-trim ps, ipw.py:152-164; for r0
        # trim=0 so these are the raw fitted propensities.)
        assert r0.model_info["pscore_min"] > 0.01
        assert r0.model_info["pscore_max"] < 0.99
        assert abs(r0.estimate - r1.estimate) < 1e-12, (
            f"trim=0.01 should be a no-op here: " f"{r0.estimate!r} vs {r1.estimate!r}"
        )
        # And (a fortiori) within the 4-sigma combined band.
        band = 4.0 * np.sqrt(r0.se**2 + r1.se**2)
        assert abs(r0.estimate - r1.estimate) <= band

    def test_trim_caps_max_weight_on_poor_overlap_dgp(self, overlap_poor_data):
        """On poor overlap, trim=0.01 caps max 1/ps weight at exactly 100.

        The DGP pushes fitted ps below 1e-4 (guarded), so the untrimmed
        max ATE weight 1/ps_min exceeds 1e4; after np.clip the reported
        pscore_min equals trim exactly (model_info is computed from the
        post-trim pscore, ipw.py:124 then ipw.py:152-164), shrinking the
        max inverse-propensity weight to 1/0.01 = 100.
        Tolerance 1e-12 on pscore_min == trim: np.clip sets the value
        bitwise-exactly to the float 0.01.
        """
        r0 = sp.ipw(
            overlap_poor_data,
            y="y",
            treat="t",
            covariates=["x"],
            estimand="ATE",
            trim=0.0,
            n_bootstrap=10,
            seed=4,
        )
        r1 = sp.ipw(
            overlap_poor_data,
            y="y",
            treat="t",
            covariates=["x"],
            estimand="ATE",
            trim=0.01,
            n_bootstrap=10,
            seed=4,
        )
        ps_min_raw = r0.model_info["pscore_min"]
        ps_min_trim = r1.model_info["pscore_min"]
        # Guard: overlap really is poor (observed ps_min ~5.7e-6).
        assert ps_min_raw < 0.01, f"DGP not poor-overlap: ps_min={ps_min_raw}"
        # Clip lands exactly on the trim boundary.
        assert ps_min_trim == pytest.approx(0.01, abs=1e-12)
        # Max inverse-propensity weight shrinks (observed ~1.8e5 -> 100).
        assert 1.0 / ps_min_trim < 1.0 / ps_min_raw
        # Weights genuinely changed -> point estimate moved
        # (observed |diff| ~0.025, far above float noise).
        assert abs(r0.estimate - r1.estimate) > 1e-10


# ---------------------------------------------------------------------------
# F. Bootstrap SE calibration vs Monte-Carlo SD
# ---------------------------------------------------------------------------


class TestBootstrapSECalibration:
    """Mean bootstrap SE must track the empirical SD across MC draws."""

    def test_bootstrap_se_within_3x_of_monte_carlo_sd(self):
        """30 independent DGP replications, n=300, n_bootstrap=100.

        Tolerance [1/3, 3] x empirical SD: with 30 reps the SD itself
        has ~13% relative MC error and the bootstrap adds ~7% noise per
        rep (averaged out over 30), so a correctly calibrated SE sits
        well inside the band (observed ratio 0.94) while an
        order-of-magnitude SE bug (variance-vs-SD, missing sqrt(n))
        lands far outside it.
        """
        n, reps = 300, 30
        estimates, ses = [], []
        for rep in range(reps):
            rng = np.random.default_rng(50_000 + rep)
            x1 = rng.normal(size=n)
            x2 = rng.normal(size=n)
            p = 1.0 / (1.0 + np.exp(-(0.5 * x1 - 0.4 * x2)))
            t = (rng.uniform(size=n) < p).astype(int)
            y = 1.0 + 1.0 * x1 - 0.6 * x2 + 1.5 * t + rng.normal(scale=1.0, size=n)
            df = pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})
            r = sp.ipw(
                df,
                y="y",
                treat="t",
                covariates=["x1", "x2"],
                estimand="ATE",
                n_bootstrap=100,
                seed=rep,
            )
            estimates.append(r.estimate)
            ses.append(r.se)
        emp_sd = float(np.std(estimates, ddof=1))
        mean_se = float(np.mean(ses))
        assert emp_sd / 3.0 < mean_se < emp_sd * 3.0, (
            f"Bootstrap SE miscalibrated: mean boot SE {mean_se:.4f} vs "
            f"Monte-Carlo SD {emp_sd:.4f} (ratio {mean_se / emp_sd:.2f})"
        )
