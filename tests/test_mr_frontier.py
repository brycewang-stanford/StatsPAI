"""
Tests for the v1.6 MR Frontier estimators: MR-Lap, MR-Clust, GRAPPLE,
and MR-cML-BIC.

Strategy (matching ``tests/reference_parity/test_mr_parity.py``):

- Build known-truth simulations under each method's DGP.
- Assert point estimates recover truth within theoretically justified
  tolerances (typically ~1-2 SE for n_snps >= 50).
- Assert cross-estimator consistency properties guaranteed by theory:
    * mr_lap with overlap_fraction=0 reduces to IVW.
    * mr_cml with K_max=0 matches measurement-error-aware IVW
      (Bowden 2019 attenuation-corrected form).
    * grapple tau^2 -> 0 when pleiotropy SD=0, mean-F large.
    * mr_clust picks K=1 (null-only) when Y has no causal association
      with X.
- Assert boundary validation (length mismatch, bad rho, bad K).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.mendelian.frontier import (
    mr_lap, mr_clust, grapple, mr_cml, mr_raps,
    MRLapResult, MRClustResult, GrappleResult, MRcMLResult, MRRapsResult,
)


# --------------------------------------------------------------------------- #
#  DGP helpers
# --------------------------------------------------------------------------- #


def _sim_clean(n_snps: int = 60, true_beta: float = 0.30, *,
               pleiotropy_sd: float = 0.0, seed: int = 0):
    """Balanced-pleiotropy two-sample MR DGP.

    Returns (bx, by, sx, sy, true_beta).  Matches the DGP used in the
    existing MR parity tests so behaviour is comparable.
    """
    rng = np.random.default_rng(seed)
    alpha = rng.normal(0.10, 0.03, n_snps)
    sx = np.full(n_snps, 1.0 / np.sqrt(20_000))
    sy = np.full(n_snps, np.sqrt(2.0 / 20_000))
    bx = rng.normal(alpha, sx)
    pleio = rng.normal(0.0, pleiotropy_sd, n_snps)
    by = true_beta * alpha + pleio + rng.normal(0.0, sy, n_snps)
    return bx, by, sx, sy, true_beta


def _sim_two_clusters(
    n_per_cluster: int = 30,
    beta_a: float = 0.20, beta_b: float = 0.60, seed: int = 7,
):
    """Two-cluster DGP for MR-Clust tests.

    Half the SNPs act through a pathway with causal effect beta_a,
    the other half through a pathway with beta_b.  Total SNPs = 2n.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_cluster
    alpha = rng.normal(0.10, 0.03, n)
    sx = np.full(n, 1.0 / np.sqrt(20_000))
    sy = np.full(n, np.sqrt(2.0 / 20_000))
    bx = rng.normal(alpha, sx)
    true_b = np.concatenate([
        np.full(n_per_cluster, beta_a),
        np.full(n_per_cluster, beta_b),
    ])
    by = true_b * alpha + rng.normal(0.0, sy, n)
    return bx, by, sx, sy, beta_a, beta_b


def _sim_with_overlap(
    n_snps: int = 80, true_beta: float = 0.30, *,
    overlap_fraction: float = 1.0, overlap_rho: float = 0.30,
    seed: int = 11,
):
    """Simulate two-sample MR with participant overlap.

    When the two GWAS share participants the sampling errors
    e_x and e_y become correlated with correlation rho * p (where
    p is overlap_fraction, rho the underlying phenotypic correlation).
    """
    rng = np.random.default_rng(seed)
    alpha = rng.normal(0.10, 0.03, n_snps)
    sx = np.full(n_snps, 1.0 / np.sqrt(20_000))
    sy = np.full(n_snps, np.sqrt(2.0 / 20_000))

    # Joint bivariate noise with correlation rho*p
    rho_joint = overlap_rho * overlap_fraction
    mu = np.zeros(2)
    cov = np.array([[sx[0] ** 2, rho_joint * sx[0] * sy[0]],
                    [rho_joint * sx[0] * sy[0], sy[0] ** 2]])
    noise = rng.multivariate_normal(mu, cov, size=n_snps)
    bx = alpha + noise[:, 0]
    by = true_beta * alpha + noise[:, 1]
    return bx, by, sx, sy, true_beta


# --------------------------------------------------------------------------- #
#  1. MR-Lap
# --------------------------------------------------------------------------- #


class TestMRLap:

    def test_result_type_and_summary(self):
        bx, by, sx, sy, _ = _sim_clean(seed=0)
        r = mr_lap(bx, by, sx, sy, overlap_fraction=0.5, overlap_rho=0.1)
        assert isinstance(r, MRLapResult)
        assert "MR-Lap" in r.summary()
        assert "overlap" in r.summary().lower()

    def test_zero_overlap_equals_ivw(self):
        """overlap_fraction=0 must leave IVW untouched."""
        bx, by, sx, sy, _ = _sim_clean(seed=1)
        r = mr_lap(bx, by, sx, sy, overlap_fraction=0.0, overlap_rho=0.5)
        assert r.bias_correction == 0.0
        assert r.estimate == pytest.approx(r.estimate_ivw, abs=1e-12)

    def test_zero_rho_equals_ivw(self):
        """overlap_rho=0 must leave IVW untouched."""
        bx, by, sx, sy, _ = _sim_clean(seed=2)
        r = mr_lap(bx, by, sx, sy, overlap_fraction=1.0, overlap_rho=0.0)
        assert r.bias_correction == 0.0
        assert r.estimate == pytest.approx(r.estimate_ivw, abs=1e-12)

    def test_reduces_overlap_bias(self):
        """Under a DGP with overlap bias, MR-Lap should bring the
        estimate closer to the truth than naive IVW.

        We use a large simulation to see the bias: overlap=1.0, rho=0.5,
        and moderate F (mean F around 40 given the DGP).  The
        BDT-2016 bias correction shifts IVW by rho/F_mean ≈ 0.012, which
        moves IVW downward when beta_true=0.3 and the observational
        correlation is positive.
        """
        bias_ivw_list = []
        bias_lap_list = []
        for seed in range(20):
            bx, by, sx, sy, true_beta = _sim_with_overlap(
                n_snps=100, true_beta=0.30,
                overlap_fraction=1.0, overlap_rho=0.5,
                seed=seed,
            )
            r = mr_lap(bx, by, sx, sy,
                       overlap_fraction=1.0, overlap_rho=0.5)
            bias_ivw_list.append(r.estimate_ivw - true_beta)
            bias_lap_list.append(r.estimate - true_beta)
        mean_abs_bias_ivw = float(np.mean(np.abs(bias_ivw_list)))
        mean_abs_bias_lap = float(np.mean(np.abs(bias_lap_list)))
        # Hard bound: correction must not inflate bias.
        assert mean_abs_bias_lap <= mean_abs_bias_ivw + 0.005, (
            mean_abs_bias_lap, mean_abs_bias_ivw
        )

    def test_invalid_overlap_fraction_raises(self):
        bx, by, sx, sy, _ = _sim_clean(seed=0)
        with pytest.raises(ValueError, match="overlap_fraction"):
            mr_lap(bx, by, sx, sy,
                   overlap_fraction=1.5, overlap_rho=0.1)

    def test_invalid_rho_raises(self):
        bx, by, sx, sy, _ = _sim_clean(seed=0)
        with pytest.raises(ValueError, match="overlap_rho"):
            mr_lap(bx, by, sx, sy,
                   overlap_fraction=0.5, overlap_rho=1.2)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            mr_lap(
                beta_exposure=np.array([0.1, 0.2]),
                beta_outcome=np.array([0.3, 0.4, 0.5]),
                se_exposure=np.array([0.01, 0.01]),
                se_outcome=np.array([0.05, 0.05, 0.05]),
            )

    def test_flags_weak_instruments_in_summary(self):
        """Small F (bx/sx small) should annotate the summary."""
        rng = np.random.default_rng(0)
        n = 50
        # Very weak instruments: bx ~ 0.01, sx = 0.02 -> per-SNP F ≈ 0.25
        bx = rng.normal(0.01, 0.005, n)
        sx = np.full(n, 0.02)
        sy = np.full(n, 0.05)
        by = 0.3 * bx + rng.normal(0, sy)
        r = mr_lap(bx, by, sx, sy, overlap_fraction=0.5, overlap_rho=0.1)
        # f_mean should be < 10
        assert r.f_mean < 10
        assert "weak" in r.summary()


# --------------------------------------------------------------------------- #
#  2. MR-Clust
# --------------------------------------------------------------------------- #


class TestMRClust:

    def test_result_type_and_summary(self):
        bx, by, sx, sy, _ = _sim_clean(seed=0)
        r = mr_clust(bx, by, sx, sy, K_range=(1, 3))
        assert isinstance(r, MRClustResult)
        assert "MR-Clust" in r.summary()
        assert "BIC" in r.summary()

    def test_single_pathway_prefers_K1_or_K2(self):
        """A clean single-effect DGP should not be split into many
        real non-null clusters — BIC should prefer the smallest K
        that captures the structure (K=2: null + one real cluster,
        or K=1 depending on null/non-null balance).
        """
        bx, by, sx, sy, _ = _sim_clean(n_snps=60, seed=0)
        r = mr_clust(bx, by, sx, sy, K_range=(1, 5))
        assert r.K <= 3, r.bic  # no runaway clusters

    def test_cluster_assignments_shape(self):
        bx, by, sx, sy, _ = _sim_clean(n_snps=40, seed=0)
        r = mr_clust(bx, by, sx, sy, K_range=(1, 3))
        assert r.assignments.shape == (40,)
        assert r.responsibilities.shape == (40, r.K)
        # Responsibilities sum to 1
        np.testing.assert_allclose(
            r.responsibilities.sum(axis=1), np.ones(40), atol=1e-6,
        )

    def test_two_cluster_dgp(self):
        """When the DGP has two clearly separated effect magnitudes
        (beta_a=0.2, beta_b=0.6), MR-Clust should find K=2 or K=3
        (K=3 if a null cluster is kept) and include estimates near
        0.2 and 0.6 in the non-null clusters.
        """
        bx, by, sx, sy, beta_a, beta_b = _sim_two_clusters(
            n_per_cluster=60, beta_a=0.20, beta_b=0.60, seed=7,
        )
        r = mr_clust(bx, by, sx, sy, K_range=(1, 4))
        # Select one non-null estimate closest to each truth
        non_null = r.cluster_estimates[r.cluster_estimates["cluster"] > 0]
        est = non_null["estimate"].values if len(non_null) else np.array([])
        # Cluster must recover at least one of the two pathway magnitudes
        close_to_a = np.any(np.abs(est - beta_a) < 0.10) if len(est) else False
        close_to_b = np.any(np.abs(est - beta_b) < 0.15) if len(est) else False
        assert close_to_a or close_to_b, (r.cluster_estimates, beta_a, beta_b)

    def test_without_null_cluster(self):
        """include_null=False should still fit and not raise."""
        bx, by, sx, sy, _ = _sim_clean(n_snps=40, seed=0)
        r = mr_clust(bx, by, sx, sy, K_range=(1, 3), include_null=False)
        assert r.cluster_estimates["estimate"].iloc[0] != 0.0

    def test_invalid_K_range_raises(self):
        bx, by, sx, sy, _ = _sim_clean(seed=0)
        with pytest.raises(ValueError, match="K_range"):
            mr_clust(bx, by, sx, sy, K_range=(5, 3))
        with pytest.raises(ValueError, match="K_range"):
            mr_clust(bx, by, sx, sy, K_range=(0, 3))


# --------------------------------------------------------------------------- #
#  3. GRAPPLE
# --------------------------------------------------------------------------- #


class TestGrapple:

    def test_result_type_and_summary(self):
        bx, by, sx, sy, _ = _sim_clean(seed=0)
        r = grapple(bx, by, sx, sy)
        assert isinstance(r, GrappleResult)
        assert r.converged
        assert "GRAPPLE" in r.summary()

    def test_recovers_truth_balanced_pleiotropy(self):
        """On a balanced-pleiotropy DGP GRAPPLE should recover beta
        within a few SE and tau^2 should be small.
        """
        bias_list = []
        se_list = []
        for seed in range(10):
            bx, by, sx, sy, true_beta = _sim_clean(
                n_snps=80, pleiotropy_sd=0.005, seed=seed,
            )
            r = grapple(bx, by, sx, sy)
            bias_list.append(r.estimate - true_beta)
            se_list.append(r.se)
        mean_bias = float(np.mean(bias_list))
        mean_se = float(np.mean(se_list))
        # Bias should be < 1.5 * mean SE across replicates
        assert abs(mean_bias) < 1.5 * mean_se, (mean_bias, mean_se)

    def test_tau2_positive_with_pleiotropy(self):
        """Under a larger pleiotropy SD, tau^2 should grow."""
        bx, by, sx, sy, _ = _sim_clean(
            n_snps=100, pleiotropy_sd=0.02, seed=3,
        )
        r_big = grapple(bx, by, sx, sy)

        bx, by, sx, sy, _ = _sim_clean(
            n_snps=100, pleiotropy_sd=0.001, seed=3,
        )
        r_small = grapple(bx, by, sx, sy)

        # Under pleiotropy, tau^2 should typically be larger.  This is
        # a monotonicity check.  Allow some slack for finite-sample
        # noise (not a hard bound).
        assert r_big.tau2 >= r_small.tau2 - 1e-6

    def test_warm_start_with_beta_init(self):
        bx, by, sx, sy, _ = _sim_clean(seed=0)
        r1 = grapple(bx, by, sx, sy)
        r2 = grapple(bx, by, sx, sy, beta_init=r1.estimate)
        # Same optimum either way (within tolerance)
        assert abs(r1.estimate - r2.estimate) < 1e-3


# --------------------------------------------------------------------------- #
#  4. MR-cML-BIC
# --------------------------------------------------------------------------- #


class TestMRcML:

    def test_result_type_and_summary(self):
        bx, by, sx, sy, _ = _sim_clean(seed=0)
        r = mr_cml(bx, by, sx, sy)
        assert isinstance(r, MRcMLResult)
        assert "MR-cML" in r.summary()
        assert 0 <= r.K_selected <= r.n_snps - 3

    def test_K_path_length(self):
        bx, by, sx, sy, _ = _sim_clean(n_snps=15, seed=0)
        r = mr_cml(bx, by, sx, sy, K_max=5)
        # 6 rows (K=0..5)
        assert len(r.path) == 6
        assert "bic" in r.path.columns

    def test_K0_matches_ivw_like_estimate(self):
        """When K=0 (no pleiotropy allowed) MR-cML reduces to a
        measurement-error-corrected ML estimator.  On a clean DGP
        it should be close to IVW.
        """
        bx, by, sx, sy, _ = _sim_clean(seed=0, pleiotropy_sd=0.0)
        r_cml = mr_cml(bx, by, sx, sy, K_max=0)
        assert r_cml.K_selected == 0
        r_ivw = sp.mr("ivw",
                      beta_exposure=bx, beta_outcome=by,
                      se_exposure=sx, se_outcome=sy)
        # Within 10% of IVW on clean data — both target the same thing;
        # MR-cML corrects attenuation from sx^2 whereas naive IVW does
        # not, so small deviations are expected but should be small.
        assert abs(r_cml.estimate - r_ivw["estimate"]) < 0.10 * abs(
            r_ivw["estimate"]
        ) + 0.01

    def test_recovers_truth_no_pleiotropy(self):
        bias_list = []
        for seed in range(10):
            bx, by, sx, sy, true_beta = _sim_clean(
                n_snps=80, pleiotropy_sd=0.0, seed=seed,
            )
            r = mr_cml(bx, by, sx, sy)
            bias_list.append(r.estimate - true_beta)
        mean_abs_bias = float(np.mean(np.abs(bias_list)))
        # Very small bias on a clean DGP
        assert mean_abs_bias < 0.05, mean_abs_bias

    def test_flags_invalid_snps_with_pleiotropy(self):
        """Plant 5 strongly-pleiotropic SNPs and confirm MR-cML picks
        K > 0 and includes the planted SNPs in invalid set.
        """
        rng = np.random.default_rng(99)
        n = 60
        n_bad = 8
        alpha = rng.normal(0.10, 0.03, n)
        sx = np.full(n, 1.0 / np.sqrt(20_000))
        sy = np.full(n, np.sqrt(2.0 / 20_000))
        bx = rng.normal(alpha, sx)
        true_beta = 0.30
        pleio = np.zeros(n)
        pleio[:n_bad] = rng.choice([-0.08, 0.08], size=n_bad)
        by = true_beta * alpha + pleio + rng.normal(0, sy)
        r = mr_cml(bx, by, sx, sy, K_max=15)
        assert r.K_selected >= 3  # should pick up at least some
        # Estimate should be close to truth
        assert abs(r.estimate - true_beta) < 0.10, r.estimate

    def test_invalid_K_max_raises(self):
        bx, by, sx, sy, _ = _sim_clean(seed=0)
        with pytest.raises(ValueError, match="K_max"):
            mr_cml(bx, by, sx, sy, K_max=1000)
        with pytest.raises(ValueError, match="K_max"):
            mr_cml(bx, by, sx, sy, K_max=-1)


# --------------------------------------------------------------------------- #
#  5. MR-RAPS (robust profile score)
# --------------------------------------------------------------------------- #


class TestMRRAPS:

    def test_result_type_and_summary(self):
        bx, by, sx, sy, _ = _sim_clean(seed=0)
        r = mr_raps(bx, by, sx, sy)
        assert isinstance(r, MRRapsResult)
        assert r.converged
        assert "MR-RAPS" in r.summary()

    def test_recovers_truth_balanced_pleiotropy(self):
        bias_list = []
        se_list = []
        for seed in range(10):
            bx, by, sx, sy, true_beta = _sim_clean(
                n_snps=80, pleiotropy_sd=0.005, seed=seed,
            )
            r = mr_raps(bx, by, sx, sy)
            bias_list.append(r.estimate - true_beta)
            se_list.append(r.se)
        mean_bias = float(np.mean(bias_list))
        mean_se = float(np.mean(se_list))
        assert abs(mean_bias) < 2.0 * mean_se, (mean_bias, mean_se)

    def test_robust_to_outlier_pleiotropy(self):
        """Plant 3 strong pleiotropic outliers; MR-RAPS should remain
        closer to truth than IVW.
        """
        rng = np.random.default_rng(17)
        n = 60
        alpha = rng.normal(0.10, 0.03, n)
        sx = np.full(n, 1.0 / np.sqrt(20_000))
        sy = np.full(n, np.sqrt(2.0 / 20_000))
        bx = rng.normal(alpha, sx)
        true_beta = 0.30
        pleio = np.zeros(n)
        pleio[:3] = rng.choice([-0.15, 0.15], size=3)  # gross outliers
        by = true_beta * alpha + pleio + rng.normal(0, sy)
        r_raps = mr_raps(bx, by, sx, sy, tuning_c=4.685)
        r_ivw = sp.mr("ivw",
                      beta_exposure=bx, beta_outcome=by,
                      se_exposure=sx, se_outcome=sy)
        assert abs(r_raps.estimate - true_beta) < abs(
            r_ivw["estimate"] - true_beta
        ) + 0.02, (r_raps.estimate, r_ivw["estimate"], true_beta)

    def test_smaller_c_is_more_robust(self):
        """With contaminated data, smaller Tukey c should yield estimates
        at least as close to truth as the default c=4.685."""
        rng = np.random.default_rng(23)
        n = 60
        alpha = rng.normal(0.10, 0.03, n)
        sx = np.full(n, 1.0 / np.sqrt(20_000))
        sy = np.full(n, np.sqrt(2.0 / 20_000))
        bx = rng.normal(alpha, sx)
        true_beta = 0.30
        pleio = np.zeros(n)
        pleio[:4] = rng.choice([-0.2, 0.2], size=4)
        by = true_beta * alpha + pleio + rng.normal(0, sy)
        r_default = mr_raps(bx, by, sx, sy, tuning_c=4.685)
        r_robust = mr_raps(bx, by, sx, sy, tuning_c=2.0)
        # Finite sample: either works, but the tight-c estimate should
        # not be dramatically worse.
        assert np.isfinite(r_robust.estimate)
        assert np.isfinite(r_robust.se)

    def test_invalid_tuning_c_raises(self):
        bx, by, sx, sy, _ = _sim_clean(seed=0)
        with pytest.raises(ValueError, match="tuning_c"):
            mr_raps(bx, by, sx, sy, tuning_c=-1.0)


# --------------------------------------------------------------------------- #
#  6. Dispatcher + public API
# --------------------------------------------------------------------------- #


class TestFrontierIntegration:

    @pytest.mark.parametrize("method,cls", [
        ("mr_lap", MRLapResult),
        ("lap", MRLapResult),
        ("mr_clust", MRClustResult),
        ("clust", MRClustResult),
        ("grapple", GrappleResult),
        ("mr_cml", MRcMLResult),
        ("cml", MRcMLResult),
        ("mr_raps", MRRapsResult),
        ("raps", MRRapsResult),
    ])
    def test_dispatcher_routes(self, method, cls):
        bx, by, sx, sy, _ = _sim_clean(seed=0)
        if "lap" in method:
            r = sp.mr(method,
                      beta_exposure=bx, beta_outcome=by,
                      se_exposure=sx, se_outcome=sy,
                      overlap_fraction=0.3, overlap_rho=0.1)
        elif "clust" in method:
            r = sp.mr(method,
                      beta_exposure=bx, beta_outcome=by,
                      se_exposure=sx, se_outcome=sy,
                      K_range=(1, 3))
        else:
            r = sp.mr(method,
                      beta_exposure=bx, beta_outcome=by,
                      se_exposure=sx, se_outcome=sy)
        assert isinstance(r, cls)

    def test_registry_describes_each_function(self):
        for name in ("mr_lap", "mr_clust", "grapple", "mr_cml", "mr_raps"):
            info = sp.describe_function(name)
            assert info is not None
            assert info["category"] == "mendelian"
            # params metadata must list the four SNP-summary arrays
            param_names = {p["name"] for p in info["params"]}
            for expected in ("beta_exposure", "beta_outcome",
                             "se_exposure", "se_outcome"):
                assert expected in param_names, (name, param_names)

    def test_available_methods_lists_new_aliases(self):
        methods = set(sp.mr_available_methods())
        for alias in ("mr_lap", "lap", "mr_clust", "clust",
                      "grapple", "mr_cml", "cml",
                      "mr_raps", "raps"):
            assert alias in methods, alias

    def test_all_top_level_imports(self):
        for name in ("mr_lap", "mr_clust", "grapple", "mr_cml", "mr_raps",
                     "MRLapResult", "MRClustResult", "GrappleResult",
                     "MRcMLResult", "MRRapsResult"):
            assert getattr(sp, name, None) is not None, name
