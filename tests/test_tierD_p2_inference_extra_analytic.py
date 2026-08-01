"""Tier D P2 known-truth upgrades — CR3 jackknife, PATE, wild-cluster CIs.

Part of the P1/P2 "Tier D analytic special-cases" campaign. These entry
points carried only weak (boolean / shape / not-None) asserts. Each test
below anchors on a pre-known number or decision:

    sp.cr3_jackknife_vcov     CR3 is the leave-one-cluster-out jackknife
                              variance with (G-1)/G scaling -- reproduced
                              BY HAND to machine precision. CR1 = CR0 x the
                              finite-sample factor (>=1) exactly, so the
                              CR1 >= CR0 SE ordering is a theorem; CR3 is
                              conservative on average.
    sp.pate                   with a randomized DGP and a KNOWN constant
                              treatment effect tau, PATE = SATE = tau, so
                              the estimate recovers tau within tolerance.
    sp.subcluster_wild_bootstrap   wild cluster bootstrap-t is deterministic
    sp.wild_cluster_ci_inv         given a seed: true null -> large p / CI
                              covers 0; strong effect -> CI excludes 0 and
                              brackets beta_hat; same seed -> identical out.

Purely additive -- no estimator numerics changed (campaign red line).

NB (semantics): sp.pate returns a single PATE point estimate + bootstrap
SE/CI; it does NOT return a SATE-vs-PATE variance decomposition, so the
"PATE variance >= SATE variance" check is not applicable here and is
intentionally omitted (confirmed from the docstring + CausalResult fields).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# sp.cr3_jackknife_vcov — leave-one-cluster-out jackknife variance
# ---------------------------------------------------------------------------
class TestCR3JackknifeAnalytic:
    def _design(self, seed=7, G=6, n_per=5):
        rng = np.random.default_rng(seed)
        n = G * n_per
        cluster = np.repeat(np.arange(G), n_per)
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = X @ np.array([1.0, 2.0]) + rng.normal(0, 1, n)
        return X, y, cluster, G

    def test_matches_hand_leave_one_cluster_out(self):
        # CR3 = (G-1)/G * sum_g (b_(-g) - b_full)(b_(-g) - b_full)'
        X, y, cluster, G = self._design()
        coef_full = np.linalg.lstsq(X, y, rcond=None)[0]
        coefs = np.empty((G, X.shape[1]))
        for g in range(G):
            m = cluster != g
            coefs[g] = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
        diff = coefs - coef_full
        V_hand = (G - 1) / G * (diff.T @ diff)
        V = sp.cr3_jackknife_vcov(X, y, cluster)
        np.testing.assert_allclose(V, V_hand, atol=1e-8, rtol=0)

    def test_scaling_factor_is_g_minus_one_over_g(self):
        # Recompute WITHOUT the (G-1)/G scaling and confirm the helper is
        # exactly (G-1)/G times the raw jackknife outer product.
        X, y, cluster, G = self._design(seed=3, G=8, n_per=6)
        coef_full = np.linalg.lstsq(X, y, rcond=None)[0]
        coefs = np.empty((G, X.shape[1]))
        for g in range(G):
            m = cluster != g
            coefs[g] = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
        diff = coefs - coef_full
        raw = diff.T @ diff
        V = sp.cr3_jackknife_vcov(X, y, cluster)
        np.testing.assert_allclose(V, (G - 1) / G * raw, atol=1e-8, rtol=0)

    def test_symmetric_and_psd(self):
        X, y, cluster, G = self._design(seed=11, G=10, n_per=8)
        V = sp.cr3_jackknife_vcov(X, y, cluster)
        np.testing.assert_allclose(V, V.T, atol=1e-10)
        eig = np.linalg.eigvalsh(0.5 * (V + V.T))
        assert eig.min() >= -1e-10

    def test_cr1_equals_cr0_times_finite_sample_factor(self):
        # Known truth: CR1 = CR0 * sqrt(G/(G-1) * (n-1)/(n-k)) elementwise,
        # so CR1 >= CR0 always (factor >= 1). This is the rigorous ordering.
        X, y, cluster, G = self._design(seed=3, G=10, n_per=8)
        n, k = X.shape
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        XtX_inv = np.linalg.inv(X.T @ X)
        meat0 = np.zeros((k, k))
        for g in range(G):
            m = cluster == g
            u = X[m].T @ resid[m]
            meat0 += np.outer(u, u)
        se0 = np.sqrt(np.diag(XtX_inv @ meat0 @ XtX_inv))
        se1 = np.asarray(sp.cluster_robust_se(X, resid, cluster))
        factor = np.sqrt((G / (G - 1)) * ((n - 1) / (n - k)))
        np.testing.assert_allclose(se1, se0 * factor, rtol=1e-10)
        assert np.all(se1 >= se0)

    def test_cr3_conservative_on_average_vs_cr1(self):
        # CR3 is "often more conservative" (Bell-McCaffrey): the mean SE
        # ratio CR3/CR1 across coefficients is >= 1 on a balanced design.
        # (Strict per-coefficient CR3 >= CR1 is NOT a theorem, so we only
        # anchor the on-average tendency here.)
        X, y, cluster, G = self._design(seed=3, G=10, n_per=8)
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        se1 = np.asarray(sp.cluster_robust_se(X, resid, cluster))
        se3 = np.sqrt(np.diag(sp.cr3_jackknife_vcov(X, y, cluster)))
        assert np.mean(se3 / se1) >= 1.0


# ---------------------------------------------------------------------------
# sp.pate — population average treatment effect recovers known tau
# ---------------------------------------------------------------------------
class TestPATEKnownTau:
    TAU = 3.0

    def _data(self):
        # Randomized treatment, CONSTANT effect tau, with selection-on-X:
        # the experiment oversamples high "age", so reweighting to the target
        # matters -- yet PATE = SATE = tau because the effect is homogeneous.
        rng = np.random.default_rng(0)
        nt = 4000
        df_t = pd.DataFrame(
            {
                "age": rng.normal(0.0, 1.0, nt),
                "edu": rng.normal(0.0, 1.0, nt),
            }
        )
        ne = 3000
        age_e = rng.normal(0.8, 1.0, ne)
        edu_e = rng.normal(0.0, 1.0, ne)
        d = rng.integers(0, 2, ne)
        y = 1.0 + 0.5 * age_e + 0.3 * edu_e + self.TAU * d + rng.normal(0, 1, ne)
        df_e = pd.DataFrame({"y": y, "d": d, "age": age_e, "edu": edu_e})
        return df_e, df_t

    @pytest.mark.parametrize("method", ["ipw", "calibration"])
    def test_recovers_constant_tau(self, method):
        df_e, df_t = self._data()
        r = sp.pate(
            df_e,
            df_t,
            y="y",
            treatment="d",
            covariates=["age", "edu"],
            method=method,
            n_boot=200,
            seed=1,
        )
        # Constant effect -> PATE = SATE = tau regardless of weighting.
        assert r.estimate == pytest.approx(self.TAU, abs=0.25)

    def test_ci_covers_true_tau(self):
        df_e, df_t = self._data()
        r = sp.pate(
            df_e,
            df_t,
            y="y",
            treatment="d",
            covariates=["age", "edu"],
            method="ipw",
            n_boot=300,
            seed=2,
        )
        lo, hi = r.ci
        assert lo <= self.TAU <= hi

    def test_aipw_doubly_robust_recovers_tau(self):
        df_e, df_t = self._data()
        r = sp.pate(
            df_e,
            df_t,
            y="y",
            treatment="d",
            covariates=["age", "edu"],
            method="aipw",
            n_boot=200,
            seed=3,
        )
        assert r.estimate == pytest.approx(self.TAU, abs=0.3)


# ---------------------------------------------------------------------------
# sp.subcluster_wild_bootstrap — wild cluster bootstrap-t
# ---------------------------------------------------------------------------
class TestSubclusterWildBootstrap:
    def _data(self, seed, effect, G=12, n_per=30):
        rng = np.random.default_rng(seed)
        n = G * n_per
        cl = np.repeat(np.arange(G), n_per)
        cl_eff = rng.normal(0, 1, G)[cl]
        x = rng.normal(0, 1, n)
        y = 1.0 + effect * x + cl_eff + rng.normal(0, 1, n)
        return pd.DataFrame({"y": y, "x": x, "cl": cl})

    def test_strong_effect_rejects_and_ci_excludes_zero(self):
        df = self._data(0, effect=2.0)
        r = sp.subcluster_wild_bootstrap(
            df, "y", ["x"], "cl", test_var="x", n_boot=999, seed=42
        )
        lo, hi = r["ci_boot"]
        assert r["p_boot"] < 0.05
        assert not (lo <= 0.0 <= hi)  # CI excludes the null
        assert lo < r["beta_hat"] < hi  # CI brackets point est

    def test_true_null_not_rejected_and_ci_covers_zero(self):
        df = self._data(1, effect=0.0)
        r = sp.subcluster_wild_bootstrap(
            df, "y", ["x"], "cl", test_var="x", n_boot=999, seed=42
        )
        lo, hi = r["ci_boot"]
        assert r["p_boot"] > 0.05  # large p under true null
        assert lo <= 0.0 <= hi  # CI covers the null
        assert lo < r["beta_hat"] < hi

    def test_deterministic_given_seed(self):
        df = self._data(0, effect=2.0)
        a = sp.subcluster_wild_bootstrap(
            df, "y", ["x"], "cl", test_var="x", n_boot=499, seed=7
        )
        b = sp.subcluster_wild_bootstrap(
            df, "y", ["x"], "cl", test_var="x", n_boot=499, seed=7
        )
        assert a["p_boot"] == b["p_boot"]
        assert a["ci_boot"] == b["ci_boot"]
        assert a["t_stat"] == b["t_stat"]


# ---------------------------------------------------------------------------
# sp.wild_cluster_ci_inv — CI by bootstrap p-value inversion
# ---------------------------------------------------------------------------
class TestWildClusterCIInversion:
    def _data(self, seed, effect, G=12, n_per=30):
        rng = np.random.default_rng(seed)
        n = G * n_per
        cl = np.repeat(np.arange(G), n_per)
        cl_eff = rng.normal(0, 1, G)[cl]
        x = rng.normal(0, 1, n)
        y = 1.0 + effect * x + cl_eff + rng.normal(0, 1, n)
        return pd.DataFrame({"y": y, "x": x, "cl": cl})

    def test_ci_brackets_point_estimate(self):
        df = self._data(0, effect=2.0)
        r = sp.wild_cluster_ci_inv(
            df, "y", ["x"], "cl", test_var="x", n_boot=499, seed=42
        )
        lo, hi = r["ci"]
        assert lo < r["beta_hat"] < hi  # lower < beta < upper

    def test_strong_effect_ci_excludes_zero(self):
        df = self._data(0, effect=2.0)
        r = sp.wild_cluster_ci_inv(
            df, "y", ["x"], "cl", test_var="x", n_boot=499, seed=42
        )
        lo, hi = r["ci"]
        assert not (lo <= 0.0 <= hi)

    def test_true_null_ci_covers_zero(self):
        df = self._data(1, effect=0.0)
        r = sp.wild_cluster_ci_inv(
            df, "y", ["x"], "cl", test_var="x", n_boot=499, seed=42
        )
        lo, hi = r["ci"]
        assert lo <= 0.0 <= hi

    def test_deterministic_given_seed(self):
        df = self._data(0, effect=2.0)
        a = sp.wild_cluster_ci_inv(
            df, "y", ["x"], "cl", test_var="x", n_boot=299, grid_size=21, seed=5
        )
        b = sp.wild_cluster_ci_inv(
            df, "y", ["x"], "cl", test_var="x", n_boot=299, grid_size=21, seed=5
        )
        assert a["ci"] == b["ci"]
        np.testing.assert_array_equal(a["p_grid"], b["p_grid"])
