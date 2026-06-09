"""Tier D P2 known-truth upgrades — randomization inference & cluster-robust SE.

Part of the P1/P2 "Tier D analytic special-cases" campaign (see
``.tierd_campaign/CAMPAIGN.md``). Both were graded ``weak`` by
``scripts/tierd_classify.py``. Each anchors to a known truth:

    sp.fisher_exact       Fisher randomization test: the observed statistic is
                          the difference in means; a strong effect rejects the
                          sharp null, a null DGP does not.
    sp.cluster_robust_se  with singleton clusters reduces exactly to the HC0
                          sandwich times the CR1 finite-sample factor.

Purely additive — no estimator numerics changed (campaign red line).

NB (Tier D finding): ``sp.granger_causality`` had a placeholder Wald variance
and was off by orders of magnitude; the fix is guarded separately in
``tests/test_tierD_p2_timeseries_analytic.py``.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# sp.fisher_exact — Fisher randomization (permutation) test
# ---------------------------------------------------------------------------
class TestFisherRandomizationAnalytic:

    def test_observed_statistic_is_difference_in_means(self):
        rng = np.random.default_rng(0)
        n = 200
        t = rng.integers(0, 2, n)
        y = 2.0 * t + rng.normal(0, 1, n)
        df = pd.DataFrame({"y": y, "t": t})
        res = sp.fisher_exact(df, y="y", treatment="t", n_perm=2000, seed=1)
        diff = y[t == 1].mean() - y[t == 0].mean()
        assert res.statistic == pytest.approx(diff, abs=1e-9)

    def test_strong_effect_rejects_sharp_null(self):
        rng = np.random.default_rng(0)
        n = 200
        t = rng.integers(0, 2, n)
        y = 2.0 * t + rng.normal(0, 1, n)  # large constant effect
        res = sp.fisher_exact(
            pd.DataFrame({"y": y, "t": t}), y="y", treatment="t", n_perm=2000, seed=1
        )
        assert res.p_value < 0.01

    def test_no_effect_does_not_reject(self):
        rng = np.random.default_rng(0)
        n = 200
        t = rng.integers(0, 2, n)
        y = rng.normal(0, 1, n)  # treatment unrelated to outcome
        res = sp.fisher_exact(
            pd.DataFrame({"y": y, "t": t}), y="y", treatment="t", n_perm=2000, seed=1
        )
        assert res.p_value > 0.10


# ---------------------------------------------------------------------------
# sp.cluster_robust_se — clustered sandwich variance
# ---------------------------------------------------------------------------
class TestClusterRobustSEAnalytic:

    def test_singleton_clusters_equal_hc0_with_cr1_factor(self):
        # One observation per cluster -> the clustered meat equals the HC0 meat,
        # so the SE equals the HC0 sandwich SE times the CR1 finite-sample
        # factor sqrt( G/(G-1) * (N-1)/(N-K) ).
        rng = np.random.default_rng(0)
        n = 300
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        beta = np.array([1.0, 2.0])
        y = X @ beta + rng.normal(0, 1, n)
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        se = np.asarray(sp.cluster_robust_se(X, resid, np.arange(n)))

        XtX_inv = np.linalg.inv(X.T @ X)
        meat = (X * resid[:, None]).T @ (X * resid[:, None])
        hc0 = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
        G, K = n, X.shape[1]
        factor = np.sqrt((G / (G - 1)) * ((n - 1) / (n - K)))
        np.testing.assert_allclose(se, hc0 * factor, rtol=1e-6)

    def test_se_positive_and_two_dimensional(self):
        rng = np.random.default_rng(1)
        n = 400
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n), rng.normal(0, 1, n)])
        beta = np.array([0.5, 1.0, -1.0])
        y = X @ beta + rng.normal(0, 1, n)
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        clusters = rng.integers(0, 20, n)  # 20 clusters
        se = np.asarray(sp.cluster_robust_se(X, resid, clusters))
        assert se.shape[0] == 3
        assert np.all(se > 0)
