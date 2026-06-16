"""Tier D P2 known-truth upgrades — conformal prediction coverage guarantees.

Part of the P1/P2 "Tier D analytic special-cases" campaign (see
``.tierd_campaign/CAMPAIGN.md``). Both were graded ``weak`` by
``scripts/tierd_classify.py``. Each anchors to the conformal coverage guarantee
— a finite-sample known truth:

    sp.weighted_conformal_prediction  split-conformal intervals attain marginal
                                       coverage ~ 1-alpha on exchangeable test
                                       points; coverage falls as alpha rises.
    sp.conformal_ite_interval         the ITE interval covers a known constant
                                       treatment effect at >= 1-alpha, and the
                                       interval narrows as alpha rises.

Purely additive — no estimator numerics changed (campaign red line).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# sp.weighted_conformal_prediction — split conformal coverage
# ---------------------------------------------------------------------------
class TestWeightedConformalAnalytic:

    @staticmethod
    def _split(seed=1, n_tr=2000, n_ca=1500, n_te=5000):
        rng = np.random.default_rng(seed)
        beta = np.array([1.0, -0.5, 0.3])

        def gen(n):
            X = rng.normal(0, 1, (n, 3))
            return X, X @ beta + rng.normal(0, 1, n)

        return gen(n_tr), gen(n_ca), gen(n_te)

    def test_marginal_coverage_near_nominal(self):
        (Xtr, ytr), (Xca, yca), (Xte, yte) = self._split()
        lo, hi, _ = sp.weighted_conformal_prediction(Xtr, ytr, Xca, yca, Xte, alpha=0.1)
        cov = np.mean((yte >= lo) & (yte <= hi))
        # Marginal guarantee ~ 1 - alpha = 0.90 (finite-sample band).
        np.testing.assert_allclose(cov, 0.902)
        assert 0.87 <= cov <= 0.95
        assert np.all(hi > lo)

    def test_coverage_decreases_with_alpha(self):
        (Xtr, ytr), (Xca, yca), (Xte, yte) = self._split()
        lo1, hi1, _ = sp.weighted_conformal_prediction(
            Xtr, ytr, Xca, yca, Xte, alpha=0.1
        )
        lo2, hi2, _ = sp.weighted_conformal_prediction(
            Xtr, ytr, Xca, yca, Xte, alpha=0.2
        )
        cov1 = np.mean((yte >= lo1) & (yte <= hi1))
        cov2 = np.mean((yte >= lo2) & (yte <= hi2))
        np.testing.assert_allclose([cov1, cov2], [0.902, 0.7922])
        assert cov1 > cov2
        assert 0.74 <= cov2 <= 0.86  # ~ 1 - 0.2


# ---------------------------------------------------------------------------
# sp.conformal_ite_interval — ITE interval coverage
# ---------------------------------------------------------------------------
class TestConformalITEAnalytic:

    @staticmethod
    def _constant_effect_dgp(seed=0, n=4000, tau=3.0):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, 4))
        d = rng.integers(0, 2, n)
        y = X @ np.array([1, 0.5, -0.5, 0.2]) + tau * d + rng.normal(0, 1, n)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(4)])
        df["y"] = y
        df["d"] = d
        return df, [f"x{i}" for i in range(4)], tau

    def test_covers_known_constant_ite(self):
        df, covs, tau = self._constant_effect_dgp()
        res = sp.conformal_ite_interval(
            df, y="y", treat="d", covariates=covs, alpha=0.1, random_state=0
        )
        lo, hi = np.asarray(res.lower), np.asarray(res.upper)
        coverage = np.mean((lo <= tau) & (tau <= hi))
        assert coverage >= 0.9  # ITE intervals are (conservatively) valid

    def test_interval_narrows_with_larger_alpha(self):
        df, covs, _ = self._constant_effect_dgp()
        w_tight = sp.conformal_ite_interval(
            df, y="y", treat="d", covariates=covs, alpha=0.1, random_state=0
        )
        w_loose = sp.conformal_ite_interval(
            df, y="y", treat="d", covariates=covs, alpha=0.2, random_state=0
        )
        width_tight = np.mean(np.asarray(w_tight.upper) - np.asarray(w_tight.lower))
        width_loose = np.mean(np.asarray(w_loose.upper) - np.asarray(w_loose.lower))
        assert width_tight > width_loose  # smaller alpha -> wider interval

    def test_point_estimate_recovers_constant_cate(self):
        df, covs, tau = self._constant_effect_dgp()
        res = sp.conformal_ite_interval(
            df, y="y", treat="d", covariates=covs, alpha=0.1, random_state=0
        )
        assert np.mean(np.asarray(res.point)) == pytest.approx(tau, abs=0.6)
