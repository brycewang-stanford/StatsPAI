"""Tier D P2 known-truth anchors — cointegration, CUSUM, IRF.

Part of the P1/P2 "Tier D analytic special-cases" campaign. These entry
points were graded ``weak`` (boolean/shape/not-None asserts only). Each
test below anchors to a truth that is known a priori from the DGP or from
a closed-form identity:

    sp.engle_granger  cointegrated pair rejects no-cointegration & the
                      step-1 OLS recovers the true cointegrating slope;
                      two independent random walks fail to reject.
    sp.johansen       two series sharing one stochastic trend imply
                      cointegration rank r = 1; independent walks -> r = 0.
    sp.cusum_test     a clear mean shift is detected (rejects stability).
    sp.irf            non-orthogonal IRF of an AR(1)/VAR(1) equals
                      (A^h)[resp, imp] exactly; for a scalar AR(1) that is
                      phi^h, and the orthogonal own-shock IRF is
                      phi^h * sqrt(sigma_u).

Purely additive — no estimator numerics changed (campaign red line).

FIXED SINCE (cusum_test): these tests were quarantined in June 2026 with the
stability direction marked xfail, because ``sp.cusum_test`` compared
max|CUSUM| against a *flat* constant (1.358) instead of the
Brown-Durbin-Evans boundary ``a*(sqrt(T) + 2t/sqrt(T))`` — empirical size
0.31-0.35 against a nominal 0.05, not shrinking with n. ``main`` has since
adopted the BDE boundary, so the xfail now passes and is kept as a plain
size anchor.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# sp.engle_granger — two-step cointegration test
# ---------------------------------------------------------------------------
class TestEngleGrangerAnalytic:
    @staticmethod
    def _cointegrated(seed=42, T=600, slope=1.5, intercept=2.0):
        # y1 is a random walk (I(1)); y2 = intercept + slope*y1 + I(0) noise.
        # By construction y2 and y1 are cointegrated with slope `slope`.
        rng = np.random.default_rng(seed)
        y1 = np.cumsum(rng.normal(size=T))
        y2 = intercept + slope * y1 + rng.normal(scale=0.7, size=T)
        return pd.DataFrame({"y2": y2, "y1": y1})

    @staticmethod
    def _independent_walks(seed=99, T=600):
        rng = np.random.default_rng(seed)
        a = np.cumsum(rng.normal(size=T))
        b = np.cumsum(rng.normal(size=T))
        return pd.DataFrame({"a": a, "b": b})

    def test_cointegrated_pair_rejects_no_cointegration(self):
        df = self._cointegrated()
        r = sp.engle_granger(df, variables=["y2", "y1"])
        # residual ADF stat must fall below the 5% Engle-Granger CV.
        assert r.test_stats < r.critical_values[1]
        assert r.rank == 1

    def test_recovers_cointegrating_slope(self):
        # Step-1 OLS coefficient on y1 is the cointegrating slope (=1.5).
        # eigenvectors holds beta = [intercept, slope].
        df = self._cointegrated(slope=1.5, intercept=2.0)
        r = sp.engle_granger(df, variables=["y2", "y1"])
        beta = np.asarray(r.eigenvectors)
        assert beta[1] == pytest.approx(1.5, abs=0.05)
        assert beta[0] == pytest.approx(2.0, abs=0.5)

    def test_independent_walks_fail_to_reject(self):
        df = self._independent_walks()
        r = sp.engle_granger(df, variables=["a", "b"])
        # spurious-regression residuals stay I(1): ADF stat above the CV.
        assert r.test_stats > r.critical_values[1]
        assert r.rank == 0


# ---------------------------------------------------------------------------
# sp.johansen — cointegration rank test
# ---------------------------------------------------------------------------
class TestJohansenAnalytic:
    @staticmethod
    def _one_common_trend(seed=7, T=600):
        # Two series driven by a single I(1) stochastic trend -> rank 1
        # (k - r = 1 common trend, so exactly one cointegrating relation).
        rng = np.random.default_rng(seed)
        trend = np.cumsum(rng.normal(size=T))
        y1 = trend + rng.normal(scale=0.6, size=T)
        y2 = 3.0 + 2.0 * trend + rng.normal(scale=0.6, size=T)
        return pd.DataFrame({"y1": y1, "y2": y2})

    @staticmethod
    def _two_independent_trends(seed=8, T=600):
        rng = np.random.default_rng(seed)
        a = np.cumsum(rng.normal(size=T))
        b = np.cumsum(rng.normal(size=T))
        return pd.DataFrame({"a": a, "b": b})

    def test_shared_trend_implies_rank_one_trace(self):
        df = self._one_common_trend()
        r = sp.johansen(df, variables=["y1", "y2"], lags=1, test="trace")
        # r <= 0 strongly rejected; r <= 1 not rejected -> rank 1.
        assert r.test_stats[0] > r.critical_values[0]
        assert r.test_stats[1] < r.critical_values[1]
        assert r.rank == 1

    def test_shared_trend_implies_rank_one_maxeig(self):
        df = self._one_common_trend()
        r = sp.johansen(df, variables=["y1", "y2"], lags=1, test="maxeig")
        assert r.test_stats[0] > r.critical_values[0]
        assert r.test_stats[1] < r.critical_values[1]
        assert r.rank == 1

    def test_independent_trends_imply_rank_zero(self):
        df = self._two_independent_trends()
        r = sp.johansen(df, variables=["a", "b"], lags=1, test="trace")
        # no cointegration: even the r <= 0 hypothesis is not rejected.
        assert r.test_stats[0] < r.critical_values[0]
        assert r.rank == 0


# ---------------------------------------------------------------------------
# sp.cusum_test — parameter-stability test
# ---------------------------------------------------------------------------
class TestCusumAnalytic:
    @staticmethod
    def _mean_shift(seed=3, n=400, jump=8.0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        y = 1.0 + 2.0 * x + rng.normal(scale=1.0, size=n)
        y[n // 2 :] += jump  # intercept jumps at the midpoint
        return pd.DataFrame({"y": y, "x": x})

    @staticmethod
    def _stable(seed=3, n=400):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        y = 1.0 + 2.0 * x + rng.normal(scale=1.0, size=n)
        return pd.DataFrame({"y": y, "x": x})

    def test_mean_shift_is_detected(self):
        # A large intercept break must push the CUSUM path across the boundary.
        # (power: 50/50 replications reject; the break dominates noise.)
        #
        # ``critical_value`` is the Brown-Durbin-Evans crossing boundary, which
        # is *linear in the recursion index* — one value per residual, not a
        # scalar. This assertion used to read ``max_cusum > critical_value``,
        # which was written against an earlier scalar-threshold implementation
        # and now raises "truth value of an array is ambiguous". The boundary
        # narrows at the start and widens at the end, so comparing the path
        # maximum against a single number is not the test anyway: the question
        # is whether the path crosses *anywhere*.
        df = self._mean_shift()
        r = sp.cusum_test(df, y="y", x=["x"])
        assert np.any(np.abs(r["cusum"]) > r["critical_value"])
        assert r["reject"] is True or r["reject"] == np.True_

    def test_mean_shift_intercept_only_model(self):
        rng = np.random.default_rng(0)
        n = 400
        y = rng.normal(loc=0.0, scale=1.0, size=n)
        y[n // 2 :] += 6.0
        df = pd.DataFrame({"y": y})
        r = sp.cusum_test(df, y="y")
        assert r["reject"] is True or r["reject"] == np.True_

    def test_stable_series_does_not_reject(self):
        """Size anchor for a bug that has since been fixed.

        This was written as an ``xfail``: ``cusum_test`` compared max|CUSUM|
        against a *flat* constant (1.358) rather than the Brown-Durbin-Evans
        boundary ``a*(sqrt(T) + 2t/sqrt(T))``. Under H0 the standardised CUSUM
        behaves like a Brownian motion whose maximum routinely clears a flat
        threshold, so a stable series rejected about a third of the time —
        empirical size 0.31-0.35 over 300 replications at n=200/400/1000,
        against a nominal 0.05, and not shrinking with n.

        ``sp.cusum_test`` now uses the BDE boundary, so it passes. Kept as a
        plain assertion rather than deleted: a false-positive rate of one in
        three is the kind of defect that reappears the moment someone
        "simplifies" the boundary back to a constant.
        """
        df = self._stable()
        r = sp.cusum_test(df, y="y", x=["x"])
        assert r["reject"] is False or r["reject"] == np.False_


# ---------------------------------------------------------------------------
# sp.irf — impulse response functions (closed-form anchors)
# ---------------------------------------------------------------------------
class TestIRFAnalytic:
    @staticmethod
    def _ar1(seed=0, T=4000, phi=0.6, scale=1.0):
        rng = np.random.default_rng(seed)
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = phi * y[t - 1] + rng.normal(scale=scale)
        return pd.DataFrame({"y": y})

    @staticmethod
    def _var1(seed=11, T=6000):
        # y_t = A y_{t-1} + e_t, e_t ~ N(0, I).
        rng = np.random.default_rng(seed)
        A = np.array([[0.5, 0.1], [0.0, 0.4]])
        y = np.zeros((T, 2))
        for t in range(1, T):
            y[t] = A @ y[t - 1] + rng.normal(size=2)
        return pd.DataFrame({"y1": y[:, 0], "y2": y[:, 1]})

    def test_ar1_nonorthogonal_irf_equals_phi_to_the_h(self):
        # For a scalar AR(1) the non-orthogonal IRF at horizon h is phi^h
        # exactly (Phi_h = phi^h, P = I when orthogonal=False).
        df = self._ar1(phi=0.6)
        v = sp.var(df, variables=["y"], lags=1)
        phi_hat = float(v.coefs["y"].loc["L1.y", "coef"])
        assert phi_hat == pytest.approx(0.6, abs=0.02)  # DGP recovery
        res = sp.irf(v, periods=6, orthogonal=False)
        vals = res["irf"]["y -> y"]
        expected = np.array([phi_hat**h for h in range(7)])
        assert np.allclose(vals, expected, atol=1e-10)

    def test_ar1_orthogonal_own_shock_scales_by_sqrt_sigma(self):
        # Orthogonal own-shock IRF = phi^h * sqrt(sigma_u) for k = 1.
        df = self._ar1(phi=0.6, scale=2.0)
        v = sp.var(df, variables=["y"], lags=1)
        phi_hat = float(v.coefs["y"].loc["L1.y", "coef"])
        sig = (
            float(np.asarray(v.sigma_u)[0, 0])
            if not isinstance(v.sigma_u, pd.DataFrame)
            else float(v.sigma_u.values[0, 0])
        )
        res = sp.irf(v, periods=5, orthogonal=True)
        vals = res["irf"]["y -> y"]
        expected = np.array([phi_hat**h * np.sqrt(sig) for h in range(6)])
        assert np.allclose(vals, expected, atol=1e-10)
        assert vals[0] == pytest.approx(np.sqrt(sig), abs=1e-10)

    def test_var1_nonorthogonal_irf_is_power_of_A(self):
        # Non-orthogonal IRF imp->resp at horizon h equals (A^h)[resp, imp].
        df = self._var1()
        v = sp.var(df, variables=["y1", "y2"], lags=1)

        def coef(eq, name):
            return float(v.coefs[eq].loc[name, "coef"])

        A = np.array(
            [
                [coef("y1", "L1.y1"), coef("y1", "L1.y2")],
                [coef("y2", "L1.y1"), coef("y2", "L1.y2")],
            ]
        )
        # DGP recovery sanity check on the lead diagonal.
        assert A[0, 0] == pytest.approx(0.5, abs=0.03)
        assert A[1, 1] == pytest.approx(0.4, abs=0.03)

        res = sp.irf(v, periods=3, orthogonal=False)
        names = ["y1", "y2"]
        for h in range(4):
            Ah = np.linalg.matrix_power(A, h)
            for ii, imp in enumerate(names):
                for ri, resp in enumerate(names):
                    got = res["irf"][f"{imp} -> {resp}"][h]
                    assert got == pytest.approx(Ah[ri, ii], abs=1e-10)
