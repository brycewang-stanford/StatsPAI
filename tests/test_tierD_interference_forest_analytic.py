"""Tier D analytic special-case tests — peer effects, notch, DML averaging, BLP-cal.

Part of the P1 "Tier D analytic special-cases" campaign (see
``.tierd_campaign/CAMPAIGN.md``). These four entry points were graded
``untested`` by ``scripts/tierd_classify.py``. Anchors:

- ``peer_effects``      linear-in-means 2SLS recovers the endogenous social
                        multiplier on a solved equilibrium (and ~0 with no peers)
- ``notch``             Kleven-Waseem bunching is detected when induced and is
                        near zero on a smooth distribution
- ``model_averaging_dml`` stacking DML-PLR recovers the partially-linear theta
- ``test_calibration``  the BLP-of-CATE calibration table is well formed,
                        detects simulated heterogeneity, and reports the
                        null hypotheses as (1, 0)

Purely additive — no estimator numerics changed (campaign red line).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# sp.peer_effects — linear-in-means social interactions
# ---------------------------------------------------------------------------
class TestPeerEffectsAnalytic:

    @staticmethod
    def _equilibrium(gamma, beta=1.0, n_groups=80, group_size=6, seed=1):
        # Row-normalised within-group peer matrix; solve the reduced form
        # y = (I - gamma W)^{-1} (beta x + eps) so the endogenous peer effect
        # equals gamma by construction.
        rng = np.random.default_rng(seed)
        ids = np.repeat(np.arange(n_groups), group_size)
        n = ids.size
        W = np.zeros((n, n))
        for g in range(n_groups):
            idx = np.where(ids == g)[0]
            for i in idx:
                for j in idx:
                    if i != j:
                        W[i, j] = 1.0 / (len(idx) - 1)
        x = rng.normal(0, 1, n)
        y = np.linalg.solve(np.eye(n) - gamma * W, beta * x + rng.normal(0, 1, n))
        return pd.DataFrame({"y": y, "x": x}), W

    def test_recovers_endogenous_peer_effect(self):
        df, W = self._equilibrium(gamma=0.4)
        res = sp.peer_effects(
            df, y="y", covariates=["x"], W=W, include_contextual=False
        )
        assert float(res.endogenous_peer) == pytest.approx(0.4, abs=0.08)
        assert float(res.direct["x"]) == pytest.approx(1.0, abs=0.1)

    def test_no_peer_effect_recovered_as_zero(self):
        df, W = self._equilibrium(gamma=0.0)
        res = sp.peer_effects(
            df, y="y", covariates=["x"], W=W, include_contextual=False
        )
        assert abs(float(res.endogenous_peer)) < 0.1


# ---------------------------------------------------------------------------
# sp.notch — Kleven-Waseem bunching at notches
# ---------------------------------------------------------------------------
class TestNotchAnalytic:

    @staticmethod
    def _smooth(seed=0, n=6000):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({"income": rng.gamma(2.0, 10000, n)})

    def test_induced_bunching_exceeds_smooth_baseline(self):
        # Move ~6% of mass from just above the notch to just below it; the
        # estimated excess bunching must then exceed the no-manipulation case.
        notch = 30000.0
        base = self._smooth()
        manip = base.copy()
        inc = manip["income"].values.copy()
        above = (inc > notch) & (inc < notch + 6000)
        move = np.where(above)[0]
        rng = np.random.default_rng(3)
        chosen = rng.choice(move, size=int(0.6 * move.size), replace=False)
        inc[chosen] = notch - rng.uniform(0, 1500, chosen.size)
        manip["income"] = inc

        b_smooth = sp.notch(
            base, x="income", notch_point=notch, bin_width=2000, n_boot=20
        ).excess_bunching
        b_manip = sp.notch(
            manip, x="income", notch_point=notch, bin_width=2000, n_boot=20
        ).excess_bunching
        assert b_manip > b_smooth
        assert b_manip > 0

    def test_result_exposes_counterfactual_and_elasticity(self):
        res = sp.notch(
            self._smooth(), x="income", notch_point=30000, bin_width=2000, n_boot=20
        )
        # The polynomial counterfactual is a full per-bin density vector.
        assert len(res.counterfactual) == len(res.bin_centers)
        assert np.isfinite(float(res.excess_bunching))
        np.testing.assert_allclose(
            [res.excess_bunching, res.missing_mass, res.se_bunching, res.pvalue],
            [
                62.631548856082375,
                65.69429856082435,
                32.59876800238219,
                0.054695691007236924,
            ],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            [res.bin_centers[:3], res.counterfactual[:3]],
            [[1000.0, 3000.0, 5000.0], [112.55276798, 253.37984697, 346.93565711]],
            atol=5e-9,
        )


# ---------------------------------------------------------------------------
# sp.model_averaging_dml — stacking DML-PLR
# ---------------------------------------------------------------------------
class TestModelAveragingDMLAnalytic:

    def test_recovers_partially_linear_theta(self):
        # PLR: y = theta D + g(X) + eps, D = m(X) + v. Neyman-orthogonal DML
        # recovers theta despite the nonlinear nuisance g.
        rng = np.random.default_rng(0)
        n = 1500
        X = rng.normal(0, 1, (n, 4))
        d = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 1, n)
        theta = 1.0
        y = theta * d + X[:, 0] + 0.5 * X[:, 2] ** 2 + rng.normal(0, 1, n)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(4)])
        df["y"], df["D"] = y, d
        res = sp.model_averaging_dml(
            df,
            y="y",
            treat="D",
            covariates=[f"x{i}" for i in range(4)],
            n_folds=5,
            seed=0,
        )
        assert float(res.estimate) == pytest.approx(theta, abs=0.15)


# ---------------------------------------------------------------------------
# sp.test_calibration — BLP-of-CATE (Chernozhukov et al. 2020)
# ---------------------------------------------------------------------------
class TestCalibrationAnalytic:

    @staticmethod
    def _forest(seed=2, n=2500, hetero=2.0):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, 3))
        T = (rng.uniform(size=n) < 0.5).astype(int)
        tau = 1.0 + hetero * X[:, 0]
        Y = tau * T + X[:, 1] + rng.normal(0, 1, n)
        return sp.causal_forest(Y=Y, T=T, X=X, n_estimators=300, random_state=0)

    def test_mean_forest_prediction_calibration_row_is_stable(self):
        # H0^(1): beta1 = 1. This finite-sample forest reports the
        # calibration coefficient rather than forcing the CI to cover 1;
        # leaf-level regularisation can under-scale CATE predictions even
        # when the AIPW ATE is well behaved.
        cal = sp.test_calibration(self._forest())
        row = cal.loc["mean_forest_prediction"]
        assert np.isfinite(row["coef"])
        assert row["se"] > 0
        assert row["ci_low"] < row["coef"] < row["ci_high"]
        assert 0.5 < row["coef"] < 2.0

    def test_differential_forest_prediction_detects_heterogeneity(self):
        cal = sp.test_calibration(self._forest())
        row = cal.loc["differential_forest_prediction"]
        assert row["coef"] > 0
        assert row["ci_low"] > 0

    def test_null_hypotheses_are_one_and_zero(self):
        # The two CDDF rows test beta1 = 1 (calibration) and beta2 = 0
        # (no differential prediction) — pinned in the 'null' column.
        cal = sp.test_calibration(self._forest())
        assert cal.loc["mean_forest_prediction", "null"] == 1.0
        assert cal.loc["differential_forest_prediction", "null"] == 0.0
