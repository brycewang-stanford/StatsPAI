"""Tier D P2 known-truth anchors — limited-dependent & system estimators.

Part of the P1/P2 "Tier D analytic special-cases" campaign. These entry
points were graded ``weak`` by ``scripts/tierd_classify.py`` (only
boolean/shape/not-None asserts). Each test below simulates from the
estimator's OWN data-generating process with KNOWN parameters and recovers
them within a justified tolerance:

    sp.betareg    Beta-distributed y in (0,1), mean=logistic(Xb), known phi
    sp.fracreg    fractional y in [0,1] with E[y|X]=logistic(Xb) (quasi-MLE)
    sp.biprobit   bivariate probit, known per-equation coefs and known rho
    sp.etregress  endogenous treatment (Heckman control-function), known delta
    sp.sqreg      location-shift DGP -> common slope across tau, ordered consts
    sp.three_sls  known simultaneous system + 3SLS==2SLS identity (uncorr eqs)
    sp.lasso_iv   sparse-instrument IV, known first stage, known structural coef

MLE estimators carry finite-sample bias, so tolerances are sized to N and
annotated. Purely additive — no estimator numerics changed (campaign red
line).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# sp.betareg — Beta regression (Ferrari & Cribari-Neto 2004)
# ---------------------------------------------------------------------------
class TestBetaregAnalytic:
    @staticmethod
    def _dgp(seed=0, n=5000, phi=20.0):
        rng = np.random.default_rng(seed)
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        b0, b1, b2 = 0.3, 1.0, -0.7
        mu = 1.0 / (1.0 + np.exp(-(b0 + b1 * x1 + b2 * x2)))
        y = rng.beta(mu * phi, (1.0 - mu) * phi)
        y = np.clip(y, 1e-6, 1 - 1e-6)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        return df, (b0, b1, b2), phi

    def test_recovers_mean_coefficients(self):
        df, (b0, b1, b2), _ = self._dgp()
        r = sp.betareg(df, y="y", x=["x1", "x2"])
        # MLE at N=5000: tolerance ~0.05 on the index coefficients.
        assert r.params["_cons"] == pytest.approx(b0, abs=0.05)
        assert r.params["x1"] == pytest.approx(b1, abs=0.05)
        assert r.params["x2"] == pytest.approx(b2, abs=0.05)

    def test_recovers_precision_phi(self):
        # Precision equation has an intercept-only term log(phi) = _cons_phi.
        df, _, phi = self._dgp()
        r = sp.betareg(df, y="y", x=["x1", "x2"])
        phi_hat = float(np.exp(r.params["_cons_phi"]))
        assert phi_hat == pytest.approx(phi, rel=0.08)


# ---------------------------------------------------------------------------
# sp.fracreg — fractional response model (Papke & Wooldridge 1996)
# ---------------------------------------------------------------------------
class TestFracregAnalytic:
    @staticmethod
    def _dgp(seed=1, n=8000):
        # Any DGP with the correct conditional mean is recovered by the
        # quasi-MLE; here y is Beta-distributed with mean logistic(Xb).
        rng = np.random.default_rng(seed)
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        b0, b1, b2 = 0.2, 0.9, -0.6
        mu = 1.0 / (1.0 + np.exp(-(b0 + b1 * x1 + b2 * x2)))
        phi = 8.0
        y = rng.beta(mu * phi, (1.0 - mu) * phi)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        return df, (b0, b1, b2)

    def test_recovers_index_coefficients(self):
        df, (b0, b1, b2) = self._dgp()
        r = sp.fracreg(df, y="y", x=["x1", "x2"])
        # Quasi-MLE consistent for the index coefs; N=8000 -> abs~0.05.
        assert r.params["_cons"] == pytest.approx(b0, abs=0.05)
        assert r.params["x1"] == pytest.approx(b1, abs=0.05)
        assert r.params["x2"] == pytest.approx(b2, abs=0.05)

    def test_probit_link_also_recovers_sign_and_scale(self):
        # Generate from a probit-mean DGP and recover via link='probit'.
        rng = np.random.default_rng(11)
        n = 8000
        from scipy import stats

        x1 = rng.normal(0, 1, n)
        b0, b1 = 0.1, 0.8
        mu = stats.norm.cdf(b0 + b1 * x1)
        y = rng.beta(mu * 8.0, (1.0 - mu) * 8.0)
        df = pd.DataFrame({"y": y, "x1": x1})
        r = sp.fracreg(df, y="y", x=["x1"], link="probit")
        assert r.params["_cons"] == pytest.approx(b0, abs=0.06)
        assert r.params["x1"] == pytest.approx(b1, abs=0.06)


# ---------------------------------------------------------------------------
# sp.biprobit — bivariate probit with known correlation rho
# ---------------------------------------------------------------------------
class TestBiprobitAnalytic:
    @staticmethod
    def _dgp(seed=42, n=2500, rho=0.5):
        rng = np.random.default_rng(seed)
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        a0, a1 = 0.3, 0.8
        c0, c1 = -0.2, 0.6
        cov = [[1.0, rho], [rho, 1.0]]
        e = rng.multivariate_normal([0.0, 0.0], cov, size=n)
        y1 = ((a0 + a1 * x1 + e[:, 0]) > 0).astype(int)
        y2 = ((c0 + c1 * x2 + e[:, 1]) > 0).astype(int)
        df = pd.DataFrame({"y1": y1, "y2": y2, "x1": x1, "x2": x2})
        return df, (a0, a1, c0, c1), rho

    def test_recovers_both_equations_and_rho(self):
        df, (a0, a1, c0, c1), rho = self._dgp()
        r = sp.biprobit(df, y1="y1", y2="y2", x1=["x1"], x2=["x2"])
        # MLE at N=2500 on binary outcomes -> abs~0.08 on coefs, ~0.07 rho.
        assert r.params["eq1._cons"] == pytest.approx(a0, abs=0.08)
        assert r.params["eq1.x1"] == pytest.approx(a1, abs=0.08)
        assert r.params["eq2._cons"] == pytest.approx(c0, abs=0.08)
        assert r.params["eq2.x2"] == pytest.approx(c1, abs=0.08)
        assert r.params["rho"] == pytest.approx(rho, abs=0.07)
        assert float(r.model_info["rho"]) == pytest.approx(rho, abs=0.07)


# ---------------------------------------------------------------------------
# sp.etregress — endogenous treatment (Heckman control function)
# ---------------------------------------------------------------------------
class TestEtregressAnalytic:
    @staticmethod
    def _dgp(seed=3, n=8000, delta=2.0, rho=0.6):
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 1, n)
        zinst = rng.normal(0, 1, n)  # excluded instrument for selection
        cov = [[1.0, rho], [rho, 1.0]]
        ev = rng.multivariate_normal([0.0, 0.0], cov, size=n)
        eps, u = ev[:, 0], ev[:, 1]
        dstar = 0.0 + 0.7 * x + 0.9 * zinst + u
        d = (dstar > 0).astype(int)
        b0, b1 = 1.0, 0.5
        y = b0 + b1 * x + delta * d + eps
        df = pd.DataFrame({"y": y, "x": x, "D": d, "zinst": zinst})
        return df, (b0, b1, delta), rho

    def test_recovers_treatment_effect(self):
        df, (b0, b1, delta), _ = self._dgp()
        r = sp.etregress(df, y="y", x=["x"], treatment="D", z=["x", "zinst"])
        # Control-function correction recovers delta at N=8000.
        assert r.params["D"] == pytest.approx(delta, abs=0.1)
        assert float(r.model_info["treatment_effect"]) == pytest.approx(delta, abs=0.1)
        assert r.params["_cons"] == pytest.approx(b0, abs=0.1)
        assert r.params["x"] == pytest.approx(b1, abs=0.1)

    def test_corrects_positive_selection_bias(self):
        # rho>0 biases naive OLS of y on D upward; etregress must undo it.
        df, (b0, b1, delta), rho = self._dgp()
        r = sp.etregress(df, y="y", x=["x"], treatment="D", z=["x", "zinst"])
        n = len(df)
        Xn = np.column_stack([np.ones(n), df["x"].values, df["D"].values])
        naive = np.linalg.lstsq(Xn, df["y"].values, rcond=None)[0][2]
        assert naive > delta + 0.3  # naive OLS is upward-biased
        assert abs(r.params["D"] - delta) < abs(naive - delta)
        # selection_corr is a control-function proxy, positive like rho.
        assert float(r.diagnostics["selection_corr"]) > 0.3


# ---------------------------------------------------------------------------
# sp.sqreg — simultaneous quantile regression
# ---------------------------------------------------------------------------
class TestSqregAnalytic:
    def test_location_shift_shares_slope_across_quantiles(self):
        # y = Xb + eps with eps independent of X -> every conditional
        # quantile has the SAME slope b; only the intercept shifts by the
        # quantile of eps.
        rng = np.random.default_rng(0)
        n = 6000
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        b0, b1, b2 = 1.0, 2.0, -1.0
        eps = rng.normal(0, 1, n)  # independent of X
        y = b0 + b1 * x1 + b2 * x2 + eps
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        taus = [0.25, 0.5, 0.75]
        tab = sp.sqreg(df, y="y", x=["x1", "x2"], quantiles=taus).set_index("variable")
        for tau in taus:
            assert tab.loc["x1", f"Q({tau})"] == pytest.approx(b1, abs=0.08)
            assert tab.loc["x2", f"Q({tau})"] == pytest.approx(b2, abs=0.08)
        # Intercept ordering matches the standard-normal error quantiles.
        from scipy import stats

        consts = [tab.loc["const", f"Q({t})"] for t in taus]
        assert consts[0] < consts[1] < consts[2]
        for tau, c in zip(taus, consts):
            expected = b0 + stats.norm.ppf(tau)  # known eps quantile
            assert c == pytest.approx(expected, abs=0.1)


# ---------------------------------------------------------------------------
# sp.three_sls — three-stage least squares on a known system
# ---------------------------------------------------------------------------
class TestThreeSLSAnalytic:
    def test_recovers_simultaneous_system_coefficients(self):
        # Genuine simultaneity: q and p determined jointly by demand/supply.
        rng = np.random.default_rng(0)
        n = 4000
        income = rng.normal(0, 1, n)
        cost = rng.normal(0, 1, n)
        e1 = rng.normal(0, 1, n)
        e2 = rng.normal(0, 1, n)
        a0, a1, a2 = 1.0, -0.8, 0.5  # demand: q = a0 + a1 p + a2 income + e1
        b0, b1, b2 = 0.5, 0.6, 0.4  # supply: q = b0 + b1 p + b2 cost  + e2
        p = ((b0 - a0) + b2 * cost - a2 * income + (e2 - e1)) / (a1 - b1)
        q = a0 + a1 * p + a2 * income + e1
        df = pd.DataFrame({"q": q, "p": p, "income": income, "cost": cost})
        res = sp.three_sls(
            equations={
                "demand": ("q", ["income"], ["p"]),
                "supply": ("q", ["cost"], ["p"]),
            },
            data=df,
            instruments=["income", "cost"],
        )
        dem = res.equations["demand"]["params"]
        sup = res.equations["supply"]["params"]
        assert dem["p"] == pytest.approx(a1, abs=0.1)
        assert dem["income"] == pytest.approx(a2, abs=0.1)
        assert sup["p"] == pytest.approx(b1, abs=0.1)
        assert sup["cost"] == pytest.approx(b2, abs=0.1)

    def test_equals_2sls_when_cross_equation_errors_uncorrelated(self):
        # Identity: with diagonal cross-equation error covariance, 3SLS
        # collapses to equation-by-equation 2SLS.
        rng = np.random.default_rng(7)
        n = 4000
        z1 = rng.normal(0, 1, n)
        z2 = rng.normal(0, 1, n)
        w = rng.normal(0, 1, n)
        e1 = rng.normal(0, 1, n)
        e2 = rng.normal(0, 1, n)  # independent of e1 across equations
        xe1 = 0.8 * z1 + 0.5 * w + rng.normal(0, 1, n) + 0.4 * e1
        xe2 = 0.7 * z2 + 0.6 * w + rng.normal(0, 1, n) + 0.4 * e2
        y1 = 1.0 + 1.5 * xe1 + e1
        y2 = 0.5 - 0.9 * xe2 + e2
        df = pd.DataFrame(
            {
                "y1": y1,
                "y2": y2,
                "xe1": xe1,
                "xe2": xe2,
                "z1": z1,
                "z2": z2,
                "w": w,
            }
        )
        res = sp.three_sls(
            equations={
                "eq1": ("y1", [], ["xe1"]),
                "eq2": ("y2", [], ["xe2"]),
            },
            data=df,
            instruments=["z1", "z2", "w"],
        )
        iv1 = sp.ivreg("y1 ~ 1 + (xe1 ~ z1 + z2 + w)", data=df)
        iv2 = sp.ivreg("y2 ~ 1 + (xe2 ~ z1 + z2 + w)", data=df)
        assert res.equations["eq1"]["params"]["xe1"] == pytest.approx(
            float(iv1.params["xe1"]), rel=5e-3
        )
        assert res.equations["eq2"]["params"]["xe2"] == pytest.approx(
            float(iv2.params["xe2"]), rel=5e-3
        )
        # And both anchor to the known structural truth.
        assert res.equations["eq1"]["params"]["xe1"] == pytest.approx(1.5, abs=0.1)
        assert res.equations["eq2"]["params"]["xe2"] == pytest.approx(-0.9, abs=0.1)


# ---------------------------------------------------------------------------
# sp.lasso_iv — LASSO-selected instruments (Belloni et al. 2012)
# ---------------------------------------------------------------------------
class TestLassoIVAnalytic:
    @staticmethod
    def _dgp(seed=0, n=3000, p_inst=20, struct=1.5):
        rng = np.random.default_rng(seed)
        Z = rng.normal(0, 1, (n, p_inst))
        pi = np.zeros(p_inst)
        pi[:3] = [0.8, 0.7, 0.6]  # only first 3 instruments relevant
        u = rng.normal(0, 1, n)  # induces endogeneity
        endog = Z @ pi + u + 0.5 * rng.normal(0, 1, n)
        y = struct * endog + u + rng.normal(0, 1, n)
        cols = {f"z{i}": Z[:, i] for i in range(p_inst)}
        df = pd.DataFrame({"y": y, "xe": endog, **cols})
        return df, [f"z{i}" for i in range(p_inst)], struct

    def test_recovers_structural_coefficient_with_sparse_instruments(self):
        df, zcols, struct = self._dgp()
        r = sp.lasso_iv(df, y="y", x_endog=["xe"], z=zcols)
        assert float(r.params["xe"]) == pytest.approx(struct, abs=0.1)
        # LASSO keeps a small, relevant subset of the 20 candidates.
        assert r.diagnostics["N instruments"] <= 6
        assert r.diagnostics["First-stage F (xe)"] > 50  # strong first stage
