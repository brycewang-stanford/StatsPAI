"""Tier D P2 known-truth upgrades — collinearity diagnostic & system estimators.

Part of the P1/P2 "Tier D analytic special-cases" campaign (see
``.tierd_campaign/CAMPAIGN.md``). These entry points were graded ``weak`` by
``scripts/tierd_classify.py`` (only boolean/shape/provenance asserts). Each now
anchors to a known truth:

    sp.vif    variance inflation factor VIF_j = 1 / (1 - R^2_j) exactly
    sp.sureg  SUR with identical regressors == OLS equation-by-equation (Kruskal)
    sp.jive   jackknife IV recovers the known structural coefficient

Purely additive — no estimator numerics changed (campaign red line).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# sp.vif — variance inflation factor
# ---------------------------------------------------------------------------
class TestVIFAnalytic:

    def test_vif_equals_one_over_one_minus_r2(self):
        # VIF_j is exactly 1/(1 - R^2_j), R^2_j from regressing x_j on the rest.
        rng = np.random.default_rng(0)
        n = 4000
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        x3 = 0.7 * x1 + 0.3 * x2 + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        tab = sp.vif(df, x=["x1", "x2", "x3"]).set_index("variable")
        cols = ["x1", "x2", "x3"]
        for j, name in enumerate(cols):
            others = [c for c in cols if c != name]
            X = np.column_stack([np.ones(n)] + [df[c].values for c in others])
            b, *_ = np.linalg.lstsq(X, df[name].values, rcond=None)
            resid = df[name].values - X @ b
            r2 = (
                1
                - (resid @ resid)
                / ((df[name].values - df[name].values.mean()) ** 2).sum()
            )
            assert tab.loc[name, "VIF"] == pytest.approx(1.0 / (1.0 - r2), abs=0.01)

    def test_orthogonal_covariates_have_vif_one(self):
        rng = np.random.default_rng(1)
        n = 5000
        df = pd.DataFrame({f"x{i}": rng.normal(0, 1, n) for i in range(1, 4)})
        tab = sp.vif(df, x=["x1", "x2", "x3"]).set_index("variable")
        assert np.allclose(tab["VIF"].values, 1.0, atol=0.1)


# ---------------------------------------------------------------------------
# sp.sureg — seemingly unrelated regression
# ---------------------------------------------------------------------------
class TestSURegAnalytic:

    @staticmethod
    def _system(seed=0, n=4000):
        rng = np.random.default_rng(seed)
        z1 = rng.normal(0, 1, n)
        z2 = rng.normal(0, 1, n)
        ya = 1.0 + 2.0 * z1 - 1.0 * z2 + rng.normal(0, 1, n)
        yb = 0.5 - 1.5 * z1 + 0.8 * z2 + rng.normal(0, 1, n)
        return pd.DataFrame({"ya": ya, "yb": yb, "z1": z1, "z2": z2})

    def test_recovers_true_system_coefficients(self):
        df = self._system()
        sur = sp.sureg(
            {"eqA": ("ya", ["z1", "z2"]), "eqB": ("yb", ["z1", "z2"])}, data=df
        )
        a = sur.equations["eqA"]["params"]
        b = sur.equations["eqB"]["params"]
        assert a["z1"] == pytest.approx(2.0, abs=0.1)
        assert a["z2"] == pytest.approx(-1.0, abs=0.1)
        assert b["z1"] == pytest.approx(-1.5, abs=0.1)
        assert b["z2"] == pytest.approx(0.8, abs=0.1)

    def test_identical_regressors_equals_ols(self):
        # Kruskal's theorem: when every equation has the same regressor matrix,
        # the SUR/FGLS estimator collapses to equation-by-equation OLS.
        df = self._system()
        sur = sp.sureg(
            {"eqA": ("ya", ["z1", "z2"]), "eqB": ("yb", ["z1", "z2"])}, data=df
        )
        ols_a = sp.regress("ya ~ z1 + z2", data=df)
        a = sur.equations["eqA"]["params"]
        assert a["z1"] == pytest.approx(float(ols_a.params["z1"]), rel=1e-3)
        assert a["z2"] == pytest.approx(float(ols_a.params["z2"]), rel=1e-3)


# ---------------------------------------------------------------------------
# sp.jive — jackknife instrumental variables
# ---------------------------------------------------------------------------
class TestJIVEAnalytic:

    @staticmethod
    def _iv_system(seed=0, n=4000):
        rng = np.random.default_rng(seed)
        Z = rng.normal(0, 1, (n, 3))
        endog = Z @ np.array([0.6, 0.5, 0.4]) + rng.normal(0, 1, n)
        y = 1.5 * endog + rng.normal(0, 1, n)
        return pd.DataFrame(
            {"y": y, "xe": endog, "z0": Z[:, 0], "z1": Z[:, 1], "z2": Z[:, 2]}
        )

    def test_recovers_structural_coefficient(self):
        df = self._iv_system()
        res = sp.jive(df, y="y", x_endog=["xe"], z=["z0", "z1", "z2"])
        assert float(res.params["xe"]) == pytest.approx(1.5, abs=0.1)

    def test_jive2_variant_also_recovers(self):
        df = self._iv_system(seed=3)
        res = sp.jive(df, y="y", x_endog=["xe"], z=["z0", "z1", "z2"], variant="jive2")
        assert float(res.params["xe"]) == pytest.approx(1.5, abs=0.1)
