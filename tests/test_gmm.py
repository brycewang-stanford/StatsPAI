"""
Tests for Arellano-Bond / Blundell-Bond dynamic panel GMM.
"""

import warnings

import pytest
import numpy as np
import pandas as pd

from statspai.gmm import xtabond
from statspai.core.results import CausalResult


@pytest.fixture
def dynamic_panel():
    """Simulated dynamic panel: Y_it = 0.5*Y_{i,t-1} + 1.0*X_it + α_i + ε_it.

    50 units × 10 periods. True ρ = 0.5, true β_x = 1.0.
    """
    rng = np.random.default_rng(42)
    N, T = 50, 10
    rho_true = 0.5
    beta_x_true = 1.0
    alpha = rng.normal(0, 1, N)  # unit FE

    rows = []
    for i in range(N):
        y_prev = rng.normal(0, 1)
        for t in range(T):
            x = rng.normal(0, 1)
            eps = rng.normal(0, 0.5)
            y = rho_true * y_prev + beta_x_true * x + alpha[i] + eps
            rows.append({"id": i, "time": t, "y": y, "x": x})
            y_prev = y

    return pd.DataFrame(rows)


class TestArellanoBond:
    def test_basic_run(self, dynamic_panel):
        result = xtabond(dynamic_panel, y="y", x=["x"], id="id", time="time")
        assert isinstance(result, CausalResult)
        assert "Arellano-Bond" in result.method
        # Point estimate (AR coef) must be finite and inside the stationary
        # band implied by the DGP (true ρ = 0.5 < 1); SE finite and modest.
        assert np.isfinite(result.estimate)
        assert -1.0 < result.estimate < 1.0
        assert np.isfinite(result.se) and 0 < result.se < 1.0

    def test_rho_positive(self, dynamic_panel):
        """AR coefficient should be positive (true ρ = 0.5)."""
        result = xtabond(dynamic_panel, y="y", x=["x"], id="id", time="time")
        assert result.estimate > 0

    def test_rho_magnitude(self, dynamic_panel):
        """ρ̂ should be in reasonable range of 0.5."""
        result = xtabond(dynamic_panel, y="y", x=["x"], id="id", time="time")
        assert abs(result.estimate - 0.5) < 0.3

    def test_x_coefficient(self, dynamic_panel):
        """β_x should be positive and recover the planted true = 1.0."""
        result = xtabond(dynamic_panel, y="y", x=["x"], id="id", time="time")
        row = result.detail[result.detail["variable"] == "x"]
        x_coef = row["coefficient"].values[0]
        x_se = row["se"].values[0]
        assert x_coef > 0
        # Recovery band: true β_x = 1.0. xtabond is consistent here, so a
        # generous 30% relative band comfortably covers BLAS/seed noise
        # while still catching a sign flip or order-of-magnitude regression.
        assert abs(x_coef - 1.0) < 0.3
        assert np.isfinite(x_se) and x_se > 0

    def test_ar2_not_reject(self, dynamic_panel):
        """AR(2) test should not reject (DGP has AR(1) only)."""
        result = xtabond(dynamic_panel, y="y", x=["x"], id="id", time="time")
        assert result.model_info["ar2_p"] > 0.01

    def test_hansen_test(self, dynamic_panel):
        """Hansen overid stats should be present and well-formed.

        Hansen J is only a valid overid test under two-step (efficient)
        weighting, so the one-step run reports the keys (p may be NaN) and
        the two-step run must return a usable p-value in [0, 1] with a
        non-negative statistic on the correct (n_instruments - k) df. The
        AR(1)-only DGP is correctly specified, so the test should NOT reject.
        """
        mi = xtabond(dynamic_panel, y="y", x=["x"], id="id", time="time").model_info
        assert "hansen_p" in mi and "hansen_stat" in mi and "hansen_df" in mi

        mi2 = xtabond(
            dynamic_panel, y="y", x=["x"], id="id", time="time", twostep=True
        ).model_info
        assert np.isfinite(mi2["hansen_stat"]) and mi2["hansen_stat"] >= 0
        assert 0.0 <= mi2["hansen_p"] <= 1.0
        assert mi2["hansen_df"] == mi2["n_instruments"] - mi2["n_regressors"]
        # Correctly specified model: overid restrictions should not reject.
        assert mi2["hansen_p"] > 0.01

    def test_n_instruments(self, dynamic_panel):
        result = xtabond(dynamic_panel, y="y", x=["x"], id="id", time="time")
        mi = result.model_info
        # 2 regressors (L1.y + x); for identification the GMM-style design
        # must be over-identified (more instruments than parameters), and the
        # overid df reported to the user must equal that excess count.
        assert mi["n_regressors"] == 2
        assert mi["n_instruments"] > mi["n_regressors"]
        assert mi["sargan_df"] == mi["n_instruments"] - mi["n_regressors"]

    def test_twostep(self, dynamic_panel):
        """Two-step GMM should work and recover the same DGP as one-step."""
        result = xtabond(
            dynamic_panel, y="y", x=["x"], id="id", time="time", twostep=True
        )
        assert isinstance(result, CausalResult)
        assert result.model_info["twostep"] is True
        # Two-step is a re-weighting of the same moments: estimate stays in
        # the stationary band and close to the planted ρ = 0.5; SE finite.
        assert np.isfinite(result.estimate) and 0 < result.estimate < 1.0
        assert abs(result.estimate - 0.5) < 0.3
        assert np.isfinite(result.se) and result.se > 0
        # One- and two-step point estimates should agree to within a few SEs.
        onestep = xtabond(
            dynamic_panel, y="y", x=["x"], id="id", time="time", twostep=False
        )
        assert abs(result.estimate - onestep.estimate) < 0.1

    def test_system_gmm_not_implemented(self, dynamic_panel):
        """System (Blundell-Bond) GMM is gated until it has a parity ref.

        Proper system GMM needs a stacked level equation; rather than
        return an unvalidated (and previously distorted) estimate, the
        method raises loudly and points at the difference estimator.
        """
        with pytest.raises(NotImplementedError, match="system GMM"):
            xtabond(
                dynamic_panel, y="y", x=["x"], id="id", time="time", method="system"
            )

    @staticmethod
    def _parity_panel():
        """Balanced AR(1) parity DGP (matches tests/r_parity/50_xtabond.py)."""
        rng = np.random.default_rng(42)
        N, T = 100, 8
        rows, y_prev = [], np.zeros(N)
        for t in range(T):
            xx = rng.normal(0, 1, N)
            yy = 0.5 * y_prev + 0.3 * xx + rng.normal(0, 1, N)
            for i in range(N):
                rows.append({"id": i, "time": t, "y": float(yy[i]), "x": float(xx[i])})
            y_prev = yy
        return pd.DataFrame(rows)

    def test_parity_matches_stata_xtabond(self):
        """Difference GMM (one-step robust) must match Stata's xtabond to
        machine precision on the parity DGP.

        Stata `xtabond y x, lags(1) noconstant vce(robust)`:
            L1.y = 0.39117889 (se 0.04632272); x = 0.21695482 (se 0.04361645)
        """
        df = self._parity_panel()
        res = xtabond(
            df,
            y="y",
            x=["x"],
            id="id",
            time="time",
            lags=1,
            gmm_lags=(2, None),
            method="difference",
            twostep=False,
            robust=True,
        )
        d = {
            r["variable"]: (r["coefficient"], r["se"]) for _, r in res.detail.iterrows()
        }
        assert abs(d["L1.y"][0] - 0.39117889) < 1e-5
        assert abs(d["L1.y"][1] - 0.04632272) < 1e-5
        assert abs(d["x"][0] - 0.21695482) < 1e-5
        assert abs(d["x"][1] - 0.04361645) < 1e-5

    def test_parity_all_variants_vs_stata(self):
        """SEs, Sargan/Hansen, and AR tests must match Stata across the
        one-step (robust/non-robust) and two-step (conventional/Windmeijer)
        variants. Ground truth: Stata 18 `xtabond y x, lags(1) noconstant
        [vce(robust)] [twostep]`.
        """
        df = self._parity_panel()
        # (twostep, robust): seL, sex, overid (1-step Sargan / 2-step Hansen),
        #                    ar1_z, ar2_z, betaL
        cases = {
            (False, False): (
                0.05186975,
                0.04906476,
                13.22746,
                -9.8663,
                -0.66112,
                0.39117889,
            ),
            (False, True): (
                0.04632272,
                0.04361645,
                13.22746,
                -6.8961,
                -0.72073,
                0.39117889,
            ),
            (True, False): (
                0.04038861,
                0.03527729,
                12.62734,
                -6.9592,
                -0.62856,
                0.40206281,
            ),
            (True, True): (
                0.04906360,
                0.04024339,
                12.62734,
                -6.6252,
                -0.62642,
                0.40206281,
            ),
        }
        for (ts, rob), (seL, sex, overid, ar1, ar2, bL) in cases.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = xtabond(
                    df,
                    y="y",
                    x=["x"],
                    id="id",
                    time="time",
                    lags=1,
                    gmm_lags=(2, None),
                    twostep=ts,
                    robust=rob,
                )
            d = {
                r["variable"]: (r["coefficient"], r["se"])
                for _, r in res.detail.iterrows()
            }
            mi = res.model_info
            tag = f"twostep={ts}, robust={rob}"
            assert abs(d["L1.y"][0] - bL) < 1e-5, f"betaL {tag}"
            assert abs(d["L1.y"][1] - seL) < 2e-4, f"seL {tag}"
            assert abs(d["x"][1] - sex) < 2e-4, f"sex {tag}"
            got_overid = mi["hansen_stat"] if ts else mi["sargan_stat"]
            assert abs(got_overid - overid) < 1e-3, f"overid {tag}"
            # AR(1)/AR(2): exact for one-step; ~0.1%/~few% for two-step
            ar_tol = 5e-3 if not ts else 6e-2
            assert abs(mi["ar1_z"] - ar1) < abs(ar1) * ar_tol, f"ar1 {tag}"
            assert abs(mi["ar2_z"] - ar2) < max(abs(ar2) * 5e-2, 1e-2), f"ar2 {tag}"

    def test_internal_gap_warns(self):
        """Panels with interior gaps emit a parity-caveat warning."""
        df = self._parity_panel()
        df = df[~((df["id"] == 0) & (df["time"] == 3))]  # drop an interior obs
        with pytest.warns(UserWarning, match="internal time gaps"):
            xtabond(df, y="y", x=["x"], id="id", time="time")

    def test_no_exogenous(self, dynamic_panel):
        """Should work with only lagged Y (no X) and still identify ρ."""
        result = xtabond(dynamic_panel, y="y", id="id", time="time")
        assert isinstance(result, CausalResult)
        # Only the AR(1) term is estimated: exactly one row, named L1.y.
        assert list(result.detail["variable"]) == ["L1.y"]
        assert result.model_info["n_regressors"] == 1
        # Even without the strong exogenous regressor, ρ̂ stays in the
        # stationary band and recovers the planted true ρ = 0.5 (wider band
        # since dropping x inflates the SE).
        assert np.isfinite(result.estimate) and 0 < result.estimate < 1.0
        assert abs(result.estimate - 0.5) < 0.35
        assert np.isfinite(result.se) and result.se > 0

    def test_detail_table(self, dynamic_panel):
        result = xtabond(dynamic_panel, y="y", x=["x"], id="id", time="time")
        det = result.detail
        for col in ("variable", "coefficient", "se", "z", "pvalue"):
            assert col in det.columns
        # One row per regressor (L1.y + x), all numeric fields finite, SE > 0,
        # p-values are valid probabilities, and z is internally consistent
        # with coef/se. The leading row (L1.y) must equal result.estimate.
        assert set(det["variable"]) == {"L1.y", "x"}
        assert len(det) == 2
        assert np.all(np.isfinite(det["coefficient"]))
        assert np.all(det["se"] > 0)
        assert np.all((det["pvalue"] >= 0) & (det["pvalue"] <= 1))
        np.testing.assert_allclose(det["z"], det["coefficient"] / det["se"], rtol=1e-6)
        lag_row = det[det["variable"] == "L1.y"]
        assert abs(lag_row["coefficient"].values[0] - result.estimate) < 1e-12

    def test_summary(self, dynamic_panel):
        result = xtabond(dynamic_panel, y="y", x=["x"], id="id", time="time")
        s = result.summary()
        assert "Arellano-Bond" in s or "GMM" in s

    def test_cite(self, dynamic_panel):
        result = xtabond(dynamic_panel, y="y", x=["x"], id="id", time="time")
        bib = result.cite()
        assert "arellano" in bib.lower()


class TestIntegration:
    def test_import(self):
        import statspai as sp

        assert hasattr(sp, "xtabond")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
