"""
Tests for Panel regression module (wrapping linearmodels).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import EconometricResults
from statspai.panel import PanelRegression, panel


@pytest.fixture
def panel_data():
    """
    Simulated panel: 50 entities, 10 periods.
    DGP: y_it = alpha_i + 2*x1_it + 3*x2_it + eps_it
    """
    rng = np.random.default_rng(42)
    n_entities = 50
    n_periods = 10

    records = []
    for i in range(n_entities):
        alpha_i = rng.normal(5, 2)  # entity fixed effect
        for t in range(1, n_periods + 1):
            x1 = rng.normal(0, 1)
            x2 = rng.normal(0, 1)
            eps = rng.normal(0, 0.5)
            y = alpha_i + 2 * x1 + 3 * x2 + eps
            records.append(
                {
                    "entity": f"e{i}",
                    "time": t,
                    "y": y,
                    "x1": x1,
                    "x2": x2,
                }
            )

    return pd.DataFrame(records)


class TestFixedEffects:
    def test_fe_basic(self, panel_data):
        """FE should recover slopes ≈ 2, 3 (wipes out alpha_i)."""
        result = panel(
            panel_data, "y ~ x1 + x2", entity="entity", time="time", method="fe"
        )

        assert isinstance(result, EconometricResults)
        assert abs(result.params["x1"] - 2.0) < 0.3
        assert abs(result.params["x2"] - 3.0) < 0.3

    def test_fe_robust(self, panel_data):
        """FE with robust SE should still recover slopes ≈ 2, 3 with valid SEs."""
        result = panel(
            panel_data,
            "y ~ x1 + x2",
            entity="entity",
            time="time",
            method="fe",
            robust="robust",
        )
        assert isinstance(result, EconometricResults)
        # robust SE only changes inference, not the point estimates
        assert abs(result.params["x1"] - 2.0) < 0.3
        assert abs(result.params["x2"] - 3.0) < 0.3
        # SEs must be finite, strictly positive, and not absurd
        for v in ["x1", "x2"]:
            se = float(result.std_errors[v])
            assert np.isfinite(se) and 0.0 < se < 1.0
        # both slopes are highly significant given the planted true effect
        assert np.all(np.asarray(result.pvalues) < 0.05)

    def test_fe_clustered(self, panel_data):
        """FE clustered by entity: same point estimates, valid finite SEs."""
        result = panel(
            panel_data,
            "y ~ x1 + x2",
            entity="entity",
            time="time",
            method="fe",
            cluster="entity",
        )
        assert isinstance(result, EconometricResults)
        assert abs(result.params["x1"] - 2.0) < 0.3
        assert abs(result.params["x2"] - 3.0) < 0.3
        for v in ["x1", "x2"]:
            se = float(result.std_errors[v])
            assert np.isfinite(se) and 0.0 < se < 1.0
        # CI must bracket the point estimate for each slope
        ci = result.conf_int()
        for v in ["x1", "x2"]:
            lo, hi = float(ci.loc[v].iloc[0]), float(ci.loc[v].iloc[1])
            assert lo < float(result.params[v]) < hi

    def test_fe_time_cluster_warns_and_records_few_clusters(self, panel_data):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = panel(
                panel_data,
                "y ~ x1 + x2",
                entity="entity",
                time="time",
                method="fe",
                cluster="time",
            )

        typed = [
            w.message for w in caught if isinstance(w.message, sp.AssumptionWarning)
        ]
        assert typed
        assert result.model_info["n_clusters"] == 10
        few = [v for v in result.violations() if v.get("test") == "few_clusters"]
        assert few
        assert "sp.wild_cluster_bootstrap" in few[0]["alternatives"]
        assert "sp.wild_cluster_bootstrap" in typed[0].alternative_functions


class TestRandomEffects:
    def test_re_basic(self, panel_data):
        """RE should also estimate slopes approximately."""
        result = panel(
            panel_data, "y ~ x1 + x2", entity="entity", time="time", method="re"
        )

        assert isinstance(result, EconometricResults)
        assert abs(result.params["x1"] - 2.0) < 0.5
        assert abs(result.params["x2"] - 3.0) < 0.5


class TestPooledOLS:
    def test_pooled_basic(self, panel_data):
        """Pooled OLS should work and recover slopes (alpha_i ~ uncorrelated w/ x)."""
        result = panel(
            panel_data, "y ~ x1 + x2", entity="entity", time="time", method="pooled"
        )

        assert isinstance(result, EconometricResults)
        assert len(result.params) == 3  # const + x1 + x2
        assert list(result.params.index) == ["const", "x1", "x2"]
        # alpha_i is drawn independently of x1/x2, so pooled OLS is still
        # consistent for the slopes (wider band than FE for the extra noise).
        assert abs(result.params["x1"] - 2.0) < 0.5
        assert abs(result.params["x2"] - 3.0) < 0.5
        # const ~ E[alpha_i] = 5
        assert abs(result.params["const"] - 5.0) < 1.0
        for v in ["const", "x1", "x2"]:
            se = float(result.std_errors[v])
            assert np.isfinite(se) and se > 0.0


class TestFirstDifference:
    def test_fd_basic(self, panel_data):
        """First difference also differences out alpha_i -> slopes ≈ 2, 3."""
        result = panel(
            panel_data, "y ~ x1 + x2", entity="entity", time="time", method="fd"
        )

        assert isinstance(result, EconometricResults)
        # differencing removes the entity effect, so FD is consistent here
        assert abs(result.params["x1"] - 2.0) < 0.3
        assert abs(result.params["x2"] - 3.0) < 0.3
        for v in ["x1", "x2"]:
            se = float(result.std_errors[v])
            assert np.isfinite(se) and 0.0 < se < 1.0
        assert np.all(np.asarray(result.pvalues) < 0.05)


class TestBetween:
    def test_between_basic(self, panel_data):
        """Between estimator: only 50 entity means -> noisy but well-formed.

        x1/x2 averaged within entity over 10 i.i.d. draws shrink toward 0,
        so the between slopes are high-variance and NOT a reliable recovery
        of (2, 3) here; we assert structure + finiteness + the const ~ 5
        instead of a tight slope band (kept robust, not seed-fragile).
        """
        result = panel(
            panel_data, "y ~ x1 + x2", entity="entity", time="time", method="be"
        )

        assert isinstance(result, EconometricResults)
        assert list(result.params.index) == ["const", "x1", "x2"]
        # const estimates E[alpha_i] = 5 from the 50 entity means
        assert abs(result.params["const"] - 5.0) < 1.0
        # slopes stay finite and within a generous plausible range
        for v in ["const", "x1", "x2"]:
            assert np.isfinite(float(result.params[v]))
            se = float(result.std_errors[v])
            assert np.isfinite(se) and se > 0.0
        assert abs(result.params["x1"]) < 10.0
        assert abs(result.params["x2"]) < 10.0
        # CI brackets each estimate
        ci = result.conf_int()
        for v in ["const", "x1", "x2"]:
            lo, hi = float(ci.loc[v].iloc[0]), float(ci.loc[v].iloc[1])
            assert lo < float(result.params[v]) < hi


class TestPanelGeneral:
    def test_summary(self, panel_data):
        result = panel(
            panel_data, "y ~ x1 + x2", entity="entity", time="time", method="fe"
        )
        s = result.summary()
        assert isinstance(s, str)
        assert "Panel FE" in s

    def test_diagnostics(self, panel_data):
        result = panel(
            panel_data, "y ~ x1 + x2", entity="entity", time="time", method="fe"
        )
        assert "R-squared" in result.diagnostics
        r2 = float(result.diagnostics["R-squared"])
        # well-specified low-noise DGP -> within R^2 should be high but valid
        assert 0.0 <= r2 <= 1.0
        assert r2 > 0.9
        # entity/period counts must match the simulated panel (50 x 10)
        assert int(result.diagnostics["N entities"]) == 50
        assert int(result.diagnostics["N time periods"]) == 10
        # F-stat for joint significance is large and positive here
        assert float(result.diagnostics["F-statistic"]) > 0.0
        assert 0.0 <= float(result.diagnostics["F p-value"]) <= 1.0

    def test_residuals(self, panel_data):
        result = panel(
            panel_data, "y ~ x1 + x2", entity="entity", time="time", method="fe"
        )
        resid = result.residuals()
        assert resid is not None
        assert len(resid) == len(panel_data)
        resid = np.asarray(resid, dtype=float)
        assert np.all(np.isfinite(resid))
        # FE residuals are mean-zero by construction (within transform)
        assert abs(float(np.mean(resid))) < 1e-6
        # residual scale tracks the planted eps ~ N(0, 0.5)
        assert 0.1 < float(np.std(resid)) < 1.0

    def test_fitted_values(self, panel_data):
        result = panel(
            panel_data, "y ~ x1 + x2", entity="entity", time="time", method="fe"
        )
        fitted = result.fitted_values()
        assert fitted is not None
        assert len(fitted) == len(panel_data)
        fitted = np.asarray(fitted, dtype=float)
        assert np.all(np.isfinite(fitted))
        # FE fitted values exclude the entity effects (within transform), so
        # they track y's within-variation, not its full level -> corr ~ 0.88.
        y = panel_data["y"].to_numpy(dtype=float)
        corr = float(np.corrcoef(fitted, y)[0, 1])
        assert corr > 0.8

    def test_repr(self, panel_data):
        result = panel(
            panel_data, "y ~ x1 + x2", entity="entity", time="time", method="fe"
        )
        assert "Panel FE" in repr(result)

    # --- Error handling ---

    def test_invalid_method(self, panel_data):
        with pytest.raises(ValueError, match="method must be"):
            panel(panel_data, "y ~ x1", entity="entity", time="time", method="invalid")

    def test_missing_column(self, panel_data):
        with pytest.raises(ValueError, match="not found"):
            panel(
                panel_data, "y ~ nonexistent", entity="entity", time="time", method="fe"
            )

    def test_no_tilde(self, panel_data):
        with pytest.raises(ValueError, match="must contain"):
            panel(panel_data, "y x1", entity="entity", time="time", method="fe")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
