"""
Tests for the new features:
- diagnose_result (method-aware diagnostic battery)
- ipw (standalone IPW estimator)
- dag (DAG declaration + adjustment sets)
- event_study (traditional OLS event study)
- augsynth (Augmented Synthetic Control)
"""

import numpy as np
import pandas as pd
import pytest


# ====================================================================== #
#  Test diagnose_result
# ====================================================================== #

class TestDiagnoseResult:
    def test_diagnose_ols(self):
        from statspai import regress
        from statspai.diagnostics.battery import diagnose_result
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame({"y": rng.randn(n), "x1": rng.randn(n), "x2": rng.randn(n)})
        result = regress("y ~ x1 + x2", data=df)
        diag = diagnose_result(result, print_results=False)
        assert diag["method_type"] == "ols"
        assert len(diag["checks"]) > 0

    def test_diagnose_did(self):
        from statspai import did
        from statspai.diagnostics.battery import diagnose_result
        rng = np.random.RandomState(42)
        n = 400
        df = pd.DataFrame({
            "y": rng.randn(n),
            "treated": np.repeat([0, 1], n // 2),
            "post": np.tile([0, 1], n // 2),
        })
        df.loc[(df["treated"] == 1) & (df["post"] == 1), "y"] += 2.0
        result = did(df, y="y", treat="treated", time="post")
        diag = diagnose_result(result, print_results=False)
        assert diag["method_type"] == "did"

    def test_diagnose_iv(self):
        from statspai import ivreg
        from statspai.diagnostics.battery import diagnose_result
        rng = np.random.RandomState(42)
        n = 300
        z = rng.randn(n)
        x = 0.8 * z + rng.randn(n) * 0.5
        y = 2.0 * x + rng.randn(n)
        df = pd.DataFrame({"y": y, "x": x, "z": z})
        result = ivreg("y ~ (x ~ z)", data=df)
        diag = diagnose_result(result, print_results=False)
        assert diag["method_type"] == "iv"

    def test_diagnose_unknown(self):
        from statspai.diagnostics.battery import diagnose_result
        # Mock result with no recognizable method
        class FakeResult:
            model_info = {"model_type": "unknown_model"}
            params = pd.Series([1.0, 2.0])
        diag = diagnose_result(FakeResult(), print_results=False)
        assert diag["method_type"] == "generic"


# ====================================================================== #
#  Test IPW
# ====================================================================== #

class TestIPW:
    def _make_data(self, n=500, ate=2.0, seed=42):
        rng = np.random.RandomState(seed)
        x1 = rng.randn(n)
        x2 = rng.randn(n)
        ps = 1 / (1 + np.exp(-(0.5 * x1 - 0.3 * x2)))
        treat = rng.binomial(1, ps)
        y = ate * treat + x1 + 0.5 * x2 + rng.randn(n)
        return pd.DataFrame({"y": y, "treat": treat, "x1": x1, "x2": x2})

    def test_ipw_ate(self):
        from statspai import ipw
        df = self._make_data()
        result = ipw(df, y="y", treat="treat", covariates=["x1", "x2"],
                     estimand="ATE", n_bootstrap=100, seed=42)
        assert hasattr(result, "estimate")
        assert hasattr(result, "se")
        assert result.se > 0
        # ATE should be roughly 2.0
        assert abs(result.estimate - 2.0) < 1.5

    def test_ipw_att(self):
        from statspai import ipw
        df = self._make_data()
        result = ipw(df, y="y", treat="treat", covariates=["x1", "x2"],
                     estimand="ATT", n_bootstrap=50, seed=42)
        assert result.model_info["estimand"] == "ATT"
        assert result.model_info["n_treated"] > 0

    def test_ipw_trimming(self):
        from statspai import ipw
        df = self._make_data()
        result = ipw(df, y="y", treat="treat", covariates=["x1", "x2"],
                     trim=0.1, n_bootstrap=50, seed=42)
        assert result.model_info["trim"] == 0.1
        assert result.model_info["pscore_min"] >= 0.1

    def test_ipw_bad_estimand(self):
        from statspai import ipw
        df = self._make_data(n=50)
        with pytest.raises(ValueError, match="estimand"):
            ipw(df, y="y", treat="treat", covariates=["x1"], estimand="WRONG")


# ====================================================================== #
#  Test DAG
# ====================================================================== #

class TestDAG:
    def test_basic_dag(self):
        from statspai import dag
        g = dag("Z -> X; Z -> Y; X -> Y")
        assert len(g.observed_nodes) == 3
        assert ("Z", "X") in g.edges
        assert ("X", "Y") in g.edges
        assert ("Z", "Y") in g.edges

    def test_chain(self):
        from statspai import dag
        g = dag("A -> B -> C -> D")
        assert g.children("A") == {"B"}
        assert g.children("B") == {"C"}
        assert g.children("C") == {"D"}

    def test_ancestors(self):
        from statspai import dag
        g = dag("A -> B -> C; A -> C")
        assert g.ancestors("C") == {"A", "B"}
        assert g.ancestors("A") == set()

    def test_descendants(self):
        from statspai import dag
        g = dag("A -> B -> C")
        assert g.descendants("A") == {"B", "C"}

    def test_adjustment_sets_confounding(self):
        from statspai import dag
        # Classic confounding: Z -> X, Z -> Y, X -> Y
        g = dag("Z -> X; Z -> Y; X -> Y")
        adj = g.adjustment_sets("X", "Y")
        assert {"Z"} in adj

    def test_adjustment_sets_no_confounding(self):
        from statspai import dag
        # No backdoor path: X -> Y
        g = dag("X -> Y")
        adj = g.adjustment_sets("X", "Y")
        assert set() in adj  # empty set is valid

    def test_adjustment_sets_mediator(self):
        from statspai import dag
        # X -> M -> Y; no confounding — don't condition on M for total effect
        g = dag("X -> M -> Y")
        adj = g.adjustment_sets("X", "Y")
        # Empty set should be valid (no backdoor paths)
        assert set() in adj

    def test_bidirected_edge(self):
        from statspai import dag
        g = dag("X <-> Y; X -> Y")
        # Should have a latent node
        assert any(n.startswith("_L_") for n in g.nodes)

    def test_d_separation(self):
        from statspai import dag
        g = dag("X -> Z -> Y")
        # X and Y are d-separated given Z
        assert g.d_separated("X", "Y", {"Z"})
        # X and Y are NOT d-separated without conditioning
        assert not g.d_separated("X", "Y")

    def test_collider(self):
        from statspai import dag
        g = dag("X -> M; Y -> M")
        # M is a collider on path X-M-Y
        assert g.is_collider("M", ["X", "M", "Y"])

    def test_repr(self):
        from statspai import dag
        g = dag("X -> Y")
        assert "DAG" in repr(g)


# ====================================================================== #
#  Test Event Study
# ====================================================================== #

class TestEventStudy:
    def _make_panel(self, n_units=50, n_periods=10, treat_time=6, effect=3.0, seed=42):
        rng = np.random.RandomState(seed)
        rows = []
        for i in range(n_units):
            treated = i < n_units // 2
            for t in range(1, n_periods + 1):
                y = rng.randn() + (effect if treated and t >= treat_time else 0)
                rows.append({
                    "unit": i,
                    "time": t,
                    "y": y,
                    "treat_time": treat_time if treated else np.nan,
                })
        return pd.DataFrame(rows)

    def test_event_study_basic(self):
        from statspai import event_study
        df = self._make_panel()
        result = event_study(df, y="y", treat_time="treat_time",
                             time="time", unit="unit", window=(-3, 3))
        assert hasattr(result, "estimate")
        assert "event_study" in result.model_info
        es = result.model_info["event_study"]
        assert isinstance(es, pd.DataFrame)
        assert "relative_time" in es.columns
        assert "estimate" in es.columns

    def test_event_study_pretrend(self):
        from statspai import event_study
        df = self._make_panel()
        result = event_study(df, y="y", treat_time="treat_time",
                             time="time", unit="unit")
        pretrend = result.model_info.get("pretrend_test")
        assert pretrend is not None
        assert "pvalue" in pretrend

    def test_event_study_post_positive(self):
        from statspai import event_study
        df = self._make_panel(effect=5.0)
        result = event_study(df, y="y", treat_time="treat_time",
                             time="time", unit="unit")
        # The overall ATT should be positive
        assert result.estimate > 0

    def test_event_study_ref_period_zero(self):
        from statspai import event_study
        df = self._make_panel()
        es = result = event_study(df, y="y", treat_time="treat_time",
                                   time="time", unit="unit")
        es_df = result.model_info["event_study"]
        ref_row = es_df[es_df["relative_time"] == -1]
        assert len(ref_row) == 1
        assert ref_row.iloc[0]["estimate"] == 0.0


# ====================================================================== #
#  Test Augmented Synthetic Control
# ====================================================================== #

class TestAugSynth:
    def _make_synth_data(self, n_units=10, n_periods=20, treat_time=11, effect=5.0, seed=42):
        rng = np.random.RandomState(seed)
        rows = []
        # Common factor
        factor = np.cumsum(rng.randn(n_periods) * 0.5)
        for i in range(n_units):
            loading = 1.0 + rng.randn() * 0.3
            for t in range(1, n_periods + 1):
                y = loading * factor[t - 1] + rng.randn() * 0.5
                if i == 0 and t >= treat_time:
                    y += effect
                rows.append({"unit": i, "time": t, "y": y})
        return pd.DataFrame(rows)

    def test_augsynth_basic(self):
        from statspai import augsynth
        df = self._make_synth_data()
        result = augsynth(df, outcome="y", unit="unit", time="time",
                          treated_unit=0, treatment_time=11)
        assert hasattr(result, "estimate")
        assert hasattr(result, "se")
        assert result.model_info["n_donors"] == 9
        assert result.model_info["pre_rmspe"] >= 0

    def test_augsynth_effect_direction(self):
        from statspai import augsynth
        df = self._make_synth_data(effect=10.0)
        result = augsynth(df, outcome="y", unit="unit", time="time",
                          treated_unit=0, treatment_time=11)
        # Effect should be positive (true effect = 10)
        assert result.estimate > 0

    def test_augsynth_detail(self):
        from statspai import augsynth
        df = self._make_synth_data()
        result = augsynth(df, outcome="y", unit="unit", time="time",
                          treated_unit=0, treatment_time=11)
        detail = result.detail
        assert isinstance(detail, pd.DataFrame)
        assert "effect" in detail.columns
        assert len(detail) == 10  # 10 post-treatment periods
