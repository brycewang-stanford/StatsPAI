"""Tests for CausalResult.violations / .to_agent_summary and the
matching EconometricResults methods."""

import json

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult, EconometricResults


# ====================================================================== #
#  CausalResult
# ====================================================================== #


def _make_causal_result(model_info=None, method="did_2x2"):
    return CausalResult(
        method=method,
        estimand="ATT",
        estimate=1.5,
        se=0.5,
        pvalue=0.003,
        ci=(0.5, 2.5),
        alpha=0.05,
        n_obs=1000,
        model_info=model_info or {},
        _citation_key="did_2x2",
    )


class TestCausalViolations:
    def test_no_violations_when_clean(self):
        r = _make_causal_result(model_info={"pretrend_test": {"pvalue": 0.50}})
        assert r.violations() == []

    def test_pretrend_error_when_significant(self):
        r = _make_causal_result(model_info={"pretrend_test": {"pvalue": 0.01}})
        vs = r.violations()
        assert len(vs) == 1
        v = vs[0]
        assert v["test"] == "pretrend"
        assert v["severity"] == "error"
        assert v["value"] == pytest.approx(0.01)
        assert "sp.sensitivity_rr" in v["alternatives"]
        assert "Rambachan" in v["recovery_hint"]

    def test_pretrend_warning_when_borderline(self):
        r = _make_causal_result(model_info={"pretrend_test": {"pvalue": 0.08}})
        vs = r.violations()
        assert len(vs) == 1
        assert vs[0]["severity"] == "warning"

    def test_mccrary_triggers_only_for_rd(self):
        rd = _make_causal_result(
            model_info={"mccrary": {"pvalue": 0.001}},
            method="rdrobust",
        )
        assert any(v["test"] == "mccrary_density" for v in rd.violations())

        did = _make_causal_result(
            model_info={"mccrary": {"pvalue": 0.001}},
            method="did_2x2",
        )
        # DID result with mccrary field should not be flagged as RD issue
        assert not any(v["test"] == "mccrary_density" for v in did.violations())

    def test_rhat_flagged(self):
        r = _make_causal_result(model_info={"rhat_max": 1.15})
        vs = r.violations()
        assert any(v["test"] == "rhat" and v["severity"] == "error" for v in vs)

    def test_divergences_flagged(self):
        r = _make_causal_result(model_info={"divergences": 7})
        vs = r.violations()
        assert any(v["test"] == "divergences" for v in vs)

    def test_nan_estimate_flagged(self):
        r = _make_causal_result()
        r.estimate = float("nan")
        vs = r.violations()
        assert any(v["test"] == "estimate_finite" for v in vs)

    def test_negative_se_flagged(self):
        r = _make_causal_result()
        r.se = -1.0
        vs = r.violations()
        assert any(v["test"] == "se_positive" for v in vs)

    def test_weak_iv_flagged_only_for_iv(self):
        iv_res = _make_causal_result(
            model_info={"first_stage_f": 3.5},
            method="2sls_iv",
        )
        assert any(v["test"] == "weak_instrument" for v in iv_res.violations())

        did = _make_causal_result(
            model_info={"first_stage_f": 3.5},
            method="did_2x2",
        )
        # DID result is not IV, so weak-IV rule shouldn't fire
        assert not any(v["test"] == "weak_instrument" for v in did.violations())


class TestCausalAgentSummary:
    def test_keys_present(self):
        r = _make_causal_result()
        s = r.to_agent_summary()
        for key in (
            "kind", "method", "method_family", "estimand",
            "point", "n_obs", "diagnostics", "violations",
            "next_steps", "citation_key",
        ):
            assert key in s

    def test_point_estimates(self):
        r = _make_causal_result()
        s = r.to_agent_summary()
        assert s["point"]["estimate"] == pytest.approx(1.5)
        assert s["point"]["se"] == pytest.approx(0.5)
        assert s["point"]["ci"] == [pytest.approx(0.5), pytest.approx(2.5)]

    def test_method_family_inferred(self):
        r = _make_causal_result(method="callaway_santanna")
        assert r.to_agent_summary()["method_family"] == "did"

    def test_json_roundtrip(self):
        r = _make_causal_result(
            model_info={
                "pretrend_test": {"pvalue": 0.01},
                "matrix": np.ones((3, 3)),  # non-scalar: should be stringified
            }
        )
        s = r.to_agent_summary()
        # Must be JSON-serialisable
        text = json.dumps(s, default=str)
        assert "pretrend" in text
        assert "<ndarray" in text or "matrix" in text

    def test_violations_are_list_of_dicts(self):
        r = _make_causal_result(model_info={"pretrend_test": {"pvalue": 0.01}})
        s = r.to_agent_summary()
        assert isinstance(s["violations"], list)
        assert all(isinstance(v, dict) for v in s["violations"])
        assert any(v["test"] == "pretrend" for v in s["violations"])


# ====================================================================== #
#  EconometricResults
# ====================================================================== #


class TestEconometricAgentSummary:
    def test_regress_agent_summary(self):
        np.random.seed(0)
        df = pd.DataFrame({
            "y": np.random.randn(200),
            "x": np.random.randn(200),
        })
        r = sp.regress("y ~ x", data=df)
        s = r.to_agent_summary()

        assert s["kind"] == "econometric_result"
        assert s["n_obs"] == 200
        terms = {c["term"] for c in s["coefficients"]}
        assert "Intercept" in terms
        assert "x" in terms
        for c in s["coefficients"]:
            assert isinstance(c["estimate"], float)
            assert isinstance(c["std_error"], float)
            assert isinstance(c["p_value"], float)

    def test_regress_violations_empty_on_good_data(self):
        np.random.seed(0)
        df = pd.DataFrame({
            "y": np.random.randn(200),
            "x": np.random.randn(200),
        })
        r = sp.regress("y ~ x", data=df)
        assert r.violations() == []

    def test_agent_summary_json_safe(self):
        np.random.seed(0)
        df = pd.DataFrame({
            "y": np.random.randn(100),
            "x": np.random.randn(100),
        })
        r = sp.regress("y ~ x", data=df)
        s = r.to_agent_summary()
        # Must round-trip through json
        json.dumps(s, default=str)
