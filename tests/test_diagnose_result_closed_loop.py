"""Tests for the sp.diagnose_result ↔ result.violations() closed loop.

Verifies that:
1. ``diagnose_result`` returns the traditional family battery output.
2. The output now carries a ``violations`` key sourced from
   ``result.violations()`` — structured items with severity + recovery
   hints.
3. The output also carries a ``next_steps`` key from
   ``result.next_steps(print_result=False)``.
4. All three coexist without breaking the existing interface.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult


def _did_result_with_pretrend_violation():
    return CausalResult(
        method="did_2x2",
        estimand="ATT",
        estimate=1.5,
        se=0.5,
        pvalue=0.003,
        ci=(0.5, 2.5),
        alpha=0.05,
        n_obs=1000,
        model_info={
            "pretrend_test": {
                "pvalue": 0.01, "statistic": 9.2, "df": 3,
            },
        },
    )


def _clean_did_result():
    return CausalResult(
        method="did_2x2",
        estimand="ATT",
        estimate=1.5,
        se=0.5,
        pvalue=0.003,
        ci=(0.5, 2.5),
        alpha=0.05,
        n_obs=1000,
        model_info={
            "pretrend_test": {
                "pvalue": 0.80, "statistic": 1.2, "df": 3,
            },
        },
    )


class TestClosedLoop:
    def test_violations_key_present(self):
        out = sp.diagnose_result(
            _did_result_with_pretrend_violation(), print_results=False,
        )
        assert "violations" in out
        assert isinstance(out["violations"], list)

    def test_next_steps_key_present(self):
        out = sp.diagnose_result(
            _did_result_with_pretrend_violation(), print_results=False,
        )
        assert "next_steps" in out
        assert isinstance(out["next_steps"], list)
        assert len(out["next_steps"]) > 0

    def test_violation_surfaces_pretrend(self):
        out = sp.diagnose_result(
            _did_result_with_pretrend_violation(), print_results=False,
        )
        tests = {v["test"] for v in out["violations"]}
        assert "pretrend" in tests

    def test_violation_carries_recovery_hint(self):
        out = sp.diagnose_result(
            _did_result_with_pretrend_violation(), print_results=False,
        )
        pretrend = next(
            v for v in out["violations"] if v["test"] == "pretrend"
        )
        assert pretrend["recovery_hint"]
        assert "sp.sensitivity_rr" in pretrend["alternatives"]

    def test_clean_result_has_empty_violations(self):
        out = sp.diagnose_result(_clean_did_result(), print_results=False)
        assert out["violations"] == []

    def test_backward_compatibility_checks_still_there(self):
        out = sp.diagnose_result(
            _did_result_with_pretrend_violation(), print_results=False,
        )
        # Existing interface untouched
        assert "method_type" in out
        assert "checks" in out
        assert out["method_type"] == "did"

    def test_regress_result_also_loops(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "y": rng.normal(size=200),
            "x": rng.normal(size=200),
        })
        r = sp.regress("y ~ x", data=df)
        out = sp.diagnose_result(r, print_results=False)
        assert "violations" in out
        assert "next_steps" in out
        # Clean random data → empty violations
        assert out["violations"] == []

    def test_print_does_not_crash(self, capsys):
        # Exercise the _print_battery path — this is the agent-facing
        # human readable view
        sp.diagnose_result(
            _did_result_with_pretrend_violation(), print_results=True,
        )
        captured = capsys.readouterr().out
        # Print should include the new Structured violations block
        assert "Structured violations" in captured
        assert "pretrend" in captured
