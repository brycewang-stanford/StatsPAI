"""Tests for core/next_steps.py — method-aware next-step recommendations.

Covers
------
* ``Step`` dataclass construction, repr, to_dict.
* ``_format_steps`` and ``_steps_repr_html`` rendering.
* ``_detect_family`` classification of method strings.
* ``econometric_next_steps`` for OLS / IV / panel results.
* ``causal_next_steps`` for DiD / RD / IV / matching / synth / DML / HTE /
  mediation families — including family-specific steps and universal steps.
"""

import re
from typing import Any, Dict, List, Optional

import pytest

from statspai.core.next_steps import (
    Step,
    _format_steps,
    _steps_repr_html,
    econometric_next_steps,
    causal_next_steps,
    _detect_family,
)


# ======================================================================
#  Step basics
# ======================================================================


class TestStep:
    def test_defaults(self):
        s = Step(action="sp.test()", reason="just testing")
        assert s.action == "sp.test()"
        assert s.reason == "just testing"
        assert s.priority == "recommended"
        assert s.category == "diagnostics"

    def test_custom_priority(self):
        s = Step("x", "y", priority="essential", category="robustness")
        assert s.priority == "essential"
        assert s.category == "robustness"

    def test_repr(self):
        s = Step("sp.estat(r)", "Post-estimation checks", priority="essential")
        r = repr(s)
        assert "sp.estat(r)" in r
        assert "Post-estimation" in r

    def test_to_dict(self):
        s = Step("sp.test()", "reason", "optional", "export")
        d = s.to_dict()
        assert d["action"] == "sp.test()"
        assert d["priority"] == "optional"
        assert d["category"] == "export"


# ======================================================================
#  Formatters
# ======================================================================


class TestFormatSteps:
    def test_empty(self):
        output = _format_steps([])
        assert "Suggested Next Steps" in output

    def test_groups_by_priority(self):
        steps = [
            Step("a", "opt", priority="optional"),
            Step("b", "ess", priority="essential"),
            Step("c", "rec", priority="recommended"),
            Step("d", "ess2", priority="essential"),
        ]
        output = _format_steps(steps)
        # Essential first, then recommended, then optional
        ess_idx = output.index("Essential")
        rec_idx = output.index("Recommended")
        opt_idx = output.index("Optional")
        assert ess_idx < rec_idx < opt_idx


class TestStepsReprHtml:
    def test_contains_html_structure(self):
        steps = [
            Step("sp.test()", "A test step", priority="essential", category="diagnostics"),
        ]
        html = _steps_repr_html(steps)
        assert "sp-ns" in html
        assert "sp.test()" in html
        assert "essential" in html

    def test_multiple_categories(self):
        steps = [
            Step("a", "diag", category="diagnostics"),
            Step("b", "rob", category="robustness"),
            Step("c", "exp", category="export"),
        ]
        html = _steps_repr_html(steps)
        assert "Diagnostics" in html
        assert "Robustness" in html
        assert "Export" in html

    def test_empty(self):
        html = _steps_repr_html([])
        assert "sp-ns" in html


# ======================================================================
#  _detect_family
# ======================================================================


class TestDetectFamily:
    def test_synth_before_did(self):
        """sdid must match synth, not did (order matters)."""
        assert _detect_family("sdid") == "synth"
        assert _detect_family("augsynth") == "synth"

    def test_did(self):
        for m in ("did", "diff_in_diff", "callaway", "sun_abraham",
                  "staggered", "imputation", "twfe"):
            assert _detect_family(m) == "did", f"{m} -> did"

    def test_rd(self):
        for m in ("rd", "rdrobust", "discontinuity", "kink", "rdit"):
            assert _detect_family(m) == "rd", f"{m} -> rd"

    def test_iv(self):
        for m in ("2sls", "iv", "instrumental",
                  "bartik", "shift-share", "deepiv"):
            assert _detect_family(m) == "iv", f"{m} -> iv"

    def test_matching(self):
        for m in ("matching", "psm", "cem", "ipw", "entropy_bal", "aipw", "tmle"):
            assert _detect_family(m) == "matching", f"{m} -> matching"

    def test_dml(self):
        for m in ("dml", "double_ml", "debiased_lasso"):
            assert _detect_family(m) == "dml", f"{m} -> dml"

    def test_hte(self):
        for m in ("metalearner", "slearner", "tlearner", "xlearner",
                  "rlearner", "drlearner", "causal forest", "bcf", "cate"):
            assert _detect_family(m) == "hte", f"{m} -> hte"

    def test_mediation(self):
        assert _detect_family("mediation") == "mediation"

    def test_generic(self):
        assert _detect_family("ols") == "generic"
        assert _detect_family("logit") == "generic"
        assert _detect_family("probit") == "generic"
        assert _detect_family("") == "generic"
        assert _detect_family("unknown_estimator") == "generic"

    def test_case_insensitive(self):
        assert _detect_family("DID") == "did"
        assert _detect_family("IV") == "iv"
        assert _detect_family("RDD") == "rd"
        assert _detect_family("SYNTH") == "synth"


# ======================================================================
#  Mock-helpers for result objects
# ======================================================================


def _mock_econ_result(
    model_type: str = "ols",
    robust: str = "nonrobust",
    has_residuals: bool = True,
    has_X: bool = True,
    nobs: int = 100,
    n_params: int = 3,
    dep_var: str = "y",
) -> Any:
    """Build a minimal mock of EconometricResults."""

    class MockEconResult:
        def __init__(self):
            self.model_info: Dict[str, Any] = {"model_type": model_type, "robust": robust}
            self.data_info: Dict[str, Any] = {
                "residuals": [0.1] * nobs if has_residuals else None,
                "X": [[1]] * nobs if has_X else None,
                "nobs": nobs,
                "dependent_var": dep_var,
            }
            self.params: List[float] = [0.5] * n_params

    return MockEconResult()


def _mock_causal_result(
    method: str = "did",
    has_event_study: bool = False,
    has_detail: bool = False,
    estimate: float = 1.0,
    se: float = 0.5,
    method_family_hint: str = "",
) -> Any:
    """Build a minimal mock of CausalResult."""

    class MockDetail:
        pass

    class MockCausalResult:
        def __init__(self):
            self.method: str = method
            self.model_info: Dict[str, Any] = {}
            self.detail: Any = MockDetail() if has_detail else None
            if has_detail:
                self.detail.columns = ["relative_time", "estimate", "se"]
            self.estimate: float = estimate
            self.se: float = se
            if has_event_study:
                self.model_info["event_study"] = True

    return MockCausalResult()


# ======================================================================
#  econometric_next_steps
# ======================================================================


class TestEconometricNextSteps:
    def test_ols_default(self):
        steps = econometric_next_steps(_mock_econ_result())
        assert len(steps) >= 5
        actions = [s.action for s in steps]
        # Core steps
        assert any("sp.estat(result, 'all')" in a for a in actions)
        assert any("sp.margins" in a for a in actions)
        assert any("sp.oster_delta" in a for a in actions)
        assert any("sp.esttab" in a for a in actions)

    def test_ols_no_residuals(self):
        """Without residuals/X, estat shortcut to IC."""
        steps = econometric_next_steps(
            _mock_econ_result(has_residuals=False, has_X=False))
        actions = [s.action for s in steps]
        assert any("sp.estat(result, 'ic')" in a for a in actions)

    def test_nonrobust_triggers_robust_se(self):
        steps = econometric_next_steps(
            _mock_econ_result(robust="nonrobust"))
        actions = [s.action for s in steps]
        assert any("robust='hc1'" in a for a in actions)

    def test_robust_se_already_set(self):
        steps = econometric_next_steps(
            _mock_econ_result(robust="hc1"))
        actions = [s.action for s in steps]
        # Should not suggest robust SE re-estimation
        assert not any("robust='hc1'" in a for a in actions)

    def test_iv_adds_specific_steps(self):
        steps = econometric_next_steps(
            _mock_econ_result(model_type="iv"))
        actions = [s.action for s in steps]
        assert any("firststage" in a for a in actions)
        assert any("overid" in a for a in actions)
        assert any("endogenous" in a for a in actions)
        assert any("kitagawa" in a for a in actions)

    def test_many_params_triggers_stepwise(self):
        steps = econometric_next_steps(
            _mock_econ_result(n_params=10))
        actions = [s.action for s in steps]
        assert any("stepwise" in a for a in actions)

    def test_few_params_no_stepwise(self):
        steps = econometric_next_steps(
            _mock_econ_result(n_params=2))
        actions = [s.action for s in steps]
        assert not any("stepwise" in a for a in actions)

    def test_panel_adds_hausman(self):
        steps = econometric_next_steps(
            _mock_econ_result(model_type="panel_fe"))
        actions = [s.action for s in steps]
        assert any("hausman" in a for a in actions)
        assert any("twoway_cluster" in a for a in actions)


# ======================================================================
#  causal_next_steps
# ======================================================================


class TestCausalNextSteps:
    def test_did_family_basic(self):
        steps = causal_next_steps(_mock_causal_result(method="did"))
        actions = [s.action for s in steps]
        # No event study → suggests building one
        assert any("event_study" in a for a in actions)

    def test_did_with_event_study(self):
        """With event study → no need to suggest building one."""
        steps = causal_next_steps(_mock_causal_result(
            method="did", has_event_study=True))
        actions = [s.action for s in steps]
        assert any("pretrends_test" in a for a in actions)
        assert any("sensitivity_rr" in a for a in actions)
        assert not any("event_study(data=df" in a for a in actions)

    def test_did_twfe_suggests_alternatives(self):
        steps = causal_next_steps(_mock_causal_result(
            method="twfe"))
        actions = [s.action for s in steps]
        assert any("bacon" in a for a in actions)
        assert any("did_imputation" in a for a in actions)

    def test_rd_family(self):
        steps = causal_next_steps(_mock_causal_result(method="rdrobust"))
        actions = [s.action for s in steps]
        assert any("mccrary" in a for a in actions)
        assert any("rd_honest" in a for a in actions)
        assert any("rdbalance" in a for a in actions)
        assert any("rdbwsensitivity" in a for a in actions)
        assert any("rdplacebo" in a for a in actions)
        assert any("result.plot()" in a for a in actions)

    def test_iv_family(self):
        steps = causal_next_steps(_mock_causal_result(method="iv"))
        actions = [s.action for s in steps]
        assert any("firststage" in a for a in actions)
        assert any("anderson_rubin" in a for a in actions)
        assert any("kitagawa" in a for a in actions)
        assert any("iv_bounds" in a for a in actions)

    def test_matching_family(self):
        steps = causal_next_steps(_mock_causal_result(method="psm"))
        actions = [s.action for s in steps]
        assert any("ps_balance" in a for a in actions)
        assert any("overlap_plot" in a for a in actions)
        assert any("love_plot" in a for a in actions)
        assert any("trimming" in a for a in actions)
        assert any("sensemakr" in a for a in actions)

    def test_synth_family(self):
        steps = causal_next_steps(_mock_causal_result(method="synth"))
        actions = [s.action for s in steps]
        assert any("result.plot()" in a for a in actions)
        assert any("diagnose_result" in a for a in actions)
        assert any("conformal_synth" in a for a in actions)

    def test_dml_family(self):
        steps = causal_next_steps(_mock_causal_result(method="dml"))
        actions = [s.action for s in steps]
        assert any("diagnose_result" in a for a in actions)
        assert any("cate_summary" in a for a in actions)

    def test_hte_family(self):
        steps = causal_next_steps(_mock_causal_result(method="causal forest"))
        actions = [s.action for s in steps]
        assert any("cate_summary" in a for a in actions)
        assert any("gate_test" in a for a in actions)
        assert any("blp_test" in a for a in actions)
        assert any("compare_metalearners" in a for a in actions)

    def test_mediation_family(self):
        steps = causal_next_steps(_mock_causal_result(method="mediation"))
        actions = [s.action for s in steps]
        assert any("result.plot()" in a for a in actions)
        assert any("sensemakr" in a for a in actions)

    def test_universal_evalue(self):
        """E-value appears for all families except synth."""
        for method in ("did", "rd", "iv", "psm", "dml",
                       "causal_forest", "mediation"):
            steps = causal_next_steps(_mock_causal_result(method=method))
            actions = [s.action for s in steps]
            assert any("evalue" in a for a in actions), f"{method} missing evalue"
        # Synth should NOT have evalue (has own sensitivity)
        steps_synth = causal_next_steps(_mock_causal_result(method="synth"))
        actions_synth = [s.action for s in steps_synth]
        assert not any("evalue" in a for a in actions_synth)

    def test_universal_subgroup(self):
        """subgroup_analysis appears for all families."""
        for method in ("did", "rd", "iv", "synth"):
            steps = causal_next_steps(_mock_causal_result(method=method))
            actions = [s.action for s in steps]
            assert any("subgroup" in a for a in actions)

    def test_universal_export(self):
        """regtable and cite appear for all families."""
        for method in ("did", "rd", "iv"):
            steps = causal_next_steps(_mock_causal_result(method=method))
            actions = [s.action for s in steps]
            assert any("regtable" in a for a in actions)
            assert any(".cite()" in a for a in actions)

    def test_long_output_is_stable(self):
        """Run with every method family; verify total step count range."""
        for method in ("did", "rd", "iv", "psm", "synth", "dml",
                       "causal_forest", "mediation", "generic_estimator"):
            steps = causal_next_steps(_mock_causal_result(method=method))
            # Every family should generate at least 3 universal steps
            assert len(steps) >= 3, f"{method}: only {len(steps)} steps"
            # No family should generate more than 15 (sanity cap)
            assert len(steps) <= 15, f"{method}: {len(steps)} steps (too many)"
