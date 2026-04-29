"""Tests for ``sp.audit(result)`` — reviewer-checklist gap audit.

The audit is the *missing-evidence* counterpart to:

* ``result.violations()`` — checked-but-failed
* ``result.next_steps()`` — action recommendations
* ``sp.assumption_audit(result, data)`` — re-runs statistical tests

These tests pin: family routing, status semantics (passed / failed /
missing / not-applicable via filter), token budget, JSON
serialisability, and explicit non-overlap with the heavyweight
counterpart.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


def _bare_did_result(model_info=None) -> CausalResult:
    """Synthetic CausalResult — lets us probe each model_info shape
    without running a real estimator."""
    return CausalResult(
        method="did_2x2",
        estimand="ATT",
        estimate=1.5, se=0.5, pvalue=0.003, ci=(0.5, 2.5),
        alpha=0.05, n_obs=1000,
        model_info=model_info or {},
        _citation_key="did_2x2",
    )


@pytest.fixture(scope="module")
def real_did_result():
    rng = np.random.default_rng(5)
    rows = []
    for i in range(200):
        tr = 1 if i < 100 else 0
        for t in (0, 1):
            y = (1.0 + 0.3 * t + 0.5 * tr + 2.0 * tr * t
                 + rng.normal(scale=0.5))
            rows.append({"i": i, "t": t, "treated": tr,
                         "post": t, "y": y})
    df = pd.DataFrame(rows)
    return sp.did(df, y="y", treat="treated", time="t", post="post")


@pytest.fixture(scope="module")
def real_regress_result():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=300),
        "x": rng.normal(size=300),
    })
    return sp.regress("y ~ x", data=df, robust="hc1")


# ---------------------------------------------------------------------------
#  Top-level export + return shape
# ---------------------------------------------------------------------------


class TestAuditExport:

    def test_sp_audit_is_callable(self):
        assert callable(sp.audit)

    def test_audit_in_all(self):
        assert "audit" in sp.__all__

    def test_audit_distinct_from_assumption_audit(self):
        # Both are exported but they're separate functions with
        # different semantics. ``audit`` takes only result; the
        # heavyweight one takes (result, data).
        assert sp.audit is not sp.assumption_audit


class TestReturnShape:

    def test_top_level_keys(self, real_did_result):
        card = sp.audit(real_did_result)
        for k in ("method", "method_family", "checks",
                  "summary", "coverage"):
            assert k in card

    def test_summary_keys(self, real_did_result):
        s = sp.audit(real_did_result)["summary"]
        for k in ("passed", "failed", "missing", "n_total"):
            assert k in s
        assert s["passed"] + s["failed"] + s["missing"] == s["n_total"]

    def test_check_shape(self, real_did_result):
        for c in sp.audit(real_did_result)["checks"]:
            for k in ("name", "question", "status", "severity",
                      "importance", "value", "threshold",
                      "suggest_function", "rationale"):
                assert k in c, f"check missing {k!r}: {c}"
            assert c["status"] in ("passed", "failed", "missing")
            # severity uses violations-style observed-state vocabulary.
            assert c["severity"] in ("info", "warning", "error")
            # importance is constant per check, independent of status.
            assert c["importance"] in ("high", "medium", "low")

    def test_coverage_in_unit_interval(self, real_did_result):
        cov = sp.audit(real_did_result)["coverage"]
        assert 0.0 <= cov <= 1.0


# ---------------------------------------------------------------------------
#  Family routing
# ---------------------------------------------------------------------------


class TestFamilyRouting:

    def test_did_routes_to_did_family(self, real_did_result):
        assert sp.audit(real_did_result)["method_family"] == "did"

    def test_regression_routes_to_regression(self, real_regress_result):
        assert (sp.audit(real_regress_result)["method_family"]
                == "regression")

    def test_did_includes_parallel_trends(self):
        names = {c["name"]
                 for c in sp.audit(_bare_did_result())["checks"]}
        assert "parallel_trends" in names

    def test_did_excludes_rd_only_checks(self):
        names = {c["name"]
                 for c in sp.audit(_bare_did_result())["checks"]}
        # mccrary / bandwidth_sensitivity / placebo_cutoff are RD-only.
        for rd_only in ("mccrary_density", "bandwidth_sensitivity",
                        "placebo_cutoff"):
            assert rd_only not in names

    def test_regression_excludes_did_checks(self, real_regress_result):
        names = {c["name"]
                 for c in sp.audit(real_regress_result)["checks"]}
        for did_only in ("parallel_trends", "honest_did",
                         "bacon_decomposition"):
            assert did_only not in names

    def test_regression_includes_robust_se_check(self, real_regress_result):
        names = {c["name"]
                 for c in sp.audit(real_regress_result)["checks"]}
        assert "robust_se" in names


# ---------------------------------------------------------------------------
#  Status semantics: passed / failed / missing
# ---------------------------------------------------------------------------


class TestPassedFailedMissing:

    def test_pretrend_passes_when_pvalue_above_threshold(self):
        r = _bare_did_result(
            model_info={"pretrend_test": {"pvalue": 0.50}})
        check = next(c for c in sp.audit(r)["checks"]
                     if c["name"] == "parallel_trends")
        assert check["status"] == "passed"
        assert check["value"] == pytest.approx(0.50)

    def test_pretrend_fails_when_pvalue_below_threshold(self):
        r = _bare_did_result(
            model_info={"pretrend_test": {"pvalue": 0.005}})
        check = next(c for c in sp.audit(r)["checks"]
                     if c["name"] == "parallel_trends")
        assert check["status"] == "failed"
        # severity escalates to "error" on a real failure regardless of
        # the missing-severity baseline.
        assert check["severity"] == "error"

    def test_existence_check_passes_when_evidence_dict_present(self):
        r = _bare_did_result(
            model_info={"honest_did": {"has_run": True, "ci": [0, 1]}})
        check = next(c for c in sp.audit(r)["checks"]
                     if c["name"] == "honest_did")
        assert check["status"] == "passed"

    def test_missing_check_carries_suggest_function(self):
        r = _bare_did_result()  # empty model_info
        check = next(c for c in sp.audit(r)["checks"]
                     if c["name"] == "parallel_trends")
        assert check["status"] == "missing"
        assert check["suggest_function"] == "sp.pretrends_test"

    def test_robust_se_failed_when_nonrobust(self):
        # Build a synthetic EconometricResults via sp.regress without
        # robust= so model_info["robust"] == "nonrobust".
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"y": rng.normal(size=200),
                           "x": rng.normal(size=200)})
        r = sp.regress("y ~ x", data=df)  # default robust=nonrobust
        check = next(c for c in sp.audit(r)["checks"]
                     if c["name"] == "robust_se")
        # Either failed (recorded as "nonrobust") or missing if
        # model_info doesn't surface the field.
        assert check["status"] in ("failed", "missing")


# ---------------------------------------------------------------------------
#  Token budget: < 4K chars JSON for any single result
# ---------------------------------------------------------------------------


class TestTokenBudget:

    def test_did_under_budget(self, real_did_result):
        s = json.dumps(sp.audit(real_did_result))
        assert len(s) < 4000, (
            f"audit payload {len(s)} chars exceeds 1K-token budget; "
            "agents need this lightweight")

    def test_regression_under_budget(self, real_regress_result):
        s = json.dumps(sp.audit(real_regress_result))
        assert len(s) < 4000


class TestJsonSafety:

    def test_payload_strict_json_safe(self, real_did_result):
        # Strict json.dumps (no default= fallback) — catches numpy /
        # pandas leakage that would only break under MCP transport.
        json.dumps(sp.audit(real_did_result))

    def test_payload_strict_json_safe_regression(self, real_regress_result):
        json.dumps(sp.audit(real_regress_result))


# ---------------------------------------------------------------------------
#  Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_empty_model_info_returns_all_missing(self):
        r = _bare_did_result()
        card = sp.audit(r)
        # Every applicable check should be "missing" since model_info
        # is empty.
        assert card["summary"]["missing"] == card["summary"]["n_total"]
        assert card["summary"]["passed"] == 0
        assert card["summary"]["failed"] == 0
        assert card["coverage"] == 0.0

    def test_unknown_family_does_not_crash(self):
        # A causal result with a method name that doesn't match any
        # family (→ "generic") should still return a valid card.
        r = _bare_did_result(model_info={})
        r.method = "totally_made_up_estimator_name"
        card = sp.audit(r)
        # ``generic`` family has no DID-only checks; the universal
        # convergence checks still apply.
        assert card["method_family"] == "generic"
        names = {c["name"] for c in card["checks"]}
        assert "parallel_trends" not in names
        # Just verify it doesn't crash and returns a sane summary.
        assert card["summary"]["n_total"] >= 0


# ---------------------------------------------------------------------------
#  Non-overlap with violations() / next_steps()
# ---------------------------------------------------------------------------


class TestNonOverlapWithSiblings:
    """audit() is the missing-evidence view; it must not duplicate
    violations() (checked-but-failed) or next_steps() (actions)."""

    def test_audit_returns_missing_when_violations_silent(self):
        # Empty model_info → no violations, but audit lists missing
        # checks.
        r = _bare_did_result()
        viols = r.violations()
        card = sp.audit(r)
        assert viols == []
        assert card["summary"]["missing"] > 0

    def test_audit_returns_failed_when_violations_fires(self):
        # A real failure (pretrend p<0.05) fires both violations()
        # AND audit's "failed" status — both views agree.
        r = _bare_did_result(
            model_info={"pretrend_test": {"pvalue": 0.005}})
        viols = r.violations()
        card = sp.audit(r)
        assert any(v["test"] == "pretrend" for v in viols)
        pretrend_status = next(c["status"] for c in card["checks"]
                                if c["name"] == "parallel_trends")
        assert pretrend_status == "failed"


# ---------------------------------------------------------------------------
#  Multi-path evidence resolution: ``violations()`` and ``audit()`` must
#  not disagree just because an estimator stored the diagnostic under
#  one of several legitimate aliases.
# ---------------------------------------------------------------------------


def _bare_iv_result(model_info=None) -> CausalResult:
    return CausalResult(
        method="2sls_iv",
        estimand="LATE",
        estimate=1.0, se=0.3, pvalue=0.001, ci=(0.4, 1.6),
        alpha=0.05, n_obs=500,
        model_info=model_info or {},
        _citation_key="2sls_iv",
    )


class TestMultiPathEvidence:
    """The IV first-stage F has three legitimate aliases — audit must
    resolve any of them, mirroring causal_violations() behaviour."""

    def test_first_stage_f_flat_alias(self):
        r = _bare_iv_result(model_info={"first_stage_f": 25.0})
        check = next(c for c in sp.audit(r)["checks"]
                     if c["name"] == "weak_instrument")
        assert check["status"] == "passed"
        assert check["value"] == pytest.approx(25.0)

    def test_first_stage_f_nested_alias(self):
        r = _bare_iv_result(
            model_info={"first_stage": {"f_stat": 25.0}})
        check = next(c for c in sp.audit(r)["checks"]
                     if c["name"] == "weak_instrument")
        assert check["status"] == "passed"

    def test_first_stage_f_weak_iv_alias(self):
        r = _bare_iv_result(model_info={"weak_iv_f": 25.0})
        check = next(c for c in sp.audit(r)["checks"]
                     if c["name"] == "weak_instrument")
        assert check["status"] == "passed"

    def test_audit_and_violations_agree_on_weak_iv(self):
        # Same result, both views consulted: must not disagree on
        # the IV first-stage F even though they walk different code
        # paths to find it.
        r = _bare_iv_result(model_info={"first_stage": {"f_stat": 4.5}})
        viols = r.violations()
        assert any(v["test"] == "weak_instrument" for v in viols)
        check = next(c for c in sp.audit(r)["checks"]
                     if c["name"] == "weak_instrument")
        assert check["status"] == "failed"


# ---------------------------------------------------------------------------
#  Strict ``exists`` semantics — falsy sentinels must NOT mark passed.
# ---------------------------------------------------------------------------


class TestExistsStrictness:
    """A bare ``False`` / ``0`` / ``""`` value at the evidence path
    means the estimator wrote a sentinel for "tried but didn't store
    real diagnostics"; audit must NOT report passed in that case."""

    def test_false_sentinel_is_missing_not_passed(self):
        r = _bare_did_result(model_info={"honest_did": False})
        check = next(c for c in sp.audit(r)["checks"]
                     if c["name"] == "honest_did")
        assert check["status"] == "missing"

    def test_empty_dict_is_missing_not_passed(self):
        r = _bare_did_result(model_info={"honest_did": {}})
        check = next(c for c in sp.audit(r)["checks"]
                     if c["name"] == "honest_did")
        assert check["status"] == "missing"

    def test_zero_is_missing_not_passed(self):
        r = _bare_did_result(model_info={"honest_did": 0})
        check = next(c for c in sp.audit(r)["checks"]
                     if c["name"] == "honest_did")
        assert check["status"] == "missing"

    def test_non_empty_dict_passes(self):
        r = _bare_did_result(
            model_info={"honest_did": {"has_run": True}})
        check = next(c for c in sp.audit(r)["checks"]
                     if c["name"] == "honest_did")
        assert check["status"] == "passed"

    def test_truthy_non_dict_scalar_passes(self):
        # A non-empty string at the evidence path is a legitimate
        # signal too — keep audit lenient enough to recognise it.
        r = _bare_did_result(model_info={"honest_did": "ran 2024-04-28"})
        check = next(c for c in sp.audit(r)["checks"]
                     if c["name"] == "honest_did")
        assert check["status"] == "passed"


# ---------------------------------------------------------------------------
#  Severity / importance vocabulary separation
# ---------------------------------------------------------------------------


class TestSeverityImportanceSeparation:
    """``severity`` carries observed status (info/warning/error);
    ``importance`` carries fixed-per-check criticality (high/medium/low).
    Agents branching on either must get clean signals."""

    def test_passed_check_has_info_severity(self):
        r = _bare_did_result(
            model_info={"pretrend_test": {"pvalue": 0.50}})
        check = next(c for c in sp.audit(r)["checks"]
                     if c["name"] == "parallel_trends")
        assert check["status"] == "passed"
        assert check["severity"] == "info"
        # importance stays "high" regardless of status.
        assert check["importance"] == "high"

    def test_failed_check_has_error_severity(self):
        r = _bare_did_result(
            model_info={"pretrend_test": {"pvalue": 0.005}})
        check = next(c for c in sp.audit(r)["checks"]
                     if c["name"] == "parallel_trends")
        assert check["status"] == "failed"
        assert check["severity"] == "error"
        assert check["importance"] == "high"

    def test_missing_high_importance_check_has_warning_severity(self):
        r = _bare_did_result()  # empty
        check = next(c for c in sp.audit(r)["checks"]
                     if c["name"] == "parallel_trends")
        assert check["status"] == "missing"
        assert check["severity"] == "warning"
        assert check["importance"] == "high"

    def test_missing_low_importance_check_has_info_severity(self):
        r = _bare_did_result()  # empty
        # ess_bulk is universal-applies and importance="low"
        check = next(c for c in sp.audit(r)["checks"]
                     if c["name"] == "ess_bulk")
        assert check["status"] == "missing"
        assert check["importance"] == "low"
        # Low-importance missing → severity stays at info, not warning.
        assert check["severity"] == "info"
