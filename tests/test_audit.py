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
        estimate=1.5,
        se=0.5,
        pvalue=0.003,
        ci=(0.5, 2.5),
        alpha=0.05,
        n_obs=1000,
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
            y = 1.0 + 0.3 * t + 0.5 * tr + 2.0 * tr * t + rng.normal(scale=0.5)
            rows.append({"i": i, "t": t, "treated": tr, "post": t, "y": y})
    df = pd.DataFrame(rows)
    return sp.did(df, y="y", treat="treated", time="t", post="post")


@pytest.fixture(scope="module")
def real_regress_result():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=300),
            "x": rng.normal(size=300),
        }
    )
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
        for k in ("method", "method_family", "checks", "summary", "coverage"):
            assert k in card

    def test_summary_keys(self, real_did_result):
        s = sp.audit(real_did_result)["summary"]
        for k in ("passed", "failed", "missing", "n_total"):
            assert k in s
        assert s["passed"] + s["failed"] + s["missing"] == s["n_total"]

    def test_check_shape(self, real_did_result):
        for c in sp.audit(real_did_result)["checks"]:
            for k in (
                "name",
                "question",
                "status",
                "severity",
                "importance",
                "value",
                "threshold",
                "suggest_function",
                "rationale",
            ):
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
        assert sp.audit(real_regress_result)["method_family"] == "regression"

    def test_did_includes_parallel_trends(self):
        names = {c["name"] for c in sp.audit(_bare_did_result())["checks"]}
        assert "parallel_trends" in names

    def test_did_excludes_rd_only_checks(self):
        names = {c["name"] for c in sp.audit(_bare_did_result())["checks"]}
        # mccrary / bandwidth_sensitivity / placebo_cutoff are RD-only.
        for rd_only in ("mccrary_density", "bandwidth_sensitivity", "placebo_cutoff"):
            assert rd_only not in names

    def test_regression_excludes_did_checks(self, real_regress_result):
        names = {c["name"] for c in sp.audit(real_regress_result)["checks"]}
        for did_only in ("parallel_trends", "honest_did", "bacon_decomposition"):
            assert did_only not in names

    def test_regression_includes_robust_se_check(self, real_regress_result):
        names = {c["name"] for c in sp.audit(real_regress_result)["checks"]}
        assert "robust_se" in names

    def test_observational_regression_checks_require_declared_treatment(
        self, real_regress_result
    ):
        descriptive = {c["name"] for c in sp.audit(real_regress_result)["checks"]}
        causal = {
            c["name"] for c in sp.audit(real_regress_result, treatment="x")["checks"]
        }

        treatment_only = {"overlap", "balance_after", "ovb_sensitivity"}
        assert descriptive.isdisjoint(treatment_only)
        assert treatment_only <= causal


# ---------------------------------------------------------------------------
#  Status semantics: passed / failed / missing
# ---------------------------------------------------------------------------


class TestPassedFailedMissing:
    def test_pretrend_passes_when_pvalue_above_threshold(self):
        r = _bare_did_result(model_info={"pretrend_test": {"pvalue": 0.50}})
        check = next(c for c in sp.audit(r)["checks"] if c["name"] == "parallel_trends")
        assert check["status"] == "passed"
        assert check["value"] == pytest.approx(0.50)

    def test_pretrend_fails_when_pvalue_below_threshold(self):
        r = _bare_did_result(model_info={"pretrend_test": {"pvalue": 0.005}})
        check = next(c for c in sp.audit(r)["checks"] if c["name"] == "parallel_trends")
        assert check["status"] == "failed"
        # severity escalates to "error" on a real failure regardless of
        # the missing-severity baseline.
        assert check["severity"] == "error"

    def test_existence_check_passes_when_evidence_dict_present(self):
        r = _bare_did_result(model_info={"honest_did": {"has_run": True, "ci": [0, 1]}})
        check = next(c for c in sp.audit(r)["checks"] if c["name"] == "honest_did")
        assert check["status"] == "passed"

    def test_missing_check_carries_suggest_function(self):
        r = _bare_did_result()  # empty model_info
        check = next(c for c in sp.audit(r)["checks"] if c["name"] == "parallel_trends")
        assert check["status"] == "missing"
        assert check["suggest_function"] == "sp.pretrends_test"

    def test_robust_se_failed_when_nonrobust(self):
        # Build a synthetic EconometricResults via sp.regress without
        # robust= so model_info["robust"] == "nonrobust".
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"y": rng.normal(size=200), "x": rng.normal(size=200)})
        r = sp.regress("y ~ x", data=df)  # default robust=nonrobust
        check = next(c for c in sp.audit(r)["checks"] if c["name"] == "robust_se")
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
            "agents need this lightweight"
        )

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
        r = _bare_did_result(model_info={"pretrend_test": {"pvalue": 0.005}})
        viols = r.violations()
        card = sp.audit(r)
        assert any(v["test"] == "pretrend" for v in viols)
        pretrend_status = next(
            c["status"] for c in card["checks"] if c["name"] == "parallel_trends"
        )
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
        estimate=1.0,
        se=0.3,
        pvalue=0.001,
        ci=(0.4, 1.6),
        alpha=0.05,
        n_obs=500,
        model_info=model_info or {},
        _citation_key="2sls_iv",
    )


class TestMultiPathEvidence:
    """The IV first-stage F has three legitimate aliases — audit must
    resolve any of them, mirroring causal_violations() behaviour."""

    def test_first_stage_f_flat_alias(self):
        r = _bare_iv_result(model_info={"first_stage_f": 25.0})
        check = next(c for c in sp.audit(r)["checks"] if c["name"] == "weak_instrument")
        assert check["status"] == "passed"
        assert check["value"] == pytest.approx(25.0)

    def test_first_stage_f_nested_alias(self):
        r = _bare_iv_result(model_info={"first_stage": {"f_stat": 25.0}})
        check = next(c for c in sp.audit(r)["checks"] if c["name"] == "weak_instrument")
        assert check["status"] == "passed"

    def test_first_stage_f_weak_iv_alias(self):
        r = _bare_iv_result(model_info={"weak_iv_f": 25.0})
        check = next(c for c in sp.audit(r)["checks"] if c["name"] == "weak_instrument")
        assert check["status"] == "passed"

    def test_audit_and_violations_agree_on_weak_iv(self):
        # Same result, both views consulted: must not disagree on
        # the IV first-stage F even though they walk different code
        # paths to find it.
        r = _bare_iv_result(model_info={"first_stage": {"f_stat": 4.5}})
        viols = r.violations()
        assert any(v["test"] == "weak_instrument" for v in viols)
        check = next(c for c in sp.audit(r)["checks"] if c["name"] == "weak_instrument")
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
        check = next(c for c in sp.audit(r)["checks"] if c["name"] == "honest_did")
        assert check["status"] == "missing"

    def test_empty_dict_is_missing_not_passed(self):
        r = _bare_did_result(model_info={"honest_did": {}})
        check = next(c for c in sp.audit(r)["checks"] if c["name"] == "honest_did")
        assert check["status"] == "missing"

    def test_zero_is_missing_not_passed(self):
        r = _bare_did_result(model_info={"honest_did": 0})
        check = next(c for c in sp.audit(r)["checks"] if c["name"] == "honest_did")
        assert check["status"] == "missing"

    def test_non_empty_dict_passes(self):
        r = _bare_did_result(model_info={"honest_did": {"has_run": True}})
        check = next(c for c in sp.audit(r)["checks"] if c["name"] == "honest_did")
        assert check["status"] == "passed"

    def test_truthy_non_dict_scalar_passes(self):
        # A non-empty string at the evidence path is a legitimate
        # signal too — keep audit lenient enough to recognise it.
        r = _bare_did_result(model_info={"honest_did": "ran 2024-04-28"})
        check = next(c for c in sp.audit(r)["checks"] if c["name"] == "honest_did")
        assert check["status"] == "passed"


# ---------------------------------------------------------------------------
#  Severity / importance vocabulary separation
# ---------------------------------------------------------------------------


class TestSeverityImportanceSeparation:
    """``severity`` carries observed status (info/warning/error);
    ``importance`` carries fixed-per-check criticality (high/medium/low).
    Agents branching on either must get clean signals."""

    def test_passed_check_has_info_severity(self):
        r = _bare_did_result(model_info={"pretrend_test": {"pvalue": 0.50}})
        check = next(c for c in sp.audit(r)["checks"] if c["name"] == "parallel_trends")
        assert check["status"] == "passed"
        assert check["severity"] == "info"
        # importance stays "high" regardless of status.
        assert check["importance"] == "high"

    def test_failed_check_has_error_severity(self):
        r = _bare_did_result(model_info={"pretrend_test": {"pvalue": 0.005}})
        check = next(c for c in sp.audit(r)["checks"] if c["name"] == "parallel_trends")
        assert check["status"] == "failed"
        assert check["severity"] == "error"
        assert check["importance"] == "high"

    def test_missing_high_importance_check_has_warning_severity(self):
        r = _bare_did_result()  # empty
        check = next(c for c in sp.audit(r)["checks"] if c["name"] == "parallel_trends")
        assert check["status"] == "missing"
        assert check["severity"] == "warning"
        assert check["importance"] == "high"

    def test_missing_low_importance_check_has_info_severity(self):
        # A Bayesian DID result that wrote the diagnostics dict but omitted
        # ess_bulk: still missing, still low-importance, still info severity.
        r = _bare_did_result(
            model_info={
                "model_type": "Bayesian DID",
                "rhat_max": 1.001,
                # no ess_bulk_min
            }
        )
        check = next(c for c in sp.audit(r)["checks"] if c["name"] == "ess_bulk")
        assert check["status"] == "missing"
        assert check["importance"] == "low"
        # Low-importance missing → severity stays at info, not warning.
        assert check["severity"] == "info"


class TestModelTypeGatedChecks:
    """Cox / Tobit / Heckman each route to a broad family (regression /
    generic) they share with plain OLS, but carry a signature assumption the
    others must not be asked about. The ``model_type_any`` gate makes audit
    surface each proactively — ``passed`` when the assumption holds, ``failed``
    when violated — while a plain OLS never sees the survival/limited-dep
    question. Pins that audit stays a faithful, non-crying-wolf superset of
    result.violations() for these estimators."""

    @staticmethod
    def _status(result, name):
        for c in sp.audit(result)["checks"]:
            if c["name"] == name:
                return c["status"]
        return "ABSENT"

    @staticmethod
    def _is_superset(result):
        names = {c["name"] for c in sp.audit(result)["checks"]}
        return {v["test"] for v in result.violations()} <= names

    def test_cox_proportional_hazards_passed_and_failed(self):
        n = 700
        # PH holds: textbook proportional-hazards DGP.
        xg = np.random.default_rng(3).normal(size=n)
        tg = -np.log(np.random.default_rng(3).uniform(size=n)) / np.exp(0.8 * xg)
        good = sp.cox(
            data=pd.DataFrame({"t": tg + 0.001, "d": np.ones(n, int), "x": xg}),
            duration="t",
            event="d",
            x=["x"],
        )
        # PH violated: covariate trends with failure time.
        rb = np.random.default_rng(11)
        tb = np.sort(rb.exponential(1.0, n)) + 0.01
        xb = np.linspace(-2, 2, n) + rb.normal(0, 0.3, n)
        bad = sp.cox(
            data=pd.DataFrame({"t": tb, "d": np.ones(n, int), "x": xb}),
            duration="t",
            event="d",
            x=["x"],
        )
        assert self._status(good, "proportional_hazards") == "passed"
        assert self._status(bad, "proportional_hazards") == "failed"
        # No double-count when the violation also fires the fold-in.
        names = [c["name"] for c in sp.audit(bad)["checks"]]
        assert names.count("proportional_hazards") == 1
        assert self._is_superset(good) and self._is_superset(bad)

    def test_plain_ols_never_asked_proportional_hazards(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=400)
        ols = sp.regress(
            "y ~ x", data=pd.DataFrame({"y": x + rng.normal(size=400), "x": x})
        )
        assert self._status(ols, "proportional_hazards") == "ABSENT"

    def test_tobit_extreme_censoring_passed_and_failed(self):
        x = np.random.default_rng(5).normal(size=800)
        y_good = np.maximum(1 + 2 * x + np.random.default_rng(6).normal(size=800), 0)
        y_bad = np.maximum(-3 + x + np.random.default_rng(6).normal(size=800), 0)
        good = sp.tobit(pd.DataFrame({"y": y_good, "x": x}), y="y", x=["x"], ll=0)
        bad = sp.tobit(pd.DataFrame({"y": y_bad, "x": x}), y="y", x=["x"], ll=0)
        assert self._status(good, "extreme_censoring") == "passed"
        assert self._status(bad, "extreme_censoring") == "failed"
        assert self._is_superset(bad)

    def test_heckman_rho_boundary_passed_and_failed(self):
        r5 = np.random.default_rng(5)
        m = 2000
        zc, xc, uc = r5.normal(size=m), r5.normal(size=m), r5.normal(size=m)
        eps = 0.6 * uc + np.sqrt(1 - 0.36) * r5.normal(size=m)  # rho ~ 0.6, interior
        selc = 0.3 + zc + 0.5 * xc + uc > 0
        yc = 1 + 2 * xc + 3 * eps
        gdf = pd.DataFrame(
            {"y": np.where(selc, yc, np.nan), "x": xc, "z": zc, "s": selc.astype(int)}
        )
        good = sp.heckman(gdf, y="y", x=["x"], select="s", z=["z"])
        r9 = np.random.default_rng(9)
        k = 600
        zb, xb, ub = r9.normal(size=k), r9.normal(size=k), r9.normal(size=k)
        selb = 0.5 + 0.8 * zb + ub > 0
        yb = 1 + 2 * xb + 3 * ub  # outcome error == selection error => rho -> 1
        bdf = pd.DataFrame(
            {"y": np.where(selb, yb, np.nan), "x": xb, "z": zb, "s": selb.astype(int)}
        )
        bad = sp.heckman(bdf, y="y", x=["x"], select="s", z=["z"])
        assert self._status(good, "heckman_rho_boundary") == "passed"
        assert self._status(bad, "heckman_rho_boundary") == "failed"
        assert self._is_superset(bad)


class TestMcmcGate:
    """MCMC convergence checks (rhat / ess) only apply when the result could
    actually report them. On a frequentist MLE that has no sampler, "missing
    convergence_rhat" is noise — a reviewer asked "has your OLS converged?"
    is being told something that doesn't exist. The gate keeps the check
    *live* for Bayesian results (so an actually-converged / non-converged
    verdict is still surfaced) and silences it for everyone else."""

    @staticmethod
    def _names(result):
        return {c["name"] for c in sp.audit(result)["checks"]}

    def test_frequentist_ols_has_no_mcmc_checks(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=400)
        ols = sp.regress(
            "y ~ x", data=pd.DataFrame({"y": x + rng.normal(size=400), "x": x})
        )
        names = self._names(ols)
        assert "convergence_rhat" not in names
        assert "ess_bulk" not in names

    def test_frequentist_tobit_has_no_mcmc_checks(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=400)
        y = np.maximum(1 + 2 * x + rng.normal(size=400), 0)
        tb = sp.tobit(pd.DataFrame({"y": y, "x": x}), y="y", x=["x"], ll=0)
        names = self._names(tb)
        assert "convergence_rhat" not in names
        assert "ess_bulk" not in names

    def test_bayesian_did_surfaces_mcmc_checks(self):
        # A DID result that carries rhat evidence should keep the MCMC checks
        # surfaced (a high rhat would then be a real "failed" finding).
        r = _bare_did_result(
            model_info={
                "model_type": "Bayesian DID",
                "rhat_max": 1.001,
                "ess_bulk_min": 800,
            }
        )
        names = self._names(r)
        assert "convergence_rhat" in names
        assert "ess_bulk" in names
        rhat = next(c for c in sp.audit(r)["checks"] if c["name"] == "convergence_rhat")
        ess = next(c for c in sp.audit(r)["checks"] if c["name"] == "ess_bulk")
        assert rhat["status"] == "passed"
        assert ess["status"] == "passed"

    def test_bayesian_method_signature_unlocks_mcmc(self):
        # A result with no model_info rhat but a "Bayesian" method label
        # is also treated as Bayesian — the signature alone is sufficient.
        r = _bare_did_result(model_info={})
        # Override method to a Bayesian signature without touching model_info
        r.method = "Bayesian DiD"
        r.model_info["model_type"] = "Bayesian DiD"
        names = self._names(r)
        assert "convergence_rhat" in names
        assert "ess_bulk" in names


class TestRequiresEvidenceGate:
    """The requires_evidence / requires_signature gates let a check be
    scoped to a regime whose data may or may not be present. The
    motivating use is MCMC convergence (rhat/ess only on Bayesian
    results) but the same mechanism must work for any future
    "this assumption only matters when the model collected X" check.
    Pins the OR semantics (evidence OR signature), the diagnostic
    sub-dict fallback, and the empty-gate pass-through."""

    def test_evidence_gate_fires_on_stored_key(self):
        # n_clusters is the canonical example: a "panel few-clusters"
        # check should be silent on a plain OLS that never recorded one.
        from statspai.smart.audit import _check_evidence_predicate

        assert _check_evidence_predicate(("n_clusters",), {"n_clusters": 12}, {})
        assert not _check_evidence_predicate(("n_clusters",), {"x": 1}, {})

    def test_evidence_gate_falls_back_to_diagnostics(self):
        # panel OLS exposes n_clusters in diagnostics (not model_info);
        # the gate must look at the merged view, not just model_info.
        from statspai.smart.audit import _check_evidence_predicate

        assert _check_evidence_predicate(("n_clusters",), {}, {"n_clusters": 12})

    def test_evidence_gate_empty_is_no_op(self):
        from statspai.smart.audit import _check_evidence_predicate

        assert _check_evidence_predicate((), {"x": 1}, {})  # pass-through

    def test_signature_gate_substring_match(self):
        from statspai.smart.audit import _check_signature_predicate

        assert _check_signature_predicate(("bayes", "mcmc"), "bayesian did")
        assert _check_signature_predicate(("bayes", "mcmc"), "MCMC sampler")
        assert not _check_signature_predicate(("bayes", "mcmc"), "did_2x2")
        assert _check_signature_predicate((), "anything")  # pass-through
