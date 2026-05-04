"""Tests for the workflow-layer degradation tracking surface.

CLAUDE.md §3.7 ("失败要响亮") forbids silent ``except Exception: pass``
in best-effort orchestration code.  These tests pin down the contract:
when a sub-step of ``sp.paper`` / ``sp.causal_workflow`` /
``sp.assumption_audit`` fails, the user must see a
``WorkflowDegradedWarning`` *and* be able to introspect what was skipped
via ``draft.degradations`` / ``workflow.degradations``.

The tests deliberately trigger failures by feeding pathological inputs
(non-numeric covariates, CI shaped wrong, ``cite()`` raising, etc.)
rather than by mocking, so they double as regression tests against
future code paths that try to silently swallow these errors again.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from statspai.workflow._degradation import (
    WorkflowDegradedWarning,
    record_degradation,
)


# --------------------------------------------------------------------- #
#  Helper itself
# --------------------------------------------------------------------- #


def test_record_degradation_warns_and_returns_entry():
    bag: list = []
    with pytest.warns(WorkflowDegradedWarning, match="custom section"):
        entry = record_degradation(
            bag,
            section="custom section",
            exc=ValueError("boom"),
            detail="ctx=1",
        )
    assert entry["section"] == "custom section"
    assert entry["error_type"] == "ValueError"
    assert entry["message"] == "boom"
    assert entry["detail"] == "ctx=1"
    assert bag == [entry]


def test_record_degradation_appends_to_object_attr():
    class Holder:
        def __init__(self):
            self.degradations: list = []

    h = Holder()
    with pytest.warns(WorkflowDegradedWarning):
        record_degradation(h, section="x", exc=RuntimeError("y"))
    assert len(h.degradations) == 1
    assert h.degradations[0]["error_type"] == "RuntimeError"


def test_record_degradation_tolerates_no_target():
    """Passing target=None still warns; just nowhere to record."""
    with pytest.warns(WorkflowDegradedWarning):
        record_degradation(None, section="loose", exc=KeyError("k"))


# --------------------------------------------------------------------- #
#  _eda_block — covariate balance table
# --------------------------------------------------------------------- #


def test_eda_block_records_when_balance_table_fails():
    """A non-numeric covariate breaks .mean(); the EDA section must
    still render and the failure must surface."""
    from statspai.workflow.paper import _eda_block

    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0],
        "treated": [0, 1, 0, 1],
        # ``string_col`` is not numeric → groupby().mean() raises.
        "string_col": ["a", "b", "c", "d"],
    })
    bag: list = []
    with pytest.warns(
        WorkflowDegradedWarning, match="EDA covariate balance table"
    ):
        out = _eda_block(
            df, y="y", treatment="treated",
            covariates=["string_col"], degradations=bag,
        )
    # Section still renders something useful (sample size + missingness).
    assert "Sample size" in out
    assert len(bag) == 1
    assert bag[0]["section"] == "EDA covariate balance table"
    # Partial markdown rows ("| covariate |") must NOT leak when the
    # build aborted — the dropped-table fix in _eda_block guarantees
    # all-or-nothing balance output.
    assert "| covariate |" not in out


# --------------------------------------------------------------------- #
#  _section_from_workflow — CI + Results serialisation
# --------------------------------------------------------------------- #


class _FakeResultBadCI:
    """Looks like a CausalResult but its ``ci`` attribute is malformed."""

    estimate = 1.5
    se = 0.2
    estimand = "ATT"
    ci = "not-a-tuple"   # float(ci[0]) → ValueError
    pvalue = 0.01
    n_obs = 100


class _FakeWorkflow:
    """Minimal stand-in for CausalWorkflow that _section_from_workflow needs."""

    def __init__(self, result):
        self.result = result
        self.diagnostics = type("D", (), {
            "verdict": "OK", "findings": []
        })()
        self.recommendation = None
        self.robustness_findings = {}
        self._robustness_report = None
        self.treatment = None


def test_section_from_workflow_records_ci_failure():
    from statspai.workflow.paper import _section_from_workflow

    wf = _FakeWorkflow(_FakeResultBadCI())
    bag: list = []
    with pytest.warns(WorkflowDegradedWarning, match="95% CI rendering"):
        sections = _section_from_workflow(wf, degradations=bag)
    # Estimate + SE still rendered (since float(estimate) succeeds).
    assert "1.5000" in sections["Results"]
    # CI line should be missing.
    assert "95% CI" not in sections["Results"]
    assert any("CI" in d["section"] for d in bag)


# --------------------------------------------------------------------- #
#  paper() top-level — DAG + citation + provenance
# --------------------------------------------------------------------- #


def _make_observational_df(seed: int = 42, n: int = 200):
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    treat = (0.5 * x1 + 0.3 * x2 + rng.standard_normal(n) > 0).astype(int)
    wage = (1.2 * treat + 0.4 * x1 + 0.3 * x2 + rng.standard_normal(n))
    return pd.DataFrame({
        "wage": wage, "trained": treat, "edu": x1, "experience": x2,
    })


class _ExplodingDAG:
    """Object that quacks like a DAG but raises during render.

    ``_render_dag_section`` is intentionally tolerant of missing
    methods (it falls back via ``getattr`` defaults), so we have to
    blow up at the very first access — ``nodes`` is the first thing
    the renderer touches after the ``dag is None`` short-circuit.
    """

    edges = [("trained", "wage")]

    @property
    def nodes(self):
        raise RuntimeError("simulated DAG render failure")


def test_paper_records_dag_render_failure():
    """Passing a DAG that explodes must surface the failure, not vanish."""
    import statspai as sp

    df = _make_observational_df()
    bag_seen: list = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        draft = sp.paper(
            df, "effect of trained on wage",
            covariates=["edu", "experience"],
            dag=_ExplodingDAG(),
        )
        bag_seen.extend(draft.degradations)

    # At least one WorkflowDegradedWarning fired specifically for the
    # DAG appendix.
    dag_warnings = [
        w for w in caught
        if issubclass(w.category, WorkflowDegradedWarning)
        and "DAG" in str(w.message)
    ]
    assert dag_warnings, "Expected a DAG-related WorkflowDegradedWarning"
    assert any(
        "DAG" in d.get("section", "") for d in bag_seen
    ), f"Expected DAG section in degradations, got {bag_seen}"
    # Draft is still produced and renderable.
    assert "## Question" in draft.to_markdown()


# --------------------------------------------------------------------- #
#  CausalWorkflow.run(full=True) — sub-stage failure
# --------------------------------------------------------------------- #


def test_causal_workflow_run_records_substage_failure(monkeypatch):
    """A broken compare_estimators must surface as a degradation, not silence."""
    import statspai as sp
    from statspai.workflow.causal_workflow import CausalWorkflow

    df = _make_observational_df()

    def _explode(self, *a, **kw):
        raise RuntimeError("synthetic compare failure")

    monkeypatch.setattr(CausalWorkflow, "compare_estimators", _explode)

    with pytest.warns(WorkflowDegradedWarning, match="compare_estimators"):
        wf = sp.causal(
            df, y="wage", treatment="trained",
            covariates=["edu", "experience"],
            design="rct",  # cheapest path
        ).run(full=True)

    sections = [d["section"] for d in wf.degradations]
    assert any("compare_estimators" in s for s in sections), sections
    # The legacy free-text pipeline_notes channel must still fire too,
    # so the rendered HTML / markdown report still surfaces it.
    assert any("compare_estimators" in n for n in wf.pipeline_notes)


# --------------------------------------------------------------------- #
#  assumption_audit — Oster bounds / VIF / RESET / BP failure
# --------------------------------------------------------------------- #


def test_assumption_audit_records_underlying_failure(monkeypatch):
    """When a per-assumption test raises, audit must fire warnings AND
    append an inconclusive ``AssumptionCheck`` (not silently drop it)."""
    import statspai as sp

    df = _make_observational_df()
    res = sp.regress("wage ~ trained + edu + experience", data=df)

    # Force every backing test to blow up so we see all four sites fire.
    def _boom(*a, **kw):
        raise RuntimeError("synthetic assumption failure")

    for name in ("reset_test", "het_test", "vif", "oster_bounds"):
        monkeypatch.setattr(sp, name, _boom, raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        audit = sp.assumption_audit(res, verbose=False)

    degraded = [
        w for w in caught
        if issubclass(w.category, WorkflowDegradedWarning)
        and "_audit_linear" in str(w.message)
    ]
    # All four backing tests should have fired a degradation warning.
    assert len(degraded) >= 4, f"Got {len(degraded)} degraded warnings"

    # Each broken assumption should still surface as an inconclusive
    # check (passed=None) with the actual error message in detail.
    inconclusive = [c for c in audit.checks if c.passed is None]
    assumption_names = {c.assumption for c in inconclusive}
    assert {"Linearity", "Homoskedasticity",
            "No multicollinearity", "Robustness to unobservables"} \
        <= assumption_names
    for c in inconclusive:
        assert "RuntimeError" in c.detail, c.detail
