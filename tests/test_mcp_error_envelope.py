"""Structured ``StatsPAIError`` payloads must survive the MCP transport.

The ``execute_tool`` error envelope and the MCP server's
``tools/call`` response should carry ``error_kind``, ``recovery_hint``,
``diagnostics`` and ``alternative_functions`` so an LLM-driven agent
can branch on the failure programmatically — instead of regex-parsing
the free-text ``error`` message.

These tests pin the contract so future refactors of the agent layer
don't accidentally drop the structured payload.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from statspai.agent import execute_tool
from statspai.agent.mcp_server import handle_request
from statspai.exceptions import StatsPAIError


# ---------------------------------------------------------------------------
#  Fixtures: a dataset that triggers MethodIncompatibility from sp.did
# ---------------------------------------------------------------------------

@pytest.fixture
def under_identified_iv_df() -> pd.DataFrame:
    """Two endogenous regressors with only one instrument — ``sp.ivreg``
    raises ``MethodIncompatibility`` with a structured payload."""
    rng = np.random.default_rng(0)
    n = 200
    z = rng.normal(size=n)
    d1 = 0.5 * z + rng.normal(size=n)
    d2 = rng.normal(size=n)
    y = d1 + d2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d1": d1, "d2": d2, "z": z})


# ---------------------------------------------------------------------------
#  execute_tool: legacy fields preserved + structured fields added
# ---------------------------------------------------------------------------

class TestExecuteToolStructuredError:

    def test_legacy_fields_still_present(self, under_identified_iv_df):
        out = execute_tool(
            "ivreg",
            {"formula": "y ~ (d1 + d2 ~ z)"},
            data=under_identified_iv_df,
        )
        # Backward compat: pre-existing fields remain unchanged.
        for k in ("error", "tool", "arguments", "remediation"):
            assert k in out, f"legacy field {k!r} missing"
        assert "MethodIncompatibility" in out["error"]
        assert out["tool"] == "ivreg"

    def test_error_kind_surfaced(self, under_identified_iv_df):
        out = execute_tool(
            "ivreg",
            {"formula": "y ~ (d1 + d2 ~ z)"},
            data=under_identified_iv_df,
        )
        # New: agents can branch on the typed code without parsing
        # free-text error messages.
        assert out["error_kind"] == "method_incompatibility"

    def test_error_payload_has_full_to_dict_shape(self, under_identified_iv_df):
        out = execute_tool(
            "ivreg",
            {"formula": "y ~ (d1 + d2 ~ z)"},
            data=under_identified_iv_df,
        )
        payload = out["error_payload"]
        # Mirrors StatsPAIError.to_dict() exactly.
        for k in ("kind", "class", "message", "recovery_hint",
                  "diagnostics", "alternative_functions"):
            assert k in payload, f"error_payload missing {k!r}"
        assert payload["kind"] == "method_incompatibility"
        assert payload["class"] == "MethodIncompatibility"

    def test_diagnostics_round_trip(self, under_identified_iv_df):
        out = execute_tool(
            "ivreg",
            {"formula": "y ~ (d1 + d2 ~ z)"},
            data=under_identified_iv_df,
        )
        diag = out["error_payload"]["diagnostics"]
        # The IV migration sets n_instruments < n_endogenous diagnostics.
        assert diag.get("n_instruments") == 1
        assert diag.get("n_endogenous") == 2

    def test_alternative_functions_round_trip(self, under_identified_iv_df):
        out = execute_tool(
            "ivreg",
            {"formula": "y ~ (d1 + d2 ~ z)"},
            data=under_identified_iv_df,
        )
        alts = out["error_payload"]["alternative_functions"]
        assert isinstance(alts, list)
        assert "sp.bounds" in alts

    def test_recovery_hint_present(self, under_identified_iv_df):
        out = execute_tool(
            "ivreg",
            {"formula": "y ~ (d1 + d2 ~ z)"},
            data=under_identified_iv_df,
        )
        hint = out["error_payload"]["recovery_hint"]
        assert isinstance(hint, str)
        assert hint  # non-empty

    def test_envelope_is_json_serialisable(self, under_identified_iv_df):
        out = execute_tool(
            "ivreg",
            {"formula": "y ~ (d1 + d2 ~ z)"},
            data=under_identified_iv_df,
        )
        # Strict serialisation (no ``default=`` fallback) — catches any
        # numpy / pandas leakage that would only surface inside the MCP
        # round-trip otherwise.
        s = json.dumps(out)
        reloaded = json.loads(s)
        assert reloaded["error_kind"] == "method_incompatibility"

    def test_non_statspai_error_omits_structured_fields(self):
        # KeyError from a missing column is NOT a StatsPAIError → no
        # error_kind / error_payload should appear (so agents using
        # ``"error_kind" in out`` for branching get a clean signal).
        df = pd.DataFrame({"y": [1.0, 2.0], "treated": [0, 1],
                           "t": [0, 1]})
        out = execute_tool(
            "regress",
            {"formula": "y ~ does_not_exist"},
            data=df,
        )
        assert "error" in out
        assert "error_kind" not in out
        assert "error_payload" not in out


# ---------------------------------------------------------------------------
#  Subclass coverage: every concrete StatsPAIError code surfaces correctly
# ---------------------------------------------------------------------------

class TestAllStatsPAIErrorSubclassesRoundTrip:
    """Each concrete ``StatsPAIError`` subclass must surface its own
    ``code`` through ``execute_tool`` regardless of what specific
    estimator raised it. We monkey-patch a fake tool into the registry
    so this test stays decoupled from estimator-side migrations."""

    @pytest.fixture
    def patched_tool(self, monkeypatch):
        from statspai.agent import tools as agent_tools

        # Marker so we can clean up cleanly even if the test errors.
        fake_spec = {
            'name': '_fake_error_tool',
            'description': 'test-only tool that raises a chosen error',
            'input_schema': {'type': 'object', 'properties': {},
                             'required': []},
            'statspai_fn': '_fake_error_tool',
            'serializer': lambda r: r,
        }
        agent_tools.TOOL_REGISTRY.append(fake_spec)

        def _restore():
            agent_tools.TOOL_REGISTRY.remove(fake_spec)
        monkeypatch.setattr(agent_tools, '_resolve_fn',
                            lambda fn_name: _raiser_factory.fn)
        yield
        _restore()

    @pytest.mark.parametrize("cls_name,expected_code,extra", [
        ("AssumptionViolation", "assumption_violation",
         {"diagnostics": {"pretrend_pvalue": 0.001}}),
        ("IdentificationFailure", "identification_failure",
         {"alternative_functions": ["sp.bounds"]}),
        ("DataInsufficient", "data_insufficient",
         {"diagnostics": {"n_treated": 1}}),
        ("ConvergenceFailure", "convergence_failure",
         {"diagnostics": {"rhat_max": 1.15, "ess_min": 120}}),
        ("NumericalInstability", "numerical_instability",
         {"diagnostics": {"condition_number": 1e18}}),
        ("MethodIncompatibility", "method_incompatibility",
         {"diagnostics": {"got": "continuous", "expected": "binary"}}),
    ])
    def test_subclass_round_trips(self, cls_name, expected_code, extra,
                                   patched_tool):
        from statspai import exceptions as spx
        cls = getattr(spx, cls_name)

        def _raiser(**_kwargs):
            raise cls(f"test {cls_name}", recovery_hint="try X", **extra)
        _raiser_factory.fn = _raiser

        out = execute_tool('_fake_error_tool', {})
        assert out["error_kind"] == expected_code, (
            f"{cls_name} surfaced as {out.get('error_kind')!r}, "
            f"expected {expected_code!r}"
        )
        payload = out["error_payload"]
        assert payload["class"] == cls_name
        assert payload["recovery_hint"] == "try X"
        # Any diagnostics passed in should round-trip intact.
        if "diagnostics" in extra:
            for k, v in extra["diagnostics"].items():
                assert payload["diagnostics"][k] == v
        if "alternative_functions" in extra:
            assert (payload["alternative_functions"]
                    == extra["alternative_functions"])
        # Strict JSON serialisation — no string fallback.
        json.dumps(out)


class _raiser_factory:
    """Holder so the parametrized test can swap the raise target without
    rebuilding the monkeypatch each call."""
    fn = None  # set by each test


# ---------------------------------------------------------------------------
#  Defensive: malformed diagnostics shouldn't crash the error handler
# ---------------------------------------------------------------------------

class TestMalformedDiagnosticsFallback:

    def test_unserialisable_diagnostics_degrade_gracefully(self,
                                                            monkeypatch):
        """If a caller stuffs a non-serialisable value into
        ``diagnostics`` and ``e.to_dict()`` somehow blows up, the
        envelope must still carry ``error_kind`` and a minimal
        ``error_payload`` so the agent isn't left with nothing.
        """
        from statspai.agent import tools as agent_tools
        from statspai.exceptions import AssumptionViolation

        # Build an exception whose to_dict() raises on access.
        class _BadError(AssumptionViolation):
            def to_dict(self):  # type: ignore[override]
                raise RuntimeError("intentionally broken to_dict")

        def _raiser(**_kwargs):
            raise _BadError("violation", recovery_hint="hint")

        spec = {
            'name': '_bad_tool',
            'description': 'broken to_dict',
            'input_schema': {'type': 'object', 'properties': {},
                             'required': []},
            'statspai_fn': '_bad_tool',
            'serializer': lambda r: r,
        }
        agent_tools.TOOL_REGISTRY.append(spec)
        try:
            monkeypatch.setattr(agent_tools, '_resolve_fn',
                                lambda fn_name: _raiser)
            out = execute_tool('_bad_tool', {})
            # The error_kind still surfaces because e.code is a class
            # attribute, never raised from.
            assert out["error_kind"] == "assumption_violation"
            # Payload degrades to the minimal-but-useful shape, never
            # disappears.
            assert out["error_payload"]["kind"] == "assumption_violation"
            assert out["error_payload"]["class"] == "_BadError"
            assert "violation" in out["error_payload"]["message"]
            json.dumps(out)
        finally:
            agent_tools.TOOL_REGISTRY.remove(spec)


# ---------------------------------------------------------------------------
#  MCP server: structured error survives the JSON-RPC envelope
# ---------------------------------------------------------------------------

class TestMcpToolsCallErrorEnvelope:

    def test_tools_call_propagates_structured_error(self, tmp_path,
                                                     under_identified_iv_df):
        # Persist the IV frame as CSV — the MCP server expects a
        # ``data_path`` argument and loads the frame itself.
        csv = tmp_path / "under_identified.csv"
        under_identified_iv_df.to_csv(csv, index=False)
        request = json.dumps({
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/call",
            "params": {
                "name": "ivreg",
                "arguments": {
                    "formula": "y ~ (d1 + d2 ~ z)",
                    "data_path": str(csv),
                },
            },
        })
        response = handle_request(request)
        assert response is not None
        msg = json.loads(response)
        assert msg["id"] == 42
        assert "result" in msg
        result = msg["result"]
        # MCP wraps the tool result under a content text block.
        assert result["isError"] is True
        text = result["content"][0]["text"]
        payload = json.loads(text)
        # The structured payload must survive the MCP envelope.
        assert payload["error_kind"] == "method_incompatibility"
        assert "sp.bounds" in (
            payload["error_payload"]["alternative_functions"])

    def test_statspai_error_class_export_unchanged(self):
        # Smoke check: the bridge depends on isinstance() so the error
        # class must remain importable from the canonical path.
        import statspai
        assert hasattr(statspai, "StatsPAIError")
        assert statspai.StatsPAIError is StatsPAIError
