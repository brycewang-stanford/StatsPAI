"""Contract tests for the MCP / agent tool surface.

An LLM agent drives StatsPAI by reading the published tool manifest and
calling tools with the parameters it advertises. The contract these
tests lock down is therefore simple but load-bearing:

1.  **No advertised parameter may crash the dispatcher.** Calling a tool
    with a parameter listed in its own ``input_schema`` must never raise
    an unhandled exception nor produce a ``TypeError: got an unexpected
    keyword argument`` — the agent followed the manifest faithfully and
    deserves a structured result, not a JSON-RPC ``-32000``.

2.  **Auto / registry tools must have zero schema↔signature drift.** The
    manifest schema for an auto-dispatched tool is generated from the
    registry ``ParamSpec`` list and forwarded to ``sp.<name>(**kwargs)``;
    a ParamSpec naming an argument the function does not accept would
    crash mid-workflow.

3.  **Curated tools must not rely on transparent argument drops.** The
    dispatcher still reports any dropped advertised parameter under
    ``_unsupported_args``, but the ratchet now expects the drop set to
    stay empty.

These caught four real defects on 2026-06-21 (``detect_design`` hint
forwarding, ``sensitivity`` pointing at a non-existent callable, and the
``honest_did`` / ``spec_curve`` legacy schemas). The current contract
keeps those fixed rather than carrying historical schema exceptions. See
``plans/2026-06-21-agent-runtime-orchestrator/WORKLOG.md``.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent.pipeline_tools import PIPELINE_TOOL_NAMES
from statspai.agent.tools import TOOL_REGISTRY, execute_tool, tool_manifest
from statspai.agent.tools._dispatch import _filter_to_signature
from statspai.agent.workflow_tools import WORKFLOW_TOOL_NAMES

# Server-handled envelope keys are stripped before estimator dispatch
# (see ``mcp_server._handle_tools_call``); they are not estimator args.
ENVELOPE = {
    "data_path",
    "data_columns",
    "data_sample_n",
    "result_id",
    "as_handle",
    "detail",
    "data",
}

CURATED_REGISTRY = {t["name"] for t in TOOL_REGISTRY}
CURATED = CURATED_REGISTRY | set(WORKFLOW_TOOL_NAMES) | set(PIPELINE_TOOL_NAMES)

# Curated TOOL_REGISTRY specs whose advertised ``input_schema`` describes
# a stale / aspirational API the estimator no longer accepts. This set is
# intentionally empty: known historical offenders (``honest_did``,
# ``sensitivity``, ``spec_curve``) now advertise callable-compatible
# schemas, and ``schemas/tools.json`` is in sync. Any new entry here must
# come with a follow-up schema fix, not just a broader allowance.
KNOWN_STALE_SCHEMA = set()


def _advertised(tool: dict) -> list:
    """Advertised, non-envelope parameter names for a manifest tool."""
    schema = tool.get("input_schema") or tool.get("inputSchema") or {}
    props = schema.get("properties") or {}
    return [name for name in props if name not in ENVELOPE]


def _dummy_value(name: str, tool: dict):
    """A type-appropriate dummy for an advertised parameter."""
    schema = tool.get("input_schema") or tool.get("inputSchema") or {}
    spec = (schema.get("properties") or {}).get(name) or {}
    typ = spec.get("type")
    if typ == "array":
        return ["x1"]
    if typ in ("number", "integer"):
        return 1
    if typ == "boolean":
        return True
    if typ == "object":
        return {}
    return "x1"


@pytest.fixture(scope="module")
def fixture_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame(
        {
            "y": np.arange(n, dtype=float) + rng.normal(size=n),
            "treat": [0, 1] * (n // 2),
            "d": [0, 1] * (n // 2),
            "post": [0] * (n // 2) + [1] * (n // 2),
            "time": [0] * (n // 2) + [1] * (n // 2),
            "x1": np.linspace(0, 1, n),
            "x2": np.linspace(1, 2, n),
            "unit": list(range(n // 2)) * 2,
            "i": list(range(n // 2)) * 2,
            "g": [0, 2005] * (n // 2),
            "t": [2000] * (n // 2) + [2001] * (n // 2),
        }
    )


@pytest.fixture(scope="module")
def manifest_by_name() -> dict:
    return {t["name"]: t for t in tool_manifest()}


# ---------------------------------------------------------------------------
# Part A — auto / registry tools: static schema↔signature parity
# ---------------------------------------------------------------------------


class TestNoAutoToolDrift:
    def test_auto_tool_params_subset_of_signature(self, manifest_by_name):
        """Every advertised param of an auto tool is accepted by its fn.

        Auto tools forward ``sp.<name>(**kwargs)`` after filtering to the
        registry ParamSpec, so a param advertised but absent from the
        function signature would raise ``TypeError`` for any agent that
        supplies it.
        """
        drift = {}
        for name, tool in manifest_by_name.items():
            if name in CURATED:
                continue
            fn = getattr(sp, name, None)
            if fn is None or not callable(fn):
                drift[name] = "does not resolve to a sp.<name> callable"
                continue
            try:
                params = inspect.signature(fn).parameters
            except (TypeError, ValueError):
                continue
            if any(p.kind == p.VAR_KEYWORD for p in params.values()):
                continue
            # Call-time spellings the wrapper accepts (``@accepts_aliases``,
            # and ``vce=`` from ``@accepts_vce``) are forwarded by the
            # dispatcher too.
            accepted = set(params) | set(getattr(fn, "__statspai_aliases__", {}) or {})
            missing = sorted(set(_advertised(tool)) - accepted)
            if missing:
                drift[name] = missing
        assert drift == {}, (
            "Auto-tool schema↔signature drift (advertised params the "
            f"function cannot accept): {drift}"
        )


# ---------------------------------------------------------------------------
# Part B — curated tools: never raise, transparent degradation, ratchet
# ---------------------------------------------------------------------------


class TestCuratedToolsNeverRaise:
    def test_curated_dispatch_returns_dict_without_raising(self, fixture_frame):
        """Dispatching a curated tool with its advertised params yields a dict.

        The agent following the manifest must always get a structured
        envelope back — an error dict is fine, an unhandled exception is
        a broken contract.
        """
        manifest = {t["name"]: t for t in tool_manifest()}
        raised = {}
        for name in sorted(CURATED_REGISTRY):
            tool = manifest.get(name)
            if tool is None:
                continue
            args = {p: _dummy_value(p, tool) for p in _advertised(tool)}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = execute_tool(name, args, data=fixture_frame.copy())
            except Exception as exc:  # noqa: BLE001 - contract: must not raise
                raised[name] = f"{type(exc).__name__}: {exc}"
                continue
            if not isinstance(out, dict):
                raised[name] = f"returned {type(out).__name__}, not dict"
        assert raised == {}, f"Curated tools raised / returned non-dict: {raised}"


class TestSchemaDriftRatchet:
    def test_curated_drops_match_known_stale_set(self, fixture_frame):
        """The set of curated tools that drop advertised params is ratcheted.

        After the dispatch hardening, advertised-but-unaccepted params are
        dropped (not crashed) and reported under ``_unsupported_args``.
        The current expected set is empty. Any entry is unintended drift
        that must be fixed, or explicitly added here with justification
        and a follow-up schema correction.
        """
        manifest = {t["name"]: t for t in tool_manifest()}
        dropping = set()
        for name in sorted(CURATED_REGISTRY):
            tool = manifest.get(name)
            if tool is None:
                continue
            args = {p: _dummy_value(p, tool) for p in _advertised(tool)}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = execute_tool(name, args, data=fixture_frame.copy())
            if isinstance(out, dict) and out.get("_unsupported_args"):
                dropping.add(name)
        new_drift = dropping - KNOWN_STALE_SCHEMA
        assert new_drift == set(), (
            "New curated tool(s) silently dropping advertised params "
            f"(schema drift): {sorted(new_drift)}. Fix the advertised "
            "schema (and regen schemas/*.json) or document why."
        )
        # Ratchet pressure: flag if a known-stale schema was fixed so the
        # entry can be removed from KNOWN_STALE_SCHEMA.
        assert dropping <= KNOWN_STALE_SCHEMA


class TestNoBadArgumentForAdvertisedParam:
    def test_advertised_params_never_trigger_bad_argument(self, fixture_frame):
        """No tool blames one of its *own* advertised params as unexpected.

        A ``bad_argument`` remediation naming a param the tool itself
        advertises is the signature of schema↔dispatch drift.
        """
        manifest = {t["name"]: t for t in tool_manifest()}
        offenders = {}
        for name in sorted(CURATED_REGISTRY):
            tool = manifest.get(name)
            if tool is None:
                continue
            advertised = set(_advertised(tool))
            args = {p: _dummy_value(p, tool) for p in advertised}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = execute_tool(name, args, data=fixture_frame.copy())
            if not isinstance(out, dict):
                continue
            err = str(out.get("error") or "")
            for param in advertised:
                if f"unexpected keyword argument '{param}'" in err:
                    offenders[name] = param
        assert offenders == {}, (
            "Tool(s) reject their own advertised parameter as unexpected: "
            f"{offenders}"
        )


# ---------------------------------------------------------------------------
# Regression: detect_design must honor its advertised column hints
# ---------------------------------------------------------------------------


class TestDetectDesignHints:
    def test_id_and_time_hints_are_honored(self, fixture_frame):
        out = execute_tool(
            "detect_design",
            {"id_col_hint": "unit", "time_col_hint": "time"},
            data=fixture_frame.copy(),
        )
        assert isinstance(out, dict)
        assert "error" not in out, out
        # detect_design returns a design verdict, not a TypeError envelope.
        assert any(k in out for k in ("design", "primary", "candidates", "ranked"))


# ---------------------------------------------------------------------------
# Unit: the dispatch-hardening primitives
# ---------------------------------------------------------------------------


class TestFilterToSignature:
    def test_drops_unaccepted_and_reports_them(self):
        def f(a, b=1):
            return a + b

        kept, dropped = _filter_to_signature(f, {"a": 1, "b": 2, "zzz": 9})
        assert kept == {"a": 1, "b": 2}
        assert dropped == ["zzz"]

    def test_var_keyword_passes_everything(self):
        def f(a, **kw):
            return a

        kept, dropped = _filter_to_signature(f, {"a": 1, "anything": 2})
        assert kept == {"a": 1, "anything": 2}
        assert dropped == []


class TestResolveGuardNeverRaises:
    def test_unresolvable_statspai_fn_returns_envelope(
        self, monkeypatch, fixture_frame
    ):
        """A curated spec whose statspai_fn vanishes yields an envelope.

        Guards the ``sensitivity`` crash class: a mis-pointed ``statspai_fn``
        must produce an actionable error dict, never an unhandled raise.
        """
        import statspai.agent.tools as tools_pkg

        def _boom(fn_name):
            raise ValueError(f"Tool {fn_name!r} not found on statspai.")

        # The dispatcher resolves via the parent package pointer.
        monkeypatch.setattr(tools_pkg, "_resolve_fn", _boom)
        out = execute_tool("regress", {"formula": "y ~ x1"}, data=fixture_frame.copy())
        assert isinstance(out, dict)
        assert "error" in out
        assert out.get("hint")  # points the agent at statspai://functions
