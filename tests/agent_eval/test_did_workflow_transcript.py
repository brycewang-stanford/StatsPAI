"""End-to-end agent-workflow transcript (Day-7 regression net).

This is the integration test that proves an agent can actually *drive*
StatsPAI through the recommended MCP workflow — not just that individual
units pass.  It exercises the real ``execute_tool`` dispatch + result-cache
handle chaining + output enrichment, following the sequence the MCP server
instructions advertise:

    detect_design -> preflight -> fit(as_handle) -> audit_result
                  -> sensitivity_from_result

It also pins the *failure* UX: a method/design mismatch and a stale handle
must come back as recoverable error envelopes (with a hint), never a crash
— that recoverability is what lets an agent self-correct.

No network / R / Stata; deterministic synthetic panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

#: Tool names the MCP manifest actually exposes — a follow-up naming
#: anything else would send an agent to a tool that does not exist.
from statspai.agent.mcp_server import tool_manifest
from statspai.agent.tools import execute_tool

_MANIFEST_TOOLS = frozenset(t["name"] for t in tool_manifest())


def _did_panel(seed: int = 0, n: int = 600) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n // 2), 2),
            "time": np.tile([0, 1], n // 2),
        }
    )
    df["treat"] = (df["id"] % 2 == 0).astype(int)
    df["post"] = df["time"]
    df["y"] = 1.0 + 0.5 * df["treat"] * df["post"] + rng.normal(0, 1, len(df))
    return df


def _ok(out) -> bool:
    return isinstance(out, dict) and not out.get("error")


# --------------------------------------------------------------------------- #
#  Happy path: the full recommended transcript chains end to end
# --------------------------------------------------------------------------- #


def test_full_did_agent_transcript():
    df = _did_panel()

    # 1. Identify the study shape from raw columns.
    design = execute_tool("detect_design", {}, data=df)
    assert _ok(design), design
    # Naming a key proves nothing about what was detected.
    assert design["design"] == "panel"
    top = design["candidates"][0]
    assert top["unit"] == "id" and top["time"] == "time"
    assert top["n_units"] == len(df) // 2 and top["n_periods"] == 2

    # 2. Pre-fit identification checks — must return a verdict, not crash.
    pre = execute_tool(
        "preflight",
        {"method": "did", "y": "y", "treat": "treat", "time": "post"},
        data=df,
    )
    assert _ok(pre), pre
    # "verdict" existing is satisfied by FAIL; this panel is well-formed.
    assert pre["verdict"] == "PASS", [
        c for c in pre["checks"] if c.get("status") != "passed"
    ]

    # 3. Fit, caching the result so downstream tools chain by handle.
    fit = execute_tool(
        "did",
        {"y": "y", "treat": "treat", "time": "post"},
        data=df,
        detail="agent",
        as_handle=True,
    )
    assert _ok(fit), fit
    # Pin the number against the 2x2 difference-in-means, not its type.
    cells = df.groupby(["treat", "post"])["y"].mean()
    hand_did = (cells[(1, 1)] - cells[(1, 0)]) - (cells[(0, 1)] - cells[(0, 0)])
    assert fit["estimate"] == pytest.approx(float(hand_did), rel=1e-9)
    assert fit["n_obs"] == len(df)
    rid = fit.get("result_id")
    assert isinstance(rid, str) and rid.startswith("r_"), fit

    # 4. Reviewer-grade audit, by handle only (no data re-sent).
    audit = execute_tool("audit_result", {"result_id": rid}, data=None)
    assert _ok(audit), audit
    # A fresh fit has done no robustness work, so coverage is 0 and every
    # check is outstanding — assert that, rather than that the keys exist.
    assert audit["checks"], "audit_result returned no checks"
    assert audit["coverage"] == pytest.approx(0.0)
    assert all(c["status"] == "missing" for c in audit["checks"])
    assert any(c["name"] == "parallel_trends" for c in audit["checks"])
    assert all(
        c.get("suggest_function") for c in audit["checks"]
    ), "a missing check must name the function that would satisfy it"

    # 5. Design-agnostic sensitivity off the same handle.
    sens = execute_tool(
        "sensitivity_from_result",
        {"result_id": rid, "method": "evalue"},
        data=None,
    )
    assert _ok(sens), sens
    assert sens.get("source_result_id") == rid
    # The handle assertion alone passed while this tool returned an empty
    # payload. Check the E-value against sp.evalue_from_result.
    expected = sp.evalue_from_result(sp.did(df, y="y", treat="treat", time="post"))
    assert sens["evalue_estimate"] == pytest.approx(
        expected["evalue_estimate"], rel=1e-9
    )


# --------------------------------------------------------------------------- #
#  Result-handle chaining contract
# --------------------------------------------------------------------------- #


def test_handle_is_reusable_and_stale_handle_errors_cleanly():
    df = _did_panel(seed=1)
    fit = execute_tool(
        "did",
        {"y": "y", "treat": "treat", "time": "post"},
        data=df,
        as_handle=True,
    )
    rid = fit["result_id"]

    # Same handle drives two different downstream tools.
    assert _ok(execute_tool("audit_result", {"result_id": rid}, data=None))
    assert _ok(
        execute_tool(
            "sensitivity_from_result",
            {"result_id": rid, "method": "evalue"},
            data=None,
        )
    )

    # A handle that was never cached returns a clean, explanatory error.
    bad = execute_tool("audit_result", {"result_id": "r_deadbeef00000000"}, data=None)
    assert isinstance(bad, dict) and bad.get("error")
    assert "not found" in bad["error"].lower()


# --------------------------------------------------------------------------- #
#  Failure UX: a mismatch is recoverable, not fatal
# --------------------------------------------------------------------------- #


def test_method_mismatch_returns_recoverable_envelope():
    """honest_did needs an event study; a plain 2x2 must fail *gracefully*."""
    df = _did_panel(seed=2)
    fit = execute_tool(
        "did",
        {"y": "y", "treat": "treat", "time": "post"},
        data=df,
        as_handle=True,
    )
    out = execute_tool(
        "honest_did_from_result",
        {"result_id": fit["result_id"], "method": "relative_magnitude"},
        data=None,
    )
    # Not a crash — an envelope the agent can read and route around.
    assert isinstance(out, dict) and out.get("error")
    # Carries enough context to self-correct (a hint or the upstream cause).
    assert out.get("hint") or out.get("upstream_error")


# --------------------------------------------------------------------------- #
#  Enrichment reaches the agent
# --------------------------------------------------------------------------- #


def test_fit_payload_carries_enrichment_for_the_agent():
    df = _did_panel(seed=3)
    fit = execute_tool(
        "did",
        {"y": "y", "treat": "treat", "time": "post"},
        data=df,
        detail="agent",
        as_handle=True,
    )
    # The agent gets a citation handle, ready-to-run next calls, and prose.
    assert "citation_key" in fit
    assert isinstance(fit.get("next_calls"), list) and fit["next_calls"]
    assert isinstance(fit.get("narrative"), str) and fit["narrative"]
    assert fit["citation_key"] == "did_2x2"
    assert "Difference-in-Differences" in fit["narrative"]

    # Every advertised next-call names a real tool, and — the part that
    # matters to an agent — the readiness flag must be honest: a call
    # marked ready must dispatch without an argument error, and one marked
    # not-ready must say which arguments it still needs.
    from statspai.agent import execute_tool as _dispatch

    assert fit["next_calls"], "no follow-ups advertised"
    for nc in fit["next_calls"]:
        assert nc.get("tool")
        assert nc["tool"] in _MANIFEST_TOOLS, nc["tool"]
        assert isinstance(nc.get("ready"), bool)
        if nc["ready"]:
            out = _dispatch(nc["tool"], dict(nc.get("arguments") or {}), data=df)
            err = out.get("error", "") if isinstance(out, dict) else ""
            assert "missing" not in str(err).lower(), (
                f"{nc['tool']} is advertised ready but rejects its own "
                f"pre-filled arguments: {err}"
            )
        else:
            assert nc.get("missing_arguments"), nc
