"""Contract for the ``interpret_result`` MCP workflow tool.

``interpret_result`` closes the "explain this estimate in words" loop:

* When the MCP client advertised sampling, it reuses the agent's own
  model (via :class:`SamplingLLMClient`) — no API key — and grounds the
  explanation in the result's own numbers.
* With no sampling available it degrades to a deterministic structured
  brief; it never fabricates a narrative.
* A bad / missing handle returns a friendly, actionable error.
* It is exposed in the MCP manifest as a dataless tool (``data_path`` is
  injected but NOT required) so strict-schema clients can dispatch it.

The sampling round-trip is mocked exactly as in
``test_sampling_llm_loop`` — the fake writer synchronously routes a
scripted reply for the matching request id, so no network / no key.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from statspai.agent import execute_tool, mcp_handle_request
from statspai.agent import _sampling

# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_sampling():
    """Install a fake sampling-capable MCP client echoing a scripted reply."""
    prior_cap = _sampling.get_capability()
    state = {"reply": "Interpreted by the agent's own model."}

    def fake_writer(payload_str: str) -> None:
        req = json.loads(payload_str)
        _sampling.route_response(
            {
                "jsonrpc": "2.0",
                "id": req["id"],
                "result": {
                    "role": "assistant",
                    "content": {"type": "text", "text": state["reply"]},
                    "model": "mock-model",
                    "stopReason": "endTurn",
                },
            }
        )

    _sampling.set_capability(True)
    _sampling.set_writer(fake_writer)

    def set_reply(text: str) -> None:
        state["reply"] = text

    try:
        yield set_reply
    finally:
        _sampling.set_writer(None)
        _sampling.set_capability(prior_cap)


def _toy_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        treat = i % 2
        for t in (0, 1):
            y = 1.0 + 0.5 * t + 0.4 * treat * t + rng.normal(scale=0.1)
            rows.append({"id": i, "time": t, "treat": treat, "y": y})
    return pd.DataFrame(rows)


def _fit_handle() -> str:
    out = execute_tool(
        "did",
        {"y": "y", "treat": "treat", "time": "time"},
        data=_toy_panel(),
        as_handle=True,
    )
    assert "result_id" in out, out
    return out["result_id"]


# --------------------------------------------------------------------------- #
#  Deterministic fallback (no sampling)
# --------------------------------------------------------------------------- #


def test_interpret_falls_back_to_deterministic_without_sampling():
    """No advertised sampling -> deterministic brief, never an LLM call."""
    prior = _sampling.get_capability()
    _sampling.set_capability(False)
    try:
        rid = _fit_handle()
        out = execute_tool("interpret_result", {"result_id": rid})
    finally:
        _sampling.set_capability(prior)

    assert out["backend"] == "deterministic"
    assert isinstance(out["interpretation"], str) and out["interpretation"]
    assert out["result_id"] == rid
    # Grounding payload is attached so the agent can see the numbers.
    assert "summary" in out
    assert "sampling" in out.get("note", "").lower()


def test_interpret_missing_handle_is_friendly():
    out = execute_tool("interpret_result", {"result_id": "r_definitely_missing"})
    assert "error" in out
    assert "not found" in out["error"] or "result_id" in out["error"]


# --------------------------------------------------------------------------- #
#  LLM-in-the-loop path
# --------------------------------------------------------------------------- #


def test_interpret_uses_the_agents_model_when_sampling_advertised(mock_sampling):
    mock_sampling("The DiD ATT is positive and precisely estimated.")
    rid = _fit_handle()

    out = execute_tool(
        "interpret_result",
        {
            "result_id": rid,
            "question": "is the effect meaningful?",
            "audience": "policymaker",
        },
    )

    assert out["backend"] == "mcp_sampling"
    assert out["interpretation"] == "The DiD ATT is positive and precisely estimated."
    assert out["audience"] == "policymaker"
    assert out["question"] == "is the effect meaningful?"


def test_interpret_falls_back_loudly_on_sampling_error(monkeypatch):
    """A mid-call sampling failure degrades to the brief AND surfaces it."""
    prior_cap = _sampling.get_capability()
    _sampling.set_capability(True)

    def boom(*a, **k):  # noqa: ANN001
        raise _sampling.SamplingTimeoutError("client never replied")

    # Force resolve_llm_client to hand back a client whose chat() raises.
    monkeypatch.setattr(_sampling, "request_sampling", boom)
    _sampling.set_writer(lambda line: None)
    try:
        rid = _fit_handle()
        out = execute_tool("interpret_result", {"result_id": rid})
    finally:
        _sampling.set_writer(None)
        _sampling.set_capability(prior_cap)

    assert out["backend"] == "deterministic"
    assert "sampling_error" in out
    assert "SamplingTimeoutError" in out["sampling_error"]
    assert isinstance(out["interpretation"], str) and out["interpretation"]


# --------------------------------------------------------------------------- #
#  Manifest exposure (dataless)
# --------------------------------------------------------------------------- #


def _rpc(method, params=None, request_id=1):
    msg = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
    return json.loads(mcp_handle_request(json.dumps(msg)))


def test_interpret_result_is_in_manifest_and_dataless():
    msg = _rpc("tools/list")
    tools = {t["name"]: t for t in msg["result"]["tools"]}
    assert "interpret_result" in tools, "interpret_result not advertised"
    schema = tools["interpret_result"]["inputSchema"]
    # data_path is injected for client convenience but must NOT be required.
    assert "data_path" not in schema.get("required", [])
    assert "result_id" in schema.get("required", [])
