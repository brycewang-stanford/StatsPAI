"""MCP protocol-level transcript for an empirical-analysis agent loop.

The direct ``execute_tool`` transcript proves the Python dispatcher can
drive a workflow. This file pins what an actual MCP client sees over
JSON-RPC: ``structuredContent``, ``isError``, result handles,
ready-to-run follow-up calls, and provenance from the loaded CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent.mcp_server import handle_request


def _rpc(method: str, params: dict, request_id: int = 1) -> dict:
    raw = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
    )
    response = handle_request(raw)
    assert response is not None, f"{method} returned no response"
    return json.loads(response)


def _did_csv(tmp_path: Path, seed: int = 0, n: int = 600) -> Path:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n // 2), 2),
            "time": np.tile([0, 1], n // 2),
        }
    )
    df["treat"] = (df["id"] % 2 == 0).astype(int)
    df["post"] = df["time"]
    df["y"] = 1.0 + 0.5 * df["treat"] * df["post"] + rng.normal(0, 1, len(df))
    path = tmp_path / "did_panel.csv"
    df.to_csv(path, index=False)
    return path


def _call_tool(name: str, arguments: dict, request_id: int = 1) -> dict:
    msg = _rpc(
        "tools/call",
        {"name": name, "arguments": arguments},
        request_id=request_id,
    )
    assert "result" in msg, msg
    result = msg["result"]
    payload = result.get("structuredContent")
    assert isinstance(payload, dict), result
    assert payload == json.loads(result["content"][0]["text"])
    return result


def test_mcp_agent_empirical_analysis_transcript(tmp_path: Path) -> None:
    csv = _did_csv(tmp_path)

    design = _call_tool(
        "detect_design",
        {"data_path": str(csv)},
        request_id=10,
    )
    assert design["isError"] is False
    assert "design" in design["structuredContent"]
    assert "candidates" in design["structuredContent"]

    preflight = _call_tool(
        "preflight",
        {
            "data_path": str(csv),
            "method": "did",
            "y": "y",
            "treatment": "treat",
            "time": "post",
        },
        request_id=11,
    )
    assert preflight["isError"] is False
    assert "verdict" in preflight["structuredContent"]
    assert "checks" in preflight["structuredContent"]

    fit = _call_tool(
        "did",
        {
            "data_path": str(csv),
            "y": "y",
            "treat": "treat",
            "time": "post",
            "as_handle": True,
            "detail": "agent",
        },
        request_id=12,
    )
    payload = fit["structuredContent"]
    assert fit["isError"] is False
    assert isinstance(payload.get("estimate"), (int, float))
    rid = payload.get("result_id")
    assert isinstance(rid, str) and rid.startswith("r_"), payload
    assert payload["data_provenance"]["source"] == str(csv)
    assert payload["data_provenance"]["format"] == "csv"
    assert payload["next_calls"][0]["tool"] == "audit_result"
    assert payload["next_calls"][0]["arguments"]["result_id"] == rid

    audit = _call_tool(
        "audit_result",
        {"result_id": rid},
        request_id=13,
    )
    assert audit["isError"] is False
    assert "checks" in audit["structuredContent"]

    sensitivity = _call_tool(
        "sensitivity_from_result",
        {"result_id": rid, "method": "evalue"},
        request_id=14,
    )
    assert sensitivity["isError"] is False
    assert sensitivity["structuredContent"]["source_result_id"] == rid

    stale = _call_tool(
        "audit_result",
        {"result_id": "r_deadbeef00000000"},
        request_id=15,
    )
    assert stale["isError"] is True
    assert "not found" in stale["structuredContent"]["error"].lower()


# ---------------------------------------------------------------------------
# Runtime-orchestrator lane regressions (2026-06-21): the tool-contract
# fixes, exercised at the MCP protocol level an agent actually sees.
# ---------------------------------------------------------------------------


def _staggered_panel_csv(tmp_path: Path, seed: int = 3) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(60):
        cohort = int(rng.choice([0, 2005, 2008]))
        for year in range(2000, 2011):
            treated = 1 if cohort > 0 and year >= cohort else 0
            y = (
                1.0
                + 0.2 * (year - 2000)
                + 0.03 * unit
                + (1.2 if treated else 0.0)
                + rng.normal()
            )
            rows.append((unit, year, cohort, y))
    df = pd.DataFrame(rows, columns=["unit", "year", "cohort", "y"])
    path = tmp_path / "staggered_panel.csv"
    df.to_csv(path, index=False)
    return path


def _cross_section_csv(tmp_path: Path, seed: int = 4, n: int = 400) -> Path:
    rng = np.random.default_rng(seed)
    treat = rng.integers(0, 2, n).astype(float)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 0.5 * treat + 0.3 * x1 + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "treat": treat, "x1": x1, "x2": x2})
    path = tmp_path / "cross_section.csv"
    df.to_csv(path, index=False)
    return path


def test_detect_design_honors_column_hints(tmp_path: Path) -> None:
    """The advertised ``id_col_hint`` / ``time_col_hint`` must not crash.

    Regression for the schema↔dispatch drift where these hints were
    forwarded verbatim to ``sp.detect_design(unit=, time=)``.
    """
    csv = _did_csv(tmp_path)
    out = _call_tool(
        "detect_design",
        {"data_path": str(csv), "id_col_hint": "id", "time_col_hint": "post"},
        request_id=20,
    )
    assert out["isError"] is False
    assert "design" in out["structuredContent"]


def test_curated_result_handle_injection(tmp_path: Path) -> None:
    """Curated result-consuming tools resolve a ``result_id`` handle.

    ``honest_did`` and ``sensitivity`` operate on a fitted ``result``; the
    dispatcher injects the cached object from ``result_id`` so an agent can
    chain them off an ``as_handle`` fit without re-passing data.
    """
    panel = _staggered_panel_csv(tmp_path)
    cs = _call_tool(
        "callaway_santanna",
        {
            "data_path": str(panel),
            "y": "y",
            "g": "cohort",
            "t": "year",
            "i": "unit",
            "as_handle": True,
        },
        request_id=21,
    )
    rid = cs["structuredContent"]["result_id"]
    honest = _call_tool(
        "honest_did",
        {"e": 0, "method": "smoothness", "result_id": rid},
        request_id=22,
    )
    assert honest["isError"] is False, honest["structuredContent"]
    assert "_unsupported_args" not in honest["structuredContent"]

    cs_csv = _cross_section_csv(tmp_path)
    reg = _call_tool(
        "regress",
        {"data_path": str(cs_csv), "formula": "y ~ treat + x1 + x2", "as_handle": True},
        request_id=23,
    )
    reg_rid = reg["structuredContent"]["result_id"]
    sens = _call_tool(
        "sensitivity",
        {"y": "y", "treat": "treat", "controls": ["x1", "x2"], "result_id": reg_rid},
        request_id=24,
    )
    assert sens["isError"] is False, sens["structuredContent"]
    payload = sens["structuredContent"]
    assert "_unsupported_args" not in payload

    # `isError is False` alone is worthless here. Analysing the *intercept*
    # also does not error — it returns a number that answers a question
    # nobody asked, and this assertion stayed green through exactly that
    # bug. Check the value, against the treatment coefficient computed
    # independently: standardise by the outcome SD, map to a risk ratio
    # (VanderWeele & Ding 2017, RR = exp(0.91 * d)), then
    # E = RR + sqrt(RR * (RR - 1)).
    import math

    frame = pd.read_csv(cs_csv)
    fit = sp.regress("y ~ treat + x1 + x2", frame, robust="HC1")
    d = float(fit.params["treat"]) / float(frame["y"].std(ddof=1))
    rr = math.exp(0.91 * d)
    if rr < 1.0:
        rr = 1.0 / rr
    expected = rr + math.sqrt(rr * (rr - 1.0))

    assert payload["e_value_point"] == pytest.approx(expected, rel=1e-9), (
        "the MCP sensitivity tool must describe `treat`, not whatever "
        "coefficient happens to come first"
    )
    assert payload["rr_observed"] == pytest.approx(rr, rel=1e-9)


def test_spec_curve_multiverse_schema(tmp_path: Path) -> None:
    """spec_curve runs from its corrected (y / x / controls) schema."""
    csv = _cross_section_csv(tmp_path)
    out = _call_tool(
        "spec_curve",
        {
            "data_path": str(csv),
            "y": "y",
            "x": "treat",
            "controls": [["x1"], ["x1", "x2"]],
            "se_types": ["classical", "hc1"],
        },
        request_id=25,
    )
    assert out["isError"] is False, out["structuredContent"]
    payload = out["structuredContent"]
    assert "_unsupported_args" not in payload
    assert payload.get("n_specs", 0) >= 1
