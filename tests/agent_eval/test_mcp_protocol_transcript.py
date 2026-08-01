"""MCP protocol-level transcript for an empirical-analysis agent loop.

The direct ``execute_tool`` transcript proves the Python dispatcher can
drive a workflow. This file pins what an actual MCP client sees over
JSON-RPC: ``structuredContent``, ``isError``, result handles,
ready-to-run follow-up calls, and provenance from the loaded CSV.
"""

from __future__ import annotations

import json
import math
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
    # Not just "a design key exists" — this CSV is an id x time panel, so
    # name it, and check the detector read the panel's real shape.
    dpayload = design["structuredContent"]
    assert dpayload["design"] == "panel"
    top = dpayload["candidates"][0]
    assert top["unit"] == "id" and top["time"] == "time"
    assert top["n_units"] == 300 and top["n_periods"] == 2
    assert dpayload["n_obs"] == 600

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
    # "verdict" merely existing is satisfied by FAIL. This panel is
    # well-formed, so demand PASS: preflight used to reject the canonical
    # `treatment=` spelling its own schema advertises and fail every time.
    ppayload = preflight["structuredContent"]
    assert ppayload["verdict"] == "PASS", [
        c for c in ppayload["checks"] if c.get("status") != "passed"
    ]
    assert all(c["status"] == "passed" for c in ppayload["checks"])

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
    # isinstance(..., float) passes for any number at all. Pin the value
    # against the 2x2 difference-in-means computed straight from the CSV.
    frame = pd.read_csv(csv)
    cells = frame.groupby(["treat", "post"])["y"].mean()
    hand_did = (cells[(1, 1)] - cells[(1, 0)]) - (cells[(0, 1)] - cells[(0, 0)])
    assert payload["estimate"] == pytest.approx(float(hand_did), rel=1e-9)
    assert payload["n_obs"] == len(frame)
    assert payload["estimand"] == "ATT"
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
    # An empty checks list would satisfy `"checks" in payload`.
    checks = audit["structuredContent"]["checks"]
    assert checks, "audit_result returned no checks"
    assert {"name", "status"} <= set(checks[0])
    assert any(
        c["name"] == "parallel_trends" for c in checks
    ), "a DiD audit that never mentions parallel trends is not an audit"

    sensitivity = _call_tool(
        "sensitivity_from_result",
        {"result_id": rid, "method": "evalue"},
        request_id=14,
    )
    assert sensitivity["isError"] is False
    spayload = sensitivity["structuredContent"]
    assert spayload["source_result_id"] == rid
    # This tool returned nothing but source_result_id — a plain-dict result
    # serialised to an empty payload — and the handle assertion above stayed
    # green through it. Check the number, against sp.evalue_from_result.
    expected_ev = sp.evalue_from_result(
        sp.did(frame, y="y", treat="treat", time="post")
    )
    assert spayload["evalue_estimate"] == pytest.approx(
        expected_ev["evalue_estimate"], rel=1e-9
    )
    assert spayload["rr_estimate"] == pytest.approx(
        expected_ev["rr_estimate"], rel=1e-9
    )

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
    # Hints must steer the detector, not merely be tolerated.
    hinted = out["structuredContent"]
    assert hinted["design"] == "panel"
    assert hinted["candidates"][0]["unit"] == "id"
    assert hinted["candidates"][0]["time"] == "post"


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
    hpayload = honest["structuredContent"]
    assert "_unsupported_args" not in hpayload

    # honest_did returns a sensitivity curve over M, so assert the property
    # that makes it a sensitivity curve: relaxing the smoothness bound can
    # only widen the interval, and once the breakdown point is crossed the
    # null stays inside. A degenerate or constant payload fails this;
    # "no unsupported args" would not notice.
    order = sorted(hpayload["M"], key=lambda k: int(k))
    m_grid = [hpayload["M"][k] for k in order]
    lows = [hpayload["ci_lower"][k] for k in order]
    highs = [hpayload["ci_upper"][k] for k in order]
    rejects = [hpayload["rejects_zero"][k] for k in order]

    assert len(m_grid) >= 3
    assert m_grid == sorted(m_grid), "M grid must be increasing"
    assert m_grid[0] == pytest.approx(0.0), "the grid should start at M=0"
    assert all(math.isfinite(v) for v in lows + highs)
    assert all(lo < hi for lo, hi in zip(lows, highs))

    widths = [hi - lo for lo, hi in zip(lows, highs)]
    assert all(
        b >= a - 1e-9 for a, b in zip(widths, widths[1:])
    ), f"widening M must not tighten the interval: {widths}"
    assert widths[-1] > widths[0], "the curve never responds to M"

    # Rejection can only be lost as M grows, never regained.
    if False in rejects:
        first_false = rejects.index(False)
        assert not any(rejects[first_false:])

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
    # 2 control sets x 2 SE types is 4 specifications, not "at least 1".
    assert payload["n_specs"] == 4
    frame = pd.read_csv(csv)
    # The SE type does not move the point estimate, so each control set
    # contributes its beta twice across the four specifications.
    betas = [
        float(sp.regress(f"y ~ treat + {' + '.join(ctrl)}", frame).params["treat"])
        for ctrl in (["x1"], ["x1", "x2"])
    ]
    assert payload["median_estimate"] == pytest.approx(
        float(np.median(betas * 2)), rel=1e-6
    )
    assert 0.0 <= payload["share_significant"] <= 1.0
    assert 0.0 <= payload["share_positive"] <= 1.0
