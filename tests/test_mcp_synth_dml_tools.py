"""synth and dml are curated MCP tools (default profile, not core).

Before 2026-10-17 both were reachable only through call_function: they
were named in the core profile list but, not being curated, were filtered
out of every tools/list. These check that the curated tools are listed
where they should be and return the same numbers as the direct calls.
"""

from __future__ import annotations

import json
import warnings

import pytest

import statspai as sp
from statspai.agent import mcp_server as m


def _rpc(method, params, _i=[0]):
    _i[0] += 1
    reply = json.loads(
        m.handle_request(
            json.dumps(
                {"jsonrpc": "2.0", "id": _i[0], "method": method, "params": params}
            )
        )
    )
    assert "result" in reply, reply
    return reply["result"]


def _listed(profile):
    m.set_tool_profile(profile)
    try:
        return {t["name"] for t in _rpc("tools/list", {})["tools"]}
    finally:
        m.set_tool_profile(None)


def test_synth_and_dml_are_curated_but_not_core():
    assert {"synth", "dml"} <= _listed("curated")
    assert not {"synth", "dml"} & _listed("core")


def _load(tmp_path, frame, name):
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    out = _rpc(
        "tools/call", {"name": "load_data", "arguments": {"data_path": str(path)}}
    )
    return out["structuredContent"]["data_id"]


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_synth_tool_matches_the_direct_call(tmp_path):
    ca = sp.load_dataset("california_prop99")
    args = dict(
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        placebo=False,
    )
    data_id = _load(tmp_path, ca, "prop99")
    out = _rpc(
        "tools/call", {"name": "synth", "arguments": {**args, "data_id": data_id}}
    )
    assert out["isError"] is False, out
    direct = sp.synth(ca, **args)
    assert out["structuredContent"]["estimate"] == pytest.approx(
        direct.estimate, rel=1e-12
    )


def test_dml_tool_matches_the_direct_call(tmp_path):
    nsw = sp.load_dataset("nsw_dw")
    args = dict(
        y="re78",
        treat="treat",
        covariates=["age", "education", "re74", "re75"],
        model_y="lasso",
        model_d="lasso",
        random_state=7,
    )
    data_id = _load(tmp_path, nsw, "nsw")
    out = _rpc("tools/call", {"name": "dml", "arguments": {**args, "data_id": data_id}})
    assert out["isError"] is False, out
    direct = sp.dml(nsw, **args)
    body = out["structuredContent"]
    assert body["estimate"] == pytest.approx(direct.estimate, rel=1e-12)
    assert body["se"] == pytest.approx(direct.se, rel=1e-12)
