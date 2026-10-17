"""Per-observation arrays capped in a tool payload stay readable in full.

Tool payloads show 20 values of residuals / fitted values / scores and
list the cut under ``arrays_capped``. Before 2026-10-17 nothing in the
MCP surface returned the rest, so an agent that needed the residuals had
no way to get them. Each capped entry now names a resource that does.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent import mcp_server as m
from statspai.agent.tools._helpers import OBS_ARRAY_KEEP


def _rpc(method, params, _i=[0]):
    _i[0] += 1
    return json.loads(
        m.handle_request(
            json.dumps(
                {"jsonrpc": "2.0", "id": _i[0], "method": method, "params": params}
            )
        )
    )


@pytest.fixture()
def fit(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": np.cumsum(rng.normal(size=300))})
    path = tmp_path / "ts.csv"
    df.to_csv(path, index=False)
    data_id = _rpc(
        "tools/call", {"name": "load_data", "arguments": {"data_path": str(path)}}
    )["result"]["structuredContent"]["data_id"]
    args = {"function": "arima", "arguments": {"y": "y", "order": [1, 1, 0]}}
    out = _rpc(
        "tools/call",
        {
            "name": "call_function",
            "arguments": {**args, "data_id": data_id, "as_handle": True},
        },
    )["result"]["structuredContent"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = sp.arima("y", order=(1, 1, 0), data=pd.read_csv(path))
    return out, direct


def test_capped_residuals_are_readable_in_full(fit):
    out, direct = fit
    capped = {c["path"]: c for c in out["arrays_capped"]}
    entry = capped["/residuals"]
    assert entry["shown"] == OBS_ARRAY_KEEP and entry["total"] == 300
    body = json.loads(
        _rpc("resources/read", {"uri": entry["full"]})["result"]["contents"][0]["text"]
    )
    assert body["n"] == 300
    np.testing.assert_allclose(
        body["values"], np.asarray(direct.residuals, dtype=float), rtol=1e-12
    )
    # The payload's 20 values are the head of the same array.
    assert out["residuals"] == body["values"][:OBS_ARRAY_KEEP]


def test_only_per_observation_arrays_are_served(fit):
    out, _ = fit
    rid = out["result_id"]
    reply = _rpc("resources/read", {"uri": f"statspai://result/{rid}/arrays/params"})
    assert "error" in reply and "per-observation" in reply["error"]["message"]
