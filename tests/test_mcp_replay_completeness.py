"""``replay_completeness``: a replay line says what re-running it needs.

``replay`` writes the dataset as ``data=data`` and a fitted handle as
``result=result_<id>``; inline data is recorded by hash only and an
argument without a literal form as ``<TypeName>``. The string is an audit
record in all of those cases but a runnable call in only some
(2026-10-02 review, A4), so the envelope labels which.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent._replay import REPLAY_LEVELS, replay_completeness
from statspai.agent.mcp_server import handle_request


def _call(name, **arguments):
    raw = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    return json.loads(handle_request(raw))["result"]["structuredContent"]


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=120)})
    df["y"] = 1.0 + 0.5 * df["x"] + rng.normal(size=120)
    return df


def test_levels_are_ordered_weakest_first():
    assert REPLAY_LEVELS == ("call_only", "session_replayable", "standalone")


def test_unit_weakest_component_decides():
    local = {"source_type": "local", "source": "/d/a.csv", "sha256": "abc"}
    out = replay_completeness("sp.regress(data=data, formula='y ~ x')", local)
    assert out["level"] == "standalone"
    assert out["needs"] == ["file /d/a.csv (sha256 abc)"]

    out = replay_completeness("sp.regress(data=data, weights=<Series>)", local)
    assert out["level"] == "call_only" and "<TypeName>" in out["needs"][0]

    out = replay_completeness("sp.audit(result=result_r_1)", None, result_id="r_1")
    assert out["level"] == "session_replayable"

    out = replay_completeness("sp.regress(data=data, formula='y ~ x')", None)
    assert out["level"] == "call_only"

    out = replay_completeness("sp.dag_example(name='frontdoor')", None)
    assert out == {"level": "standalone", "needs": []}


def test_local_file_fit_is_standalone_and_really_reruns(frame, tmp_path):
    path = tmp_path / "ols.csv"
    frame.to_csv(path, index=False)
    sc = _call("regress", formula="y ~ x", data_path=str(path))
    rc = sc["replay_completeness"]
    assert rc["level"] == "standalone"
    assert str(path) in rc["needs"][0] and "sha256" in rc["needs"][0]
    # The label is a claim about a new process: honour it with only the
    # string and the file named in ``needs``.
    call = sc["replay"].split("  #", 1)[0]
    res = eval(call, {"sp": sp, "data": pd.read_csv(path)})  # noqa: S307
    assert res.params["x"] == pytest.approx(
        sc["coefficients"]["x"]["estimate"], rel=1e-12
    )


def test_handle_and_inline_data_are_not_standalone(frame, tmp_path):
    path = tmp_path / "ols.csv"
    frame.to_csv(path, index=False)
    data_id = _call("load_data", data_path=str(path))["data_id"]
    sc = _call("regress", formula="y ~ x", data_id=data_id)
    assert sc["replay_completeness"]["level"] == "session_replayable"
    assert data_id in sc["replay_completeness"]["needs"][0]

    sc = _call("regress", formula="y ~ x", data_records=frame.to_dict("records"))
    assert sc["replay_completeness"]["level"] == "call_only"
    assert "inline" in sc["replay_completeness"]["needs"][0]


def test_follow_up_on_a_result_handle_is_session_bound(tmp_path):
    rng = np.random.default_rng(1)
    z, u = rng.normal(size=300), rng.normal(size=300)
    d = 0.8 * z + u + rng.normal(size=300)
    iv = pd.DataFrame({"y": 1 + 2 * d + u + rng.normal(size=300), "d": d, "z": z})
    path = tmp_path / "iv.csv"
    iv.to_csv(path, index=False)
    rid = _call("ivreg", formula="y ~ (d ~ z)", data_path=str(path), as_handle=True)[
        "result_id"
    ]
    follow = _call("estat", result_id=rid, test="firststage")
    assert f"result=result_{rid}" in follow["replay"]
    rc = follow["replay_completeness"]
    assert rc["level"] == "session_replayable" and rid in rc["needs"][0]
