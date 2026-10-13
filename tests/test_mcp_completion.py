"""``completion/complete`` for resource templates and prompt arguments."""

from __future__ import annotations

import json

import pandas as pd

from statspai.agent import mcp_server as m
from statspai.agent._completion import MAX_VALUES
from statspai.agent._data_cache import DATA_CACHE
from statspai.agent._result_cache import RESULT_CACHE


def _rpc(method, params=None):
    line = m.handle_request(
        json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        )
    )
    return json.loads(line)


def _complete(ref, name, value=""):
    return _rpc(
        "completion/complete", {"ref": ref, "argument": {"name": name, "value": value}}
    )


def _template(uri):
    return {"type": "ref/resource", "uri": uri}


def test_capability_is_advertised():
    caps = _rpc("initialize", {})["result"]["capabilities"]
    assert caps["completions"] == {}


def test_every_advertised_template_is_completable():
    templates = _rpc("resources/templates/list")["result"]["resourceTemplates"]
    assert templates
    for t in templates:
        variable = t["uriTemplate"].split("{", 1)[1].split("}", 1)[0]
        out = _complete(_template(t["uriTemplate"]), variable)
        assert "result" in out, (t["uriTemplate"], out)


def test_function_names_rank_prefix_matches_first_and_are_capped():
    c = _complete(_template("statspai://function/{name}"), "name", "did")["result"][
        "completion"
    ]
    assert c["values"][0] == "did"
    first_non_prefix = next(
        (i for i, v in enumerate(c["values"]) if not v.startswith("did")),
        len(c["values"]),
    )
    assert all("did" in v for v in c["values"])
    assert all(v.startswith("did") for v in c["values"][:first_non_prefix])
    assert not any(v.startswith("did") for v in c["values"][first_non_prefix:])

    everything = _complete(_template("statspai://function/{name}"), "name")["result"][
        "completion"
    ]
    assert len(everything["values"]) == MAX_VALUES
    assert everything["total"] > MAX_VALUES and everything["hasMore"] is True


def test_guide_families_are_the_ones_the_resource_serves():
    values = _complete(_template("statspai://guide/{family}"), "family")["result"][
        "completion"
    ]["values"]
    assert values
    for family in values:
        read = _rpc("resources/read", {"uri": f"statspai://guide/{family}"})
        assert "result" in read, family


def test_live_handles_complete_and_dropped_ones_do_not():
    rid = RESULT_CACHE.put({"estimate": 1.0}, tool="did")
    did = DATA_CACHE.put(pd.DataFrame({"a": [1, 2]}), tool="load_data")
    try:
        got = _complete(_template("statspai://result/{id}"), "id", rid[:5])
        assert rid in got["result"]["completion"]["values"]
        got = _complete(_template("statspai://data/{id}"), "id", "d_")
        assert did in got["result"]["completion"]["values"]
        prompt = {"type": "ref/prompt", "name": "robustness_followup"}
        got = _complete(prompt, "result_id")
        assert rid in got["result"]["completion"]["values"]
    finally:
        RESULT_CACHE.evict(rid)
        DATA_CACHE.evict(did)
    got = _complete(_template("statspai://result/{id}"), "id")
    assert rid not in got["result"]["completion"]["values"]


def test_unenumerable_argument_completes_to_nothing():
    prompt = {"type": "ref/prompt", "name": "audit_did_result"}
    c = _complete(prompt, "y", "wa")["result"]["completion"]
    assert c == {"values": [], "total": 0, "hasMore": False}


def test_unknown_references_are_invalid_params():
    for ref in (
        {"type": "ref/prompt", "name": "no_such_prompt"},
        _template("statspai://nope/{x}"),
        {"type": "ref/tool", "name": "did"},
    ):
        out = _complete(ref, "x")
        assert out["error"]["code"] == -32602, ref
    bad = _rpc("completion/complete", {"ref": "did", "argument": {}})
    assert bad["error"]["code"] == -32602
    named = _complete({"type": "ref/prompt", "name": "no_such_prompt"}, "x")
    assert named["error"]["message"] == "Unknown prompt: 'no_such_prompt'"
