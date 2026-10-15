"""Every ``next_calls`` entry marked ``ready: true`` must run as given.

The server instructions tell a client to dispatch a follow-up directly
when it is marked ready. An audit on 2026-10-16 found 6 of 11 ready
follow-ups failing verbatim: ``rdplot`` / ``rddensity`` / ``rdsensitivity``
after ``rdrobust`` and ``bacon_decomposition`` after ``did`` arrived with a
``result_id`` they do not read and without the dataset they do; the
E-value follow-up after ``regress`` / ``ivreg`` targeted results that
carry no single causal estimate. This test drives the real JSON-RPC
handler: load a dataset, fit with ``as_handle``, then dispatch each ready
follow-up exactly as the server wrote it.
"""

from __future__ import annotations

import json
import os
import warnings

import pytest

import statspai as sp
from statspai.agent.mcp_server import handle_request

os.environ.setdefault("MPLBACKEND", "Agg")


def _call(name, arguments, _counter=[0]):
    _counter[0] += 1
    raw = handle_request(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": _counter[0],
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
    )
    reply = json.loads(raw)
    assert "result" in reply, reply
    return reply["result"]


def _load(tmp_path, frame, name):
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    out = _call("load_data", {"data_path": str(path), "detail": "minimal"})
    return out["structuredContent"]["data_id"]


_CASES = {
    # A non-zero cutoff: a follow-up that dropped ``c`` would run at 0.
    "rdrobust": ("lee_2008_senate", {"y": "y", "x": "x", "c": 5}),
    "did": (
        "mpdta",
        {"y": "lemp", "treat": "treat", "time": "year", "id": "countyreal"},
    ),
    "callaway_santanna": (
        "mpdta",
        {"y": "lemp", "g": "first_treat", "t": "year", "i": "countyreal"},
    ),
    "regress": ("nsw_dw", {"formula": "re78 ~ treat + age + education"}),
    "ivreg": ("card_1995", {"formula": "lwage ~ (educ ~ nearc4) + exper"}),
}


@pytest.mark.parametrize("tool", sorted(_CASES))
def test_ready_followups_dispatch_verbatim(tool, tmp_path):
    dataset, args = _CASES[tool]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = getattr(sp.datasets, dataset)()
    data_id = _load(tmp_path, frame, dataset)
    fit = _call(tool, {**args, "data_id": data_id, "as_handle": True})
    assert fit["isError"] is False, fit
    payload = fit["structuredContent"]
    ready = [c for c in payload.get("next_calls", []) if c.get("ready")]
    failures = []
    for follow in ready:
        out = _call(follow["tool"], follow["arguments"])
        body = out.get("structuredContent") or {}
        if out.get("isError") or body.get("error") or body.get("_unsupported_args"):
            failures.append((follow, body.get("error"), body.get("_unsupported_args")))
    assert not failures, failures
    if tool == "rdrobust":
        for follow in ready:
            assert follow["arguments"].get("c") == 5, follow
    # Data-reading follow-ups run on the fit's own dataset.
    for follow in ready:
        if "result_id" not in follow["arguments"]:
            assert follow["arguments"].get("data_id") == data_id, follow
