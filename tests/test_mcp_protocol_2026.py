"""MCP ``2026-07-28`` (per-request metadata, no handshake) beside the handshake.

The server is dual-era: a request whose ``_meta`` names a protocol version,
or a ``server/discover`` probe, is served under the per-request rules; an
``initialize`` request selects the handshake revisions exactly as before.
Every expectation below is a clause of the 2026-07-28 specification
(Versioning, Overview, Caching, server/discover, stdio, Streamable HTTP).
"""

from __future__ import annotations

import http.client
import io
import json
import threading

import pytest

from statspai.agent import _http_transport as ht
from statspai.agent import _protocol, _sampling
from statspai.agent import mcp_server as m

V = "2026-07-28"
META = {
    _protocol.META_PROTOCOL_VERSION: V,
    _protocol.META_CLIENT_CAPABILITIES: {},
    _protocol.META_CLIENT_INFO: {"name": "pytest", "version": "0"},
}
SERVER_INFO = _protocol.META_SERVER_INFO


def _rpc(method, params=None, request_id=1):
    line = json.dumps(
        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    )
    return json.loads(m.handle_request(line))


def _modern(method, request_id=1, meta=None, **params):
    return _rpc(method, {"_meta": dict(meta or META), **params}, request_id)


# --------------------------------------------------------------------- #
#  Version negotiation and metadata validation
# --------------------------------------------------------------------- #


def test_discover_lists_versions_capabilities_and_identity():
    res = _modern("server/discover")["result"]
    assert res["resultType"] == "complete"
    assert res["supportedVersions"][0] == V
    # The handshake revisions stay listed: a dual-era client may pick one.
    assert set(m.SUPPORTED_PROTOCOL_VERSIONS) <= set(res["supportedVersions"])
    assert set(res["capabilities"]) >= {"tools", "resources", "prompts"}
    assert res["instructions"] == m._SESSION_INSTRUCTIONS
    ident = res["_meta"][SERVER_INFO]
    assert ident["name"] == m.SERVER_NAME and ident["version"] == m.SERVER_VERSION
    assert ident["title"] == "StatsPAI" and ident["websiteUrl"].startswith("https://")
    assert res["ttlMs"] >= 0 and res["cacheScope"] in ("public", "private")


def test_unsupported_version_is_32022_with_supported_list():
    meta = {**META, _protocol.META_PROTOCOL_VERSION: "1900-01-01"}
    err = _modern("tools/list", meta=meta)["error"]
    assert err["code"] == -32022
    assert err["data"]["requested"] == "1900-01-01"
    assert err["data"]["supported"][0] == V


def test_handshake_version_in_meta_points_to_initialize():
    # 2025-11-25 is served, but only after an initialize handshake.
    meta = {**META, _protocol.META_PROTOCOL_VERSION: "2025-11-25"}
    err = _modern("tools/list", meta=meta)["error"]
    assert err["code"] == -32022
    assert "initialize" in err["message"]


@pytest.mark.parametrize(
    "drop", [_protocol.META_PROTOCOL_VERSION, _protocol.META_CLIENT_CAPABILITIES]
)
def test_missing_required_meta_field_is_invalid_params(drop):
    meta = {k: v for k, v in META.items() if k != drop}
    if drop == _protocol.META_PROTOCOL_VERSION:
        # Without the version the request reads as a handshake request,
        # except for server/discover, which only exists per-request.
        err = _modern("server/discover", meta=meta)["error"]
    else:
        err = _modern("tools/list", meta=meta)["error"]
    assert err["code"] == -32602
    assert drop in err["message"]


def test_methods_removed_in_2026_are_not_found():
    for method in ("ping", "logging/setLevel"):
        assert _modern(method)["error"]["code"] == -32601


def test_initialize_still_selects_the_handshake():
    res = _rpc("initialize", {"protocolVersion": "2025-11-25", "capabilities": {}})[
        "result"
    ]
    assert res["protocolVersion"] == "2025-11-25"
    assert res["serverInfo"]["title"] == "StatsPAI"
    # Handshake results are not decorated with per-request fields.
    legacy = _rpc("tools/list", {})["result"]
    assert "resultType" not in legacy and "ttlMs" not in legacy


# --------------------------------------------------------------------- #
#  Result decoration
# --------------------------------------------------------------------- #


def test_tools_list_carries_result_type_identity_and_cache_hints():
    res = _modern("tools/list")["result"]
    assert res["resultType"] == "complete"
    assert res["_meta"][SERVER_INFO]["version"] == m.SERVER_VERSION
    assert res["cacheScope"] == "public" and res["ttlMs"] > 0
    names = [t["name"] for t in res["tools"]]
    # Same catalogue as the handshake path, in a deterministic order.
    assert names == [t["name"] for t in _rpc("tools/list", {})["result"]["tools"]]


def test_spliced_tools_list_equals_decoding_and_decorating():
    encoded = m._tools_list_result_json()
    params = {"_meta": META}
    spliced = json.loads(_protocol.decorate_result_json("tools/list", params, encoded))
    direct = _protocol.decorate_result("tools/list", params, json.loads(encoded))
    assert spliced == direct
    assert json.loads(_protocol.decorate_result_json("tools/list", params, "{}")) == (
        _protocol.decorate_result("tools/list", params, {})
    )


def test_handle_contents_are_private_and_stale_catalogue_public():
    loaded = _modern(
        "tools/call", name="load_data", arguments={"data_csv": "y,x\n1,2\n3,4\n"}
    )["result"]
    assert loaded["resultType"] == "complete" and not loaded["isError"]
    data_uri = loaded["structuredContent"]["data_uri"]
    handle = _modern("resources/read", uri=data_uri)["result"]
    assert (handle["cacheScope"], handle["ttlMs"]) == ("private", 0)
    catalogue = _modern("resources/read", uri="statspai://arguments")["result"]
    assert catalogue["cacheScope"] == "public" and catalogue["ttlMs"] > 0


def test_resource_not_found_uses_32602_not_32002():
    err = _modern("resources/read", uri="statspai://result/r_missing")["error"]
    assert err["code"] == -32602
    # The handshake path keeps the code of its own revision.
    legacy = _rpc("resources/read", {"uri": "statspai://result/r_missing"})["error"]
    assert legacy["code"] == -32002


def test_error_codes_avoid_the_grandfathered_band():
    assert _protocol.map_error_code(-32000) == -32603
    assert _protocol.map_error_code(-32002) == -32602
    assert _protocol.map_error_code(-32602) == -32602


def test_no_sampling_request_inside_a_per_request_call():
    _sampling.set_capability(True)
    try:
        assert _sampling.get_capability() is True
        with _protocol.modern_request():
            assert _sampling.get_capability() is False
        assert _sampling.get_capability() is True
    finally:
        _sampling.set_capability(False)


# --------------------------------------------------------------------- #
#  stdio: probe, subscriptions/listen, cancellation
# --------------------------------------------------------------------- #


def _stdio(lines):
    out = io.StringIO()
    m.serve_stdio(stdin=[json.dumps(x) + "\n" for x in lines], stdout=out)
    return [json.loads(x) for x in out.getvalue().splitlines() if x.strip()]


def test_stdio_probe_then_call_without_handshake():
    msgs = _stdio(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "server/discover",
                "params": {"_meta": META},
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "_meta": META,
                    "name": "bibtex",
                    "arguments": {"keys": ["callaway2021difference"]},
                },
            },
        ]
    )
    by_id = {x["id"]: x for x in msgs if "id" in x}
    assert by_id[1]["result"]["supportedVersions"][0] == V
    call = by_id[2]["result"]
    assert call["resultType"] == "complete" and call["isError"] is False
    assert "callaway2021difference" in call["structuredContent"]["bibtex"]


def test_stdio_listen_acknowledged_cancelled_silently():
    listen = {
        "jsonrpc": "2.0",
        "id": "sub-1",
        "method": "subscriptions/listen",
        "params": {"_meta": META, "notifications": {"toolsListChanged": True}},
    }
    cancel = {
        "jsonrpc": "2.0",
        "method": "notifications/cancelled",
        "params": {"requestId": "sub-1"},
    }
    msgs = _stdio([listen, cancel])
    assert len(msgs) == 1
    ack = msgs[0]
    assert ack["method"] == "notifications/subscriptions/acknowledged"
    assert ack["params"]["_meta"][_protocol.META_SUBSCRIPTION_ID] == "sub-1"
    # Static lists: no notification type is honoured.
    assert ack["params"]["notifications"] == {}


def test_stdio_listen_open_at_shutdown_gets_teardown_result():
    listen = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "subscriptions/listen",
        "params": {"_meta": META, "notifications": {}},
    }
    msgs = _stdio([listen])
    assert msgs[0]["method"] == "notifications/subscriptions/acknowledged"
    end = msgs[1]
    assert end["id"] == 7 and end["result"]["resultType"] == "complete"
    assert end["result"]["_meta"][_protocol.META_SUBSCRIPTION_ID] == 7


def test_stdio_listen_with_bad_version_is_refused():
    meta = {**META, _protocol.META_PROTOCOL_VERSION: "1900-01-01"}
    msgs = _stdio(
        [
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "subscriptions/listen",
                "params": {"_meta": meta, "notifications": {}},
            }
        ]
    )
    assert msgs == [
        {
            "jsonrpc": "2.0",
            "id": 3,
            "error": {
                "code": -32022,
                "message": "Unsupported protocol version",
                "data": {
                    "supported": list(_protocol.all_supported_versions()),
                    "requested": "1900-01-01",
                },
            },
        }
    ]


# --------------------------------------------------------------------- #
#  Streamable HTTP
# --------------------------------------------------------------------- #


@pytest.fixture
def http_server():
    srv = ht.make_http_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(5)


def _post(srv, msg, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=60)
    h = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    h.update(headers or {})
    try:
        conn.request("POST", "/mcp", body=json.dumps(msg).encode(), headers=h)
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, dict(resp.getheaders()), (json.loads(raw) if raw else None)
    finally:
        conn.close()


def _modern_msg(method, request_id=1, meta=None, **params):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {"_meta": dict(meta or META), **params},
    }


def _headers(method, name=None, version=V):
    h = {"MCP-Protocol-Version": version, "Mcp-Method": method}
    if name is not None:
        h["Mcp-Name"] = name
    return h


def test_http_call_without_session(http_server):
    msg = _modern_msg(
        "tools/call", name="bibtex", arguments={"keys": ["callaway2021difference"]}
    )
    status, hdrs, body = _post(http_server, msg, _headers("tools/call", "bibtex"))
    assert status == 200, body
    assert "Mcp-Session-Id" not in hdrs
    assert body["result"]["resultType"] == "complete"


def test_http_base64_name_header_is_decoded(http_server):
    msg = _modern_msg("tools/call", name="bibtex", arguments={"keys": []})
    status, _, body = _post(
        http_server, msg, _headers("tools/call", "=?base64?YmlidGV4?=")
    )
    assert status == 200, body


@pytest.mark.parametrize(
    "headers, fragment",
    [
        ({"MCP-Protocol-Version": V, "Mcp-Method": "tools/call"}, "Mcp-Name"),
        (_headers("tools/list", "bibtex"), "Mcp-Method"),
        (_headers("tools/call", "regress"), "Mcp-Name"),
        ({"MCP-Protocol-Version": V, "Mcp-Name": "bibtex"}, "Mcp-Method"),
    ],
)
def test_http_header_mismatch_is_400_32020(http_server, headers, fragment):
    msg = _modern_msg("tools/call", name="bibtex", arguments={"keys": []})
    status, _, body = _post(http_server, msg, headers)
    assert status == 400
    assert body["error"]["code"] == -32020
    assert fragment in body["error"]["message"]


def test_http_unsupported_version_is_400_32022(http_server):
    meta = {**META, _protocol.META_PROTOCOL_VERSION: "1900-01-01"}
    msg = _modern_msg("tools/list", meta=meta)
    status, _, body = _post(
        http_server, msg, _headers("tools/list", version="1900-01-01")
    )
    assert status == 400
    assert body["error"]["code"] == -32022
    assert body["error"]["data"]["requested"] == "1900-01-01"


def test_http_unknown_method_is_404_32601(http_server):
    status, _, body = _post(http_server, _modern_msg("ping"), _headers("ping"))
    assert status == 404 and body["error"]["code"] == -32601


def test_http_modern_header_without_meta_is_400_32602(http_server):
    msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    status, _, body = _post(http_server, msg, _headers("tools/list"))
    assert status == 400 and body["error"]["code"] == -32602


def test_http_session_header_is_ignored(http_server):
    headers = {**_headers("tools/list"), "Mcp-Session-Id": "never-issued"}
    status, _, body = _post(http_server, _modern_msg("tools/list"), headers)
    assert status == 200 and body["result"]["resultType"] == "complete"


def test_http_listen_stream_is_acknowledged(http_server):
    conn = http.client.HTTPConnection(
        "127.0.0.1", http_server.server_address[1], timeout=30
    )
    msg = _modern_msg("subscriptions/listen", request_id="s", notifications={})
    try:
        conn.request(
            "POST",
            "/mcp",
            body=json.dumps(msg).encode(),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                **_headers("subscriptions/listen"),
            },
        )
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type") == "text/event-stream"
        lines = []
        while not lines or not lines[-1].startswith("data:"):
            lines.append(resp.fp.readline().decode().strip())
        ack = json.loads(lines[-1][len("data:") :])
        assert ack["method"] == "notifications/subscriptions/acknowledged"
        assert ack["params"]["_meta"][_protocol.META_SUBSCRIPTION_ID] == "s"
    finally:
        conn.close()


def test_http_handshake_clients_are_unaffected(http_server):
    status, hdrs, body = _post(
        http_server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "capabilities": {}},
        },
        {"MCP-Protocol-Version": "2025-11-25"},
    )
    assert status == 200 and "Mcp-Session-Id" in hdrs
    assert body["result"]["protocolVersion"] == "2025-11-25"
