#!/usr/bin/env python3
"""Drive statspai-mcp with the official MCP Python SDK, both protocol eras.

Run from any environment that has the official SDK (``pip install mcp``,
2.3 or later speaks 2026-07-28); the server runs under ``--server-python``,
an interpreter that has StatsPAI::

    uv venv /tmp/sdkenv -p 3.12 && uv pip install -p /tmp/sdkenv/bin/python mcp
    /tmp/sdkenv/bin/python scripts/mcp_sdk_interop.py \\
        --server-python .venv/bin/python [--src src]

For stdio and Streamable HTTP, and for each SDK connect mode (``auto``
probes ``server/discover`` and falls back to ``initialize``; ``2026-07-28``
pins the per-request revision; ``legacy`` forces the handshake), it loads
a small staggered panel, fits Callaway-Sant'Anna with a handle, runs honest
DiD from the handle, lists tools / resources / prompts and reads
``statspai://handles``. Exit 1 if any combination fails or the six ATTs
differ. The SDK is not a StatsPAI dependency, so this is a script, not a
test; 2026-10-16 with ``mcp`` 2.3.0 all six passed (before the dual-era
change the pinned 2026-07-28 client could not list tools over stdio and was
refused over HTTP).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _panel_csv(path: Path) -> None:
    rows = ["id,year,y,g"]
    for i in range(60):
        g = 2003 if i < 20 else (2004 if i < 40 else 0)
        for t in range(2000, 2006):
            treated = g and t >= g
            y = 1.0 + 0.1 * i + 0.2 * (t - 2000) + (0.5 if treated else 0.0)
            y += ((i * 7 + t * 3) % 11) / 50.0
            rows.append(f"{i},{t},{y:.4f},{g}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


async def _session(server: object, mode: str, data: Path) -> dict:
    from mcp import Client

    out: dict = {"mode": mode}
    async with Client(server, mode=mode) as c:
        out["protocol"] = c.session.protocol_version
        out["tools"] = len((await c.list_tools()).tools)
        r = await c.call_tool("load_data", {"data_path": str(data)})
        did = r.structured_content["data_id"]
        r = await c.call_tool(
            "callaway_santanna",
            {
                "data_id": did,
                "y": "y",
                "g": "g",
                "t": "year",
                "i": "id",
                "as_handle": True,
                "detail": "minimal",
            },
        )
        if r.is_error:
            raise RuntimeError(json.dumps(r.structured_content)[:400])
        out["att"] = round(r.structured_content["estimate"], 10)
        rid = r.structured_content["result_id"]
        r = await c.call_tool("honest_did_from_result", {"result_id": rid})
        out["honest_rows"] = len(r.structured_content["rows"])
        out["resources"] = len((await c.list_resources()).resources)
        handles = await c.read_resource("statspai://handles")
        out["handle_listed"] = rid in handles.contents[0].text
        out["prompts"] = len((await c.list_prompts()).prompts)
    return out


def _innermost(exc: BaseException) -> BaseException:
    while hasattr(exc, "exceptions") and exc.exceptions:  # ExceptionGroup
        exc = exc.exceptions[0]
    return exc


async def _main(args: argparse.Namespace) -> int:
    from mcp.client.stdio import StdioServerParameters

    tmp = Path(tempfile.mkdtemp(prefix="statspai-mcp-interop-"))
    data = tmp / "panel.csv"
    _panel_csv(data)
    env = {**os.environ, "STATSPAI_MCP_DATA_ROOTS": str(tmp)}
    if args.src:
        env["PYTHONPATH"] = str(Path(args.src).resolve())
    cmd = [args.server_python, "-m", "statspai.agent.mcp_server"]
    results = []

    async def run(transport: str, server: object) -> None:
        for mode in ("auto", "2026-07-28", "legacy"):
            started = time.time()
            try:
                row = await _session(server, mode, data)
            except Exception as exc:  # report every combination
                inner = _innermost(exc)
                row = {"mode": mode, "failed": f"{type(inner).__name__}: {inner}"}
            row["transport"] = transport
            row["seconds"] = round(time.time() - started, 1)
            results.append(row)
            print(json.dumps(row))

    await run(
        "stdio",
        StdioServerParameters(command=cmd[0], args=cmd[1:], env=env),
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    proc = subprocess.Popen(
        cmd + ["--transport", "http", "--port", str(port)],
        env=env,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(150):
            try:
                socket.create_connection(("127.0.0.1", port), 0.2).close()
                break
            except OSError:
                time.sleep(0.2)
        await run("http", f"http://127.0.0.1:{port}/mcp")
    finally:
        proc.terminate()
        proc.wait(10)
    failed = [r for r in results if "failed" in r]
    atts = {r["att"] for r in results if "att" in r}
    if failed or len(atts) != 1:
        print(f"FAIL: {len(failed)} combination(s) failed; ATTs {sorted(atts)}")
        return 1
    print(f"OK: {len(results)} combinations, ATT {atts.pop()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--server-python",
        required=True,
        help="interpreter with StatsPAI that runs the server",
    )
    parser.add_argument(
        "--src", help="put this source tree on the server's PYTHONPATH (worktrees)"
    )
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
