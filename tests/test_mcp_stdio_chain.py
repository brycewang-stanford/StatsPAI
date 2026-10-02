"""One whole analysis over a real ``python -m statspai.agent.mcp_server`` pipe.

``tests/agent_eval/test_mcp_protocol_transcript.py`` drives the same kind
of chain through ``handle_request`` in-process: it proves the envelopes,
not the stdio lifecycle (reader thread, worker pool, handles living across
requests in one server process, stdout staying pure JSON-RPC). This test
runs the chain an agent actually runs, end to end, in a subprocess:

initialize -> tools/list -> route_estimator -> load_data -> transform_data
-> fit (two covariance options) -> audit_result -> brief_result
-> resources/read (data and result handles) -> tight output budget
-> stale handle.

It also pins the three things the 2026-10-02 review asked a fast gate to
hold: configuration-level evidence follows the options of *this* call, an
audit check that was not run is not reported as passed, and a result cut
to a byte budget says so.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest


class _Server:
    def __init__(self) -> None:
        env = dict(os.environ)
        env.setdefault("STATSPAI_SKIP_RUST", "1")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "statspai.agent.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            # posix_spawn, not fork: on macOS a fork taken after an
            # estimator ran in this process (earlier tests in the same
            # session) intermittently segfaults in the child before exec.
            close_fds=False,
            env=env,
        )
        self._next_id = 1
        self.unparseable: list = []

    def rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        rid = self._next_id
        self._next_id += 1
        self.proc.stdin.write(
            json.dumps(
                {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
            )
            + "\n"
        )
        self.proc.stdin.flush()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if line == "":
                if self.proc.poll() is not None:
                    err = self.proc.stderr.read() if self.proc.stderr else ""
                    raise RuntimeError(f"server exited early: {err[-2000:]}")
                time.sleep(0.01)
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self.unparseable.append(line)
                continue
            if msg.get("id") == rid:
                return msg
        raise TimeoutError(f"no response to {method}")

    def tool(self, name: str, **arguments: Any) -> Dict[str, Any]:
        msg = self.rpc("tools/call", {"name": name, "arguments": arguments})
        assert "result" in msg, msg
        result = msg["result"]
        # The text block is the same object, for clients without
        # structuredContent support.
        assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
        return result

    def close(self) -> int:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        try:
            return self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            raise


@pytest.fixture
def iv_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    z, u, x = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    d = 0.8 * z + u + rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "y": 1 + 2 * d + x + u + rng.normal(size=n),
            "d": d,
            "z": z,
            "x": x,
            "inc": np.exp(rng.normal(size=n)),
        }
    )
    path = tmp_path / "iv.csv"
    frame.to_csv(path, index=False)
    return path, frame


def test_full_analysis_chain_over_stdio(iv_csv):
    import statspai as sp

    path, frame = iv_csv
    srv = _Server()
    try:
        init = srv.rpc(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest-chain", "version": "0"},
            },
        )["result"]
        assert init["serverInfo"]["version"] == sp.__version__

        names = {t["name"] for t in srv.rpc("tools/list")["result"]["tools"]}
        chain = {
            "route_estimator",
            "load_data",
            "transform_data",
            "ivreg",
            "audit_result",
            "brief_result",
        }
        assert chain <= names, chain - names

        # -- route ----------------------------------------------------
        route = srv.tool("route_estimator", family="iv")
        assert route["isError"] is False
        assert route["structuredContent"]["family"] == "iv"
        assert route["structuredContent"]["questions"]

        # -- load -> transform (handles live in the server process) ---
        loaded = srv.tool("load_data", data_path=str(path))["structuredContent"]
        assert loaded["n_rows"] == len(frame)
        derived = srv.tool(
            "transform_data",
            data_id=loaded["data_id"],
            operations=[{"op": "assign", "column": "linc", "expr": "log(inc)"}],
        )["structuredContent"]
        assert derived["parent_id"] == loaded["data_id"]
        assert "linc" in derived["columns"]

        # -- fit: the number, against the library called directly -----
        formula = "y ~ x + linc + (d ~ z)"
        local = frame.assign(linc=np.log(frame["inc"]))
        fits = {}
        for vce in ("hc1", "hc3"):
            res = srv.tool(
                "ivreg",
                formula=formula,
                data_id=derived["data_id"],
                robust=vce,
                as_handle=True,
            )
            assert res["isError"] is False, res["structuredContent"]
            fits[vce] = res["structuredContent"]
            direct = sp.ivreg(formula, data=local, robust=vce)
            coef = fits[vce]["coefficients"]["d"]
            assert coef["estimate"] == pytest.approx(direct.params["d"], rel=1e-10)
            assert coef["std_error"] == pytest.approx(direct.std_errors["d"], rel=1e-10)

        # Lineage back to the file, with its hash.
        prov = fits["hc1"]["data_provenance"]
        assert [step["tool"] for step in prov["lineage"]] == [
            "transform_data",
            "load_data",
        ]
        assert prov["root"]["sha256"] and prov["root"]["source"] == str(path)
        assert "robust='hc1'" in fits["hc1"]["replay"]

        # -- evidence is about this call's options (review A2) --------
        ev1 = fits["hc1"]["result_card"]["evidence"]
        ev3 = fits["hc3"]["result_card"]["evidence"]
        assert ev1["level"] == ev3["level"] == "configuration"
        assert ev1["outputs"]["se"] == "reference" and ev1["status"] == "covered"
        assert ev3["outputs"]["estimate"] == "reference"
        assert ev3["outputs"]["se"] == "not_covered"
        assert ev3["status"] == "estimate_only"

        # -- audit: not run is not passed (review A3) -----------------
        rid = fits["hc1"]["result_id"]
        audit = srv.tool("audit_result", result_id=rid)["structuredContent"]
        status = {c["name"]: c["status"] for c in audit["checks"]}
        assert status["weak_instrument"] == "passed"
        assert status["anderson_rubin_ci"] == "missing"
        assert audit["summary"]["missing"] >= 1 and audit["coverage"] < 1

        # -- follow-up on the handle, and both resources --------------
        brief = srv.tool("brief_result", result_id=rid)["structuredContent"]
        assert brief["result_id"] == rid and "N=300" in brief["brief"]
        for uri, key in (
            (fits["hc1"]["result_uri"], "coefficients"),
            (derived["data_uri"], "lineage"),
        ):
            read = srv.rpc("resources/read", {"uri": uri})["result"]["contents"][0]
            assert key in json.loads(read["text"])

        # -- a budget that cuts the result says so (review M1/M2) -----
        tight = srv.tool(
            "ivreg",
            formula=formula,
            data_id=derived["data_id"],
            robust="hc3",
            max_output_bytes=1500,
        )["structuredContent"]
        ledger = tight["output_budget"]
        assert ledger["scope"] == "structuredContent"
        assert ledger["actual_bytes"] == len(
            json.dumps(tight, separators=(",", ":"), allow_nan=False)
        )
        assert tight["truncated"]
        if ledger["status"] == "truncated":
            assert ledger["actual_bytes"] <= 1500
        else:
            assert ledger["status"] == "unavoidable_overflow"
            assert ledger["oversized_fields"]
        # The evidence block is never cut to fit.
        if "result_card" in tight:
            assert tight["result_card"]["evidence"] == ev3

        # -- stale handles are errors, not empty successes ------------
        stale = srv.tool("audit_result", result_id="r_deadbeef00000000")
        assert stale["isError"] is True
        assert stale["structuredContent"]["error_kind"]
        gone = srv.rpc("resources/read", {"uri": "statspai://data/d_deadbeef"})
        assert gone["error"]["code"] == -32002

        # -- still alive, stdout was JSON-RPC only, clean exit on EOF --
        assert srv.rpc("ping")["result"] == {}
        assert srv.unparseable == []
    finally:
        code = srv.close()
    assert code == 0
