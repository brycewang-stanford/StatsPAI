"""Follow-up calls over a real ``python -m statspai.agent.mcp_server`` pipe.

The 2026-11-13 agent probe ran sixteen analyses through the server and
dispatched every ``next_calls`` item marked ``ready``. Several failed, some
answered about another model, and ``audit_result`` listed a check as
missing after the agent had run it. This file runs the same loops in one
server process and pins:

* every ``ready=true`` follow-up of a fit dispatches without an error;
* an IV follow-up reports the coefficient of the fitted model;
* a plotting function called by name returns an image block;
* ``audit_result`` changes when a suggested check is run, and only then;
* a result's ``citation_key`` resolves through ``bibtex``;
* a weak-instrument-robust interval arrives with its endpoints.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest

import statspai as sp


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
            close_fds=False,  # posix_spawn: see tests/test_mcp_stdio_chain.py
            env=env,
        )
        self._next_id = 1

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
        deadline = time.monotonic() + 180
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
                continue
            if msg.get("id") == rid:
                return msg
        raise TimeoutError(f"no response to {method}")

    def tool(self, name: str, **arguments: Any) -> Dict[str, Any]:
        msg = self.rpc("tools/call", {"name": name, "arguments": arguments})
        assert "result" in msg, msg
        return msg["result"]

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    root = tmp_path_factory.mktemp("probe")
    rng = np.random.default_rng(20261113)

    n = 600
    z, u, x = rng.normal(size=(3, n))
    d = 0.8 * z + u + rng.normal(size=n)
    iv = pd.DataFrame(
        {"y": 1 + 2 * d + x + u + rng.normal(size=n), "d": d, "z": z, "x": x}
    )

    r = rng.uniform(-1, 1, 1500)
    take = (rng.random(1500) < np.where(r >= 0, 0.8, 0.2)).astype(int)
    rd = pd.DataFrame(
        {
            "score": r,
            "takeup": take,
            "y": 1 + 0.8 * r + 3 * take + rng.normal(size=1500),
        }
    )

    m = 200
    g = rng.choice([0, 2013, 2016], size=m, p=[0.4, 0.3, 0.3])
    a = rng.normal(size=m)
    stag = pd.concat(
        [
            pd.DataFrame(
                {
                    "id": np.arange(m),
                    "year": year,
                    "first_treat": g,
                    "d": ((g > 0) & (year >= g)).astype(int),
                    "y": a
                    + 0.2 * (year - 2010)
                    + ((g > 0) & (year >= g)) * 1.0
                    + rng.normal(size=m),
                }
            )
            for year in range(2010, 2020)
        ],
        ignore_index=True,
    )

    k = 800
    x1, x2 = rng.normal(size=(2, k))
    t = (rng.random(k) < 1 / (1 + np.exp(-0.6 * x1))).astype(int)
    obs = pd.DataFrame(
        {
            "y": 1 + 1.5 * t + x1 + 0.5 * x2 + rng.normal(size=k),
            "d": t,
            "x1": x1,
            "x2": x2,
        }
    )

    paths = {}
    for name, frame in (("iv", iv), ("rd", rd), ("stag", stag), ("obs", obs)):
        paths[name] = str(root / f"{name}.csv")
        frame.to_csv(paths[name], index=False)
    return paths, {"iv": iv, "rd": rd, "stag": stag, "obs": obs}


@pytest.fixture(scope="module")
def srv():
    server = _Server()
    server.rpc(
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "pytest-followups", "version": "0"},
        },
    )
    yield server
    server.close()


def _dispatch_ready(srv, fit):
    """Dispatch every ready follow-up; return ``{tool: result}``."""
    out = {}
    for call in fit["structuredContent"].get("next_calls", []):
        if call.get("ready"):
            out[call["tool"]] = srv.tool(call["tool"], **call["arguments"])
    return out


def test_iv_followups_run_and_describe_the_fitted_model(srv, frames):
    paths, data = frames
    fit = srv.tool(
        "ivreg", formula="y ~ x + (d ~ z)", data_path=paths["iv"], as_handle=True
    )
    sc = fit["structuredContent"]
    tools = [c["tool"] for c in sc["next_calls"]]
    assert tools == ["audit_result", "effective_f_test", "anderson_rubin_test"]
    assert all(c["ready"] for c in sc["next_calls"])
    results = _dispatch_ready(srv, fit)
    assert not any(r.get("isError") for r in results.values())
    direct = sp.ivreg("y ~ x + (d ~ z)", data=data["iv"])
    ar = results["anderson_rubin_test"]["structuredContent"]
    assert ar["beta_2sls"] == pytest.approx(float(direct.params["d"]), rel=1e-9)
    assert sc["coefficients"]["d"]["estimate"] == pytest.approx(
        float(direct.params["d"]), rel=1e-12
    )

    # the audit sees the check once it has been run, and not before
    rid = sc["result_id"]
    before = {
        c["name"]: c["status"]
        for c in results["audit_result"]["structuredContent"]["checks"]
    }
    after_report = srv.tool("audit_result", result_id=rid)["structuredContent"]
    after = {c["name"]: c["status"] for c in after_report["checks"]}
    assert before["anderson_rubin_ci"] == "missing"
    assert after["anderson_rubin_ci"] == "passed"
    assert after_report["summary"]["missing"] == 0


def test_weak_iv_interval_arrives_with_its_endpoints(srv, frames):
    paths, data = frames
    out = srv.tool(
        "anderson_rubin_ci",
        y="y",
        endog="d",
        instruments=["z"],
        exog=["x"],
        data_path=paths["iv"],
    )
    assert not out.get("isError")
    sc = out["structuredContent"]
    direct = sp.anderson_rubin_ci("y", "d", ["z"], exog=["x"], data=data["iv"])
    assert sc["lower"] == pytest.approx(direct.lower, rel=1e-9)
    assert sc["upper"] == pytest.approx(direct.upper, rel=1e-9)
    assert sc["is_empty"] is False


def test_fuzzy_rd_followups_keep_the_design_and_the_plot_is_a_picture(srv, frames):
    paths, data = frames
    fit = srv.tool(
        "rdrobust",
        y="y",
        x="score",
        c=0,
        fuzzy="takeup",
        data_path=paths["rd"],
        as_handle=True,
    )
    sc = fit["structuredContent"]
    results = _dispatch_ready(srv, fit)
    assert set(results) == {
        "audit_result",
        "rdplot",
        "rddensity",
        "rdbwsensitivity",
        "rdplacebo",
    }
    assert not any(r.get("isError") for r in results.values())

    image = [b for b in results["rdplot"]["content"] if b["type"] == "image"]
    assert len(image) == 1 and image[0]["mimeType"] == "image/png"
    assert base64.b64decode(image[0]["data"])[:8] == b"\x89PNG\r\n\x1a\n"

    # the placebo table's row at the true cutoff is the fuzzy estimate
    placebo = results["rdplacebo"]["structuredContent"]
    at_cutoff = [i for i, c in placebo["cutoff"].items() if c == 0][0]
    assert placebo["estimate"][at_cutoff] == pytest.approx(sc["estimate"], rel=1e-9)
    direct = sp.rdrobust(data["rd"], y="y", x="score", c=0, fuzzy="takeup")
    assert sc["estimate"] == pytest.approx(direct.estimate, rel=1e-12)

    rid = sc["result_id"]
    report = srv.tool("audit_result", result_id=rid)["structuredContent"]
    status = {c["name"]: c["status"] for c in report["checks"]}
    assert status["bandwidth_sensitivity"] == "passed"
    assert status["placebo_cutoff"] == "passed"


def test_did_audit_follows_what_the_session_ran(srv, frames):
    paths, data = frames
    fit = srv.tool(
        "callaway_santanna",
        y="y",
        g="first_treat",
        t="year",
        i="id",
        data_path=paths["stag"],
        as_handle=True,
    )
    sc = fit["structuredContent"]
    rid = sc["result_id"]
    first = srv.tool("audit_result", result_id=rid)["structuredContent"]
    status = {c["name"]: c["status"] for c in first["checks"]}
    assert status["honest_did"] == "missing"
    assert status["bacon_decomposition"] == "missing"

    honest = srv.tool("honest_did_from_result", result_id=rid, method="SD")
    assert not honest.get("isError")
    mid = srv.tool("audit_result", result_id=rid)["structuredContent"]
    status = {c["name"]: c["status"] for c in mid["checks"]}
    assert status["honest_did"] == "passed"
    assert status["bacon_decomposition"] == "missing"  # not run yet

    # a decomposition of another outcome does not count for this fit
    other = srv.tool(
        "bacon_decomposition",
        y="d",
        treat="d",
        time="year",
        id="id",
        data_path=paths["stag"],
    )
    still = srv.tool("audit_result", result_id=rid)["structuredContent"]
    assert {c["name"]: c["status"] for c in still["checks"]}["bacon_decomposition"] == (
        "missing"
    ), other["structuredContent"].get("error")

    bacon = srv.tool(
        "bacon_decomposition",
        y="y",
        treat="d",
        time="year",
        id="id",
        data_path=paths["stag"],
    )
    assert not bacon.get("isError")
    last = srv.tool("audit_result", result_id=rid)["structuredContent"]
    assert {c["name"]: c["status"] for c in last["checks"]}["bacon_decomposition"] == (
        "passed"
    )
    assert last["summary"]["missing"] == 0
    direct = sp.callaway_santanna(
        data["stag"], y="y", g="first_treat", t="year", i="id"
    )
    assert sc["estimate"] == pytest.approx(direct.estimate, rel=1e-12)

    # the result's citation_key is an estimator name; bibtex resolves it
    bib = srv.tool("bibtex", keys=[sc["citation_key"]])["structuredContent"]
    assert bib["resolved_from_function"] == {
        sc["citation_key"]: ["callaway2021difference"]
    }
    assert bib["unknown_keys"] == []


def test_two_period_did_offers_nothing_that_needs_a_pre_period(srv, frames):
    paths, data = frames
    two = data["stag"].query("year in (2012, 2013) and first_treat in (0, 2013)").copy()
    two["post"] = (two["year"] == 2013).astype(int)
    two["treated"] = (two["first_treat"] == 2013).astype(int)
    fit = srv.tool(
        "did",
        y="y",
        treat="treated",
        time="post",
        data_records=two[["y", "treated", "post"]].to_dict(orient="records"),
        as_handle=True,
    )
    sc = fit["structuredContent"]
    # the 2x2 estimate is the difference of the four cell-mean differences
    cell = two.groupby(["treated", "post"])["y"].mean()
    by_hand = (cell[1, 1] - cell[1, 0]) - (cell[0, 1] - cell[0, 0])
    assert sc["estimate"] == pytest.approx(by_hand, rel=1e-10)
    tools = {c["tool"]: c for c in sc["next_calls"]}
    assert "honest_did_from_result" not in tools
    results = _dispatch_ready(srv, fit)
    assert not any(r.get("isError") for r in results.values())
    report = results["audit_result"]["structuredContent"]
    # untested stays missing; nothing that cannot run is suggested
    assert [c["suggest_function"] for c in report["checks"]] == [None, None, None]
    assert {c["name"]: c["status"] for c in report["checks"]} == {
        "parallel_trends": "missing",
        "honest_did": "missing",
        "bacon_decomposition": "not_applicable",
    }
    assert (
        sc["result_card"]["inference"]["reference_distribution"] == f"t({len(two) - 4})"
    )


def test_match_fit_reports_balance_once_and_correctly(srv, frames):
    paths, data = frames
    fit = srv.tool(
        "match",
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        data_path=paths["obs"],
        as_handle=True,
    )
    sc = fit["structuredContent"]
    assert (
        sc["diagnostics"]["max_abs_smd_after"]
        < 0.25
        < sc["diagnostics"]["max_abs_smd_before"]
    )
    # One threshold since the probe's owner decision 6: the fit's own
    # violations() and the audit judge the matched |SMD| at 0.10 (this
    # test used to require silence below 0.25).
    after = sc["diagnostics"]["max_abs_smd_after"]
    flags = [v for v in sc["violations"] if v["test"] == "balance"]
    assert bool(flags) == (after > 0.10)
    assert all(v["value"] == after and v["threshold"] == 0.10 for v in flags)
    results = _dispatch_ready(srv, fit)
    assert set(results) == {"audit_result", "plot_from_result"}
    assert not any(r.get("isError") for r in results.values())
    checks = {
        c["name"]: c for c in results["audit_result"]["structuredContent"]["checks"]
    }
    balance = checks["balance_after"]
    # the audit reads the matched sample's SMD and judges it at the same
    # threshold
    assert balance["value"] == after and balance["threshold"] == 0.10
    assert balance["status"] == ("failed" if flags else "passed")
    assert [b["type"] for b in results["plot_from_result"]["content"]] == [
        "text",
        "image",
    ]


def test_an_error_stays_small(srv, frames):
    paths, _ = frames
    out = srv.tool("match", y="y", treat="y", covariates=["x1"], data_path=paths["obs"])
    assert out["isError"] is True
    assert len(json.dumps(out)) < 12_000
    rem = out["structuredContent"]["remediation"]
    assert rem["fix"].startswith("Matching assumes a binary treatment")
