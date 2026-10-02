"""A reproduction bundle really reproduces, in a process that never saw the fit.

``replay`` is one line with ``data=data`` in it. The bundle
(``statspai://result/<id>/bundle``) is what turns a server session into
something a new process can re-run: the file and its hash, the
``transform_data`` steps, the call, and the numbers to compare
(review item A4). The central test runs load -> transform -> fit in one
server, closes it, and executes the exported script in a separate
interpreter.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from statspai.agent._replay import build_bundle, replay_transforms
from statspai.agent.mcp_server import handle_request


def _rpc(method, params):
    raw = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    return json.loads(handle_request(raw))


def _call(name, **arguments):
    return _rpc("tools/call", {"name": name, "arguments": arguments})["result"][
        "structuredContent"
    ]


def _bundle(result_id):
    out = _rpc("resources/read", {"uri": f"statspai://result/{result_id}/bundle"})
    return json.loads(out["result"]["contents"][0]["text"])


@pytest.fixture
def csv(tmp_path):
    rng = np.random.default_rng(5)
    n = 300
    frame = pd.DataFrame({"x": rng.normal(size=n), "inc": np.exp(rng.normal(size=n))})
    frame["y"] = 1 + 0.5 * frame["x"] + 0.3 * np.log(frame["inc"]) + rng.normal(size=n)
    frame.loc[::17, "inc"] = np.nan
    path = tmp_path / "survey.csv"
    frame.to_csv(path, index=False)
    return str(path), frame


def _run_script(script, tmp_path):
    path = tmp_path / "reproduce.py"
    path.write_text(script, encoding="utf-8")
    env = dict(os.environ)
    env.setdefault("STATSPAI_SKIP_RUST", "1")
    return subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        close_fds=False,  # posix_spawn; see test_mcp_stdio_subprocess
    )


def test_bundle_reruns_a_transformed_fit_in_a_new_process(csv, tmp_path):
    path, frame = csv
    data_id = _call("load_data", data_path=path)["data_id"]
    derived = _call(
        "transform_data",
        data_id=data_id,
        operations=[
            {"op": "dropna", "columns": ["inc"]},
            {"op": "assign", "column": "linc", "expr": "log(inc)"},
        ],
    )["data_id"]
    fit = _call("regress", formula="y ~ x + linc", data_id=derived, as_handle=True)
    assert fit["replay_completeness"]["level"] == "session_replayable"
    bundle = _bundle(fit["result_id"])
    assert fit["replay_completeness"]["bundle_uri"].endswith(
        f"{fit['result_id']}/bundle"
    )

    assert bundle["completeness"] == "standalone" and bundle["needs"] == []
    assert bundle["source"]["source"] == path and len(bundle["source"]["sha256"]) == 64
    assert [s["op"] for s in bundle["steps"]] == ["dropna", "assign"]
    assert bundle["call"] == "sp.regress(data=data, formula='y ~ x + linc')"
    # The numbers the re-run is held to are the ones the server reported.
    for name, row in fit["coefficients"].items():
        assert bundle["expected"]["coefficients"][name] == row["estimate"]

    done = _run_script(bundle["script"], tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "reproduced: 3 number(s) match" in done.stdout

    # And the numbers are right, not merely self-consistent.
    clean = frame.dropna(subset=["inc"])
    design = np.column_stack([np.ones(len(clean)), clean["x"], np.log(clean["inc"])])
    beta = np.linalg.lstsq(design, clean["y"].to_numpy(), rcond=None)[0]
    assert bundle["expected"]["coefficients"]["x"] == pytest.approx(beta[1], rel=1e-9)
    assert bundle["expected"]["coefficients"]["linc"] == pytest.approx(
        beta[2], rel=1e-9
    )


def test_script_refuses_a_data_file_that_changed(csv, tmp_path):
    path, frame = csv
    fit = _call("regress", formula="y ~ x", data_path=path, as_handle=True)
    script = _bundle(fit["result_id"])["script"]
    frame.assign(y=frame["y"] + 1.0).to_csv(path, index=False)
    done = _run_script(script, tmp_path)
    assert done.returncode != 0
    assert "is not the file that was analysed" in done.stderr


def test_inline_data_gives_no_script_and_says_why(csv):
    _, frame = csv
    fit = _call(
        "regress",
        formula="y ~ x",
        data_records=frame[["y", "x"]].head(60).to_dict("records"),
        as_handle=True,
    )
    bundle = _bundle(fit["result_id"])
    assert bundle["completeness"] == "call_only"
    assert "script" not in bundle
    assert "inline" in bundle["needs"][0]
    assert bundle["call"].startswith("sp.regress(")
    assert (
        bundle["expected"]["coefficients"]["x"] == fit["coefficients"]["x"]["estimate"]
    )


def test_bundle_of_a_dead_handle_is_a_resource_error():
    out = _rpc("resources/read", {"uri": "statspai://result/r_00000000/bundle"})
    assert out["error"]["code"] == -32002
    assert "bundle" in out["error"]["message"]


def test_replay_transforms_matches_the_server_side_result(csv):
    path, frame = csv
    steps = [
        {"op": "dropna", "columns": ["inc"]},
        {"op": "assign", "column": "linc", "expr": "log(inc)"},
    ]
    got = replay_transforms(frame, steps)
    assert len(got) == int(frame["inc"].notna().sum()) < len(frame)
    np.testing.assert_allclose(got["linc"], np.log(got["inc"]), rtol=0, atol=0)


def test_bundle_unit_placeholder_argument_blocks_the_script():
    bundle = build_bundle(
        result_id="r_1",
        replay="sp.regress(data=data, formula='y ~ x', weights=<Series>)  # data = ...",
        data_provenance={
            "source_type": "local",
            "source": "/d.csv",
            "sha256": "ab" * 32,
        },
        payload={"estimate": 1.5},
    )
    assert bundle["completeness"] == "call_only" and "script" not in bundle
    assert "<TypeName>" in bundle["needs"][0]
    assert bundle["expected"] == {"estimate": 1.5}
