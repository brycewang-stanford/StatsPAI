"""MCP defects found by the 2026-10-16 agent audit, each pinned to a value.

Every test drives the real JSON-RPC handler and compares what comes back
with the same computation made directly in Python, so a tool that answers
without computing (or computes something else) fails here.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent import mcp_server as m
from statspai.agent._prompts import PROMPTS
from statspai.agent._result_cache import RESULT_CACHE
from statspai.agent.mcp_server import handle_request

os.environ.setdefault("MPLBACKEND", "Agg")

SRC = Path(__file__).resolve().parents[1] / "src"


def _rpc(method, params, _counter=[0]):
    _counter[0] += 1
    raw = handle_request(
        json.dumps(
            {"jsonrpc": "2.0", "id": _counter[0], "method": method, "params": params}
        )
    )
    return json.loads(raw)


def _call(name, arguments):
    reply = _rpc("tools/call", {"name": name, "arguments": arguments})
    assert "result" in reply, reply
    return reply["result"]


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def mpdta():
    df = _quiet(sp.datasets.mpdta)
    df["region"] = df["countyreal"] % 20
    return df


@pytest.fixture(scope="module")
def mpdta_id(mpdta, tmp_path_factory):
    path = tmp_path_factory.mktemp("w3") / "mpdta.csv"
    mpdta.to_csv(path, index=False)
    out = _call("load_data", {"data_path": str(path), "detail": "minimal"})
    return out["structuredContent"]["data_id"]


@pytest.fixture(scope="module")
def nsw():
    return _quiet(sp.datasets.nsw_dw)


@pytest.fixture(scope="module")
def nsw_id(nsw, tmp_path_factory):
    path = tmp_path_factory.mktemp("w3") / "nsw.csv"
    nsw.to_csv(path, index=False)
    out = _call("load_data", {"data_path": str(path), "detail": "minimal"})
    return out["structuredContent"]["data_id"]


_CS = {"y": "lemp", "g": "first_treat", "t": "year", "i": "countyreal"}


# ---------------------------------------------------------------------------
# 1. An argument the function cannot bind refuses the call
# ---------------------------------------------------------------------------


def test_misspelt_cluster_is_refused_and_the_suggestion_clusters(mpdta, mpdta_id):
    out = _call(
        "callaway_santanna",
        {**_CS, "data_id": mpdta_id, "cluster_var": "region", "detail": "minimal"},
    )
    body = out["structuredContent"]
    assert out["isError"] is True
    assert body["error_kind"] == "unknown_argument"
    assert body["did_you_mean"]["cluster_var"][0] == "clustervars"
    assert "estimate" not in body  # the call did not run unclustered

    fixed = _call(
        "callaway_santanna",
        {**_CS, "data_id": mpdta_id, "clustervars": "region", "detail": "minimal"},
    )["structuredContent"]
    ref = _quiet(sp.callaway_santanna, mpdta, **_CS, clustervars="region")
    plain = _quiet(sp.callaway_santanna, mpdta, **_CS)
    assert fixed["se"] == pytest.approx(ref.se, rel=1e-12)
    assert abs(fixed["se"] - plain.se) > 1e-4  # clustering changed the SE


def test_alias_spelling_binds_instead_of_being_dropped(mpdta, mpdta_id):
    # ``cluster`` is an @accepts_aliases spelling of ``clustervars``; it used
    # to be dropped on the curated path, giving unclustered SEs.
    body = _call(
        "callaway_santanna",
        {**_CS, "data_id": mpdta_id, "cluster": "region", "detail": "minimal"},
    )["structuredContent"]
    ref = _quiet(sp.callaway_santanna, mpdta, **_CS, clustervars="region")
    assert body["se"] == pytest.approx(ref.se, rel=1e-12)


def test_call_function_refuses_unknown_arguments_on_the_registry_path():
    out = _call(
        "call_function",
        {
            "function": "adjust_pvalues",
            "arguments": {"pvalues": [0.01, 0.04, 0.03], "methd": "holm"},
        },
    )
    body = out["structuredContent"]
    assert body["error_kind"] == "unknown_argument"
    assert body["did_you_mean"] == {"methd": ["method"]}
    fixed = _call(
        "call_function",
        {
            "function": "adjust_pvalues",
            "arguments": {"pvalues": [0.01, 0.04, 0.03], "method": "holm"},
        },
    )["structuredContent"]
    ref = sp.adjust_pvalues([0.01, 0.04, 0.03], method="holm")
    assert fixed["value"] == pytest.approx(list(ref), rel=1e-12)
    assert fixed["value"] == pytest.approx([0.03, 0.06, 0.06], rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Staggered preflight checks the cohort column, not a binary treatment
# ---------------------------------------------------------------------------


def test_staggered_preflight_passes_on_cohort_coding(mpdta, mpdta_id):
    body = _call(
        "preflight", {**_CS, "method": "callaway_santanna", "data_id": mpdta_id}
    )["structuredContent"]
    assert body["verdict"] == "PASS", body
    cohort = next(c for c in body["checks"] if c["name"] == "cohort_coding")
    assert cohort["evidence"]["cohorts"] == [2004, 2006, 2007]
    assert cohort["evidence"]["time_min"] == 2003
    never = int((mpdta["first_treat"] == 0).sum())
    assert cohort["evidence"]["n_never_treated_rows"] == never


def test_staggered_preflight_rejects_a_time_varying_indicator(mpdta):
    out = sp.preflight(mpdta, "callaway_santanna", **{**_CS, "g": "treat"})
    check = next(c for c in out["checks"] if c["name"] == "cohort_coding")
    varying = int((mpdta.groupby("countyreal")["treat"].nunique() > 1).sum())
    assert out["verdict"] == "FAIL"
    assert check["evidence"]["n_units_varying"] == varying > 0


def test_staggered_preflight_flags_always_treated_cohorts(mpdta):
    df = mpdta.copy()
    df.loc[df["first_treat"] == 2004, "first_treat"] = 2003
    out = sp.preflight(
        df,
        "did_imputation",
        y="lemp",
        first_treat="first_treat",
        time="year",
        group="countyreal",
    )
    check = next(c for c in out["checks"] if c["name"] == "cohort_coding")
    assert out["verdict"] == "WARN"
    assert check["evidence"]["always_treated_cohorts"] == [2003]
    assert check["evidence"]["cohorts"][0] == 2003


# ---------------------------------------------------------------------------
# 3. recommend keeps the library's content and emits a runnable call
# ---------------------------------------------------------------------------


def test_recommend_entries_carry_reasoning_and_a_ready_call(nsw, nsw_id):
    args = {"y": "re78", "treatment": "treat", "covariates": ["age", "education"]}
    body = _call("recommend", {**args, "data_id": nsw_id})["structuredContent"]
    lib = _quiet(sp.recommend, nsw, **args)
    top = body["top_recommendations"][0]
    assert top["function"] == lib.recommendations[0]["function"]
    assert top["reasoning"] == lib.recommendations[0]["reason"]
    assert top["assumptions"] == lib.recommendations[0]["assumptions"]
    call = body["next_calls"][0]
    assert call["ready"] is True and call["tool"] == top["function"]
    fit = _call(call["tool"], call["arguments"])["structuredContent"]
    params = {k: v for k, v in lib.recommendations[0]["params"].items() if k != "data"}
    ref = _quiet(getattr(sp, top["function"]), nsw, **params)
    assert fit["estimate"] == pytest.approx(float(ref.estimate), rel=1e-10)


def test_recommend_names_a_derived_cohort_instead_of_calling_it_ready(mpdta, mpdta_id):
    body = _call(
        "recommend",
        {
            "y": "lemp",
            "treatment": "treat",
            "id": "countyreal",
            "time": "year",
            "data_id": mpdta_id,
        },
    )["structuredContent"]
    call = body["next_calls"][0]
    assert call["tool"] == "callaway_santanna"
    assert call["ready"] is False and "g" in call["missing_arguments"]
    assert "_cohort_treat" in call["note"]
    fit = _call(
        call["tool"], {**call["arguments"], "g": "first_treat", "detail": "minimal"}
    )["structuredContent"]
    ref = _quiet(sp.callaway_santanna, mpdta, **_CS)
    assert fit["estimate"] == pytest.approx(ref.estimate, rel=1e-12)


# ---------------------------------------------------------------------------
# 4. pipeline_did passes the unit id
# ---------------------------------------------------------------------------


def test_pipeline_did_fits_the_panel_it_was_given(mpdta, mpdta_id):
    body = _call(
        "pipeline_did",
        {
            "data_id": mpdta_id,
            "y": "lemp",
            "treat": "treat",
            "time": "year",
            "id": "countyreal",
            "detail": "minimal",
        },
    )["structuredContent"]
    stages = {s["name"]: s["status"] for s in body["stages"]}
    assert stages["estimate"] == "ok", body["stages"]
    ref = _quiet(sp.did, mpdta, y="lemp", treat="treat", time="year", id="countyreal")
    fitted = RESULT_CACHE.get(body["result_id"])
    assert fitted.estimate == pytest.approx(ref.estimate, rel=1e-12)


# ---------------------------------------------------------------------------
# 5. No data argument is required; a dataless call names the four sources
# ---------------------------------------------------------------------------


def test_data_id_only_call_runs_and_no_tool_requires_data_path(nsw, nsw_id):
    for tool in m._build_mcp_tools("curated"):
        assert "data_path" not in tool["inputSchema"]["required"], tool["name"]
    body = _call(
        "regress", {"formula": "re78 ~ treat", "data_id": nsw_id, "detail": "standard"}
    )["structuredContent"]
    ref = sp.regress("re78 ~ treat", data=nsw)
    assert body["coefficients"]["treat"]["estimate"] == pytest.approx(
        float(ref.params["treat"]), rel=1e-12
    )


@pytest.mark.parametrize(
    "tool,args",
    [
        ("regress", {"formula": "y ~ x"}),
        ("callaway_santanna", dict(_CS)),
        ("preflight", {"method": "did"}),
        ("pipeline_did", {"y": "y", "treat": "d", "time": "t"}),
        ("call_function", {"function": "regress", "arguments": {"formula": "y ~ x"}}),
    ],
)
def test_a_call_without_data_names_the_four_sources(tool, args):
    body = _call(tool, args)["structuredContent"]
    assert body["error_kind"] == "missing_data", body
    assert body["data_sources"] == ["data_id", "data_path", "data_records", "data_csv"]
    assert len(body["data_sources"]) == 4


# ---------------------------------------------------------------------------
# 6. The core profile lists what the instructions use, and nothing filtered
# ---------------------------------------------------------------------------


def test_core_profile_lists_every_named_tool_and_the_data_tools():
    core = {t["name"] for t in m._build_mcp_tools("core")}
    assert core == set(m._CORE_PROFILE_TOOLS)
    instructions = _rpc("initialize", {"protocolVersion": m.MCP_PROTOCOL_VERSION})[
        "result"
    ]["instructions"]
    curated = {t["name"] for t in m._build_mcp_tools("curated")}
    # Tool names in the text: snake_case identifiers, plus the three
    # one-word tools the workflow steps name ("causal" / "sensitivity"
    # also occur there as plain English).
    words = set(re.findall(r"[a-z_]+", instructions))
    named = {
        w
        for w in words & curated
        if "_" in w or w in {"preflight", "recommend", "bibtex"}
    }
    assert {"load_data", "describe_data", "transform_data"} <= named
    assert named <= core, sorted(named - core)
    assert len(core) == 25


# ---------------------------------------------------------------------------
# 7. Plots render on the worker thread (macOS default backend refuses)
# ---------------------------------------------------------------------------


def test_server_selects_agg_without_importing_matplotlib(tmp_path):
    code = textwrap.dedent("""
        import json, os, sys
        import statspai as sp
        from statspai.agent.mcp_server import handle_request
        before = "matplotlib" in sys.modules
        req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "bibtex",
                          "arguments": {"keys": ["callaway2021difference"]}}}
        handle_request(json.dumps(req))
        import matplotlib
        print(json.dumps({"before": before, "env": os.environ.get("MPLBACKEND"),
                          "backend": matplotlib.get_backend().lower()}))
        """)
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == {"before": False, "env": "Agg", "backend": "agg"}
    assert len(out) == 3


def test_server_keeps_a_backend_the_user_chose(tmp_path):
    code = (
        "import json, os; from statspai.agent.mcp_server import "
        "_ensure_headless_matplotlib as f; f(); print(os.environ['MPLBACKEND'])"
    )
    env = dict(os.environ, MPLBACKEND="svg")
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert proc.stdout.strip().splitlines()[-1] == "svg"
    assert proc.returncode == 0


# ---------------------------------------------------------------------------
# 8. Unknown names come with suggestions
# ---------------------------------------------------------------------------


def test_unknown_function_and_tool_names_suggest_the_close_one():
    body = _call("call_function", {"function": "callaway_santana"})["structuredContent"]
    assert body["did_you_mean"][0] == "callaway_santanna"
    reply = _rpc("tools/call", {"name": "regres", "arguments": {}})
    assert reply["error"]["code"] == -32602
    assert reply["error"]["data"]["did_you_mean"][0] == "regress"
    assert "Did you mean: regress" in reply["error"]["message"]


# ---------------------------------------------------------------------------
# 9. Remediation is phrased in MCP arguments over MCP
# ---------------------------------------------------------------------------


def test_remediation_speaks_mcp_arguments():
    body = _call("regress", {"formula": "y ~ x"})["structuredContent"]
    rem = body["remediation"]
    assert "data_id" in rem["fix"] and "sp.regress" not in rem["fix"]
    assert rem["python_fix"] == "Pass data=... to sp.regress."
    assert rem["data_sources"][0] == "data_id"

    from statspai.agent.remediation import mcp_remediation, remediate

    raw = remediate(RuntimeError("novel failure"), {"tool": "regress"})
    assert "sp.agent.tool_manifest()" in raw["fix"]
    mcp = mcp_remediation(raw, tool="regress")
    assert "tools/list" in mcp["fix"] and "sp.agent" not in mcp["fix"]
    assert mcp["fix"].count("describe_function(name='regress')") == 1


# ---------------------------------------------------------------------------
# 10. Prompts: no literal placeholders, real audit statuses, data_id
# ---------------------------------------------------------------------------


def _prompt(name, args):
    reply = _rpc("prompts/get", {"name": name, "arguments": args})
    if "error" in reply:
        return reply["error"]
    return reply["result"]["messages"][0]["content"]["text"]


def test_optional_prompt_arguments_leave_no_placeholder():
    rd = _prompt("audit_rd_result", {"data_id": "d_1", "y": "y", "x": "x"})
    assert "{c}" not in rd and "c=" not in rd and "data_id=d_1" in rd
    rd5 = _prompt(
        "audit_rd_result", {"data_path": "/a.csv", "y": "y", "x": "x", "c": 5}
    )
    assert "c=5" in rd5 and "data_path=/a.csv" in rd5
    design = _prompt("design_then_estimate", {"data_id": "d_1", "outcome": "y"})
    assert "{treatment}" not in design and "treatment=" not in design


def test_every_prompt_renders_with_a_data_handle_and_no_leftovers():
    rendered = 0
    for p in PROMPTS:
        names = {a["name"] for a in p["arguments"]}
        args = {a["name"]: "V" for a in p["arguments"] if a["required"]}
        if "data_id" in names:
            args["data_id"] = "d_abc"
        text = _prompt(p["name"], args)
        assert isinstance(text, str), (p["name"], text)
        left = set(re.findall(r"\{(\w+)\}", text)) & (names | {"data_source"})
        assert not left, (p["name"], left)
        if "data_id" in names:
            assert "data_id=d_abc" in text, p["name"]
        rendered += 1
    assert rendered == len(PROMPTS) == 13


def test_prompt_needs_exactly_one_data_source_and_real_statuses():
    both = _prompt(
        "audit_iv_result", {"data_id": "d_1", "data_path": "/x", "formula": "y ~ x"}
    )
    neither = _prompt("audit_iv_result", {"formula": "y ~ x"})
    assert both["code"] == neither["code"] == -32602
    paper = _prompt("paper_render", {"result_id": "r_1"})
    assert "'present'" not in paper
    assert "status='passed'" in paper and "status='failed'" in paper


# ---------------------------------------------------------------------------
# 11 / 12. Serialization: no private keys, capped arrays, plain dicts
# ---------------------------------------------------------------------------


def test_private_keys_are_stripped_and_arrays_capped(mpdta_id, tmp_path):
    cs = _call(
        "callaway_santanna", {**_CS, "data_id": mpdta_id, "max_output_bytes": 0}
    )["structuredContent"]
    assert not [k for k in cs.get("diagnostics", {}) if k.startswith("_")]
    assert cs["estimate"] == pytest.approx(-0.03297651380965374, rel=1e-9)

    rng = np.random.default_rng(0)
    y = np.zeros(400)
    e = rng.normal(size=400)
    for t in range(1, 400):
        y[t] = 0.6 * y[t - 1] + e[t]
    path = tmp_path / "ar.csv"
    pd.DataFrame({"y": y}).to_csv(path, index=False)
    body = _call(
        "call_function",
        {
            "function": "arima",
            "data_path": str(path),
            "arguments": {"y": "y", "order": [1, 0, 0]},
            "max_output_bytes": 0,
        },
    )["structuredContent"]
    assert not [k for k in body if k.startswith("_")]
    capped = {c["path"]: c for c in body["arrays_capped"]}
    assert capped["/residuals"]["total"] == 400
    # The reference reads the same CSV bytes the server read.
    ref = _quiet(sp.arima, "y", data=pd.read_csv(path), order=(1, 0, 0))
    full = np.asarray(ref.to_dict(detail="agent")["residuals"], dtype=float)
    assert np.allclose(body["residuals"], full[:20], rtol=1e-12, atol=1e-12)
    assert len(json.dumps(body)) < 12_000


def test_underscore_names_inside_tables_are_data_and_kept():
    from statspai.agent.tools._helpers import _scrub_payload

    out = _scrub_payload(
        {
            "_model": "x",
            "coefficients": {"_cons": {"estimate": 1.5}},
            "diagnostics": {"_cluster_ids": [1, 2], "f": 3.0},
        },
        "agent",
    )
    assert out == {
        "coefficients": {"_cons": {"estimate": 1.5}},
        "diagnostics": {"f": 3.0},
    }
    assert out["coefficients"]["_cons"]["estimate"] == pytest.approx(1.5)


def test_plain_dict_results_serialize_and_honour_detail(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(size=n)
    x3 = x2 + rng.normal(size=n)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    path = tmp_path / "pc.csv"
    frame.to_csv(path, index=False)
    pc = _call("call_function", {"function": "pc_algorithm", "data_path": str(path)})
    body = pc["structuredContent"]
    ref = sp.pc_algorithm(frame)
    assert body["separating_sets"] == {
        "|".join(k): sorted(v) for k, v in ref["separating_sets"].items()
    }
    assert body["n_edges"] == ref["n_edges"] == 2

    d = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "d": rng.integers(0, 2, n),
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
        }
    )
    tree_path = tmp_path / "pt.csv"
    d.to_csv(tree_path, index=False)
    sizes = {}
    for detail in ("minimal", "agent"):
        body = _call(
            "call_function",
            {
                "function": "policy_tree",
                "data_path": str(tree_path),
                "arguments": {"y": "y", "treat": "d", "covariates": ["a", "b"]},
                "detail": detail,
                "max_output_bytes": 0,
            },
        )["structuredContent"]
        sizes[detail] = len(json.dumps(body))
        # Same CSV bytes as the server (the in-memory floats differ in the
        # last digit, enough to move a cross-fitted value).
        ref = _quiet(
            sp.policy_tree,
            pd.read_csv(tree_path),
            y="y",
            treat="d",
            covariates=["a", "b"],
        )
        assert body["value_policy"] == pytest.approx(ref["value_policy"], rel=1e-12)
    assert sizes["minimal"] < sizes["agent"] < 10_000


# ---------------------------------------------------------------------------
# 13. Oster / Cinelli-Hazlett from a cached fit
# ---------------------------------------------------------------------------


def test_ols_sensitivity_runs_from_a_regress_handle(nsw, nsw_id):
    fit = _call(
        "regress",
        {
            "formula": "re78 ~ treat + age + education + re74",
            "data_id": nsw_id,
            "as_handle": True,
            "detail": "minimal",
        },
    )["structuredContent"]
    rid = fit["result_id"]
    controls = ["age", "education", "re74"]
    oster = _call(
        "sensitivity_from_result",
        {"result_id": rid, "method": "oster", "treat": "treat"},
    )["structuredContent"]
    ref = sp.oster_bounds(nsw, y="re78", treat="treat", controls=controls)
    assert oster["delta_star"] == pytest.approx(ref["delta_star"], rel=1e-10)
    ch = _call(
        "sensitivity_from_result",
        {
            "result_id": rid,
            "method": "cinelli_hazlett",
            "treat": "treat",
            "benchmark_covariate": "re74",
        },
    )["structuredContent"]
    ref = sp.sensemakr(
        nsw, y="re78", treat="treat", controls=controls, benchmark=["re74"]
    )
    assert ch["rv_q"] == pytest.approx(ref["rv_q"], rel=1e-10)


def test_ols_sensitivity_on_a_cs_fit_names_what_it_needs(mpdta_id):
    fit = _call(
        "callaway_santanna",
        {**_CS, "data_id": mpdta_id, "as_handle": True, "detail": "minimal"},
    )["structuredContent"]
    assert fit["estimate"] == pytest.approx(-0.03297651380965374, rel=1e-9)
    for method in ("oster", "cinelli_hazlett"):
        body = _call(
            "sensitivity_from_result", {"result_id": fit["result_id"], "method": method}
        )["structuredContent"]
        assert body["error_kind"] == "missing_arguments"
        assert body["missing_arguments"] == ["y", "treat", "controls"]


# ---------------------------------------------------------------------------
# 14. Descriptions an MCP client can use
# ---------------------------------------------------------------------------


def test_hint_and_handle_arguments_are_described():
    from statspai.agent.tools import tool_manifest

    tools = {t["name"]: t for t in tool_manifest(curated_only=True)}
    props = tools["detect_design"]["input_schema"]["properties"]
    assert props["id_col_hint"]["description"] and props["time_col_hint"]["description"]
    brief = tools["brief"]
    assert brief["input_schema"]["properties"]["result_id"]["description"]
    assert "in scope" not in brief["description"]
    assert len(brief["description"]) < 120
