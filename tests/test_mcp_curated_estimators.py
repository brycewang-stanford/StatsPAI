"""The curated MCP profile lists the estimators applied work reaches for.

2026-10-19 audit: the curated ``tools/list`` advertised ``nbreg`` and
``xtnbreg`` but not ``feols``, ``iv``, ``match``, ``sun_abraham`` or
``sdid``, so an agent wanting fixed effects or matching had to discover
them through ``search_functions``. The count models stay callable by name
(their hand-written schema is kept), they are just not advertised.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import statspai as sp
from statspai.agent import mcp_server as m
from statspai.agent.tools import execute_tool

ADDED = ("feols", "iv", "match", "sun_abraham", "sdid")
DEMOTED = ("nbreg", "xtnbreg")


def _names(profile):
    return {t["name"] for t in m._build_mcp_tools(profile)}


def test_curated_lists_the_core_estimators():
    names = _names("curated")
    assert set(ADDED) <= names
    assert not (set(DEMOTED) & names)


def test_demoted_tools_stay_in_the_full_profile_and_callable():
    assert set(DEMOTED) <= _names("full")
    rng = np.random.default_rng(0)
    df = sp.dgp_rct(n=400, seed=0)
    mu = np.exp(1.0 + 0.3 * df["treatment"])
    df["count"] = rng.negative_binomial(2.0, 2.0 / (2.0 + mu))
    out = execute_tool("nbreg", {"formula": "count ~ treatment"}, data=df)
    assert "error" not in out, out.get("error")


@pytest.mark.parametrize(
    "tool, arguments, check",
    [
        (
            "feols",
            {"fml": "lemp ~ treat | countyreal + year", "vcov": {"CRV1": "countyreal"}},
            lambda df: sp.feols(
                "lemp ~ treat | countyreal + year",
                data=df,
                vcov={"CRV1": "countyreal"},
            ).params["treat"],
        ),
        (
            "sun_abraham",
            {"y": "lemp", "g": "first_treat", "t": "year", "i": "countyreal"},
            lambda df: sp.sun_abraham(
                df, y="lemp", g="first_treat", t="year", i="countyreal"
            ).estimate,
        ),
    ],
)
def test_new_tools_dispatch_to_the_same_fit(tool, arguments, check):
    df = sp.datasets.mpdta()
    out = execute_tool(tool, dict(arguments), data=df)
    assert "error" not in out, out.get("error")
    # The tool runs the function it names: same number as the direct call.
    assert out["estimate"] == pytest.approx(float(check(df)), rel=1e-12)


def test_iv_and_match_tools_run():
    card = sp.datasets.card_1995()
    out = execute_tool(
        "iv",
        {"formula": "lwage ~ exper + (educ ~ nearc4 + nearc2)", "method": "liml"},
        data=card,
    )
    direct = sp.iv("lwage ~ exper + (educ ~ nearc4 + nearc2)", data=card, method="liml")
    assert out["coefficients"]["educ"]["estimate"] == pytest.approx(
        direct.params["educ"], rel=1e-12
    )
    wage = sp.cps_wage()
    out = execute_tool(
        "match",
        {"y": "log_wage", "treat": "union", "covariates": ["education", "tenure"]},
        data=wage,
    )
    ref = sp.match(
        wage, y="log_wage", treat="union", covariates=["education", "tenure"]
    )
    assert out["estimate"] == pytest.approx(ref.estimate, rel=1e-12)


def test_sdid_tool_runs():
    df = sp.datasets.california_prop99()
    args = {
        "y": "cigsale",
        "unit": "state",
        "time": "year",
        "treat": "treated",
        "se_method": "noinference",
    }
    out = execute_tool("sdid", dict(args), data=df)
    ref = sp.sdid(df, **args)
    assert out["estimate"] == pytest.approx(ref.estimate, rel=1e-12)


def test_input_schemas_name_real_parameters():
    import inspect

    from statspai.agent.tools._specs import TOOL_REGISTRY

    for spec in TOOL_REGISTRY:
        if spec["name"] not in ADDED:
            continue
        fn = getattr(sp, spec["statspai_fn"])
        params = set(inspect.signature(fn).parameters)
        params |= set(getattr(fn, "__statspai_aliases__", {}) or {})
        takes_kwargs = any(
            p.kind is p.VAR_KEYWORD for p in inspect.signature(fn).parameters.values()
        )
        for prop in spec["input_schema"]["properties"]:
            assert prop in params or takes_kwargs, (spec["name"], prop)
    json.dumps([t for t in m._build_mcp_tools("curated")])


def test_describe_function_sends_the_parameters_once():
    from statspai.agent.workflow_tools import _tool_describe_function

    out = _tool_describe_function({"name": "callaway_santanna"})
    assert "params" not in out
    props = out["schema"]["parameters"]["properties"]
    # Everything the dropped list carried is in the schema.
    for p in sp.describe_function("callaway_santanna")["params"]:
        assert p["name"] in props
        if p.get("enum"):
            assert props[p["name"]]["enum"] == p["enum"]
    assert set(out["schema"]) == {"name", "parameters"}
    full = len(json.dumps(sp.describe_function("callaway_santanna"), default=str))
    full += len(json.dumps(sp.function_schema("callaway_santanna"), default=str))
    assert len(json.dumps(out, default=str)) < full


def _rpc(method, params):
    line = m.handle_request(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    )
    return json.loads(line)


def test_detail_full_is_the_agent_payload():
    records = sp.dgp_rct(n=200, seed=1).to_dict("records")
    calls = [
        _rpc(
            "tools/call",
            {
                "name": "regress",
                "arguments": {
                    "formula": "y ~ treatment",
                    "data_records": records,
                    "detail": level,
                },
            },
        )["result"]["structuredContent"]
        for level in ("full", "agent")
    ]
    assert not calls[0].get("error"), calls[0]
    assert set(calls[0]) == set(calls[1])
    assert calls[0]["estimate"] == calls[1]["estimate"]
