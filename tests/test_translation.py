"""Tests for Module 2 — from_stata / from_r translators.

Round-trip dictionary: a representative Stata command per Tier-1
target, plus parser stress tests (comments, options, abbreviations).
"""
from __future__ import annotations

import json

import pytest

from statspai.agent._translation import (
    from_stata,
    from_r,
    STATA_COMMAND_MAP,
    R_FUNCTION_MAP,
)
from statspai.agent._translation._stata_lexer import (
    parse as stata_parse,
    StataParseError,
)
from statspai.agent import execute_tool, mcp_handle_request


def _rpc(method, params=None, request_id=1):
    msg = {"jsonrpc": "2.0", "id": request_id, "method": method,
           "params": params or {}}
    return json.loads(mcp_handle_request(json.dumps(msg)))


# ----------------------------------------------------------------------
# Lexer
# ----------------------------------------------------------------------

class TestStataLexer:
    def test_basic_command(self):
        c = stata_parse("regress y x1 x2")
        assert c.command == "regress"
        assert c.varlist == ["y", "x1", "x2"]
        assert c.options == {}

    def test_options_parsed(self):
        c = stata_parse("xtreg y x, fe vce(cluster id)")
        assert c.command == "xtreg"
        assert c.varlist == ["y", "x"]
        assert "fe" in c.options
        assert c.options["fe"] is None
        assert c.options["vce"] == "cluster id"

    def test_nested_parens_dont_break_split(self):
        c = stata_parse("rdrobust y x, kernel(triangular) bwselect(mserd)")
        assert c.options["kernel"] == "triangular"
        assert c.options["bwselect"] == "mserd"

    def test_if_in_extracted(self):
        c = stata_parse("regress y x if year > 2010 in 1/100")
        assert c.if_cond == "year > 2010"
        assert c.in_range == "1/100"
        assert c.varlist == ["y", "x"]

    def test_comments_stripped(self):
        c = stata_parse("regress y x // baseline spec")
        assert c.varlist == ["y", "x"]
        c = stata_parse("regress y x /* TODO */")
        assert c.varlist == ["y", "x"]

    def test_multi_command_rejected(self):
        with pytest.raises(StataParseError):
            stata_parse("reg y x; reg z w")

    def test_empty_rejected(self):
        with pytest.raises(StataParseError):
            stata_parse("")
        with pytest.raises(StataParseError):
            stata_parse("   ")

    def test_prefix_handled(self):
        c = stata_parse("quietly: reg y x")
        assert c.command == "reg"


# ----------------------------------------------------------------------
# Tier-1 translation correctness
# ----------------------------------------------------------------------

#: Round-trip dictionary: (Stata command, expected sp tool, required keys
#: in arguments). Each row asserts the LLM-recoverable shape: which sp
#: function and which arguments. The exact serialisation of options
#: ("hc1" vs "robust") is locked down in dedicated tests below.
TIER1_ROUND_TRIPS = [
    # regress
    ("regress y x1 x2", "regress", {"formula": "y ~ x1 + x2"}),
    ("reg wage education experience, robust",
     "regress", {"formula": "wage ~ education + experience", "robust": "hc1"}),
    ("regress y x, vce(cluster id)",
     "regress", {"formula": "y ~ x", "cluster": "id"}),
    # xtreg
    ("xtreg y x, fe i(worker)",
     "fixest", {"formula": "y ~ x", "fe": ["worker"]}),
    ("xtreg y x, fe vce(cluster id) i(id)",
     "fixest", {"formula": "y ~ x", "fe": ["id"], "cluster": "id"}),
    # reghdfe
    ("reghdfe y x, absorb(id year) cluster(id)",
     "fixest", {"formula": "y ~ x", "fe": ["id", "year"], "cluster": "id"}),
    ("reghdfe wage edu exp, absorb(firm year)",
     "fixest", {"formula": "wage ~ edu + exp", "fe": ["firm", "year"]}),
    # ivreg2
    ("ivreg2 y x1 (d = z1 z2)",
     "ivreg", {"formula": "y ~ x1 + (d ~ z1 z2)"}),
    ("ivregress y (d = z), cluster(id)",
     "ivreg", {"formula": "y ~ (d ~ z)"}),
    # csdid
    ("csdid wage, ivar(worker_id) tvar(year) gvar(first_treat)",
     "callaway_santanna",
     {"y": "wage", "i": "worker_id", "t": "year", "g": "first_treat"}),
    # did_imputation
    ("did_imputation y, treatment(treat) horizons(0 1 2)",
     "did_imputation", {"y": "y", "treat": "treat", "horizons": [0, 1, 2]}),
    # synth
    ("synth gdp inflation unemployment, trunit(7) trperiod(1990) "
     "unit(state) time(year)",
     "synth", {"outcome": "gdp", "treated_unit": 7, "treatment_time": 1990,
                "unit": "state", "time": "year"}),
    # rdrobust
    ("rdrobust y x, c(0.5)",
     "rdrobust", {"y": "y", "x": "x", "c": 0.5}),
    ("rdrobust wage age, c(18) kernel(uniform)",
     "rdrobust", {"y": "wage", "x": "age", "c": 18.0, "kernel": "uniform"}),
]


@pytest.mark.parametrize("stata,tool,subset", TIER1_ROUND_TRIPS)
def test_tier1_round_trip(stata, tool, subset):
    out = from_stata(stata)
    assert out["ok"] is True, out
    assert out["tool"] == tool, out
    for k, v in subset.items():
        assert out["arguments"].get(k) == v, (
            f"{stata!r}: arguments[{k!r}] = "
            f"{out['arguments'].get(k)!r}, expected {v!r}")
    # python_code is always non-empty
    assert out["python_code"]
    # JSON-serializable
    json.dumps(out)


class TestStataDispatchPolicy:
    def test_unknown_returns_suggestions(self):
        out = from_stata("xtbreg y x")  # typo of xtreg
        assert out["ok"] is False
        assert "xtreg" in out["suggestions"]

    def test_empty_handled(self):
        out = from_stata("")
        assert out["ok"] is False
        assert "parse_error" in out["error"]

    def test_if_clause_surfaces_note(self):
        out = from_stata("regress y x if year > 2010")
        assert out["ok"] is True
        assert any("query" in n for n in out["notes"])


# ----------------------------------------------------------------------
# R translator
# ----------------------------------------------------------------------

R_ROUND_TRIPS = [
    ("feols(y ~ x, data = df)",
     "fixest", {"formula": "y ~ x", "fe": []}),
    ("feols(y ~ x | id, data = df)",
     "fixest", {"formula": "y ~ x", "fe": ["id"]}),
    ("feols(y ~ x | id + year, data = df, cluster = \"id\")",
     "fixest", {"formula": "y ~ x", "fe": ["id", "year"], "cluster": "id"}),
    ("feols(y ~ x | id^year | (d ~ z), data = df)",
     "fixest", {"formula": "y ~ x + (d ~ z)", "fe": ["id^year"]}),
    ("lm(y ~ x + z, data = df)",
     "regress", {"formula": "y ~ x + z"}),
    ("att_gt(yname=\"y\", gname=\"g\", tname=\"t\", idname=\"id\", data=df)",
     "callaway_santanna",
     {"y": "y", "g": "g", "t": "t", "i": "id"}),
]


@pytest.mark.parametrize("rcall,tool,subset", R_ROUND_TRIPS)
def test_r_round_trip(rcall, tool, subset):
    out = from_r(rcall)
    assert out["ok"] is True, out
    assert out["tool"] == tool, out
    for k, v in subset.items():
        assert out["arguments"].get(k) == v, (
            f"{rcall!r}: arguments[{k!r}] = "
            f"{out['arguments'].get(k)!r}, expected {v!r}")


class TestRDispatchPolicy:
    def test_unknown_returns_suggestions(self):
        out = from_r("ffeols(y ~ x, data = df)")  # typo
        assert out["ok"] is False
        assert "feols" in out["suggestions"]

    def test_non_function_call_rejected(self):
        out = from_r("y <- x + 1")
        assert out["ok"] is False


# ----------------------------------------------------------------------
# MCP integration: workflow tool dispatch
# ----------------------------------------------------------------------

class TestExecuteToolFromStata:
    def test_via_execute_tool(self):
        out = execute_tool("from_stata",
                           {"command": "reghdfe y x, absorb(id) cluster(id)"})
        assert out["ok"] is True
        assert out["tool"] == "fixest"
        assert out["source"] == "stata"

    def test_missing_command(self):
        out = execute_tool("from_stata", {})
        assert "error" in out


class TestExecuteToolFromR:
    def test_via_execute_tool(self):
        out = execute_tool("from_r",
                           {"expression": "feols(y ~ x | id, data=df)"})
        assert out["ok"] is True
        assert out["tool"] == "fixest"

    def test_missing_expression(self):
        out = execute_tool("from_r", {})
        assert "error" in out


# ----------------------------------------------------------------------
# MCP RPC: tools/list shows them, tools/call dispatches them
# ----------------------------------------------------------------------

class TestRpcSurface:
    def test_translators_in_manifest(self):
        msg = _rpc("tools/list", {})
        names = {t["name"] for t in msg["result"]["tools"]}
        assert "from_stata" in names
        assert "from_r" in names

    def test_translators_dataless(self):
        msg = _rpc("tools/list", {})
        for t in msg["result"]["tools"]:
            if t["name"] in ("from_stata", "from_r"):
                assert "data_path" not in t["inputSchema"]["required"], (
                    f"{t['name']} should be dataless")

    def test_rpc_translation(self):
        msg = _rpc("tools/call", {
            "name": "from_stata",
            "arguments": {"command": "rdrobust y x, c(0)"},
        })
        body = json.loads(msg["result"]["content"][0]["text"])
        assert body["tool"] == "rdrobust"
        assert body["arguments"]["c"] == 0.0


# ----------------------------------------------------------------------
# Coverage — every Tier-1 entry has a test
# ----------------------------------------------------------------------

class TestTier1Coverage:
    def test_every_tier1_command_has_round_trip(self):
        # Every command in the dispatch map (de-duped by handler) must
        # appear in the round-trip dictionary at least once.
        covered = set()
        for stata, _, _ in TIER1_ROUND_TRIPS:
            cmd = stata.split()[0].lower()
            covered.add(cmd)
        # Aliased entries map to the same handler — collapse to one
        # canonical representative per handler.
        canonical = {
            id(h): name for name, h in STATA_COMMAND_MAP.items()
        }
        canonicals = set(STATA_COMMAND_MAP.keys())
        # Each handler should have at least one alias covered
        # (e.g. either "regress" OR "reg" appearing in TIER1_ROUND_TRIPS
        # is sufficient).
        handler_to_aliases = {}
        for alias, h in STATA_COMMAND_MAP.items():
            handler_to_aliases.setdefault(id(h), []).append(alias)
        uncovered_handlers = [
            aliases for aliases in handler_to_aliases.values()
            if not any(a in covered for a in aliases)
        ]
        assert not uncovered_handlers, (
            f"these handler aliases have no round-trip test: "
            f"{uncovered_handlers}")
