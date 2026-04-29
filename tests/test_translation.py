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
# Tier-2 round-trip dictionary
# ----------------------------------------------------------------------

TIER2_ROUND_TRIPS = [
    # GLM family
    ("probit y x1 x2", "probit", {"formula": "y ~ x1 + x2"}),
    ("logit treated age income, vce(cluster fid)",
     "logit", {"formula": "treated ~ age + income", "cluster": "fid"}),
    ("poisson visits age, robust",
     "poisson", {"formula": "visits ~ age", "robust": "hc1"}),
    ("nbreg counts x1 x2",
     "nbreg", {"formula": "counts ~ x1 + x2"}),
    # Censored regression
    ("tobit hours wage kids, ll(0) ul(80)",
     "tobit", {"formula": "hours ~ wage + kids",
                "lower": 0.0, "upper": 80.0}),
    # Selection
    ("heckman wage education, select(employed = age kids)",
     "heckman",
     {"formula": "wage ~ education",
      "select_formula": "employed ~ age + kids"}),
    # RD ancillary
    ("rdplot y x, c(0)", "rdplot", {"y": "y", "x": "x", "c": 0.0}),
    ("rddensity x, c(0.5)", "rddensity", {"x": "x", "c": 0.5}),
    # teffects
    ("teffects ipw (y) (treat z1 z2)",
     "ipw", {"y": "y", "treat": "treat", "covariates": ["z1", "z2"]}),
    ("teffects nnmatch (y x1) (treat)",
     "match",
     {"y": "y", "treat": "treat", "method": "nn"}),
    # Postestimation
    ("margins, dydx(treat)",
     "margins", {"variables": [], "dydx": ["treat"]}),
    ("contrast x1",
     "contrast", {"terms": ["x1"]}),
    ("test x1 x2",
     "test", {"terms": ["x1", "x2"]}),
    # Panel declaration (no-op)
    ("xtset id year",
     "xtset", {"id": "id", "time": "year"}),
    ("tsset year", "xtset", {"id": "year"}),
]


TIER3_ROUND_TRIPS = [
    # Poisson HDFE
    ("ppmlhdfe trade gravity, absorb(orig dest year) cluster(orig)",
     "ppmlhdfe",
     {"formula": "trade ~ gravity",
      "fe": ["orig", "dest", "year"], "cluster": "orig"}),
    # Multinomial / ordinal
    ("mlogit choice age income, baseoutcome(1)",
     "glm", {"formula": "choice ~ age + income",
              "family": "multinomial", "base_outcome": 1}),
    ("oprobit grade x1 x2",
     "glm", {"formula": "grade ~ x1 + x2",
              "family": "ordered_probit"}),
    # Dynamic panel GMM
    ("xtabond y x1 x2, twostep robust i(firm)",
     "xtabond",
     {"y": "y", "x": ["x1", "x2"], "id": "firm",
      "twostep": True, "robust": True}),
    ("xtdpdsys y x1, twostep i(unit)",
     "xtdpdsys",
     {"y": "y", "x": ["x1"], "id": "unit", "twostep": True}),
    # Bunching
    ("bunching income, c(50000) bw(2000)",
     "bunching", {"x": "income", "c": 50000.0, "bandwidth": 2000.0}),
    # boottest (post-estimation)
    ("boottest x1=0, reps(999) cluster(id)",
     "wild_cluster_bootstrap",
     {"hypothesis": ["x1=0"], "B": 999, "cluster": "id"}),
    # mi estimate: passes through with a translation note.
    ("mi estimate: reg y x", "mi_estimate", {}),
]


@pytest.mark.parametrize("stata,tool,subset", TIER3_ROUND_TRIPS)
def test_tier3_round_trip(stata, tool, subset):
    out = from_stata(stata)
    assert out["ok"] is True, out
    assert out["tool"] == tool, out
    for k, v in subset.items():
        assert out["arguments"].get(k) == v, (
            f"{stata!r}: arguments[{k!r}] = "
            f"{out['arguments'].get(k)!r}, expected {v!r}")
    json.dumps(out)


class TestTier3EdgeCases:
    def test_mi_estimate_emits_hint(self):
        out = from_stata("mi estimate: reg y x")
        # ``mi estimate: reg`` parses as ``mi`` command (Stata abbreviation
        # parsing) → the handler emits a translation hint instead of
        # silently dropping the inner command.
        assert out["ok"] is True
        assert out["tool"] == "mi_estimate"
        assert any("inner command" in n for n in out["notes"])

    def test_xtabond_without_panel_id_emits_placeholder(self):
        out = from_stata("xtabond y x")
        assert out["ok"] is True
        # Panel id is unknown — handler leaves a placeholder + note.
        assert out["arguments"]["id"] is None
        assert any("xtset" in n for n in out["notes"])

    def test_bunching_default_cutoff(self):
        out = from_stata("bunching income")
        assert out["ok"] is True
        assert out["arguments"]["c"] == 0.0


@pytest.mark.parametrize("stata,tool,subset", TIER2_ROUND_TRIPS)
def test_tier2_round_trip(stata, tool, subset):
    out = from_stata(stata)
    assert out["ok"] is True, out
    assert out["tool"] == tool, out
    for k, v in subset.items():
        assert out["arguments"].get(k) == v, (
            f"{stata!r}: arguments[{k!r}] = "
            f"{out['arguments'].get(k)!r}, expected {v!r}")
    json.dumps(out)


class TestTier2EdgeCases:
    def test_heckman_without_select(self):
        out = from_stata("heckman y x")
        assert out["ok"] is False
        assert "select" in out["error"]

    def test_heckman_malformed_select(self):
        # No `=` in select() — must be `select(d = z)`
        out = from_stata("heckman y x, select(z1 z2)")
        assert out["ok"] is False
        assert "selectvar" in out["error"]

    def test_teffects_unsupported_method(self):
        out = from_stata("teffects ml (y x) (treat)")
        assert out["ok"] is False
        assert "ml" in out["error"]

    def test_xtset_handles_time_only_form(self):
        out = from_stata("tsset year")
        assert out["ok"] is True
        assert out["arguments"]["id"] == "year"

    def test_tobit_string_bounds_ignored(self):
        # ``ll(.)`` is Stata's missing literal; we should ignore.
        out = from_stata("tobit y x, ll(.) ul(.)")
        assert out["ok"] is True
        # Neither lower nor upper survives — that's correct.
        assert "lower" not in out["arguments"]
        assert "upper" not in out["arguments"]


# ----------------------------------------------------------------------
# Coverage — every Tier-1 entry has a test
# ----------------------------------------------------------------------

class TestStataHandlerCoverage:
    def test_every_handler_has_round_trip(self):
        # Every handler in the dispatch map (de-duped by handler id) must
        # appear in TIER1_ROUND_TRIPS or TIER2_ROUND_TRIPS at least once.
        covered = set()
        import re
        for stata, _, _ in (TIER1_ROUND_TRIPS + TIER2_ROUND_TRIPS
                             + TIER3_ROUND_TRIPS):
            head = stata.split()[0]
            # Strip trailing punctuation: ``margins, dydx(...)`` →
            # ``margins`` (the comma is the option-separator, not part
            # of the command name).
            cmd = re.sub(r"[^a-z_].*$", "", head.lower())
            covered.add(cmd)
        # Each handler should have at least one alias covered.
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
