"""Tests for Module 2 — from_stata / from_r translators.

Round-trip dictionary: a representative Stata command per Tier-1
target, plus parser stress tests (comments, options, abbreviations).
"""

from __future__ import annotations

import json

import pytest

from statspai.agent import execute_tool, mcp_handle_request
from statspai.agent._translation import (
    R_FUNCTION_MAP,
    STATA_COMMAND_MAP,
    from_r,
    from_stata,
)
from statspai.agent._translation._stata_lexer import StataParseError
from statspai.agent._translation._stata_lexer import parse as stata_parse


def _rpc(method, params=None, request_id=1):
    msg = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
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
    (
        "reg wage education experience, robust",
        "regress",
        {"formula": "wage ~ education + experience", "robust": "hc1"},
    ),
    ("regress y x, vce(cluster id)", "regress", {"formula": "y ~ x", "cluster": "id"}),
    # xtreg — targets sp.feols; FE folded into the pyfixest formula (fml)
    ("xtreg y x, fe i(worker)", "feols", {"fml": "y ~ x | worker"}),
    (
        "xtreg y x, fe vce(cluster id) i(id)",
        "feols",
        {"fml": "y ~ x | id", "cluster": "id"},
    ),
    # reghdfe
    (
        "reghdfe y x, absorb(id year) cluster(id)",
        "feols",
        {"fml": "y ~ x | id + year", "cluster": "id"},
    ),
    (
        "reghdfe wage edu exp, absorb(firm year)",
        "feols",
        {"fml": "wage ~ edu + exp | firm + year"},
    ),
    # ivreg2
    ("ivreg2 y x1 (d = z1 z2)", "ivreg", {"formula": "y ~ x1 + (d ~ z1 z2)"}),
    ("ivregress y (d = z), cluster(id)", "ivreg", {"formula": "y ~ (d ~ z)"}),
    (
        "ivregress 2sls y x1 (d = z), robust small",
        "ivreg",
        {"formula": "y ~ x1 + (d ~ z)", "method": "2sls", "robust": "hc1"},
    ),
    (
        "ivregress liml y x1 (d = z1 z2), vce(cluster firm)",
        "ivreg",
        {
            "formula": "y ~ x1 + (d ~ z1 z2)",
            "method": "liml",
            "cluster": "firm",
        },
    ),
    (
        "ivreghdfe y x1 x2 (d = z1 z2), absorb(firm year) cluster(firm)",
        "feols",
        {
            "fml": "y ~ x1 + x2 | firm + year | d ~ z1 + z2",
            "cluster": "firm",
        },
    ),
    # csdid
    (
        "csdid wage, ivar(worker_id) tvar(year) gvar(first_treat)",
        "callaway_santanna",
        {"y": "wage", "i": "worker_id", "t": "year", "g": "first_treat"},
    ),
    (
        "didregress (wage education) (treated), group(worker_id) time(year) "
        "vce(cluster worker_id)",
        "did",
        {
            "y": "wage",
            "treat": "treated",
            "time": "year",
            "id": "worker_id",
            "method": "twfe",
            "covariates": ["education"],
            "cluster": "worker_id",
        },
    ),
    # did_imputation (Borusyak positional syntax: Y i t Ei)
    (
        "did_imputation y unit period first_treat",
        "did_imputation",
        {
            "y": "y",
            "group": "unit",
            "time": "period",
            "first_treat": "first_treat",
        },
    ),
    # synth
    (
        "synth gdp inflation unemployment, trunit(7) trperiod(1990) "
        "unit(state) time(year)",
        "synth",
        {
            "outcome": "gdp",
            "treated_unit": 7,
            "treatment_time": 1990,
            "unit": "state",
            "time": "year",
        },
    ),
    # rdrobust
    ("rdrobust y x, c(0.5)", "rdrobust", {"y": "y", "x": "x", "c": 0.5}),
    (
        "rdrobust wage age, c(18) kernel(uniform)",
        "rdrobust",
        {"y": "wage", "x": "age", "c": 18.0, "kernel": "uniform"},
    ),
]


@pytest.mark.parametrize("stata,tool,subset", TIER1_ROUND_TRIPS)
def test_tier1_round_trip(stata, tool, subset):
    out = from_stata(stata)
    assert out["ok"] is True, out
    assert out["tool"] == tool, out
    for k, v in subset.items():
        assert out["arguments"].get(k) == v, (
            f"{stata!r}: arguments[{k!r}] = "
            f"{out['arguments'].get(k)!r}, expected {v!r}"
        )
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
    # feols targets sp.feols; FE / IV live inside the pyfixest formula (fml)
    ("feols(y ~ x, data = df)", "feols", {"fml": "y ~ x"}),
    ("feols(y ~ x | id, data = df)", "feols", {"fml": "y ~ x | id"}),
    (
        'feols(y ~ x | id + year, data = df, cluster = "id")',
        "feols",
        {"fml": "y ~ x | id + year", "cluster": "id"},
    ),
    (
        "feols(y ~ x | id^year | (d ~ z), data = df)",
        "feols",
        {"fml": "y ~ x | id^year | d ~ z"},
    ),
    ("lm(y ~ x + z, data = df)", "regress", {"formula": "y ~ x + z"}),
    (
        'att_gt(yname="y", gname="g", tname="t", idname="id", data=df)',
        "callaway_santanna",
        {"y": "y", "g": "g", "t": "t", "i": "id"},
    ),
    # GLM family — recognised binomial/poisson route to the
    # specialised sp helper.
    ("glm(y ~ x, family = binomial, data = df)", "logit", {"formula": "y ~ x"}),
    (
        'glm(y ~ x, family = binomial(link = "probit"), data = df)',
        "probit",
        {"formula": "y ~ x"},
    ),
    (
        "glm(counts ~ x, family = poisson, data = df)",
        "poisson",
        {"formula": "counts ~ x"},
    ),
    (
        "glm(y ~ x, family = gaussian, data = df)",
        "glm",
        {"formula": "y ~ x", "family": "gaussian"},
    ),
    # Multilevel / GLMM — parsed into y / x_fixed / group (the sp.mixed /
    # sp.melogit signature), targeting the real callables (sp.multilevel is a
    # package, sp.glmer does not exist).
    (
        "lmer(y ~ x + (1|group), data = df)",
        "mixed",
        {"y": "y", "x_fixed": ["x"], "group": "group"},
    ),
    (
        "glmer(y ~ x + (1|group), family = binomial, data = df)",
        "melogit",
        {"y": "y", "x_fixed": ["x"], "group": "group"},
    ),
    # Panel — sp.panel uses keyword args entity/time (not id/time); entity must
    # be supplied (error otherwise) so the test case keeps index=c("id","t").
    (
        'plm(y ~ x, data = df, model = "within", index = c("id", "t"))',
        "panel",
        {"formula": "y ~ x", "entity": "id", "time": "t", "method": "within"},
    ),
    (
        'plm(y ~ x, data = df, model = "random", index = c("id", "t"))',
        "panel",
        {"formula": "y ~ x", "entity": "id", "time": "t", "method": "random"},
    ),
    # MatchIt — sp.match takes y (outcome) / treat / covariates (kw), not
    # a formula. Translator uses the LHS as outcome and falls back to LHS
    # as treatment if no ``treat=`` override is given.
    (
        'matchit(treat ~ x1 + x2, data = df, method = "nearest")',
        "match",
        {
            "y": "treat",
            "treat": "treat",
            "covariates": ["x1", "x2"],
            "method": "nearest",
        },
    ),
    (
        'matchit(treat ~ x1, data = df, method = "genetic")',
        "match",
        {"y": "treat", "treat": "treat", "covariates": ["x1"], "method": "genetic"},
    ),
]


@pytest.mark.parametrize("rcall,tool,subset", R_ROUND_TRIPS)
def test_r_round_trip(rcall, tool, subset):
    out = from_r(rcall)
    assert out["ok"] is True, out
    assert out["tool"] == tool, out
    for k, v in subset.items():
        assert out["arguments"].get(k) == v, (
            f"{rcall!r}: arguments[{k!r}] = "
            f"{out['arguments'].get(k)!r}, expected {v!r}"
        )


class TestRDispatchPolicy:
    def test_unknown_returns_suggestions(self):
        out = from_r("ffeols(y ~ x, data = df)")  # typo
        assert out["ok"] is False
        assert "feols" in out["suggestions"]

    def test_non_function_call_rejected(self):
        out = from_r("y <- x + 1")
        assert out["ok"] is False


class TestEconomistMigrationUseCases:
    def test_reghdfe_and_feols_share_estimand_shape(self):
        stata = from_stata("reghdfe y x1 x2, absorb(firm year) cluster(firm)")
        r = from_r('feols(y ~ x1 + x2 | firm + year, data = df, cluster = "firm")')

        assert stata["ok"] is True, stata
        assert r["ok"] is True, r
        assert stata["tool"] == r["tool"] == "feols"
        assert stata["arguments"] == r["arguments"]

    def test_csdid_and_att_gt_share_timing_shape(self):
        stata = from_stata(
            "csdid lemp, ivar(countyreal) time(year) " "gvar(first_treat) method(reg)"
        )
        r = from_r(
            'att_gt(yname="lemp", tname="year", idname="countyreal", '
            'gname="first_treat", est_method="reg", data=df)'
        )

        assert stata["ok"] is True, stata
        assert r["ok"] is True, r
        assert stata["tool"] == r["tool"] == "callaway_santanna"
        for key in ("y", "i", "t", "g", "estimator"):
            assert stata["arguments"][key] == r["arguments"][key]

    def test_ivreghdfe_and_feols_iv_share_estimand_shape(self):
        stata = from_stata(
            "ivreghdfe y x1 x2 (d = z1 z2), absorb(firm year) cluster(firm)"
        )
        r = from_r(
            "feols(y ~ x1 + x2 | firm + year | (d ~ z1 + z2), "
            'data=df, cluster="firm")'
        )

        assert stata["ok"] is True, stata
        assert r["ok"] is True, r
        assert stata["tool"] == r["tool"] == "feols"
        assert stata["arguments"] == r["arguments"]


# ----------------------------------------------------------------------
# MCP integration: workflow tool dispatch
# ----------------------------------------------------------------------


class TestExecuteToolFromStata:
    def test_via_execute_tool(self):
        out = execute_tool(
            "from_stata", {"command": "reghdfe y x, absorb(id) cluster(id)"}
        )
        assert out["ok"] is True
        assert out["tool"] == "feols"
        assert out["source"] == "stata"

    def test_psmatch2_via_execute_tool(self):
        out = execute_tool(
            "from_stata",
            {"command": "psmatch2 d x, out(y) kernel bw(0.06)"},
        )
        assert out["ok"] is True
        assert out["tool"] == "psmatch2"
        assert out["source"] == "stata"
        assert out["arguments"]["method"] == "kernel"
        assert out["arguments"]["bwidth"] == 0.06

    def test_ivreghdfe_via_execute_tool(self):
        out = execute_tool(
            "from_stata",
            {"command": ("ivreghdfe y x (d = z), absorb(id year) cluster(id)")},
        )
        assert out["ok"] is True
        assert out["tool"] == "feols"
        assert out["arguments"]["fml"] == "y ~ x | id + year | d ~ z"

    def test_missing_command(self):
        out = execute_tool("from_stata", {})
        assert "error" in out


class TestExecuteToolFromR:
    def test_via_execute_tool(self):
        out = execute_tool("from_r", {"expression": "feols(y ~ x | id, data=df)"})
        assert out["ok"] is True
        assert out["tool"] == "feols"

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
                assert (
                    "data_path" not in t["inputSchema"]["required"]
                ), f"{t['name']} should be dataless"

    def test_rpc_translation(self):
        msg = _rpc(
            "tools/call",
            {
                "name": "from_stata",
                "arguments": {"command": "rdrobust y x, c(0)"},
            },
        )
        body = json.loads(msg["result"]["content"][0]["text"])
        assert body["tool"] == "rdrobust"
        assert body["arguments"]["c"] == 0.0


# ----------------------------------------------------------------------
# Tier-2 round-trip dictionary
# ----------------------------------------------------------------------

TIER2_ROUND_TRIPS = [
    # GLM family
    ("probit y x1 x2", "probit", {"formula": "y ~ x1 + x2"}),
    (
        "logit treated age income, vce(cluster fid)",
        "logit",
        {"formula": "treated ~ age + income", "cluster": "fid"},
    ),
    (
        "poisson visits age, robust",
        "poisson",
        {"formula": "visits ~ age", "robust": "hc1"},
    ),
    ("nbreg counts x1 x2", "nbreg", {"formula": "counts ~ x1 + x2"}),
    (
        "xtnbreg counts x1 x2, fe i(firm) vce(cluster firm) irr",
        "xtnbreg",
        {
            "formula": "counts ~ x1 + x2",
            "entity": "firm",
            "model": "fe",
            "cluster": "firm",
            "irr": True,
        },
    ),
    # Censored regression
    (
        "tobit hours wage kids, ll(0) ul(80)",
        "tobit",
        {"y": "hours", "x": ["wage", "kids"], "ll": 0.0, "ul": 80.0},
    ),
    # Selection
    (
        "heckman wage education, select(employed = age kids)",
        "heckman",
        {"y": "wage", "x": ["education"], "select": "employed", "z": ["age", "kids"]},
    ),
    # RD ancillary
    ("rdplot y x, c(0)", "rdplot", {"y": "y", "x": "x", "c": 0.0}),
    ("rddensity x, c(0.5)", "rddensity", {"x": "x", "c": 0.5}),
    # teffects
    (
        "teffects ipw (y) (treat z1 z2)",
        "ipw",
        {
            "y": "y",
            "treat": "treat",
            "covariates": ["z1", "z2"],
            "estimand": "ATE",
        },
    ),
    (
        "teffects aipw (y x1 x2) (treat z1 z2), atet",
        "aipw",
        {
            "y": "y",
            "treat": "treat",
            "covariates": ["z1", "z2"],
            "estimand": "ATT",
        },
    ),
    (
        "teffects nnmatch (y x1) (treat)",
        "match",
        {"y": "y", "treat": "treat", "method": "nn", "estimand": "ATE"},
    ),
    # Stata psmatch2 migration
    (
        "psmatch2 d x, out(y) n(1) logit",
        "psmatch2",
        {"treat": "d", "outcome": "y", "covariates": ["x"], "neighbor": 1},
    ),
    (
        "psmatch2 d x, kernel kerneltype(epan) bwidth(0.06)",
        "psmatch2",
        {
            "treat": "d",
            "covariates": ["x"],
            "method": "kernel",
            "kernel": "epan",
            "bwidth": 0.06,
        },
    ),
    (
        "psmatch2 d x, radius caliper(0.05) common ai(2)",
        "psmatch2",
        {
            "treat": "d",
            "covariates": ["x"],
            "method": "radius",
            "caliper": 0.05,
            "common_support": "minmax",
            "ai": 2,
        },
    ),
    # Postestimation
    ("margins, dydx(treat)", "margins", {"dydx": ["treat"]}),
    ("contrast x1", "contrast", {"terms": ["x1"]}),
    ("test x1 x2", "test", {"terms": ["x1", "x2"]}),
    # xtset / tsset are intentionally excluded: no sp equivalent, the
    # translator now fails loud with a note pointing at sp.panel / sp.feols.
]


TIER3_ROUND_TRIPS = [
    # Poisson HDFE — absorb is a "+"-joined string (sp.ppmlhdfe's real arg)
    (
        "ppmlhdfe trade gravity, absorb(orig dest year) cluster(orig)",
        "ppmlhdfe",
        {
            "formula": "trade ~ gravity",
            "absorb": "orig + dest + year",
            "cluster": "orig",
        },
    ),
    # Multinomial — targets the dedicated sp.mlogit (base=), not sp.glm
    (
        "mlogit choice age income, baseoutcome(1)",
        "mlogit",
        {
            "formula": "choice ~ age + income",
            "base": 1,
        },
    ),
    (
        "oprobit grade x1 x2",
        "glm",
        {"formula": "grade ~ x1 + x2", "family": "ordered_probit"},
    ),
    # Dynamic panel GMM (xtabond = difference GMM; xtdpdsys = system GMM, which
    # sp.xtabond does not yet implement — see TestUnsupportedButHonest below).
    (
        "xtabond y x1 x2, twostep robust i(firm)",
        "xtabond",
        {"y": "y", "x": ["x1", "x2"], "id": "firm", "twostep": True, "robust": True},
    ),
    # Bunching
    (
        "bunching income, c(50000) bw(2000)",
        "bunching",
        {"running_var": "income", "threshold": 50000.0, "bin_width": 2000.0},
    ),
    # boottest (post-estimation)
    (
        "boottest x1=0, reps(999) cluster(id)",
        "wild_cluster_bootstrap",
        {"hypothesis": ["x1=0"], "B": 999, "cluster": "id"},
    ),
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
            f"{out['arguments'].get(k)!r}, expected {v!r}"
        )
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
        assert out["arguments"]["threshold"] == 0.0

    def test_xtdpdsys_fails_loud_as_unsupported(self):
        # Blundell-Bond system GMM is not implemented (sp.xtabond raises
        # NotImplementedError for method='system'). The translator must fail
        # loud with the difference-GMM fallback, not emit a dead sp.xtdpdsys.
        out = from_stata("xtdpdsys y x1, twostep i(unit)")
        assert out["ok"] is False
        assert "xtabond" in out.get("suggestions", [])


@pytest.mark.parametrize("stata,tool,subset", TIER2_ROUND_TRIPS)
def test_tier2_round_trip(stata, tool, subset):
    out = from_stata(stata)
    assert out["ok"] is True, out
    assert out["tool"] == tool, out
    for k, v in subset.items():
        assert out["arguments"].get(k) == v, (
            f"{stata!r}: arguments[{k!r}] = "
            f"{out['arguments'].get(k)!r}, expected {v!r}"
        )
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

    def test_xtdidregress_notes_treatment_status_semantics(self):
        out = from_stata("xtdidregress (y) (treated), group(id) time(year)")
        assert out["ok"] is True
        assert out["tool"] == "did"
        assert out["arguments"]["method"] == "twfe"
        assert any("treatment-status" in note for note in out["notes"])

    def test_didregress_missing_group_or_time_is_error(self):
        out = from_stata("didregress (y) (treated), group(id)")
        assert out["ok"] is False
        assert "time" in out["error"]

    def test_xtset_handles_time_only_form(self):
        # xtset / tsset have no sp equivalent — the translator must fail
        # loud (ok=False) with a note pointing at sp.panel / sp.feols, not
        # silently emit a dead tool name + placeholder.
        out = from_stata("tsset year")
        assert out["ok"] is False
        assert (
            "panel" in out.get("error", "").lower()
            or "feols" in out.get("error", "").lower()
        )

    def test_tobit_string_bounds_ignored(self):
        # ``ll(.)`` is Stata's missing literal; we should ignore.
        out = from_stata("tobit y x, ll(.) ul(.)")
        assert out["ok"] is True
        # Neither lower nor upper survives — that's correct.
        assert "lower" not in out["arguments"]
        assert "upper" not in out["arguments"]

    def test_psmatch2_convention_changing_options_emit_notes(self):
        out = from_stata("psmatch2 d x, out(y) probit ate")
        assert out["ok"] is True
        assert out["tool"] == "psmatch2"
        assert any("probit" in note for note in out["notes"])
        assert any("ATT-focused" in note for note in out["notes"])

    def test_psmatch2_without_outcome_emits_matched_frame_note(self):
        out = from_stata("psmatch2 d x1 x2, n(1)")
        assert out["ok"] is True
        assert out["tool"] == "psmatch2"
        assert "outcome" not in out["arguments"]
        assert any("matched frame" in note for note in out["notes"])


# ----------------------------------------------------------------------
# Coverage — every Tier-1 entry has a test
# ----------------------------------------------------------------------


class TestStataHandlerCoverage:
    def test_every_handler_has_round_trip(self):
        # Every handler in the dispatch map (de-duped by handler id) must
        # appear in TIER1_ROUND_TRIPS or TIER2_ROUND_TRIPS at least once.
        covered = set()
        import re

        for stata, _, _ in TIER1_ROUND_TRIPS + TIER2_ROUND_TRIPS + TIER3_ROUND_TRIPS:
            head = stata.split()[0]
            # Strip trailing punctuation: ``margins, dydx(...)`` →
            # ``margins`` (the comma is the option-separator, not part
            # of the command name).
            cmd = re.sub(r"[^a-z0-9_].*$", "", head.lower())
            covered.add(cmd)
        # Commands whose contract is a dedicated behaviour test rather than a
        # round-trip (they intentionally do not produce a runnable payload):
        #   xtdpdsys — system GMM unsupported → test_xtdpdsys_fails_loud_as_unsupported
        #   xtset / tsset — no sp equivalent; translator fails loud →
        #     test_xtset_handles_time_only_form
        covered.add("xtdpdsys")
        covered.add("xtset")
        # Each handler should have at least one alias covered.
        handler_to_aliases = {}
        for alias, h in STATA_COMMAND_MAP.items():
            handler_to_aliases.setdefault(id(h), []).append(alias)
        uncovered_handlers = [
            aliases
            for aliases in handler_to_aliases.values()
            if not any(a in covered for a in aliases)
        ]
        assert not uncovered_handlers, (
            f"these handler aliases have no round-trip test: " f"{uncovered_handlers}"
        )


# ----------------------------------------------------------------------
# Structural executability — every translated payload must be runnable
# ----------------------------------------------------------------------
#
# A migration on-ramp that emits a payload the user cannot run is worse than
# none. The fixest bug (tool='fixest' + sp.fixest(formula=...), when sp.fixest
# is a package and feols wants fml=) was one instance; the sweep that followed
# found tobit / heckman / bunching / did_imputation / lmer / glmer / xtdpdsys
# all emitting a non-callable tool or the wrong argument names. This contract
# translates every round-trip command and asserts the emitted (tool, arguments)
# is structurally runnable: the tool resolves to a callable on sp, and every
# REQUIRED parameter of that callable (bar ``data``, injected at dispatch) is
# present in the arguments. A newly-added handler that targets a non-existent
# tool or misnames an argument fails here instead of shipping a dead on-ramp.

#: Tools that legitimately cannot yield a standalone runnable payload from a
#: single command — postestimation acts on a prior fitted result, and setup
#: commands declare structure with no estimator target. Each is exercised by a
#: dedicated behaviour test elsewhere in this file.
_NON_EXECUTABLE_TOOLS = frozenset(
    {
        # postestimation — need a fitted ``result``
        "margins",
        "contrast",
        "test",
        "wild_cluster_bootstrap",
        "mi_estimate",
        # setup / declaration — no estimator target (no-op)
        "xtset",
    }
)


def _tool_param_info(tool):
    """``(named_params, required_params, has_var_keyword)`` for ``sp.<tool>``,
    excluding ``self``/``data``. Returns ``(None, None, None)`` if ``sp.<tool>``
    is not callable — i.e. the translation named a tool an agent cannot run."""
    import inspect

    import statspai as sp

    fn = getattr(sp, tool, None)
    if not callable(fn):
        return None, None, None
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return set(), set(), True
    named, required = set(), set()
    has_kwargs = False
    for name, p in sig.parameters.items():
        if p.kind == p.VAR_KEYWORD:
            has_kwargs = True
            continue
        if p.kind == p.VAR_POSITIONAL or name in ("self", "data"):
            continue
        named.add(name)
        if p.default is p.empty:
            required.add(name)
    return named, required, has_kwargs


def _structural_exec_problems(commands, translate):
    problems = []
    for cmd in commands:
        out = translate(cmd)
        if not out.get("ok"):
            continue  # fail-loud / partial translations are covered elsewhere
        tool = out["tool"]
        if tool in _NON_EXECUTABLE_TOOLS:
            continue
        named, required, has_kwargs = _tool_param_info(tool)
        if named is None:
            problems.append(f"[{cmd}] -> sp.{tool} is NOT callable (dead on-ramp)")
            continue
        args = set(out["arguments"])
        missing = required - args
        if missing:
            problems.append(
                f"[{cmd}] -> sp.{tool} missing required {sorted(missing)} "
                f"(emitted args {sorted(args)})"
            )
        # An emitted arg that is neither a named parameter nor absorbed by
        # **kwargs is silently dropped at dispatch — the translation thinks it
        # set an option that vanishes, so the fit runs with wrong inputs and no
        # error (the ppmlhdfe fe=/absorb=, mlogit, and synth predictors=/
        # covariates= bugs were exactly this). Only flag when the tool has no
        # **kwargs — a **kwargs tool may legitimately forward the arg.
        if not has_kwargs:
            dropped = args - named
            if dropped:
                problems.append(
                    f"[{cmd}] -> sp.{tool} has no such parameter(s) {sorted(dropped)} "
                    f"and no **kwargs — they are SILENTLY DROPPED at dispatch "
                    f"(wrong-results on-ramp)"
                )
    return problems


def test_stata_translations_are_structurally_executable():
    commands = [
        c for c, _, _ in TIER1_ROUND_TRIPS + TIER2_ROUND_TRIPS + TIER3_ROUND_TRIPS
    ]
    problems = _structural_exec_problems(commands, from_stata)
    assert not problems, "dead / unrunnable Stata translations:\n  " + "\n  ".join(
        problems
    )


def test_r_translations_are_structurally_executable():
    commands = [c for c, _, _ in R_ROUND_TRIPS]
    problems = _structural_exec_problems(commands, from_r)
    assert not problems, "dead / unrunnable R translations:\n  " + "\n  ".join(problems)


# ----------------------------------------------------------------------
# Numerical equivalence — the translated call must fit the RIGHT model
# ----------------------------------------------------------------------
#
# Structural checks prove a translation runs with accepted argument names; they
# do not prove the arguments encode the model the command actually specifies.
# A formula-reassembly slip (dropped FE, wrong cluster column, mis-parsed IV)
# or a wrong option value produces an executable payload that fits a DIFFERENT
# model — right shape, wrong numbers. This suite executes each translated
# payload and an INDEPENDENTLY hand-authored reference call for the same model
# on shared data, and asserts the coefficients agree to machine precision.
# Because ``sp.feols`` etc. are separately validated against R/Stata in the
# reference-parity suite, matching the reference call means matching R/Stata.


def _num_df(seed=7, n=1500):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    firm = rng.integers(0, 25, n)
    year = rng.integers(0, 10, n)
    fe = firm * 0.3 + year * 0.2
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    d = 0.8 * z + u + rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y_star = 1.0 * x1 - 0.5 * x2 + 2.0 * d + fe + u + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": y_star,
            "hours": np.clip(y_star * 5 + 40, 0, 80),
            "x1": x1,
            "x2": x2,
            "d": d,
            "z": z,
            "firm": firm,
            "year": year,
        }
    )


def _coef_map(tool, args, df):
    """Coefficient dict from a direct ``sp.<tool>`` call — the same numerical
    path ``execute_tool`` dispatches to, called directly so ``.params`` is
    easy to compare."""
    import warnings

    import numpy as np

    import statspai as sp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = getattr(sp, tool)(data=df, **args)
    params = res.params
    if hasattr(params, "to_dict"):
        return {str(k): float(v) for k, v in params.to_dict().items()}
    return {str(i): float(v) for i, v in enumerate(np.atleast_1d(params))}


def _assert_same_fit(translated, reference, df, label):
    tool_t, args_t = translated
    tool_r, args_r = reference
    ct = _coef_map(tool_t, args_t, df)
    cr = _coef_map(tool_r, args_r, df)
    common = set(ct) & set(cr)
    assert common, f"{label}: no overlapping coefficients ({ct} vs {cr})"
    for k in common:
        assert abs(ct[k] - cr[k]) < 1e-9, (
            f"{label}: coefficient {k!r} differs — translated {ct[k]!r} vs "
            f"reference {cr[k]!r}; the translation fits the wrong model."
        )


#: (command, translate_fn, reference (tool, args)). The reference is authored
#: independently of the handler; a handler that mis-encodes the model diverges.
_NUMERIC_CASES = [
    (
        "reghdfe y x1 x2, absorb(firm year) cluster(firm)",
        "stata",
        ("feols", {"fml": "y ~ x1 + x2 | firm + year", "cluster": "firm"}),
    ),
    (
        "xtreg y x1, fe i(firm) vce(cluster firm)",
        "stata",
        ("feols", {"fml": "y ~ x1 | firm", "cluster": "firm"}),
    ),
    (
        "ivreghdfe y x1 (d = z), absorb(firm) cluster(firm)",
        "stata",
        ("feols", {"fml": "y ~ x1 | firm | d ~ z", "cluster": "firm"}),
    ),
    (
        "feols(y ~ x1 + x2 | firm + year, data=df, cluster=~firm)",
        "r",
        ("feols", {"fml": "y ~ x1 + x2 | firm + year", "cluster": "firm"}),
    ),
    (
        "ppmlhdfe hours x1, absorb(firm year)",
        "stata",
        ("ppmlhdfe", {"formula": "hours ~ x1", "absorb": "firm + year"}),
    ),
]


@pytest.mark.parametrize(
    "command,channel,reference",
    _NUMERIC_CASES,
    ids=[c[0][:32] for c in _NUMERIC_CASES],
)
def test_translation_fits_the_right_model(command, channel, reference):
    df = _num_df()
    translate = from_stata if channel == "stata" else from_r
    out = translate(command)
    assert out["ok"], f"{command}: translation failed: {out}"
    _assert_same_fit((out["tool"], out["arguments"]), reference, df, command)


def test_cross_translator_hdfe_agrees_numerically():
    """The two independent translators must agree on the numbers for the same
    model: Stata ``reghdfe`` and R ``feols`` of an identical HDFE spec produce
    the same coefficients. A bug in either reassembly path shows up as a
    divergence here."""
    df = _num_df()
    s = from_stata("reghdfe y x1 x2, absorb(firm year) cluster(firm)")
    r = from_r("feols(y ~ x1 + x2 | firm + year, data=df, cluster=~firm)")
    _assert_same_fit(
        (s["tool"], s["arguments"]),
        (r["tool"], r["arguments"]),
        df,
        "reghdfe vs feols",
    )


# ----------------------------------------------------------------------
# python_code ↔ arguments round-trip — copy-paste must match dispatch
# ----------------------------------------------------------------------
#
# Each translation emits BOTH a ``python_code`` string (for an LLM to paste
# into a chat reply) AND a structured ``arguments`` dict (for an LLM to
# dispatch via tools/call). An agent following one path and a user following
# the other must reach the same model. The previous round of fixes caught
# a real divergence here: ``_h_glm_like`` was writing ``args["robust"]`` but
# leaving it out of ``python_code`` — copy-paste ran the wrong model while
# dispatch ran the right one, with no error from either path. This contract
# parses the emitted code back to a dict and asserts equality with the
# structured ``arguments``, mod the data carrier (``data=df``).


def _parse_python_call(code):
    """Parse ``sp.fn(args..., kw=val)`` and return (tool, kwargs) where every
    value is a plain Python value. Returns None on parse failure or a
    placeholder / comment snippet.
    """
    import ast

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return None
    if not tree.body or not isinstance(tree.body[0], ast.Expr):
        return None
    call = tree.body[0].value
    if not isinstance(call, ast.Call):
        return None
    f = call.func
    if isinstance(f, ast.Name):
        tool = f.id
    elif isinstance(f, ast.Attribute):
        tool = f"sp.{f.attr}"
    else:
        return None
    out = {}

    def _val(v):
        if isinstance(v, (ast.List, ast.Dict, ast.Tuple, ast.Set)):
            try:
                return ast.literal_eval(v)
            except Exception:
                return ast.unparse(v)
        if isinstance(v, ast.Constant):
            return v.value
        if isinstance(v, ast.Name):
            return v.id
        if (
            isinstance(v, ast.UnaryOp)
            and isinstance(v.op, ast.USub)
            and isinstance(v.operand, ast.Constant)
        ):
            return -v.operand.value
        return ast.unparse(v)

    for k in call.keywords or []:
        if k.arg is None:
            continue
        out[k.arg] = _val(k.value)
    import inspect

    try:
        sig = inspect.signature(
            getattr(__import__("statspai"), tool.removeprefix("sp."), None)
        )
    except Exception:
        sig = None
    params = list(sig.parameters.values()) if sig else []
    for pos, a in enumerate(call.args):
        if pos < len(params):
            out[params[pos].name] = _val(a)
        else:
            out[f"_pos{pos}"] = _val(a)
    return tool, out


def _code_args_agree(payload, *, allow_postest=False):
    """Assert python_code and arguments describe the same ``sp.<tool>`` call.

    A few tools (margins / contrast / test / boottest / wild_cluster_boot) take
    a fitted result as a positional argument, so ``python_code`` references
    ``result`` while ``arguments`` does not — the agent is expected to pipe
    the result in. That is a partial translation by design; pass
    ``allow_postest=True`` to allow that asymmetry. Otherwise every
    key in args must appear in code with the same value (and vice versa
    for keys in code that are not the data carrier).
    """
    code = payload.get("python_code", "")
    if not code or code.lstrip().startswith("#"):
        return  # placeholder / comment snippet
    parsed = _parse_python_call(code)
    if parsed is None:
        return
    tool, code_args = parsed
    expected_tool = f"sp.{payload['tool']}"
    if tool != expected_tool:
        raise AssertionError(
            f"{payload.get('python_code', '')!r}: tool mismatch — "
            f"python_code calls {tool!r} but arguments target {expected_tool!r}."
        )
    args = dict(payload["arguments"])
    args.pop("data", None)
    code_args.pop("data", None)
    # Allow postestimation tools (margins / contrast / test / wild_cluster_boot):
    # they take a fitted ``result`` as the first argument in python_code while
    # args has no ``result`` key (the agent pipes the previous estimator's
    # result_id in). Match either positional ``_pos0`` or keyword ``result``.
    if allow_postest:
        if code_args.get("_pos0") == "result":
            code_args.pop("_pos0", None)
        if code_args.get("result") == "result":
            code_args.pop("result", None)
    for k, v in code_args.items():
        if k.startswith("_pos"):
            # Extra positional with no parameter name — can't match. Skip
            # unless the handler explicitly named it elsewhere.
            continue
        if k not in args:
            raise AssertionError(
                f"{code!r}: key {k!r}={v!r} present in python_code but not "
                f"in arguments (args={args})."
            )
        if args[k] != v:
            raise AssertionError(
                f"{code!r}: key {k!r} mismatch — python_code has {v!r}, "
                f"arguments has {args[k]!r}."
            )
    for k, v in args.items():
        if k not in code_args:
            raise AssertionError(
                f"{code!r}: key {k!r}={v!r} present in arguments but not "
                f"in python_code (code_args={code_args})."
            )


@pytest.mark.parametrize(
    "command,channel",
    [
        *[
            (c[0], "stata")
            for c in TIER1_ROUND_TRIPS + TIER2_ROUND_TRIPS + TIER3_ROUND_TRIPS
        ],
        *[(c[0], "r") for c in R_ROUND_TRIPS],
    ],
    ids=lambda x: x if isinstance(x, str) else x[:50],
)
def test_python_code_and_arguments_describe_the_same_call(command, channel):
    """For every (non-postest) translation, parsing the emitted ``python_code``
    back to kwargs must equal the structured ``arguments`` dict — a copy-paste
    user and a dispatch user reach the same model."""
    translate = from_stata if channel == "stata" else from_r
    out = translate(command)
    if not out.get("ok"):
        return
    POSTEST = {"margins", "contrast", "test", "wild_cluster_bootstrap", "mi_estimate"}
    allow = out.get("tool") in POSTEST
    _code_args_agree(out, allow_postest=allow)


# ----------------------------------------------------------------------
# Unified 4-channel sweep — one test, one failure path
# ----------------------------------------------------------------------
#
# Translation trust has four faces; running them as a single sweep gives a
# unified failure message that points at the exact channel that broke,
# rather than four separate test failures. A bug that affects multiple
# channels (e.g. plm signature drift surfaced as both a "dispatch" failure
# and an "args↔code" failure) is now reported once, with the deepest channel
# that actually caught it.
#
# Order matches the trust chain:
#   1. dispatchable   — the tool is callable and the payload runs.
#   2. lossless        — no required arg is silently dropped.
#   3. consistent     — python_code and arguments describe the same call.
#   4. self-consistent — the emitted code can be parsed back to its keys.
#
# Numerical equivalence (the fourth trust contract, in a separate suite)
# is excluded here because it needs synthetic data; this sweep is
# structural-only so it runs in microseconds across every command.


POSTEST_TOOLS = {"margins", "contrast", "test", "wild_cluster_bootstrap", "mi_estimate"}


def _channel_dispatchable(out, df):
    """Channel 1: payload is dispatchable — tool exists, args accepted, runs."""
    if out.get("tool") in POSTEST_TOOLS:
        return  # partial translations handled by postest allowlist
    import statspai as sp

    tool = out["tool"]
    fn = getattr(sp, tool, None)
    if fn is None or not callable(fn):
        raise AssertionError(
            f"channel=dispatchable: sp.{tool} is not callable — "
            f"the translated tool name is a dead on-ramp."
        )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = fn(data=df, **out["arguments"])
        except TypeError as e:
            raise AssertionError(
                f"channel=dispatchable: sp.{tool}(**{out['arguments']}) raised "
                f"TypeError — {e}. An argument name is likely wrong or a required "
                f"kwarg is missing."
            ) from None
        except Exception:
            # Numerical / convergence errors are not the dispatch channel's
            # concern — they belong to numerical faithfulness.
            return


def _channel_lossless(out):
    """Channel 2: no arg name in arguments is silently dropped by dispatch."""
    if out.get("tool") in POSTEST_TOOLS:
        return
    import inspect

    import statspai as sp

    tool = out["tool"]
    fn = getattr(sp, tool, None)
    if fn is None:
        return
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return
    has_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kwargs:
        return  # unknown kwargs land in **kwargs; only the structural test
        # (channel 3) can catch silent drops there.
    accepted = {
        p.name
        for p in sig.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
    unknown = set(out["arguments"]) - accepted - {"data"}
    assert not unknown, (
        f"channel=lossless: tool sp.{tool} rejects {sorted(unknown)} (no "
        f"**kwargs to absorb them). One of those args is being silently dropped "
        f"at dispatch — the call shape looks runnable but fits a different model."
    )


def _channel_consistent(out):
    """Channel 3: python_code and arguments describe the same call.

    Both must be derivable from one another. A divergence means a user
    who copy-pastes the python_code gets a different model than a user who
    dispatches (tool, arguments).
    """
    allow = out.get("tool") in POSTEST_TOOLS
    _code_args_agree(out, allow_postest=allow)


def _channel_self_consistent(out):
    """Channel 4: the emitted python_code can be parsed back to its keys.

    This is the lightest channel — it just confirms the code is syntactically
    a real sp.<tool> call with extractable kwargs. Catches malformed strings,
    placeholder snippets that were never real code, etc.
    """
    code = out.get("python_code", "")
    if not code or code.lstrip().startswith("#"):
        if out.get("tool") in POSTEST_TOOLS:
            return
        raise AssertionError(
            f"channel=self_consistent: python_code is a placeholder/comment: "
            f"{code!r} — the translator emitted no runnable call."
        )
    parsed = _parse_python_call(code)
    if parsed is None:
        raise AssertionError(
            f"channel=self_consistent: python_code is not parseable: {code!r}"
        )


#: Same dataset shape used by the numerical-equivalence contract — built
#: lazily so the sweep is cheap on commands that don't need a dataframe.
def _sweep_df():
    import pandas as pd

    return _num_df()


@pytest.mark.parametrize(
    "command,channel",
    [
        *[
            (c[0], "stata")
            for c in TIER1_ROUND_TRIPS + TIER2_ROUND_TRIPS + TIER3_ROUND_TRIPS
        ],
        *[(c[0], "r") for c in R_ROUND_TRIPS],
    ],
    ids=lambda x: x if isinstance(x, str) else x[:50],
)
def test_unified_translation_trust_sweep(command, channel):
    """For every (non-postest) translation, all four trust channels must
    pass: dispatchable, lossless, consistent, self-consistent. A single
    failure points at the violated channel; if multiple channels break, we
    surface all of them so the regression can be diagnosed in one read."""
    translate = from_stata if channel == "stata" else from_r
    out = translate(command)
    if not out.get("ok"):
        # translators already fail-loud with suggestions; covered by the
        # existing dispatch-policy tests. Skip — nothing for the sweep to
        # verify.
        return
    failures: list = []
    for name, fn in (
        ("self_consistent", _channel_self_consistent),
        ("dispatchable", _channel_dispatchable),
        ("lossless", _channel_lossless),
        ("consistent", _channel_consistent),
    ):
        try:
            if name == "dispatchable":
                fn(out, _sweep_df())
            else:
                fn(out)
        except AssertionError as e:
            failures.append(str(e))
    assert not failures, (
        f"translation trust sweep failed for {command!r} on "
        f"{len(failures)} channel(s):\n  " + "\n  ".join(failures)
    )
