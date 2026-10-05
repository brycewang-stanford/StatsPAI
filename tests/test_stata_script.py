"""``sp.stata`` on a do-file snippet: comments, continuations, macros, xtset.

A pasted snippet is not a list of one-line commands. ``///`` continues a
command, variable lists live in ``global`` / ``local`` macros, and the
panel id of ``xtreg, fe`` is declared by an earlier ``xtset``. These are
resolved by reading the snippet; what only Stata can evaluate (a macro
set by an expression, a loop) is refused.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent._translation._stata_script import (
    MacroTable,
    ScriptError,
    control_flow,
    panel_declaration,
    split_commands,
)
from statspai.exceptions import MethodIncompatibility


def test_split_commands_resolves_comments_and_continuations():
    text = """
    * a comment line
    reg y x ///  continued
        , r      // trailing comment
    /* a block
       comment reg y z */
    sum y; lincom x
    di "a // not a comment"
    """
    assert split_commands(text) == [
        "reg y x , r",
        "sum y",
        "lincom x",
        'di "a // not a comment"',
    ]


def test_split_commands_follows_delimit():
    text = "#delimit ;\nreg y x\n  , r ;\nsum y ;\n#delimit cr\nreg y z\n"
    assert split_commands(text) == ["reg y x , r", "sum y", "reg y z"]
    assert split_commands("#d ;\nreg y x ;") == ["reg y x"]


def test_a_url_is_not_a_comment():
    assert split_commands('use "http://x.org/d.dta"') == ['use "http://x.org/d.dta"']


def test_macros_are_expanded_where_they_were_defined_by_text():
    t = MacroTable()
    for line in ['global ctrl "age educ"', "local fe id year", "gl more $ctrl tenure"]:
        assert t.define(line)
    assert not t.define("reg y x")
    assert t.expand("reghdfe y x ${more}, a(`fe')") == (
        "reghdfe y x age educ tenure, a(id year)"
    )
    # a value is fixed when the macro is defined, as in Stata
    t.define('global ctrl "other"')
    assert t.expand("$more") == "age educ tenure"


@pytest.mark.parametrize(
    "definition",
    ["local n = _N", "local k : word count a b", "local ++i", "global s = 2 * 3"],
)
def test_a_macro_only_stata_can_evaluate_is_refused_on_use(definition):
    t = MacroTable()
    assert t.define(definition)
    name = definition.split()[1].lstrip("+")
    use = f"${name}" if definition.startswith("global") else f"`{name}'"
    with pytest.raises(ScriptError, match="only known to Stata"):
        t.expand(f"reg y x {use}")


def test_literal_expressions_are_values():
    t = MacroTable()
    t.define("local k = 3")
    t.define('local y = "wage"')
    assert t.expand("reg `y' x in 1/`k'") == "reg wage x in 1/3"


def test_an_undefined_macro_is_refused():
    # Stata would expand it to nothing and run a different regression
    with pytest.raises(ScriptError, match="not defined"):
        MacroTable().expand("reg y x $controls")


def test_a_compound_quote_is_not_a_macro():
    assert MacroTable().expand('title(`"a b"\')') == 'title(`"a b"\')'


def test_control_flow_and_panel_declarations_are_recognised():
    assert control_flow("foreach v of varlist a b {") == "foreach"
    assert control_flow("forv i = 1/3 {") == "forvalues"
    assert control_flow("}") == "}"
    assert control_flow("reg y x if z == 1") is None
    assert control_flow("format x %9.2f") is None
    assert panel_declaration("xtset id year") == ("id", "year")
    assert panel_declaration("xtset id") == ("id", None)
    assert panel_declaration("tsset year") == (None, "year")
    assert panel_declaration("xtset id year, yearly") == ("id", "year")
    assert panel_declaration("reg y x") is None


# ---------------------------------------------------------------------------
# sp.stata
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(20261001)
    n_id, n_t = 60, 10
    d = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n_id), n_t),
            "year": np.tile(np.arange(n_t), n_id),
            "x": rng.normal(size=n_id * n_t),
            "age": rng.normal(size=n_id * n_t),
            "educ": rng.normal(size=n_id * n_t),
        }
    )
    d["y"] = (
        1
        + 0.5 * d.x
        + 0.2 * d.age
        + np.repeat(rng.normal(size=n_id), n_t)
        + rng.normal(size=n_id * n_t)
    )
    return d


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def test_a_snippet_with_macros_and_continuations_runs_like_the_direct_call(df):
    script = """
    * baseline specification
    global ctrl "age educ"
    local fe id year
    reghdfe y x $ctrl ///
        , a(`fe') cl(id)   // main
    """
    got = _quiet(sp.stata, script, data=df)
    want = _quiet(sp.hdfe_ols, "y ~ x + age + educ | id + year", data=df, cluster="id")
    assert got.params["x"] == want.params["x"]
    assert got.std_errors["x"] == want.std_errors["x"]


def test_xtset_supplies_the_panel_id(df):
    got = _quiet(sp.stata, "xtset id year\nxtreg y x, fe r", data=df)
    want = _quiet(sp.feols, "y ~ x | id", data=df, cluster="id")
    assert float(got.params["x"]) == float(want.params["x"])
    assert float(got.std_errors["x"]) == float(want.std_errors["x"])
    # an explicit i() is not overridden
    out = sp.from_stata("xtreg y x, fe i(year)")
    assert out["arguments"]["fml"] == "y ~ x | year"


def test_xtset_supplies_id_and_time_to_xtabond(df):
    got = _quiet(sp.stata, "xtset id year\nxtabond y x, lags(1)", data=df)
    want = _quiet(
        sp.xtabond, data=df, y="y", x=["x"], id="id", time="year", robust=False
    )
    pd.testing.assert_frame_equal(got.detail, want.detail)
    # with another time variable the instruments, hence the estimates, change
    assert sp.from_stata("xtabond y x, i(id) t(year)")["arguments"]["time"] == "year"


@pytest.mark.parametrize(
    "script, match",
    [
        ("reg y x $controls", "not defined"),
        ("local k : sysdir STATA\nreg y x `k'", "not implemented"),
        ("mata\n x = 1\nend", "control flow"),
        ("xtreg y x, fe", "cannot be run as written"),
    ],
)
def test_what_cannot_be_resolved_by_reading_is_refused(df, script, match):
    with pytest.raises(MethodIncompatibility, match=match):
        _quiet(sp.stata, script, data=df)
