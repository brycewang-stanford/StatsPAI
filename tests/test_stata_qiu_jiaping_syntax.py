"""``sp.stata`` behaviours found while replaying Qiu Jiaping's textbook.

The expected values of the data steps were printed by Stata 18 on the same
lines (``list`` after each block); they are quoted in the tests.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent._translation._stata_run import StataSession


def _run(source: str, data=None) -> StataSession:
    session = StataSession(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for line in source.strip().splitlines():
            session.run(line.strip())
    return session


# ------------------------------------------------- replace, row after row
def test_replace_reads_the_rows_it_has_just_written():
    """Stata replaces one observation after another, so a reference to an
    earlier row of the variable sees its new value."""
    data = _run("""
        clear
        set obs 6
        gen t = _n
        tsset t
        gen x = 1
        replace x = 2*l.x + 1 if t > 1
        gen y = 1
        replace y = y[_n-1] + y[_n-2] if t > 2
        gen z = t + 1
        replace z = z/z[1]
        gen w = t
        replace w = w[_n+1]
        gen u = 1 in 1
        replace u = 2*l.u in 2/6
        gen du = u - l.u
        """).data
    assert data["x"].tolist() == [1, 3, 7, 15, 31, 63]
    assert data["y"].tolist() == [1, 1, 2, 3, 5, 8]
    # the first row becomes 1, and every later row is divided by that 1
    assert data["z"].tolist() == [1, 3, 4, 5, 6, 7]
    # a later row is read before it is replaced
    assert data["w"].iloc[:5].tolist() == [2, 3, 4, 5, 6] and np.isnan(
        data["w"].iloc[5]
    )
    assert data["u"].tolist() == [1, 2, 4, 8, 16, 32]
    assert data["du"].iloc[1:].tolist() == [1, 2, 4, 8, 16]


def test_replace_within_groups_carries_forward():
    data = _run("""
        clear
        set obs 6
        gen g = _n > 3
        gen v = _n if mod(_n, 3) == 1
        bysort g: replace v = v[_n-1] if missing(v)
        """).data
    assert data["v"].tolist() == [1, 1, 1, 4, 4, 4]


def test_replace_within_groups_three_shapes():
    """A running total (the value reads nothing but its row), a value that
    counts within the group, and a condition that reads the row above.
    Stata lists a = 1 3 6 10 | 5 11 18 26 | 9 19 30 42,
    b = 1 3 6 10 | 5 7 10 14 | 9 11 14 18 and
    c = 2 2 3 6 | 10 16 17 18 | 18 28 29 30."""
    data = _run("""
        clear
        set obs 12
        gen id = ceil(_n/4)
        gen double a = _n
        bysort id: replace a = a[_n-1] + a if _n > 1
        gen double b = _n
        bysort id: replace b = b[_n-1] + _n if _n > 1
        gen double c = _n
        bysort id: replace c = c[1] + c if c[_n-1] > 2
        """).data
    assert data["a"].tolist() == [1, 3, 6, 10, 5, 11, 18, 26, 9, 19, 30, 42]
    assert data["b"].tolist() == [1, 3, 6, 10, 5, 7, 10, 14, 9, 11, 14, 18]
    # a missing value is larger than 2, so the first row of a group changes
    assert data["c"].tolist() == [2, 2, 3, 6, 10, 16, 17, 18, 18, 28, 29, 30]


def test_recursive_replace_draws_once():
    """An AR(1) series built in place: the innovations are drawn once, and
    what is left after the lag is taken out is the innovation."""
    data = _run("""
        clear
        set obs 400
        set seed 12345
        gen t = _n
        tsset t
        gen double e = rnormal(0, 1)
        gen double x = 0 in 1
        replace x = 0.5*l.x + e in 2/400
        gen double back = x - 0.5*l.x
        """).data
    assert np.allclose(data["back"].iloc[1:], data["e"].iloc[1:], atol=1e-12)
    assert data["x"].notna().all()


def test_a_lag_is_read_from_the_data_as_they_are():
    """gen x = t in 1/3 ; gen lx = l.x ; replace x = 10 in 4/8 ; gen lx2 = l.x
    Stata lists lx2 = . 1 2 3 10 10 10 10."""
    data = _run("""
        clear
        set obs 8
        gen t = _n
        tsset t
        gen x = t in 1/3
        gen lx = l.x
        replace x = 10 in 4/8
        gen lx2 = l.x
        """).data
    assert np.isnan(data["lx2"].iloc[0])
    assert data["lx2"].iloc[1:].tolist() == [1, 2, 3, 10, 10, 10, 10]
    assert (
        data["lx"].iloc[1:4].tolist() == [1, 2, 3] and data["lx"].iloc[4:].isna().all()
    )


def test_a_chain_longer_than_the_limit_is_refused():
    session = _run("clear\nset obs 20005\ngen t = _n\ngen x = 1 in 1")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="chains more than"):
        session.run("replace x = x[_n-1] + 1 in 2/l")


# ------------------------------------------------------------ small syntax
def test_clear_all_empties_the_data():
    out = sp.stata(
        "clear all\nset obs 5\ngen x = 1\nclear all\nset obs 5\ngen x = 2\nsum x"
    )
    assert float(out.loc["x", "Mean"]) == 2.0


def test_clear_all_drops_stored_estimates():
    frame = pd.DataFrame({"y": [1.0, 2, 4, 3, 6], "x": [1.0, 2, 3, 4, 5]})
    session = _run("reg y x\nestimates store m1\nscalar k = 3", frame)
    assert "m1" in session.estimates
    session.run("clear all")
    assert not session.estimates and not session.stored["scalars"]
    assert session.last is None


def test_an_option_may_stand_apart_from_its_parenthesis():
    code = sp.from_stata("reg y x, cluster (id)")["python_code"]
    assert code == "sp.regress('y ~ x', cluster='id', data=df)"
    heck = sp.from_stata("heckman w educ age, select (educ age kids) twostep")
    assert heck["arguments"]["z"] == ["educ", "age", "kids"]
    assert heck["untranslated_options"] == []


def test_describe_abbreviations_are_skipped():
    frame = pd.DataFrame({"y": [1.0, 2, 4, 3, 6], "x": [1.0, 2, 3, 4, 5]})
    for line in ("d", "des y x", "desc", "describe y", "d, short"):
        assert _run(line, frame).output is None
    # a variable called d is still a variable
    session = _run("gen d = x > 2\nsum d", frame)
    assert float(session.output.loc["d", "Mean"]) == 0.6


def test_vif_is_estat_vif():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(size=(60, 3)), columns=["y", "a", "b"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = sp.stata("reg y a b\nvif", frame)
        new = sp.stata("reg y a b\nestat vif", frame)
    assert old["mean_vif"] == new["mean_vif"]


# -------------------------------------------------- matching, by its lines
@pytest.fixture(scope="module")
def lalonde() -> pd.DataFrame:
    return sp.datasets.nsw_lalonde().drop(columns=["race"])


COVS = "age educ black hispanic married nodegree re74 re75"


def test_pscore_writes_its_variables(lalonde):
    session = _run(
        f"pscore treat {COVS}, pscore(ps1) blockid(b1) logit comsup", lalonde
    )
    data = session.data
    assert {"ps1", "b1", "comsup"} <= set(data.columns)
    assert int(data["comsup"].sum()) == 557  # Stata: 557 in the region
    assert int(data["b1"].max()) == 7
    assert session.output.n_blocks == 7
    # the name is taken: Stata stops before fitting anything
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="already defined"):
        session.run(f"pscore treat {COVS}, pscore(ps1) logit")
    # comsup is dropped and written again, as the command does
    session.run(f"pscore treat {COVS}, pscore(ps2)")
    assert "comsup" not in session.data.columns


def test_attnd_and_psmatch2_on_the_stored_score(lalonde):
    """attnd re78 treat, pscore(ps1) -> r(attnd) = 1968.799716424065
    psmatch2 treat, pscore(ps1) outcome(re78) neighbor(1) ties -> the same"""
    session = _run(f"pscore treat {COVS}, pscore(ps1) logit", lalonde)
    session.run("attnd re78 treat, pscore(ps1)")
    r = session.stored["r"]
    assert r["attnd"] == pytest.approx(1968.799716424065, rel=5e-9)
    assert r["seattnd"] == pytest.approx(1008.408250646113, rel=5e-9)
    assert r["ntnd"] == 185 and r["ncnd"] == 88
    session.run("psmatch2 treat, pscore(ps1) outcome(re78) neighbor(1) ties")
    assert session.stored["r"]["att"] == pytest.approx(1968.7997158559, rel=1e-11)
    assert session.stored["r"]["seatt"] == pytest.approx(1008.4082506356, rel=1e-9)
    session.run("attnd re78 treat, pscore(ps1) comsup")
    assert session.stored["r"]["attnd"] == pytest.approx(1981.076824530195, rel=5e-9)
    session.run("display r(attnd)")
    assert session.output == pytest.approx(1981.076824530195, rel=5e-9)


def test_pstest_reads_what_psmatch2_left(lalonde):
    session = _run(
        f"psmatch2 treat {COVS}, logit outcome(re78) neighbor(1) ties\n"
        "pstest age educ re74, both",
        lalonde,
    )
    table = session.output.table
    direct = sp.psmatch2(lalonde, treat="treat", covariates=COVS.split(),
                         outcome="re78", ties=True).pstest(["age", "educ", "re74"])  # fmt: skip
    pd.testing.assert_frame_equal(table, direct.table)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="not implemented"):
        session.run("pstest age, treated(treat)")
    fresh = StataSession(lalonde)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="follow a psmatch2"):
        fresh.run("pstest age")


def test_bootstrap_of_a_psmatch2_result(lalonde):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sp.stata(
            "set seed 7\n"
            f"bootstrap r(att), reps(20): psmatch2 treat {COVS}, logit "
            "outcome(re78) neighbor(1) common ties",
            lalonde,
        )
    direct = sp.psmatch2(lalonde, treat="treat", covariates=COVS.split(),
                         outcome="re78", ties=True, common_support="minmax")  # fmt: skip
    assert float(np.ravel(out.estimate)[0]) == pytest.approx(direct.att, rel=1e-12)
    assert float(np.ravel(out.se)[0]) > 0


def test_translations_of_the_matching_commands():
    cases = {
        "psmatch2 d, pscore(ps) out(y) n(1) ties":
            "sp.psmatch2(data=df, treat='d', pscore='ps', outcome='y', neighbor=1, "
            "ties=True)",
        "attnd y d, pscore(ps) comsup":
            "sp.psmatch2(data=df, treat='d', outcome='y', pscore='ps', ties=True, "
            "common_support='treated')",
        "attnd y d x1 x2, logit boot reps(100)":
            "sp.psmatch2(data=df, treat='d', outcome='y', covariates=['x1', 'x2'], "
            "ps_model='logit', ties=True, se='bootstrap', bootstrap_reps=100)",
        "pscore d x1 x2, pscore(ps) blockid(b) comsup numblo(4) level(0.05)":
            "sp.pscore(df, treat='d', covariates=['x1', 'x2'], ps_model='probit', "
            "common_support=True, level=0.05, n_blocks=4)",
        "DCdensity z, breakpoint(0.5) b(0.01) h(0.2) nograph":
            "sp.mccrary_test(df, x='z', c=0.5, bin_width=0.01, bw=0.2)",
        "loneway y g, level(90) exact":
            "sp.loneway(df, y='y', by='g', alpha=0.1, exact=True)",
    }  # fmt: skip
    for line, code in cases.items():
        out = sp.from_stata(line)
        assert out["python_code"] == code, line
        assert out["untranslated_options"] == [], line
    # the matching would run on the linear index: not what the call does
    assert sp.from_stata("attnd y d x, index")["untranslated_options"] == ["index"]
    assert not sp.from_stata("pstest x1 x2, both")["ok"]
    assert not sp.from_stata("psmatch2 d, out(y)")["ok"]


def test_loneway_and_dcdensity_lines():
    frame = pd.DataFrame({"y": np.array([1.0, 2, 3, 11, 12, 13, 21, 22, 23])
                          + np.arange(1, 10) % 4, "g": np.repeat([1, 2, 3], 3)})  # fmt: skip
    session = _run("loneway y g", frame)
    # r(rho), r(se), r(sd_b) printed by Stata on these nine rows
    r = session.stored["r"]
    assert r["rho"] == pytest.approx(0.967479674796748, rel=1e-13)
    assert r["se"] == pytest.approx(0.0367371180606702, rel=1e-12)
    assert r["sd_b"] == pytest.approx(9.620579793107874, rel=1e-13)
    senate = sp.datasets.lee_2008_senate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        line = sp.stata("DCdensity x, breakpoint(0) generate(Xj Yj r0 fhat se) nograph",
                        senate)  # fmt: skip
        direct = sp.mccrary_test(senate, "x", c=0)
    assert line.estimate == direct.estimate and line.se == direct.se
