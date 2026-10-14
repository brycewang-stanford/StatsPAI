"""October review of ``sp.stata``: data steps and estimation.

Every expected number below was produced by Stata 18 MP on the same data
(the snippet is in each test) and is quoted next to its assertion. The
defects these pin: frequency-weighted ``collapse`` ignored the weights,
``margins, post`` and ``predict(xb)`` were ignored, ``[pw=]`` after a
likelihood estimator carried N/(N-k), weighted fits could not be followed
by ``margins``, ``L.x`` left helper columns in the data, ``delta()`` was
ignored, ``F.D.x`` was refused, ``egen group()`` / ``tag()`` counted ""
as a group, cluster fits stored the wrong ``e(df_r)`` and no
``e(N_clust)``, ``gsort`` misplaced missing values, storage types were not
enforced, tied ``xtile`` cutpoints were merged, three Stata errors were
accepted, and some factor-variable notation leaked a patsy error or could
not be read back through ``_b[]``.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent._translation._stata_run import StataSession
from statspai.agent._translation._stata_script import split_commands
from statspai.exceptions import MethodIncompatibility

NAN = np.nan

#: the 60-row design every estimation test below uses (Stata code, run
#: verbatim on both sides)
D = """
set obs 60
gen double x1 = mod(_n*37, 101)/10
gen double x2 = mod(_n*53, 97)/10
gen g = mod(_n, 3) + 1
gen h = mod(_n, 2)
gen w = mod(_n, 4) + 1
gen cl = ceil(_n/6)
replace cl = . in 5/7
gen double y = 1 + .5*x1 - .3*x2 + g + mod(_n*13, 17)/5
gen yb = (y > 4.5)
replace yb = 1 - yb in 1/8
"""


def run(code: str, data: Optional[pd.DataFrame] = None) -> StataSession:
    session = StataSession(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for line in split_commands(code):
            session.run(line)
    return session


def value(session: StataSession, expr: str) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(session.value(expr))


def close(got: float, want: float, rtol: float = 1e-7) -> None:
    np.testing.assert_allclose(got, want, rtol=rtol, atol=1e-12)


def same(got, want) -> None:
    np.testing.assert_allclose(
        np.asarray(got, dtype=float), np.asarray(want, dtype=float), equal_nan=True
    )


# ------------------------------------------------- 1. collapse [fw=]
@pytest.fixture()
def fw_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "g": [1, 1, 1, 2, 2, 2, 3.0],
            "x": [1, 5, NAN, 3, 4, 9, 7],
            "w": [2, 1, 3, 0, 1, NAN, 0],
        }
    )


def test_collapse_frequency_weights_stand_for_rows(fw_data):
    s = run(
        "collapse (mean) m=x (sum) s=x (sd) sd=x (count) n=x (p50) md=x "
        "(min) mn=x (max) mx=x (first) f=x (last) l=x (firstnm) fn=x "
        "(lastnm) ln=x [fw=w], by(g)",
        fw_data,
    )
    out = s.data
    # Stata 18: g=1: 2.333333 7 2.309401 3 1 1 5 1 . 1 5
    #           g=2: 4 4 . 1 4 4 4 4 4 4 4 ; group 3 (weight 0) is gone
    same(out["g"], [1, 2])
    close(out["m"][0], 7 / 3)
    same(out["s"], [7, 4])
    close(out["sd"][0], 2.309401076758503)
    assert np.isnan(out["sd"][1])
    same(out["n"], [3, 1])
    same(out["md"], [1, 4])
    same(out["mn"], [1, 4])
    same(out["mx"], [5, 4])
    same(out["f"], [1, 4])
    same(out["l"], [NAN, 4])
    same(out["fn"], [1, 4])
    same(out["ln"], [5, 4])


def test_collapse_frequency_weights_without_by_and_rawsum(fw_data):
    out = run(
        "collapse (mean) x (sum) s=x (sd) sd=x (count) n=x (p50) m=x "
        "(rawsum) r=x [fw=w]",
        fw_data,
    ).data
    # Stata 18: x 2.75, s 11, sd 2.061553, n 4, m 2.5; rawsum by g was
    # 6 / 4, so 10 over the rows with a positive weight
    close(out["x"][0], 2.75)
    close(out["s"][0], 11)
    close(out["sd"][0], 2.0615528128088303)
    close(out["n"][0], 4)
    close(out["m"][0], 2.5)
    close(out["r"][0], 10)


def test_collapse_frequency_weights_finder_case():
    code = """
set obs 3
gen g = cond(_n < 3, 1, 2)
gen x = _n*2 - 1
gen w = _n
collapse (mean) x (sum) s=x (sd) sd=x (count) n=x [fw=w], by(g)
"""
    out = run(code).data
    # Stata 18, g = 1: 2.3333333 / 7 / 1.1547005 / 3 (sp gave 2 / 4 / 1.41 / 2)
    close(out["x"][0], 2.3333333, rtol=1e-7)
    close(out["s"][0], 7)
    close(out["sd"][0], 1.1547005, rtol=1e-7)
    close(out["n"][0], 3)


def test_collapse_noninteger_frequency_weight_is_refused(fw_data):
    fw_data.loc[0, "w"] = 1.5
    # Stata 18: "may not use noninteger frequency weights"
    with pytest.raises(MethodIncompatibility, match="noninteger frequency"):
        run("collapse (mean) x [fw=w], by(g)", fw_data)


# ------------------------------------------- 2. / 3. / 5. margins
def test_margins_post_dydx_replaces_the_estimates():
    s = run(D + "qui logit yb x1 x2\nqui margins, dydx(*) post")
    # Stata 18: _b[x1] .0456990373109084497, _se[x1] .0182725101126413568
    close(value(s, "_b[x1]"), 0.0456990373109084497, rtol=1e-8)
    close(value(s, "_se[x1]"), 0.0182725101126413568, rtol=1e-8)
    close(value(s, "e(N)"), 60)
    test = run(D + "qui logit yb x1 x2\nqui margins, dydx(*) post")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test.run("test x1 = x2")
        chi2 = test.output["chi2"]
        test.run("lincom x1 - x2")
        lin = test.output
    # Stata 18: test chi2 10.701216485407679; lincom .0871988501728312437
    # se .0266559511165890412
    close(chi2, 10.701216485407679, rtol=1e-8)
    close(lin["estimate"], 0.0871988501728312437, rtol=1e-8)
    close(lin["se"], 0.0266559511165890412, rtol=1e-8)


def test_margins_post_names_follow_stata():
    s = run(D + "qui logit yb x1 i.g\nqui margins, dydx(g x1) post\nmatrix b = e(b)")
    # Stata 18: e(b) = x1 .04236822, 1b.g 0, 2.g .01474597, 3.g .12791318
    # (model order, whatever the order of dydx())
    same([value(s, f"el(b,1,{k})") for k in range(1, 5)][1], 0.0)
    close(value(s, "el(b,1,1)"), 0.04236822, rtol=1e-6)
    close(value(s, "_b[2.g]"), 0.01474597, rtol=1e-6)
    close(value(s, "_b[3.g]"), 0.12791318, rtol=1e-6)
    close(value(s, "_se[3.g]"), 0.144436, rtol=1e-5)

    s = run(D + "qui reg y x1 i.g\nqui margins g, post\nmatrix V = e(V)")
    # Stata 18: _b[1.g] 4.734945015626427889, _se[3.g] .297772665315268914,
    # V[2,1] 3.8948586420155e-06, V[3,1] -.000393380722842697
    close(value(s, "_b[1.g]"), 4.734945015626427889, rtol=1e-10)
    close(value(s, "_se[3.g]"), 0.297772665315268914, rtol=1e-10)
    close(value(s, "el(V,2,1)"), 3.8948586420155e-06, rtol=1e-6)
    close(value(s, "el(V,3,1)"), -0.000393380722842697, rtol=1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.run("test 1.g = 2.g")
        f_stat = s.output["statistic"]
        s.run("lincom 3.g - 1.g")
        lin = s.output
    # Stata 18: F(1, 56) 5.879727562434210242; lincom 2.075730518590880536
    # se .4220379760136915293
    close(f_stat, 5.879727562434210242, rtol=1e-8)
    close(lin["estimate"], 2.075730518590880536, rtol=1e-10)
    close(lin["se"], 0.4220379760136915293, rtol=1e-8)

    s = run(D + "qui reg y c.x1##c.x1 x2\nqui margins, at(x1=(2 5)) post")
    # Stata 18: _b[1._at] 4.2473976, _b[2._at] 5.590850081505410962,
    # _se[2._at] .255542548948249626; lincom 2._at - 1._at
    # 1.343452526015357584 se .2845914345900720233
    close(value(s, "_b[1._at]"), 4.2473976, rtol=1e-7)
    close(value(s, "_b[2._at]"), 5.590850081505410962, rtol=1e-10)
    close(value(s, "_se[2._at]"), 0.255542548948249626, rtol=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.run("lincom 2._at - 1._at")
    close(s.output["estimate"], 1.343452526015357584, rtol=1e-10)
    close(s.output["se"], 0.2845914345900720233, rtol=1e-8)

    s = run(D + "qui reg y x1 x2\nqui margins, post")
    # Stata 18: _cons 5.7668333, se .1668971
    close(value(s, "_b[_cons]"), 5.7668333, rtol=1e-7)
    close(value(s, "_se[_cons]"), 0.1668971, rtol=1e-6)


def test_margins_post_is_refused_for_tables_it_cannot_name():
    with pytest.raises(MethodIncompatibility, match="post"):
        run(D + "qui reg y x1 x2\nmargins, over(g) post")


def test_margins_predict_xb_is_the_linear_index():
    s = run(D + "qui logit yb x1 x2\nqui margins, predict(xb) post")
    # Stata 18: .6327671630270911285, se .2933627991842022165 (sp gave the
    # mean probability .633333 / .05823)
    close(value(s, "_b[_cons]"), 0.6327671630270911285, rtol=1e-8)
    close(value(s, "_se[_cons]"), 0.2933627991842022165, rtol=1e-8)
    s = run(D + "qui logit yb x1 x2\nmargins, dydx(*) predict(xb)")
    # Stata 18: the coefficients, .2246249 / -.2039844
    close(s.output["dy/dx"].tolist()[0], 0.2246249, rtol=1e-6)
    with pytest.raises(MethodIncompatibility, match="predict"):
        run(D + "qui reg y x1 x2\nmargins, predict(stdp)")


def test_margins_after_weighted_fits():
    s = run(D + "qui logit yb x1 x2 [pw=w]\nqui margins, dydx(*) post")
    # Stata 18 r(b): .041448452836319642 -.046830140961026799,
    # r(V)[1,1] .000390355860467654
    close(value(s, "_b[x1]"), 0.041448452836319642, rtol=1e-7)
    close(value(s, "_b[x2]"), -0.046830140961026799, rtol=1e-7)
    close(value(s, "_se[x1]") ** 2, 0.000390355860467654, rtol=1e-6)

    s = run(D + "qui probit yb x1 i.g [pw=w]\nqui margins g, post\nmatrix V = e(V)")
    # Stata 18: .65061387851615593, V[1,1] .013266999637861591,
    # V[2,1] .000076154785068058
    close(value(s, "_b[1.g]"), 0.65061387851615593, rtol=1e-7)
    close(value(s, "_se[1.g]") ** 2, 0.013266999637861591, rtol=1e-6)
    close(value(s, "el(V,2,1)"), 0.000076154785068058, rtol=1e-5)

    s = run(
        D + "qui logit yb x1 x2 [pw=w], vce(cluster cl)\nqui margins, dydx(x1) post"
    )
    # Stata 18: .047911805811587362, V .000590352311514531
    close(value(s, "_b[x1]"), 0.047911805811587362, rtol=1e-7)
    close(value(s, "_se[x1]") ** 2, 0.000590352311514531, rtol=1e-6)

    s = run(D + "qui reg y x1 x2 [aw=w]\nqui margins, dydx(x1) post")
    # Stata 18: .5015805893435800744 se .0591980099305199348
    close(value(s, "_b[x1]"), 0.5015805893435800744, rtol=1e-10)
    close(value(s, "_se[x1]"), 0.0591980099305199348, rtol=1e-10)


# --------------------------------------- 4. [pw=] after likelihood fits
@pytest.mark.parametrize(
    "line, want",
    [
        # Stata 18 robust SEs of x1 (sp had sqrt(59/58) more)
        ("logit yb x1 [pw=w]", 0.0980456391384460502),
        ("probit yb x1 [pw=w]", 0.0603029257995002233),
        ("probit yb x1 i.g [pw=w]", 0.0629657349102349245),
        ("logit yb x1 [pw=w], vce(cluster cl)", 0.1334426059222359462),
        ("probit yb x1 x2 [pw=w], vce(cluster cl)", 0.0899420638662265931),
        ("poisson cnt x1 [pw=w]", 0.0346176773031923873),
        ("cloglog yb x1 [pw=w]", 0.0658677642040734895),
    ],
)
def test_pweights_after_likelihood_estimators(line, want):
    s = run(D + "gen cnt = mod(_n*7, 5)\nqui " + line)
    close(value(s, "_se[x1]"), want, rtol=1e-6)


def test_pweight_translation_names_stata_robust():
    code = sp.from_stata("logit yb x1 [pw=w]")["python_code"]
    assert "robust='robust'" in code
    assert "robust='hc1'" in sp.from_stata("regress y x1 [pw=w]")["python_code"]


# ------------------------------------------ 6. / 7. / 8. ts operators
def test_lag_helpers_do_not_stay_in_the_data():
    code = """
set obs 6
gen t = _n
gen x = _n^2
tsset t
gen lx = L.x
egen rs = rowtotal(x*)
local n = 0
foreach v of varlist x* {
    local n = `n' + 1
}
gen nvars = `n'
"""
    s = run(code)
    # Stata 18: rs[2] 4 (x only), one variable matches x*
    assert list(s.data.columns) == ["t", "x", "lx", "rs", "nvars"]
    same(s.data["rs"][1], 4)
    same(s.data["nvars"][0], 1)


def test_predict_and_sample_after_a_lag_regression():
    code = """
set obs 8
gen t = _n
gen double x = mod(_n*5, 7)
gen double y = mod(_n*3, 11)
tsset t
reg y L.x
predict yh
predict se, stdp
count if e(sample)
egen rs = rowtotal(x*)
"""
    s = run(code)
    # Stata 18: count 7, yh[3] 5.571428775787353516 (float), yh[1] .,
    # se[3] 1.348846793174743652, rs[2] 3
    close(s.data["se"][2], 1.348846793174743652, rtol=1e-12)
    close(value(s, "r(N)"), 7)
    close(s.data["yh"][2], 5.571428775787353516, rtol=1e-12)
    assert np.isnan(s.data["yh"][0])
    same(s.data["rs"][1], 3)
    assert "x_L1" not in s.data.columns


def test_tsset_delta():
    code = """
set obs 6
gen t = _n*5
gen x = _n
tsset t, delta(5)
gen lx = L.x
gen dx = D.x
gen fx = F2.x
"""
    out = run(code).data
    # Stata 18: lx . 1 2 3 4 5; dx . 1 1 1 1 1; fx 3 4 5 6 . .
    same(out["lx"], [NAN, 1, 2, 3, 4, 5])
    same(out["dx"], [NAN, 1, 1, 1, 1, 1])
    same(out["fx"], [3, 4, 5, 6, NAN, NAN])
    panel = """
set obs 6
gen id = ceil(_n/3)
gen t = mod(_n-1,3)*2
gen x = _n
xtset id t, delta(2)
gen lx = L.x
"""
    # Stata 18: . 1 2 . 4 5
    same(run(panel).data["lx"], [NAN, 1, 2, NAN, 4, 5])
    off_grid = """
set obs 4
gen t = cond(_n<4, _n*5, 17)
gen x = _n
tsset t, delta(5)
"""
    # Stata 18: "time values with period less than delta() found", r(451)
    with pytest.raises(MethodIncompatibility, match="less than delta"):
        run(off_grid)


def test_stacked_operators_with_dots():
    code = """
set obs 8
gen t = _n
gen x = _n^2
tsset t
gen fd = F.D.x
gen ld = L.D.x
"""
    out = run(code).data
    # F.D.x = x[t+1] - x[t]; L.D.x = x[t-1] - x[t-2] (Stata 18: as FD. / LD.)
    x = np.arange(1, 9, dtype=float) ** 2
    same(out["fd"], np.r_[x[1:] - x[:-1], NAN])
    same(out["ld"], np.r_[NAN, NAN, x[1:-1] - x[:-2]])
    reg = """
set obs 25
gen t = _n
gen double x = mod(_n*37, 31)/3
gen double y = 1 + .4*x + mod(_n*11, 7)/2
tsset t
reg D.y D.x L.D.x
"""
    s = run(reg)
    # Stata 18: e(N) 23, _b[D.x] .5580645161290321843,
    # _b[LD.x] -.095967741935483894
    close(value(s, "e(N)"), 23)
    close(value(s, "_b[D.x]"), 0.5580645161290321843, rtol=1e-10)
    close(value(s, "_b[LD.x]"), -0.095967741935483894, rtol=1e-10)


# ------------------------------------------------ 9. egen group / tag
def test_egen_group_and_tag_treat_empty_string_as_missing():
    code = """
set obs 6
gen s = ""
replace s = "a" in 1
replace s = "b" in 3
replace s = "a" in 5
replace s = "c" in 6
gen g = cond(_n == 6, ., mod(_n, 2) + 1)
egen g1 = group(s)
egen g2 = group(s), missing
egen g3 = group(g s)
egen g4 = group(g s), missing
egen t1 = tag(s)
egen t2 = tag(s), missing
egen t3 = tag(g s)
"""
    out = run(code).data
    # Stata 18 list output
    same(out["g1"], [1, NAN, 2, NAN, 1, 3])
    same(out["g2"], [2, 1, 3, 1, 2, 4])
    same(out["g3"], [1, NAN, 2, NAN, 1, NAN])
    same(out["g4"], [2, 1, 3, 1, 2, 4])
    same(out["t1"], [1, 0, 1, 0, 0, 1])
    same(out["t2"], [1, 1, 1, 0, 0, 1])
    same(out["t3"], [1, 0, 1, 0, 0, 0])


# ---------------------------------------- 10. e(df_r) / e(N_clust)
@pytest.mark.parametrize(
    "line, df_r, n_clust",
    [
        # Stata 18: display e(df_r) " " e(N_clust)
        ("reg y x1 x2, vce(cluster cl)", 9, 10),
        ("reg y x1 x2 [pw=w], vce(cluster cl)", 9, 10),
        ("areg y x1 x2, absorb(g) vce(cluster cl)", 9, 10),
        ("reghdfe y x1, absorb(g) vce(cluster cl)", 9, 10),
        ("xtset id t\nqui xtreg y x1 x2, fe vce(cluster id)", 14, 15),
        ("xtset id t\nqui xtreg y x1 x2, re vce(cluster id)", None, 15),
        ("logit yb x1 x2, vce(cluster cl)", None, 10),
        ("ivregress 2sls y x2 (x1 = h), vce(cluster cl)", None, 10),
        ("reg y x1 x2, vce(robust)", 57, None),
    ],
)
def test_cluster_degrees_of_freedom(line, df_r, n_clust):
    s = run(D + "gen id = ceil(_n/4)\ngen t = mod(_n-1, 4)\nqui " + line)
    e = s.stored["e"]
    assert e.get("df_r") == df_r
    assert e.get("N_clust") == n_clust


def test_cluster_rmse_keeps_n_minus_k():
    s = run(D + "qui reg y x1 x2, vce(cluster cl)")
    # Stata 18: Root MSE 1.2863 with N = 57, k = 3
    close(value(s, "e(rmse)"), 1.2863, rtol=1e-4)


# ------------------------------------------------------ 11. gsort
@pytest.mark.parametrize(
    "line, order",
    [
        # Stata 18 (one missing value, at id 2)
        ("gsort -a", [6, 4, 3, 5, 1, 2]),
        ("gsort -a, mfirst", [2, 6, 4, 3, 5, 1]),
        ("gsort +a, mfirst", [1, 5, 3, 4, 6, 2]),
        ("gsort -s", [6, 2, 5, 4, 1, 3]),
        ("gsort -s, mfirst", [6, 2, 5, 4, 1, 3]),
        ("gsort s, mfirst", [3, 1, 4, 5, 2, 6]),
    ],
)
def test_gsort_missing_values(line, order):
    code = """
set obs 6
gen id = _n
gen a = cond(_n==2, ., cond(_n==5, 2, _n))
gen s = cond(_n==3, "", word("q w e r t y", _n))
"""
    out = run(code + line).data
    assert out["id"].astype(int).tolist() == order


# ---------------------------------------------- 12. storage types
def test_generate_enforces_storage_types():
    code = """
set obs 1
gen byte b1 = 300
gen byte b2 = 100
gen byte b3 = 101
gen byte b4 = -127
gen byte b5 = -128
gen byte b6 = 1.7
gen byte b7 = -1.7
gen byte b9 = 100.5
gen int i1 = 40000
gen int i2 = 32740
gen int i3 = 32741
gen long l1 = 3000000000
gen long l2 = 2147483620
gen long l3 = 2147483621
gen long l6 = 2.5
gen str3 s1 = "abcdef"
gen str3 s2 = "ab"
gen str2 s4 = "éa"
gen byte r = 5
replace r = 300
gen str2 r3 = "ab"
replace r3 = "abcdef"
"""
    row = run(code).data.iloc[0]
    # Stata 18 list: b1 . b2 100 b3 . b4 -127 b5 . b6 1 b7 -1 b9 100
    # i1 . i2 32740 i3 . l1 . l2 2147483620 l3 . l6 2 s1 abc s2 ab s4 é;
    # replace promotes: r 300 ("byte now int"), r3 abcdef ("str2 now str6")
    for name in ("b1", "b3", "b5", "i1", "i3", "l1", "l3"):
        assert np.isnan(row[name]), name
    want = {"b2": 100, "b4": -127, "b6": 1, "b7": -1, "b9": 100, "i2": 32740,
            "l2": 2147483620, "l6": 2, "r": 300}  # fmt: skip
    for name, v in want.items():
        assert row[name] == v, name
    assert (row["s1"], row["s2"], row["s4"], row["r3"]) == ("abc", "ab", "é", "abcdef")


def test_str_width_that_cuts_a_character_is_refused():
    # Stata 18 stores `gen str3 s = "éèa"` as "é" plus one byte of "è"
    with pytest.raises(MethodIncompatibility, match="inside a character"):
        run('set obs 1\ngen str3 s = "éèa"')


# ---------------------------------------------- 13. xtile cutpoints
def test_xtile_tied_cutpoints():
    code = """
set obs 9
gen x = ceil(_n/3)
xtile q = x, cutpoints(x)
gen c = cond(_n <= 3, 2, .)
replace c = 1 in 4
gen y = _n - 4
xtile q2 = y, cutpoints(c)
"""
    out = run(code).data
    # Stata 18: q 1 1 1 4 4 4 7 7 7; q2 1 1 1 1 1 2 5 5 5
    same(out["q"], [1, 1, 1, 4, 4, 4, 7, 7, 7])
    same(out["q2"], [1, 1, 1, 1, 1, 2, 5, 5, 5])


# ------------------------------------------ 14. what Stata refuses
def test_append_type_mismatch():
    master_num = 'set obs 2\ngen v = "a"\nsave s\nclear\nset obs 2\ngen v = _n\n'
    # Stata 18: "variable v is float in master but str1 in using data", r(106)
    with pytest.raises(MethodIncompatibility, match="r\\(106\\)"):
        run(master_num + "append using s")
    # with force the using values are missing: 1 2 . .
    same(run(master_num + "append using s, force").data["v"], [1, 2, NAN, NAN])
    master_str = 'set obs 4\ngen v = _n\nsave n\nclear\nset obs 2\ngen v = "z"\n'
    # Stata 18: z z "" "" "" ""
    out = run(master_str + "append using n, force").data["v"].tolist()
    assert out == ["z", "z", "", "", "", ""]


@pytest.mark.parametrize(
    "values, message",
    [
        # Stata 18 r(452)
        ("cond(_n<3, 1.5, 2)", "noninteger"),
        ("cond(_n<3, -1, 2)", "negative"),
    ],
)
def test_factor_levels_must_be_nonnegative_integers(values, message):
    code = f"set obs 6\ngen y = _n\ngen gg = {values}\nreg y i.gg"
    with pytest.raises(MethodIncompatibility, match=message):
        run(code)


def test_recode_else_with_missing_is_refused():
    base = "set obs 6\ngen x = _n\n"
    # Stata 18: "keywords else/* and missing/nonmissing may not be
    # combined", r(198)
    with pytest.raises(MethodIncompatibility, match="may not be"):
        run(base + "recode x (1=5) (missing=9) (else=3)")
    # (nonmissing=3) with (missing=9) is fine: 5 3 3 3 3 3
    same(run(base + "recode x (1=5) (missing=9) (nonmissing=3)").data["x"],
         [5, 3, 3, 3, 3, 3])  # fmt: skip


# ------------------------------ 15. / 16. factor-variable notation
D15 = D.replace("gen h =", "replace g = 2 in 1/5\ngen h =")


@pytest.mark.parametrize(
    "line, probes",
    [
        # Stata 18 e(b) (g = 1, 2, 3 has 19, 23, 18 rows)
        ("reg y ib(last).g x1",
         {"_b[1.g]": -1.91593517694342785, "_b[2.g]": -0.92157456205349686,
          "_b[_cons]": 4.23805302645716786}),
        ("reg y ib(freq).g x1",
         {"_b[1.g]": -0.99436061488993166, "_b[3.g]": 0.921574562053495083,
          "_b[_cons]": 3.31647846440367333}),
        ("reg y ib(first).g x1",
         {"_b[2.g]": 0.994360614889930439, "_b[3.g]": 1.9159351769434263}),
        ("reg y i.g#c.x1",
         {"_b[1.g#c.x1]": 0.337642931289061754,
          "_se[2.g#c.x1]": 0.0734039191097964755,
          "_b[_cons]": 3.42063790546076252}),
        ("reg y ibn.g x1, noconstant",
         {"_b[1.g]": 2.322117849513742449, "_se[3.g]": 0.4103129196441696602,
          "_b[x1]": 0.483145933881841738}),
    ],
)  # fmt: skip
def test_factor_notation_and_names(line, probes):
    s = run(D15 + "qui " + line)
    for expr, want in probes.items():
        close(value(s, expr), want, rtol=1e-10)


@pytest.mark.parametrize(
    "line, message",
    [
        ("reg y i(2 3).g x1", "level list"),
        ("reg y x1 o.x2", "omitted operator"),
    ],
)
def test_unsupported_factor_operators_are_refused(line, message):
    with pytest.raises(MethodIncompatibility, match=message):
        run(D15 + line)
    out = sp.from_stata("reg y ib(last).g x1")
    assert not out["ok"] and "ib#.var" in out["error"]


def test_tsset_delta_reaches_operators_inside_expressions():
    # Stata 18: set obs 6 / gen t = _n*5 / gen x = _n / tsset t, delta(5)
    #           summarize L.x  ->  r(mean) = 3, r(N) = 5
    s = run("""
set obs 6
gen t = _n*5
gen x = _n
tsset t, delta(5)
quietly summarize L.x
""")
    close(value(s, "r(mean)"), 3.0)
    close(value(s, "r(N)"), 5.0)
