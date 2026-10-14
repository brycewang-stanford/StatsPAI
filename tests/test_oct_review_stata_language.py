"""The language layer of ``sp.stata`` against Stata 18 (October 2026 review).

Each expected value below is what Stata 18 MP printed for the same lines;
the Stata output is quoted next to the assertion. The items: string
concatenation, ``round()``, the numeric display formats (``string()``,
``display %fmt``), string literals in the command tokeniser, macro names
built from macros, compound quotes from ``levelsof``, ``capture { }``
blocks, scalars against variable abbreviations, ``regexs()`` across
commands, ``date()`` with a topyear, the refusal of an unknown command, and
``tempvar`` clean-up.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent._translation import _stata_run
from statspai.agent._translation._stata_functions import _string, stata_format
from statspai.agent._translation._stata_run import StataSession
from statspai.agent._translation._stata_script import split_commands
from statspai.exceptions import MethodIncompatibility


def _session(code: str, data: pd.DataFrame | None = None) -> StataSession:
    s = StataSession(pd.DataFrame({"x": [1.0, 2.0]}) if data is None else data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for line in split_commands(code):
            s.run(line)
    return s


def _shown(code: str) -> str:
    """What the last `display` of ``code`` prints."""
    return str(_session(code).output)


# ------------------------------------------------------------ 1. strings
def test_string_concatenation_of_literals():
    # . display ("a" + "b")        -> ab
    assert _shown('display ("a" + "b")') == "ab"
    # . local s = "a" + "b" / display "`s'"   -> ab
    assert _shown('local s = "a" + "b"\ndisplay "`s\'"') == "ab"
    # . local s2 = "ab" + string(1) / display "`s2'"   -> ab1
    assert _shown('local s2 = "ab" + string(1)\ndisplay "`s2\'"') == "ab1"
    # . gen s3 = "a" + "b"   -> s3 = ab in every row
    s = _session('gen s3 = "a" + "b"')
    assert s.data["s3"].tolist() == ["ab", "ab"]


def test_display_of_a_leading_string_then_plus_is_an_error():
    # . display "a" + "b"   -> a+"b" invalid name, r(198)
    with pytest.raises(MethodIncompatibility):
        _session('display "a" + "b"')


# --------------------------------------------------------------- 2. round
def test_round_is_y_times_floor_of_x_over_y_plus_half():
    # . display round(-1.5) " " round(-0.5) " " round(-2.5,1) " "
    # >   round(-0.25,.5) " " round(7,0) " " round(-3,2) " " round(3,2) " "
    # >   round(2.5) " " round(0.5) " " round(1.5,-1) " " round(.,1) " "
    # >   round(1,.) " " round(-7,0) " " round(1.25, .1) " " round(-1.25,.1)
    # -1 0 -2 0 7 -2 4 3 1 1 . . -7 1.3 -1.2
    code = (
        'display round(-1.5) " " round(-0.5) " " round(-2.5,1) " " '
        'round(-0.25,.5) " " round(7,0) " " round(-3,2) " " round(3,2) " " '
        'round(2.5) " " round(0.5) " " round(1.5,-1) " " round(.,1) " " '
        'round(1,.) " " round(-7,0) " " round(1.25, .1) " " round(-1.25,.1)'
    )
    assert _shown(code) == "-1 0 -2 0 7 -2 4 3 1 1 . . -7 1.3 -1.2"


@pytest.mark.parametrize(
    "expr, stata",
    [
        # . display %21.0g round(...), Stata 18
        ("round(0.285,.01)", ".2800000000000000266"),
        ("round(1.005,.01)", "1"),
        ("round(2.675,.01)", "2.68000000000000016"),
        ("round(4.35,.1)", "4.299999999999999822"),
        ("round(-1.25,.1)", "-1.200000000000000178"),
        ("round(123.456, 0.01)", "123.460000000000008"),
        ("round(5, 0.3)", "5.099999999999999645"),
        ("round(-0.5, -1)", "-1"),
    ],
)
def test_round_matches_stata_to_the_last_bit(expr, stata):
    assert _shown(f"display %21.0g {expr}").strip() == stata


# ------------------------------------------------- 3. numeric formats
# string(x, fmt), Stata 18 (a selection of a 2,340-cell grid checked in the
# review; every cell agreed)
_STRING_CASES = [
    # default format %9.0g
    (1 / 3, None, ".3333333"),
    (123456789, None, "1.23e+08"),
    (1e10, None, "1.00e+10"),
    (12345678, None, "1.23e+07"),
    (1234567, None, "1234567"),
    (1e-5, None, ".00001"),
    (0.00001234, None, ".0000123"),
    (1.23456789e-7, None, "1.23e-07"),
    (9999999.5, None, "1.00e+07"),
    (0.999999999, None, "1"),
    (1.5e300, None, "1.5e+300"),
    (1e-300, None, "1.0e-300"),
    (-1 / 3, None, "-.3333333"),
    (np.nan, None, "."),
    # %g of other widths and digits
    (1 / 3, "%4.0g", ".33"),
    (0.05, "%4.0g", "5.0e-02"),
    (12.5, "%4.0g", "13"),
    (0.125, "%4.0g", ".13"),
    (1.5e300, "%8.0g", "2.e+300"),
    (1234567, "%8.0g", "1.2e+06"),
    (123456789, "%10.0g", "1.235e+08"),
    (1e10, "%12.0g", "1.00000e+10"),
    (1e15, "%18.0g", "1000000000000000"),
    (1 / 3, "%21.0g", ".3333333333333333148"),
    (0.01, "%21.0g", ".0100000000000000002"),
    (123.456, "%9.3g", "123"),
    (12345.678, "%9.3g", "12346"),
    (99.5, "%9.1g", "1.0e+02"),
    (9.96, "%9.1g", "10"),
    (3.5, "%4.1g", "3.5"),
    (0.0996, "%4.1g", "1.0e-01"),
    (3.5, "%2.0g", "3.5e+00"),
    (0.000012345, "%7.1g", "1.2e-05"),
    (123456789012, "%12.3g", "1.23e+11"),
    (0.000123, "%24.17g", ".00012300000000000001"),
    (1 / 3, "%-9.0g", ".3333333 "),
    # %f
    (0.5, "%9.3f", "0.500"),
    (-0.5, "%9.0f", "-0"),
    (2.5, "%9.0f", "2"),
    (0.125, "%9.2f", "0.12"),
    (0.285, "%4.2f", "0.28"),
    (1234567, "%9.3f", "1.23e+06"),
    (12345678, "%9.1f", "1.2e+07"),
    (12345678, "%02.0f", "12345678"),
    (123456789, "%3.0f", "1.2e+08"),
    (-1234.5678, "%6.2f", "-1234.57"),
    (-12345.678, "%6.2f", "-1.2e+04"),
    (5, "%02.0f", "05"),
    (-1, "%010.3f", "-00001.000"),
    (12345, "%20.15f", "1.2345000000000e+04"),
    (1.5e300, "%6.2f", "2.e+300"),
    # %e
    (0, "%10.3e", "0.000e+00"),
    (0.125, "%8.1e", "1.3e-01"),
    (1.5e300, "%10.3e", "1.50e+300"),
    (123456789, "%8.4e", "1.2e+08"),
    (3.5, "%24.17e", "3.50000000000000000e+00"),
    # commas
    (1234567, "%9.0fc", "1,234,567"),
    (12345678, "%9.0fc", "12345678"),
    (123456789, "%12.2fc", "123456789.00"),
    (1234.5678, "%8.0gc", "1,234.6"),
    (123456.7, "%8.0gc", "123457"),
    (1e10, "%15.0gc", "10000000000"),
    (-987654321, "%15.0gc", "-987,654,321"),
]


@pytest.mark.parametrize("value, fmt, stata", _STRING_CASES)
def test_string_formats_as_stata(value, fmt, stata):
    got = _string(value) if fmt is None else _string(value, fmt)
    assert got == stata


def test_string_of_float_data_uses_the_stored_float():
    # . gen float x = 1/3 / gen float y = 123456789 / gen float z = 0.1
    # . display string(x) "|" string(y) "|" string(z)
    #   .3333333|1.23e+08|.1
    # . display string(x, "%18.0g") "|" string(z,"%18.0g")
    #   .3333333432674408|.1000000014901161
    s = _session(
        "clear\nset obs 1\ngen float x = 1/3\ngen float y = 123456789\n"
        "gen float z = 0.1\n"
        'display string(x) "|" string(y) "|" string(z) "|" '
        'string(x, "%18.0g") "|" string(z,"%18.0g")'
    )
    assert str(s.output) == (".3333333|1.23e+08|.1|.3333333432674408|.1000000014901161")


@pytest.mark.parametrize(
    "line, stata",
    [
        # . display "[" %9.2f 1/3 "]"      -> [     0.33]
        ('display "[" %9.2f 1/3 "]"', "[     0.33]"),
        # . display "[" 1/3 "]"            -> [.33333333]
        ('display "[" 1/3 "]"', "[.33333333]"),
        # . display 0.00001                -> .00001
        ("display 0.00001", ".00001"),
        # . display 123456789              -> 1.235e+08
        ("display 123456789", "1.235e+08"),
        # . display 1e10                   -> 1.000e+10
        ("display 1e10", "1.000e+10"),
        # . display "[" %9.0g 1/3 "]" "[" %-9.2f 2 "]" "[" %9.0fc 1234567 "]"
        # >   "[" %5.0fc 12345 "]" "[" %6.0fc 12345 "]"
        # [ .3333333][2.00     ][1,234,567][12345][12,345]
        (
            'display "[" %9.0g 1/3 "]" "[" %-9.2f 2 "]" "[" %9.0fc 1234567 "]" '
            '"[" %5.0fc 12345 "]" "[" %6.0fc 12345 "]"',
            "[ .3333333][2.00     ][1,234,567][12345][12,345]",
        ),
        # . display "[" 2 "]" "[" 1e15 "]" "[" 123456789 "]"
        # [2][1.000e+15][1.235e+08]
        (
            'display "[" 2 "]" "[" 1e15 "]" "[" 123456789 "]"',
            "[2][1.000e+15][1.235e+08]",
        ),
        # . display %09.0g 1/3 "|" %09.2e 1/3 "|" %-9.0gc 1234567 "|"
        # >   %12.0gc 1234567.25 "|" %9.0gc 1234567
        #  .3333333| 3.33e-01|1234567  | 1,234,567.3|  1234567
        (
            'display %09.0g 1/3 "|" %09.2e 1/3 "|" %-9.0gc 1234567 "|" '
            '%12.0gc 1234567.25 "|" %9.0gc 1234567',
            " .3333333| 3.33e-01|1234567  | 1,234,567.3|  1234567",
        ),
    ],
)
def test_display_prints_as_stata(line, stata):
    assert _shown(line) == stata


def test_display_of_one_number_is_still_that_number():
    out = _session("display %9.2f 1/3").output
    assert isinstance(out, float) and out == pytest.approx(1 / 3)
    assert str(out) == "     0.33"  # Stata: display %9.2f 1/3


def test_display_extended_function_keeps_the_padding():
    # . local fmt : display %6.2f 2/3 / display "[`fmt']"   -> [  0.67]
    # . local f2 : display %9.0g 1/3 / display "[`f2']"     -> [ .3333333]
    s = _session("local fmt : display %6.2f 2/3\nlocal f2 : display %9.0g 1/3")
    assert s._macros.locals["fmt"] == "  0.67"
    assert s._macros.locals["f2"] == " .3333333"


def test_stata_format_without_pad_is_unchanged_api():
    assert stata_format(1 / 3, "%9.2f") == "0.33"
    assert stata_format(1 / 3, "%9.2f", pad=True) == "     0.33"


# ------------------------------------------------ 4./5. string literals
def test_blanks_inside_a_string_literal_are_kept():
    # . gen s = "a  b" / gen n = strlen(s)   -> n = 4
    s = _session('gen s = "a  b"\ngen n = strlen(s)')
    assert s.data["s"].iloc[0] == "a  b"
    assert s.data["n"].iloc[0] == 4


def test_comma_comment_marker_and_semicolon_inside_strings():
    # . gen t = "1,234" / gen u = "http://a.b/c" / gen w = "a;b"
    # . list   -> t = 1,234   u = http://a.b/c   w = a;b
    # . display "a // b" "|" "c /* d */ e"   -> a // b|c /* d */ e
    s = _session(
        'gen t = "1,234"\ngen u = "http://a.b/c"\ngen w = "a;b"\n'
        'gen v = "a // b" // a real comment\n'
        'display "a // b" "|" "c /* d */ e"'
    )
    assert s.data["t"].iloc[0] == "1,234"
    assert s.data["u"].iloc[0] == "http://a.b/c"
    assert s.data["w"].iloc[0] == "a;b"
    assert s.data["v"].iloc[0] == "a // b"
    assert str(s.output) == "a // b|c /* d */ e"


# ---------------------------------------------------- 6. macro names
def test_a_macro_name_built_from_a_macro():
    # . local v x / local m`v' = 5 / local m = 9 / display "mx=`mx' m=`m'"
    # mx=5 m=9
    code = "local v x\nlocal m`v' = 5\nlocal m = 9\ndisplay \"mx=`mx' m=`m'\""
    assert _shown(code) == "mx=5 m=9"
    # . scalar c`v' = 3 / display cx -> 3; global g`v' = 7 / display "$gx" -> 7
    # . local m`v'y "hi" / display "`mxy'" -> hi
    s = _session("local v x\nscalar c`v' = 3\nglobal g`v' = 7\nlocal m`v'y \"hi\"")
    assert s.stored["scalars"]["cx"] == 3.0
    assert s._macros.globals["gx"] == "7"
    assert s._macros.locals["mxy"] == "hi"


# ----------------------------------------------------- 7. levelsof
def test_levelsof_string_values_with_blanks_loop_back():
    # . levelsof s, local(ls)   -> `"x y"' `"z"'
    # . foreach v of local ls { local ++j / replace k = `j' if s == "`v'" }
    # . list   -> k = 1 2 1 2;  local n : word count `ls' -> n=2
    code = (
        "clear\nset obs 4\n"
        'gen s = cond(mod(_n,2), "x y", "z")\ngen k = 0\n'
        "levelsof s, local(ls)\nlocal j = 0\n"
        "foreach v of local ls {\n    local ++j\n"
        "    replace k = `j' if s == \"`v'\"\n}\n"
        "local n : word count `ls'"
    )
    s = _session(code)
    assert s.data["k"].tolist() == [1, 2, 1, 2]
    assert s._macros.locals["n"] == "2"


# ---------------------------------------------------- 8. capture { }
def test_capture_block_sets_rc_and_keeps_what_ran():
    # . capture { gen y = x / gen y = 2 }   -> display _rc: 110, y kept
    # . capture { gen z = nosuch }          -> 111
    # . capture { replace x = x + 1 }       -> 0, x replaced
    s = _session(
        "clear\nset obs 3\ngen x = _n\n"
        "capture {\n gen y = x\n gen y = 2\n}\nscalar r1 = _rc\n"
        "capture {\n gen z = nosuch\n}\nscalar r2 = _rc\n"
        "capture {\n replace x = x + 1\n}\nscalar r3 = _rc"
    )
    scalars = s.stored["scalars"]
    assert (scalars["r1"], scalars["r2"], scalars["r3"]) == (110.0, 111.0, 0.0)
    assert s.data["y"].tolist() == [1, 2, 3]
    assert s.data["x"].tolist() == [2, 3, 4]
    assert "z" not in s.data.columns


# ------------------------------------------- 9. scalar or variable
def test_unique_variable_abbreviation_beats_a_scalar():
    # . gen cl = _n / scalar c = 100 / gen z = c / list   -> z = 1 2 3
    s = _session("clear\nset obs 3\ngen cl = _n\nscalar c = 100\ngen z = c")
    assert s.data["z"].tolist() == [1, 2, 3]
    # with cl and clb, c is ambiguous and the scalar is taken:
    # . gen z2 = c / display z2[1]   -> 100
    s = _session(
        "clear\nset obs 3\ngen cl = _n\ngen clb = 5\nscalar c = 100\ngen z2 = c"
    )
    assert s.data["z2"].tolist() == [100, 100, 100]
    # . scalar cl2 = 50 / gen z4 = cl2 -> 50 (no variable starts with cl2)
    s = _session("clear\nset obs 3\ngen cl = _n\nscalar cl2 = 50\ngen z4 = cl2")
    assert s.data["z4"].tolist() == [50, 50, 50]
    # no scalar and an ambiguous abbreviation:
    # . capture noisily gen z5 = c  -> c ambiguous abbreviation, r(111)
    with pytest.raises(MethodIncompatibility):
        _session("clear\nset obs 3\ngen cl = _n\ngen clb = 5\ngen z5 = c")


# ----------------------------------------------------- 10. regexs
def test_regexs_reads_each_row_within_one_command():
    # . gen g0 = regexs(0) if regexm(s, "[0-9]")   -> 1, 9, (missing)
    s = _session(
        'clear\nset obs 3\ngen s = "ab12cd"\nreplace s = "x9" in 2\n'
        'replace s = "none" in 3\ngen g0 = regexs(0) if regexm(s, "[0-9]")'
    )
    assert s.data["g0"].tolist() == ["1", "9", ""]


def test_regexs_after_an_earlier_commands_regexm_is_refused():
    # Stata 18 keeps only the last evaluated observation's match:
    # . gen m = regexm(s, "([a-z]+)([0-9]+)") / gen g1 = regexs(1) if m
    #   -> g1 empty in every row (row 3, "none", matched nothing)
    # . gen m2 = regexm(s, "([a-z]+)") if _n < 3 / gen t = regexs(1)
    #   -> t = "x" in every row (row 2 was evaluated last)
    with pytest.raises(MethodIncompatibility, match="regexs"):
        _session(
            'clear\nset obs 3\ngen s = "ab12cd"\nreplace s = "x9" in 2\n'
            'replace s = "none" in 3\ngen m = regexm(s, "([a-z]+)([0-9]+)")\n'
            "gen g1 = regexs(1) if m"
        )


def test_regexs_after_a_scalar_regexm_in_an_if_block():
    # . display regexm("q77", "([0-9]+)") " " regexs(1)   -> 1 77
    code = 'if regexm("q77", "([0-9]+)") {\n  local a = regexs(1)\n}'
    assert _session(code)._macros.locals["a"] == "77"


# ------------------------------------------------------ 11. date()
def test_date_with_a_topyear():
    # . display date("01/02/20","MDY",2050) " " date("01/02/60","MDY",2050)
    # >   " " date("01/02/50","MDY",2050) " " date("01/02/51","MDY",2050)
    # 21916 1 32874 -3286
    # . display date("01/02/20","MDY") " " date("01/02/2020","MDY",2050) " "
    # >   date("01/02/2060","MDY",2050) " " date("01/02/20","MDY",.) " "
    # >   date("01/02/5","MDY",2050)
    # . 21916 36526 . .
    # . display date("01/02/20","MD19Y") " " date("01/02/20","MD20Y") " "
    # >   date("01/02/2020","MD19Y")
    # -14609 21916 21916
    # . display date("01/02/20","MDY",1999) " " date("01/02/99","MDY",1999)
    # >   " " date("01/02/00","MDY",1999) " " date("01/02/20","MDY",2050.7)
    # >   " " date("200102","YMD",2050) " " date("1jan20","DMY",2050) " "
    # >   date("01/02/120","MDY",2050)
    # -14609 14246 -21913 21916 21916 21915 -672045
    # . display date("01/02/20","MD19Y",2050) " " date("01/02/20","MD20Y",1999)
    # . .
    # . display date("01/02/20","MDY",-5) " " date("01/02/20","MDY",50)
    # . .
    cases = {
        'date("01/02/20","MDY",2050)': 21916,
        'date("01/02/60","MDY",2050)': 1,
        'date("01/02/50","MDY",2050)': 32874,
        'date("01/02/51","MDY",2050)': -3286,
        'date("01/02/20","MDY")': np.nan,
        'date("01/02/2020","MDY",2050)': 21916,
        'date("01/02/2060","MDY",2050)': 36526,
        'date("01/02/20","MDY",.)': np.nan,
        'date("01/02/5","MDY",2050)': np.nan,
        'date("01/02/20","MD19Y")': -14609,
        'date("01/02/20","MD20Y")': 21916,
        'date("01/02/2020","MD19Y")': 21916,
        'date("01/02/20","MDY",1999)': -14609,
        'date("01/02/99","MDY",1999)': 14246,
        'date("01/02/00","MDY",1999)': -21913,
        'date("01/02/20","MDY",2050.7)': 21916,
        'date("200102","YMD",2050)': 21916,
        'date("1jan20","DMY",2050)': 21915,
        'date("01/02/120","MDY",2050)': -672045,
        'date("01/02/20","MD19Y",2050)': np.nan,
        'date("01/02/20","MD20Y",1999)': np.nan,
        'date("01/02/20","MDY",-5)': np.nan,
        'date("01/02/20","MDY",50)': np.nan,
    }
    s = _session("clear\nset obs 1")
    for expr, stata in cases.items():
        got = s.value(expr)
        if np.isnan(stata):
            assert np.isnan(got), expr
        else:
            assert got == stata, expr


# ----------------------------------------- 12. unknown command message
def test_unknown_command_is_not_offered_an_unrelated_one():
    # Stata 18 runs tsfill; sp.stata does not, and says so
    with pytest.raises(MethodIncompatibility) as err:
        _session("clear\nset obs 2\ngen t = _n\ntsset t\ntsfill")
    text = str(err.value)
    assert "xtserial" not in text
    assert "not supported" in text or "not translated" in text


def test_a_typo_still_gets_a_suggestion():
    with pytest.raises(MethodIncompatibility, match="Did you mean: .*summarize"):
        _session("summarise x")


# ----------------------------------------------------- 13. tempvar
def test_tempvar_names_and_their_end():
    # . tempvar tv / display "`tv'"   -> __000000 (tempname next: __000001)
    s = _session("tempvar tv\ntempname nm")
    assert s._macros.locals["tv"] == "__000000"
    assert s._macros.locals["nm"] == "__000001"
    # a program's tempvar is gone when it ends:
    # . program define mk / tempvar t / gen `t' = x * 2 / gen y = `t' + 1 / end
    # . mk / describe, simple   -> x  y;   display y[2] -> 5
    s = _session(
        "clear\nset obs 2\ngen x = _n\nprogram define mk\n    tempvar t\n"
        "    gen `t' = x * 2\n    gen y = `t' + 1\nend\nmk"
    )
    assert list(s.data.columns) == ["x", "y"]
    assert s.data["y"].tolist() == [3, 5]


def test_tempvar_is_dropped_at_the_end_of_the_snippet(monkeypatch):
    # do-file: tempvar t / gen `t' = x * 2 / gen y = `t' + 1
    # inside: describe, simple -> x  __000002  y;  after the do-file: x  y
    made = []

    class Recording(StataSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(_stata_run, "StataSession", Recording)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp.stata(
            "clear\nset obs 2\ngen x = _n\ntempvar t\ngen `t' = x * 2\n"
            "gen y = `t' + 1",
            pd.DataFrame({"a": [1.0]}),
        )
    assert list(made[0].data.columns) == ["x", "y"]


def test_numbers_in_macros_are_written_in_18_0g():
    # . local a = 1/3 / local b = 1e20 / local c = 2/3*1e-10 / local d = 0.1
    # . display "`a'|`b'|`c'|`d'"
    # .3333333333333333|1.00000000000e+20|6.66666666667e-11|.1
    code = (
        "local a = 1/3\nlocal b = 1e20\nlocal c = 2/3*1e-10\nlocal d = 0.1\n"
        "display \"`a'|`b'|`c'|`d'\""
    )
    assert _shown(code) == ".3333333333333333|1.00000000000e+20|6.66666666667e-11|.1"
    # . forvalues v = 0(0.1)0.35 { display "`v'" }   -> 0 .1 .2 .3
    loop = 'local s ""\nforvalues v = 0(0.1)0.35 {\n local s "`s\' `v\'"\n}'
    s = _session(loop)
    assert s._macros.locals["s"].split() == ["0", ".1", ".2", ".3"]
