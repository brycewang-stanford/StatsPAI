"""sp.xtset / sp.lag / sp.lead / sp.diff / sp.reshape / sp.collapse and
sp.stata(..., return_data=True).

Every expected number below was printed by Stata 18 MP (batch mode,
2026-10-19) on the same rows; the do-file is reproduced in the comments
of each fixture. Stata stores a mean / sd / median of a ``float``
variable in single precision, the public functions keep double precision,
so those columns are compared at float32 resolution (rtol 1e-7); counts,
sums of exactly representable values and lags are exact.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import ColumnNotFound, MethodIncompatibility

NA = np.nan
F32 = dict(rtol=1e-7, atol=0)


# --------------------------------------------------------------------------
# xtset: Stata's r(balanced), r(gaps), r(tmin), r(tmax)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ids, ts, balanced, gaps, tmax",
    [
        # input id t / 1 1 / 1 2 / 2 1 / 2 2 -> strongly balanced, gaps 0
        ([1, 1, 2, 2], [1, 2, 1, 2], "strongly balanced", False, 2),
        # replace t = 3 in 4 -> weakly balanced, but with a gap
        ([1, 1, 2, 2], [1, 2, 1, 3], "weakly balanced", True, 3),
        # 1 1 / 1 3 / 2 1 / 2 3 -> strongly balanced, but with gaps
        ([1, 1, 2, 2], [1, 3, 1, 3], "strongly balanced", True, 3),
        # 1 1 / 1 2 / 1 4 / 2 1 / 2 2 / 2 3 / 3 2 -> unbalanced, with a gap
        ([1, 1, 1, 2, 2, 2, 3], [1, 2, 4, 1, 2, 3, 2], "unbalanced", True, 4),
    ],
)
def test_xtset_describes_the_panel_as_stata(ids, ts, balanced, gaps, tmax):
    df = pd.DataFrame({"id": ids, "t": ts})
    decl = sp.xtset(df, id="id", time="t")
    assert (decl.balanced, decl.gaps, decl.t_min, decl.t_max) == (
        balanced,
        gaps,
        1.0,
        float(tmax),
    )
    assert decl.n_units == len(set(ids))
    assert sp.xtset(df).to_dict() == decl.to_dict()  # the stored declaration


def test_xtset_delta_closes_the_gap_and_refuses_a_shorter_step():
    df = pd.DataFrame({"id": [1, 1, 2, 2], "t": [1, 3, 1, 3]})
    # Stata: xtset id t, delta(2) -> strongly balanced, r(gaps) = 0
    assert sp.xtset(df, id="id", time="t", delta=2).gaps is False
    # Stata r(451): time values with period less than delta() found
    with pytest.raises(MethodIncompatibility, match="less than delta"):
        sp.xtset(pd.DataFrame({"t": [1, 2, 4]}), time="t", delta=2)


def test_xtset_ignores_rows_without_a_period_but_lags_refuse_them():
    # Stata accepts missing periods in xtset (they are not described)
    df = pd.DataFrame({"id": [1, 1, 2], "t": [1.0, 2.0, NA], "x": [1.0, 2, 3]})
    decl = sp.xtset(df, id="id", time="t")
    assert (decl.t_min, decl.t_max, decl.gaps) == (1.0, 2.0, False)
    with pytest.raises(MethodIncompatibility, match="missing or non-integer"):
        sp.lag(df, "x")


@pytest.mark.parametrize(
    "ts, message",
    [
        ([1.5, 2.5], "non-integer"),  # Stata: must contain only integer values
        ([1, 1], "repeated time values"),  # Stata: repeated time values
    ],
)
def test_xtset_refuses_what_stata_refuses(ts, message):
    df = pd.DataFrame({"id": [1, 1], "t": ts})
    with pytest.raises(MethodIncompatibility, match=message):
        sp.xtset(df, id="id", time="t")


def test_xtset_leaves_the_rows_alone_and_names_missing_columns():
    df = pd.DataFrame({"id": [2, 1], "t": [1, 1]})
    sp.xtset(df, id="id", time="t")
    assert df["id"].tolist() == [2, 1]
    with pytest.raises(ColumnNotFound):
        sp.xtset(df, id="firm", time="t")


# --------------------------------------------------------------------------
# lag / lead / diff: an unsorted, unbalanced panel with gaps
# --------------------------------------------------------------------------


@pytest.fixture()
def panel():
    # input id t x (in this row order); xtset id t; gen double lx = L.x ...
    return pd.DataFrame(
        {
            "id": [2, 1, 1, 3, 2, 1, 2, 3, 3, 1],
            "t": [3, 4, 1, 2, 1, 2, 2, 5, 4, 6],
            "x": [7.5, 9.25, 1.5, 6.125, 2.75, 4.5, NA, 11, 8, 13.5],
        }
    )


# Stata 18, `list obs id t x lx fx dx d2x l2x ldx f2x` in the input order
STATA_OPS = {
    "L.x": [NA, NA, NA, NA, NA, 1.5, 2.75, 8, NA, NA],
    "F.x": [NA, NA, 4.5, NA, NA, NA, 7.5, NA, 11, NA],
    "D.x": [NA, NA, NA, NA, NA, 3, NA, 3, NA, NA],
    "D2.x": [NA] * 10,
    "L2.x": [2.75, 4.5, NA, NA, NA, NA, NA, NA, 6.125, 9.25],
    "F2.x": [NA, 13.5, NA, 8, 7.5, 9.25, NA, NA, NA, NA],
}


def test_lags_match_stata_on_an_unsorted_gapped_panel(panel):
    sp.xtset(panel, id="id", time="t")
    got = {
        "L.x": sp.lag(panel, "x"),
        "F.x": sp.lead(panel, "x"),
        "D.x": sp.diff(panel, "x"),
        "D2.x": sp.diff(panel, "x", 2),
        "L2.x": sp.lag(panel, "x", 2),
        "F2.x": sp.lead(panel, "x", 2),
    }
    for name, series in got.items():
        assert series.name == name
        assert series.index.equals(panel.index)
        np.testing.assert_array_equal(series.to_numpy(), STATA_OPS[name])
    # a row shift would put 9.25 (t = 4) after 4.5 (t = 2) in panel 1
    assert np.isnan(got["L.x"].iloc[1])


def test_explicit_panel_overrides_and_lists_give_a_frame(panel):
    frame = sp.lag(panel, "x", n=[0, 1, 2], id="id", time="t")
    assert list(frame.columns) == ["x", "L.x", "L2.x"]
    np.testing.assert_array_equal(frame["L2.x"].to_numpy(), STATA_OPS["L2.x"])
    np.testing.assert_array_equal(frame["x"].to_numpy(), panel["x"].to_numpy())


def test_delta_two_steps_two_periods():
    # tsset t, delta(2): L.x at 9 looks for 7, which is absent
    df = pd.DataFrame({"t": [1, 3, 5, 9], "x": [1.0, 4, 9, 16]})
    np.testing.assert_array_equal(
        sp.lag(df, "x", time="t", delta=2).to_numpy(), [NA, 1, 4, NA]
    )
    sp.xtset(df, time="t", delta=2)
    np.testing.assert_array_equal(sp.diff(df, "x").to_numpy(), [NA, 3, 5, NA])


def test_period_time_variable_steps_one_period():
    df = pd.DataFrame(
        {"m": pd.period_range("2020-01", periods=4, freq="M"), "x": [1.0, 2, 4, 8]}
    ).drop(index=2)
    np.testing.assert_array_equal(sp.lag(df, "x", time="m").to_numpy(), [NA, 1, NA])


def test_lag_without_a_time_variable_names_the_fix():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(MethodIncompatibility, match="no time variable") as err:
        sp.lag(df, "x")
    assert "sp.xtset" in err.value.recovery_hint
    with pytest.raises(MethodIncompatibility, match="order"):
        sp.lag(df.assign(t=[1, 2]), "x", n=-1, time="t")


# --------------------------------------------------------------------------
# reshape
# --------------------------------------------------------------------------


@pytest.fixture()
def long_inc():
    # input hid year inc sex
    return pd.DataFrame(
        {
            "hid": [1, 1, 2, 2, 3],
            "year": [80, 81, 80, 82, 81],
            "inc": [5.5, 6, 7, 8.25, 9],
            "sex": [0, 0, 1, 1, 0],
        }
    )


def test_reshape_wide_and_back_as_stata(long_inc):
    wide = sp.reshape(long_inc, "wide", stubs="inc", id="hid", j="year")
    # Stata: hid inc80 inc81 inc82 sex
    assert list(wide.columns) == ["hid", "inc80", "inc81", "inc82", "sex"]
    expected = [[1, 5.5, 6, NA, 0], [2, 7, NA, 8.25, 1], [3, NA, 9, NA, 0]]
    np.testing.assert_array_equal(wide.to_numpy(dtype=float), expected)
    back = sp.reshape(wide, "long", stubs=["inc"], i="hid", j="year")
    assert list(back.columns) == ["hid", "year", "inc", "sex"]
    np.testing.assert_array_equal(
        back.to_numpy(dtype=float),
        [
            [1, 80, 5.5, 0],
            [1, 81, 6, 0],
            [1, 82, NA, 0],
            [2, 80, 7, 1],
            [2, 81, NA, 1],
            [2, 82, 8.25, 1],
            [3, 80, NA, 0],
            [3, 81, 9, 0],
            [3, 82, NA, 0],
        ],
    )


def test_reshape_refuses_what_stata_refuses(long_inc):
    dup = pd.concat([long_inc, long_inc.iloc[[0]]])
    with pytest.raises(MethodIncompatibility, match="not unique"):
        sp.reshape(dup, "wide", stubs="inc", id="hid", j="year")
    varying = long_inc.assign(sex=[0, 1, 1, 1, 0])
    with pytest.raises(MethodIncompatibility, match="not constant"):
        sp.reshape(varying, "wide", stubs="inc", id="hid", j="year")
    with pytest.raises(MethodIncompatibility, match="'wide' or 'long'"):
        sp.reshape(long_inc, "across", stubs="inc", id="hid", j="year")


def test_sp_stata_reshape_runs_the_same_code(long_inc):
    run = sp.stata("reshape wide inc, i(hid) j(year)", data=long_inc, return_data=True)
    pd.testing.assert_frame_equal(
        run.data, sp.reshape(long_inc, "wide", stubs="inc", id="hid", j="year")
    )


# --------------------------------------------------------------------------
# collapse
# --------------------------------------------------------------------------


@pytest.fixture()
def grouped():
    # input g h x y w f
    return pd.DataFrame(
        {
            "g": [1, 1, 1, 2, 2, 2, NA, NA],
            "h": [1, 1, 2, 1, 1, 2, 1, 2],
            "x": [1.5, 2.5, 4, NA, 7, 9.5, 3, 6],
            "y": [2, NA, 5, 1, 8, 4, 3, NA],
            "w": [1, 2, 0.5, 1, 3, 2, 1, 4],
            "f": [2, 1, 3, 1, 2, 1, 1, 2],
        }
    )


def test_collapse_statistics_match_stata(grouped):
    out = sp.collapse(
        grouped,
        "(mean) mx=x (sd) sx=x (sum) tx=x (count) nx=x (median) medx=x "
        "(min) minx=x (max) maxx=x (firstnm) fx=x (lastnm) lx=x (first) f0=x "
        "(last) l0=x",
        by="g",
    )
    # the group with a missing g is listed last
    np.testing.assert_array_equal(out["g"].to_numpy(), [1, 2, NA])
    np.testing.assert_allclose(out["mx"], [2.666666746139526367, 8.25, 4.5], **F32)
    np.testing.assert_allclose(
        out["sx"],
        [1.258305788040161133, 1.767766952514648438, 2.121320247650146484],
        **F32,
    )
    exact = {
        "tx": [8, 16.5, 9],
        "nx": [3, 2, 2],
        "medx": [2.5, 8.25, 4.5],
        "minx": [1.5, 7, 3],
        "maxx": [4, 9.5, 6],
        "fx": [1.5, 7, 3],
        "lx": [4, 9.5, 6],
        "f0": [1.5, NA, 3],  # (first) takes the first row, missing or not
        "l0": [4, 9.5, 6],
    }
    for name, want in exact.items():
        np.testing.assert_array_equal(out[name].to_numpy(dtype=float), want)


def test_collapse_cw_and_dict_spelling(grouped):
    out = sp.collapse(
        grouped,
        {"mx": ("mean", "x"), "my": ("mean", "y"), "nx": ("count", "x"),
         "ny": ("count", "y")},
        by=["g"],
        cw=True,
    )  # fmt: skip
    np.testing.assert_array_equal(
        out[["mx", "my", "nx", "ny"]].to_numpy(dtype=float),
        [[2.75, 3.5, 2, 2], [8.25, 6, 2, 2], [3, 3, 1, 1]],
    )


def test_collapse_weights_match_stata(grouped):
    aw = sp.collapse(
        grouped,
        "(mean) mx=x (sum) tx=x (count) nx=x (sd) sx=x (median) medx=x",
        by="g",
        weights="w",
    )
    np.testing.assert_allclose(
        aw["mx"], [2.428571462631225586, 8, 5.400000095367431641], **F32
    )
    np.testing.assert_allclose(
        aw["tx"], [7.285714285714285587, 16, 10.80000000000000071], rtol=1e-15
    )
    np.testing.assert_array_equal(aw["nx"], [3, 2, 2])
    np.testing.assert_allclose(
        aw["sx"],
        [0.9502953290939331055, 1.732050776481628418, 1.697056293487548828],
        **F32,
    )
    np.testing.assert_array_equal(aw["medx"], [2.5, 7, 6])

    fw = sp.collapse(
        grouped,
        "(mean) mx=x (sum) tx=x (count) nx=x (sd) sx=x (rawsum) rx=x",
        by="g",
        weights="f",
        weight_type="fweight",
    )
    np.testing.assert_allclose(
        fw["mx"], [2.916666746139526367, 7.833333492279052734, 5], **F32
    )
    np.testing.assert_array_equal(fw["tx"], [17.5, 23.5, 15])
    np.testing.assert_array_equal(fw["nx"], [6, 3, 3])
    np.testing.assert_allclose(
        fw["sx"],
        [1.241638660430908203, 1.443375706672668457, 1.732050776481628418],
        **F32,
    )
    np.testing.assert_array_equal(fw["rx"], [8, 16.5, 9])

    pw = sp.collapse(
        grouped,
        "(mean) mx=x (sum) tx=x (count) nx=x",
        by=["g", "h"],
        weights="w",
        weight_type="pweight",
    )
    np.testing.assert_allclose(pw["mx"], [2.166666746139526367, 4, 7, 9.5, 3, 6], **F32)
    np.testing.assert_array_equal(pw["tx"], [6.5, 2, 21, 19, 3, 24])
    np.testing.assert_array_equal(pw["nx"], [3, 0.5, 3, 2, 1, 4])


def test_collapse_without_by_is_one_row(grouped):
    out = sp.collapse(grouped, "(mean) mx=x (sum) tx=x")
    np.testing.assert_allclose(out["mx"], [4.785714149475097656], **F32)
    assert out["tx"].tolist() == [33.5]


def test_collapse_refuses_what_stata_refuses(grouped):
    with pytest.raises(MethodIncompatibility, match="noninteger frequency"):
        sp.collapse(grouped, "(mean) x", weights="w", weight_type="fweight")
    with pytest.raises(MethodIncompatibility, match="not allowed with pweights"):
        sp.collapse(grouped, "(sd) x", weights="w", weight_type="pweight")
    with pytest.raises(MethodIncompatibility, match="used twice"):
        sp.collapse(grouped, "(mean) g=x", by="g")
    with pytest.raises(MethodIncompatibility, match="not implemented"):
        sp.collapse(grouped, "(p75) x")
    with pytest.raises(ColumnNotFound):
        sp.collapse(grouped, "(mean) z")


def test_collapse_leaves_the_input_alone(grouped):
    before = grouped.copy()
    sp.collapse(grouped, "(mean) x", by="g", weights="w")
    pd.testing.assert_frame_equal(grouped, before)


# --------------------------------------------------------------------------
# sp.stata(..., return_data=True)
# --------------------------------------------------------------------------


def test_return_data_gives_the_dataset_the_commands_left():
    df = pd.DataFrame({"g": [1, 1, 2], "x": [1.0, 3.0, 5.0]})
    run = sp.stata(
        "generate double y = 2 * x\ncollapse (mean) y, by(g)",
        data=df,
        return_data=True,
    )
    assert run.output is None or run.output is not df
    np.testing.assert_array_equal(run.data["y"].to_numpy(), [4.0, 10.0])
    assert list(df.columns) == ["g", "x"]  # the caller's frame is untouched
    output, data = run  # a named tuple
    assert data is run.data


def test_return_data_restores_preserved_data_as_a_do_file_end_does():
    # Stata 18: `preserve` / `drop if _n > 2` in a do-file run by another
    # do-file; `count` afterwards gives 10, not 2
    df = pd.DataFrame({"x": np.arange(10.0)})
    run = sp.stata("preserve\ndrop if x > 1\ncount", data=df, return_data=True)
    assert run.output == 2
    assert len(run.data) == 10
