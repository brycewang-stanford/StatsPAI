"""Time-series operators in sp.stata: L. F. D. and lag lists.

Stata resolves a lag against the time variable of ``tsset`` / ``xtset``:
``L.x`` at time t is x at t - 1 in the same panel, and missing when that
period is absent ([U] 11.4.4 Time-series varlists). The expected values
here are written out from that definition, on data with a gap and on an
unsorted panel, where ``shift`` on the rows gives something else.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent._translation._stata_tsops import rewrite_ts_operators
from statspai.exceptions import MethodIncompatibility

NA = np.nan


def _run(text, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.stata(text, data=data)


@pytest.fixture()
def gapped():
    # t = 1, 2, 3, 5, 6: period 4 is absent
    return pd.DataFrame({"t": [1, 2, 3, 5, 6], "x": [10.0, 12.0, 15.0, 19.0, 24.0]})


@pytest.mark.parametrize(
    "term, want",
    [
        ("L.x", [NA, 10, 12, NA, 19]),
        ("l.x", [NA, 10, 12, NA, 19]),
        ("L1.x", [NA, 10, 12, NA, 19]),
        ("L2.x", [NA, NA, 10, 15, NA]),
        ("F.x", [12, 15, NA, 24, NA]),
        ("D.x", [NA, 2, 3, NA, 5]),
        # D2 is the difference of the difference
        ("D2.x", [NA, NA, 1, NA, NA]),
        # LD = lag of the difference
        ("LD.x", [NA, NA, 2, NA, NA]),
        ("L2D.x", [NA, NA, NA, 3, NA]),
    ],
)
def test_operator_values_follow_the_time_variable(gapped, term, want):
    line, cols = rewrite_ts_operators(f"gen z = {term}", gapped, (None, "t"))
    (name,) = cols
    assert line == f"gen z = {name}"
    np.testing.assert_array_equal(cols[name], np.array(want, dtype=float))


def test_row_shift_would_be_wrong_across_the_gap(gapped):
    _, cols = rewrite_ts_operators("gen z = L.x", gapped, (None, "t"))
    by_time = cols["x_L1"]
    by_row = gapped.x.shift(1).to_numpy()
    assert np.isnan(by_time[3]) and by_row[3] == 15.0


def test_lag_list_expands_in_order(gapped):
    line, cols = rewrite_ts_operators("reg y L(0/2).x", gapped, (None, "t"))
    assert line == "reg y x x_L1 x_L2"
    assert list(cols) == ["x_L1", "x_L2"]


def test_panel_lags_stay_inside_the_unit_whatever_the_row_order():
    panel = pd.DataFrame(
        {
            "id": np.repeat([1, 2], 4),
            "yr": [2000, 2001, 2002, 2003] * 2,
            "v": [1.0, 2.0, 4.0, 8.0, 100.0, 200.0, 400.0, 800.0],
        }
    ).sample(frac=1, random_state=3)
    _, cols = rewrite_ts_operators("gen z = L.v", panel, ("id", "yr"))
    got = pd.Series(cols["v_L1"], index=panel.index).sort_index().to_numpy()
    np.testing.assert_array_equal(got, [NA, 1, 2, 4, NA, 100, 200, 400])


def test_period_time_variable():
    data = pd.DataFrame(
        {"q": pd.period_range("2001Q1", periods=4, freq="Q"), "x": [1.0, 2.0, 4.0, 7.0]}
    )
    _, cols = rewrite_ts_operators("gen z = D.x", data, (None, "q"))
    np.testing.assert_array_equal(cols["x_D1"], [NA, 1, 2, 3])


# ----------------------------------------------------------- in sp.stata
@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(6)
    n = 160
    x = rng.normal(size=n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.5 * y[i - 1] + 0.4 * x[i - 1] + rng.normal()
    return pd.DataFrame({"t": np.arange(n), "y": y, "x": x})


def test_regression_with_operators_is_the_adl(series):
    got = _run("tsset t\nreg y L.y L(1/2).x, r", series)
    assert list(got.params.index) == ["Intercept", "y_L1", "x_L1", "x_L2"]
    want = sp.ardl(series, "y", "x", lags=1, x_lags=2, vce="hc1")
    np.testing.assert_allclose(
        got.params.to_numpy(), want.params.to_numpy(), rtol=1e-12
    )
    np.testing.assert_allclose(
        got.std_errors.to_numpy(), want.std_errors.to_numpy(), rtol=1e-12
    )
    assert "x_L1" not in series.columns  # the caller's frame is untouched


def test_test_and_lincom_name_the_same_coefficients(series):
    a = _run("tsset t; reg y L.y L(1/2).x; test L.x L2.x", series)
    b = _run("tsset t; reg y L.y L(1/2).x; test x_L1 x_L2", series)
    assert a == b
    lin = _run("tsset t; reg y L.y L(1/2).x; lincom L.x + L2.x", series)
    fit = _run("tsset t; reg y L.y L(1/2).x", series)
    assert lin["estimate"] == pytest.approx(fit.params["x_L1"] + fit.params["x_L2"])


def test_newey_and_dfuller_after_tsset(series):
    got = _run("tsset t; newey y L.x, lag(3)", series)
    lagged = series.assign(x_L1=series.x.shift(1))
    want = sp.regress("y ~ x_L1", data=lagged, robust="hac", hac_lags=3, hac_small=True)
    np.testing.assert_allclose(got.std_errors.to_numpy(), want.std_errors.to_numpy())


def test_operators_in_generate_and_if(series):
    out = _run("tsset t; gen double dy = D.y; summarize dy if L.x > 0", series)
    dy = series.y.diff()
    keep = series.x.shift(1) > 0  # L.x is missing at t = 0, and D.y is too
    assert out.loc["dy", "N"] == int((keep & dy.notna()).sum())
    np.testing.assert_allclose(out.loc["dy", "Mean"], dy[keep].mean())


@pytest.mark.parametrize(
    "text, reason",
    [
        ("reg y L.y", "tsset time"),  # no time variable declared
        ("tsset t; reg y S12.y", "not resolved"),  # seasonal difference
        ("tsset t; reg y L.nope", "not resolved"),
        ("tsset t; reg y L(3/1).x", "empty lag list"),
    ],
)
def test_what_cannot_be_resolved_is_refused(series, text, reason):
    with pytest.raises(MethodIncompatibility, match=reason):
        _run(text, series)


def test_bad_time_variables_are_refused(series):
    dated = series.assign(day=pd.date_range("2020-01-01", periods=len(series)))
    with pytest.raises(MethodIncompatibility, match="no unit step"):
        _run("tsset day; reg y L.y", dated)
    twice = pd.concat([series, series.iloc[:3]])
    with pytest.raises(MethodIncompatibility, match="repeated time values"):
        _run("tsset t; reg y L.y", twice)


def test_a_file_name_is_not_mistaken_for_an_operator(series):
    # `d.csv` looks like D.csv; csv is not a variable, so the text is left
    # alone and the line is handled as the export it is
    with pytest.warns(UserWarning, match="skipped 'export'"):
        got = sp.stata("tsset t; export delimited d.csv; reg y x", data=series)
    assert got.nobs == len(series)
