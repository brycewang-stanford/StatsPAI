"""sp.abnormal_returns: the trading calendar and incomplete windows.

iangow/era_statpai#38. Relative days count the days the market was open
(``calendar='market'``, the dates of the return file), not the rows of the
security, and ``incomplete=`` says what happens to an event whose window
has a day without a return. Every expected number below is computed from
the return file by hand, with the rule of Gow and Ding's Event studies
chapter: number the market's trading days, take the days of the window,
add up the returns that are there.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _file(seed=7, n_days=120, n_sec=4):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", periods=n_days)
    mkt = rng.normal(0, 0.01, n_days)
    rows = [
        pd.DataFrame(
            {
                "id": j,
                "date": days,
                "ret": 0.9 * mkt + rng.normal(0, 0.012, n_days),
                "mkt": mkt,
            }
        )
        for j in range(n_sec)
    ]
    return pd.concat(rows, ignore_index=True), days


def _quiet(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.abnormal_returns(*args, **kwargs)


def _by_hand(data, days, unit, day0, lo, hi):
    """The chapter's sums: the rows of ``unit`` on market days
    ``day0 + lo .. day0 + hi`` that have a return."""
    td = pd.Series(np.arange(len(days)), index=days)
    own = data[data["id"] == unit].dropna(subset=["ret"])
    pos = td.reindex(own["date"]).to_numpy()
    at = int(td[day0])
    inside = own[(pos >= at + lo) & (pos <= at + hi)]
    return (
        float(inside["ret"].sum()),
        float((inside["ret"] - inside["mkt"]).sum()),
        len(inside),
    )


KW = dict(model="market_adjusted", market="mkt", inference=False)


def test_a_day_without_a_row_is_still_a_trading_day():
    data, days = _file()
    gap = data[~((data["id"] == 1) & (data["date"] == days[79]))]
    events = pd.DataFrame({"id": [1], "event_date": [days[80]]})

    # day -1 of the market is days[79], which security 1 has no row for
    with pytest.warns(UserWarning, match="missing return in the event window"):
        with pytest.raises(Exception, match="no usable event"):
            sp.abnormal_returns(gap, events, event_window=(-1, 1), **KW)

    res = sp.abnormal_returns(gap, events, event_window=(-1, 1), incomplete="sum", **KW)
    raw, car, n = _by_hand(gap, days, 1, days[80], -1, 1)
    row = res.events.iloc[0]
    assert (row["window_days"], row["n_days"], n) == (3, 2, 2)
    assert row["raw"] == pytest.approx(raw, abs=1e-15)
    assert row["car"] == pytest.approx(car, abs=1e-15)
    assert np.isnan(res.ar.loc[-1].iloc[0]) and res.ar.loc[[0, 1]].notna().all().all()

    # the security's own rows: day -1 is days[78], one market day earlier
    own = sp.abnormal_returns(
        gap, events, event_window=(-1, 1), calendar="security", **KW
    )
    sec = data[data["id"] == 1].set_index("date")
    want = sec.loc[[days[78], days[80], days[81]]]
    assert own.events["n_days"].iloc[0] == 3
    assert own.events["car"].iloc[0] == pytest.approx(
        float((want["ret"] - want["mkt"]).sum()), abs=1e-15
    )


def test_the_three_calendars_agree_on_a_complete_file():
    data, days = _file()
    events = pd.DataFrame({"id": range(4), "event_date": days[[70, 75, 80, 85]]})
    base = sp.abnormal_returns(
        data, events, market="mkt", estimation_window=(-60, -5), calendar="security"
    )
    for calendar in ("market", days, list(days), pd.Series(days), days.to_numpy()):
        got = sp.abnormal_returns(
            data, events, market="mkt", estimation_window=(-60, -5), calendar=calendar
        )
        pd.testing.assert_frame_equal(got.events, base.events, check_exact=True)
        pd.testing.assert_frame_equal(got.tests, base.tests, check_exact=True)
    assert (base.events["n_days"] == base.events["window_days"]).all()


def test_a_given_calendar_counts_days_no_security_has():
    data, days = _file()
    thin = data[data["date"] != days[79]]  # no security traded that day
    events = pd.DataFrame({"id": [0, 1], "event_date": [days[80], days[80]]})
    # the file's own dates: days[79] never happened, the window is whole
    seen = sp.abnormal_returns(thin, events, event_window=(-1, 1), **KW)
    assert (seen.events["n_days"] == 3).all()
    # the index file says the market was open: the window lacks a day
    told = sp.abnormal_returns(
        thin, events, event_window=(-1, 1), calendar=days, incomplete="sum", **KW
    )
    assert (told.events["n_days"] == 2).all()
    for unit in (0, 1):
        _, car, _ = _by_hand(thin, days, unit, days[80], -1, 1)
        got = told.events.loc[told.events["id"] == unit, "car"].iloc[0]
        assert got == pytest.approx(car, abs=1e-15)


def test_the_chapter_sums_on_an_unbalanced_file():
    data, days = _file(seed=11, n_days=200, n_sec=12)
    rng = np.random.default_rng(3)
    keep = rng.random(len(data)) > 0.08
    keep |= (data["id"] == 0).to_numpy()  # one security shows every day
    thin = data[keep].copy()
    thin.loc[thin.sample(40, random_state=1).index, "ret"] = np.nan
    events = pd.DataFrame(
        {
            "id": rng.integers(1, 12, 60),
            "event_date": days[rng.integers(5, 195, 60)],
        }
    )
    res = sp.abnormal_returns(
        thin, events, event_window=(-2, 2), incomplete="sum", **KW
    )
    assert res.n_events + len(res.skipped) == 60
    short = 0
    for label, row in res.events.iterrows():
        raw, car, n = _by_hand(thin, days, row["id"], row["event_day"], -2, 2)
        assert row["n_days"] == n and row["window_days"] == 5
        assert row["raw"] == pytest.approx(raw, abs=1e-15)
        assert row["car"] == pytest.approx(car, abs=1e-15)
        short += n < 5
    assert short > 5  # the design has incomplete windows to sum

    # the default keeps exactly the complete ones, with the same numbers
    strict = _quiet(thin, events, event_window=(-2, 2), **KW)
    whole = res.events[res.events["n_days"] == 5]
    pd.testing.assert_frame_equal(strict.events, whole, check_exact=True)


def test_zero_counts_a_missing_return_against_the_market():
    data, days = _file()
    holed = data.copy()
    at = holed.index[(holed["id"] == 2) & (holed["date"] == days[60])][0]
    holed.loc[at, "ret"] = np.nan
    events = pd.DataFrame({"id": [2], "event_date": [days[61]]})
    res = sp.abnormal_returns(
        holed, events, event_window=(-2, 2), incomplete="zero", **KW
    )
    sec = data[data["id"] == 2].set_index("date")
    window = sec.loc[days[59:64]]
    raw = float(window["ret"].drop(days[60]).sum())
    row = res.events.iloc[0]
    assert (row["window_days"], row["n_days"]) == (5, 4)
    assert row["raw"] == pytest.approx(raw, abs=1e-15)
    # the market return of the missing day stays in the benchmark
    assert row["car"] == pytest.approx(raw - float(window["mkt"].sum()), abs=1e-15)
    assert res.returns.loc[-1].iloc[0] == 0.0

    # no row at all: the benchmark of that day is unknown, the day is left out
    gone = data.drop(index=at)
    left = sp.abnormal_returns(
        gone, events, event_window=(-2, 2), incomplete="zero", **KW
    )
    summed = sp.abnormal_returns(
        gone, events, event_window=(-2, 2), incomplete="sum", **KW
    )
    pd.testing.assert_frame_equal(left.events, summed.events, check_exact=True)
    assert np.isnan(left.ar.loc[-1].iloc[0])


def test_zero_does_not_fill_before_the_first_return():
    data, days = _file()
    late = data[~((data["id"] == 3) & (data["date"] < days[50]))].copy()
    events = pd.DataFrame({"id": [3], "event_date": [days[51]]})
    res = sp.abnormal_returns(
        late, events, event_window=(-3, 1), incomplete="zero", **KW
    )
    # days -3 and -2 precede the listing: left out, not counted as zero
    assert res.events["n_days"].iloc[0] == 3
    assert res.ar.iloc[:2, 0].isna().all()


def test_a_window_past_the_end_of_the_file():
    data, days = _file()
    events = pd.DataFrame({"id": [0, 1], "event_date": [days[118], days[60]]})
    with pytest.warns(UserWarning, match="outside the return series"):
        strict = sp.abnormal_returns(data, events, event_window=(-1, 3), **KW)
    assert strict.n_events == 1
    res = sp.abnormal_returns(
        data, events, event_window=(-1, 3), incomplete="sum", **KW
    )
    row = res.events.iloc[0]
    _, car, n = _by_hand(data, days, 0, days[118], -1, 3)
    assert (row["window_days"], row["n_days"], n) == (5, 3, 3)
    assert row["car"] == pytest.approx(car, abs=1e-15)


def test_the_variance_of_an_incomplete_car_uses_its_own_days():
    data, days = _file(n_days=160)
    gap = data[~((data["id"] == 1) & (data["date"] == days[139]))]
    events = pd.DataFrame({"id": [0, 1], "event_date": [days[140], days[140]]})
    res = sp.abnormal_returns(
        gap,
        events,
        model="market_adjusted",
        market="mkt",
        event_window=(-1, 1),
        estimation_window=(-100, -5),
        incomplete="sum",
    )
    ev = res.events.set_index("id")
    assert ev.loc[1, "n_days"] == 2 and ev.loc[0, "n_days"] == 3
    for unit in (0, 1):
        assert ev.loc[unit, "se"] ** 2 == pytest.approx(
            ev.loc[unit, "resid_var"] * ev.loc[unit, "n_days"], rel=1e-12
        )
    # the estimation days are market days too: security 1 has all 96
    assert ev.loc[1, "n_est"] == 96


def test_estimation_days_are_market_days():
    data, days = _file(n_days=160)
    # security 1 has no rows on twenty market days of the estimation window
    cut = data[~((data["id"] == 1) & data["date"].isin(days[60:80]))]
    events = pd.DataFrame({"id": [0, 1], "event_date": [days[140], days[140]]})
    kw = dict(market="mkt", event_window=(0, 0), estimation_window=(-100, -5))
    market = sp.abnormal_returns(cut, events, **kw).events.set_index("id")
    own = sp.abnormal_returns(cut, events, calendar="security", **kw)
    assert market.loc[1, "n_est"] == 76  # 96 market days, 20 without a row
    assert own.events.set_index("id").loc[1, "n_est"] == 96


def test_polars_reads_the_calendar_from_the_whole_file():
    pl = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    data, days = _file(seed=5, n_days=150, n_sec=8)
    rng = np.random.default_rng(2)
    keep = (rng.random(len(data)) > 0.1) | (data["id"] == 7).to_numpy()
    thin = data[keep].reset_index(drop=True)
    # the event securities miss days that only the others show
    events = pd.DataFrame({"id": [1, 2, 3], "event_date": days[[100, 110, 120]]})
    want = sp.abnormal_returns(
        thin, events, event_window=(-3, 3), incomplete="sum", **KW
    )
    assert (want.events["n_days"] < 7).any()
    got = sp.abnormal_returns(
        pl.from_pandas(thin),
        pl.from_pandas(events),
        event_window=(-3, 3),
        incomplete="sum",
        **KW,
    )
    cols = ["id", "n_est", "window_days", "n_days", "raw", "car"]
    pd.testing.assert_frame_equal(got.events[cols], want.events[cols], check_exact=True)
    assert list(got.events["event_day"]) == list(want.events["event_day"])


def test_refusals():
    data, days = _file()
    events = pd.DataFrame({"id": [0], "event_date": [days[60]]})
    with pytest.raises(MethodIncompatibility, match="incomplete must be"):
        sp.abnormal_returns(data, events, incomplete="drop", **KW)
    with pytest.raises(MethodIncompatibility, match="calendar must be"):
        sp.abnormal_returns(data, events, calendar="nyse", **KW)
    with pytest.raises(MethodIncompatibility, match="calendar must be"):
        sp.abnormal_returns(data, events, calendar=[], **KW)
    # a calendar that lacks a date of the security is not a calendar of it
    with pytest.raises(MethodIncompatibility, match="not in calendar"):
        sp.abnormal_returns(data, events, calendar=days[::2], **KW)
