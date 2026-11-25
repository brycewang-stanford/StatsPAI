"""GitHub #58: ``sp.abnormal_returns(inference=False)`` returns abnormal
returns and CARs with no standard errors or tests; the market-adjusted
model then reads the event window alone."""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def _issue_frames():
    days = pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-15"])
    returns = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "date": list(days) * 2,
            "ret": [0.01, 0.02, -0.01, 0.005, 0.0, 0.007],
            "mkt": [0.0, 0.005, 0.002] * 2,
        }
    )
    events = pd.DataFrame({"id": [1, 2], "event_date": [days[1]] * 2})
    return returns, events


def _panel(n_ids=12, n_days=320, seed=5):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2019-01-01", periods=n_days)
    mkt = rng.normal(0, 0.01, n_days)
    rows = [
        pd.DataFrame(
            {
                "id": j,
                "date": days,
                "ret": 0.8 * mkt + rng.normal(0, 0.012, n_days),
                "mkt": mkt,
            }
        )
        for j in range(n_ids)
    ]
    events = pd.DataFrame({"id": range(n_ids), "event_date": days[290:302]})
    return pd.concat(rows, ignore_index=True), events, days


def test_the_example_of_the_issue_gives_the_two_sums():
    returns, events = _issue_frames()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = sp.abnormal_returns(
            returns,
            events,
            model="market_adjusted",
            market="mkt",
            event_window=(-1, 1),
            inference=False,
        )
    # (0.01 - 0) + (0.02 - 0.005) + (-0.01 - 0.002); 0.005 - 0.005 + 0.005
    assert res.events["car"].tolist() == pytest.approx([0.013, 0.005], abs=1e-15)
    assert res.ar[0].tolist() == pytest.approx([0.01, 0.015, -0.012], abs=1e-15)
    assert res.caar == pytest.approx(0.009, abs=1e-15)
    assert res.aar["caar"].iloc[-1] == pytest.approx(0.009, abs=1e-15)
    assert res.skipped == []
    assert res.inference is False
    assert res.tests.empty
    assert np.isnan(res.se) and np.isnan(res.pvalue)
    assert "se" not in res.events.columns
    # the returns themselves, next to the abnormal returns
    assert res.events["raw"].tolist() == pytest.approx([0.02, 0.012], abs=1e-15)
    assert res.returns[1].tolist() == [0.005, 0.0, 0.007]
    assert "inference=False" in res.summary()


def test_one_event_is_enough_without_inference():
    returns, events = _issue_frames()
    res = sp.abnormal_returns(
        returns,
        events.iloc[:1],
        model="market_adjusted",
        market="mkt",
        inference=False,
    )
    assert res.n_events == 1
    assert res.caar == pytest.approx(0.013, abs=1e-15)


def test_the_default_still_needs_the_estimation_window():
    returns, events = _issue_frames()
    with pytest.warns(UserWarning, match="left out"):
        with pytest.raises(DataInsufficient, match="inference=False"):
            sp.abnormal_returns(returns, events, model="market_adjusted", market="mkt")


@pytest.mark.parametrize("model", ["market_adjusted", "market", "mean_adjusted"])
def test_returns_equal_those_of_the_run_with_inference(model):
    data, events, _ = _panel()
    kw = dict(model=model, event_window=(-2, 2))
    if model != "mean_adjusted":
        kw["market"] = "mkt"
    full = sp.abnormal_returns(data, events, **kw)
    only = sp.abnormal_returns(data, events, inference=False, **kw)
    pd.testing.assert_frame_equal(only.ar, full.ar, check_exact=True)
    pd.testing.assert_frame_equal(only.aar, full.aar, check_exact=True)
    np.testing.assert_array_equal(only.events["car"], full.events["car"])
    assert only.caar == full.caar


def test_market_adjusted_ignores_the_estimation_window_without_inference():
    data, events, days = _panel()
    late = data[data["date"] >= days[288]]  # two days before the first event
    res = sp.abnormal_returns(
        late,
        events,
        model="market_adjusted",
        market="mkt",
        event_window=(-1, 1),
        estimation_window=(-5, 0),  # overlaps the event window: not read
        inference=False,
    )
    full = sp.abnormal_returns(
        data, events, model="market_adjusted", market="mkt", event_window=(-1, 1)
    )
    np.testing.assert_array_equal(res.events["car"], full.events["car"])
    assert (res.events["n_est"] == 0).all()


def test_a_fitted_model_still_needs_its_estimation_window():
    data, events, days = _panel()
    late = data[data["date"] >= days[288]]
    with pytest.warns(UserWarning, match="estimation-window returns"):
        with pytest.raises(DataInsufficient, match="no usable event"):
            sp.abnormal_returns(late, events, market="mkt", inference=False)
    with pytest.raises(MethodIncompatibility, match="must end before"):
        sp.abnormal_returns(
            data, events, market="mkt", estimation_window=(-50, 0), inference=False
        )


def test_incomplete_event_windows_are_still_left_out():
    returns, events = _issue_frames()
    returns.loc[2, "ret"] = np.nan
    with pytest.warns(UserWarning, match="missing return in the event window"):
        res = sp.abnormal_returns(
            returns, events, model="market_adjusted", market="mkt", inference=False
        )
    assert res.events["id"].tolist() == [2]
    with pytest.warns(UserWarning, match="outside the return series"):
        with pytest.raises(DataInsufficient, match="no usable event"):
            sp.abnormal_returns(
                returns,
                events,
                model="market_adjusted",
                market="mkt",
                event_window=(-2, 2),
                inference=False,
            )


def test_inference_must_be_a_boolean():
    returns, events = _issue_frames()
    with pytest.raises(MethodIncompatibility, match="True or False"):
        sp.abnormal_returns(
            returns, events, model="market_adjusted", market="mkt", inference="no"
        )


def test_securities_without_an_event_do_not_change_the_result():
    data, events, _ = _panel()
    some = events.iloc[:5]
    all_rows = sp.abnormal_returns(data, some, market="mkt")
    own_rows = sp.abnormal_returns(data[data["id"] < 5], some, market="mkt")
    pd.testing.assert_frame_equal(all_rows.events, own_rows.events, check_exact=True)
    pd.testing.assert_frame_equal(all_rows.tests, own_rows.tests, check_exact=True)


def test_the_returns_only_result_serialises():
    returns, events = _issue_frames()
    res = sp.abnormal_returns(
        returns, events, model="market_adjusted", market="mkt", inference=False
    )
    d = res.to_dict()
    assert d["estimate"] == pytest.approx(0.009, abs=1e-15)
    assert "inference=False" in repr(res)


# --- events of several days (iangow/era_statpai#38) -------------------------


def test_event_end_gives_each_event_its_own_window():
    data, events, days = _panel()
    events = events.assign(end=events["event_date"])
    # event 0 runs three trading days, event 1 ends on a Saturday (-> Monday)
    events.loc[0, "end"] = days[292]
    events.loc[1, "end"] = days[291] + pd.Timedelta(days=4 - days[291].weekday() + 1)
    res = sp.abnormal_returns(
        data,
        events,
        model="market_adjusted",
        market="mkt",
        event_end="end",
        event_window=(-1, 1),
        inference=False,
    )
    by_id = {j: g.reset_index(drop=True) for j, g in data.groupby("id")}

    def want(j, first, last):
        g = by_id[j]
        a = g.index[g["date"] == first][0] - 1
        b = g.index[g["date"] == last][0] + 1
        part = g.iloc[a : b + 1]
        return part["ret"].sum(), (part["ret"] - part["mkt"]).sum(), len(part)

    monday = days[days > events.loc[1, "end"]][0]
    assert events.loc[1, "end"].weekday() == 5
    for label, first, last in ((0, days[290], days[292]), (1, days[291], monday)):
        raw, car, n = want(label, first, last)
        assert res.events.loc[label, "raw"] == pytest.approx(raw, rel=1e-13)
        assert res.events.loc[label, "car"] == pytest.approx(car, rel=1e-13)
        assert res.ar[label].notna().sum() == n
        assert res.events.loc[label, "end_day"] == last
    # the other events are one-day events: three days each, as without event_end
    plain = sp.abnormal_returns(
        data, events, model="market_adjusted", market="mkt", inference=False
    )
    np.testing.assert_array_equal(
        res.events["car"].to_numpy()[2:], plain.events["car"].to_numpy()[2:]
    )
    assert res.ar.index.tolist() == list(range(-1, res.ar.shape[0] - 1))
    assert res.aar["n"].iloc[0] == len(events) and res.aar["n"].iloc[-1] < len(events)
    assert res.caar == pytest.approx(res.events["car"].mean(), rel=1e-15)


def test_event_end_variance_uses_the_length_of_each_window():
    data, events, days = _panel()
    events = events.assign(end=events["event_date"])
    events.loc[0, "end"] = days[294]  # five event days, seven in the window
    res = sp.abnormal_returns(
        data, events, model="market_adjusted", market="mkt", event_end="end"
    )
    ev = res.events
    np.testing.assert_allclose(
        ev["se"].to_numpy() ** 2,
        ev["resid_var"].to_numpy() * res.ar.notna().sum().to_numpy(),
        rtol=1e-12,
    )
    assert res.ar[0].notna().sum() == 7 and res.ar[1].notna().sum() == 3


def test_event_end_equal_to_the_event_date_changes_nothing():
    data, events, _ = _panel()
    same = events.assign(end=events["event_date"])
    same.loc[3, "end"] = pd.NaT  # a missing end date: a one-day event
    a = sp.abnormal_returns(data, events, market="mkt")
    b = sp.abnormal_returns(data, same, market="mkt", event_end="end")
    pd.testing.assert_frame_equal(
        a.events, b.events.drop(columns=["event_end", "end_day"]), check_exact=True
    )
    pd.testing.assert_frame_equal(a.tests, b.tests, check_exact=True)
    pd.testing.assert_frame_equal(a.ar, b.ar, check_exact=True)


def test_event_end_refusals():
    from statspai.exceptions import ColumnNotFound

    data, events, days = _panel()
    with pytest.raises(ColumnNotFound):
        sp.abnormal_returns(data, events, market="mkt", event_end="end")
    bad = events.assign(end=events["event_date"])
    bad.loc[0, "end"] = days[200]  # before the event date
    bad.loc[1, "end"] = days[-1]  # the window then runs off the series
    with pytest.warns(UserWarning, match="left out"):
        res = sp.abnormal_returns(data, bad, market="mkt", event_end="end")
    assert [s["reason"] for s in res.skipped] == [
        "event end before the event date",
        "event window runs outside the return series",
    ]


def test_raw_is_the_sum_of_the_returns_in_the_window():
    data, events, days = _panel()
    res = sp.abnormal_returns(data, events, market="mkt", event_window=(-2, 2))
    g = data[data["id"] == 4].reset_index(drop=True)
    i = g.index[g["date"] == events.loc[4, "event_date"]][0]
    assert res.events.loc[4, "raw"] == pytest.approx(
        g["ret"].iloc[i - 2 : i + 3].sum(), rel=1e-13
    )
    np.testing.assert_array_equal(
        res.returns[4].to_numpy(), g["ret"].iloc[i - 2 : i + 3].to_numpy()
    )
