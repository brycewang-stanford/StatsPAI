"""``sp.*`` accepts polars, Arrow and interchange-protocol frames.

The shim (``statspai/_frame_input.py``) is left out of the parity and perf
provenance traces on the strength of the first test: pandas arguments
reach the function as the very same objects.
"""

from __future__ import annotations

import inspect
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai import _frame_input as fi

pa = pytest.importorskip("pyarrow")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    n = 300
    d = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "g": rng.integers(0, 3, n),
            "w": rng.uniform(0.5, 2, n),
        }
    )
    d["y"] = 1 + 2 * d.x + 0.5 * d.g + rng.normal(size=n)
    d["b"] = (d.y > 1.5).astype(int)
    return d


def test_pandas_and_everything_else_pass_through_by_identity(data):
    seen = {}

    def probe(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs

    wrapped = fi.wrap(probe)
    arr = np.arange(3.0)
    obj = object()
    series = data["x"]
    wrapped(data, arr, obj, series, "y ~ x", 1.5, None, [data], k=data, s=series)
    assert all(a is b for a, b in zip(seen["args"], (data, arr, obj, series)))
    assert seen["args"][7][0] is data
    assert seen["kwargs"]["k"] is data and seen["kwargs"]["s"] is series


def test_arrow_objects_become_pandas(data):
    table = pa.Table.from_pandas(data, preserve_index=False)
    pd.testing.assert_frame_equal(fi.convert(table), data)
    batch = table.to_batches()[0]
    pd.testing.assert_frame_equal(fi.convert(batch), data)
    col = fi.convert(table.column("x"))
    np.testing.assert_array_equal(col.to_numpy(), data.x.to_numpy())


def test_any_arrow_stream_or_interchange_object(data):
    table = pa.Table.from_pandas(data, preserve_index=False)

    class ArrowStream:
        def __arrow_c_stream__(self, requested_schema=None):
            return table.__arrow_c_stream__(requested_schema)

    class Interchange:
        def __dataframe__(self, nan_as_null=False, allow_copy=True):
            return data.__dataframe__(nan_as_null, allow_copy)

    pd.testing.assert_frame_equal(fi.convert(ArrowStream()), data)
    pd.testing.assert_frame_equal(fi.convert(Interchange()), data)


def test_polars_frames_become_pandas(data):
    pl = pytest.importorskip("polars")
    frame = pl.from_pandas(data)
    pd.testing.assert_frame_equal(fi.convert(frame), data, check_dtype=False)
    pd.testing.assert_frame_equal(fi.convert(frame.lazy()), data, check_dtype=False)
    np.testing.assert_array_equal(fi.convert(frame["x"]).to_numpy(), data.x)


@pytest.mark.parametrize("kind", ["polars", "polars_lazy", "arrow"])
def test_estimators_give_the_pandas_numbers(data, kind):
    if kind.startswith("polars"):
        pl = pytest.importorskip("polars")
        other = pl.from_pandas(data)
        if kind == "polars_lazy":
            other = other.lazy()
    else:
        other = pa.Table.from_pandas(data, preserve_index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.regress("y ~ x + C(g)", data=data, weights="w")
        b = sp.regress("y ~ x + C(g)", data=other, weights="w")
        pd.testing.assert_series_equal(a.params, b.params)
        pd.testing.assert_series_equal(a.std_errors, b.std_errors)
        la = sp.logit("b ~ x", data=data)
        lb = sp.logit("b ~ x", data=other)
        # newdata= is converted too
        me_a = sp.avg_slopes(la, "x", newdata=data)
        me_b = sp.avg_slopes(lb, "x", newdata=other)
    np.testing.assert_allclose(me_a.estimate, me_b.estimate, rtol=1e-12)
    np.testing.assert_allclose(me_a.std_error, me_b.std_error, rtol=1e-12)


def test_staggered_did_on_a_polars_panel():
    pl = pytest.importorskip("polars")
    df = sp.datasets.mpdta()
    kw = dict(y="lemp", g="first_treat", t="year", i="countyreal")
    a = sp.callaway_santanna(df, **kw)
    b = sp.callaway_santanna(pl.from_pandas(df), **kw)
    assert a.estimate == pytest.approx(b.estimate, rel=1e-12)
    assert a.se == pytest.approx(b.se, rel=1e-12)


def test_wrapping_keeps_signature_pickling_and_categories():
    assert getattr(sp.regress, fi.ATTR) is sp.regress.__wrapped__
    assert inspect.signature(sp.regress) == inspect.signature(sp.regress.__wrapped__)
    # pickle finds the function by module.qualname: it must be this object.
    assert pickle.loads(pickle.dumps(sp.regress)) is sp.regress
    from statspai.regression import ols

    assert ols.regress is sp.regress
    assert sp.regress.__module__ == "statspai.regression.ols"
    assert sp.describe_function("regress")["category"] == "regression"
    # never wrapped twice
    assert fi.wrap(sp.regress) is sp.regress


def test_every_public_function_is_wrapped():
    unwrapped = [
        name
        for name in sp.__all__
        if inspect.isfunction(sp.__dict__.get(name))
        and getattr(sp.__dict__[name], fi.ATTR, None) is None
    ]
    assert unwrapped == []


def test_lazy_exports_are_wrapped_when_resolved():
    from statspai import _LAZY_ATTRS

    name = next(
        n
        for n, (mod, _) in _LAZY_ATTRS.items()
        if mod.startswith(("mht", "survey", "bounds"))
        and inspect.isfunction(getattr(sp, n))
    )
    assert getattr(getattr(sp, name), fi.ATTR, None) is not None


def test_pyarrow_missing_message_for_a_foreign_stream(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_pyarrow(name, *a, **k):
        if name == "pyarrow":
            raise ImportError("no pyarrow")
        return real_import(name, *a, **k)

    class ArrowStream:
        def __arrow_c_stream__(self, requested_schema=None):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(builtins, "__import__", no_pyarrow)
    with pytest.raises(ImportError, match="pip install pyarrow"):
        fi.convert(ArrowStream())


def test_warnings_still_point_at_the_caller():
    # The wrapper adds a frame; its body is compiled under an importlib
    # bootstrap file name, which CPython's warnings skip when counting
    # stacklevel. Without that, every warning from a wrapped sp.* function
    # would be reported from statspai/_frame_input.py.
    def noisy(x):
        warnings.warn("from the estimator", UserWarning, stacklevel=2)
        return x

    wrapped = fi.wrap(noisy)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        wrapped(1)
    assert rec[0].filename == __file__


# --- reads_part: a polars frame is cut in polars before it is converted -----


def _event_frames():
    rng = np.random.default_rng(8)
    days = pd.bdate_range("2019-01-01", periods=320)
    mkt = rng.normal(0, 0.01, 320)
    rows = [
        pd.DataFrame(
            {
                "id": j,
                "date": days,
                "ret": 0.8 * mkt + rng.normal(0, 0.012, 320),
                "mkt": mkt,
                "other": rng.normal(size=320),
            }
        )
        for j in range(30)
    ]
    events = pd.DataFrame({"id": range(6), "event_date": days[290:296]})
    return pd.concat(rows, ignore_index=True), events


def test_reads_part_gives_the_pandas_result():
    pl = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    data, events = _event_frames()
    want = sp.abnormal_returns(data, events, market="mkt")
    for frame, ev in (
        (pl.from_pandas(data), pl.from_pandas(events)),
        (pl.from_pandas(data).lazy(), pl.from_pandas(events).lazy()),
        (pl.from_pandas(data), events),
    ):
        got = sp.abnormal_returns(frame, ev, market="mkt")
        pd.testing.assert_frame_equal(
            got.events.drop(columns=["event_date", "event_day"]),
            want.events.drop(columns=["event_date", "event_day"]),
            check_exact=True,
        )
        pd.testing.assert_frame_equal(got.tests, want.tests, check_exact=True)


def test_reads_part_hands_over_only_the_part(monkeypatch):
    pl = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    from statspai import _frame_input

    data, events = _event_frames()
    seen = []
    original = _frame_input._polars

    def spy(value):
        seen.append((value.height, list(value.columns)))
        return original(value)

    monkeypatch.setattr(_frame_input, "_polars", spy)
    sp.abnormal_returns(pl.from_pandas(data), pl.from_pandas(events), market="mkt")
    # six of thirty securities, four of five columns; then the event list
    assert seen[0] == (6 * 320, ["id", "date", "ret", "mkt"])


def test_reads_part_leaves_pandas_and_bad_calls_alone():
    pl = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    from statspai import _frame_input
    from statspai.exceptions import ColumnNotFound

    data, events = _event_frames()
    fn = _frame_input.bare(sp.abnormal_returns)
    # pandas: no cut, decided on the type alone
    assert _frame_input._cut(fn, lambda a: 1 / 0, (data, events), {}) is None
    # a column that does not exist is reported by the function itself
    with pytest.raises(ColumnNotFound):
        sp.abnormal_returns(pl.from_pandas(data), events, market="nope")
    # event ids of another type: the filter is skipped, the function then
    # reports that it found no returns for those securities
    text = events.assign(id=events["id"].astype(str))
    with pytest.warns(UserWarning, match="left out"):
        with pytest.raises(Exception, match="usable event"):
            sp.abnormal_returns(
                pl.from_pandas(data), pl.from_pandas(text), market="mkt"
            )
