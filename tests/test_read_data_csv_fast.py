"""``sp.read_data`` reads large CSV files with pyarrow, as pandas would.

The fast path must give pandas' frame: same columns in the same order,
same types, the same missing values, dates left as text. The one intended
difference is that decimals are rounded correctly: a column written with
``repr`` (shortest round-trip text) reads back as exactly the doubles that
were written, which pandas' default converter does not guarantee.
The size threshold is lowered so the files here stay small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.utils import _csv_fast

pytest.importorskip("pyarrow")


@pytest.fixture(autouse=True)
def _small_threshold(monkeypatch):
    monkeypatch.setattr(_csv_fast, "_MIN_BYTES", 0)


def _frame(n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    m = rng.normal(size=n)
    m[rng.random(n) < 0.2] = np.nan
    k = rng.integers(0, 50, n).astype(float)
    k[::7] = np.nan
    return pd.DataFrame(
        {
            "y": rng.normal(size=n) * 1e3,
            "id": rng.integers(0, 10**9, n),
            "m": m,
            "k": k,
            "s": rng.choice(["a", 'he said "x", y', "NA", "c d"], n),
            "date": rng.choice(["2020-01-01", "2021-06-30"], n),
            "stamp": rng.choice(["2020-01-01 10:00:00", "2021-06-30 23:59:59"], n),
            "flag": rng.choice([True, False], n),
            "empty": np.full(n, np.nan),
        }
    )


def test_fast_path_matches_pandas_and_rounds_exactly(tmp_path, monkeypatch):
    df = _frame()
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    calls = []
    real = _csv_fast.read_csv_fast
    monkeypatch.setattr(
        _csv_fast, "read_csv_fast", lambda *a: calls.append(1) or real(*a)
    )
    got = sp.read_data(str(path))
    assert calls and real(str(path), None, {}) is not None
    ref = pd.read_csv(path)
    assert list(got.columns) == list(ref.columns)
    assert got.dtypes.to_dict() == ref.dtypes.to_dict()
    for c in ["id", "s", "date", "stamp", "flag", "empty", "k"]:
        pd.testing.assert_series_equal(got[c], ref[c])
    # a missing text cell is NaN, as pandas has it (pyarrow gives None)
    assert not any(v is None for v in got["s"])
    # decimals: exactly the doubles written (repr round-trips)
    np.testing.assert_array_equal(got["y"].to_numpy(), df["y"].to_numpy())
    np.testing.assert_array_equal(got["m"].to_numpy(), df["m"].to_numpy())
    assert isinstance(got.index, pd.RangeIndex)


def test_usecols_keeps_the_file_order_and_tsv(tmp_path):
    df = _frame(500)
    path = tmp_path / "d.tsv"
    df.to_csv(path, index=False, sep="\t")
    got = sp.read_data(str(path), usecols=["k", "id"])
    assert list(got.columns) == ["id", "k"]
    pd.testing.assert_frame_equal(got, pd.read_csv(path, sep="\t", usecols=["k", "id"]))


@pytest.mark.parametrize(
    "text",
    [
        # a number column that turns into text after the first rows
        "a,b\n" + "1,2\n" * 3000 + "x,3\n",
        # numbers with a leading blank: pandas reads floats
        "a,b\n" + " 1.5,2\n" * 50,
        # duplicate column names: pandas renames them
        "a,a\n" + "1,2\n" * 50,
    ],
)
def test_falls_back_to_pandas_when_types_could_differ(tmp_path, text):
    path = tmp_path / "x.csv"
    path.write_text(text, encoding="utf-8")
    pd.testing.assert_frame_equal(sp.read_data(str(path)), pd.read_csv(path))


def test_other_keywords_go_to_pandas(tmp_path):
    path = tmp_path / "x.csv"
    _frame(200).to_csv(path, index=False)
    assert _csv_fast.read_csv_fast(str(path), None, {"nrows": 5}) is None
    assert _csv_fast.read_csv_fast(str(path), "latin-1", {}) is None
    got = sp.read_data(str(path), nrows=5)
    pd.testing.assert_frame_equal(got, pd.read_csv(path, nrows=5))
