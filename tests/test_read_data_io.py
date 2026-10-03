"""``sp.read_data`` on .dta files, with and without the optional pyreadstat.

Regression: the pyreadstat import sat outside the ``try`` block, so the
documented pandas fallback was unreachable and a core install raised
``ModuleNotFoundError`` on every .dta file.
"""

import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def labelled_dta(tmp_path):
    path = tmp_path / "lab.dta"
    df = pd.DataFrame(
        {"wage": [10.0, 12.0, np.nan], "female": np.array([0, 1, 1], dtype="int8")}
    )
    df.to_stata(
        path,
        write_index=False,
        variable_labels={"wage": "Monthly wage in CNY"},
        value_labels={"female": {0: "male", 1: "female"}},
    )
    return path


def test_dta_without_pyreadstat_keeps_labels(labelled_dta):
    with mock.patch.dict(sys.modules, {"pyreadstat": None}):
        df = sp.read_data(str(labelled_dta))
    assert list(df.columns) == ["wage", "female"]
    # numeric codes kept (same layout as the pyreadstat path), not categoricals
    assert pd.api.types.is_numeric_dtype(df["female"])
    assert df["female"].tolist() == [0, 1, 1]
    assert np.isnan(df["wage"].iloc[2])
    assert df.attrs["_labels"] == {"wage": "Monthly wage in CNY"}
    assert df.attrs["_value_labels"] == {"female": {0: "male", 1: "female"}}
    assert sp.get_label(df, "wage") == "Monthly wage in CNY"


def test_dta_with_pyreadstat_matches_fallback(labelled_dta):
    pytest.importorskip("pyreadstat")
    with mock.patch.dict(sys.modules, {"pyreadstat": None}):
        fallback = sp.read_data(str(labelled_dta))
    native = sp.read_data(str(labelled_dta))
    np.testing.assert_allclose(
        native.to_numpy(dtype=float), fallback.to_numpy(dtype=float)
    )
    assert native.attrs["_labels"] == fallback.attrs["_labels"]


# --- sp.write_data ---------------------------------------------------------


def _labelled_frame():
    df = pd.DataFrame(
        {
            "wage": [10.0, 12.0, np.nan],
            "female": np.array([0, 1, 1], dtype="int8"),
            "name": ["a", "b", "c"],
        }
    )
    sp.label_vars(df, {"wage": "月工资 (CNY)", "female": "Sex"})
    df.attrs["_value_labels"] = {"female": {0: "male", 1: "female"}}
    df.attrs["_data_label"] = "Survey wave 1"
    return df


def test_write_data_dta_round_trip_keeps_all_labels(tmp_path):
    df = _labelled_frame()
    out = sp.write_data(df, tmp_path / "out.dta")
    assert out == tmp_path / "out.dta"
    back = sp.read_data(str(out))
    assert list(back.columns) == ["wage", "female", "name"]  # no index column
    assert back["female"].tolist() == [0, 1, 1]
    assert np.isnan(back["wage"].iloc[2])
    # non-ASCII label survives: the default format is 118 (UTF-8)
    assert back.attrs["_labels"] == {"wage": "月工资 (CNY)", "female": "Sex"}
    assert back.attrs["_value_labels"] == {"female": {0: "male", 1: "female"}}
    assert back.attrs["_data_label"] == "Survey wave 1"


def test_write_data_matches_what_pandas_reads_back(tmp_path):
    """Independent check: read the file with pandas, not with sp.read_data."""
    out = sp.write_data(_labelled_frame(), tmp_path / "out.dta")
    with pd.read_stata(out, iterator=True) as reader:
        assert reader.variable_labels() == {
            "wage": "月工资 (CNY)",
            "female": "Sex",
            "name": "",
        }
        assert reader.data_label == "Survey wave 1"
    # pandas' default applies the value labels
    assert pd.read_stata(out)["female"].astype(str).tolist() == [
        "male",
        "female",
        "female",
    ]


def test_write_data_explicit_labels_override_attrs(tmp_path):
    df = _labelled_frame()
    out = sp.write_data(
        df,
        tmp_path / "out.dta",
        labels={"female": "Female (=1)"},
        value_labels={"female": {0: "no", 1: "yes"}},
        data_label="Override",
    )
    back = sp.read_data(str(out))
    assert back.attrs["_labels"] == {"wage": "月工资 (CNY)", "female": "Female (=1)"}
    assert back.attrs["_value_labels"] == {"female": {0: "no", 1: "yes"}}
    assert back.attrs["_data_label"] == "Override"
    # the caller's frame is not modified
    assert df.attrs["_labels"]["female"] == "Sex"


def test_write_data_skips_labels_of_dropped_columns(tmp_path):
    df = _labelled_frame()[["wage", "name"]]  # attrs still mention 'female'
    assert "female" in df.attrs["_value_labels"]
    back = sp.read_data(str(sp.write_data(df, tmp_path / "out.dta")))
    assert back.attrs["_labels"] == {"wage": "月工资 (CNY)"}
    assert "_value_labels" not in back.attrs


def test_write_data_rejects_what_stata_cannot_store(tmp_path):
    df = _labelled_frame()
    with pytest.raises(ValueError, match=r"80.*\['wage'\]"):
        sp.write_data(df, tmp_path / "a.dta", labels={"wage": "x" * 81})
    with pytest.raises(ValueError, match="not in the data"):
        sp.write_data(df, tmp_path / "a.dta", labels={"nope": "x"})
    with pytest.raises(ValueError, match="numeric"):
        sp.write_data(df, tmp_path / "a.dta", value_labels={"name": {1: "x"}})
    with pytest.raises(ValueError, match="integer codes"):
        sp.write_data(df, tmp_path / "a.dta", value_labels={"wage": {1.5: "x"}})
    with pytest.raises(ValueError, match="Unsupported file format"):
        sp.write_data(df, tmp_path / "a.sav")
    assert not (tmp_path / "a.dta").exists()


def test_write_data_csv_warns_that_labels_are_dropped(tmp_path):
    df = _labelled_frame()
    with pytest.warns(UserWarning, match="cannot store variable or value labels"):
        out = sp.write_data(df, tmp_path / "out.csv")
    assert list(pd.read_csv(out).columns) == ["wage", "female", "name"]
    # no labels, no warning
    plain = pd.DataFrame({"x": [1, 2]})
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sp.write_data(plain, tmp_path / "plain.csv")


def test_write_data_parquet_round_trip_restores_integer_codes(tmp_path):
    pytest.importorskip("pyarrow")
    if tuple(int(x) for x in pd.__version__.split(".")[:2]) < (2, 1):
        pytest.skip("pandas < 2.1 does not persist DataFrame.attrs in Parquet")
    out = sp.write_data(_labelled_frame(), tmp_path / "out.parquet")
    back = sp.read_data(str(out))
    assert back.attrs["_labels"] == {"wage": "月工资 (CNY)", "female": "Sex"}
    # JSON turned the codes into strings; read_data turns them back
    assert back.attrs["_value_labels"] == {"female": {0: "male", 1: "female"}}


def test_read_data_columns_subset_keeps_labels_aligned(tmp_path):
    out = sp.write_data(_labelled_frame(), tmp_path / "out.dta")
    with mock.patch.dict(sys.modules, {"pyreadstat": None}):
        back = sp.read_data(str(out), columns=["female"])
    assert list(back.columns) == ["female"]
    assert back.attrs["_labels"] == {"female": "Sex"}
    assert back.attrs["_value_labels"] == {"female": {0: "male", 1: "female"}}


# --- Stata dates: the same frame with and without pyreadstat --------------

_DATE_UNITS = {
    "d": "td",
    "c": "tc",
    "w": "tw",
    "m": "tm",
    "q": "tq",
    "h": "th",
    "y": "ty",
}


@pytest.fixture
def dated_dta(tmp_path):
    """One column per Stata date unit, with a missing value in each."""
    stamps = pd.to_datetime(
        ["1958-11-03 00:00", "1960-01-01 00:00", "2019-12-30 10:30", "2024-07-15 00:00"]
    )
    df = pd.DataFrame({name: stamps for name in _DATE_UNITS})
    df.loc[1, list(_DATE_UNITS)] = pd.NaT
    df["x"] = [1.0, 2.0, 3.0, 4.0]
    path = tmp_path / "dates.dta"
    # a copy: pandas rewrites the dict it is given ("td" becomes "%td")
    df.to_stata(path, write_index=False, convert_dates=dict(_DATE_UNITS), version=118)
    return path


def test_period_date_conversion_matches_pandas(dated_dta):
    # The helper the pyreadstat path uses, checked against what pandas
    # itself returns for the same file; runs without pyreadstat installed.
    from statspai.utils.io import _convert_stata_period_dates

    with mock.patch.dict(sys.modules, {"pyreadstat": None}):
        expected = sp.read_data(str(dated_dta))
    counts = pd.read_stata(dated_dta, convert_dates=False)
    formats = {name: f"%{unit}" for name, unit in _DATE_UNITS.items()}
    _convert_stata_period_dates(counts, formats)
    for name in ("w", "m", "q", "h", "y"):
        assert pd.api.types.is_datetime64_any_dtype(counts[name]), name
        pd.testing.assert_series_equal(
            counts[name].astype("datetime64[ns]"),
            expected[name].astype("datetime64[ns]"),
        )
    # %td / %tc are pyreadstat's job, and a plain number is left alone
    assert pd.api.types.is_numeric_dtype(counts["d"])
    assert counts["x"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_period_date_out_of_range_warns_and_keeps_counts():
    from statspai.utils.io import _convert_stata_period_dates

    # Year 0 is not a date in any datetime64 resolution. (Year 1500 is out
    # of range for pandas 2's nanoseconds but converts under pandas 3.)
    df = pd.DataFrame({"y": [0.0, 2020.0]})
    with pytest.warns(UserWarning, match="could not be converted"):
        _convert_stata_period_dates(df, {"y": "%ty"})
    assert df["y"].tolist() == [0.0, 2020.0]


def test_dta_dates_with_pyreadstat_match_fallback(dated_dta):
    pytest.importorskip("pyreadstat")
    with mock.patch.dict(sys.modules, {"pyreadstat": None}):
        fallback = sp.read_data(str(dated_dta))
    native = sp.read_data(str(dated_dta))
    for name in _DATE_UNITS:
        assert pd.api.types.is_datetime64_any_dtype(native[name]), name
        pd.testing.assert_series_equal(
            native[name].astype("datetime64[ns]"),
            fallback[name].astype("datetime64[ns]"),
        )
    assert native.attrs["_formats"] == fallback.attrs["_formats"]


@pytest.mark.parametrize("with_pyreadstat", [False, True])
def test_write_data_round_trip_keeps_the_date_unit(
    dated_dta, tmp_path, with_pyreadstat
):
    if with_pyreadstat:
        pytest.importorskip("pyreadstat")
        patch = {}
    else:
        patch = {"pyreadstat": None}
    with mock.patch.dict(sys.modules, patch):
        first = sp.read_data(str(dated_dta))
        out = sp.write_data(first, tmp_path / "again.dta")
        second = sp.read_data(str(out))
    assert second.attrs["_formats"] == {n: f"%{u}" for n, u in _DATE_UNITS.items()}
    for name in _DATE_UNITS:
        pd.testing.assert_series_equal(
            second[name].astype("datetime64[ns]"), first[name].astype("datetime64[ns]")
        )


def test_write_data_explicit_convert_dates_wins(dated_dta, tmp_path):
    with mock.patch.dict(sys.modules, {"pyreadstat": None}):
        first = sp.read_data(str(dated_dta))
        out = sp.write_data(first, tmp_path / "again.dta", convert_dates={"d": "tm"})
        second = sp.read_data(str(out))
    assert second.attrs["_formats"]["d"] == "%tm"


# --- numeric widths: Stata's storage type is not numpy's arithmetic type ---


@pytest.fixture
def typed_dta(tmp_path):
    """One column per Stata numeric storage type."""
    df = pd.DataFrame(
        {
            "b": np.array([1, 50, 100], dtype="int8"),
            "i": np.array([1, 500, 32000], dtype="int16"),
            "l": np.array([1, 70000, 2_000_000_000], dtype="int32"),
            "f": np.array([0.1, 1.5, 2.25], dtype="float32"),
            "d": np.array([0.1, 1.5, 2.25], dtype="float64"),
        }
    )
    path = tmp_path / "typed.dta"
    df.to_stata(path, write_index=False, version=118)
    return path


def test_dta_numerics_are_widened_without_pyreadstat(typed_dta):
    with mock.patch.dict(sys.modules, {"pyreadstat": None}):
        df = sp.read_data(str(typed_dta))
    assert [str(df[c].dtype) for c in "bilfd"] == ["int64"] * 3 + ["float64"] * 2
    # the point of widening: a byte squared no longer wraps past 127
    assert (df["b"] ** 2).tolist() == [1, 2500, 10000]
    # a Stata float keeps its float32 value exactly
    assert (
        df["f"].tolist() == np.array([0.1, 1.5, 2.25], "float32").astype(float).tolist()
    )


def test_dta_numeric_dtypes_with_pyreadstat_match_fallback(typed_dta):
    pytest.importorskip("pyreadstat")
    with mock.patch.dict(sys.modules, {"pyreadstat": None}):
        fallback = sp.read_data(str(typed_dta))
    native = sp.read_data(str(typed_dta))
    pd.testing.assert_frame_equal(native, fallback)


def test_write_data_round_trip_keeps_storage_types(typed_dta, tmp_path):
    with mock.patch.dict(sys.modules, {"pyreadstat": None}):
        df = sp.read_data(str(typed_dta))
    before = df.copy()
    out = sp.write_data(df, tmp_path / "again.dta")
    pd.testing.assert_frame_equal(df, before)  # the caller's frame is untouched
    stored = pd.read_stata(out)
    assert [str(stored[c].dtype) for c in "bilfd"] == [
        "int8",
        "int16",
        "int32",
        "float32",
        "float64",
    ]
    pd.testing.assert_frame_equal(stored, pd.read_stata(typed_dta))
    assert out.stat().st_size == typed_dta.stat().st_size


def test_write_data_compress_never_rounds(tmp_path):
    df = pd.DataFrame(
        {
            "over_byte": np.array([1, 101], dtype="int64"),  # 101 is .a in a byte
            "big": np.array([1, 2**40], dtype="int64"),
            "frac": np.array([0.1, np.nan]),  # 0.1 is not a float32
            "half": np.array([0.5, np.nan]),  # 0.5 is
        }
    )
    stored = pd.read_stata(sp.write_data(df, tmp_path / "c.dta"))
    assert str(stored["over_byte"].dtype) == "int16"
    assert stored["over_byte"].tolist() == [1, 101]
    assert stored["big"].tolist() == [1, 2**40]
    assert str(stored["frac"].dtype) == "float64" and stored["frac"][0] == 0.1
    assert str(stored["half"].dtype) == "float32" and np.isnan(stored["half"][1])
