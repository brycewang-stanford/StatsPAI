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
