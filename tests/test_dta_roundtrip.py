"""What a .dta file stores beyond rows and plain labels survives a round trip.

The fixtures are written by Stata itself
(``tests/fixtures/dta_labels/make_roundtrip_fixtures.do``): a value-label
set shared by two variables under its own name, notes, characteristics, an
``xtset`` declaration, display formats, extended missing values in every
numeric storage type, and a format-120 file with an alias variable.

The files this module writes were opened in Stata 18 when the writer was
built: ``dtaverify`` found them valid in formats 117, 118 and 119, and
``cf _all using <fixture>, all`` matched every variable, extended missing
values included.  The tests here hold the same facts from the Python side.
"""

import sys
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.utils import _dta_layout

FIXTURES = Path(__file__).parent / "fixtures" / "dta_labels"
RELEASES = {"meta118": 118, "meta117": 117, "meta115": 115}


def _read(path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return sp.read_data(str(path), **kwargs)


@pytest.fixture(params=sorted(RELEASES))
def fixture(request):
    return FIXTURES / f"{request.param}.dta"


@pytest.fixture(params=["pandas", "pyreadstat"], autouse=True)
def reader(request):
    """Every test runs on both readers ``read_data`` may use."""
    if request.param == "pyreadstat":
        pytest.importorskip("pyreadstat")
        yield request.param
    else:
        with mock.patch.dict(sys.modules, {"pyreadstat": None}):
            yield request.param


def test_read_keeps_set_names_notes_and_characteristics(fixture):
    df = _read(fixture)
    assert _dta_layout.dta_release(fixture) == RELEASES[fixture.stem]
    assert df.attrs["_value_label_names"] == {"q1": "yesno", "q2": "yesno"}
    assert df.attrs["_notes"] == {
        "_dta": ["first dataset note", "second dataset note"],
        "wage": ["top-coded"],
    }
    chars = df.attrs["_characteristics"]
    assert chars["score"] == {"unit": "points"}
    assert chars["_dta"]["source"] == "make_roundtrip_fixtures.do"
    # the xtset declaration lives in the dataset's characteristics
    assert chars["_dta"]["_TSpanel"] == "id" and chars["_dta"]["_TStvar"] == "year"
    assert df.attrs["_formats"] == {"wage": "%12.2fc", "day": "%tdCCYY-NN-DD"}


def test_columns_subset_keeps_only_their_metadata(reader):
    subset = {"columns" if reader == "pandas" else "usecols": ["id", "q1"]}
    df = _read(FIXTURES / "meta118.dta", **subset)
    assert df.attrs["_value_label_names"] == {"q1": "yesno"}
    assert df.attrs["_notes"] == {"_dta": ["first dataset note", "second dataset note"]}
    assert set(df.attrs["_characteristics"]) == {"_dta"}


def test_missing_codes_and_dates_read_together(fixture):
    """A date column with a missing value used to break extended_missing='column'."""
    plain = _read(fixture)
    coded = _read(fixture, extended_missing="column")
    codes = [c for c in coded.columns if c.endswith("__miss")]
    assert codes == [
        f"{v}__miss" for v in ("q1", "score", "big", "ratio", "wage", "day")
    ]
    pd.testing.assert_frame_equal(coded.drop(columns=codes), plain)
    assert coded["q1__miss"].dropna().tolist() == [".a", ".z"]
    assert coded["day__miss"].dropna().tolist() == [".a"]


@pytest.mark.parametrize("version", [114, 117, 118, 119])
def test_round_trip_is_lossless(fixture, tmp_path, version):
    df = _read(fixture, extended_missing="column")
    out = sp.write_data(df, tmp_path / "back.dta", version=version)
    back = _read(out, extended_missing="column")
    pd.testing.assert_frame_equal(back, df)
    assert back.attrs == df.attrs

    found = _dta_layout.read_layout(out)
    # one set in the file, shared, under the name Stata gave it
    assert [r.name for r in found.records] == ["yesno"]
    assert found.records[0].table == {0: "No", 1: "Yes", ".a": "Refused"}
    assert dict(zip(found.names, found.set_names))["q2"] == "yesno"
    assert not [n for n in found.names if n.endswith("__miss")]


def test_written_cells_hold_statas_missing_codes(tmp_path):
    """Byte, int, long, float and double each store .a ... .z their own way."""
    codes = [None, ".a", ".z", None]
    df = pd.DataFrame({"x": [1.0, np.nan, np.nan, np.nan], "x__miss": codes})
    expected = {
        # the value of '.' for the type, per `help dta`; .a is one step above
        "byte": (b"\x66", b"\x7f", b"\x65"),
        "float": (
            (0x7F000800).to_bytes(4, "little"),
            (0x7F00D000).to_bytes(4, "little"),
            (0x7F000000).to_bytes(4, "little"),
        ),
        "double": (
            (0x7FE0010000000000).to_bytes(8, "little"),
            (0x7FE01A0000000000).to_bytes(8, "little"),
            (0x7FE0000000000000).to_bytes(8, "little"),
        ),
    }
    for kind, value in (("float", 1.0), ("double", 0.1)):
        df.loc[0, "x"] = value
        out = sp.write_data(df, tmp_path / f"{kind}.dta")
        found = _dta_layout.read_layout(out)
        assert found.types == [kind]
        raw = out.read_bytes()[found.data_offset :]
        width = found.row_width
        cells = [raw[i * width : (i + 1) * width] for i in range(1, 4)]
        assert tuple(cells) == expected[kind]
    # the integer types are reached by patching a file pandas wrote as such
    ints = pd.DataFrame({"x": np.array([1, 2, 3, 4], dtype="int8")})
    ints.to_stata(tmp_path / "byte.dta", write_index=False, version=118)
    _dta_layout.patch_dta(
        tmp_path / "byte.dta",
        missing_codes={"x": (np.array([1, 2]), np.array([1, 26]))},
    )
    found = _dta_layout.read_layout(tmp_path / "byte.dta")
    raw = (tmp_path / "byte.dta").read_bytes()[found.data_offset :]
    assert (raw[1:2], raw[2:3]) == expected["byte"][:2]
    back = _read(tmp_path / "byte.dta", extended_missing="column")
    assert back["x__miss"].fillna("").tolist() == ["", ".a", ".z", ""]


def test_extended_missing_nan_writes_the_code_column_as_text(tmp_path):
    df = pd.DataFrame({"x": [1.0, np.nan], "x__miss": [None, ".a"]})
    out = sp.write_data(df, tmp_path / "a.dta", extended_missing="nan")
    assert _dta_layout.read_layout(out).names == ["x", "x__miss"]
    with pytest.raises(ValueError, match="extended_missing must be one of"):
        sp.write_data(df, tmp_path / "a.dta", extended_missing="fold")


def test_a_code_on_a_row_that_has_a_value_is_refused(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0], "x__miss": [None, ".a"]})
    with pytest.raises(ValueError, match="marks 1 rows.*'x' is not missing"):
        sp.write_data(df, tmp_path / "a.dta")
    assert not list(tmp_path.iterdir())


def test_a_column_that_only_looks_like_codes_is_an_ordinary_column(tmp_path):
    df = pd.DataFrame({"x": [1.0, np.nan], "x__miss": ["why", "because"]})
    out = sp.write_data(df, tmp_path / "a.dta")
    assert _read(out)["x__miss"].tolist() == ["why", "because"]


def test_changed_labels_leave_a_shared_set(tmp_path):
    df = _read(FIXTURES / "meta118.dta")
    sp.label_values(df, "q2", {0: "Never", 1: "Yes"})
    out = sp.write_data(df, tmp_path / "a.dta")
    found = _dta_layout.read_layout(out)
    assert {r.name: r.table for r in found.records} == {
        "yesno": {0: "No", 1: "Yes", ".a": "Refused"},
        "q2": {0: "Never", 1: "Yes"},
    }
    assert _read(out).attrs["_value_label_names"] == {"q1": "yesno", "q2": "q2"}

    sp.label_values(df, "q2", None)
    assert df.attrs["_value_label_names"] == {"q1": "yesno"}


def test_dropped_columns_take_their_notes_along(tmp_path):
    df = _read(FIXTURES / "meta118.dta").drop(columns=["wage", "score"])
    back = _read(sp.write_data(df, tmp_path / "a.dta"))
    assert back.attrs["_notes"] == {
        "_dta": ["first dataset note", "second dataset note"]
    }
    assert set(back.attrs["_characteristics"]) == {"_dta"}


def test_a_format_stata_would_refuse_is_left_out_with_a_warning(tmp_path):
    df = pd.DataFrame({"x": [1.5, 2.5], "s": ["a", "b"]})
    df.attrs["_formats"] = {"x": "%9s", "s": "%-12s"}
    with pytest.warns(UserWarning, match="x: %9s"):
        out = sp.write_data(df, tmp_path / "a.dta")
    found = _dta_layout.read_layout(out)
    assert dict(zip(found.names, found.formats)) == {"x": "%9.0g", "s": "%-12s"}


def test_text_a_latin1_file_cannot_hold_fails_before_anything_is_written(tmp_path):
    df = pd.DataFrame({"x": [1.0]})
    df.attrs["_notes"] = {"_dta": ["数据说明"]}
    target = tmp_path / "a.dta"
    target.write_bytes(b"what was there")
    with pytest.raises(ValueError, match="format-114 file cannot"):
        sp.write_data(df, target, version=114)
    assert target.read_bytes() == b"what was there"
    assert [p.name for p in tmp_path.iterdir()] == ["a.dta"]


def test_format_120_is_read_without_its_alias_variable():
    with pytest.warns(UserWarning, match=r"format-120.*alias variables \(val\)"):
        df = sp.read_data(str(FIXTURES / "alias120.dta"))
    assert list(df.columns) == ["id", "own", "other"]
    assert df["own"].tolist() == [7, 7, 7] and df["id"].tolist() == [1, 2, 3]
    assert df.attrs["_alias_variables"] == ["val"]
    assert df.attrs["_value_labels"] == {"own": {7: "Seven"}}
    assert df.attrs["_value_label_names"] == {"own": "seven"}
    assert df.attrs["_notes"] == {"own": ["kept beside an alias"]}


def test_format_120_copy_is_a_valid_118_file(tmp_path):
    converted, alias = _dta_layout.without_alias_variables(FIXTURES / "alias120.dta")
    try:
        found = _dta_layout.read_layout(converted)
        assert found.release == 118 and alias == ["val"]
        assert found.names == ["id", "own", "other"]
        # every section is where the map says, and the file ends where it says
        assert found.map[13] == Path(converted).stat().st_size
        assert pd.read_stata(converted)["other"].tolist() == [1, 2, 3]
    finally:
        Path(converted).unlink()
    assert _dta_layout.without_alias_variables(FIXTURES / "meta118.dta") == (None, [])


def test_stata_session_knows_the_sets_a_file_came_with():
    df = _read(FIXTURES / "meta118.dta")
    from statspai.agent._translation._stata_datastep import DataSteps

    steps = DataSteps(df)
    steps.apply_label('label define yesno 2 "Maybe", add')
    steps.apply_label("label values score yesno")
    out = steps.data
    assert out.attrs["_value_labels"]["q1"] == {0: "No", 1: "Yes", 2: "Maybe"}
    assert out.attrs["_value_labels"]["q2"] == {0: "No", 1: "Yes", 2: "Maybe"}
    assert out.attrs["_value_label_names"]["score"] == "yesno"
