"""The MCP data loader reads .dta the way ``sp.read_data`` does.

Regression: the loader called bare ``pd.read_stata``, which (a) dropped
variable labels and (b) turned every value-labelled column into a string
categorical, so ``foreign`` (0 "Domestic" / 1 "Foreign") reached the
estimators as text instead of the 0/1 codes Stata's own commands use.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent import _data_loader as dl
from statspai.agent._data_cache import describe_frame


@pytest.fixture
def auto_dta(tmp_path, monkeypatch):
    monkeypatch.delenv("STATSPAI_MCP_DATA_ROOTS", raising=False)
    rng = np.random.default_rng(0)
    n = 400
    foreign = rng.integers(0, 2, n).astype("int8")
    weight = rng.normal(3000, 500, n)
    df = pd.DataFrame(
        {
            "price": 6000 + 1500.0 * foreign + 2.0 * weight + rng.normal(0, 300, n),
            "weight": weight,
            "foreign": foreign,
        }
    )
    path = tmp_path / "auto.dta"
    sp.write_data(
        df,
        path,
        labels={"price": "Price", "foreign": "Car origin"},
        value_labels={"foreign": {0: "Domestic", 1: "Foreign"}},
        data_label="1978 automobile data",
    )
    return str(path), df


def _assert_labelled(got):
    assert pd.api.types.is_integer_dtype(got["foreign"])
    assert got.attrs["_labels"] == {"price": "Price", "foreign": "Car origin"}
    assert got.attrs["_value_labels"] == {"foreign": {0: "Domestic", 1: "Foreign"}}
    assert got.attrs["_data_label"] == "1978 automobile data"


def test_local_load_keeps_codes_and_labels(auto_dta):
    path, df = auto_dta
    got = dl.load_dataframe(path)
    _assert_labelled(got)
    assert got["foreign"].tolist() == df["foreign"].tolist()


def test_column_projection_restricts_labels(auto_dta):
    path, _ = auto_dta
    got = dl.load_dataframe(path, columns=["foreign", "weight"])
    assert list(got.columns) == ["foreign", "weight"]
    assert got.attrs["_labels"] == {"foreign": "Car origin"}
    assert got.attrs["_value_labels"] == {"foreign": {0: "Domestic", 1: "Foreign"}}


def test_streamed_sample_matches_whole_file_sample(auto_dta, monkeypatch):
    path, _ = auto_dta
    whole = dl.load_dataframe(path, sample_n=50)
    monkeypatch.setenv("STATSPAI_MCP_MAX_DATA_BYTES", "1")  # force streaming
    monkeypatch.setattr(dl, "_STREAM_CHUNK_ROWS", 64)
    streamed = dl.load_dataframe(path, sample_n=50)
    _assert_labelled(streamed)
    pd.testing.assert_frame_equal(streamed, whole, check_dtype=False)


def test_estimate_on_loaded_frame_uses_numeric_codes(auto_dta):
    """The loaded frame gives the same fit as the frame that was written."""
    path, df = auto_dta
    loaded = sp.regress("price ~ weight + foreign", data=dl.load_dataframe(path))
    direct = sp.regress("price ~ weight + foreign", data=df)
    # same numbers in, same numbers out; dta stores doubles exactly
    np.testing.assert_allclose(
        loaded.params["foreign"], direct.params["foreign"], rtol=1e-12
    )
    assert list(loaded.params.index) == list(direct.params.index)


def test_describe_frame_reports_labels(auto_dta):
    path, df = auto_dta
    desc = describe_frame(dl.load_dataframe(path))
    assert desc["variable_labels"] == {"price": "Price", "foreign": "Car origin"}
    assert desc["value_labels"] == {"foreign": {"0": "Domestic", "1": "Foreign"}}
    assert desc["data_label"] == "1978 automobile data"
    # a frame without labels describes exactly as before
    plain = describe_frame(df)
    assert not {"variable_labels", "value_labels", "data_label"} & set(plain)


# ---------------------------------------------------------------------------
# Display formats, bounded label payloads, labels through transforms
# ---------------------------------------------------------------------------


@pytest.fixture
def survey_dta(tmp_path, monkeypatch):
    """A file with dates at two frequencies, a money format and shared labels."""
    monkeypatch.delenv("STATSPAI_MCP_DATA_ROOTS", raising=False)
    n = 40
    df = pd.DataFrame(
        {
            "id": np.arange(n, dtype="int32"),
            "wave": pd.to_datetime("2020-01-01") + pd.to_timedelta(np.arange(n), "D"),
            "month": pd.date_range("2020-01-01", periods=n, freq="MS"),
            "income": np.linspace(1000.0, 9000.0, n),
            "q1": (np.arange(n) % 5 + 1).astype("int8"),
            "q2": (np.arange(n) % 5 + 1).astype("int8"),
            "q3": (np.arange(n) % 2).astype("int8"),
            "essay": ["x" * 3000] + ["short"] * (n - 1),
        }
    )
    likert = {1: "Never", 2: "Rarely", 3: "Sometimes", 4: "Often", 5: "Always"}
    path = tmp_path / "survey.dta"
    df.to_stata(
        path,
        write_index=False,
        version=118,
        convert_dates={"wave": "td", "month": "tm"},
        variable_labels={"q1": "How often do you exercise", "income": "Annual income"},
        value_labels={"q1": likert, "q2": likert, "q3": {0: "No", 1: "Yes"}},
        convert_strl=["essay"],
    )
    return str(path)


def test_informative_display_formats_are_kept(survey_dta):
    got = dl.load_dataframe(survey_dta)
    # %td and %tm both arrive as datetime64, so the format is the only place
    # the time unit survives; default formats (%9.0g, %9s) are not reported.
    assert got.attrs["_formats"] == {"wave": "%td", "month": "%tm"}
    assert sp.read_data(survey_dta).attrs["_formats"] == {"wave": "%td", "month": "%tm"}
    out = describe_frame(got)
    assert out["display_formats"] == {"wave": "%td", "month": "%tm"}
    projected = dl.load_dataframe(survey_dta, columns=["id", "month"])
    assert projected.attrs["_formats"] == {"month": "%tm"}


def test_projection_keeps_value_labels_and_formats(survey_dta):
    for cols in (["q2"], ["q3", "q2"], ["q2", "id"]):
        got = dl.load_dataframe(survey_dta, columns=cols)
        assert got.attrs["_value_labels"]["q2"][5] == "Always", cols
    from statspai.utils.io import stata_label_attrs

    attrs = stata_label_attrs(survey_dta, ["q2", "month"])
    assert attrs["_value_labels"]["q2"][1] == "Never"
    assert attrs["_formats"] == {"month": "%tm"}


class _FakeReader:
    """A pandas StataReader's metadata surface, for the layouts it can be in.

    pandas names every value-label set it writes after its variable, so a
    file where the two differ (Stata's ``label values region regionlbl``)
    cannot be produced from here; the layouts are reproduced directly.
    """

    data_label = ""

    def __init__(self, lbllist, fmtlist):
        self._varlist = ["id", "region", "joined"]
        self._lbllist = lbllist
        self._fmtlist = fmtlist

    def variable_labels(self):
        return {"id": "", "region": "Census region", "joined": ""}

    def value_labels(self):
        return {"regionlbl": {1: "North", 2: "South"}}


def test_label_set_named_differently_from_its_variable():
    from statspai.utils.io import _stata_reader_attrs

    # Before any rows are read: lists cover every variable, in file order.
    whole = _FakeReader(["", "regionlbl", ""], ["%12.0g", "%8.0g", "%td"])
    attrs = _stata_reader_attrs(whole, ["joined", "region"])
    assert attrs["_value_labels"] == {"region": {1: "North", 2: "South"}}
    assert attrs["_formats"] == {"joined": "%td"}

    # After read(columns=[...]): lists narrowed to the request, in its order.
    narrowed = _FakeReader(["", "regionlbl"], ["%td", "%8.0g"])
    attrs = _stata_reader_attrs(narrowed, ["joined", "region"])
    assert attrs["_value_labels"] == {"region": {1: "North", 2: "South"}}
    assert attrs["_formats"] == {"joined": "%td"}

    # Lists gone (a future pandas): no formats, and value labels only where
    # the set happens to be named after the variable — here it is not.
    bare = _FakeReader(None, None)
    attrs = _stata_reader_attrs(bare, ["joined", "region"])
    assert "_value_labels" not in attrs and "_formats" not in attrs
    assert attrs["_labels"] == {"region": "Census region"}


def test_default_formats_are_not_informative():
    from statspai.utils.io import _informative_formats

    kept = _informative_formats(
        {
            "a": "%9.0g",
            "b": "%8.0g",
            "c": "%12.0g",
            "d": "%10.0g",
            "e": "%18s",
            "f": "%-18s",
            "g": "%td",
            "h": "%12.2fc",
            "i": "%tdCCYY-NN-DD",
            "j": "",
            "k": None,
        }
    )
    assert kept == {"g": "%td", "h": "%12.2fc", "i": "%tdCCYY-NN-DD"}


def test_shared_value_label_is_spelled_out_once(survey_dta):
    out = describe_frame(dl.load_dataframe(survey_dta))
    vl = out["value_labels"]
    assert vl["q1"] == {
        "1": "Never",
        "2": "Rarely",
        "3": "Sometimes",
        "4": "Often",
        "5": "Always",
    }
    assert vl["q2"] == {"same_as": "q1"}
    assert vl["q3"] == {"0": "No", "1": "Yes"}
    assert "value_labels_truncated" not in out


def test_long_value_label_is_cut_and_counted():
    from statspai.agent._data_cache import VALUE_LABEL_MAX_ENTRIES

    df = pd.DataFrame({"county": [1, 2, 3]})
    df.attrs["_value_labels"] = {"county": {k: f"County {k}" for k in range(1, 3001)}}
    out = describe_frame(df)
    assert len(out["value_labels"]["county"]) == VALUE_LABEL_MAX_ENTRIES
    assert out["value_labels"]["county"]["1"] == "County 1"
    assert out["value_labels_truncated"] == {"county": 3000}


def test_stata_missing_value_codes_print_as_stata_writes_them():
    df = pd.DataFrame({"region": [1.0, 2.0, np.nan]})
    df.attrs["_value_labels"] = {
        "region": {
            1: "North",
            2147483621: "Sysmiss",
            2147483622: "Refused",
            2147483647: "Z",
        }
    }
    assert describe_frame(df)["value_labels"]["region"] == {
        "1": "North",
        ".": "Sysmiss",
        ".a": "Refused",
        ".z": "Z",
    }


def test_long_strings_in_head_are_clipped(survey_dta):
    out = describe_frame(dl.load_dataframe(survey_dta))
    essay = out["head"][0]["essay"]
    assert essay.startswith("x" * 200) and essay.endswith("… (+2800 chars)")
    assert out["head"][1]["essay"] == "short"


def test_frames_without_labels_describe_exactly_as_before():
    out = describe_frame(pd.DataFrame({"x": [1.0, 2.0], "s": ["a", "b"]}))
    assert set(out) == {
        "n_rows",
        "n_cols",
        "columns",
        "dtypes",
        "missing",
        "head",
        "numeric_summary",
    }


def test_labels_follow_columns_through_transform_steps(survey_dta):
    from statspai.agent.workflow_tools import _apply_transform, _carry_label_attrs

    df = dl.load_dataframe(survey_dta)

    def run(frame, step):
        out = _apply_transform(frame, step)
        out.attrs = _carry_label_attrs(frame.attrs, out, step)
        return out

    renamed = run(df, {"op": "rename", "mapping": {"q1": "exercise"}})
    assert renamed.attrs["_labels"]["exercise"] == "How often do you exercise"
    assert "q1" not in renamed.attrs["_labels"]
    assert renamed.attrs["_value_labels"]["exercise"][1] == "Never"

    selected = run(renamed, {"op": "select", "columns": ["id", "month", "q3"]})
    assert "_labels" not in selected.attrs  # none of the kept columns is labelled
    assert set(selected.attrs["_value_labels"]) == {"q3"}
    assert selected.attrs["_formats"] == {"month": "%tm"}

    # A column that assign overwrote no longer means what its labels said.
    recoded = run(df, {"op": "assign", "column": "q3", "expr": "q3 * 10"})
    assert "q3" not in recoded.attrs["_value_labels"]
    assert "q1" in recoded.attrs["_value_labels"]
    added = run(df, {"op": "assign", "column": "log_income", "expr": "log(income)"})
    assert added.attrs["_labels"]["income"] == "Annual income"
