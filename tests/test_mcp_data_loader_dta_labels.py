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
