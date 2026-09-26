"""MCP data loader: streamed sampling for over-cap files (review F03).

Before: a file over ``STATSPAI_MCP_MAX_DATA_BYTES`` was rejected with advice
to pass ``data_sample_n`` — which could never help, because the cap check ran
first and sampling happened only after a full load.
"""

import numpy as np
import pandas as pd
import pytest

from statspai.agent import _data_loader as dl
from statspai.exceptions import MethodIncompatibility


@pytest.fixture
def frame():
    rng = np.random.default_rng(7)
    n = 2_503
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "g": rng.integers(0, 5, size=n),
        }
    )


def _write(frame, tmp_path, ext):
    path = tmp_path / f"d{ext}"
    if ext in (".csv", ".tsv"):
        frame.to_csv(path, index=False, sep="\t" if ext == ".tsv" else ",")
    elif ext == ".parquet":
        frame.to_parquet(path, index=False)
    elif ext == ".jsonl":
        frame.to_json(path, orient="records", lines=True)
    elif ext == ".dta":
        frame.to_stata(path, write_index=False)
    return path


@pytest.mark.parametrize("ext", [".csv", ".tsv", ".parquet", ".jsonl", ".dta"])
def test_streamed_sample_equals_in_memory_sample(frame, tmp_path, monkeypatch, ext):
    if ext == ".parquet":
        pytest.importorskip("pyarrow")
    path = str(_write(frame, tmp_path, ext))
    monkeypatch.setattr(dl, "_STREAM_CHUNK_ROWS", 97)  # many uneven chunks

    monkeypatch.setenv("STATSPAI_MCP_MAX_DATA_BYTES", "0")  # no cap -> in memory
    dl._load_local_cached.cache_clear()
    in_mem = dl.load_dataframe(path, sample_n=200)

    monkeypatch.setenv("STATSPAI_MCP_MAX_DATA_BYTES", "1")  # over cap -> streamed
    streamed = dl.load_dataframe(path, sample_n=200)

    assert len(streamed) == 200
    assert streamed["id"].tolist() == in_mem["id"].tolist()
    assert streamed["id"].is_monotonic_increasing  # file order kept
    np.testing.assert_allclose(streamed["y"].to_numpy(), in_mem["y"].to_numpy())


def test_sample_is_invariant_to_column_projection(frame, tmp_path, monkeypatch):
    path = str(_write(frame, tmp_path, ".csv"))
    monkeypatch.setenv("STATSPAI_MCP_MAX_DATA_BYTES", "1")
    full = dl.load_dataframe(path, sample_n=50)
    proj = dl.load_dataframe(path, columns=["id", "y"], sample_n=50)
    assert list(proj.columns) == ["id", "y"]
    assert proj["id"].tolist() == full["id"].tolist()


def test_sample_larger_than_file_returns_everything(frame, tmp_path, monkeypatch):
    path = str(_write(frame, tmp_path, ".csv"))
    monkeypatch.setenv("STATSPAI_MCP_MAX_DATA_BYTES", "1")
    out = dl.load_dataframe(path, sample_n=10**6)
    assert out["id"].tolist() == frame["id"].tolist()


def test_over_cap_without_sample_gives_advice_that_works(frame, tmp_path, monkeypatch):
    path = str(_write(frame, tmp_path, ".csv"))
    monkeypatch.setenv("STATSPAI_MCP_MAX_DATA_BYTES", "1")
    with pytest.raises(MethodIncompatibility, match="data_sample_n") as ei:
        dl.load_dataframe(path)
    # following the advice succeeds
    assert "streamed" in str(ei.value)
    assert len(dl.load_dataframe(path, sample_n=5)) == 5


def test_over_cap_non_streamable_format_does_not_suggest_sampling(
    frame, tmp_path, monkeypatch
):
    pytest.importorskip("openpyxl")
    path = tmp_path / "d.xlsx"
    frame.head(20).to_excel(path, index=False)
    monkeypatch.setenv("STATSPAI_MCP_MAX_DATA_BYTES", "1")
    with pytest.raises(MethodIncompatibility, match="cannot reduce peak memory"):
        dl.load_dataframe(str(path), sample_n=5)


def test_parquet_cap_uses_uncompressed_projected_size(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    n = 50_000
    df = pd.DataFrame({"zeros": np.zeros(n), "keep": np.arange(n, dtype=float)})
    path = tmp_path / "z.parquet"
    df.to_parquet(path, index=False)
    disk = path.stat().st_size
    est_all = dl.estimated_load_bytes(str(path))
    est_keep = dl.estimated_load_bytes(str(path), ["keep"])
    assert est_all > disk  # compression hid the in-memory size
    assert 0 < est_keep < est_all


def test_provenance_records_sampling_rule():
    prov = dl.data_provenance("/nonexistent/x.csv", sample_n=3)
    assert prov["sample_method"] == dl.SAMPLE_METHOD
    assert prov["sample_seed"] == 0
