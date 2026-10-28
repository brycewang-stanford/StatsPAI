"""sp.datasets.china_shock: the Autor-Dorn-Hanson (2013) workfile.

Offline tests point the loader at a small archive built in memory; the
check against the real file skips unless the Agarwal-Singh replication zip
is cached or ``STATSPAI_FETCH_DATASETS`` allows the download.
"""

import importlib.util
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.datasets import _china_shock, _fetch
from statspai.exceptions import DataInsufficient

SPEC = _china_shock.AGARWAL_SINGH_PACKAGE


def _archive(tmp_path):
    """A zip laid out like the replication package, with a two-row .dta."""
    df = pd.DataFrame(
        {
            "czone": [100.0, 100.0],
            "yr": np.array([1990, 2000], dtype=np.int16),
            "t2": np.array([0, 1], dtype=np.int8),
            "d_tradeusch_pw": np.array([0.1, 1.7], dtype=np.float32),
        }
    )
    dta = io.BytesIO()
    df.to_stata(dta, write_index=False)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(_china_shock.DTA_MEMBER, dta.getvalue())
    path = tmp_path / "pkg.zip"
    path.write_bytes(buf.getvalue())
    return path


def test_loader_reads_the_member_and_widens_float32(tmp_path, monkeypatch):
    path = _archive(tmp_path)
    monkeypatch.setattr(_china_shock, "fetch", lambda spec, download=True: path)
    df = sp.datasets.china_shock()
    assert list(df.columns) == ["czone", "yr", "t2", "d_tradeusch_pw"]
    assert df["d_tradeusch_pw"].dtype == np.float64
    # widening keeps the stored single-precision value, not the decimal
    assert df["d_tradeusch_pw"].iloc[0] == float(np.float32(0.1))
    assert df["yr"].dtype == np.int16
    assert df.attrs["simulated"] is False
    assert df.attrs["sha256"] == SPEC.sha256
    assert "CC-BY-4.0" in df.attrs["licence"]


def test_listed_without_downloading(tmp_path, monkeypatch):
    monkeypatch.setenv(_fetch.ENV_DATA_DIR, str(tmp_path))
    table = sp.list_datasets()
    row = table.set_index("name").loc["china_shock"]
    assert row["delivery"] == "fetched"
    assert not bool(row["simulated"])
    assert row["n_obs"] == 1444
    assert not row["available_offline"]
    assert list(row["columns"]) == list(_china_shock.COLUMNS)
    assert len(_china_shock.COLUMNS) == 191


def test_offline_without_cache_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(_fetch.ENV_DATA_DIR, str(tmp_path))
    with pytest.raises(DataInsufficient, match="sha256|SHA-256|download"):
        sp.datasets.china_shock(download=False)


def test_load_dataset_dispatches():
    assert "china_shock" in sp.datasets.__all__
    assert sp.load_dataset.__module__ == "statspai.datasets"


@pytest.fixture(scope="module")
def real():
    if not (_fetch.is_cached(SPEC) or _fetch.fetch_allowed_in_tests()):
        pytest.skip(
            "Agarwal-Singh replication zip not cached; set "
            "STATSPAI_FETCH_DATASETS=1 to download it (9.6 MB)."
        )
    return sp.datasets.china_shock()


def test_real_file_shape_and_columns(real):
    assert real.shape == (1444, 191)
    assert tuple(real.columns) == _china_shock.COLUMNS
    assert not any(real.dtypes == np.float32)
    assert real["czone"].nunique() == 722
    assert sorted(real["yr"].unique().tolist()) == [1990, 2000]


def test_real_file_equals_the_parity_design(real):
    # The ADH parity test reads the same member through its own loader,
    # which checks the .dta's SHA-256.
    here = Path(__file__).parent / "reference_parity" / "_fixtures"
    spec = importlib.util.spec_from_file_location(
        "_noisy_adh_design", here / "_generate_noisy_covariates_adh_data.py"
    )
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    ref = g.load_adh(_fetch.fetch(SPEC))
    # (it casts every column to float64; the dummies stay int8 here)
    pd.testing.assert_frame_equal(
        real[list(ref.columns)].astype("float64"), ref, check_exact=True
    )
