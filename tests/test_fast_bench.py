"""Tests for ``statspai.fast.hdfe_bench`` — HDFE kernel benchmark harness."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from statspai.fast import HDFEBenchResult, hdfe_bench


def test_hdfe_bench_returns_result_type():
    r = hdfe_bench(n_list=(500,), n_groups=10, repeat=1, seed=0)
    assert isinstance(r, HDFEBenchResult)
    df = r.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    # At least numpy + numba rows, probably rust too (marked unavailable)
    assert set(df["backend"]).issuperset({"numpy", "numba", "rust"})


def test_hdfe_bench_dry_run_is_fast():
    """Ensures CI can include the harness without timing out."""
    import time

    t0 = time.perf_counter()
    hdfe_bench(n_list=(100,), n_groups=5, repeat=1, seed=0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"Dry-run took {elapsed:.2f}s"


def test_hdfe_bench_numpy_numba_agree_on_small_dgp():
    """Numba path must match numpy to within atol=1e-10 or bench raises."""
    r = hdfe_bench(n_list=(1_000,), n_groups=20, repeat=1, seed=1, atol=1e-10)
    df = r.to_dataframe()
    numba = df[df["backend"] == "numba"]
    assert not numba.empty
    if numba.iloc[0]["available"]:
        # When numba is available its max err vs ref must be <= atol
        err = float(numba.iloc[0]["max_abs_err_vs_ref"])
        assert err <= 1e-10, f"Numba drift {err:.2e} exceeds atol"


def test_hdfe_bench_unavailable_paths_are_recorded_not_crashed():
    """A missing backend (e.g. Rust) must produce a NaN row, not raise."""
    r = hdfe_bench(n_list=(200,), n_groups=5, repeat=1, seed=0)
    df = r.to_dataframe()
    rust_row = df[df["backend"] == "rust"]
    assert not rust_row.empty
    # Expect rust to be unavailable pre-1.0
    if not rust_row.iloc[0]["available"]:
        assert np.isnan(rust_row.iloc[0]["wall_time_s"])


def test_hdfe_bench_summary_string():
    r = hdfe_bench(n_list=(200,), n_groups=5, repeat=1, seed=0)
    s = r.summary()
    assert isinstance(s, str)
    assert "hdfe_bench" in s
    assert "numpy" in s
    # Each backend should show either AVAILABLE or unavailable
    assert ("AVAILABLE" in s) or ("unavailable" in s)


def test_hdfe_bench_result_protocol_json_safe():
    r = hdfe_bench(n_list=(100,), n_groups=5, repeat=1, seed=0)

    full = r.to_dict()
    agent = r.to_agent_summary(max_rows=2)

    assert full["kind"] == "fast_hdfe_bench_result"
    assert full["reference"] == "numpy"
    assert set(full["backends"]).issuperset({"numpy", "numba", "rust"})
    assert len(full["results"]) >= 3
    assert agent["kind"] == "fast_hdfe_bench_agent_summary"
    assert len(agent["results"]) == 2
    assert agent["truncated_rows"] == len(full["results"]) - 2
    assert agent["best_by_n"]
    json.dumps(full)
    json.dumps(agent)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_list": (0,)}, "n_list"),
        ({"n_list": ()}, "n_list"),
        ({"n_list": "100"}, "n_list"),
        ({"n_list": 100}, "n_list"),
        ({"n_groups": 0}, "n_groups"),
        ({"n_groups": True}, "n_groups"),
        ({"repeat": 0}, "repeat"),
        ({"repeat": True}, "repeat"),
        ({"atol": -1.0}, "atol"),
        ({"atol": np.nan}, "atol"),
    ],
)
def test_hdfe_bench_rejects_invalid_config(kwargs, message):
    params = {"n_list": (100,), "n_groups": 5, "repeat": 1, "seed": 0}
    params.update(kwargs)
    with pytest.raises(ValueError, match=message):
        hdfe_bench(**params)
