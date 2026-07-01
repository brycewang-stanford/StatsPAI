"""sp.demean must stay pinned to the textbook mean-within projection.

Regression lock for module 68. ``sp.demean(solver="map")`` computes
``y - mean_i(y)`` (and analogously for ``X``) — the within transformation
used by every absorbing-FE estimator in the ``sp.feols`` family and by
``sp.xtabond``. The estimator is purely algorithmic, so agreement to
machine precision is the honest expectation.

The committed R golden (``tests/r_parity/68_demean_within.R``) computes
the same projection with a hand-rolled ``sum()`` per-id loop and emits the
same row indices the Python side reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai import demean

_R_GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "r_parity"
    / "results"
    / "68_demean_within_R.json"
)


def _make_data(seed: int = 42, N: int = 20, T: int = 8) -> pd.DataFrame:
    """Rebuild the module-68 DGP. The draw order (x1, x2, x3, y) must match
    the committed CSV byte-for-byte — the harness's make_data in
    68_demean_within.py uses that exact sequence."""
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(N), T)
    years = np.tile(np.arange(T), N)
    x1 = rng.normal(size=N * T)
    x2 = rng.normal(size=N * T)
    x3 = rng.normal(size=N * T)
    y = rng.normal(size=N * T)
    return pd.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "x3": x3, "id": ids, "year": years}
    )


def _r_reference():
    rows = json.loads(_R_GOLDEN.read_text(encoding="utf-8"))["rows"]
    return {r["statistic"]: r for r in rows}


def test_demean_y_matches_within_projection():
    df = _make_data()
    y = df["y"].to_numpy()
    fe = pd.DataFrame({"id": df["id"].to_numpy()})
    dem_y, _ = demean(y, fe, solver="map")
    ref = _r_reference()
    assert float(dem_y[0]) == pytest.approx(ref["demean_y"]["estimate"], rel=1e-6)


@pytest.mark.parametrize("col", ["x1", "x2", "x3"])
@pytest.mark.parametrize("row_offset", [0, "mid", "last"])
def test_demean_x_matches_within_projection(col, row_offset):
    df = _make_data()
    # Match the Python harness's emitted statistic names (0-indexed):
    # row0, row79, row159 — the R script emits the same labels.
    row_idx = {0: 0, "mid": 79, "last": 159}[row_offset]
    X = df[["x1", "x2", "x3"]].to_numpy()
    fe = pd.DataFrame({"id": df["id"].to_numpy()})
    dem_X, _ = demean(X, fe, solver="map")
    col_idx = {"x1": 0, "x2": 1, "x3": 2}[col]
    stat = f"demean_{col}_row{row_idx}"
    ref = _r_reference()[stat]
    assert float(dem_X[row_idx, col_idx]) == pytest.approx(ref["estimate"], rel=1e-6)


def test_demean_is_zero_per_id():
    """Within transformation must be exactly per-id-mean-zero."""
    df = _make_data()
    y = df["y"].to_numpy()
    fe = pd.DataFrame({"id": df["id"].to_numpy()})
    dem_y, _ = demean(y, fe, solver="map")
    for uid in np.unique(df["id"].to_numpy()):
        m = df["id"].to_numpy() == uid
        assert abs(dem_y[m].sum()) < 1e-10
