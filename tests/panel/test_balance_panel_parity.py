"""sp.balance_panel must keep exactly the entities observed in every period.

Regression lock for module 69. ``sp.balance_panel`` filters a long panel to
the cross-section of entities observed in every time period and sorts by
(entity, time). The estimator is a deterministic row-filter, so agreement
is exact.

The committed R golden (``tests/r_parity/69_balance_panel.R``) applies the
same `counts == n_periods` filter in base R and emits the same row indices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai import balance_panel

_R_GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "r_parity"
    / "results"
    / "69_balance_panel_R.json"
)


def _make_data() -> pd.DataFrame:
    """5 entities × 4 periods with one (id=2, t=1) obs missing (unbalanced)."""
    rows = []
    rng = np.random.default_rng(44)
    for uid in range(5):
        for t in range(4):
            if uid == 2 and t == 1:
                continue
            rows.append({"id": uid, "year": t, "y": float(rng.normal())})
    return pd.DataFrame(rows)


def _r_reference():
    rows = json.loads(_R_GOLDEN.read_text(encoding="utf-8"))["rows"]
    return {r["statistic"]: r for r in rows}


def test_balance_panel_drops_short_unit():
    df = _make_data()
    out = balance_panel(df, entity="id", time="year")
    ref = _r_reference()
    assert len(out) == int(ref["n_obs_balanced"]["estimate"])
    assert int(out["id"].nunique()) == int(ref["n_units_kept"]["estimate"])


@pytest.mark.parametrize("row_offset", [0, "mid", "last"])
def test_balance_panel_row_order_matches_r(row_offset):
    df = _make_data()
    out = balance_panel(df, entity="id", time="year").reset_index(drop=True)
    n = len(out)
    # Python harness emits statistic names row0, row<n//2>, row<n-1>.
    row_idx = {0: 0, "mid": n // 2, "last": n - 1}[row_offset]
    ref = _r_reference()
    assert int(out["id"].iloc[row_idx]) == int(ref[f"row{row_idx}_id"]["estimate"])
    assert int(out["year"].iloc[row_idx]) == int(ref[f"row{row_idx}_year"]["estimate"])
