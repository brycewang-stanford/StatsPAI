"""sp.panel_view: panelView-style treatment-status and outcome display."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _panel(reversal: bool = False):
    rows = []
    for u in range(1, 7):
        for t in range(1, 7):
            d = int(u <= 2 and t >= 4) or int(u in (3, 4) and t >= 5)
            if reversal and u == 1 and t == 6:
                d = 0
            rows.append({"id": u, "t": t, "d": d, "y": u + 0.5 * t + 2.0 * d})
    return pd.DataFrame(rows)


def test_summary_detects_staggering_and_never_treated():
    fig, ax, info = sp.panel_view(_panel(), unit="id", time="t", treat="d")
    assert info["n_units"] == 6 and info["n_periods"] == 6
    assert info["n_treated_units"] == 4 and info["n_never_treated"] == 2
    assert info["adoption_periods"] == [4, 5] and info["staggered"] is True
    assert info["has_reversals"] is False
    assert info["n_missing_cells"] == 0
    assert info["first_treat"][1] == 4 and info["first_treat"][6] is None


def test_reversals_and_missing_cells_are_reported():
    df = _panel(reversal=True)
    df = df[~((df["id"] == 5) & (df["t"] == 2))]
    _, _, info = sp.panel_view(
        df, unit="id", time="t", treat="d", y="y", type="outcome"
    )
    assert info["has_reversals"] is True
    assert info["n_missing_cells"] == 1


def test_outcome_type_requires_y_and_validates_type():
    df = _panel()
    with pytest.raises(ValueError):
        sp.panel_view(df, unit="id", time="t", treat="d", type="outcome")
    with pytest.raises(ValueError):
        sp.panel_view(df, unit="id", time="t", treat="d", type="bivariate")


def test_registered():
    assert sp.describe_function("panel_view")["name"] == "panel_view"
