"""``didregress / xtdidregress ..., aggregate()`` through ``sp.stata``.

Executed equivalence against Stata 18: the ten lines of
``reference_parity/_fixtures/_generate_didregress_dlang_Stata.do`` run
through ``sp.stata`` on the same CSVs, and the ATET, its standard error, the
number of cells and the confidence bounds Stata reported
(``didregress_dlang_Stata.csv``) are reproduced to 1e-10 (bounds 1e-9): the
translation writes ``dlang_df='stata'`` for the Donald-Lang fits, so the
interval uses Stata's degrees of freedom. ``level(90)`` with
``aggregate(dlang)`` was run in Stata 18 MP on 2026-10-20 (bounds .7378461
and 1.552145 as printed).
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

FX = pathlib.Path(__file__).parent / "reference_parity" / "_fixtures"

LINES = {
    "rcs_dlang_xw": "didregress (y x w) (d), group(g) time(t) aggregate(dlang)",
    "rcs_varying_xw": (
        "didregress (y x w) (d), group(g) time(t) aggregate(dlang, varying)"
    ),
    "rcs_dlang_xwz": "didregress (y x w z) (d), group(g) time(t) aggregate(dlang)",
    "rcs_varying_xwz": (
        "didregress (y x w z) (d), group(g) time(t) aggregate(dlang, varying)"
    ),
    "rcs_dlang_none": "didregress (y) (d), group(g) time(t) aggregate(dlang)",
    "rcs_standard_xw": ("didregress (y x w) (d), group(g) time(t) aggregate(standard)"),
    "rcs_standard_xwz": (
        "didregress (y x w z) (d), group(g) time(t) aggregate(standard)"
    ),
    "panel_dlang_x": (
        "xtset id t\nxtdidregress (y x) (d), group(g) time(t) aggregate(dlang)"
    ),
    "panel_varying_x": (
        "xtset id t\n"
        "xtdidregress (y x) (d), group(g) time(t) aggregate(dlang, varying)"
    ),
    "panel_standard_x": (
        "xtset id t\nxtdidregress (y x) (d), group(g) time(t) aggregate(standard)"
    ),
}


@pytest.fixture(scope="module")
def ref():
    return pd.read_csv(FX / "didregress_dlang_Stata.csv").set_index("case")


@pytest.fixture(scope="module")
def data():
    return {
        "rcs": pd.read_csv(FX / "didregress_dlang.csv"),
        "panel": pd.read_csv(FX / "didregress_dlang_panel.csv"),
    }


def _run(commands, df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.stata(commands, data=df)


@pytest.mark.parametrize("case", sorted(LINES))
def test_stata_lines_run_to_statas_numbers(case, ref, data):
    res = _run(LINES[case], data[case.split("_")[0]])
    r = ref.loc[case]
    assert abs(res.estimate - r.b) < 1e-10
    assert abs(res.se - r.se) < 1e-10
    assert res.model_info["n_cells"] == r.N
    assert abs(res.ci[0] - r.lb) < 1e-9 and abs(res.ci[1] - r.ub) < 1e-9


def test_level_with_dlang_uses_statas_degrees_of_freedom(data):
    res = _run(
        "didregress (y x w) (d), group(g) time(t) aggregate(dlang) level(90)",
        data["rcs"],
    )
    np.testing.assert_allclose(res.ci, [0.7378461, 1.552145], rtol=5e-7)


def test_translation_arguments():
    tr = sp.from_stata(
        "didregress (y x w) (d), group(g) time(t) aggregate(dlang, varying)"
    )
    assert tr["complete"]
    assert tr["arguments"]["aggregate"] == "dlang_varying"
    assert tr["arguments"]["dlang_df"] == "stata"
    tr = sp.from_stata("didregress (y x) (d), group(g) time(t) aggregate(standard)")
    assert tr["arguments"]["aggregate"] == "standard"
    assert "dlang_df" not in tr["arguments"]


@pytest.mark.parametrize(
    "line",
    [
        # Stata 18: "aggregate() with method dlang not allowed with vcetype
        # cluster g" / "... with wild bootstrap"
        "didregress (y x w) (d), group(g) time(t) aggregate(dlang) vce(cluster g)",
        "didregress (y x w) (d), group(g) time(t) aggregate(dlang) wildbootstrap",
    ],
)
def test_combinations_stata_refuses_are_refused(line):
    out = sp.from_stata(line)
    assert out["ok"] is False and "aggregate(dlang)" in out["error"]


def test_unknown_aggregate_is_flagged_and_sp_stata_refuses(data):
    line = "didregress (y x w) (d), group(g) time(t) aggregate(bogus)"
    tr = sp.from_stata(line)
    assert tr["ok"] and tr["dropped"] == ["aggregate"]
    with pytest.raises(MethodIncompatibility, match="aggregate"):
        _run(line, data["rcs"])
