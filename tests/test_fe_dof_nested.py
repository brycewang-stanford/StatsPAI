"""did._core.fe_dof_not_nested: the fixest/reghdfe nested-K rule."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statspai.did._core import fe_dof_not_nested


def _panel(n_units: int = 6, n_periods: int = 4) -> pd.DataFrame:
    rows = [
        {"unit": u, "time": t, "state": u % 2}
        for u in range(n_units)
        for t in range(n_periods)
    ]
    return pd.DataFrame(rows)


def test_unit_effect_nested_in_unit_cluster_counts_only_time_levels():
    df = _panel()
    # unit nested in cluster=unit -> 0; time has 4 levels -> K_fe = 4
    assert fe_dof_not_nested(df, ["unit", "time"], "unit") == 4


def test_two_non_nested_effects_lose_one_collinear_level():
    df = _panel()
    # cluster on state: unit is nested in state (each unit has one state)
    # so only time counts -> 4; cluster on time instead: unit (6) not
    # nested, time nested -> 6.
    assert fe_dof_not_nested(df, ["unit", "time"], "state") == 4
    assert fe_dof_not_nested(df, ["unit", "time"], "time") == 6
    # cluster on a variable nesting neither: 6 + 4 - (2 - 1) = 9
    df["cl"] = np.arange(len(df)) % 5
    assert fe_dof_not_nested(df, ["unit", "time"], "cl") == 9


def test_no_fixed_effects_gives_zero():
    df = _panel()
    assert fe_dof_not_nested(df, [], "unit") == 0
