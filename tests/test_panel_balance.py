"""Tests for ``sp.balance_panel`` (previously untested public function).

``balance_panel`` keeps only entities observed in *every* time period —
deterministic, so the expected output is fully analytic.
"""

import pandas as pd
import pytest
import numpy as np

import statspai as sp


def test_balance_panel_keeps_only_fully_observed_entities():
    # id 1 and 3 appear in all of t in {1,2,3}; id 2 misses t=3.
    df = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 3, 3, 3],
            "t": [1, 2, 3, 1, 2, 1, 2, 3],
            "y": range(8),
        }
    )
    out = sp.balance_panel(df, entity="id", time="t")
    np.testing.assert_allclose(len(out), 6)
    np.testing.assert_allclose(out["id"].nunique(), 2)
    assert sorted(out["id"].unique().tolist()) == [1, 3]
    assert len(out) == 6
    # Surviving rows are untouched (no reindexing of values).
    assert set(out["y"]) == {0, 1, 2, 5, 6, 7}


def test_balance_panel_idempotent_on_balanced_input():
    df = pd.DataFrame({"id": [1, 1, 2, 2], "t": [1, 2, 1, 2], "y": [10, 11, 12, 13]})
    out = sp.balance_panel(df, entity="id", time="t")
    pd.testing.assert_frame_equal(out.reset_index(drop=True), df.reset_index(drop=True))


def test_balance_panel_empty_when_no_entity_is_complete():
    # Each entity observed in a disjoint set of periods -> none complete.
    df = pd.DataFrame({"id": [1, 2, 3], "t": [1, 2, 3], "y": [0, 1, 2]})
    out = sp.balance_panel(df, entity="id", time="t")
    assert len(out) == 0


def test_balance_panel_does_not_mutate_input():
    df = pd.DataFrame({"id": [1, 1, 2], "t": [1, 2, 1], "y": [0, 1, 2]})
    before = df.copy()
    sp.balance_panel(df, entity="id", time="t")
    pd.testing.assert_frame_equal(df, before)


def test_balance_panel_missing_column_errors():
    df = pd.DataFrame({"id": [1, 1], "t": [1, 2], "y": [0, 1]})
    with pytest.raises((KeyError, ValueError)):
        sp.balance_panel(df, entity="id", time="not_a_col")
