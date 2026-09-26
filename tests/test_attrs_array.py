"""Result tables that carry a covariance matrix in ``attrs`` must still print.

pandas truncates a wide or long frame for display with ``concat``, which
compares the pieces' ``attrs`` with ``dict ==``; a raw ndarray there raised
``ValueError: The truth value of an array ... is ambiguous``.
"""

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.utils._attrs import attrs_array


def test_equality_is_a_single_bool():
    V = attrs_array([[1.0, 0.5], [0.5, np.nan]])
    assert (V == np.array([[1.0, 0.5], [0.5, np.nan]])) is True
    assert (V == np.eye(2)) is False
    assert (V == np.ones(3)) is False
    assert (V != np.eye(2)) is True


def test_behaves_as_an_array_otherwise():
    raw = np.array([[4.0, 1.0], [1.0, 9.0]])
    V = attrs_array(raw)
    np.testing.assert_array_equal(np.sqrt(np.diag(V)), [2.0, 3.0])
    assert type(V @ np.ones(2)) is np.ndarray
    assert type(V * 2) is np.ndarray
    assert float(V[1, 1]) == 9.0
    np.testing.assert_array_equal(np.equal(V, raw), np.ones((2, 2), bool))
    with pytest.raises(ValueError):
        V[0, 0] = 0.0  # read-only
    raw[0, 0] = -1.0
    assert V[0, 0] == 4.0  # a copy, not a view of the caller's array


def test_survives_deepcopy_and_pickle():
    V = attrs_array(np.arange(4.0).reshape(2, 2))
    for W in (copy.deepcopy(V), pickle.loads(pickle.dumps(V))):
        assert (W == V) is True
        np.testing.assert_array_equal(np.asarray(W), np.asarray(V))


def test_frame_with_matrix_in_attrs_prints_and_concats():
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(80, 20)))
    df.attrs["vcov"] = attrs_array(np.eye(3))
    with pd.option_context(
        "display.width", 60, "display.max_rows", 10, "display.max_columns", 5
    ):
        repr(df)
    out = pd.concat([df.iloc[:5], df.iloc[5:10]])
    assert (out.attrs["vcov"] == np.eye(3)) is True


@pytest.fixture(scope="module")
def fe_forest():
    df = sp.datasets.currency_union_panel(seed=0)
    cf = sp.causal_forest(
        data=df,
        y="log_trade",
        d="euro",
        x=["pre_trade", "log_gdp_prod", "log_gdppc"],
        id="pair",
        time="year",
        fe="twoway",
        n_estimators=200,
        random_state=0,
    )
    return df, cf


def test_forest_group_effects_table_prints(fe_forest):
    df, cf = fe_forest
    members = df[["country_i", "country_j"]].to_numpy()
    tables = [
        sp.forest_group_effects(cf, members=members, scale="percent"),
        sp.forest_group_effects(cf, by=df["pair"].to_numpy()),  # 105 rows
        cf.best_linear_projection(),
    ]
    with pd.option_context("display.width", 80, "display.max_rows", 20):
        for tab in tables:
            repr(tab)
            V = tab.attrs["vcov"]
            assert V.shape == (len(tab), len(tab))
            np.testing.assert_allclose(
                np.sqrt(np.diag(V)), tab.iloc[:, 1] if "se" not in tab else tab["se"]
            )
