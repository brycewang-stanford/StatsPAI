"""Multi-action ``sp.policy_tree`` vs R ``policytree`` 1.2.4 (MIT).

Fixture: ``_fixtures/policytree_multi.csv`` (four covariates, two with
ties, and four reward columns) and ``_fixtures/policytree_multi_R.json``
from ``_fixtures/_generate_policytree_multi_R.R``. Given the reward
matrix, the depth <= 2 tree is a deterministic optimum, so the assigned
action of every row must equal policytree's exactly, and so must the
achieved reward.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "policytree_multi_R.json").read_text(encoding="utf-8"))
CASES = [k for k in REF if k != "versions"]
X_COLS = ["x1", "x2", "x3", "x4"]


def _fit(case):
    d = pd.read_csv(_FIX / "policytree_multi.csv")
    G = d[case["G"]].to_numpy()
    d["arm"] = np.arange(len(d)) % G.shape[1]  # labels only: rewards are given
    d["y"] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.policy_tree(
            d,
            y="y",
            treat="arm",
            covariates=X_COLS,
            scores=G,
            max_depth=case["depth"],
            min_leaf_size=case["mns"],
            split_step=case["step"],
            search="exact",
        )
    return d, G, res


@pytest.mark.parametrize("name", CASES)
def test_actions_match_policytree(name):
    case = REF[name]
    _, G, res = _fit(case)
    want = np.asarray(case["actions"]) - 1
    np.testing.assert_array_equal(res["actions"], want)
    n = len(want)
    assert res["value_policy"] == pytest.approx(G[np.arange(n), want].mean(), rel=1e-12)


@pytest.mark.parametrize("name", CASES)
def test_root_split_matches_policytree(name):
    case = REF[name]
    _, _, res = _fit(case)
    root = case["nodes"][0]
    tree = res["tree"]._tree
    if root["is_leaf"]:
        assert tree["type"] == "leaf"
        return
    assert X_COLS[tree["feature"]] == X_COLS[int(root["split_variable"]) - 1]
    assert tree["threshold"] == pytest.approx(root["split_value"], abs=0, rel=1e-15)


def test_two_columns_reproduce_the_binary_search():
    """A = 2 on rewards (g0, g1) is the binary search on g1 - g0."""
    d = pd.read_csv(_FIX / "policytree_multi.csv")
    d["arm"] = np.arange(len(d)) % 2
    d["y"] = 0.0
    G = d[["g0", "g1"]].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multi = sp.policy_tree(
            d,
            "y",
            treat="arm",
            covariates=X_COLS,
            scores=G,
            max_depth=2,
            min_leaf_size=5,
        )
        binary = sp.policy_tree(
            d,
            "y",
            treat="arm",
            covariates=X_COLS,
            scores=G[:, 1] - G[:, 0],
            max_depth=2,
            min_leaf_size=5,
        )
    np.testing.assert_array_equal(multi["actions"], binary["policy"])


def test_three_arm_treatment_fits_with_internal_scores_and_predicts_labels():
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=n)
    arm = rng.choice(["a", "b", "c"], size=n)
    y = (
        x
        + 1.5 * (arm == "b") * (x > 0)
        + 1.5 * (arm == "c") * (x <= 0)
        + rng.normal(size=n)
    )
    df = pd.DataFrame({"y": y, "arm": arm, "x": x, "z": rng.normal(size=n)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.policy_tree(
            df,
            "y",
            treat="arm",
            covariates=["x", "z"],
            max_depth=1,
            min_leaf_size=50,
            n_folds=3,
        )
    assert res["multi_action"] and res["levels"] == ["a", "b", "c"]
    pred = res["tree"].predict(np.array([[1.5, 0.0], [-1.5, 0.0]]))
    assert list(pred) == ["b", "c"]


def test_reward_matrix_shape_is_checked():
    d = pd.read_csv(_FIX / "policytree_multi.csv")
    d["arm"] = np.arange(len(d)) % 3
    d["y"] = 0.0
    with pytest.raises(MethodIncompatibility, match="levels"):
        sp.policy_tree(
            d, "y", treat="arm", covariates=X_COLS, scores=d[["g0", "g1"]].to_numpy()
        )
    with pytest.raises(MethodIncompatibility, match="reward matrix"):
        sp.policy_tree(
            d, "y", treat="arm", covariates=X_COLS, scores=d["g0"].to_numpy()
        )
