"""``sp.ctree`` against ``partykit::ctree``.

The reference trees are written by ``_fixtures/_generate_ctree.R`` on the
synthetic data of ``_fixtures/_generate_ctree_data.py``. A conditional
inference tree is deterministic, so everything is compared: the shape of
the tree, the variable and cut point of every split, the statistic and the
adjusted p-value of every test in every inner node, and the terminal node
and prediction of every row.

Tolerances: statistics 1e-9 relative (two implementations of the same
closed-form moments); p-values 1e-7 relative, since partykit reports
``1 - (1 - p)^m`` and loses digits when ``p`` is tiny; cut points exact.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
REF = json.loads((FIX / "ctree_R.json").read_text(encoding="utf-8"))["cases"]


def _data(name):
    df = pd.read_csv(FIX / f"ctree_{name}.csv")
    if name == "multi":
        df["grade"] = pd.Categorical(
            df["grade"], categories=["low", "mid", "high", "top"], ordered=True
        )
    return df


CASES = {
    "reg_default": ("reg", {}),
    "reg_alpha30": ("reg", {"alpha": 0.3}),
    "reg_univariate": ("reg", {"test": "univariate"}),
    "reg_depth2": ("reg", {"max_depth": 2}),
    "reg_bucket40": ("reg", {"min_split": 90, "min_bucket": 40}),
    "multi_default": ("multi", {}),
    "multi_maximum": ("multi", {"split_stat": "maximum"}),
    "multi_alpha40": ("multi", {"alpha": 0.4}),
    "small_default": ("small", {"task": "classification"}),
    "small_alpha50": ("small", {"task": "classification", "alpha": 0.5}),
    "small_prob20": (
        "small",
        {"task": "classification", "alpha": 0.5, "min_prob": 0.2},
    ),
    "small_bucket1": (
        "small",
        {
            "task": "classification",
            "alpha": 0.5,
            "min_split": 4,
            "min_bucket": 1,
            "min_prob": 0.0,
        },
    ),
}


def _as_list(v):
    return v if isinstance(v, list) else [v]


@pytest.mark.parametrize("case", sorted(CASES))
def test_tree_equals_partykit(case):
    name, kwargs = CASES[case]
    df = _data(name)
    ref = REF[case]
    fit = sp.ctree(df, y="y", covariates=[c for c in df if c != "y"], **kwargs)

    assert len(fit.nodes) == len(ref["nodes"])
    for node in ref["nodes"]:
        row = fit.nodes.loc[node["id"]]
        assert row["n"] == node["n"]
        assert pd.isna(row["variable"]) == node["terminal"]
        if node["terminal"]:
            continue
        assert row["variable"] == node["variable"]
        kids = fit.nodes.index[fit.nodes["parent"] == node["id"]].tolist()
        assert kids == node["kids"]
        if "threshold" in node:
            cut = row["split"]
            col = df[node["variable"]]
            if isinstance(col.dtype, pd.CategoricalDtype):  # ordered factor
                cut = cut + 1  # partykit counts levels from 1
            assert cut == pytest.approx(node["threshold"], abs=1e-12)
        else:
            assert sorted(row["split"]) == sorted(_as_list(node["left"]))
        tests = fit.tests[fit.tests["node"] == node["id"]].set_index("variable")
        assert list(tests.index) == _as_list(node["tested"])
        np.testing.assert_allclose(
            tests["statistic"], _as_list(node["statistic"]), rtol=1e-9
        )
        np.testing.assert_allclose(
            tests["p_value"], _as_list(node["p_value"]), rtol=1e-7, atol=1e-300
        )

    assert fit.node_id.tolist() == ref["node_id"]
    if fit.task == "classification":
        assert [str(v) for v in fit.fitted] == [str(v) for v in ref["fitted"]]
    else:
        np.testing.assert_allclose(fit.fitted.astype(float), ref["fitted"], rtol=1e-12)
    assert fit.predict(df, type="node").tolist() == ref["node_id"]
