"""Reference parity: ``sp.policy_tree`` vs R ``policytree::policy_tree``.

Reads the committed Track A module-70 artifacts
(``tests/r_parity/results/70_policy_tree_{py,R}.json``) and asserts the
two exact tree searches agree on the *policy itself*, not merely on a
scalar summary of it.

Why this test exists on top of the module-70 comparator row
-----------------------------------------------------------
``compare.py`` joins the two sides on scalar statistics (policy value,
treated fraction, root split).  Two different trees can share a policy
value to many digits, so scalar agreement alone is not proof that the
two optimisers returned the same rule.  Both sides therefore also dump
the full per-row policy vector into the JSON ``extra`` block, and this
test checks them elementwise.

The comparison is fully deterministic: the AIPW score vector is computed
once on the Python side and shipped to R inside the module CSV, so the
two engines maximise the identical objective over the identical data and
the optimum is a fixed point of the problem, not of the RNG.

Reference
---------
[@athey2021policy], [@zhou2023offline]
"""

from __future__ import annotations

import json
import pathlib

import pytest

_RESULTS = pathlib.Path(__file__).parents[1] / "r_parity" / "results"
_PY = _RESULTS / "70_policy_tree_py.json"
_R = _RESULTS / "70_policy_tree_R.json"

pytestmark = pytest.mark.skipif(
    not (_PY.exists() and _R.exists()),
    reason="Track A module 70 artifacts are not materialized",
)


@pytest.fixture(scope="module")
def sides() -> tuple[dict, dict]:
    py = json.loads(_PY.read_text(encoding="utf-8"))
    r = json.loads(_R.read_text(encoding="utf-8"))
    return py, r


@pytest.fixture(scope="module")
def stats(sides) -> tuple[dict, dict]:
    py, r = sides
    return (
        {row["statistic"]: row["estimate"] for row in py["rows"]},
        {row["statistic"]: row["estimate"] for row in r["rows"]},
    )


@pytest.mark.parametrize("depth", [1, 2])
def test_policy_vectors_are_elementwise_identical(sides, depth):
    """Every unit gets the same recommendation from both engines.

    This is the strongest statement the module can make: not "the two
    trees score alike" but "the two trees *are* the same rule".
    """
    py, r = sides
    key = f"depth{depth}"
    py_policy = [int(v) for v in py["extra"]["policy"][key]]
    r_policy = [int(v) for v in r["extra"]["policy"][key]]
    assert len(py_policy) == len(r_policy) > 0
    mismatches = [i for i, (a, b) in enumerate(zip(py_policy, r_policy)) if a != b]
    assert not mismatches, (
        f"depth-{depth} policy differs on {len(mismatches)} of "
        f"{len(py_policy)} rows (first at index {mismatches[:5]}). The two "
        f"exact searches optimise the same shared-Gamma objective, so any "
        f"disagreement is a tree-search bug, not Monte Carlo noise."
    )


@pytest.mark.parametrize("depth", [1, 2])
def test_scalar_statistics_match_to_machine_tier(stats, depth):
    """Policy value, treated fraction, and root split agree to ~1e-15."""
    py, r = stats
    for name in (
        f"value_policy_d{depth}",
        f"fraction_treated_d{depth}",
        f"root_split_variable_d{depth}",
        f"root_split_value_d{depth}",
    ):
        assert name in py and name in r, name
        assert py[name] == pytest.approx(r[name], rel=1e-12, abs=1e-12), name


def test_depth2_beats_depth1_on_the_shared_objective(stats):
    """Sanity: the deeper class contains the shallower one.

    A depth-2 optimum below the depth-1 optimum would mean the search is
    not solving a nested-class maximisation and would invalidate the
    exactness claim, independently of what R reports.
    """
    py, _ = stats
    assert py["value_policy_d2"] >= py["value_policy_d1"] - 1e-12


def test_module_records_the_shared_nuisance_contract(sides):
    """The artifacts must document that Gamma was shared, not re-estimated."""
    py, r = sides
    assert py["extra"]["search"] == "exact"
    assert py["extra"]["split_step"] == 1
    assert int(py["extra"]["min_node_size"]) == int(r["extra"]["min.node.size"][0])
    assert "shared" in py["extra"]["note"].lower()
