"""``sp.ida`` against R ``pcalg::ida``.

Reference: pcalg 2.7.12 (GPL >= 2, used as a black box) from
``_fixtures/_generate_ida_pcalg.R``: 40 random linear-Gaussian DAGs, their
CPDAGs and sample covariances; ``ida(method = 'local')`` and
``ida(method = 'global')``. The multisets of possible effects agree to
1e-10. A chain checks the estimand: the possible effects are the effects
in the three DAGs of the class.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def ref():
    return json.loads((_FIX / "ida_pcalg.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("method", ["local", "global"])
def test_possible_effects_match_pcalg(ref, method):
    for k, case in enumerate(ref):
        p = len(case["A"])
        names = [f"v{i + 1}" for i in range(p)]
        A = pd.DataFrame(case["A"], index=names, columns=names)
        S = pd.DataFrame(case["S"], index=names, columns=names)
        r = sp.ida(S, A, x=names[case["x"]], y=names[case["y"]], method=method)
        want = np.sort(np.atleast_1d(case[method]))
        got = np.sort(r.effects)
        assert len(got) == len(want), k
        assert np.allclose(got, want, rtol=1e-10, atol=1e-12), k


def test_chain_known_truth():
    """a - b - c with no v-structure: the class holds a->b->c, a<-b<-c and
    a<-b->c. The effect of b on c is 0 in the second DAG and the regression
    slope otherwise; b's parent set is {}, {a} or {c}."""
    rng = np.random.default_rng(0)
    n = 200_000
    a = rng.normal(size=n)
    b = 0.8 * a + rng.normal(size=n)
    c = 0.5 * b + rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    cp = pd.DataFrame(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]], index=list("abc"), columns=list("abc")
    )
    r = sp.ida(df, cp, x="b", y="c")
    assert sorted(map(sorted, r.parent_sets)) == [[], ["a"], ["c"]]
    eff = dict(zip(map(tuple, r.parent_sets), r.effects))
    assert eff[("c",)] == 0.0
    assert eff[("a",)] == pytest.approx(0.5, abs=0.01)  # the true effect
    assert r.min_abs_effect == 0.0


def test_accepts_pc_output_and_refuses_bad_input():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    df = pd.DataFrame(
        {"x": x, "y": x + rng.normal(size=500), "z": rng.normal(size=500)}
    )
    res = sp.pc_algorithm(df)
    r = sp.ida(df, res, x="x", y="y")
    assert len(r.effects) >= 1
    with pytest.raises(sp.MethodIncompatibility, match="method"):
        sp.ida(df, res, x="x", y="y", method="optimal")
    with pytest.raises(sp.MethodIncompatibility, match="not a node"):
        sp.ida(df, res, x="x", y="w")


def test_cpdag_as_json_dict_equals_the_dataframe():
    # 2026-10-20 probe: an agent sends the CPDAG as DataFrame.to_dict()
    # (labels included); it was refused as "an array needs node names".
    rng = np.random.default_rng(0)
    a = rng.normal(size=2000)
    b = a + rng.normal(size=2000)
    c = b + rng.normal(size=2000)
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    cp = pd.DataFrame(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]], index=list("abc"), columns=list("abc")
    )
    via_json = json.loads(json.dumps(cp.to_dict()))
    r1 = sp.ida(df, cp, x="b", y="c")
    r2 = sp.ida(df, via_json, x="b", y="c")
    assert np.allclose(sorted(r1.effects), sorted(r2.effects), rtol=0, atol=0)
