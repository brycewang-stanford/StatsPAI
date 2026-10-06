"""``sp.mmpc`` and ``sp.mmhc`` against bnlearn.

Twenty-four simulated data sets (``_fixtures/_generate_mmhc_data.py``:
random DAGs on five to eight variables, half Gaussian and half categorical,
500 to 3,000 rows), with ``bnlearn`` 5.2.1 as the reference
(``_generate_mmhc_bnlearn.R``).

What is compared, and at what level:

* the skeletons of ``mmpc`` and ``si.hiton.pc``. They are sets of edges,
  so the comparison is exact, and they are equal on all 24. The algorithms
  are implemented from the papers, not from bnlearn's source, and a
  different order of admitting candidates could in principle test
  different subsets; on these data it does not change the answer.
* the BIC of the graph the hybrid search ends at. Both searches stop at a
  local optimum, so the assertion is one-sided: never below bnlearn's.
  It is equal on 18 of the 24 and higher on 6, where the hill climbing
  here leaves a plateau that ``bnlearn::hc`` stops on.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

FIX = Path(__file__).parent / "_fixtures"
REF = json.loads((FIX / "mmhc_bnlearn_R.json").read_text(encoding="utf-8"))["cases"]
_spec = importlib.util.spec_from_file_location(
    "_generate_mmhc_data", FIX / "_generate_mmhc_data.py"
)
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)
DATA = _gen.datasets()
NAMES = sorted(REF)


def _pairs(edges):
    return {frozenset(e) for e in edges}


def test_the_fixture_covers_every_data_set():
    assert NAMES == sorted(DATA) and len(NAMES) == 24
    for name in NAMES:
        assert DATA[name].shape == (REF[name]["n"], REF[name]["p"])


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("method", ["mmpc", "hiton"])
def test_skeleton_equals_bnlearn(name, method):
    out = sp.mmpc(DATA[name], method=method)
    assert _pairs(out["edges"]) == _pairs(REF[name][method])
    skel = out["skeleton"].to_numpy()
    assert (skel == skel.T).all() and not skel.diagonal().any()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "restrict, key", [("mmpc", "mmhc_score"), ("hiton", "hiton_hc_score")]
)
def test_hybrid_score_is_not_below_bnlearn(name, restrict, key):
    out = sp.mmhc(DATA[name], restrict=restrict)
    # 1e-6: both sides sum the same log-likelihood terms in another order
    assert out["score"] >= REF[name][key] - 1e-6
    # every arc lies on a candidate edge
    assert _pairs(out["edges"]) <= _pairs(out["candidate_edges"])


def test_how_often_the_hybrid_graph_is_bnlearns():
    equal = sum(
        abs(sp.mmhc(DATA[n])["score"] - REF[n]["mmhc_score"]) < 1e-6 for n in NAMES
    )
    assert equal >= 18, equal


def test_restriction_never_beats_the_restricted_optimum():
    """The hybrid result is a legal graph for plain hill climbing started
    there: restricting the candidates cannot raise the reachable score
    above what the same search finds with the restriction lifted from that
    graph."""
    for name in NAMES[:6]:
        hybrid = sp.mmhc(DATA[name])
        kept = sp.hill_climb(DATA[name], required=hybrid["edges"])
        assert kept["score"] >= hybrid["score"] - 1e-8


def test_recovers_a_known_skeleton():
    """Known truth: a chain and a collider, 4,000 rows."""
    rng = np.random.default_rng(11)
    n = 4000
    a = rng.normal(size=n)
    b = 0.8 * a + rng.normal(size=n)
    c = 0.8 * b + rng.normal(size=n)
    e = rng.normal(size=n)
    d = 0.7 * c + 0.7 * e + rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    truth = {frozenset(p) for p in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]}
    for method in ("mmpc", "hiton"):
        assert _pairs(sp.mmpc(df, method=method)["edges"]) == truth
    out = sp.mmhc(df)
    assert _pairs(out["edges"]) == truth
    assert ("c", "d") in out["edges"] and ("e", "d") in out["edges"]  # the collider


def test_max_cond_zero_is_marginal_screening():
    df = DATA["d00"]
    out = sp.mmpc(df, max_cond=0)
    corr = df.corr().to_numpy()
    n = len(df)
    from scipy import stats

    t = corr * np.sqrt((n - 2) / (1 - np.clip(corr**2, 0, 1 - 1e-12)))
    marginal = 2 * stats.t.sf(np.abs(t), n - 2) <= 0.05
    np.fill_diagonal(marginal, False)
    assert (out["skeleton"].to_numpy().astype(bool) == marginal).all()
    assert len(sp.mmpc(df)["edges"]) <= len(out["edges"])


def test_refusals():
    df = DATA["d00"]
    with pytest.raises(MethodIncompatibility, match="method must be"):
        sp.mmpc(df, method="gs")
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.mmpc(df, alpha=1.5)
    with pytest.raises(MethodIncompatibility, match="not in data"):
        sp.mmpc(df, ["v0", "nope"])
    with pytest.raises(MethodIncompatibility, match="mix"):
        sp.mmpc(df.assign(g=np.where(df["v0"] > 0, "a", "b")))
    with pytest.raises(DataInsufficient, match="constant"):
        sp.mmpc(df.assign(k=1.0))
    with pytest.raises(DataInsufficient, match="fewer than 10"):
        sp.mmpc(df.head(5))
    out = sp.mmpc(df)
    apart = next(
        (a, b) for a in df.columns for b in df.columns
        if a < b and frozenset((a, b)) not in _pairs(out["edges"])
    )  # fmt: skip
    with pytest.raises(MethodIncompatibility, match="required arc"):
        sp.mmhc(df, required=[apart])
