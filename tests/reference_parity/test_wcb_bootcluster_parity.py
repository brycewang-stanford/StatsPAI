"""``sp.wild_cluster_boot(bootcluster=)`` when the bootstrap clusters cut
across the (one-way) error clusters, against Stata ``boottest``.

Reference: ``_fixtures/wcb_bootcluster_Stata.json`` from
``_fixtures/_generate_wcb_bootcluster_Stata.do`` on
``_fixtures/wcb_bootcluster.csv`` (``_generate_wcb_bootcluster_data.py``).
Error clusters ``a`` (3) and bootstrap clusters ``b`` (3) cross in 9 cells.
boottest draws the weights for the cells: its ``bootcluster(b)`` and
``bootcluster(b a)`` give the same p-value at every null value, and it
enumerates all 2^9 Rademacher draws, so the p-values are exact on both
sides (T2, compared as multiples of 1/512). Before 2026-10-20 StatsPAI drew
the weights per ``b`` (2^3 draws): on the probe data that gave p = 0.047
where boottest gives 0.000.
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

FX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((FX / "wcb_bootcluster_Stata.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(FX / "wcb_bootcluster.csv")


def _boot(cluster, bootcluster, h0):
    fit = sp.regress("y ~ x + w", data=DATA, cluster=cluster)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = sp.wild_cluster_boot(
            fit,
            data=DATA,
            cluster=cluster,
            variable="x",
            h0=h0,
            bootcluster=bootcluster,
            n_boot=9999,
            seed=1,
        )
    return out, [str(c.message) for c in caught]


@pytest.mark.parametrize("h0", ["0", "0.2", "0.4", "0.6"])
@pytest.mark.parametrize("bc", ["b", "ba"])
def test_crossing_bootcluster_matches_boottest(h0, bc):
    ref = REF[f"h{h0}_{bc}"]
    out, msgs = _boot("a", ["b"] if bc == "b" else ["b", "a"], float(h0))
    assert out["enumerated"] and out["n_boot"] == ref["reps"] == 512
    assert out["p_boot"] * 512 == pytest.approx(ref["p"] * 512, abs=1e-9)
    assert out["t_stat"] == pytest.approx(ref["t"], rel=1e-10)
    if bc == "b":
        assert any("intersections" in m for m in msgs)


def test_crossing_bootcluster_other_way_round():
    ref = REF["clusterb_bc_a"]
    out, _ = _boot("b", "a", 0.2)
    assert out["p_boot"] * 512 == pytest.approx(ref["p"] * 512, abs=1e-9)


def test_coarser_bootcluster_is_refused():
    """A bootstrap clustering coarser than the one-way error clustering:
    boottest fails (a Mata subscript error), and this refuses."""
    df = DATA.assign(ab=DATA["a"] * 10 + DATA["b"], one=1)
    fit = sp.regress("y ~ x + w", data=df, cluster="ab")
    with pytest.raises(MethodIncompatibility, match="not nested"):
        sp.wild_cluster_boot(
            fit, data=df, cluster="ab", variable="x", bootcluster="a", n_boot=99
        )


def test_nested_bootcluster_is_used_as_given():
    """Bootstrap clusters inside the error clusters (a subcluster
    bootstrap) are not altered: same draws as naming the cells directly."""
    df = DATA.assign(cell=DATA["a"] * 10 + DATA["b"])
    fit = sp.regress("y ~ x + w", data=df, cluster="a")
    kw = dict(data=df, cluster="a", variable="x", n_boot=9999, seed=1, h0=0.2)
    p_cell = sp.wild_cluster_boot(fit, bootcluster="cell", **kw)["p_boot"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_cross = sp.wild_cluster_boot(fit, bootcluster="b", **kw)["p_boot"]
    assert np.isclose(p_cell, p_cross, rtol=0, atol=1e-15)
