"""``sp.rd_honest(cluster=)`` vs R ``RDHonest(clusterid=, se.method='EHW')``.

With clusters RDHonest's standard error is ``sqrt(sum_g (sum_{i in g} w_i
e_i)^2)`` (``w`` the estimator weights, ``e`` the joint local linear
residuals), and its bandwidth search adds a Moulton within-cluster
correlation to the preliminary variance. Cells: ``M`` and ``h`` fixed (the
SE formula), ``M`` fixed with ``h`` chosen (the Moulton-corrected search),
and the free cell (rule-of-thumb ``M``). Reference:
``_fixtures/_generate_rdhonest_cluster_R.R``.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "rdhonest_cluster_R.json").read_text(encoding="utf-8"))
D = pd.read_csv(_FIX / "rdhonest_cluster.csv", float_precision="round_trip")

CASES = {
    "fixed_M1_h0.3": dict(M=1, h=0.3),
    "fixed_M1_h0.6": dict(M=1, h=0.6),
    "fixed_M3_h0.3": dict(M=3, h=0.3),
    "fixed_M3_h0.6": dict(M=3, h=0.6),
    "bwsel_MSE": dict(M=2.4, opt_criterion="mse"),
    "bwsel_FLCI": dict(M=2.4, opt_criterion="flci"),
    "free_MSE": dict(),
}


def _fit(cluster="cl", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.rd_honest(D, y="y", x="x", cluster=cluster, **kw)


@pytest.mark.parametrize("key", sorted(CASES))
def test_matches_rdhonest(key):
    r = _fit(**CASES[key])
    ref = R[key]
    tol = 1e-9 if key.startswith("fixed") else 1e-6
    assert r.model_info["bandwidth"] == pytest.approx(ref["h"], rel=tol)
    assert r.estimate == pytest.approx(ref["estimate"], rel=tol)
    assert r.se == pytest.approx(ref["se"], rel=tol)
    assert r.model_info["bias_bound"] == pytest.approx(ref["bias"], rel=tol)
    assert r.ci[0] == pytest.approx(ref["ci_lower"], rel=tol)
    assert r.ci[1] == pytest.approx(ref["ci_upper"], rel=tol)


def test_unclustered_cell_and_cluster_count():
    un = _fit(cluster=None, M=1, h=0.3)
    assert un.se == pytest.approx(R["unclustered_fixed_M1_h0.3"]["se"], rel=1e-9)
    cl = _fit(M=1, h=0.3)
    assert cl.se > un.se  # cluster-level shocks in the design
    assert cl.model_info["n_clusters"] == 150
