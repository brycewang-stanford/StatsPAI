"""Interference estimators with a deterministic reference, against R and Stata (T2).

Same bytes on every side (``_fixtures/interference_unit_data.csv``,
``interference_cluster_data.csv``, ``peer_network_data.csv`` and
``peer_network_edges.csv``, written by
``_fixtures/_generate_proximal_interference_data.py``). References, run as
black boxes:

* R 4.5.2: ``stats::lm`` / ``stats::t.test``, ``ivreg`` 0.6.7,
  ``sandwich`` 3.1.1 (``_fixtures/_generate_proximal_interference_R.R``);
* Stata 18: ``regress``, ``collapse`` + ``ttest``, ``ivregress 2sls``
  (``_fixtures/_generate_proximal_interference_Stata.do``), with the
  network products formed in Mata.

What each function is, and so what it is compared with:

* ``sp.inward_outward_spillover``: OLS of ``y`` on own treatment, inward
  and outward exposure (and covariates), classical covariance.
* ``sp.cluster_cross_interference``: OLS of the cluster means on the
  cluster's treatment and neighbour treated share, classical covariance.
* ``sp.cluster_matched_pair``: mean of the within-pair differences of
  cluster means and the paired t-test standard error ``sd / sqrt(G)``.
* ``sp.peer_effects``: the Bramoulle-Djebbari-Fortin 2SLS of ``y`` on
  ``(1, W y, X[, W X])`` with instruments ``(1, X, W X, W^2 X, W^3 X)``,
  ``W`` the row-normalised adjacency (isolates keep a zero row), HC0
  covariance (``vcovHC(type = "HC0")``; Stata ``ivregress`` with
  ``vce(robust)`` and no ``small``). Every coefficient and SE is compared.
* ``sp.interference`` dispatches to the four by name; the routes are
  checked against the same numbers.

Tolerance 1e-9 relative on estimates and standard errors (observed at most
6e-14 against R and 7e-14 against Stata, rounding).

Not compared, and why: ``sp.network_exposure`` uses exact exposure
probabilities and a dependency-graph variance (``v'Gv / n^2``) that no
package reports (Aronow and Samii's ``interference`` package simulates the
probabilities and uses a different variance bound); ``sp.two_stage_interference``
estimates the Hudgens-Halloran randomisation estimand, while R
``inferference`` estimates IPW allocation-strategy effects of an
observational design, a different quantity; ``sp.spillover``,
``sp.cluster_staggered_rollout`` and ``sp.dnc_gnn_did`` report bootstrap
standard errors; ``sp.interference_test`` is a randomisation test;
``sp.network_hte`` cross-fits gradient boosting. Their evidence stays the
enumeration and known-truth tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "proximal_interference_R.json").read_text(encoding="utf-8"))
STATA = json.loads(
    (_FIX / "proximal_interference_Stata.json").read_text(encoding="utf-8")
)
UNIT = pd.read_csv(_FIX / "interference_unit_data.csv")
CLUSTER = pd.read_csv(_FIX / "interference_cluster_data.csv")
PEER = pd.read_csv(_FIX / "peer_network_data.csv")
EDGES = pd.read_csv(_FIX / "peer_network_edges.csv")
RTOL = 1e-9


def _check(est: float, se: float, r_ref: dict, stata_ref: list) -> None:
    assert est == pytest.approx(r_ref["estimate"], rel=RTOL)
    assert se == pytest.approx(r_ref["se"], rel=RTOL)
    assert est == pytest.approx(stata_ref[0], rel=RTOL)
    assert se == pytest.approx(stata_ref[1], rel=RTOL)


def _adjacency() -> np.ndarray:
    n = len(PEER)
    A = np.zeros((n, n))
    A[EDGES["from"] - 1, EDGES["to"] - 1] = 1.0
    A[EDGES["to"] - 1, EDGES["from"] - 1] = 1.0
    return A


@pytest.mark.parametrize("case,covariates", [("covariates", ["x"]), ("bare", None)])
@pytest.mark.parametrize("via_dispatcher", [False, True])
def test_inward_outward_equals_lm(case, covariates, via_dispatcher):
    kw = dict(
        data=UNIT,
        y="y",
        treatment="d",
        inward_exposure="e_in",
        outward_exposure="e_out",
        covariates=covariates,
    )
    res = (
        sp.interference("inward_outward", **kw)
        if via_dispatcher
        else sp.inward_outward_spillover(**kw)
    )
    ref = R["inward_outward"][case]
    _check(res.inward_effect, res.inward_se, ref["inward"], STATA[f"io_{case}_inward"])
    _check(
        res.outward_effect, res.outward_se, ref["outward"], STATA[f"io_{case}_outward"]
    )


@pytest.mark.parametrize("via_dispatcher", [False, True])
def test_cluster_cross_equals_lm_on_cluster_means(via_dispatcher):
    kw = dict(
        data=CLUSTER,
        y="y",
        cluster="cluster",
        treat="treat",
        neighbour_treat_share="nshare",
    )
    res = (
        sp.interference("cluster_cross", **kw)
        if via_dispatcher
        else sp.cluster_cross_interference(**kw)
    )
    ref = R["cluster_cross"]
    _check(
        res.direct_effect, res.direct_se, ref["direct"], STATA["cluster_cross_direct"]
    )
    _check(
        res.spillover_effect,
        res.spillover_se,
        ref["spillover"],
        STATA["cluster_cross_spillover"],
    )
    assert res.n_clusters == ref["n_clusters"] == 70


@pytest.mark.parametrize("via_dispatcher", [False, True])
def test_matched_pair_equals_paired_t_test(via_dispatcher):
    kw = dict(data=CLUSTER, y="y", cluster="cluster", treat="treat", pair="pair")
    res = (
        sp.interference("matched_pair", **kw)
        if via_dispatcher
        else sp.cluster_matched_pair(**kw)
    )
    ref = R["matched_pair"]
    _check(res.estimate, res.se, ref, STATA["matched_pair"])
    assert res.n_pairs == ref["n_pairs"] == 35


_PEER_NAMES = {
    "(Intercept)": ("(Intercept)", "_cons"),
    "WY": ("wy", "wy"),
    "xa": ("xa", "xa"),
    "xb": ("xb", "xb"),
    "W*xa": ("wxa", "wxa"),
    "W*xb": ("wxb", "wxb"),
}


@pytest.mark.parametrize(
    "case,include_contextual", [("contextual", True), ("no_contextual", False)]
)
@pytest.mark.parametrize("via_dispatcher", [False, True])
def test_peer_effects_equals_ivreg_hc0(case, include_contextual, via_dispatcher):
    kw = dict(
        data=PEER,
        y="y",
        covariates=["xa", "xb"],
        W=_adjacency(),
        include_contextual=include_contextual,
    )
    res = (
        sp.interference("peer_effects", **kw)
        if via_dispatcher
        else sp.peer_effects(**kw)
    )
    table = res.coefficients.set_index("variable")
    expected = 6 if include_contextual else 4
    assert len(table) == expected
    for name, row in table.iterrows():
        r_name, s_name = _PEER_NAMES[name]
        _check(
            row["coef"],
            row["se"],
            {
                "estimate": R["peer"][case]["coef"][r_name],
                "se": R["peer"][case]["se"][r_name],
            },
            STATA[f"peer_{case}_{s_name}"],
        )
    assert res.endogenous_peer == pytest.approx(R["peer"][case]["coef"]["wy"], rel=RTOL)
