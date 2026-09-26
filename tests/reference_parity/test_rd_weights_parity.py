"""``sp.rdrobust(weights=)`` vs R ``rdrobust(weights=)`` (review F07).

Until 1.32 ``weights=`` raised ``NotImplementedError``. R multiplies the
observation weights into the kernel weights of every local regression --
all three stages of ``rdbwselect``, the conventional and bias-corrected
fits, leverages and the sandwich meat -- before the ``w > 0`` selection;
StatsPAI now does the same on the CCT path.

Reference: ``_fixtures/rd_weights_R.json`` from
``_generate_rd_weights_R.R`` (R 4.5.2, rdrobust 4.0.0) on
``_fixtures/rd_weights_data.csv`` (n = 1500; 3 % of rows weight 0; written
once with ``numpy.random.default_rng(20260926)``, see the generator snippet
in the CHANGELOG entry). Each row compares the conventional and robust
estimates, both SEs and the main bandwidth.

Two further defects surfaced by this work are pinned here:

* the ``msecomb1`` / ``msecomb2`` recursion dropped ``fuzzy=``, so a fuzzy
  design with a comb selector got the *sharp* bandwidth (h 0.18 vs R 0.32,
  robust estimate 0.29 vs R 0.56 on this data);
* ``cluster=`` crashed with an ``IndexError`` whenever a row had a missing
  y or x, because the cluster ids were never filtered with the data.

Tolerance 1e-10 relative (observed <= 1e-12).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "rd_weights_R.json").read_text(encoding="utf-8"))
TOL = 1e-10

CASES = {
    "sharp_mserd": dict(y="y", weights="w"),
    "sharp_unweighted": dict(y="y"),
    "sharp_msetwo_hc1": dict(y="y", weights="w", bwselect="msetwo", vce="hc1"),
    "sharp_cerrd_p2": dict(y="y", weights="w", bwselect="cerrd", p=2),
    "sharp_covs": dict(y="y", weights="w", covs=["z1"]),
    "sharp_cluster": dict(y="y", weights="w", cluster="cl"),
    "sharp_hc3_uniform": dict(y="y", weights="w", vce="hc3", kernel="uniform"),
    "sharp_fixed_h": dict(y="y", weights="w", h=0.4, b=0.6),
    "fuzzy_mserd": dict(y="yf", fuzzy="d", weights="w"),
    "fuzzy_comb2": dict(y="yf", fuzzy="d", weights="w", bwselect="msecomb2"),
    "fuzzy_comb2_unweighted": dict(y="yf", fuzzy="d", bwselect="msecomb2"),
}


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "rd_weights_data.csv")


def _fit(data, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.rdrobust(data, x="x", **kw)


@pytest.mark.parametrize("key", list(CASES))
def test_matches_R_rdrobust(data, key):
    mi = _fit(data, **CASES[key]).model_info
    ref = R[key]
    h = mi["bandwidth_h"]
    h = h if isinstance(h, tuple) else (h, h)
    got = [
        mi["conventional"]["estimate"],
        mi["robust"]["estimate"],
        mi["conventional"]["se"],
        mi["robust"]["se"],
        h[0],
        h[1],
    ]
    want = [ref["coef"][0], ref["coef"][2], ref["se"][0], ref["se"][2], *ref["h"]]
    np.testing.assert_allclose(got, want, rtol=TOL)


def test_effective_n_counts_positive_weight_rows(data):
    mi = _fit(data, **CASES["sharp_mserd"]).model_info
    assert [mi["n_effective_left"], mi["n_effective_right"]] == R["sharp_mserd"]["N_h"]


def test_unit_weights_reproduce_unweighted_fit(data):
    a = _fit(data, y="y")
    b = _fit(data.assign(one=1.0), y="y", weights="one")
    assert a.estimate == b.estimate and a.se == b.se
    assert a.model_info["bandwidth_h"] == b.model_info["bandwidth_h"]


def test_weights_scale_invariant(data):
    a = _fit(data, y="y", weights="w")
    b = _fit(data.assign(w10=10 * data["w"]), y="y", weights="w10")
    assert b.estimate == pytest.approx(a.estimate, rel=1e-12)
    assert b.se == pytest.approx(a.se, rel=1e-12)


def test_negative_weights_and_weighted_bootstrap_are_refused(data):
    bad = data.assign(w=data["w"] - 1.0)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="negative"):
        _fit(bad, y="y", weights="w")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="bootstrap"):
        _fit(data, y="y", weights="w", bootstrap="rbc")


def test_missing_rows_with_cluster_no_longer_crash(data):
    holes = data.copy()
    holes.loc[holes.index[:7], "y"] = np.nan
    holes.loc[holes.index[7:10], "cl"] = np.nan
    got = _fit(holes, y="y", weights="w", cluster="cl")
    ref = _fit(holes.iloc[10:], y="y", weights="w", cluster="cl")
    assert got.estimate == pytest.approx(ref.estimate, rel=1e-13)
    assert got.se == pytest.approx(ref.se, rel=1e-13)


def test_cct_delegation_forwards_weights(data):
    pytest.importorskip("rdrobust")
    r = _fit(data, y="y", weights="w", bwselect="cct")
    assert r.model_info["robust"]["estimate"] == pytest.approx(
        R["sharp_mserd"]["coef"][2], rel=1e-8
    )
