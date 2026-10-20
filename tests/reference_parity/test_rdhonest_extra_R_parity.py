"""Honest RD branches not covered by the older RDHonest fixtures.

Black-box reference: R ``RDHonest`` 1.0.1.9000 on committed CSVs
(``_fixtures/_generate_rdhonest_extra_R.R``). Cells: selected bandwidths with
the uniform kernel (criterion minimised over the grid of data distances) and
the Epanechnikov kernel (preliminary fit always triangular), the OCI
criterion, the Taylor class, ``alpha = 0.10``, and the fuzzy default
bandwidth with clusters and with the other two kernels.

Selected bandwidths come from a numerical search over a flat criterion, so
they are held at rel 1e-6 (observed <= 1.3e-7), as in
``test_rdhonest_parity.py``; the uniform kernel picks a data point exactly.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp
from statspai.rd._rdhonest import cv_bias

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "rdhonest_extra_R.json").read_text(encoding="utf-8"))
D = {
    "curved": pd.read_csv(_FIX / "rdhonest_curved.csv", float_precision="round_trip"),
    "lee08": pd.read_csv(_FIX / "rdhonest_lee08.csv", float_precision="round_trip"),
}
FRD = pd.read_csv(_FIX / "rd_iv_frd.csv", float_precision="round_trip")
_M = {"curved": 3.0, "lee08": 0.02}
TOL = 1e-6

SHARP = {
    f"{ds}_{k}_{crit}": (ds, dict(M=_M[ds], kernel=k, opt_criterion=crit.lower()))
    for ds in ("curved", "lee08")
    for k in ("uniform", "epanechnikov")
    for crit in ("MSE", "FLCI")
}
SHARP["curved_OCI"] = ("curved", dict(M=3.0, opt_criterion="oci"))
SHARP["curved_taylor"] = ("curved", dict(M=3.0, sclass="T"))
SHARP["lee08_alpha10_FLCI"] = ("lee08", dict(M=0.02, alpha=0.1, opt_criterion="flci"))


@pytest.mark.parametrize("cell", sorted(SHARP))
def test_sharp_selected_bandwidth_matches_rdhonest(cell):
    ds, kw = SHARP[cell]
    ref = R[cell]
    res = sp.rd_honest(D[ds], y="y", x="x", c=0, **kw)
    mi = res.model_info
    assert mi["bandwidth"] == pytest.approx(ref["bandwidth"], rel=TOL)
    assert res.estimate == pytest.approx(ref["estimate"], rel=TOL)
    assert res.se == pytest.approx(ref["std.error"], rel=TOL)
    assert mi["bias_bound"] == pytest.approx(ref["maximum.bias"], rel=TOL)
    assert res.ci[0] == pytest.approx(ref["conf.low"], rel=TOL)
    assert res.ci[1] == pytest.approx(ref["conf.high"], rel=TOL)


@pytest.mark.parametrize(
    "cell, kw",
    [
        ("frd_cluster_default", dict(cluster="g")),
        ("frd_epanechnikov_default", dict(kernel="epanechnikov")),
        ("frd_uniform_default", dict(kernel="uniform")),
    ],
)
def test_fuzzy_default_bandwidth_matches_rdhonest(cell, kw):
    ref = R[cell]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.rd_bias_aware_fuzzy(FRD, y="y", x="x", fuzzy="d", **kw)
    b = res.model_info["bias_aware"]
    hon = b["rdhonest"]
    assert b["bandwidth"] == pytest.approx(ref["bandwidth"], rel=TOL)
    assert b["M_y"] == pytest.approx(ref["M.rf"], rel=TOL)
    assert b["M_d"] == pytest.approx(ref["M.fs"], rel=TOL)
    for k in ("estimate", "std.error", "maximum.bias", "conf.low", "conf.high"):
        assert hon[k] == pytest.approx(ref[k], rel=TOL), k


def test_critical_value_identity_including_extreme_levels():
    """``P(|N(t,1)| > cv) = alpha`` for every level, including alpha near 0
    and near 1 (the p-value inversion asks for alpha = 1 - 1e-12)."""
    from scipy import stats

    for t in (0.0, 0.3, 2.0, 8.0):
        for alpha in (1e-9, 0.05, 0.5, 1 - 1e-12):
            cv = cv_bias(t, alpha)
            tail = stats.norm.sf(cv - t) + stats.norm.cdf(-cv - t)
            assert tail == pytest.approx(alpha, rel=1e-9, abs=1e-15)
    with pytest.raises(ValueError, match="alpha"):
        cv_bias(1.0, 1.0)
