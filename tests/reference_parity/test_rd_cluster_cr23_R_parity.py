"""``sp.rdrobust`` cluster variances CR1 / CR2 / CR3 vs R rdrobust 4.0.0.

R rdrobust 4.0.0 (GPL-3, black box: documented arguments and outputs only)
offers ``vce='cr1' / 'cr2' / 'cr3'`` with ``cluster=`` and maps ``hc2`` /
``hc3`` onto CR2 / CR3 when clusters are given. StatsPAI used HC2 / HC3
scaled residuals with the CR1 small-sample factor instead, 7% to 18% away
on this design. The CR2 / CR3 residuals are ``(I - H_gg)^(-1/2)`` /
``(I - H_gg)^(-1)`` times the cluster's residuals, ``H_gg`` the cluster
block of the weighted hat matrix of the same fit, no small-sample factor
(``rd/_cct_side.py::cluster_adjusted_residuals``).

Fixture: ``_fixtures/_generate_rd_cluster_cr23_R.R`` on
``rd_covs_discriminating.csv``; the fuzzy treatment and the weights are
deterministic functions of ``z2`` rebuilt below.

Tolerance: 1e-9 relative (closed forms on both sides; observed <= 3e-11,
bandwidths 1e-5 because ``model_info`` rounds them). One cell is looser:
the robust SE of CR2 with ``h > b`` (9e-10 observed, budget 1e-8). There the
union window holds rows outside the ``b`` fit, ``H_gg`` is not similar to
a symmetric matrix, and its inverse square root is ill-conditioned; our
closed form and a general matrix square root (scipy ``sqrtm``) both
satisfy ``A A (I - H) = I`` to 5e-15 yet differ by 1e-9, which is where R
lands (CR3, an ordinary inverse, agrees to 1e-14).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import StatsPAIWarning

_FIX = pathlib.Path(__file__).parent / "_fixtures"
RTOL = 1e-9
RTOL_BW_ROUNDED = 1e-5

_CASES = {
    "fixed_hb": dict(p=1, h=0.5, b=0.5),
    "fixed_hneb": dict(p=1, h=0.4, b=0.7),
    "h_gt_b": dict(p=1, h=0.7, b=0.4),
    "covs": dict(p=1, h=0.5, b=0.6, covs=["z1"]),
    "fuzzy": dict(p=1, h=0.5, b=0.6, fuzzy="t"),
    "weights": dict(p=1, h=0.5, b=0.6, weights="wt"),
    "auto_p1": dict(p=1),
    "auto_p2": dict(p=2),
    "auto_covs": dict(p=1, covs=["z1"]),
    "auto_uniform": dict(p=1, kernel="uniform"),
}


@pytest.fixture(scope="module")
def rjson():
    return json.loads((_FIX / "rd_cluster_cr23_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    d = pd.read_csv(_FIX / "rd_covs_discriminating.csv")
    d["t"] = (((d.x >= 0) & (d.z2 > -0.8)) | ((d.x < 0) & (d.z2 > 0.8))).astype(float)
    d["wt"] = 1 + 0.5 * np.tanh(d.z2)
    return d


def _fit(data, vce, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.rdrobust(data, y="y", x="x", c=0, vce=vce, cluster="g", **kw)


def test_the_fixture_tells_cr1_cr2_cr3_apart(rjson):
    # Guard only: a CR1 implementation must not pass the CR2 / CR3 cells.
    for tag in ("fixed_hb", "auto_p1"):
        se = [rjson[f"{v}_{tag}"]["se_conventional"] for v in ("cr1", "cr2", "cr3")]
        assert se[0] * 1.02 < se[1] and se[1] * 1.02 < se[2], se


@pytest.mark.parametrize("vce", ["cr1", "cr2", "cr3"])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_matches_r(rjson, data, vce, case):
    ref = rjson[f"{vce}_{case}"]
    assert ref["vce_type"] == vce.upper()
    res = _fit(data, vce, **_CASES[case])
    rtol_rb = 1e-8 if (vce, case) == ("cr2", "h_gt_b") else RTOL
    est, se = res.detail["estimate"], res.detail["se"]
    assert float(est.iloc[0]) == pytest.approx(ref["coef_conventional"], rel=RTOL)
    assert float(est.iloc[1]) == pytest.approx(ref["coef_robust"], rel=RTOL)
    assert float(se.iloc[0]) == pytest.approx(ref["se_conventional"], rel=RTOL)
    assert float(se.iloc[1]) == pytest.approx(ref["se_robust"], rel=rtol_rb)
    h = float(np.ravel(res.model_info["bandwidth_h"])[0])
    assert h == pytest.approx(ref["h_left"], rel=RTOL_BW_ROUNDED)


@pytest.mark.parametrize("hc,cr", [("hc2", "cr2"), ("hc3", "cr3")])
def test_hc_with_clusters_is_the_cluster_kind(rjson, data, hc, cr):
    """R maps hc2 / hc3 + cluster to CR2 / CR3 (with a warning); so do we."""
    with pytest.warns(StatsPAIWarning, match=f"CR{cr[2]}"):
        res = sp.rdrobust(data, y="y", x="x", c=0, p=1, vce=hc, cluster="g")
    ref = rjson[f"{hc}_auto_p1"]
    assert ref == rjson[f"{cr}_auto_p1"]
    assert float(res.detail["se"].iloc[0]) == pytest.approx(
        ref["se_conventional"], rel=RTOL
    )
    assert float(res.detail["se"].iloc[1]) == pytest.approx(ref["se_robust"], rel=RTOL)


@pytest.mark.parametrize("kind", ["2", "3"])
def test_singleton_clusters_reduce_to_hc(data, kind):
    """Identity: with one observation per cluster CR2 / CR3 are HC2 / HC3
    (the leverage block is the scalar leverage; no small-sample factor)."""
    d = data.assign(one=np.arange(len(data)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cr = sp.rdrobust(d, y="y", x="x", c=0, p=1, vce=f"cr{kind}", cluster="one")
        hc = sp.rdrobust(d, y="y", x="x", c=0, p=1, vce=f"hc{kind}")
    np.testing.assert_allclose(
        cr.detail["se"].to_numpy(), hc.detail["se"].to_numpy(), rtol=1e-12
    )


def test_cluster_kind_without_cluster_falls_back_to_hc(data):
    """R: ``vce='cr2'`` without a cluster variable falls back to hc2."""
    with pytest.warns(StatsPAIWarning, match="needs cluster"):
        cr = sp.rdrobust(data, y="y", x="x", c=0, p=1, h=0.5, vce="cr2")
    hc = sp.rdrobust(data, y="y", x="x", c=0, p=1, h=0.5, vce="hc2")
    np.testing.assert_array_equal(
        cr.detail["se"].to_numpy(), hc.detail["se"].to_numpy()
    )
