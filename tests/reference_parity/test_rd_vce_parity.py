"""Reference parity: ``sp.rdrobust(vce=..., cluster=...)`` vs R rdrobust 4.0.0.

Covers the three axes the CCT path did not previously reach:

* ``vce`` -- ``sp.rdrobust`` had no such parameter at all, so an R script
  using ``vce='hc3'`` did not port. All five kinds now match.
* ``cluster`` -- was adjusted in the variance but not in the bandwidth, and
  the promotion R applies (``nn`` + ``cluster`` -> ``cr1``, whose residuals
  are ``hc1``'s) was missing entirely, understating the SE ~10x.
* ``covs`` with either of the above.

All 17 cells agree with R to <= 5.5e-11 on ``h``, ``b``, both coefficients
and both standard errors. Three defects had to be fixed together to get
there, and each was invisible until the one before it was removed:

1. The **regularisation term** ``R`` in the cascade is a sandwich variance
   too. Leaving it on ``nn`` residuals while ``V`` used ``hc*`` left ``h``
   8e-3 off *with V and B both already exact to 1e-10* -- a discrepancy
   small enough to read as accumulated float error rather than a bug.
2. ``cluster`` + ``nn`` residuals: see above.
3. ``gamma`` (the covariate coefficient) is estimated by **pooling both
   sides** in the estimator -- ``ZWZ_p = ZWZ_p_l + ZWZ_p_r`` -- while the
   bandwidth cascade solves it per side. Solving per side in both places
   left the estimate 1.2e-3 off with the bandwidth already exact at 4e-14.
   Only after (1) and (2) landed was this the largest remaining term.

The fixture is the discriminating covs/cluster design, not
``rdrobust_RDsenate``: on senate the covariates barely bind and the covs
bandwidth defect showed up 300x smaller than it really was.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"

# Closed forms on both sides: parity is a formula question, not a
# tolerance question. The one relaxation is on quantities read back out of
# ``model_info``, which rounds bandwidths to six decimals.
RTOL = 1e-9
RTOL_BW_ROUNDED = 1e-5

_VCE_KINDS = ["nn", "hc0", "hc1", "hc2", "hc3"]


@pytest.fixture(scope="module")
def rjson():
    path = _FIX / "rd_vce_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_rd_vce_R.R to build rd_vce_R.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "rd_covs_discriminating.csv")


def _s(v):
    return float(np.ravel(v)[0])


def _check(res, ref, *, bw_rtol=RTOL_BW_ROUNDED):
    __tracebackhide__ = True
    assert _s(res.model_info["bandwidth_h"]) == pytest.approx(
        ref["h_left"], rel=bw_rtol
    ), "bandwidth h"
    assert float(res.detail["estimate"][0]) == pytest.approx(
        ref["coef_conventional"], rel=RTOL
    ), "conventional coefficient"
    assert float(res.detail["se"][0]) == pytest.approx(
        ref["se_conventional"], rel=RTOL
    ), "conventional SE"
    assert float(res.detail["estimate"][1]) == pytest.approx(
        ref["coef_robust"], rel=RTOL
    ), "bias-corrected coefficient"
    assert float(res.detail["se"][1]) == pytest.approx(
        ref["se_robust"], rel=RTOL
    ), "robust SE"


# ── the fixture must be able to tell right from wrong ──────────────────── #


def test_design_is_discriminating(rjson):
    """Guard: the cells must actually differ from one another.

    If ``vce`` or ``cluster`` had no effect on R's own output, every
    assertion below would pass against an implementation that ignored both.
    That is exactly how the covs bandwidth defect survived for two months
    against a fixture where covariates barely bound.
    """
    h_nn = rjson["nn_p1"]["h_left"]
    h_hc3 = rjson["hc3_p1"]["h_left"]
    assert abs(h_hc3 - h_nn) / h_nn > 1e-3, (
        f"vce does not move h in the reference ({h_nn} vs {h_hc3}); this "
        "fixture cannot detect a vce-blind bandwidth cascade"
    )

    se_plain = rjson["nn_p1"]["se_conventional"]
    se_clust = rjson["cluster_default_p1"]["se_conventional"]
    assert se_clust / se_plain > 1.5, (
        f"clustering barely moves the SE here ({se_plain} -> {se_clust}); "
        "the design needs stronger intra-cluster correlation to be a test"
    )

    h_plain = rjson["nn_p1"]["h_left"]
    h_covs = rjson["nn_covs_p1"]["h_left"]
    assert abs(h_covs - h_plain) / h_plain > 0.5, (
        f"covariates barely move h ({h_plain} -> {h_covs}); this is the "
        "senate failure mode -- a design where a covs-blind cascade passes"
    )


# ── vce ─────────────────────────────────────────────────────────────────── #


@pytest.mark.parametrize("vce", _VCE_KINDS)
@pytest.mark.parametrize("p", [1, 2])
def test_vce_matches_r(rjson, data, vce, p):
    res = sp.rdrobust(data, y="y", x="x", c=0, p=p, vce=vce)
    _check(res, rjson[f"{vce}_p{p}"])


@pytest.mark.parametrize("vce", ["nn", "hc0", "hc2", "hc3"])
def test_vce_with_covariates_matches_r(rjson, data, vce):
    res = sp.rdrobust(data, y="y", x="x", c=0, p=1, vce=vce, covs=["z1"])
    _check(res, rjson[f"{vce}_covs_p1"])


def test_vce_is_validated(data):
    """An unknown ``vce`` must raise, not fall through to a default.

    Silently estimating something other than what was asked for is the
    failure mode this whole suite exists to catch.
    """
    with pytest.raises(ValueError, match="vce"):
        sp.rdrobust(data, y="y", x="x", c=0, vce="hc9")


def test_vce_actually_changes_the_answer(data):
    """Property test, no reference needed: ``vce`` must not be inert."""
    ses = {
        v: float(sp.rdrobust(data, y="y", x="x", c=0, vce=v).detail["se"][0])
        for v in _VCE_KINDS
    }
    assert len(set(np.round(list(ses.values()), 12))) == len(
        _VCE_KINDS
    ), f"vce is being ignored somewhere: {ses}"
    # hc0 < hc1 < hc2 < hc3 is the ordering the corrections impose.
    hc = [ses[f"hc{i}"] for i in range(4)]
    assert hc == sorted(hc), f"hc corrections are not monotone: {hc}"


# ── cluster ─────────────────────────────────────────────────────────────── #


@pytest.mark.parametrize("p", [1, 2])
def test_cluster_matches_r(rjson, data, p):
    res = sp.rdrobust(data, y="y", x="x", c=0, p=p, cluster="g")
    _check(res, rjson[f"cluster_default_p{p}"])


def test_cluster_with_covariates_matches_r(rjson, data):
    res = sp.rdrobust(data, y="y", x="x", c=0, p=1, cluster="g", covs=["z1"])
    _check(res, rjson["cluster_covs_p1"])


def test_cluster_moves_the_bandwidth(rjson, data):
    """Clustering is not inference-only -- it shifts ``h`` as well.

    Pinned because it is counter-intuitive enough that a future reader
    might "fix" the cascade to ignore ``cluster``.
    """
    h_plain = _s(sp.rdrobust(data, y="y", x="x", c=0).model_info["bandwidth_h"])
    h_clust = _s(
        sp.rdrobust(data, y="y", x="x", c=0, cluster="g").model_info["bandwidth_h"]
    )
    assert h_clust != pytest.approx(h_plain, rel=1e-6)
    assert h_clust == pytest.approx(
        rjson["cluster_default_p1"]["h_left"], rel=RTOL_BW_ROUNDED
    )


def test_cluster_se_is_not_the_nn_se(rjson, data):
    """The promotion to ``cr1`` must happen; ``nn`` residuals here are ~10x low.

    This is the specific quantity that read 0.077 against R's 0.702 before
    the promotion was implemented.
    """
    ref = rjson["cluster_default_p1"]
    got = float(sp.rdrobust(data, y="y", x="x", c=0, cluster="g").detail["se"][0])
    assert got == pytest.approx(ref["se_conventional"], rel=RTOL)
    assert got > 2 * rjson["nn_p1"]["se_conventional"]


def test_dataset_matches_the_r_side(rjson, data):
    """Guard against blaming the estimator for a data mismatch."""
    assert len(data) == int(rjson["_meta"]["n"])
    assert data["g"].nunique() == int(rjson["_meta"]["n_clusters"])
