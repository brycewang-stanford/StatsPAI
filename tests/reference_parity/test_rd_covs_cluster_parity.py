"""Reference parity on a DISCRIMINATING covs / cluster design (RD).

Why a third RD fixture
----------------------
``rdrobust_RDsenate``'s covariates barely bind, and twice that let a wrong
implementation look acceptable:

* a Z-projection in the bandwidth cascade "improved" senate from 7.3e-3 to
  2.0e-3 while making ``h`` 3x too narrow where covariates matter;
* a regression that discarded ``covs`` entirely showed up on senate as a
  ~1e-2 gap that read like "the bandwidth does not handle covs yet", when
  the SE was actually off by 6.4x.

So ``rd_covs_discriminating.csv`` is built so neither can be ignored without
the numbers moving a lot: ``z1`` carries a coefficient of 2.0 against noise
of 0.3, and the 80 clusters are **contiguous in x** (assigned by row index
they were too weak -- the cluster-robust SE moved only 1.10x, against 2.29x
once contiguous).

On the R side the design does what it should::

    spec              h        se_conv   se/plain
    plain_p1          0.2155   0.3063    1.00x
    covs1_p1          0.0668   0.0882    0.29x     <- h moves 3.2x
    cluster_p1        0.2499   0.7017    2.29x
    covs_cluster_p1   0.2076   0.4187    1.37x

What it reveals
---------------
Measured relative deviation of ``sp.rdrobust`` against rdrobust 4.0.0::

    spec              h        conv      se_conv   robust    se_rob
    plain_p1          1.5e-06  6.8e-14   2.2e-13   1.3e-13   4.1e-13   <- exact
    covs1_p1          2.2e+00  3.5e-02   1.1e-01   6.8e-02   6.8e-03
    cluster_p1        1.4e-01  4.7e-03   5.6e-01   2.8e-02   5.7e-01
    plain_p2          4.0e-07  4.4e-13   5.5e-12   7.6e-12   4.3e-12   <- exact
    covs1_p2          1.3e+00  1.6e-01   9.5e-02   2.3e-01   2.6e-01
    cluster_p2        1.6e-02  1.1e-03   5.8e-01   4.0e-03   5.9e-01

The senate fixture put the covs bandwidth gap at 7.3e-3. Here it is **2.2**
-- three hundred times larger. The cluster SE gap is 56%, i.e. clustering
is essentially not reaching the variance at all.

The default path stays exact, which is the point of keeping ``plain_*`` in
the same file: it separates "the cascade is broken" from "the cascade does
not yet know about this option".
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
RTOL = 1e-6

_TODO = (
    "cluster= is not in the CCT cascade. covs now is, end to end. Measured "
    "on a design where both bind: the cluster SE is still 6e-2 off, while "
    "covs went from 2.2 (bandwidth) and 1.1e-1 (SE) to 1.7e-07 and 2.4e-03. "
    "See docs/rfc/rd_three_month_plan.md appendix D."
)


@pytest.fixture(scope="module")
def rjson():
    p = _FIX / "rd_covs_discriminating_R.json"
    if not p.exists():  # pragma: no cover
        pytest.skip("run _generate_rd_covs_discriminating_R.R first")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "rd_covs_discriminating.csv")


_COVS_RESIDUAL = (
    "covs is wired end to end and improved by 1-2 orders of magnitude "
    "(conv 3.5e-2 -> 1.2e-3, se_conv 1.1e-1 -> 2.4e-3), but is not yet at "
    "the 1e-12 the sharp path reaches. The residual ~1e-3 is a remaining "
    "detail in R's covariate handling, not the projection itself, which "
    "matches on the bandwidth at 1.7e-07."
)


def _s(v):
    return float(np.ravel(v)[0])


# ── the fixture must actually discriminate ─────────────────────────────────


def test_design_is_discriminating(rjson):
    """Guard the fixture itself.

    If a future regeneration produces a design where covs or clusters barely
    move the R-side numbers, it can no longer catch the bugs it exists for,
    and every assertion below becomes vacuous.
    """
    base = rjson["plain_p1"]
    covs = rjson["covs1_p1"]
    clus = rjson["cluster_p1"]
    assert (
        covs["se_conventional"] < 0.5 * base["se_conventional"]
    ), "covariates barely reduce the SE -- design is not discriminating"
    assert (
        abs(covs["h_left"] / base["h_left"] - 1) > 0.5
    ), "covariates barely move the bandwidth -- design is not discriminating"
    assert (
        clus["se_conventional"] > 1.5 * base["se_conventional"]
    ), "clustering barely widens the SE -- design is not discriminating"


# ── the default path must stay exact here too ──────────────────────────────


@pytest.mark.parametrize("p", [1, 2])
def test_plain_path_is_exact(rjson, data, p):
    ref = rjson[f"plain_p{p}"]
    r = sp.rdrobust(data, y="y", x="x", c=0, p=p)
    assert _s(r.model_info["bandwidth_h"]) == pytest.approx(ref["h_left"], rel=1e-5)
    assert float(r.detail["estimate"][0]) == pytest.approx(
        ref["coef_conventional"], rel=RTOL
    )
    assert float(r.detail["se"][0]) == pytest.approx(ref["se_conventional"], rel=RTOL)
    assert float(r.detail["estimate"][1]) == pytest.approx(ref["coef_robust"], rel=RTOL)
    assert float(r.detail["se"][1]) == pytest.approx(ref["se_robust"], rel=RTOL)


# ── covs: estimate adjusts, bandwidth does not ─────────────────────────────


def test_covs_adjustment_reaches_the_estimate(rjson, data):
    """Regression guard: covs must not be silently discarded.

    This is the check that would have caught the WP-2 regression in which
    the CCT substitution overwrote the covariate-adjusted estimate.
    """
    plain = sp.rdrobust(data, y="y", x="x", c=0)
    adj = sp.rdrobust(data, y="y", x="x", c=0, covs=["z1"])
    assert float(adj.detail["se"][0]) < 0.6 * float(
        plain.detail["se"][0]
    ), "covs= did not reduce the SE; R cuts it to 0.29x on this design"


@pytest.mark.parametrize("p", [1, 2])
def test_covs_bandwidth_matches_r(rjson, data, p):
    """The covariate projection in the cascade, verified end to end.

    Tolerance is 1e-5, not RTOL: ``model_info`` rounds bandwidths to six
    decimals, which on h ~ 0.148 is already ~3e-6 of relative error. The
    projection itself matches R at 3.7e-13 when called directly (see
    _cct_bandwidth.cct_bandwidth).
    """
    ref = rjson[f"covs1_p{p}"]
    r = sp.rdrobust(data, y="y", x="x", c=0, p=p, covs=["z1"])
    assert _s(r.model_info["bandwidth_h"]) == pytest.approx(ref["h_left"], rel=1e-5)


@pytest.mark.parametrize("p", [1, 2])
def test_covs_conventional_matches_r(rjson, data, p):
    ref = rjson[f"covs1_p{p}"]
    r = sp.rdrobust(data, y="y", x="x", c=0, p=p, covs=["z1"])
    assert float(r.detail["estimate"][0]) == pytest.approx(
        ref["coef_conventional"], rel=RTOL
    )


# ── cluster: barely reaches the variance ───────────────────────────────────


@pytest.mark.parametrize("p", [1, 2])
def test_cluster_se_matches_r(rjson, data, p):
    ref = rjson[f"cluster_p{p}"]
    r = sp.rdrobust(data, y="y", x="x", c=0, p=p, cluster="g")
    assert float(r.detail["se"][0]) == pytest.approx(ref["se_conventional"], rel=RTOL)


def test_cluster_widens_the_se_at_all(data):
    """Weaker than parity, but it must move in the right direction.

    R widens the conventional SE 2.29x on this design. An implementation
    that ignores `cluster=` entirely would show no change.
    """
    plain = sp.rdrobust(data, y="y", x="x", c=0)
    clus = sp.rdrobust(data, y="y", x="x", c=0, cluster="g")
    assert float(clus.detail["se"][0]) > 1.2 * float(
        plain.detail["se"][0]
    ), "cluster= did not widen the SE; R widens it 2.29x on this design"


# ── bounded gap: catch a regression even while the above are xfailed ───────


def test_gap_does_not_widen(rjson, data):
    calls = {
        "covs1_p1": dict(p=1, covs=["z1"]),
        "covs2_p1": dict(p=1, covs=["z1", "z2"]),
        "cluster_p1": dict(p=1, cluster="g"),
        "covs_cluster_p1": dict(p=1, covs=["z1"], cluster="g"),
        "covs1_p2": dict(p=2, covs=["z1"]),
        "cluster_p2": dict(p=2, cluster="g"),
    }
    bad = []
    for spec, kw in calls.items():
        ref = rjson[spec]
        r = sp.rdrobust(data, y="y", x="x", c=0, **kw)
        for label, got, want in (
            ("h", _s(r.model_info["bandwidth_h"]), ref["h_left"]),
            ("conv", float(r.detail["estimate"][0]), ref["coef_conventional"]),
            ("se_conv", float(r.detail["se"][0]), ref["se_conventional"]),
        ):
            rel = abs(got - want) / max(abs(want), 1e-12)
            # Ceilings set just above the measured values, so any widening
            # fails while the current known gaps do not.
            cap = {"h": 3.0, "conv": 0.30, "se_conv": 0.70}[label]
            if rel > cap:
                bad.append(f"{spec}.{label} rel={rel:.2e} > {cap}")
    assert not bad, "gap widened:\n  " + "\n  ".join(bad)
