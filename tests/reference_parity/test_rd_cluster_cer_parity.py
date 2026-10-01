"""Clustered CER bandwidths against rdrobust 4.0.0 (R) on committed bytes.

The coverage-error-rate bandwidth shrinks the MSE-optimal one by
``m ** (-p / ((3 + p) (3 + 2p)))``. With clustered data rdrobust takes
``m`` to be the number of clusters, counted on each side of the cutoff and
added; StatsPAI used the number of observations until 2026-10, which made
every clustered CER bandwidth too narrow. The discrepancy was found while
checking the ``rdbwselect`` translation against Stata 18, whose
``rdbwselect`` returns the same ``h_cerrd`` as R on these bytes
(0.23049687).

Reference values: ``_generate_rd_cluster_cer_R.R`` on
``_fixtures/rd_cluster_cer.csv`` (1,200 observations, 60 clusters, each
present on both sides of the cutoff at 0.1). Tolerance 1e-8 on numbers R
printed to ten decimals.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import statspai as sp

FIXTURE = Path(__file__).parent / "_fixtures" / "rd_cluster_cer.csv"

#: (h_left, h_right, b_left, b_right) from rdrobust 4.0.0.
R_CLUSTERED = {
    "cerrd": (0.2304968741, 0.2304968741, 0.4493433897, 0.4493433897),
    "certwo": (0.3532153556, 0.2093118271, 0.6366706045, 0.4106679076),
    "cersum": (0.2434204330, 0.2434204330, 0.4375917015, 0.4375917015),
    "mserd": (0.2928357549, 0.2928357549, 0.4493433897, 0.4493433897),
}
R_UNCLUSTERED_CERRD = (0.2454704818, 0.2454704818, 0.5244884935, 0.5244884935)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


@pytest.mark.parametrize("bwselect", sorted(R_CLUSTERED))
def test_clustered_bandwidths_match_r(frame, bwselect):
    out = sp.rdbwselect(
        frame, y="y", x="x", c=0.1, covs=["z"], cluster="g", bwselect=bwselect
    ).iloc[0]
    got = (out["h_left"], out["h_right"], out["b_left"], out["b_right"])
    assert got == pytest.approx(R_CLUSTERED[bwselect], abs=1e-8)


def test_cer_factor_counts_clusters_on_each_side(frame):
    mse = sp.rdbwselect(
        frame, y="y", x="x", c=0.1, covs=["z"], cluster="g", bwselect="mserd"
    ).iloc[0]["h_left"]
    cer = sp.rdbwselect(
        frame, y="y", x="x", c=0.1, covs=["z"], cluster="g", bwselect="cerrd"
    ).iloc[0]["h_left"]
    g_left = frame.loc[frame["x"] < 0.1, "g"].nunique()
    g_right = frame.loc[frame["x"] >= 0.1, "g"].nunique()
    assert (g_left, g_right) == (60, 60)
    # p = 1: exponent -1/20, on 120 clusters and not on 1,200 observations.
    assert cer / mse == pytest.approx((g_left + g_right) ** (-1 / 20), abs=1e-12)


def test_unclustered_cer_bandwidth_is_unchanged(frame):
    out = sp.rdbwselect(frame, y="y", x="x", c=0.1, bwselect="cerrd").iloc[0]
    got = (out["h_left"], out["h_right"], out["b_left"], out["b_right"])
    assert got == pytest.approx(R_UNCLUSTERED_CERRD, abs=1e-8)


def test_rdrobust_on_a_clustered_cer_bandwidth_matches_r(frame):
    res = sp.rdrobust(
        frame, y="y", x="x", c=0.1, covs=["z"], cluster="g", bwselect="cerrd"
    )
    # R: bias-corrected coefficient 0.7784067538, robust SE 0.1322121777.
    assert res.estimate == pytest.approx(0.7784067538, abs=1e-8)
    assert res.se == pytest.approx(0.1322121777, abs=1e-8)
