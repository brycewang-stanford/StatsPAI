"""Local projections (Jordà 2005) tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.timeseries.local_projections import local_projections


@pytest.fixture(scope="module")
def ar_dgp():
    rng = np.random.default_rng(0)
    n = 300
    eps = rng.standard_normal(n)
    shock = np.zeros(n)
    shock[0] = rng.standard_normal()
    for t in range(1, n):
        shock[t] = 0.3 * shock[t - 1] + rng.standard_normal()
    y = np.zeros(n)
    for t in range(4, n):
        y[t] = (
            0.3 * y[t - 1]
            + 1.0 * shock[t]
            + 0.7 * shock[t - 1]
            + 0.4 * shock[t - 2]
            + 0.1 * shock[t - 3]
            + 0.5 * eps[t]
        )
    return pd.DataFrame({"y": y, "shock": shock})


def test_irf_recovers_impact(ar_dgp):
    res = local_projections(ar_dgp, outcome="y", shock="shock", horizons=8)
    # impact response (h=0) should be ≈ 1.0
    assert abs(res.irf[0] - 1.0) < 0.2


def test_irf_decays_after_peak(ar_dgp):
    res = local_projections(ar_dgp, outcome="y", shock="shock", horizons=12)
    # After horizon 4 the response should shrink toward zero
    assert abs(res.irf[8]) < 0.3
    assert abs(res.irf[11]) < 0.3


def test_confidence_interval_bracketing(ar_dgp):
    res = local_projections(ar_dgp, outcome="y", shock="shock", horizons=5, alpha=0.05)
    assert (res.ci_lower < res.irf).all()
    assert (res.irf < res.ci_upper).all()


def test_cumulative_mode_differs(ar_dgp):
    r1 = local_projections(ar_dgp, outcome="y", shock="shock", horizons=5)
    r2 = local_projections(
        ar_dgp, outcome="y", shock="shock", horizons=5, cumulative=True
    )
    # cumulative and level responses differ by construction
    assert np.any(np.abs(r1.irf - r2.irf) > 0.05)


def test_to_frame_shape(ar_dgp):
    res = local_projections(ar_dgp, outcome="y", shock="shock", horizons=4)
    df = res.to_frame()
    assert len(df) == 5
    assert set(df.columns) == {"horizon", "irf", "se", "ci_lower", "ci_upper", "n"}


def test_lpirfs_cholesky_matches_reference_fixture():
    """Pinned to lpirfs::lp_lin(unit Cholesky shock) on module 34 data."""
    rng = np.random.default_rng(42)
    n = 200
    y = np.zeros(n)
    x = rng.normal(0, 1, n)
    for t in range(1, n):
        y[t] = 0.6 * y[t - 1] + 0.5 * x[t - 1] + rng.normal(0, 0.5)
    df = pd.DataFrame({"y": y, "x": x})
    df["y_lag"] = df["y"].shift(1)
    df = df.iloc[1:].reset_index(drop=True)

    res = local_projections(
        df,
        outcome="y",
        shock="x",
        horizons=5,
        identification="lpirfs_cholesky",
        endog_order=["y", "x"],
    )

    np.testing.assert_allclose(
        res.irf,
        np.array(
            [
                0.0,
                0.461444592839222,
                0.299582531051504,
                0.148092245451433,
                -0.00543013383730529,
                -0.00324604091867858,
            ]
        ),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        res.se,
        np.array(
            [
                0.0,
                0.0431362326194262,
                0.0651216685321928,
                0.0691169132528259,
                0.0650565347230699,
                0.0746097068040731,
            ]
        ),
        atol=1e-12,
    )


def test_lpirfs_cholesky_rejects_direct_controls(ar_dgp):
    with pytest.raises(ValueError, match="controls are not supported"):
        local_projections(
            ar_dgp,
            outcome="y",
            shock="shock",
            controls=["shock"],
            identification="lpirfs_cholesky",
        )


def test_exported_at_sp_dot():
    import statspai as sp

    assert callable(sp.local_projections)
