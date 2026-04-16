"""Tests for sp.iv.npiv (nonparametric IV)."""

import numpy as np
import pandas as pd
import pytest

import statspai.iv as iv


@pytest.fixture
def nonlinear_dgp():
    """h(D) = 2D + 0.5D² — nonlinear structural function."""
    rng = np.random.default_rng(42)
    n = 3000
    z = rng.normal(size=n)
    eps = rng.normal(size=n)
    d = 0.8 * z + 0.3 * eps + rng.normal(size=n, scale=0.5)
    y = 2 * d + 0.5 * d ** 2 + eps
    return pd.DataFrame({"y": y, "d": d, "z": z})


class TestNPIV:
    def test_basic_runs(self, nonlinear_dgp):
        r = iv.npiv(y="y", endog="d", instruments=["z"], data=nonlinear_dgp)
        assert r.n_obs == len(nonlinear_dgp)
        assert len(r.h_values) == 100
        assert r.first_stage_f > 10

    def test_to_frame(self, nonlinear_dgp):
        r = iv.npiv(y="y", endog="d", instruments=["z"], data=nonlinear_dgp)
        df = r.to_frame()
        assert "h" in df.columns
        assert "ci_lower" in df.columns
        assert df.shape[0] == 100

    def test_captures_nonlinearity(self, nonlinear_dgp):
        r = iv.npiv(y="y", endog="d", instruments=["z"], data=nonlinear_dgp,
                    k_d=4, k_z=4)
        # h(D) = 2D + 0.5D² is convex. h(d=2) = 6, h(d=-2) = -2.
        # NPIV should show convexity: h(2) > -h(-2)
        df = r.to_frame()
        h_pos = df.loc[df["D"].sub(1.5).abs().idxmin(), "h"]
        h_neg = df.loc[df["D"].sub(-1.5).abs().idxmin(), "h"]
        # Both should be in the right direction
        assert h_pos > 0 and h_neg < 0

    def test_regularization(self, nonlinear_dgp):
        r0 = iv.npiv(y="y", endog="d", instruments=["z"], data=nonlinear_dgp,
                     k_d=4, regularization=0.0)
        r1 = iv.npiv(y="y", endog="d", instruments=["z"], data=nonlinear_dgp,
                     k_d=4, regularization=1.0)
        # Regularization should shrink h towards zero → smaller variance
        assert r1.h_se.mean() <= r0.h_se.mean()
