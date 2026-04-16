"""Tests for Bayesian IV."""

import numpy as np
import pandas as pd
import pytest

import statspai.iv as iv


@pytest.fixture
def strong_dgp():
    rng = np.random.default_rng(0)
    n = 1000
    z, eps = rng.normal(size=n), rng.normal(size=n)
    d = 0.8 * z + 0.5 * eps + rng.normal(size=n, scale=0.3)
    y = 1 + 2.0 * d + eps
    return pd.DataFrame({"y": y, "d": d, "z": z})


@pytest.fixture
def weak_dgp():
    rng = np.random.default_rng(0)
    n = 1000
    z, eps = rng.normal(size=n), rng.normal(size=n)
    d = 0.03 * z + 0.8 * eps + rng.normal(size=n, scale=0.3)
    y = 1 + 2.0 * d + eps
    return pd.DataFrame({"y": y, "d": d, "z": z})


class TestBayesianIV:
    def test_strong_iv_covers_truth(self, strong_dgp):
        r = iv.bayesian_iv(
            y="y", endog="d", instruments=["z"], data=strong_dgp,
            n_draws=5000, n_warmup=2000, random_state=0,
        )
        assert r.hpd_lower <= 2.05  # truth within reasonable tolerance
        assert r.hpd_upper >= 2.0
        assert abs(r.posterior_mean - 2.0) < 0.2
        assert r.posterior_sd < 0.2

    def test_weak_iv_much_wider(self, strong_dgp, weak_dgp):
        r_s = iv.bayesian_iv(y="y", endog="d", instruments=["z"], data=strong_dgp,
                             n_draws=3000, random_state=0)
        r_w = iv.bayesian_iv(y="y", endog="d", instruments=["z"], data=weak_dgp,
                             n_draws=3000, random_state=0)
        assert r_w.posterior_sd > r_s.posterior_sd * 2

    def test_to_frame(self, strong_dgp):
        r = iv.bayesian_iv(y="y", endog="d", instruments=["z"], data=strong_dgp,
                           n_draws=500, random_state=0)
        df = r.to_frame()
        assert df.shape == (500, 1)
        assert "beta" in df.columns
