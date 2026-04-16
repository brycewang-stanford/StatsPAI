"""Tests for BCH post-Lasso IV."""

import numpy as np
import pandas as pd
import pytest

import statspai.iv as iv


@pytest.fixture
def high_dim_dgp():
    """n=500, p=100 candidate instruments, only 5 relevant."""
    rng = np.random.default_rng(2026)
    n, p = 500, 100
    Z = rng.normal(size=(n, p))
    eps = rng.normal(size=n)
    pi_true = np.zeros(p)
    pi_true[:5] = [0.5, 0.4, 0.35, 0.3, 0.25]
    d = Z @ pi_true + 0.4 * eps + rng.normal(size=n, scale=0.3)
    y = 1 + 2.0 * d + eps
    df = pd.DataFrame({"y": y, "d": d, **{f"z{j+1}": Z[:, j] for j in range(p)}})
    return df, [f"z{j+1}" for j in range(p)]


class TestBCHPostLasso:
    def test_basic_recovers_true_beta(self, high_dim_dgp):
        df, zs = high_dim_dgp
        r = iv.bch_post_lasso_iv(y="y", endog="d", instruments=zs, data=df)
        assert abs(r.beta["d"] - 2.0) < 0.1
        assert r.n_selected >= 1
        assert r.n_selected <= len(zs)
        assert r.first_stage_f > 20  # strong on selected subset
        assert r.lambda_used > 0

    def test_outperforms_naive_2sls(self, high_dim_dgp):
        import statspai as sp
        df, zs = high_dim_dgp
        post = iv.bch_post_lasso_iv(y="y", endog="d", instruments=zs, data=df)
        naive = sp.ivreg("y ~ (d ~ " + "+".join(zs) + ")", data=df)
        # Post-Lasso should be closer to truth under many weak instruments
        assert abs(post.beta["d"] - 2.0) <= abs(naive.params["d"] - 2.0) + 0.05

    def test_lambda_formula(self):
        lam = iv.bch_lambda(n=1000, p=50, alpha=0.05, c=1.1)
        # Check monotonicity: λ ↑ in both n and p
        assert iv.bch_lambda(n=2000, p=50) > lam  # strictly increases with n
        assert iv.bch_lambda(n=1000, p=500) > lam  # increases with p

    def test_ensure_min_instruments(self, high_dim_dgp):
        df, zs = high_dim_dgp
        # Force aggressive α so LASSO selects nothing, then verify fallback
        r = iv.bch_post_lasso_iv(
            y="y", endog="d", instruments=zs, data=df,
            alpha=1e-10,  # essentially no LASSO selection
            c=50.0,       # enormous λ to starve LASSO
            ensure_min_instruments=3,
        )
        assert r.n_selected >= 3

    def test_result_summary_runs(self, high_dim_dgp):
        df, zs = high_dim_dgp
        r = iv.bch_post_lasso_iv(y="y", endog="d", instruments=zs, data=df)
        s = r.summary()
        assert "Belloni-Chen-Chernozhukov-Hansen" in s
        assert "Selected instruments" in s
