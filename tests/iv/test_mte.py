"""Smoke tests for sp.iv.mte (BMW 2017 polynomial MTR)."""

import numpy as np
import pandas as pd
import pytest

import statspai.iv as iv


def _mte_dgp(n: int = 6000, seed: int = 7):
    """Essential-heterogeneity DGP with known analytic MTE."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = rng.normal(size=n)
    # Propensity score P(Z, X) = logit(0.5*z + 0.3*x)
    eta = 0.5 * z + 0.3 * x
    p = 1 / (1 + np.exp(-eta))
    U = rng.uniform(size=n)
    D = (U < p).astype(float)
    # m_0(u, x) = 1 + 0.4*u + 0.3*x
    # m_1(u, x) = 3 - 0.6*u + 0.3*x
    # True MTE = m_1 - m_0 = 2 - u (decreasing)  -> ATE = 1.5 at any X
    eps = rng.normal(size=n, scale=0.3)
    y = (1 + 0.4 * U + 0.3 * x) + D * ((3 - 0.6 * U) - (1 + 0.4 * U)) + eps
    return pd.DataFrame({"y": y, "d": D, "z": z, "x": x}), 1.5  # true ATE


class TestMTE:
    def test_basic_runs(self):
        df, ate_true = _mte_dgp()
        r = iv.mte(y="y", treatment="d", instruments=["z"], exog=["x"], data=df,
                   poly_degree=2)
        assert r.poly_degree == 2
        assert r.mte_curve.shape[1] >= 3
        # ATE estimate should be in right ballpark (wide tolerance — small n,
        # support may not cover all of (0, 1))
        assert abs(r.ate - ate_true) < 2.0

    def test_mte_declining(self):
        df, _ = _mte_dgp(n=8000)
        r = iv.mte(y="y", treatment="d", instruments=["z"], exog=["x"], data=df,
                   poly_degree=2)
        # MTE should decline over the support
        mte_start = r.mte_curve["mte"].iloc[0]
        mte_end = r.mte_curve["mte"].iloc[-1]
        assert mte_start > mte_end  # declining

    def test_binary_treatment_required(self):
        df, _ = _mte_dgp()
        df["d_bad"] = df["d"] * 2  # non-binary
        with pytest.raises(ValueError, match="binary"):
            iv.mte(y="y", treatment="d_bad", instruments=["z"], exog=["x"], data=df)

    def test_late_reference_returned(self):
        df, _ = _mte_dgp()
        r = iv.mte(y="y", treatment="d", instruments=["z"], exog=["x"], data=df,
                   poly_degree=1)
        assert np.isfinite(r.late_2sls)
