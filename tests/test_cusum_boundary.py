"""Correctness tests for the recursive-residual CUSUM test (sp.cusum_test).

Regression guard for the fix that replaced the flat constant boundary
(``max|S_t| > 1.358``, a Kolmogorov-bridge value that rejected ~32% of stable
series at the nominal 5% level) with the Brown-Durbin-Evans (1975) *diverging
linear* boundary ``±a·(1 + 2(t-k)/(T-k))``, ``a = 0.948`` at 5%.

Reference: Brown, R.L., Durbin, J. & Evans, J.M. (1975), JRSS-B 37(2), 149-192.
"""

import numpy as np
import pandas as pd

from statspai.timeseries.structural_break import cusum_test


def test_cusum_size_under_h0():
    """Stable (H0) data must reject near alpha, not ~32%."""
    rng = np.random.default_rng(2024)
    n, reps, alpha = 150, 300, 0.05
    rej = 0
    for _ in range(reps):
        x = rng.normal(0, 1, n)
        y = 1.0 + 2.0 * x + rng.normal(0, 1, n)   # stable parameters
        res = cusum_test(pd.DataFrame({"y": y, "x": x}), y="y", x=["x"],
                         alpha=alpha)
        rej += int(res["reject"])
    fp = rej / reps
    # Buggy constant-boundary version gave ~0.32; BDE is ~0.04-0.05.
    assert fp < 0.12, f"H0 rejection rate {fp:.3f} too high (oversized test)"


def test_cusum_power_on_late_mean_shift():
    rng = np.random.default_rng(7)
    n = 200
    x = rng.normal(0, 1, n)
    y = 1.0 + 2.0 * x + rng.normal(0, 1, n)
    y[n // 2:] += 3.0
    res = cusum_test(pd.DataFrame({"y": y, "x": x}), y="y", x=["x"])
    assert res["reject"] is True


def test_cusum_boundary_is_linear_and_diverging():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 120)
    y = 1.0 + rng.normal(0, 1, 120)
    res = cusum_test(pd.DataFrame({"y": y, "x": x}), y="y", x=["x"], alpha=0.05)
    b = np.asarray(res["boundary"])
    assert res["boundary_coefficient"] == 0.948
    # Boundary increases strictly (diverging) and matches a*(1+2s).
    assert np.all(np.diff(b) > 0)
    m = len(b)
    s = np.arange(1, m + 1) / m
    np.testing.assert_allclose(b, 0.948 * (1 + 2 * s), rtol=1e-12)
    # Endpoints span a -> 3a.
    assert abs(b[-1] - 3 * 0.948) < 1e-9


def test_cusum_alpha_changes_boundary_coefficient():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 100)
    y = 1.0 + rng.normal(0, 1, 100)
    df = pd.DataFrame({"y": y, "x": x})
    a10 = cusum_test(df, y="y", x=["x"], alpha=0.10)["boundary_coefficient"]
    a05 = cusum_test(df, y="y", x=["x"], alpha=0.05)["boundary_coefficient"]
    a01 = cusum_test(df, y="y", x=["x"], alpha=0.01)["boundary_coefficient"]
    assert (a10, a05, a01) == (0.850, 0.948, 1.143)


def test_cusum_returns_expected_keys():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"y": 1.0 + rng.normal(0, 1, 80)})
    res = cusum_test(df, y="y")
    for key in ("cusum", "boundary", "boundary_coefficient", "max_cusum",
                "critical_value", "reject", "n_obs"):
        assert key in res
