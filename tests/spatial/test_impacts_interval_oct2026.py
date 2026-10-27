"""sp.impacts admissible interval, SLX / SDEM / GNS impacts, ML search
interval (option scan round 5, 2026-10-27).

Each test states the correct behaviour and failed on 1.40.2.

R references: R 4.5.2, spdep 1.4.2, spatialreg 1.4.3, run as a black box
by ``tests/spatial/_impacts_reference.R`` on the data built below (the
script reads them from CSV written with ``%.17g``, so both sides see the
same doubles).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility, NumericalInstability


def _ring(n: int, w2: float) -> np.ndarray:
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i - 1) % n] = W[i, (i + 1) % n] = 1.0
        W[i, (i - 2) % n] = W[i, (i + 2) % n] = w2
    return W


def _row_std_data():
    """Ring with half weights at distance two, row-standardised: smallest
    eigenvalue -0.5, admissible interval (-2, 1). True rho / lambda -1.4."""
    rng = np.random.default_rng(20261027)
    rng.normal(size=300)  # draws used by a companion data set in the R script
    n = 120
    W = _ring(n, 0.5)
    Wn = W / W.sum(1, keepdims=True)
    x1, x2, eps = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    A = np.eye(n) + 1.4 * Wn
    df = pd.DataFrame(
        {
            "ysar": np.linalg.solve(A, X @ [1, 1.0, -0.5] + eps),
            "ysem": X @ [1, 1.0, -0.5] + np.linalg.solve(A, eps),
            "ysdm": np.linalg.solve(
                A, X @ [1, 1.0, -0.5] + Wn @ X[:, 1:] @ [0.5, 0.3] + eps
            ),
            "yslx": X @ [1, 1.0, -0.5]
            + Wn @ X[:, 1:] @ [0.5, 0.3]
            + np.linalg.solve(np.eye(n) - 0.4 * Wn, eps),
            "x1": x1,
            "x2": x2,
        }
    )
    return W, df


def _binary_ring_near_bound():
    """Binary ring (not row-standardised): eigenvalues in [-2.25, 4], so the
    admissible interval is (-0.444, 0.25). True rho 0.248."""
    n = 100
    W = _ring(n, 1.0)
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=n)
    y = np.linalg.solve(np.eye(n) - 0.248 * W, 1 + 1.0 * x1 + rng.normal(size=n))
    return W, pd.DataFrame({"y": y, "x1": x1})


# ---------------------------------------------------------------------- #
#  ML search interval: rho / lambda below -0.99 (or within 1% of 1/omega_max)
# ---------------------------------------------------------------------- #

# spatialreg lagsarlm / errorsarlm(method = "eigen"), default interval (-2, 1).
_R_ROW_STD = {
    "sar": ("ysar", "rho", -1.368263281435526, -210.789789954509),
    "sem": ("ysem", "lambda", -1.329762451505834, -211.119678335115),
    "sdm": ("ysdm", "rho", -1.317993802588640, -210.163618071824),
}


@pytest.mark.parametrize("model", ["sar", "sem", "sdm"])
def test_ml_spatial_parameter_below_minus_099_matches_spatialreg(model):
    W, df = _row_std_data()
    y, par, r_est, r_ll = _R_ROW_STD[model]
    res = getattr(sp, model)(W, df, f"{y} ~ x1 + x2")
    # 1.40.2: -0.98999998 (the bracket end) for all three.
    assert res.params[par] == pytest.approx(r_est, rel=1e-8)
    assert res.diagnostics["Log-Likelihood"] == pytest.approx(r_ll, abs=1e-4)


def test_ml_rho_near_upper_end_of_unstandardised_w_matches_spatialreg():
    W, df = _binary_ring_near_bound()
    res = sp.sar(W, df, "y ~ x1", row_normalize=False)
    # spatialreg: rho 0.2479713895537652 (interval (-0.4444458, 0.25)); 1.40.2
    # stopped at 0.99 * 0.25 = 0.2475.
    assert res.params["rho"] == pytest.approx(0.2479713895537652, rel=1e-8)
    assert res.std_errors["rho"] == pytest.approx(8.167996609578137e-04, rel=1e-6)


# ---------------------------------------------------------------------- #
#  impacts: admissible interval from the eigenvalues of W
# ---------------------------------------------------------------------- #


def test_impacts_discards_draws_outside_eigenvalue_interval():
    W, df = _binary_ring_near_bound()
    res = sp.sar(W, df, "y ~ x1", row_normalize=False)
    ev = np.linalg.eigvalsh(W)
    lo, hi = 1.0 / ev.min(), 1.0 / ev.max()
    V = res.model_info["vcov"].to_numpy()
    draws = np.random.default_rng(3).multivariate_normal(
        res.params.to_numpy(), V, size=4000
    )
    rho = draws[:, list(res.params.index).index("rho")]
    expected = int(np.sum((rho <= lo) | (rho >= hi)))
    assert expected > 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        imp = sp.impacts(res, n_sim=4000, seed=3)
    # 1.40.2 kept every draw below 0.999, past the singular point 0.25.
    assert imp.attrs["n_draws_discarded"] == expected


def test_impacts_refuses_rho_outside_eigenvalue_interval():
    W, df = _binary_ring_near_bound()
    res = sp.sar(W, df, "y ~ x1", row_normalize=False)
    res.params["rho"] = 0.3  # inside (-0.999, 0.999), past 1 / 4
    with pytest.raises(NumericalInstability, match="invertible"):
        sp.impacts(res, n_sim=50, seed=0)


def test_impacts_sar_rho_below_minus_one_matches_spatialreg():
    W, df = _row_std_data()
    res = sp.sar(W, df, "ysar ~ x1 + x2")
    imp = sp.impacts(res, n_sim=20000, seed=0)
    # spatialreg impacts(listw =), exact dense measures.
    r_direct = [1.359950083182564, -0.972100420207997]
    r_total = [0.354524201280332, -0.253416010852400]
    np.testing.assert_allclose(imp["Direct"], r_direct, rtol=1e-8)
    np.testing.assert_allclose(imp["Total"], r_total, rtol=1e-8)
    # S-screen: spatialreg R = 20000, seeds 1-3, SE_Total of x1 0.0448 /
    # 0.0451 / 0.0447; the SD of a 20000-draw SD estimate is about 0.5%.
    assert imp.loc["x1", "SE_Total"] == pytest.approx(0.0449, rel=0.03)
    assert imp.attrs["n_draws_discarded"] == 0


# ---------------------------------------------------------------------- #
#  impacts for SLX / SDEM (exact) and GNS (simulated)
# ---------------------------------------------------------------------- #


def test_impacts_slx_equal_spatialreg_exactly():
    W, df = _row_std_data()
    imp = sp.impacts(sp.slx(W, df, "yslx ~ x1 + x2"))
    # spatialreg summary(impacts(lmSLX(...))): estimates and SEs.
    r_direct = [0.80407107700697100, -0.58246689407346386]
    r_indirect = [0.26278444152454417, 0.40973476956062560]
    r_se = {
        "SE_Direct": [0.0996982112761468, 0.1140312873347157],
        "SE_Indirect": [0.197530848855488, 0.220846340435894],
        "SE_Total": [0.222219560108872, 0.248783536066073],
    }
    np.testing.assert_allclose(imp["Direct"], r_direct, rtol=1e-12)
    np.testing.assert_allclose(imp["Indirect"], r_indirect, rtol=1e-12)
    for col, ref in r_se.items():
        np.testing.assert_allclose(imp[col], ref, rtol=1e-12)
    assert imp.attrs["se_method"] == "analytic"


def test_impacts_sdem_are_beta_theta_with_vcov_linear_combination():
    W, df = _row_std_data()
    res = sp.sdem(W, df, "yslx ~ x1 + x2")
    imp = sp.impacts(res)
    # spatialreg errorsarlm(Durbin = TRUE) impacts.
    np.testing.assert_allclose(
        imp["Direct"], [0.8192030394612632, -0.5813852507405966], rtol=1e-8
    )
    np.testing.assert_allclose(
        imp["Indirect"], [0.3695572852605635, 0.4055362091554916], rtol=1e-8
    )
    # The SEs are the exact linear combinations of the stored (OIM)
    # covariance. spatialreg's differ by the covariance convention only
    # (asymptotic vs observed information: x1 total 0.27247 vs 0.27514).
    V = res.model_info["vcov"]
    for v in ("x1", "x2"):
        b, t = v, f"W_{v}"
        se_tot = np.sqrt(V.loc[b, b] + V.loc[t, t] + 2 * V.loc[b, t])
        assert imp.loc[v, "SE_Total"] == pytest.approx(se_tot, rel=1e-12)
        assert imp.loc[v, "SE_Indirect"] == pytest.approx(
            np.sqrt(V.loc[t, t]), rel=1e-12
        )


def test_impacts_wx_total_uses_mean_row_sum_of_unstandardised_w():
    """S_k = beta I + theta W: total = beta + theta * (1'W1)/n (here 4)."""
    rng = np.random.default_rng(1)
    n = 60
    W = _ring(n, 1.0)
    x = rng.normal(size=n)
    df = pd.DataFrame({"y": 1 + x + 0.3 * W @ x + rng.normal(size=n), "x": x})
    res = sp.slx(W, df, "y ~ x", row_normalize=False)
    imp = sp.impacts(res)
    b, t = res.params["x"], res.params["W_x"]
    assert imp.loc["x", "Direct"] == pytest.approx(b, rel=1e-12)
    assert imp.loc["x", "Total"] == pytest.approx(b + 4.0 * t, rel=1e-12)


def test_impacts_gns_point_values_match_spatialreg():
    W, df = _row_std_data()
    imp = sp.impacts(sp.gns(W, df, "yslx ~ x1 + x2"), n_sim=200, seed=0)
    # spatialreg sacsarlm(Durbin = TRUE) impacts; rho / lambda agree to 2e-7.
    np.testing.assert_allclose(
        imp["Direct"], [0.816405073283980, -0.581038285103351], rtol=1e-6
    )
    np.testing.assert_allclose(
        imp["Total"], [1.19103980223911, -0.15462952224545], rtol=1e-6
    )


def test_impacts_still_refuses_sem():
    W, df = _row_std_data()
    with pytest.raises(MethodIncompatibility, match="SAR / SDM / SAC"):
        sp.impacts(sp.sem(W, df, "ysem ~ x1 + x2"), n_sim=50)
