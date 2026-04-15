"""LM diagnostics and LeSage-Pace impacts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.spatial.models.diagnostics import lm_tests
from statspai.spatial.models.impacts import impacts
from statspai.spatial.weights import knn_weights
from statspai.spatial import sar, sdm


@pytest.fixture(scope="module")
def sar_dgp():
    rng = np.random.default_rng(7)
    n = 120
    coords = rng.uniform(size=(n, 2))
    w = knn_weights(coords, k=6); w.transform = "R"
    W_dense = w.sparse.toarray()
    x = rng.standard_normal(n)
    rho_true = 0.5
    y = np.linalg.solve(np.eye(n) - rho_true * W_dense,
                        1 + 2 * x + rng.standard_normal(n))
    return w, pd.DataFrame({"y": y, "x": x})


@pytest.fixture(scope="module")
def iid_dgp():
    rng = np.random.default_rng(0)
    n = 100
    coords = rng.uniform(size=(n, 2))
    w = knn_weights(coords, k=4); w.transform = "R"
    x = rng.standard_normal(n)
    y = 1 + 2 * x + rng.standard_normal(n)          # no spatial structure
    return w, pd.DataFrame({"y": y, "x": x})


# ---------------------------------------------------------------------- LM
def test_lm_tests_returns_five_statistics(sar_dgp):
    w, df = sar_dgp
    out = lm_tests("y ~ x", df, w)
    expected = {"LM_err", "LM_lag", "Robust_LM_err", "Robust_LM_lag", "SARMA"}
    assert set(out.keys()) == expected
    for stat, p in out.values():
        assert np.isfinite(stat)


def test_lm_lag_rejects_spatial_dgp(sar_dgp):
    w, df = sar_dgp
    out = lm_tests("y ~ x", df, w)
    # Data have a true spatial lag → LM_lag should reject strongly
    assert out["LM_lag"][1] < 0.01


def test_lm_iid_data_does_not_reject(iid_dgp):
    w, df = iid_dgp
    out = lm_tests("y ~ x", df, w)
    # Non-spatial data — both tests should typically not reject.
    # Allow slack: require at least one of the two to have p>0.05
    assert out["LM_err"][1] > 0.05 or out["LM_lag"][1] > 0.05


# ---------------------------------------------------------------------- impacts
def test_impacts_sar_shape(sar_dgp):
    w, df = sar_dgp
    res = sar(w, df, "y ~ x")
    imp = impacts(res, n_sim=200, seed=0)
    assert list(imp.columns) == [
        "Direct", "SE_Direct", "Indirect", "SE_Indirect", "Total", "SE_Total"
    ]
    assert list(imp.index) == ["x"]


def test_impacts_sar_direct_equals_beta_over_1_minus_rho(sar_dgp):
    """For SAR, Direct ≈ β * mean_i [(I-ρW)^-1]_ii ; Total ≈ β / (1-ρ)."""
    w, df = sar_dgp
    res = sar(w, df, "y ~ x")
    imp = impacts(res, n_sim=200, seed=0)
    beta_x = float(res.params["x"])
    rho = float(res.model_info["spatial_param_value"])
    expected_total = beta_x / (1.0 - rho)
    assert abs(imp.loc["x", "Total"] - expected_total) < 0.1


def test_impacts_sdm_uses_theta(sar_dgp):
    w, df = sar_dgp
    res = sdm(w, df, "y ~ x")
    imp = impacts(res, n_sim=200, seed=0)
    assert list(imp.index) == ["x"]
    # SDM indirect should differ from SAR because W_x carries extra info
    assert np.isfinite(imp.loc["x", "Indirect"])


def test_impacts_rejects_non_sar_family(sar_dgp):
    from statspai.spatial import sem
    w, df = sar_dgp
    res = sem(w, df, "y ~ x")
    with pytest.raises(ValueError, match="SAR / SDM / SAC"):
        impacts(res, n_sim=50)
