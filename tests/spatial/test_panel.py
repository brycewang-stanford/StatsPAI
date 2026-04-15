"""Spatial panel (SAR-FE / SEM-FE / SDM-FE) synthetic tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.spatial.panel import spatial_panel
from statspai.spatial.weights import knn_weights


def _long_frame(Y, X, var_names=("y", "x")):
    N, T = Y.shape
    rows = []
    for i in range(N):
        for t in range(T):
            rows.append({"id": i, "t": t, var_names[0]: Y[i, t], var_names[1]: X[i, t]})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def sar_panel_dgp():
    rng = np.random.default_rng(42)
    N, T = 30, 10
    coords = rng.uniform(size=(N, 2))
    w = knn_weights(coords, k=4); w.transform = "R"
    W = w.sparse.toarray()
    mu = rng.standard_normal(N) * 2
    X = rng.standard_normal((N, T))
    eps = rng.standard_normal((N, T)) * 0.5
    rho_true, beta_true = 0.4, 1.5
    I = np.eye(N)
    A_inv = np.linalg.inv(I - rho_true * W)
    Y = A_inv @ (mu[:, None] + beta_true * X + eps)
    return w, _long_frame(Y, X), rho_true, beta_true


def test_sar_fe_recovers_parameters(sar_panel_dgp):
    w, df, rho_true, beta_true = sar_panel_dgp
    res = spatial_panel(df, "y ~ x", entity="id", time="t", W=w,
                        model="sar", effects="fe")
    assert abs(res.spatial_param_value - rho_true) < 0.1
    assert abs(res.params["x"] - beta_true) < 0.1


def test_sem_fe_runs(sar_panel_dgp):
    w, df, _, _ = sar_panel_dgp
    res = spatial_panel(df, "y ~ x", entity="id", time="t", W=w,
                        model="sem", effects="fe")
    assert np.isfinite(res.spatial_param_value)
    assert res.params["x"] > 0              # positive coefficient recoverable


def test_sdm_fe_has_lagged_covariates(sar_panel_dgp):
    w, df, _, _ = sar_panel_dgp
    res = spatial_panel(df, "y ~ x", entity="id", time="t", W=w,
                        model="sdm", effects="fe")
    assert "W_x" in res.params.index
    assert "rho" in res.params.index


def test_twoways_effects_also_work(sar_panel_dgp):
    w, df, rho_true, beta_true = sar_panel_dgp
    res = spatial_panel(df, "y ~ x", entity="id", time="t", W=w,
                        model="sar", effects="twoways")
    # ρ should still land in a plausible region
    assert abs(res.spatial_param_value - rho_true) < 0.2


def test_unbalanced_panel_raises():
    rng = np.random.default_rng(0)
    N = 4
    rows = [{"id": i, "t": t, "y": rng.standard_normal(), "x": rng.standard_normal()}
            for i in range(N) for t in range(5)]
    # drop an observation to make it unbalanced
    df = pd.DataFrame(rows[:-1])
    from statspai.spatial.weights import block_weights
    w = block_weights(np.array(["A", "A", "B", "B"]))
    with pytest.raises(ValueError, match="unbalanced"):
        spatial_panel(df, "y ~ x", entity="id", time="t", W=w,
                      model="sar", effects="fe")


def test_spatial_panel_reexported():
    import statspai as sp
    assert callable(sp.spatial_panel)
