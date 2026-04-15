"""SLX and SAC (SARAR) model smoke + shape tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.spatial.models.ml import slx, sac
from statspai.spatial.weights import knn_weights


@pytest.fixture(scope="module")
def dgp():
    rng = np.random.default_rng(0)
    n = 80
    coords = rng.uniform(size=(n, 2))
    w = knn_weights(coords, k=5); w.transform = "R"
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    y = 1.0 + 2.0 * x + 1.0 * z + rng.standard_normal(n)
    return w, pd.DataFrame({"y": y, "x": x, "z": z})


def test_slx_returns_augmented_coefficients(dgp):
    w, df = dgp
    res = slx(w, df, "y ~ x + z")
    # Expect const, x, z, W_x, W_z — 5 params
    assert len(res.params) == 5
    assert "W_x" in res.params.index
    assert "W_z" in res.params.index
    assert res.model_info["model_type"].startswith("SLX")


def test_sac_recovers_rho_and_lambda():
    """SAC identification is weak at small N — use a larger sample (N=400).

    With N=80 the MLE is systematically biased toward (ρ, λ) ≈ (0.1, 0.2)
    regardless of truth, because the two spatial coefficients induce highly
    correlated autocorrelation patterns.
    """
    rng = np.random.default_rng(7)
    n = 400
    coords = rng.uniform(size=(n, 2))
    w = knn_weights(coords, k=6); w.transform = "R"
    W_dense = w.sparse.toarray()
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    beta_true = np.array([1.0, 2.0])
    rho_true, lam_true = 0.4, 0.3
    eps = rng.standard_normal(n)
    u = np.linalg.solve(np.eye(n) - lam_true * W_dense, eps)
    y = np.linalg.solve(np.eye(n) - rho_true * W_dense, X @ beta_true + u)
    df = pd.DataFrame({"y": y, "x": X[:, 1]})
    res = sac(w, df, "y ~ x")
    rho_hat = res.model_info["spatial_param_value"]
    lam_hat = res.diagnostics["lambda_hat"]
    # Generous tolerance — SAC finite-sample bias is a real phenomenon,
    # we just need to confirm the estimator lands in the right region.
    assert abs(rho_hat - rho_true) < 0.25
    assert abs(lam_hat - lam_true) < 0.25


def test_slx_exported_on_package():
    from statspai.spatial.models import ml
    assert hasattr(ml, "slx")
    assert hasattr(ml, "sac")
