"""Tests exercising the new W-object-aware sparse SAR/SEM/SDM path."""
import numpy as np
import pandas as pd
import pytest

from statspai.spatial.weights import knn_weights
from statspai.spatial import sar, sem, sdm


@pytest.fixture(scope="module")
def dgp():
    rng = np.random.default_rng(0)
    n = 80
    coords = rng.uniform(size=(n, 2))
    w = knn_weights(coords, k=5); w.transform = "R"
    x = rng.standard_normal(n)
    y = 2.0 + 1.5 * x + rng.standard_normal(n)
    return w, pd.DataFrame({"y": y, "x": x})


def test_sar_accepts_W_object(dgp):
    w, df = dgp
    res = sar(w, df, "y ~ x")
    assert hasattr(res, "params")
    # Order in the new estimator: [const, indep..., rho]; rho is last.
    assert res.params.index[-1].lower() == "rho"


def test_sem_accepts_W_object(dgp):
    w, df = dgp
    res = sem(w, df, "y ~ x")
    assert res.params.index[-1].lower() in {"lambda", "lam"}


def test_sdm_accepts_W_object(dgp):
    w, df = dgp
    res = sdm(w, df, "y ~ x")
    # SDM has rho (last) + beta + W-lagged beta
    assert res.params.index[-1].lower() == "rho"
    lag_names = [n for n in res.params.index if n.startswith("W_")]
    assert len(lag_names) >= 1


def test_sar_large_n_sparse_no_densify(dgp):
    # n=2000 should run in under a few seconds using sparse ops
    rng = np.random.default_rng(1)
    n = 2000
    coords = rng.uniform(size=(n, 2))
    w = knn_weights(coords, k=6); w.transform = "R"
    x = rng.standard_normal(n)
    y = 1.0 + x + 0.5 * rng.standard_normal(n)
    res = sar(w, pd.DataFrame({"y": y, "x": x}), "y ~ x")
    assert np.isfinite(res.params.values).all()
