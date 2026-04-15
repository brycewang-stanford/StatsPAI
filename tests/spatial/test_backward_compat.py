"""Pin the new sparse SAR/SEM/SDM ML estimator to the legacy parameter
vectors on a fixed synthetic DGP. The acceptance gate for Tasks 12-14 is
``rtol=1e-4`` per parameter."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from statspai.spatial import sar, sem, sdm


HERE = os.path.dirname(__file__)
FIXTURE = os.path.join(HERE, "fixtures", "baseline_outputs.json")


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(42)
    n = 40
    coords = rng.uniform(size=(n, 2))
    d = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    W = (d < 0.3).astype(float)
    np.fill_diagonal(W, 0.0)
    rs = W.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    W = W / rs
    x = rng.standard_normal(n)
    y = np.linalg.solve(np.eye(n) - 0.5 * W, 1 + 2 * x + rng.standard_normal(n))
    return W, pd.DataFrame({"y": y, "x": x})


@pytest.fixture(scope="module")
def baseline():
    with open(FIXTURE) as f:
        return json.load(f)


def test_sar_backward_compat(synthetic, baseline):
    W, df = synthetic
    res = sar(W, df, "y ~ x")
    np.testing.assert_allclose(res.params.values,
                               baseline["sar"]["params"], rtol=1e-4)


def test_sem_backward_compat(synthetic, baseline):
    W, df = synthetic
    res = sem(W, df, "y ~ x")
    np.testing.assert_allclose(res.params.values,
                               baseline["sem"]["params"], rtol=1e-4)


def test_sdm_backward_compat(synthetic, baseline):
    W, df = synthetic
    res = sdm(W, df, "y ~ x")
    np.testing.assert_allclose(res.params.values,
                               baseline["sdm"]["params"], rtol=1e-4)
