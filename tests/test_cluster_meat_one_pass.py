"""Cluster meats in one pass (2026-10-19).

``core._numba_kernels.cluster_meat`` no longer sorts the rows by cluster,
and ``inference.jackknife.two_way_vcov_ols`` no longer loops over clusters
with a row mask each (O(n G): minutes at a million rows). Both must still
be ``sum_g s_g s_g'`` with ``s_g = sum_{i in g} u_i x_i``: checked here
against that definition written out cluster by cluster, for integer and
text labels, and for invariance to the row order.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core._numba_kernels import cluster_meat


def _definition(X, u, labels):
    k = X.shape[1]
    m = np.zeros((k, k))
    for g in pd.unique(labels):
        idx = labels == g
        s = X[idx].T @ u[idx]
        m += np.outer(s, s)
    return m


@pytest.mark.parametrize("text", [False, True])
def test_cluster_meat_is_the_definition_in_any_row_order(text):
    rng = np.random.default_rng(0)
    n, k = 3000, 4
    X = rng.normal(size=(n, k))
    u = rng.normal(size=n)
    lab = rng.integers(0, 70, n)
    labels = np.array([f"c{v}" for v in lab], dtype=object) if text else lab
    ref = _definition(X, u, labels)
    got = cluster_meat(X, u, labels)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())
    perm = rng.permutation(n)
    again = cluster_meat(X[perm], u[perm], labels[perm])
    np.testing.assert_allclose(again, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


def test_two_way_regress_is_the_inclusion_exclusion_sum():
    rng = np.random.default_rng(1)
    n = 2000
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "a": rng.integers(0, 40, n),
            "b": rng.integers(0, 15, n),
        }
    )
    df["y"] = 1 + 0.5 * df["x"] + rng.normal(size=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.regress("y ~ x", df, cluster=["a", "b"])
    X = np.column_stack([np.ones(n), df["x"].to_numpy()])
    yv = df["y"].to_numpy()
    bread = np.linalg.inv(X.T @ X)
    u = yv - X @ (bread @ X.T @ yv)
    ab = df["a"].to_numpy() * 100 + df["b"].to_numpy()
    meat = (
        _definition(X, u, df["a"].to_numpy())
        + _definition(X, u, df["b"].to_numpy())
        - _definition(X, u, ab)
    )
    g = 15
    V = (g / (g - 1)) * ((n - 1) / (n - 2)) * bread @ meat @ bread
    np.testing.assert_allclose(
        fit.std_errors.to_numpy(), np.sqrt(np.diag(V)), rtol=1e-10
    )
