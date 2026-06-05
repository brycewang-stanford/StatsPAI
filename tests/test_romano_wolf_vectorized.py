"""Guards for the vectorized Romano-Wolf bootstrap (sp.romano_wolf).

The per-draw / per-outcome ``_ols_fit`` calls were replaced by a vectorized
core: index numpy arrays (no DataFrame copy), share QR / (X'X)^-1 across
outcomes, and compute the first-coefficient SE in closed form. These tests pin
that the vectorized first-coef t-stat equals the per-outcome ``_ols_fit`` path
(HC1 and cluster) and that the result is seed-deterministic.
"""

import numpy as np
import pandas as pd

import statspai as sp
from statspai.mht.romano_wolf import _ols_fit


def _vectorized_first_coef_t(x_b, y_b, cluster_b):
    """The exact closed-form used inside the vectorized bootstrap."""
    n, k = x_b.shape
    q_b, r_b = np.linalg.qr(x_b)
    beta_b = np.linalg.solve(r_b, q_b.T @ y_b)
    bread = np.linalg.inv(x_b.T @ x_b)
    a = bread[0]
    resid_b = y_b - x_b @ beta_b
    u = x_b @ a
    if cluster_b is None:
        v0 = (n / (n - k)) * ((u ** 2) @ (resid_b ** 2))
    else:
        uniq = np.unique(cluster_b)
        g = len(uniq)
        dfc = g / (g - 1) * (n - 1) / (n - k)
        pos = np.searchsorted(uniq, cluster_b)
        ind = np.zeros((n, g))
        ind[np.arange(n), pos] = 1.0
        cs = ind.T @ (u[:, None] * resid_b)
        v0 = dfc * (cs ** 2).sum(axis=0)
    return beta_b[0] / np.sqrt(v0)


def _check_identity(cluster):
    rng = np.random.default_rng(0)
    n, k, S = 1200, 3, 8
    x = np.column_stack([rng.normal(0, 1, (n, k - 1)), np.ones(n)])
    y = 0.1 * x[:, [0]] + rng.normal(0, 1, (n, S))
    cl = np.repeat(np.arange(24), n // 24) if cluster else None
    t_vec = _vectorized_first_coef_t(x, y, cl)
    t_loop = np.array([_ols_fit(y[:, s], x, cl)[2] for s in range(S)])
    assert np.allclose(t_vec, t_loop, rtol=1e-9, atol=1e-11)
    assert np.max(np.abs(t_vec - t_loop)) < 1e-9


def test_vectorized_t_matches_ols_fit_hc1():
    _check_identity(cluster=False)


def test_vectorized_t_matches_ols_fit_cluster():
    _check_identity(cluster=True)


def _data(cluster=False, seed=0):
    rng = np.random.default_rng(seed)
    n, S = 1500, 5
    d = {"x": rng.normal(0, 1, n)}
    for j in range(S):
        d[f"y{j}"] = 0.1 * d["x"] + rng.normal(0, 1, n)
    if cluster:
        d["cl"] = np.repeat(np.arange(30), n // 30)
    return pd.DataFrame(d), [f"y{j}" for j in range(S)]


def test_romano_wolf_seed_deterministic():
    df, yc = _data()
    a = sp.romano_wolf(df, y=yc, x="x", n_boot=400, seed=5)
    b = sp.romano_wolf(df, y=yc, x="x", n_boot=400, seed=5)
    np.testing.assert_allclose(a.table["p_rw"].values, b.table["p_rw"].values)
    np.testing.assert_allclose(a.table["t"].values, b.table["t"].values)


def test_romano_wolf_cluster_runs_and_valid():
    df, yc = _data(cluster=True)
    out = sp.romano_wolf(df, y=yc, x="x", cluster="cl", n_boot=300, seed=2)
    pr = out.table["p_rw"].values
    assert np.all((pr >= 0) & (pr <= 1))
    # Stepdown p-values are monotone in |t| order (non-decreasing down the rank).
    order = np.argsort(-np.abs(out.table["t"].values))
    assert np.all(np.diff(pr[order]) >= -1e-12)
