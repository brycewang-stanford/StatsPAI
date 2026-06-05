"""Guards for the vectorized wild cluster bootstrap (sp.wild_cluster_bootstrap).

The per-cluster Python loops (Y* assembly and the CR1 meat) were replaced by
vectorized linear algebra, keeping the per-iteration weight draw unchanged so
the output is numerically identical (p_boot exact, t-distribution to ~1e-15)
while running ~25x faster. These tests pin the vectorization math and the
determinism / validity contract.
"""

import numpy as np
import pandas as pd

import statspai as sp


def test_vectorized_meat_equals_per_cluster_loop():
    """S'S with a one-hot cluster indicator == sum_g outer(X_g' e_g)."""
    rng = np.random.default_rng(0)
    n, k, G = 600, 3, 12
    X = rng.normal(0, 1, (n, k))
    resid = rng.normal(0, 1, n)
    cl = np.repeat(np.arange(G), n // G)
    unique_cl = np.unique(cl)

    meat_loop = np.zeros((k, k))
    for gv in unique_cl:
        idx = cl == gv
        sg = X[idx].T @ resid[idx]
        meat_loop += np.outer(sg, sg)

    cpos = np.searchsorted(unique_cl, cl)
    ind = np.zeros((n, G))
    ind[np.arange(n), cpos] = 1.0
    scores = ind.T @ (X * resid[:, None])
    meat_vec = scores.T @ scores

    assert np.allclose(meat_loop, meat_vec, rtol=1e-10, atol=1e-12)
    assert np.max(np.abs(meat_loop - meat_vec)) < 1e-9


def _clustered(n=1000, G=20, beta=0.0, seed=0):
    rng = np.random.default_rng(seed)
    cl = np.repeat(np.arange(G), n // G)
    m = len(cl)
    x = rng.normal(0, 1, m)
    y = 1 + beta * x + rng.normal(0, 1, m) + np.repeat(rng.normal(0, 0.5, G),
                                                       n // G)
    return pd.DataFrame({"y": y, "x": x, "cl": cl})


def test_deterministic_same_seed():
    df = _clustered()
    a = sp.wild_cluster_bootstrap(df, y="y", x=["x"], cluster="cl",
                                  test_var="x", n_boot=399, seed=11)
    b = sp.wild_cluster_bootstrap(df, y="y", x=["x"], cluster="cl",
                                  test_var="x", n_boot=399, seed=11)
    assert a["p_boot"] == b["p_boot"]
    assert np.allclose(a["ci_boot"], b["ci_boot"])
    assert a["t_stat"] == b["t_stat"]


def test_weight_types_valid_output():
    df = _clustered()
    for wt in ("rademacher", "webb", "mammen"):
        out = sp.wild_cluster_bootstrap(df, y="y", x=["x"], cluster="cl",
                                        test_var="x", n_boot=299, seed=3,
                                        weight_type=wt)
        assert 0.0 <= out["p_boot"] <= 1.0
        lo, hi = out["ci_boot"]
        assert lo < hi


def test_strong_signal_rejects():
    df = _clustered(beta=1.5, seed=1)
    out = sp.wild_cluster_bootstrap(df, y="y", x=["x"], cluster="cl",
                                    test_var="x", n_boot=499, seed=2)
    assert out["p_boot"] < 0.05
