"""Tests for the small-cluster inference helpers in ``inference/jackknife.py``.

Covers ``sp.jackknife_se`` and ``sp.wild_cluster_boot`` — both public and
previously untested (CLAUDE.md §5). The jackknife test is a genuine
correctness check: it re-derives the leave-one-cluster-out variance from the
documented formula independently and asserts machine-precision agreement,
rather than re-asserting the implementation against itself.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _clustered_data(n_clusters=8, per=30, beta=0.5, seed=0):
    """Balanced clustered panel with a known slope and cluster random effects."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_clusters):
        a = rng.normal(0, 1.0)  # cluster effect
        for _ in range(per):
            x = rng.normal()
            y = 1.0 + beta * x + a + rng.normal(0, 1.0)
            rows.append((y, x, g))
    return pd.DataFrame(rows, columns=["y", "x", "state"])


def _independent_cluster_jackknife_se(df, y="y", x="x", cluster="state"):
    """Reference V_jk = ((G-1)/G) * sum_g (b_g - b_bar)(b_g - b_bar)'."""
    Y = df[y].to_numpy(float)
    X = np.column_stack([np.ones(len(df)), df[x].to_numpy(float)])
    cl = df[cluster].to_numpy()
    uc = np.unique(cl)
    G = len(uc)
    beta_loo = np.array(
        [
            np.linalg.solve(X[cl != g].T @ X[cl != g], X[cl != g].T @ Y[cl != g])
            for g in uc
        ]
    )
    dev = beta_loo - beta_loo.mean(axis=0)
    V_jk = ((G - 1) / G) * (dev.T @ dev)
    return np.sqrt(np.diag(V_jk))  # [intercept_se, slope_se]


# --------------------------------------------------------------------------
# jackknife_se — correctness
# --------------------------------------------------------------------------
def test_jackknife_se_matches_reference_formula():
    df = _clustered_data()
    res = sp.regress("y ~ x", data=df, cluster="state")
    jk = sp.jackknife_se(res, data=df, cluster="state")

    ref = _independent_cluster_jackknife_se(df)
    np.testing.assert_allclose(jk.std_errors["Intercept"], ref[0], rtol=1e-10)
    np.testing.assert_allclose(jk.std_errors["x"], ref[1], rtol=1e-10)


def test_jackknife_se_metadata():
    df = _clustered_data(n_clusters=8)
    res = sp.regress("y ~ x", data=df, cluster="state")
    jk = sp.jackknife_se(res, data=df, cluster="state")

    assert jk.model_info["se_type"] == "cluster jackknife"
    assert jk.model_info["n_clusters"] == 8
    # Effective DoF is G - 1 (deliberately conservative vs asymptotic CRVE).
    assert jk.data_info["df_resid"] == 7
    # Point estimates are unchanged — only the SEs are replaced.
    np.testing.assert_allclose(jk.params.to_numpy(), res.params.to_numpy(), rtol=1e-12)


def test_jackknife_se_more_clusters_tightens_se():
    # Boundary/monotonicity sanity: with the same DGP, more clusters of the
    # same size should not blow up the slope SE relative to a tiny design.
    few = sp.jackknife_se(
        sp.regress("y ~ x", data=(d := _clustered_data(4, 30)), cluster="state"),
        data=d,
        cluster="state",
    )
    many = sp.jackknife_se(
        sp.regress("y ~ x", data=(d2 := _clustered_data(16, 30)), cluster="state"),
        data=d2,
        cluster="state",
    )
    assert many.std_errors["x"] < few.std_errors["x"]


# --------------------------------------------------------------------------
# wild_cluster_boot — contract + behaviour
# --------------------------------------------------------------------------
def test_wild_cluster_boot_contract():
    df = _clustered_data()
    res = sp.regress("y ~ x", data=df, cluster="state")
    out = sp.wild_cluster_boot(
        res, data=df, cluster="state", variable="x", n_boot=199, seed=42
    )
    expected_keys = {
        "beta_hat",
        "se_cluster",
        "t_stat",
        "p_boot",
        "ci_boot",
        "t_distribution",
        "n_clusters",
        "n_obs",
        "n_boot",
        "weight_type",
    }
    assert expected_keys <= set(out)
    assert 0.0 <= out["p_boot"] <= 1.0
    lo, hi = out["ci_boot"]
    assert lo < hi
    assert out["n_clusters"] == 8
    assert out["n_boot"] == 199
    assert len(out["t_distribution"]) == 199


def test_wild_cluster_boot_is_seed_reproducible():
    df = _clustered_data()
    res = sp.regress("y ~ x", data=df, cluster="state")
    a = sp.wild_cluster_boot(
        res, data=df, cluster="state", variable="x", n_boot=199, seed=7
    )
    b = sp.wild_cluster_boot(
        res, data=df, cluster="state", variable="x", n_boot=199, seed=7
    )
    assert a["p_boot"] == b["p_boot"]
    np.testing.assert_array_equal(a["t_distribution"], b["t_distribution"])


def test_wild_cluster_boot_rejects_strong_signal():
    # True slope = 1.5 (large): the H0=0 test should reject decisively.
    df = _clustered_data(beta=1.5, seed=1)
    res = sp.regress("y ~ x", data=df, cluster="state")
    out = sp.wild_cluster_boot(
        res, data=df, cluster="state", variable="x", n_boot=499, seed=3
    )
    assert out["p_boot"] < 0.05


@pytest.mark.parametrize("weight_type", ["rademacher", "webb", "mammen"])
def test_wild_cluster_boot_weight_schemes_run(weight_type):
    df = _clustered_data()
    res = sp.regress("y ~ x", data=df, cluster="state")
    out = sp.wild_cluster_boot(
        res,
        data=df,
        cluster="state",
        variable="x",
        n_boot=99,
        seed=0,
        weight_type=weight_type,
    )
    assert np.isfinite(out["p_boot"])


def test_wild_cluster_boot_unknown_variable_raises():
    df = _clustered_data()
    res = sp.regress("y ~ x", data=df, cluster="state")
    with pytest.raises(ValueError, match="not found"):
        sp.wild_cluster_boot(
            res, data=df, cluster="state", variable="nope", n_boot=99, seed=0
        )


def test_wild_cluster_boot_unknown_weight_raises():
    df = _clustered_data()
    res = sp.regress("y ~ x", data=df, cluster="state")
    with pytest.raises(ValueError, match="weight_type"):
        sp.wild_cluster_boot(
            res,
            data=df,
            cluster="state",
            variable="x",
            n_boot=99,
            seed=0,
            weight_type="bogus",
        )


def test_wild_cluster_boot_warns_with_few_clusters():
    df = _clustered_data(n_clusters=4, per=40)
    res = sp.regress("y ~ x", data=df, cluster="state")
    with pytest.warns(UserWarning, match="clusters"):
        sp.wild_cluster_boot(
            res, data=df, cluster="state", variable="x", n_boot=99, seed=0
        )
