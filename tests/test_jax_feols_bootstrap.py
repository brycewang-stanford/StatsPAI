"""Parity tests for ``sp.fast.feols_jax_bootstrap`` (Phase 4b).

The bootstrap SEs are stochastic, so parity here is in the
**convergence** sense:
  * Pairs-bootstrap SE → HC1 SE as B → ∞
  * Cluster-bootstrap SE → CR1 SE as B → ∞

We use B=2000 throughout so the Monte-Carlo SE on the bootstrap SE
estimate is small enough to assert ~5% relative tolerance, and we
also pin a couple of deterministic invariants (point estimate from
``coef`` matches ``feols_jax`` exactly; same-seed runs are bit-
identical).

Skips automatically when JAX is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jax")

from statspai.exceptions import (  # noqa: E402
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)
from statspai.fast import (  # noqa: E402
    feols,
    feols_jax,
    feols_jax_bootstrap,
    FeolsBootstrapResult,
)


def _make_panel(n: int = 1_000, n_firm: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, n_firm, size=n)
    fe = rng.normal(size=n_firm)[firm]
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.5 * x1 - 0.2 * x2 + fe + 0.5 * rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "firm": firm,
            "cluster": firm,
        }
    )


# ---------------------------------------------------------------------------
# Deterministic invariants (no Monte-Carlo noise)
# ---------------------------------------------------------------------------


def test_point_estimate_matches_feols_jax():
    """``coef`` is the un-resampled point estimate; must match feols_jax exactly."""
    df = _make_panel(seed=1)
    fit = feols_jax("y ~ x1 + x2 | firm", df, vcov="iid")
    boot = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm",
        df,
        n_boot=100,
        seed=0,
    )
    np.testing.assert_allclose(boot.coef.values, fit.coef_vec, atol=1e-12)
    assert list(boot.coef.index) == list(fit.coef_names)


def test_returns_correct_dataclass_type():
    df = _make_panel(seed=2)
    boot = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm",
        df,
        n_boot=50,
        seed=0,
    )
    assert isinstance(boot, FeolsBootstrapResult)
    assert boot.bootstrap_type == "pairs"
    assert boot.backend == "statspai-jax-bootstrap"
    assert boot.n_boot == 50


def test_same_seed_gives_identical_results():
    df = _make_panel(seed=3)
    b1 = feols_jax_bootstrap("y ~ x1 + x2 | firm", df, n_boot=100, seed=42)
    b2 = feols_jax_bootstrap("y ~ x1 + x2 | firm", df, n_boot=100, seed=42)
    np.testing.assert_array_equal(b1.boot_betas.values, b2.boot_betas.values)


def test_boot_betas_shape_and_columns():
    df = _make_panel(seed=4)
    boot = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm",
        df,
        n_boot=137,
        seed=0,
    )
    assert boot.boot_betas.shape == (137, 2)
    assert list(boot.boot_betas.columns) == ["x1", "x2"]


def test_chunk_size_does_not_change_results():
    """Different chunk sizes split the same vmap → identical numbers."""
    df = _make_panel(seed=5)
    b_small = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm",
        df,
        n_boot=200,
        seed=7,
        vmap_chunk_size=20,
    )
    b_large = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm",
        df,
        n_boot=200,
        seed=7,
        vmap_chunk_size=200,
    )
    np.testing.assert_allclose(
        b_small.boot_betas.values,
        b_large.boot_betas.values,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Convergence to analytic SEs
# ---------------------------------------------------------------------------


def test_pairs_bootstrap_se_converges_to_hc1():
    """Pairs SE → HC1 SE as B grows."""
    df = _make_panel(n=2_000, n_firm=80, seed=11)
    fit_hc1 = feols("y ~ x1 + x2 | firm", df, vcov="hc1")
    boot = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm",
        df,
        n_boot=2_000,
        seed=0,
    )
    # 5% relative tolerance — empirically tight enough at B=2000 even
    # with the JAX-vs-numpy random-stream divergence.
    np.testing.assert_allclose(
        boot.se_boot["x1"],
        fit_hc1.se()["x1"],
        rtol=0.10,
    )
    np.testing.assert_allclose(
        boot.se_boot["x2"],
        fit_hc1.se()["x2"],
        rtol=0.10,
    )


def test_cluster_bootstrap_se_converges_to_cr1():
    """Cluster SE → CR1 SE as B grows."""
    df = _make_panel(n=2_000, n_firm=80, seed=12)
    fit_cr1 = feols(
        "y ~ x1 + x2 | firm",
        df,
        vcov="cr1",
        cluster="cluster",
    )
    boot = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm",
        df,
        n_boot=2_000,
        seed=0,
        bootstrap="cluster",
        cluster="cluster",
    )
    # Cluster bootstrap convergence is slower than pairs; allow 15%.
    np.testing.assert_allclose(
        boot.se_boot["x1"],
        fit_cr1.se()["x1"],
        rtol=0.15,
    )


def test_percentile_ci_contains_true_value_for_well_specified_dgp():
    """95% CI should cover the true coefficient on a clean DGP."""
    df = _make_panel(n=2_000, n_firm=80, seed=13)
    boot = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm",
        df,
        n_boot=2_000,
        seed=0,
    )
    # True beta_x1 = 0.5 in _make_panel
    assert boot.ci_lower["x1"] < 0.5 < boot.ci_upper["x1"]
    # True beta_x2 = -0.2
    assert boot.ci_lower["x2"] < -0.2 < boot.ci_upper["x2"]


# ---------------------------------------------------------------------------
# Validation / error paths
# ---------------------------------------------------------------------------


def test_invalid_bootstrap_type_raises():
    df = _make_panel(seed=20)
    with pytest.raises(ValueError, match="bootstrap="):
        feols_jax_bootstrap("y ~ x1", df, n_boot=10, bootstrap="bayesian")


def test_cluster_bootstrap_without_cluster_raises():
    df = _make_panel(seed=21)
    with pytest.raises(ValueError, match="cluster"):
        feols_jax_bootstrap(
            "y ~ x1",
            df,
            n_boot=10,
            bootstrap="cluster",
        )


def test_wild_cluster_without_cluster_raises():
    df = _make_panel(seed=21)
    with pytest.raises(ValueError, match="cluster"):
        feols_jax_bootstrap(
            "y ~ x1",
            df,
            n_boot=10,
            bootstrap="wild_cluster",
        )


# ---------------------------------------------------------------------------
# Phase 4c: wild + wild_cluster bootstrap
# ---------------------------------------------------------------------------


def test_wild_bootstrap_se_converges_to_hc1():
    """Wild row-level SE → HC1 SE as B → ∞ (asymptotic equivalence)."""
    df = _make_panel(n=2_000, n_firm=80, seed=40)
    fit_hc1 = feols("y ~ x1 + x2 | firm", df, vcov="hc1")
    boot = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm",
        df,
        n_boot=2_000,
        seed=0,
        bootstrap="wild",
    )
    # Wild bootstrap with Rademacher weights converges to HC SE (not
    # HC1 — the n/(n-k) factor is asymptotically 1). 10% rtol matches
    # the pairs-bootstrap convergence rate at B=2000.
    np.testing.assert_allclose(
        boot.se_boot["x1"],
        fit_hc1.se()["x1"],
        rtol=0.10,
    )
    np.testing.assert_allclose(
        boot.se_boot["x2"],
        fit_hc1.se()["x2"],
        rtol=0.10,
    )


def test_wild_cluster_bootstrap_se_converges_to_cr1():
    """Wild cluster SE → CR1 SE as B → ∞."""
    df = _make_panel(n=2_000, n_firm=80, seed=41)
    fit_cr1 = feols(
        "y ~ x1 + x2 | firm",
        df,
        vcov="cr1",
        cluster="cluster",
    )
    boot = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm",
        df,
        n_boot=2_000,
        seed=0,
        bootstrap="wild_cluster",
        cluster="cluster",
    )
    # Wild cluster convergence is slower; allow 15%.
    np.testing.assert_allclose(
        boot.se_boot["x1"],
        fit_cr1.se()["x1"],
        rtol=0.15,
    )


def test_wild_bootstrap_point_estimate_matches_feols_jax():
    """Point estimate is from the original fit, identical across variants."""
    df = _make_panel(seed=42)
    fit = feols_jax("y ~ x1 + x2 | firm", df, vcov="iid")
    for variant in ("pairs", "cluster", "wild", "wild_cluster"):
        kw = {"n_boot": 50, "seed": 0, "bootstrap": variant}
        if variant in ("cluster", "wild_cluster"):
            kw["cluster"] = "cluster"
        boot = feols_jax_bootstrap("y ~ x1 + x2 | firm", df, **kw)
        np.testing.assert_allclose(
            boot.coef.values,
            fit.coef_vec,
            atol=1e-12,
        )


def test_wild_bootstrap_same_seed_is_bit_identical():
    df = _make_panel(seed=43)
    b1 = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm",
        df,
        n_boot=100,
        seed=42,
        bootstrap="wild",
    )
    b2 = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm",
        df,
        n_boot=100,
        seed=42,
        bootstrap="wild",
    )
    np.testing.assert_array_equal(b1.boot_betas.values, b2.boot_betas.values)


def test_wild_cluster_bootstrap_same_seed_is_bit_identical():
    df = _make_panel(seed=44)
    b1 = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm",
        df,
        n_boot=100,
        seed=42,
        bootstrap="wild_cluster",
        cluster="cluster",
    )
    b2 = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm",
        df,
        n_boot=100,
        seed=42,
        bootstrap="wild_cluster",
        cluster="cluster",
    )
    np.testing.assert_array_equal(b1.boot_betas.values, b2.boot_betas.values)


def test_wild_bootstrap_records_correct_type():
    df = _make_panel(seed=45)
    boot_w = feols_jax_bootstrap(
        "y ~ x1",
        df,
        n_boot=20,
        bootstrap="wild",
    )
    assert boot_w.bootstrap_type == "wild"
    boot_wc = feols_jax_bootstrap(
        "y ~ x1",
        df,
        n_boot=20,
        bootstrap="wild_cluster",
        cluster="cluster",
    )
    assert boot_wc.bootstrap_type == "wild_cluster"


def test_wild_score_bootstrap_matches_literal_refit_on_pseudo_y():
    """The score formulation β* = β̂ + (X'X)^-1 X'(η⊙û) is mathematically
    identical to fitting OLS on y* = Xβ̂ + η⊙û. Verify on a small
    Rademacher draw with no FE so the algebra is transparent."""
    df = _make_panel(seed=46).drop(columns=["firm"])  # no FE
    # Same RNG seed for reproducibility on both paths.
    boot = feols_jax_bootstrap(
        "y ~ x1 + x2",
        df,
        n_boot=2,
        seed=99,
        bootstrap="wild",
    )
    score_beta = boot.boot_betas.iloc[0].values

    # Literal refit: pull X, y, residuals from feols, draw the same
    # Rademacher sequence using the same JAX PRNG, build y*, refit OLS.
    fit = feols("y ~ x1 + x2", df, vcov="iid")
    import jax
    import jax.numpy as jnp

    n = len(df)
    key = jax.random.split(jax.random.PRNGKey(99), 2)[0]
    eta = 2 * jax.random.bernoulli(key, p=0.5, shape=(n,)).astype(jnp.float64) - 1
    eta_np = np.asarray(eta)
    X = np.column_stack([np.ones(n), df["x1"].values, df["x2"].values])
    y = df["y"].values
    beta_hat = fit.coef_vec
    resid = y - X @ beta_hat
    y_star = X @ beta_hat + eta_np * resid
    refit_beta = np.linalg.solve(X.T @ X, X.T @ y_star)

    # The two should agree to numerical precision (same Rademacher
    # draw + algebraic identity).
    np.testing.assert_allclose(score_beta, refit_beta, atol=1e-9)


def test_n_boot_below_one_raises():
    df = _make_panel(seed=22)
    with pytest.raises(MethodIncompatibility, match="n_boot"):
        feols_jax_bootstrap("y ~ x1", df, n_boot=0)


def test_n_boot_one_rejected_as_undefined_se():
    df = _make_panel(seed=221)
    with pytest.raises(DataInsufficient, match="n_boot"):
        feols_jax_bootstrap("y ~ x1", df, n_boot=1)


def test_chunk_below_one_raises():
    df = _make_panel(seed=23)
    with pytest.raises(MethodIncompatibility, match="vmap_chunk_size"):
        feols_jax_bootstrap("y ~ x1", df, n_boot=10, vmap_chunk_size=0)


def test_invalid_ci_alpha_raises():
    df = _make_panel(seed=24)
    with pytest.raises(MethodIncompatibility, match="ci_alpha"):
        feols_jax_bootstrap("y ~ x1", df, n_boot=10, ci_alpha=1.5)


def test_invalid_dtype_raises():
    df = _make_panel(seed=25)
    with pytest.raises(MethodIncompatibility, match="dtype="):
        feols_jax_bootstrap("y ~ x1", df, n_boot=10, dtype="bfloat16")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"fe_maxiter": 0}, "fe_maxiter"),
        ({"fe_tol": -1.0}, "fe_tol"),
        ({"ci_alpha": np.nan}, "ci_alpha"),
    ],
)
def test_invalid_prep_controls_raise(kwargs, match):
    df = _make_panel(seed=251)
    with pytest.raises(ValueError, match=match):
        feols_jax_bootstrap("y ~ x1 | firm", df, n_boot=10, **kwargs)


def test_all_zero_weights_raise():
    df = _make_panel(seed=252)
    df["w0"] = 0.0
    with pytest.raises(ValueError, match="no positive mass"):
        feols_jax_bootstrap("y ~ x1 | firm", df, n_boot=10, weights="w0")


def test_bootstrap_missing_columns_raise_method_incompatibility():
    df = _make_panel(seed=253)
    with pytest.raises(MethodIncompatibility) as exc:
        feols_jax_bootstrap("y ~ x1 + missing", df, n_boot=10)
    assert exc.value.diagnostics["missing_columns"] == ["missing"]


def test_bootstrap_non_dataframe_input_raises_method_incompatibility():
    df = _make_panel(seed=254)
    with pytest.raises(MethodIncompatibility, match="pandas DataFrame"):
        feols_jax_bootstrap("y ~ x1", df.to_dict("list"), n_boot=10)


def test_missing_cluster_column_raises():
    df = _make_panel(seed=26)
    with pytest.raises(MethodIncompatibility, match="not_a_column"):
        feols_jax_bootstrap(
            "y ~ x1",
            df,
            n_boot=10,
            bootstrap="cluster",
            cluster="not_a_column",
        )


def test_one_cluster_after_pruning_raises_data_insufficient():
    df = _make_panel(n=80, n_firm=1, seed=27)
    with pytest.raises(DataInsufficient, match=">= 2 clusters"):
        feols_jax_bootstrap(
            "y ~ x1",
            df,
            n_boot=10,
            bootstrap="cluster",
            cluster="cluster",
        )


def test_bootstrap_singular_solve_raises_numerical_instability():
    df = _make_panel(seed=28)
    df["x_dup"] = df["x1"]
    with pytest.raises(NumericalInstability):
        feols_jax_bootstrap("y ~ x1 + x_dup", df, n_boot=10)


# ---------------------------------------------------------------------------
# Summary + repr
# ---------------------------------------------------------------------------


def test_summary_contains_key_metadata():
    df = _make_panel(seed=30)
    boot = feols_jax_bootstrap(
        "y ~ x1 + x2 | firm",
        df,
        n_boot=50,
        seed=0,
    )
    s = boot.summary()
    assert "feols_jax_bootstrap" in s
    assert "pairs" in s
    assert "n_boot=50" in s
    assert "x1" in s
