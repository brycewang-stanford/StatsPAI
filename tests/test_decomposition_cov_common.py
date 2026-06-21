"""Coverage campaign (decomposition) — shared numerical primitives (_common.py).

``decomposition/_common.py`` holds the WLS / logit / bootstrap / weighted-stat /
influence-function atoms every decomposition method delegates to (CLAUDE.md §11:
"don't re-implement kernel / WLS / sandwich / influence functions"). These are
pure numerical kernels, so every test pins a closed-form value, a parity target,
or a defining mathematical property — never a smoke call.
"""

from __future__ import annotations

import numpy as np
import pytest

from statspai.decomposition import _common as C
from statspai.exceptions import MethodIncompatibility

# ── WLS ──────────────────────────────────────────────────────────────


def test_wls_uniform_weights_equals_ols():
    rng = np.random.default_rng(0)
    n = 200
    X = C.add_constant(rng.normal(size=(n, 3)))
    beta_true = np.array([1.0, 2.0, -1.0, 0.5])
    y = X @ beta_true + rng.normal(0, 0.5, n)
    beta, vcov, resid = C.wls(y, X)
    ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    np.testing.assert_allclose(beta, ols, atol=1e-10)
    # vcov is a symmetric PSD matrix with positive diagonal.
    assert vcov.shape == (4, 4)
    assert np.all(np.diag(vcov) > 0)
    np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)
    np.testing.assert_allclose(resid, y - X @ beta, atol=1e-10)


def test_wls_replicate_weights_match_expanded_ols():
    rng = np.random.default_rng(1)
    X = C.add_constant(rng.normal(size=(50, 2)))
    y = X @ np.array([0.5, 1.0, -0.5]) + rng.normal(0, 0.3, 50)
    w = rng.integers(1, 4, size=50).astype(float)
    beta_w, *_ = C.wls(y, X, w=w, robust=False)
    # Expanding each row w_i times and running OLS must give the same point.
    idx = np.repeat(np.arange(50), w.astype(int))
    ols, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
    np.testing.assert_allclose(beta_w, ols, atol=1e-8)


# ── weighted quantile / gini / ecdf ──────────────────────────────────


def test_weighted_quantile_monotone_and_bounded():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # monotone non-decreasing in q, bounded by the support
    qs = [C.weighted_quantile(y, q) for q in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert all(a <= b for a, b in zip(qs, qs[1:]))
    assert qs[0] >= 1.0 and qs[-1] <= 5.0
    # the central quantile sits near the middle of a symmetric support
    assert 2.0 <= C.weighted_quantile(y, 0.5) <= 4.0


def test_weighted_gini_properties():
    # Perfect equality → Gini 0.
    assert C.weighted_gini(np.array([5.0, 5, 5, 5]), np.ones(4)) == pytest.approx(
        0.0, abs=1e-12
    )
    # Bounded in [0, 1].
    g = C.weighted_gini(np.array([1.0, 2, 3, 10, 50]), np.ones(5))
    assert 0.0 <= g <= 1.0
    # A more unequal distribution has a strictly larger Gini.
    g_equalish = C.weighted_gini(np.array([10.0, 11, 12, 13]), np.ones(4))
    g_unequal = C.weighted_gini(np.array([1.0, 2, 3, 40]), np.ones(4))
    assert g_unequal > g_equalish


def test_weighted_ecdf_monotone_0_to_1():
    y = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    F = C.weighted_ecdf(np.sort(y), y, np.ones(5))
    assert F[0] >= 0 and F[-1] == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.diff(F) >= -1e-12)


# ── statistic_value dispatch ─────────────────────────────────────────


def test_statistic_value_closed_forms():
    rng = np.random.default_rng(2)
    y = np.abs(rng.normal(5, 2, 500)) + 0.1
    w = np.ones_like(y)
    assert C.statistic_value(y, w, "mean") == pytest.approx(float(np.average(y)))
    assert C.statistic_value(y, w, "variance") == pytest.approx(
        float(np.cov(y)), rel=1e-9
    )
    assert C.statistic_value(y, w, "std") == pytest.approx(
        float(np.sqrt(np.cov(y))), rel=1e-9
    )
    # Theil-T, Theil-L, Atkinson, Gini all non-negative for positive incomes.
    for stat in ("gini", "theil_t", "theil_l", "atkinson", "log_var"):
        assert C.statistic_value(y, w, stat) >= -1e-9
    # quantile honours tau
    assert C.statistic_value(y, w, "quantile", tau=0.5) == pytest.approx(
        C.weighted_quantile(y, 0.5)
    )


def test_statistic_value_unknown_raises():
    with pytest.raises(MethodIncompatibility, match="(?i)unknown statistic") as excinfo:
        C.statistic_value(np.array([1.0, 2.0]), np.ones(2), "not_a_stat")
    assert isinstance(excinfo.value, ValueError)
    assert excinfo.value.diagnostics["stat"] == "not_a_stat"


# ── influence function ───────────────────────────────────────────────


@pytest.mark.parametrize("stat", ["mean", "quantile", "iqr"])
def test_rif_recenters_to_statistic(stat):
    rng = np.random.default_rng(3)
    y = np.abs(rng.normal(10, 3, 600)) + 0.5
    w = np.ones_like(y)
    rif = C.influence_function(y, stat, tau=0.5, w=w)
    # The RIF recentres so that its mean equals the statistic it expands.
    assert float(np.mean(rif)) == pytest.approx(
        C.statistic_value(y, w, stat, tau=0.5), rel=1e-6, abs=1e-6
    )


def test_rif_variance_family_recenters_to_population_moment():
    # The variance / std / log_var RIFs recentre to the *population* moment
    # (ddof=0), which differs from statistic_value's np.cov (ddof=1) by 1/n.
    rng = np.random.default_rng(31)
    y = np.abs(rng.normal(10, 3, 400)) + 0.5
    pop_var = float(np.average((y - y.mean()) ** 2))
    assert float(np.mean(C.influence_function(y, "variance"))) == pytest.approx(
        pop_var, rel=1e-9
    )
    assert float(np.mean(C.influence_function(y, "std"))) == pytest.approx(
        np.sqrt(pop_var), rel=1e-9
    )
    ly = np.log(y)
    pop_logvar = float(np.average((ly - ly.mean()) ** 2))
    assert float(np.mean(C.influence_function(y, "log_var"))) == pytest.approx(
        pop_logvar, rel=1e-9
    )


def test_rif_of_mean_is_y():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    rif = C.influence_function(y, "mean")
    # RIF of the mean is the identity: RIF(y; mean) = y.
    np.testing.assert_allclose(rif, y, atol=1e-10)


# ── bootstrap CIs ────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["percentile", "basic", "normal"])
def test_bootstrap_ci_methods(method):
    rng = np.random.default_rng(4)
    boot = rng.normal(2.0, 0.5, 2000)
    point = np.array([2.0])
    # bootstrap_ci returns (se, lo, hi).
    se, lo, hi = C.bootstrap_ci(boot.reshape(-1, 1), point, alpha=0.05, method=method)
    assert np.all(lo < hi)
    assert np.all(se > 0)


def test_bootstrap_ci_percentile_endpoints():
    boot = np.linspace(0, 1, 1001).reshape(-1, 1)
    _se, lo, hi = C.bootstrap_ci(boot, np.array([0.5]), alpha=0.10, method="percentile")
    assert lo[0] == pytest.approx(0.05, abs=1e-2)
    assert hi[0] == pytest.approx(0.95, abs=1e-2)


def test_bootstrap_ci_unknown_method_raises():
    with pytest.raises(MethodIncompatibility, match="(?i)unknown method") as excinfo:
        C.bootstrap_ci(np.zeros((10, 1)), np.array([0.0]), method="nope")
    assert isinstance(excinfo.value, ValueError)
    assert excinfo.value.diagnostics["method"] == "nope"


# ── logit ────────────────────────────────────────────────────────────


def test_logit_fit_predict_recovers_signal():
    rng = np.random.default_rng(5)
    n = 2000
    X = C.add_constant(rng.normal(size=(n, 2)))
    beta_true = np.array([0.2, 1.5, -1.0])
    p = 1 / (1 + np.exp(-(X @ beta_true)))
    d = (rng.uniform(size=n) < p).astype(float)
    beta, vcov = C.logit_fit(d, X)  # signature is logit_fit(y, X, ...)
    # Signs and rough magnitudes recovered; probabilities in (0, 1).
    assert np.sign(beta[1]) == 1 and np.sign(beta[2]) == -1
    phat = C.logit_predict(beta, X)
    assert np.all((phat > 0) & (phat < 1))


# ── formula / frame helpers ──────────────────────────────────────────


def test_parse_formula():
    dep, indep = C.parse_formula("y ~ x1 + x2 + 1 + x3")
    assert dep == "y"
    assert indep == ["x1", "x2", "x3"]  # intercept token filtered
    with pytest.raises(MethodIncompatibility, match="~") as excinfo:
        C.parse_formula("no tilde here")
    assert isinstance(excinfo.value, ValueError)
    assert excinfo.value.recovery_hint


def test_weighted_quantile_hmisc_validation_uses_taxonomy():
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(MethodIncompatibility, match="same length") as excinfo:
        C.weighted_quantile_hmisc(y, 0.5, w=np.ones(2))
    assert isinstance(excinfo.value, ValueError)
    assert excinfo.value.diagnostics == {"n_y": 3, "n_w": 2}

    with pytest.raises(MethodIncompatibility, match="between 0 and 1"):
        C.weighted_quantile_hmisc(y, 1.5)


def test_prepare_frame_drops_na_and_extracts_weights():
    import pandas as pd

    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, np.nan, 4.0],
            "x": [1.0, 2.0, 3.0, 4.0],
            "wt": [1.0, 1.0, 2.0, 3.0],
        }
    )
    d2, w = C.prepare_frame(df, ["y", "x"], weights="wt")
    assert len(d2) == 3 and len(w) == 3  # NA row dropped
    np.testing.assert_array_equal(w, np.array([1.0, 1.0, 3.0]))
    # weights=None → unit weights
    d3, w3 = C.prepare_frame(df.dropna(), ["y", "x"])
    assert np.all(w3 == 1.0)


def test_prepare_frame_missing_columns_use_taxonomy():
    import pandas as pd

    df = pd.DataFrame({"y": [1.0], "x": [2.0]})
    with pytest.raises(MethodIncompatibility, match="missing") as excinfo:
        C.prepare_frame(df, ["y", "missing"], weights="wt")
    assert isinstance(excinfo.value, ValueError)
    assert excinfo.value.diagnostics["missing_columns"] == ["missing", "wt"]


# ── generic / wild bootstrap drivers ─────────────────────────────────


def test_bootstrap_stat_recovers_sample_mean():
    rng = np.random.default_rng(7)
    y = rng.normal(3.0, 1.0, 300)
    boots = C.bootstrap_stat(
        lambda idx: float(y[idx].mean()), n=300, n_boot=400, rng=rng
    )
    assert boots.shape[0] == 400
    # bootstrap distribution centres on the sample statistic
    assert float(np.mean(boots)) == pytest.approx(y.mean(), abs=0.1)


def test_bootstrap_stat_strata_and_clusters():
    rng = np.random.default_rng(8)
    y = rng.normal(size=200)
    strata = np.arange(200) % 2
    clusters = np.repeat(np.arange(20), 10)
    b_s = C.bootstrap_stat(
        lambda idx: float(y[idx].mean()), n=200, n_boot=50, rng=rng, strata=strata
    )
    b_c = C.bootstrap_stat(
        lambda idx: float(y[idx].mean()), n=200, n_boot=50, rng=rng, clusters=clusters
    )
    assert b_s.shape[0] == 50 and b_c.shape[0] == 50


@pytest.mark.parametrize("weights", ["rademacher", "mammen"])
def test_wild_bootstrap_stat(weights):
    rng = np.random.default_rng(9)
    n = 150
    fitted = np.linspace(0, 1, n)
    resid = rng.normal(0, 0.3, n)
    boots = C.wild_bootstrap_stat(
        lambda ys: float(ys.mean()), resid, fitted, n_boot=100, rng=rng, weights=weights
    )
    assert boots.shape[0] == 100
    # E[y*] = fitted (multipliers are mean-zero), so the mean statistic
    # concentrates near mean(fitted) = 0.5.
    assert float(np.mean(boots)) == pytest.approx(0.5, abs=0.15)


# ── influence functions for inequality indices ───────────────────────


@pytest.mark.parametrize("stat", ["theil_t", "theil_l", "atkinson"])
def test_rif_inequality_recenters(stat):
    rng = np.random.default_rng(10)
    y = np.abs(rng.lognormal(0, 0.5, 500)) + 0.1
    w = np.ones_like(y)
    rif = C.influence_function(y, stat, w=w)
    assert float(np.mean(rif)) == pytest.approx(
        C.statistic_value(y, w, stat), rel=1e-6, abs=1e-8
    )


def test_rif_gini_recenters_approximately():
    # The Gini RIF uses a marginally different small-sample ECDF plug-in
    # than weighted_gini's Lerman–Yitzhaki F; they agree to O(1/n).
    rng = np.random.default_rng(10)
    y = np.abs(rng.lognormal(0, 0.5, 500)) + 0.1
    w = np.ones_like(y)
    rif_mean = float(np.mean(C.influence_function(y, "gini", w=w)))
    assert rif_mean == pytest.approx(C.statistic_value(y, w, "gini"), rel=2e-2)


def test_statistic_value_iqr():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    w = np.ones_like(y)
    iqr = C.statistic_value(y, w, "iqr")
    assert iqr == pytest.approx(
        C.weighted_quantile(y, 0.75) - C.weighted_quantile(y, 0.25)
    )
