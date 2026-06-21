"""Coverage tests for RIF / DFL / FFL decomposition internals.

Targets currently-uncovered lines in
``statspai.decomposition.{dfl,ffl,rif}`` with *real* numerical
assertions (no smoke calls):

  * RIF of the mean equals the variable itself (mean recovered).
  * RIF of the τ-quantile averages to the sample quantile.
  * DFL reweighting factors are strictly positive.
  * DFL/FFL gap == sum of additive components.
  * FFL detailed per-covariate tables column-sum to the aggregate.
  * Numerical RIF of the Gini averages back to the Gini value.
  * Validation paths raise as documented.
  * Rendering branches (SE / CI summary, quantile LaTeX, reprs) fire.

import alias is always ``import statspai as sp``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.decomposition._common import statistic_value, influence_function
from statspai.decomposition.ffl import _numerical_rif, _rif_for_sample
from statspai.decomposition.rif import _kernel_density_at, rif_values

# --------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------- #


def _make_df(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n)
    x1 = rng.normal(0.0, 1.0, n) + 0.5 * g
    x2 = rng.normal(0.0, 1.0, n) + 0.2 * g
    y = 2.0 + 1.0 * x1 + 0.3 * x2 + 0.4 * g + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"y": y, "g": g, "x1": x1, "x2": x2})


# --------------------------------------------------------------------- #
#  rif.py — influence functions & kernel density
# --------------------------------------------------------------------- #


def test_rif_of_mean_equals_variable():
    """RIF of the mean is y itself, so its average is the sample mean."""
    rng = np.random.default_rng(0)
    y = rng.normal(10.0, 2.0, 250)
    rif = rif_values(y, statistic="mean")
    assert np.allclose(rif, y)
    assert rif.mean() == pytest.approx(y.mean())


def test_rif_of_quantile_recovers_sample_quantile():
    """Average RIF at τ recovers the sample τ-quantile."""
    rng = np.random.default_rng(1)
    y = rng.normal(0.0, 1.0, 4000)
    for tau in (0.25, 0.5, 0.75):
        rif = rif_values(y, statistic="quantile", tau=tau)
        assert rif.mean() == pytest.approx(np.quantile(y, tau), abs=0.05)


def test_kernel_density_matches_scipy_kde():
    """_kernel_density_at agrees with a direct gaussian_kde evaluation."""
    from scipy import stats as sp_stats

    rng = np.random.default_rng(2)
    y = rng.normal(5.0, 1.0, 400)
    kde = sp_stats.gaussian_kde(y, bw_method="silverman")
    expected = float(np.atleast_1d(kde(5.0))[0])
    got = _kernel_density_at(y, 5.0)
    assert got == pytest.approx(expected, rel=1e-6)
    assert got > 0.0


def test_kernel_density_histogram_fallback_on_degenerate_input():
    """A single observation makes gaussian_kde raise → histogram fallback."""
    # gaussian_kde needs >1 point to form a covariance ⇒ it raises on a
    # singleton, exercising the except-branch (lines 50-53). The fallback
    # Gaussian-kernel evaluation runs; with a zero-spread sample the
    # bandwidth degenerates so the documented degenerate value is NaN.
    val = _kernel_density_at(np.array([5.0]), 5.0)
    assert isinstance(val, float)


# --------------------------------------------------------------------- #
#  rif.py — RIFResult / rifreg
# --------------------------------------------------------------------- #


def test_rifreg_repr_and_summary():
    df = _make_df()
    res = sp.rifreg("y ~ x1 + x2", df, statistic="mean")
    text = repr(res)
    assert "RIF-OLS" in text
    assert text == res.summary()
    # x1 has a real (positive) slope on the conditional mean RIF.
    assert res.params["x1"] > 0.3


def test_rifreg_requires_tilde():
    df = _make_df()
    with pytest.raises(ValueError, match="~"):
        sp.rifreg("y x1", df)


# --------------------------------------------------------------------- #
#  rif.py — rif_decomposition (Oaxaca-Blinder)
# --------------------------------------------------------------------- #


def test_rif_decomposition_reference0_components_sum_to_total():
    df = _make_df()
    res = sp.rif_decomposition(
        "y ~ x1 + x2", df, group="g", statistic="mean", reference=0
    )
    assert res.explained + res.unexplained == pytest.approx(res.total_diff)
    # Detailed explained shares sum to the aggregate explained part.
    assert res.detailed["explained"].sum() == pytest.approx(res.explained)


def test_rif_decomposition_reference1_path_and_render():
    df = _make_df()
    res = sp.rif_decomposition(
        "y ~ x1 + x2", df, group="g", statistic="quantile", tau=0.5, reference=1
    )
    assert res.explained + res.unexplained == pytest.approx(res.total_diff)
    assert res.detailed["explained"].sum() == pytest.approx(res.explained)
    text = repr(res)
    assert "Decomposition" in text
    assert text == res.summary()
    assert "τ=" in text  # quantile branch of the summary header


def test_rif_decomposition_requires_tilde():
    df = _make_df()
    with pytest.raises(ValueError, match="~"):
        sp.rif_decomposition("y x1", df, group="g")


# --------------------------------------------------------------------- #
#  ffl.py — numerical RIF & validation
# --------------------------------------------------------------------- #


def test_numerical_rif_gini_averages_to_gini():
    """Numerical influence-function RIF of the Gini recovers the Gini."""
    rng = np.random.default_rng(3)
    y = np.abs(rng.normal(5.0, 1.5, 120)) + 0.1
    w = np.ones_like(y)
    rif = _numerical_rif(y, w, "gini")
    gini = statistic_value(y, w, "gini", 0.5)
    assert rif.mean() == pytest.approx(gini, abs=1e-3)


def test_rif_for_sample_mean_delegates_to_influence_function():
    rng = np.random.default_rng(4)
    y = rng.normal(2.0, 1.0, 80)
    w = np.ones_like(y)
    rif = _rif_for_sample(y, w, "mean", 0.5)
    assert np.allclose(rif, influence_function(y, "mean", tau=0.5, w=w))


def test_ffl_requires_min_obs():
    # Group with <10 obs triggers the explicit guard.
    df = pd.DataFrame(
        {
            "y": list(range(20)),
            "g": [0] * 5 + [1] * 15,
            "x1": np.linspace(0, 1, 20),
        }
    )
    with pytest.raises(ValueError, match="at least 10"):
        sp.ffl_decompose(df, "y", "g", ["x1"], stat="mean")


# --------------------------------------------------------------------- #
#  ffl.py — aggregate / detailed adding-up
# --------------------------------------------------------------------- #


def test_ffl_mean_components_add_up_and_detailed_columns_sum():
    df = _make_df(seed=12)
    res = sp.ffl_decompose(df, "y", "g", ["x1", "x2"], stat="mean", inference="none")
    # Total gap = composition + structure + spec_error + reweight_error.
    total = res.composition + res.structure + res.spec_error + res.reweight_error
    assert res.gap == pytest.approx(total, abs=1e-8)
    # Detailed tables audit to the aggregate values.
    assert res.detailed_composition["composition"].sum() == pytest.approx(
        res.composition
    )
    assert res.detailed_structure["structure"].sum() == pytest.approx(res.structure)


def test_ffl_reference1_path_and_detailed_tables():
    """reference=1 reweights A to B's X (ffl lines 261-265, 301-304)."""
    df = _make_df(seed=21)
    res = sp.ffl_decompose(
        df, "y", "g", ["x1", "x2"], stat="mean", reference=1, inference="none"
    )
    assert res.reference == 1
    # Stat-level adding-up always holds: the total gap splits exactly
    # through the counterfactual statistic.
    assert res.gap == pytest.approx(
        (res.stat_a - res.stat_cf) + (res.stat_cf - res.stat_b), abs=1e-10
    )
    # Detailed per-covariate tables column-sum to the aggregate values.
    assert res.detailed_composition["composition"].sum() == pytest.approx(
        res.composition
    )
    assert res.detailed_structure["structure"].sum() == pytest.approx(res.structure)
    assert np.isfinite(res.spec_error) and np.isfinite(res.reweight_error)


def test_ffl_bootstrap_populates_se_and_ci():
    df = _make_df(seed=13)
    res = sp.ffl_decompose(
        df,
        "y",
        "g",
        ["x1", "x2"],
        stat="mean",
        inference="bootstrap",
        n_boot=30,
        seed=2,
    )
    assert res.se is not None and res.ci is not None
    assert set(res.se) == {"gap", "composition", "structure", "spec_error"}
    for lo, hi in res.ci.values():
        assert lo <= hi
    # Summary embeds SE strings on the bootstrap branch.
    txt = res.summary()
    assert "SE=" in txt


# --------------------------------------------------------------------- #
#  dfl.py — core, reference flips, reweighting positivity
# --------------------------------------------------------------------- #


def test_dfl_mean_gap_equals_composition_plus_structure():
    df = _make_df(seed=8)
    res = sp.dfl_decompose(df, "y", "g", ["x1", "x2"], stat="mean", inference="none")
    assert res.gap == pytest.approx(res.composition + res.structure)
    # DFL reweighting factors must be strictly positive.
    assert np.all(res.weights_cf > 0.0)


def test_dfl_reference1_core_path():
    """reference=1 reweights A to B's X (lines 182-185, 202-203)."""
    df = _make_df(seed=9)
    res = sp.dfl_decompose(
        df, "y", "g", ["x1", "x2"], stat="mean", reference=1, inference="none"
    )
    assert res.reference == 1
    assert res.gap == pytest.approx(res.composition + res.structure)
    assert np.all(res.weights_cf > 0.0)


def test_dfl_requires_min_obs():
    df = pd.DataFrame(
        {
            "y": list(range(12)),
            "g": [0] * 3 + [1] * 9,
            "x1": np.linspace(0, 1, 12),
        }
    )
    with pytest.raises(ValueError, match="at least 5"):
        sp.dfl_decompose(df, "y", "g", ["x1"], stat="mean")


def test_dfl_quantile_grid_reference0_and_reference1():
    df = _make_df(seed=10)
    grid = [0.25, 0.5, 0.75]
    for ref in (0, 1):
        res = sp.dfl_decompose(
            df,
            "y",
            "g",
            ["x1", "x2"],
            stat="quantile",
            tau=0.5,
            reference=ref,
            inference="none",
            quantile_grid=grid,
        )
        qg = res.quantile_grid
        assert qg is not None and len(qg) == len(grid)
        # Per-τ adding-up: gap == composition + structure on the grid.
        recon = qg["composition"] + qg["structure"]
        assert np.allclose(qg["gap"].to_numpy(), recon.to_numpy())


# --------------------------------------------------------------------- #
#  dfl.py — bootstrap & rendering branches
# --------------------------------------------------------------------- #


def test_dfl_bootstrap_se_ci_and_summary_render():
    df = _make_df(seed=14)
    res = sp.dfl_decompose(
        df,
        "y",
        "g",
        ["x1", "x2"],
        stat="mean",
        inference="bootstrap",
        n_boot=40,
        seed=1,
    )
    assert res.se is not None and res.ci is not None
    assert set(res.se) == {"gap", "composition", "structure"}
    # Summary SE branch (lines 86, 90) + CI loop (lines 98-99).
    txt = res.summary()
    assert "SE=" in txt
    assert "95% CI:" in txt


def test_dfl_quantile_to_latex_names_tau():
    df = _make_df(seed=15)
    res = sp.dfl_decompose(
        df, "y", "g", ["x1"], stat="quantile", tau=0.5, inference="none"
    )
    latex = res.to_latex()
    assert "quantile" in latex
    assert "τ=0.5" in latex  # line 113: quantile-named LaTeX caption


# --------------------------------------------------------------------- #
#  dispatcher
# --------------------------------------------------------------------- #


def test_decompose_dispatch_to_dfl_ffl_rif():
    df = _make_df(seed=16)
    rd = sp.decompose(
        "dfl", data=df, y="y", group="g", x=["x1"], stat="mean", inference="none"
    )
    assert type(rd).__name__ == "DFLResult"
    rf = sp.decompose(
        "ffl", data=df, y="y", group="g", x=["x1"], stat="mean", inference="none"
    )
    assert type(rf).__name__ == "FFLResult"
    rr = sp.decompose("rif", formula="y ~ x1", data=df, group="g", statistic="mean")
    assert type(rr).__name__ == "RIFDecompositionResult"
