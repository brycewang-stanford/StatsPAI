"""Coverage tests for decomposition.inequality and decomposition.kitagawa.

Targets currently-uncovered branches: alternate inequality indices and
weighting paths, additive-vs-Gini subgroup contribution rules, result-object
renderers (summary / plot / to_latex / repr), validation raises, and the
Shapley random-permutation sampler for large covariate sets.

All assertions check real numerical identities:
  * inequality of a perfectly equal distribution is 0,
  * GE(2) == half-squared CV,
  * subgroup between+within(+overlap) == total index,
  * Lerman-Yitzhaki source contributions sum to the total Gini,
  * Das Gupta factor effects sum to the product-form gap,
  * Kitagawa rate+composition+interaction == raw gap.

import statspai as sp throughout (no mocks, deterministic seeds).
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")  # headless backend for .plot() branches

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.decomposition.inequality import (
    _atkinson,
    _ge_index,
    _weighted_pairwise_mad,
)

# ════════════════════════════════════════════════════════════════════════
# inequality_index — indices, equal-distribution identities, weight=None
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "index",
    ["theil_t", "theil_l", "mld", "ge0", "ge1", "ge2", "atkinson", "gini", "cv2"],
)
def test_perfectly_equal_distribution_has_zero_inequality(index):
    """Every index is exactly 0 for a perfectly equal distribution."""
    y = np.full(12, 7.0)
    val = sp.inequality_index(y, index=index)
    assert val == pytest.approx(0.0, abs=1e-12)


def test_ge2_equals_half_squared_cv():
    """GE(2) (alpha override and 'ge2'/'cv2' aliases) coincide."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    ge2_alpha = sp.inequality_index(y, alpha=2.0)
    ge2_name = sp.inequality_index(y, index="ge2")
    cv2 = sp.inequality_index(y, index="cv2")
    # Closed form: 0.5 * var / mu^2 ; mu=2.5, var=1.25 -> 0.1
    assert ge2_alpha == pytest.approx(0.1, abs=1e-12)
    assert ge2_name == pytest.approx(0.1, abs=1e-12)
    assert cv2 == pytest.approx(0.1, abs=1e-12)


def test_ge_index_weight_none_branch_matches_uniform_weights():
    """_ge_index with w=None (line 80) equals explicit uniform weights."""
    y = np.array([1.0, 2.0, 3.0, 5.0])
    assert _ge_index(y, 2.0) == pytest.approx(
        _ge_index(y, 2.0, np.ones_like(y)), abs=1e-12
    )


def test_atkinson_weight_none_branch_eps_not_one():
    """_atkinson(eps!=1) with w=None (line 95) matches uniform weights.

    Closed form for eps=0.5: A = 1 - (E[y^0.5])^2 / mu.
    """
    y = np.array([1.0, 2.0, 3.0])
    got = _atkinson(y, 0.5)
    p = 0.5  # 1 - eps
    mu = y.mean()
    expected = 1.0 - (np.mean(y**p) ** (1.0 / p)) / mu
    assert got == pytest.approx(expected, abs=1e-12)
    assert got == pytest.approx(_atkinson(y, 0.5, np.ones_like(y)), abs=1e-12)


def test_inequality_index_unknown_raises():
    with pytest.raises(ValueError, match="unknown index"):
        sp.inequality_index(np.arange(5.0), index="not_an_index")


# ════════════════════════════════════════════════════════════════════════
# subgroup_decompose — additive contribution rules + Gini (Dagum) overlap
# ════════════════════════════════════════════════════════════════════════


def _grouped_frame(seed=0, n=200, ngroups=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "y": rng.lognormal(0.0, 0.5, n),
            "g": rng.integers(0, ngroups, n),
        }
    )


@pytest.mark.parametrize(
    "index", ["theil_t", "ge1", "theil_l", "mld", "ge0", "ge2", "cv2", "atkinson"]
)
def test_additive_subgroup_between_plus_within_equals_total(index):
    """Additive GE/Theil/MLD/CV/Atkinson: between+within == total exactly.

    Exercises every contribution branch (theil_t/ge1, theil_l/mld/ge0,
    ge2/cv2, and the general fallback for atkinson).
    """
    df = _grouped_frame()
    r = sp.subgroup_decompose(df, y="y", by="g", index=index)
    assert r.between + r.within == pytest.approx(r.total, abs=1e-12)
    assert r.overlap is None
    # Per-group contributions sum to the within component.
    assert float(r.per_group["contribution"].sum()) == pytest.approx(
        r.within, abs=1e-12
    )


def test_subgroup_contribution_sum_matches_within():
    """Explicit check that per-group contributions sum to `within`."""
    df = _grouped_frame(seed=3)
    r = sp.subgroup_decompose(df, y="y", by="g", index="theil_t")
    assert float(r.per_group["contribution"].sum()) == pytest.approx(
        r.within, abs=1e-12
    )


def test_weighted_pairwise_mad_empty_group_is_zero():
    """Empty-group guard (line 304): MAD against an empty sample is 0."""
    yk = np.array([1.0, 2.0, 5.0])
    wk = np.array([1.0, 1.0, 1.0])
    assert _weighted_pairwise_mad(np.array([]), np.array([]), yk, wk) == 0.0
    assert _weighted_pairwise_mad(yk, wk, np.array([]), np.array([])) == 0.0


def test_gini_subgroup_components_sum_to_total_and_summary_overlap():
    """Dagum Gini: between+within+overlap == total; summary prints overlap."""
    df = _grouped_frame(seed=1)
    r = sp.subgroup_decompose(df, y="y", by="g", index="gini")
    assert r.overlap is not None
    assert r.between + r.within + r.overlap == pytest.approx(r.total, abs=1e-9)
    # summary() (line 186) appends the overlap line for the Gini case.
    txt = r.summary()
    assert "Overlap:" in txt
    assert "Inequality Subgroup Decomposition" in txt


# ════════════════════════════════════════════════════════════════════════
# source_decompose (Lerman-Yitzhaki) — renderers + additivity
# ════════════════════════════════════════════════════════════════════════


def test_source_decompose_contributions_sum_to_total_gini():
    rng = np.random.default_rng(2)
    n = 150
    df = pd.DataFrame(
        {
            "wage": rng.lognormal(0.0, 0.4, n),
            "capital": rng.lognormal(0.2, 0.3, n),
        }
    )
    r = sp.source_decompose(df, ["wage", "capital"])
    assert r.sources["contribution"].sum() == pytest.approx(r.total_gini, abs=1e-12)
    # __repr__ (line 458)
    rep = repr(r)
    assert "SourceDecompResult" in rep and "n_sources=2" in rep
    # plot() (lines 423-424) returns a matplotlib artifact without error.
    ax = r.plot()
    assert ax is not None
    # to_latex / summary render without error and reference the total.
    assert "Total" in r.to_latex()
    assert "Gini Source Decomposition" in r.summary()


# ════════════════════════════════════════════════════════════════════════
# shapley_inequality — renderers, unknown-index raise, large-k sampler
# ════════════════════════════════════════════════════════════════════════


def test_shapley_renderers_and_repr():
    rng = np.random.default_rng(4)
    n = 150
    df = pd.DataFrame(
        {
            "y": rng.lognormal(0.0, 0.5, n),
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
        }
    )
    r = sp.shapley_inequality(df, "y", ["x1", "x2"], index="theil_t")
    assert r.total > 0
    assert len(r.shapley) == 2
    # __repr__ (line 577)
    assert "ShapleyInequalityResult" in repr(r)
    # plot() (lines 547-548)
    ax = r.plot()
    assert ax is not None


def test_shapley_unknown_index_raises():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x1": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="unknown index"):
        sp.shapley_inequality(df, "y", ["x1"], index="bogus")


def test_shapley_large_k_uses_random_permutation_sampler():
    """k>10 (lines 645-659) warns and uses the 500-permutation sampler."""
    rng = np.random.default_rng(7)
    n, k = 60, 11
    data = {"y": rng.lognormal(0.0, 0.5, n)}
    for j in range(k):
        data[f"x{j}"] = rng.normal(0, 1, n)
    df = pd.DataFrame(data)
    xcols = [f"x{j}" for j in range(k)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = sp.shapley_inequality(df, "y", xcols, index="gini")
    msgs = [str(w.message) for w in caught]
    assert any("random permutations" in m for m in msgs)
    assert len(r.shapley) == k
    assert r.total > 0
    # Sampled Shapley values are finite real numbers.
    assert np.all(np.isfinite(r.shapley["contribution"].to_numpy()))


# ════════════════════════════════════════════════════════════════════════
# kitagawa_decompose — additivity, normalisations, branches, renderers
# ════════════════════════════════════════════════════════════════════════


def _preagg_frame():
    return pd.DataFrame(
        {
            "rate": [0.1, 0.3, 0.2, 0.4],
            "grp": [0, 0, 1, 1],
            "cat": ["x", "y", "x", "y"],
            "pop": [100.0, 100.0, 50.0, 150.0],
        }
    )


def test_kitagawa_preaggregated_components_sum_to_gap():
    """Pre-aggregated weights path (lines 147-149); components sum to gap."""
    df = _preagg_frame()
    r = sp.kitagawa_decompose(
        df, rate="rate", group="grp", by="cat", weights="pop", normalize="symmetric"
    )
    assert r.gap == pytest.approx(r.rate_a - r.rate_b, abs=1e-12)
    assert r.rate_effect + r.composition_effect + r.interaction == pytest.approx(
        r.gap, abs=1e-12
    )
    # Symmetric per-cell contributions sum to the aggregate effects.
    assert r.per_cell["rate_contrib"].sum() == pytest.approx(r.rate_effect, abs=1e-12)
    assert r.per_cell["comp_contrib"].sum() == pytest.approx(
        r.composition_effect, abs=1e-12
    )
    # summary() / __repr__ (lines 54, 69-71, 91)
    txt = r.summary()
    assert "Kitagawa Decomposition" in txt and "Rate effect" in txt
    assert "KitagawaResult" in repr(r)


@pytest.mark.parametrize("normalize", ["a", "b"])
def test_kitagawa_normalize_a_and_b_still_sum_to_gap(normalize):
    """normalize='a'/'b' branches (lines 184-189) preserve additivity."""
    df = _preagg_frame()
    r = sp.kitagawa_decompose(
        df, rate="rate", group="grp", by="cat", weights="pop", normalize=normalize
    )
    assert r.rate_effect + r.composition_effect + r.interaction == pytest.approx(
        r.gap, abs=1e-12
    )


def test_kitagawa_individual_level_aggregation():
    """weights=None individual-level path; components sum to the gap."""
    df = pd.DataFrame(
        {
            "y": [1, 0, 1, 1, 0, 1, 0, 0],
            "g": [0, 0, 0, 0, 1, 1, 1, 1],
            "cat": ["a", "a", "b", "b", "a", "a", "b", "b"],
        }
    )
    r = sp.kitagawa_decompose(df, rate="y", group="g", by="cat")
    # Group 0 overall = mean of [1,0,1,1]=0.75 ; group 1 = [0,1,0,0]=0.25.
    assert r.rate_a == pytest.approx(0.75, abs=1e-12)
    assert r.rate_b == pytest.approx(0.25, abs=1e-12)
    assert r.gap == pytest.approx(0.5, abs=1e-12)
    assert r.rate_effect + r.composition_effect + r.interaction == pytest.approx(
        r.gap, abs=1e-12
    )


def test_kitagawa_zero_population_raises():
    """All mass in one group -> empty other group -> ValueError (line 163)."""
    df = pd.DataFrame(
        {
            "y": [0.1, 0.2],
            "g": [0, 0],
            "cat": ["a", "b"],
            "pop": [10.0, 10.0],
        }
    )
    with pytest.raises(ValueError, match="Zero population"):
        sp.kitagawa_decompose(df, rate="y", group="g", by="cat", weights="pop")


# ════════════════════════════════════════════════════════════════════════
# das_gupta — product-form additivity, renderers, validation
# ════════════════════════════════════════════════════════════════════════


def test_das_gupta_effects_sum_to_gap_and_renderers():
    """Two-factor product form: effects sum to the gap; renderers run."""
    da = pd.DataFrame({"f1": [2.0], "f2": [3.0]})
    db = pd.DataFrame({"f1": [1.0], "f2": [5.0]})
    r = sp.das_gupta(da, db, ["f1", "f2"])
    assert r.rate_a == pytest.approx(6.0, abs=1e-12)
    assert r.rate_b == pytest.approx(5.0, abs=1e-12)
    assert r.gap == pytest.approx(1.0, abs=1e-12)
    assert r.factor_effects["effect"].sum() == pytest.approx(r.gap, abs=1e-12)
    # summary() (lines 234, 245-247) and to_latex (250-251) render.
    txt = r.summary()
    assert "Das Gupta" in txt
    assert "tabular" in r.to_latex()
    # __repr__ (line 282) and _repr_html_
    assert "DasGuptaResult" in repr(r)
    assert "Das Gupta" in r._repr_html_()
    # plot() (lines 250-251) returns a matplotlib artifact without error.
    ax = r.plot()
    assert ax is not None


def test_das_gupta_no_factors_raises():
    with pytest.raises(ValueError, match="Need"):
        sp.das_gupta(pd.DataFrame({"f": [1.0]}), pd.DataFrame({"f": [2.0]}), [])
