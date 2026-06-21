"""Tests for the quantile / distributional treatment effect module.

Closes the audit gap of zero direct tests for `src/statspai/qte/*`. The
goal is API + numeric-direction coverage rather than full reference
parity (the bootstrap inference is too expensive for the CI budget).

Reference numbers we check against:

- **QTE constant-shift property**: when the DGP is :math:`Y = \\tau \\cdot D
  + \\varepsilon` for any iid :math:`\\varepsilon`, every quantile
  treatment effect equals :math:`\\tau`. Both ``method='quantile_regression'``
  (Firpo 2007) and ``method='distribution'`` (PS reweighting) must
  recover this.
- **QDID parallel-trends recovery**: when the trend is shared across
  groups, QDID(τ) at every τ should equal the post-treatment shift
  applied to the treated group (Athey & Imbens 2006 Prop 3).
- **Distributional TE shape**: ``distributional_te`` should return
  treated/control CDFs that are increasing on the support grid.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def constant_qte_data():
    """Y = 1.5*D + N(0,1)  ⇒  every QTE(τ) = 1.5 (Firpo benchmark)."""
    rng = np.random.default_rng(0)
    n = 1500
    D = rng.binomial(1, 0.5, n)
    Y = 1.5 * D + rng.normal(size=n)
    return pd.DataFrame({"y": Y, "d": D})


@pytest.fixture
def parallel_trends_data():
    """Two groups, two periods. Treated group gets a +2 shift in post.
    Control follows a parallel trend (+0.5 in post). QDID(τ) should
    therefore equal +2 - +0 = +1.5 at every τ (with bootstrap noise)."""
    rng = np.random.default_rng(1)
    n_per_cell = 400
    rows = []
    for g in (0, 1):
        for t in (0, 1):
            base = 0.0 + 0.5 * t  # parallel trend +0.5 per period
            shift = 1.5 if (g == 1 and t == 1) else 0.0
            y = base + shift + rng.normal(size=n_per_cell)
            for yi in y:
                rows.append({"y": yi, "g": g, "t": t})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# qte (Firpo 2007)
# --------------------------------------------------------------------- #


class TestQTE:

    def test_quantile_regression_recovers_constant_shift(self, constant_qte_data):
        res = sp.qte(
            constant_qte_data,
            y="y",
            treatment="d",
            quantiles=[0.25, 0.5, 0.75],
            method="quantile_regression",
            n_boot=50,
            seed=0,
        )
        assert res.effects.shape == (3,)
        # Every quantile effect should be close to 1.5
        for q, eff in zip(res.quantiles, res.effects):
            assert abs(eff - 1.5) < 0.30, f"QTE(τ={q}) = {eff:.3f} (want ≈1.5)"

    def test_distribution_method(self, constant_qte_data):
        res = sp.qte(
            constant_qte_data,
            y="y",
            treatment="d",
            quantiles=[0.5],
            method="distribution",
            n_boot=50,
            seed=0,
        )
        assert abs(res.effects[0] - 1.5) < 0.30

    def test_unknown_method_raises(self, constant_qte_data):
        with pytest.raises(ValueError, match="Unknown QTE method"):
            sp.qte(
                constant_qte_data,
                y="y",
                treatment="d",
                method="not_a_real_method",
                n_boot=10,
            )

    def test_result_object(self, constant_qte_data):
        res = sp.qte(
            constant_qte_data,
            y="y",
            treatment="d",
            quantiles=[0.5],
            method="quantile_regression",
            n_boot=20,
            seed=0,
        )
        assert hasattr(res, "summary")
        assert hasattr(res, "plot")
        # ATE should also be ~1.5 (sanity)
        assert abs(res.ate - 1.5) < 0.30


# --------------------------------------------------------------------- #
# qdid (Athey-Imbens 2006)
# --------------------------------------------------------------------- #


class TestQDiD:

    def test_recovers_parallel_trends_shift(self, parallel_trends_data):
        res = sp.qdid(
            parallel_trends_data,
            y="y",
            group="g",
            time="t",
            quantiles=[0.25, 0.5, 0.75],
            n_boot=50,
            seed=2,
        )
        # QDID(τ) ≈ 1.5 at every τ (the treated-group post-treatment shift)
        for q, eff in zip(res.quantiles, res.effects):
            assert abs(eff - 1.5) < 0.30, f"QDID(τ={q}) = {eff:.3f}"

    def test_too_few_observations_raises(self):
        df = pd.DataFrame({"y": [0.0], "g": [0], "t": [0]})
        with pytest.raises(ValueError, match="Too few observations"):
            sp.qdid(df, y="y", group="g", time="t", n_boot=2)


# --------------------------------------------------------------------- #
# distributional_te (Cunningham 2021 ch.7-style distributional contrast)
# --------------------------------------------------------------------- #


class TestDistributionalTE:

    def test_smoke_constant_shift(self, constant_qte_data):
        res = sp.distributional_te(
            constant_qte_data,
            y="y",
            treatment="d",
            method="ipw",
            n_boot=20,
        )
        # CDFs should be monotone non-decreasing on the grid.
        assert np.all(np.diff(res.cdf_treated) >= -1e-10)
        assert np.all(np.diff(res.cdf_counterfactual) >= -1e-10)
        # Treated CDF should be shifted right of counterfactual at the
        # median grid point (Y = 1.5*D + ε  ⇒  treated stochastically
        # dominates control).
        mid = len(res.grid) // 2
        assert res.cdf_counterfactual[mid] >= res.cdf_treated[mid]
