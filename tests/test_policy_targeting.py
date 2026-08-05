"""Tests for ``sp.policy_targeting`` (budget-constrained CATE targeting)."""

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

TAU = np.array([2.0, 1.0, 0.5, -0.5, -2.0])


class TestCorrectness:
    def test_selects_top_effects(self):
        out = sp.policy_targeting(TAU, budget=2)
        assert out["policy"].tolist() == [1, 1, 0, 0, 0]
        assert out["expected_gain"] == pytest.approx(3.0)
        assert out["threshold"] == pytest.approx(1.0)

    def test_baselines(self):
        out = sp.policy_targeting(TAU, budget=2)
        assert out["expected_gain_treat_all"] == pytest.approx(TAU.sum())
        assert out["expected_gain_random"] == pytest.approx(2 * TAU.mean())

    def test_min_effect_guard_binds_before_budget(self):
        # Budget of 5 but only 3 units have positive predicted effect.
        out = sp.policy_targeting(TAU, budget=5)
        assert out["n_treated"] == 3
        assert out["policy"].tolist() == [1, 1, 1, 0, 0]

    def test_frac_interface(self):
        out = sp.policy_targeting(TAU, frac=0.4)
        assert out["budget"] == 2
        assert out["n_treated"] == 2

    def test_unconstrained_default_treats_positive_only(self):
        out = sp.policy_targeting(TAU)
        assert out["n_treated"] == 3

    def test_min_effect_threshold(self):
        out = sp.policy_targeting(TAU, min_effect=0.75)
        assert out["policy"].tolist() == [1, 1, 0, 0, 0]

    def test_input_order_preserved(self):
        shuffled = np.array([-2.0, 2.0, -0.5, 1.0, 0.5])
        out = sp.policy_targeting(shuffled, budget=2)
        assert out["policy"].tolist() == [0, 1, 0, 1, 0]

    def test_summary_frame(self):
        out = sp.policy_targeting(TAU, budget=2)
        row = out["summary"].iloc[0]
        assert row["n"] == 5
        assert row["n_treated"] == 2


class TestBoundaries:
    def test_both_budget_and_frac_raise(self):
        with pytest.raises(MethodIncompatibility, match="not both"):
            sp.policy_targeting(TAU, budget=2, frac=0.5)

    def test_bad_frac_raises(self):
        with pytest.raises(MethodIncompatibility, match="frac"):
            sp.policy_targeting(TAU, frac=1.5)

    def test_negative_budget_raises(self):
        with pytest.raises(MethodIncompatibility, match="non-negative"):
            sp.policy_targeting(TAU, budget=-1)

    def test_empty_raises(self):
        with pytest.raises(MethodIncompatibility, match="empty"):
            sp.policy_targeting(np.array([]))

    def test_nan_raises(self):
        with pytest.raises(MethodIncompatibility, match="finite"):
            sp.policy_targeting(np.array([1.0, np.nan]))

    def test_budget_zero(self):
        out = sp.policy_targeting(TAU, budget=0)
        assert out["n_treated"] == 0
        assert out["expected_gain"] == 0.0
        assert np.isnan(out["threshold"])

    def test_accepts_cate_result(self):
        rng = np.random.default_rng(0)
        n = 200
        df_x = rng.normal(size=(n, 2))
        d = rng.binomial(1, 0.5, n)
        y = (1 + df_x[:, 0]) * d + rng.normal(0, 0.5, n)
        cf = sp.causal_forest(Y=y, T=d, X=df_x, n_estimators=30, random_state=0)
        out = sp.policy_targeting(cf, frac=0.5)
        assert out["n_treated"] <= n // 2
        assert out["expected_gain"] >= out["expected_gain_random"]
