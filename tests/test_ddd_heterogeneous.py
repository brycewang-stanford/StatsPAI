"""Tests for sp.ddd_heterogeneous (Olden-Møen 2022 style)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _panel_with_subgroups(
    n_units: int = 120,
    n_periods: int = 8,
    effect_on_affected: float = 1.0,
    placebo_trend: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Staggered panel with a within-unit subgroup flag.

    Affected subgroup (b=1): true effect = effect_on_affected when treated.
    Unaffected subgroup (b=0): no true effect; optional differential
    calendar-time trend = placebo_trend to test DDD robustness.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = int(rng.integers(3, n_periods - 1)) if rng.random() > 0.3 else 0
        b = int(rng.random() > 0.5)
        for t in range(1, n_periods + 1):
            d = int(g > 0 and t >= g)
            eff = effect_on_affected * d * b
            # Time trend differs between subgroups to create a DID bias
            # that DDD should wash out.
            trend = 0.1 * t + placebo_trend * t * (1 - b)
            y = trend + eff + rng.normal(scale=0.3)
            rows.append({"i": u, "t": t, "g": g, "b": b, "y": y})
    return pd.DataFrame(rows)


class TestDDDHeterogeneous:
    def test_recovers_true_effect_approx(self):
        df = _panel_with_subgroups(n_units=300, effect_on_affected=1.0, seed=7)
        r = sp.ddd_heterogeneous(
            df,
            y="y",
            unit="i",
            time="t",
            cohort="g",
            subgroup="b",
            n_boot=80,
            seed=7,
        )
        # True effect = 1.0. DDD with 300 units should land within ~0.5 of
        # truth (wider than DiD because estimator subtracts a noisy placebo).
        assert abs(r.estimate - 1.0) < 0.5

    def test_robust_to_subgroup_specific_trend(self):
        """When unaffected subgroup has a differential trend (bias for naive
        DID) but affected-vs-unaffected share the same trend, DDD should
        recover the true effect better than raw DID on the affected slice."""
        df = _panel_with_subgroups(
            n_units=300,
            effect_on_affected=1.0,
            placebo_trend=0.3,  # Creates bias in subgroup-stratified DID.
            seed=11,
        )
        r = sp.ddd_heterogeneous(
            df,
            y="y",
            unit="i",
            time="t",
            cohort="g",
            subgroup="b",
            n_boot=80,
            seed=11,
        )
        # Recovery bound is wide because DDD has higher variance than DID
        # in this DGP, but the estimator should not be grossly biased by
        # the subgroup-specific trend.
        assert abs(r.estimate - 1.0) < 0.6

    def test_detail_frame_structure(self):
        df = _panel_with_subgroups(seed=3)
        r = sp.ddd_heterogeneous(
            df,
            y="y",
            unit="i",
            time="t",
            cohort="g",
            subgroup="b",
            n_boot=30,
            seed=3,
        )
        assert r.detail is not None
        assert {
            "cohort",
            "time",
            "did_affected",
            "did_placebo",
            "ddd",
            "n_treated_affected",
        } <= set(r.detail.columns)
        # Every DDD cell should satisfy ddd = did_affected - did_placebo
        diff = r.detail["did_affected"] - r.detail["did_placebo"]
        np.testing.assert_allclose(diff.values, r.detail["ddd"].values, rtol=1e-10)

    def test_placebo_joint_test_shape(self):
        df = _panel_with_subgroups(seed=4)
        r = sp.ddd_heterogeneous(
            df,
            y="y",
            unit="i",
            time="t",
            cohort="g",
            subgroup="b",
            n_boot=80,
            seed=4,
        )
        plac = r.model_info.get("placebo_joint_test")
        if plac is not None:
            assert "statistic" in plac and "df" in plac and "pvalue" in plac
            assert plac["df"] == r.model_info["n_cells"]
            assert 0 <= plac["pvalue"] <= 1

    def test_requires_never_treated(self):
        df = _panel_with_subgroups(seed=0).copy()
        # Remove all never-treated units
        df = df[df["g"] > 0]
        with pytest.raises(ValueError, match="never-treated"):
            sp.ddd_heterogeneous(
                df,
                y="y",
                unit="i",
                time="t",
                cohort="g",
                subgroup="b",
                n_boot=10,
                seed=0,
            )

    def test_requires_binary_subgroup(self):
        df = _panel_with_subgroups(seed=0).copy()
        df.loc[df.index[0], "b"] = 2
        with pytest.raises(ValueError, match="binary"):
            sp.ddd_heterogeneous(
                df,
                y="y",
                unit="i",
                time="t",
                cohort="g",
                subgroup="b",
                n_boot=10,
                seed=0,
            )

    def test_zero_effect_ci_broadly_covers_zero(self):
        covers = 0
        trials = 4
        for s in range(trials):
            df = _panel_with_subgroups(
                n_units=200,
                effect_on_affected=0.0,
                seed=200 + s,
            )
            r = sp.ddd_heterogeneous(
                df,
                y="y",
                unit="i",
                time="t",
                cohort="g",
                subgroup="b",
                n_boot=100,
                seed=200 + s,
            )
            if r.ci[0] <= 0 <= r.ci[1]:
                covers += 1
        # Tolerance is loose — 2/4 is acceptable given noise.
        assert covers >= 2, f"Only {covers}/{trials} CIs covered zero"
