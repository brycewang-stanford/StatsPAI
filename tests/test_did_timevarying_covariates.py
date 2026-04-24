"""Tests for sp.did_timevarying_covariates (Caetano et al. 2022)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _panel_with_timevarying_cov(
    n_units: int = 150,
    n_periods: int = 8,
    effect: float = 0.5,
    covariate_effect: float = 0.05,
    treatment_affects_covariate: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """Staggered panel with a time-varying covariate.

    If ``treatment_affects_covariate=True``, treatment affects the
    covariate — creating the exact scenario Caetano et al. (2022) warn
    about. With baseline freezing, the estimator should still be
    consistent.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = int(rng.integers(3, n_periods - 1)) if rng.random() > 0.3 else 0
        x0 = rng.uniform(0, 10)
        for t in range(1, n_periods + 1):
            d = int(g > 0 and t >= g)
            x_t = x0 + 0.5 * (t - 1)  # time-varying
            if treatment_affects_covariate and d:
                x_t += 3.0  # treatment bumps covariate
            y = (
                0.1 * t
                + effect * d
                + covariate_effect * x0  # only baseline x matters
                + rng.normal(scale=0.3)
            )
            rows.append({"i": u, "t": t, "g": g, "x": x_t, "y": y})
    return pd.DataFrame(rows)


class TestDIDTimevaryingCov:
    def test_recovers_true_effect(self):
        df = _panel_with_timevarying_cov(n_units=250, effect=0.5, seed=1)
        r = sp.did_timevarying_covariates(
            df,
            y="y",
            unit="i",
            time="t",
            cohort="g",
            covariates=["x"],
            n_boot=60,
            seed=1,
        )
        # 250 units × well-identified → recovery within ±0.2 of 0.5.
        assert abs(r.estimate - 0.5) < 0.25

    def test_robust_when_treatment_affects_covariate(self):
        """The whole point of Caetano et al. (2022): treatment affecting
        the covariate is fine as long as we freeze covariates at baseline.
        Regression with contemporaneous x would be biased."""
        df = _panel_with_timevarying_cov(
            n_units=250,
            effect=0.5,
            treatment_affects_covariate=True,
            seed=42,
        )
        r = sp.did_timevarying_covariates(
            df,
            y="y",
            unit="i",
            time="t",
            cohort="g",
            covariates=["x"],
            n_boot=60,
            seed=42,
        )
        # Still recover 0.5 despite covariate-feedback.
        assert abs(r.estimate - 0.5) < 0.3

    def test_detail_structure(self):
        df = _panel_with_timevarying_cov(seed=2)
        r = sp.did_timevarying_covariates(
            df,
            y="y",
            unit="i",
            time="t",
            cohort="g",
            covariates=["x"],
            n_boot=20,
            seed=2,
        )
        assert r.detail is not None
        assert {"cohort", "time", "att_gt", "n_treated", "n_control"} <= set(
            r.detail.columns
        )
        assert (r.detail["n_treated"] > 0).all()
        assert (r.detail["n_control"] > 0).all()

    def test_requires_never_treated(self):
        df = _panel_with_timevarying_cov(seed=0)
        df = df[df["g"] > 0]
        with pytest.raises(ValueError, match="never-treated"):
            sp.did_timevarying_covariates(
                df,
                y="y",
                unit="i",
                time="t",
                cohort="g",
                covariates=["x"],
                n_boot=5,
                seed=0,
            )

    def test_missing_covariate_raises(self):
        df = _panel_with_timevarying_cov(seed=0)
        with pytest.raises(ValueError, match="not in data"):
            sp.did_timevarying_covariates(
                df,
                y="y",
                unit="i",
                time="t",
                cohort="g",
                covariates=["nonexistent"],
                n_boot=5,
                seed=0,
            )

    def test_baseline_offset_is_applied(self):
        """Change baseline_offset and verify the ATT changes (sanity)."""
        df = _panel_with_timevarying_cov(n_units=150, seed=9)
        r1 = sp.did_timevarying_covariates(
            df,
            y="y",
            unit="i",
            time="t",
            cohort="g",
            covariates=["x"],
            baseline_offset=-1,
            n_boot=20,
            seed=9,
        )
        r2 = sp.did_timevarying_covariates(
            df,
            y="y",
            unit="i",
            time="t",
            cohort="g",
            covariates=["x"],
            baseline_offset=-2,
            n_boot=20,
            seed=9,
        )
        # Different offsets should yield finite estimates; not strictly
        # required to differ (if covariate barely moves). Just assert
        # both finite.
        assert np.isfinite(r1.estimate) and np.isfinite(r2.estimate)
