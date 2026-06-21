"""Tests for v1.13 RDD polish — sp.rd_flex / sp.rd_bias_aware_fuzzy /
sp.rd_discrete / sp.rd_dashboard / sp.rd_compare / sp.rd_robustness_table.

DGPs are deterministic (seeded) with known population discontinuities so
we can do parity-style recovery checks.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# --------------------------------------------------------------------------- #
# Shared DGPs
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def sharp_data():
    rng = np.random.default_rng(11)
    n = 1500
    X = rng.uniform(-1, 1, n)
    W1 = rng.normal(size=n)  # informative covariate
    W2 = rng.normal(size=n)  # noise covariate
    Y = 0.3 * X + 0.7 * W1 + 0.1 * W2 + 1.0 * (X >= 0) + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": Y, "x": X, "w1": W1, "w2": W2})
    df.attrs["true_effect"] = 1.0
    return df


@pytest.fixture(scope="module")
def fuzzy_data():
    rng = np.random.default_rng(22)
    n = 2500
    X = rng.uniform(-1, 1, n)
    # Strong first stage: P(D=1) jumps from 0.2 to 0.85
    p_left = 0.2 + 0.0 * X
    p_right = 0.85 + 0.0 * X
    p = np.where(X >= 0, p_right, p_left)
    D = rng.binomial(1, p)
    Y = 0.4 * X + 1.5 * D + rng.normal(0, 0.4, n)
    df = pd.DataFrame({"y": Y, "x": X, "d": D})
    df.attrs["true_late"] = 1.5
    return df


@pytest.fixture(scope="module")
def discrete_data():
    rng = np.random.default_rng(33)
    # 12 mass points on each side
    distinct = np.arange(-12, 12)
    counts = rng.integers(60, 120, len(distinct))
    rows = []
    for x_val, c_val in zip(distinct, counts):
        xs = np.full(c_val, x_val, dtype=float)
        # Mean function: 0.4 x + 1.0 * (x >= 0)
        mean = 0.4 * x_val + 1.0 * (x_val >= 0)
        ys = mean + rng.normal(0, 0.5, c_val)
        rows.append(pd.DataFrame({"x": xs, "y": ys}))
    df = pd.concat(rows, ignore_index=True)
    df.attrs["true_effect"] = 1.0
    return df


# --------------------------------------------------------------------------- #
# rd_flex — flexible covariate adjustment
# --------------------------------------------------------------------------- #


class TestRDFlex:
    def test_recovers_truth(self, sharp_data):
        r = sp.rd_flex(
            sharp_data,
            y="y",
            x="x",
            c=0.0,
            W=["w1", "w2"],
            learner="ridge",
            n_folds=5,
            random_state=0,
        )
        truth = sharp_data.attrs["true_effect"]
        # 5 SEs is generous and avoids flakes on different platforms
        assert abs(r.estimate - truth) <= 5 * r.se

    def test_returns_variance_reduction(self, sharp_data):
        r = sp.rd_flex(
            sharp_data,
            y="y",
            x="x",
            c=0.0,
            W=["w1"],
            learner="ridge",
            n_folds=5,
            random_state=0,
        )
        flex = r.model_info["flex"]
        assert "r2_y" in flex
        assert "var_reduction" in flex
        # An informative covariate should give non-trivial out-of-sample R²
        assert flex["r2_y"] > 0.10
        # SE should be no worse than ~5% larger than plain rdrobust
        assert flex["se_flex"] <= 1.05 * flex["se_plain"]

    def test_no_covariates_falls_back(self, sharp_data):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = sp.rd_flex(sharp_data, y="y", x="x", c=0.0, W=None)
            assert any("falling back to rdrobust" in str(wi.message) for wi in w)
        # And produces a sensible estimate
        assert np.isfinite(r.estimate)

    def test_dispatch_alias(self, sharp_data):
        r = sp.rd(
            sharp_data,
            y="y",
            x="x",
            c=0.0,
            method="flex",
            W=["w1", "w2"],
            learner="ridge",
            n_folds=5,
            random_state=0,
        )
        assert np.isfinite(r.estimate)
        assert "Flexible" in r.method or "flex" in r.method.lower()


# --------------------------------------------------------------------------- #
# rd_bias_aware_fuzzy — Noack & Rothe (2024) ECTA
# --------------------------------------------------------------------------- #


class TestRDBiasAwareFuzzy:
    def test_recovers_late(self, fuzzy_data):
        r = sp.rd_bias_aware_fuzzy(
            fuzzy_data,
            y="y",
            x="x",
            fuzzy="d",
            c=0.0,
        )
        truth = fuzzy_data.attrs["true_late"]
        assert abs(r.estimate - truth) <= 5 * r.se

    def test_bias_aware_ci_wider_than_naive(self, fuzzy_data):
        r = sp.rd_bias_aware_fuzzy(
            fuzzy_data,
            y="y",
            x="x",
            fuzzy="d",
            c=0.0,
        )
        ba = r.model_info["bias_aware"]
        ba_len = ba["bias_aware_ci"][1] - ba["bias_aware_ci"][0]
        naive_len = ba["naive_ci"][1] - ba["naive_ci"][0]
        assert ba_len >= naive_len * 0.99  # bias-aware never narrower

    def test_first_stage_F_recorded(self, fuzzy_data):
        r = sp.rd_bias_aware_fuzzy(
            fuzzy_data,
            y="y",
            x="x",
            fuzzy="d",
            c=0.0,
        )
        ba = r.model_info["bias_aware"]
        assert "first_stage_F" in ba
        assert ba["first_stage_F"] > 10  # strong first stage in DGP

    def test_dispatcher_routes(self, fuzzy_data):
        r = sp.rd(fuzzy_data, y="y", x="x", c=0.0, method="bias_aware", fuzzy="d")
        assert np.isfinite(r.estimate)


# --------------------------------------------------------------------------- #
# rd_discrete — Kolesár & Rothe (2018)
# --------------------------------------------------------------------------- #


class TestRDDiscrete:
    def test_bsd_recovers_truth(self, discrete_data):
        r = sp.rd_discrete(discrete_data, y="y", x="x", c=0.0, method="bsd")
        truth = discrete_data.attrs["true_effect"]
        ci_lo, ci_hi = r.ci
        assert ci_lo <= truth <= ci_hi

    def test_bm_recovers_truth(self, discrete_data):
        r = sp.rd_discrete(discrete_data, y="y", x="x", c=0.0, method="bm")
        truth = discrete_data.attrs["true_effect"]
        ci_lo, ci_hi = r.ci
        assert ci_lo <= truth <= ci_hi

    def test_honest_ci_at_least_naive(self, discrete_data):
        r = sp.rd_discrete(discrete_data, y="y", x="x", c=0.0)
        info = r.model_info["discrete"]
        h_len = info["honest_ci"][1] - info["honest_ci"][0]
        n_len = info["naive_ci"][1] - info["naive_ci"][0]
        assert h_len >= n_len  # honest CIs should never be narrower

    def test_too_few_mass_points_errors(self):
        df = pd.DataFrame({"y": [1.0] * 10, "x": [1.0, 2.0, 3.0] * 3 + [3.0]})
        with pytest.raises(ValueError, match="distinct mass points"):
            sp.rd_discrete(df, y="y", x="x", c=2.5)

    def test_h_filter_consistent(self, discrete_data):
        """With h=∞ (effectively) the filter is a no-op and the
        estimate must match the unfiltered call.  Exercises the B1
        re-binning fix.
        """
        r_full = sp.rd_discrete(discrete_data, y="y", x="x", c=0.0, method="bsd")
        r_h = sp.rd_discrete(discrete_data, y="y", x="x", c=0.0, method="bsd", h=1000.0)
        assert abs(r_full.estimate - r_h.estimate) < 1e-9

    def test_h_filter_drops_far_mass_points(self, discrete_data):
        """With h=4 only mass points within ±4 of the cutoff are used.
        The estimate should differ from the full-bandwidth call (the
        DGP is flat on each side, so they should still cover truth).
        """
        r_h = sp.rd_discrete(discrete_data, y="y", x="x", c=0.0, method="bsd", h=4.0)
        info = r_h.model_info["discrete"]
        # DGP has integer mass points; h=4 keeps |x-0|<=4 → x ∈ {-4,…,4}.
        # That yields 4 negative values (-4,-3,-2,-1) and 5 nonneg (0,…,4).
        assert info["n_left"] == 4
        assert info["n_right"] == 5
        truth = discrete_data.attrs["true_effect"]
        ci_lo, ci_hi = r_h.ci
        assert ci_lo <= truth <= ci_hi

    def test_dispatcher_routes(self, discrete_data):
        r = sp.rd(discrete_data, y="y", x="x", c=0.0, method="discrete")
        assert np.isfinite(r.estimate)


# --------------------------------------------------------------------------- #
# rdrobust polish — rho, mass-points warning, weak-IV warning
# --------------------------------------------------------------------------- #


class TestRDRobustPolish:
    def test_rho_param(self, sharp_data):
        r = sp.rdrobust(sharp_data, y="y", x="x", c=0.0, rho=0.8)
        # b = h / 0.8
        h = r.model_info["bandwidth_h"]
        b = r.model_info["bandwidth_b"]
        h0 = h[0] if isinstance(h, tuple) else h
        b0 = b[0] if isinstance(b, tuple) else b
        assert abs(b0 * 0.8 - h0) < 1e-9
        assert r.model_info["rho"] == 0.8

    def test_rho_b_mutually_exclusive(self, sharp_data):
        with pytest.raises(ValueError, match="mutually exclusive"):
            sp.rdrobust(sharp_data, y="y", x="x", c=0.0, b=0.3, rho=0.5)

    def test_mass_points_warning(self, discrete_data):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp.rdrobust(discrete_data, y="y", x="x", c=0.0)
            msgs = [str(wi.message) for wi in w]
            assert any("distinct values" in m and "rd_discrete" in m for m in msgs)

    def test_mass_points_warning_silenced(self, discrete_data):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp.rdrobust(discrete_data, y="y", x="x", c=0.0, warn_mass_points=False)
            msgs = [str(wi.message) for wi in w]
            assert not any("rd_discrete" in m for m in msgs)


# --------------------------------------------------------------------------- #
# rd_compare / rd_robustness_table / rd_dashboard
# --------------------------------------------------------------------------- #


class TestRDCompare:
    def test_returns_dataframe(self, sharp_data):
        out = sp.rd_compare(
            sharp_data, y="y", x="x", c=0.0, methods=("rdrobust", "honest")
        )
        assert isinstance(out, pd.DataFrame)
        assert {"method", "estimate", "se", "ci_lower", "ci_upper"}.issubset(
            out.columns
        )
        assert (out["status"] == "ok").all()

    def test_estimates_close_across_methods(self, sharp_data):
        out = sp.rd_compare(
            sharp_data, y="y", x="x", c=0.0, methods=("rdrobust", "honest")
        )
        ests = out["estimate"].to_numpy()
        # rdrobust vs honest agree within combined SE * 5
        diff = abs(ests[0] - ests[1])
        ses = out["se"].to_numpy()
        assert diff <= 5 * np.sqrt(ses[0] ** 2 + ses[1] ** 2)


class TestRDRobustnessTable:
    def test_grid_runs_all_specs(self, sharp_data):
        tbl = sp.rd_robustness_table(
            sharp_data,
            y="y",
            x="x",
            c=0.0,
            kernels=("triangular", "epanechnikov"),
            bwselects=("mserd", "cerrd"),
            polynomials=(1,),
            donuts=(0.0, 0.05),
        )
        # 2 kernels * 2 bw * 1 poly * 2 donut = 8 rows
        assert len(tbl) == 8
        assert (tbl["status"] == "ok").all()
        assert tbl["estimate_rbc"].apply(np.isfinite).all()


class TestRDDashboard:
    def test_dashboard_smoke(self, sharp_data):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pytest.skip("matplotlib not installed")
        import matplotlib

        matplotlib.use("Agg")
        fig, axes = sp.rd_dashboard(sharp_data, y="y", x="x", c=0.0, covs=["w1", "w2"])
        assert axes.shape == (2, 2)
        import matplotlib.pyplot as plt

        plt.close(fig)
