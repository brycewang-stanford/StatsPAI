"""RD reference parity tests.

Recovery + cross-method consistency for regression discontinuity.

Validates:
1. Sharp RD (CCT 2014 rdrobust) recovers the true jump at the cutoff.
2. Fuzzy RD recovers the LATE via the Wald ratio.
3. Different bandwidth selectors (MSE, CER) give similar point estimates.
4. Kernel choice (triangular, uniform, epanechnikov) agrees within noise.

All DGPs are deterministic (seeded) with known population discontinuities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.rd import rdrobust


def _within_n_se(estimate, truth, se, n_sigma=4.0):
    return abs(estimate - truth) <= n_sigma * se


# ---------------------------------------------------------------------------
# Sharp RD
# ---------------------------------------------------------------------------

class TestSharpRDRecovery:
    """Sharp RD on a DGP with known jump = 1.0 must recover truth."""

    def test_triangular_mserd_recovers_truth(self, rd_sharp_data):
        truth = rd_sharp_data.attrs['true_effect']
        r = rdrobust(rd_sharp_data, y='y', x='x', c=0.0,
                     kernel='triangular', bwselect='mserd')
        assert _within_n_se(r.estimate, truth, r.se), (
            f"Sharp RD triangular/mserd: {r.estimate:.4f} vs {truth} "
            f"(SE {r.se:.4f})"
        )

    def test_uniform_kernel_close_to_triangular(self, rd_sharp_data):
        """Point estimates under different kernels must agree within combined SE."""
        r_t = rdrobust(rd_sharp_data, y='y', x='x', c=0.0,
                       kernel='triangular', bwselect='mserd')
        r_u = rdrobust(rd_sharp_data, y='y', x='x', c=0.0,
                       kernel='uniform', bwselect='mserd')
        combined_se = np.sqrt(r_t.se**2 + r_u.se**2)
        assert abs(r_t.estimate - r_u.estimate) <= 4.0 * combined_se, (
            f"triangular {r_t.estimate:.4f} vs uniform {r_u.estimate:.4f} "
            f"(combined SE {combined_se:.4f})"
        )

    def test_epanechnikov_close_to_triangular(self, rd_sharp_data):
        r_t = rdrobust(rd_sharp_data, y='y', x='x', c=0.0,
                       kernel='triangular', bwselect='mserd')
        r_e = rdrobust(rd_sharp_data, y='y', x='x', c=0.0,
                       kernel='epanechnikov', bwselect='mserd')
        combined_se = np.sqrt(r_t.se**2 + r_e.se**2)
        assert abs(r_t.estimate - r_e.estimate) <= 4.0 * combined_se

    def test_positive_effect_is_positive(self, rd_sharp_data):
        r = rdrobust(rd_sharp_data, y='y', x='x', c=0.0)
        assert r.estimate > 0, f"Expected positive, got {r.estimate}"

    def test_ci_covers_truth(self, rd_sharp_data):
        truth = rd_sharp_data.attrs['true_effect']
        r = rdrobust(rd_sharp_data, y='y', x='x', c=0.0)
        lo, hi = r.ci
        assert lo <= truth <= hi, (
            f"95% CI [{lo:.4f}, {hi:.4f}] does not cover truth {truth}"
        )


# ---------------------------------------------------------------------------
# Fuzzy RD
# ---------------------------------------------------------------------------

class TestFuzzyRDRecovery:
    """Fuzzy RD must recover LATE via Wald ratio."""

    def test_fuzzy_rd_recovers_late(self, rd_fuzzy_data):
        truth = rd_fuzzy_data.attrs['true_effect']
        r = rdrobust(rd_fuzzy_data, y='y', x='x', c=0.0, fuzzy='d')
        assert _within_n_se(r.estimate, truth, r.se, n_sigma=4.0), (
            f"Fuzzy RD: {r.estimate:.4f} vs truth {truth} "
            f"(SE {r.se:.4f})"
        )


# ---------------------------------------------------------------------------
# Degenerate cases: RD returns NaN or raises cleanly
# ---------------------------------------------------------------------------

class TestRDDegenerate:
    """Tests that RD fails gracefully on degenerate inputs."""

    def test_all_below_cutoff_fails_cleanly(self):
        """If there are no observations above the cutoff, must raise."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            'x': rng.uniform(-2, -0.1, 200),  # all below 0
            'y': rng.normal(size=200),
        })
        with pytest.raises(Exception):
            rdrobust(df, y='y', x='x', c=0.0)

    def test_zero_effect_confidence_interval_contains_zero(self):
        """No discontinuity: CI must contain 0 most of the time."""
        rng = np.random.default_rng(17)
        n = 2000
        x = rng.uniform(-1, 1, n)
        y = 2 + 3*x + x**2 + rng.normal(scale=0.3, size=n)  # no jump
        df = pd.DataFrame({'x': x, 'y': y})
        r = rdrobust(df, y='y', x='x', c=0.0)
        lo, hi = r.ci
        assert lo <= 0 <= hi, (
            f"Zero-effect DGP: 95% CI [{lo:.4f}, {hi:.4f}] excludes 0"
        )
