"""Brown-Durbin-Evans CUSUM test — known-truth correctness guards.

``sp.cusum_test`` previously compared ``max|CUSUM|`` against a *flat* constant
(the Kolmogorov-Smirnov sup-Brownian-*bridge* value, 1.358 at 5%). The
recursive-residual CUSUM of Brown, Durbin & Evans (1975) is a Brownian
*motion* tested against the *expanding* boundary ``a * (1 + 2 (t-k)/(n-k))``
(``a = 0.948`` at 5%). The flat cut-off rejected a perfectly stable series
~30% of the time at the nominal 5% level.

These tests anchor the fix to its defining known truths:

* under H0 (no break) the empirical rejection rate is near the nominal level,
  not ~1/3 — this is the assertion the old implementation fails;
* a clear coefficient/intercept break is detected;
* the boundary has the BDE shape (``a`` at the left end, ``3a`` at the right,
  strictly increasing) and the level constant tracks ``alpha``.

Reference: Brown, Durbin & Evans (1975), ``brown1975techniques`` in
``paper.bib`` (DOI 10.1111/j.2517-6161.1975.tb01532.x).
"""

import numpy as np
import pandas as pd
import pytest

from statspai.timeseries.structural_break import cusum_test


def _stable_panel(rng, n):
    """Stable regression: no parameter break anywhere."""
    x = rng.normal(0, 1, n)
    y = 1.0 + 2.0 * x + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "x": x})


class TestCusumNullSize:
    """Under H0 the test must not over-reject (the core correctness fix)."""

    def test_empirical_size_near_nominal(self):
        # Old flat-cutoff code rejects ~0.34 of stable series; the BDE
        # expanding boundary brings this back to ~nominal. We assert a
        # generous upper band the broken implementation cannot satisfy.
        rng = np.random.default_rng(0)
        reps = 300
        n = 400
        rejects = 0
        for _ in range(reps):
            r = cusum_test(_stable_panel(rng, n), y="y", x=["x"], alpha=0.05)
            rejects += int(r["reject"])
        rate = rejects / reps
        # nominal 0.05; broken code ~0.34. Band excludes the broken regime.
        assert rate < 0.15, f"H0 rejection rate {rate:.3f} too high"

    def test_size_does_not_blow_up_with_n(self):
        # The broken flat-cutoff size stayed ~0.33 as n grew (not finite
        # sample slack). The fixed test stays near nominal at every n.
        rng = np.random.default_rng(7)
        for n in (200, 600):
            rejects = sum(
                cusum_test(_stable_panel(rng, n), y="y", x=["x"])["reject"]
                for _ in range(200)
            )
            rate = rejects / 200
            assert rate < 0.15, f"n={n}: H0 rejection rate {rate:.3f}"


class TestCusumPower:
    """A genuine structural break must be detected."""

    def test_intercept_shift_detected(self):
        rng = np.random.default_rng(1)
        n = 400
        x = rng.normal(0, 1, n)
        shift = np.concatenate([np.zeros(n // 2), np.full(n - n // 2, 5.0)])
        y = 1.0 + shift + 2.0 * x + rng.normal(0, 1, n)
        r = cusum_test(pd.DataFrame({"y": y, "x": x}), y="y", x=["x"])
        assert r["reject"] is True

    def test_slope_shift_detected(self):
        # A slope break is visible to the *mean*-based CUSUM only when it
        # moves the conditional mean, i.e. when the regressor has non-zero
        # mean (a mean-preserving slope flip on centred x is the CUSUM-of-
        # squares' job, not this test's — see the module docstring).
        rng = np.random.default_rng(2)
        n = 400
        x = rng.normal(3.0, 1.0, n)
        slope = np.concatenate([np.full(n // 2, 2.0), np.full(n - n // 2, 5.0)])
        y = 1.0 + slope * x + rng.normal(0, 1, n)
        r = cusum_test(pd.DataFrame({"y": y, "x": x}), y="y", x=["x"])
        assert r["reject"] is True


class TestCusumBoundaryShape:
    """The returned boundary must have the Brown-Durbin-Evans geometry."""

    def test_boundary_expands_from_a_to_3a(self):
        rng = np.random.default_rng(3)
        df = _stable_panel(rng, 300)
        r = cusum_test(df, y="y", x=["x"], alpha=0.05)
        b = np.asarray(r["boundary"])
        a = r["level_constant"]
        assert a == 0.948
        assert len(b) == len(r["cusum"])
        assert np.all(np.diff(b) > 0)  # strictly expanding
        assert b[0] == pytest.approx(a * (1 + 2 / len(b)), rel=1e-9)
        assert b[-1] == pytest.approx(3.0 * a, rel=1e-9)

    def test_level_constant_tracks_alpha(self):
        rng = np.random.default_rng(4)
        df = _stable_panel(rng, 250)
        assert cusum_test(df, y="y", x=["x"], alpha=0.01)["level_constant"] == 1.143
        assert cusum_test(df, y="y", x=["x"], alpha=0.05)["level_constant"] == 0.948
        assert cusum_test(df, y="y", x=["x"], alpha=0.10)["level_constant"] == 0.850

    def test_backward_compatible_keys(self):
        rng = np.random.default_rng(5)
        df = _stable_panel(rng, 200)
        r = cusum_test(df, y="y", x=["x"])
        for key in ("cusum", "max_cusum", "critical_value", "reject", "n_obs"):
            assert key in r
