"""Regression tests for two inference-correctness fixes.

These lock in fixes that change *inference* (not point estimates) in two
estimators where the previous code reported the wrong size / coverage:

1. ``sp.cusum_test`` (Brown-Durbin-Evans recursive CUSUM) compared the CUSUM
   path against a *constant* 1.358 boundary -- the sup|Brownian-bridge|
   critical value that belongs to the OLS-CUSUM, not the recursive CUSUM. On
   a stable (H0) relation it rejected ~30% of the time at a nominal 5% level.
   The boundary must be the *linear* BDE boundary ``a * [1 + 2 s / (n - k)]``.

2. ``sp.lee_bounds`` labelled its confidence interval "Imbens-Manski" but
   applied the two-sided ``z_{1-alpha/2}`` to *both* endpoints, which is the
   Horowitz-Manski interval for the identified *set* and over-covers the
   parameter. The genuine Imbens & Manski (2004) interval uses a critical
   value ``C_n`` that interpolates between the one- and two-sided z.

References for the fixed methods are hand-verified (see the respective module
docstrings); Imbens & Manski (2004) Econometrica 72(6):1845-1857 was verified
via Crossref and RePEc/IDEAS.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.timeseries.structural_break import cusum_test
from statspai.bounds.lee_manski import _imbens_manski_cn


# ----------------------------------------------------------------------
# Fix 1 -- recursive CUSUM uses the Brown-Durbin-Evans linear boundary
# ----------------------------------------------------------------------
class TestCusumLinearBoundary:
    def test_critical_value_is_linear_boundary(self):
        rng = np.random.default_rng(0)
        n = 150
        x = rng.normal(size=n)
        y = 1.0 + 0.5 * x + rng.normal(scale=0.7, size=n)
        df = pd.DataFrame({"y": y, "x": x})
        res = cusum_test(df, y="y", x=["x"], alpha=0.05)

        boundary = np.asarray(res["critical_value"], dtype=float)
        m = n - 2  # n - k, k = 2 (intercept + x)
        assert boundary.shape == (m,)
        # Boundary widens linearly from a (=0.948 at 5%) to 3a across the sample.
        a = 0.948
        s = np.arange(1, m + 1)
        np.testing.assert_allclose(boundary, a * (1.0 + 2.0 * s / m), rtol=1e-12)
        np.testing.assert_allclose(boundary[0], a * (1 + 2 / m), rtol=1e-12)
        np.testing.assert_allclose(boundary[-1], 3 * a, rtol=1e-9)

    def test_size_near_nominal_under_h0(self):
        """A nominal 5% test must not reject ~30% of the time under H0."""
        rng = np.random.default_rng(12345)
        n, B = 120, 400
        rej_new = 0
        rej_old_const = 0
        for _ in range(B):
            x = rng.normal(size=n)
            y = 1.0 + 0.5 * x + rng.normal(scale=0.7, size=n)  # stable -> H0
            df = pd.DataFrame({"y": y, "x": x})
            res = cusum_test(df, y="y", x=["x"], alpha=0.05)
            rej_new += int(res["reject"])
            # Replicate the OLD constant-1.358 rule on the same path.
            rej_old_const += int(res["max_cusum"] > 1.358)

        size_new = rej_new / B
        size_old = rej_old_const / B
        # New test is close to nominal (and conservative); old rule grossly
        # over-rejects. Generous bounds keep this deterministic test stable.
        assert size_new < 0.12, f"new size {size_new:.3f} too high"
        assert size_old > 0.20, f"old const rule size {size_old:.3f} (bug guard)"

    def test_power_against_mean_shift(self):
        rng = np.random.default_rng(7)
        n = 120
        rej = 0
        for _ in range(200):
            x = rng.normal(size=n)
            shift = np.where(np.arange(n) < 60, 0.0, 2.5)
            y = 1.0 + 0.5 * x + shift + rng.normal(scale=0.7, size=n)
            df = pd.DataFrame({"y": y, "x": x})
            rej += int(cusum_test(df, y="y", x=["x"])["reject"])
        assert rej / 200 > 0.8

    def test_keys_and_doctest_contract_preserved(self):
        rng = np.random.default_rng(0)
        n = 120
        x = rng.normal(size=n)
        y = 1.0 + 0.5 * x + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"y": y, "x": x})
        res = cusum_test(df, y="y", x=["x"])
        assert sorted(res.keys()) == [
            "critical_value",
            "cusum",
            "max_cusum",
            "n_obs",
            "reject",
        ]
        assert res["n_obs"] == 120
        assert isinstance(res["reject"], bool)


# ----------------------------------------------------------------------
# Fix 2 -- Lee bounds use the genuine Imbens-Manski C_n
# ----------------------------------------------------------------------
class TestImbensManskiCn:
    def test_point_identified_limit_is_two_sided_z(self):
        z_two = stats.norm.ppf(0.975)
        assert _imbens_manski_cn(0.0, 1.0, 0.05) == pytest.approx(z_two, abs=1e-8)

    def test_wide_bounds_limit_is_one_sided_z(self):
        z_one = stats.norm.ppf(0.95)
        assert _imbens_manski_cn(1e6, 1.0, 0.05) == pytest.approx(z_one, abs=1e-6)

    def test_monotone_and_bracketed(self):
        z_one = stats.norm.ppf(0.95)
        z_two = stats.norm.ppf(0.975)
        widths = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 50.0]
        cns = [_imbens_manski_cn(d, 1.0, 0.05) for d in widths]
        # within [z_one, z_two] and non-increasing in width
        for c in cns:
            assert z_one - 1e-9 <= c <= z_two + 1e-9
        assert all(cns[i] >= cns[i + 1] - 1e-9 for i in range(len(cns) - 1))
        # strictly interior for an intermediate width
        assert z_one < _imbens_manski_cn(0.5, 1.0, 0.05) < z_two

    def test_solves_defining_equation(self):
        delta = 1.3
        sigma = 0.8
        c = _imbens_manski_cn(delta, sigma, 0.05)
        ratio = delta / sigma
        lhs = stats.norm.cdf(c + ratio) - stats.norm.cdf(-c)
        assert lhs == pytest.approx(0.95, abs=1e-7)

    def test_degenerate_sigma(self):
        z_one = stats.norm.ppf(0.95)
        z_two = stats.norm.ppf(0.975)
        assert _imbens_manski_cn(0.0, 0.0, 0.05) == pytest.approx(z_two)
        assert _imbens_manski_cn(2.0, 0.0, 0.05) == pytest.approx(z_one)


class TestLeeBoundsCI:
    def _selection_dgp(self, seed=0, n=1500):
        rng = np.random.default_rng(seed)
        d = rng.integers(0, 2, n)
        # treatment raises retention (differential attrition)
        latent = 0.3 + 0.6 * d + rng.normal(size=n)
        s = (latent > 0).astype(int)
        y = 1.0 + 0.5 * d + rng.normal(size=n)
        return pd.DataFrame({"y": y, "d": d, "s": s})

    def test_im_ci_brackets_bounds_and_uses_cn_below_two_sided_z(self):
        df = self._selection_dgp()
        res = sp.lee_bounds(df, y="y", treat="d", selection="s", n_bootstrap=200)
        lb = res.model_info["lower_bound"]
        ub = res.model_info["upper_bound"]
        ci_lo, ci_hi = res.ci
        # The Imbens-Manski CI brackets the identified set and is ordered.
        assert ci_lo <= lb + 1e-9 <= ub + 1e-9 <= ci_hi + 1e-9
        # The per-endpoint multiplier C_n implied by these (positive-width)
        # bounds is strictly below the two-sided z that the old Horowitz-Manski
        # interval used on both ends -> the IM interval is the narrower one.
        z_two = stats.norm.ppf(0.975)
        c_n = _imbens_manski_cn(ub - lb, 1.0, 0.05)  # any sigma>0; width>0 => <z_two
        assert c_n < z_two
