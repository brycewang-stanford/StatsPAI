"""Correctness + edge tests for previously-untested public functions.

A coverage audit found ~93 registered callables with zero test mention. This
file adds analytical correctness tests (not just smoke tests) for the
highest-value standalone numerics among them: the multiple-hypothesis-testing
adjusters, the power-analysis family, survey weighted estimators, and two
regression models (cloglog, zinb).

Each MHT / power / survey assertion checks a value derivable by hand, so the
test pins behavior rather than merely exercising the code path.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# ---------------------------------------------------------------------------
# Multiple-hypothesis-testing adjusters (analytical).
# ---------------------------------------------------------------------------


class TestMHTAdjusters:
    P = [0.01, 0.02, 0.03, 0.50]  # m = 4

    def test_bonferroni_is_min_one_m_times_p(self):
        out = np.asarray(sp.bonferroni(self.P))
        np.testing.assert_allclose(out, [0.04, 0.08, 0.12, 1.0])

    def test_bonferroni_clips_at_one(self):
        out = np.asarray(sp.bonferroni([0.4, 0.9]))
        assert out.max() <= 1.0 + 1e-12
        np.testing.assert_allclose(out, [0.8, 1.0])

    def test_holm_step_down_with_monotonicity(self):
        # sorted (m-j+1)*p_(j): 4*.01, 3*.02, 2*.03, 1*.5 = .04,.06,.06,.5
        out = np.asarray(sp.holm(self.P))
        np.testing.assert_allclose(out, [0.04, 0.06, 0.06, 0.50])

    def test_benjamini_hochberg_step_up(self):
        # m/rank * p, enforced monotone from the largest: all .04 then .5
        out = np.asarray(sp.benjamini_hochberg(self.P))
        np.testing.assert_allclose(out, [0.04, 0.04, 0.04, 0.50])

    def test_adjust_pvalues_dispatches_to_method(self):
        np.testing.assert_allclose(
            np.asarray(sp.adjust_pvalues(self.P, method="holm")),
            np.asarray(sp.holm(self.P)),
        )
        np.testing.assert_allclose(
            np.asarray(sp.adjust_pvalues(self.P, method="bonferroni")),
            np.asarray(sp.bonferroni(self.P)),
        )

    def test_adjusted_pvalues_never_below_raw(self):
        # A multiplicity correction can only INCREASE p-values.
        for method in ("holm", "bonferroni", "benjamini_hochberg"):
            adj = np.asarray(sp.adjust_pvalues(self.P, method=method))
            assert np.all(adj >= np.asarray(self.P) - 1e-12)

    def test_single_pvalue_unchanged(self):
        # With m = 1 every method is the identity (clipped to 1).
        for fn in (sp.bonferroni, sp.holm, sp.benjamini_hochberg):
            np.testing.assert_allclose(np.asarray(fn([0.03])), [0.03])


# ---------------------------------------------------------------------------
# Power analysis family (analytical reference + monotonicity).
# ---------------------------------------------------------------------------


class TestPowerFamily:
    def test_power_rct_matches_two_sample_z(self):
        # n=200 total, equal split, d=0.5: ncp = 0.5*sqrt(100*100/200)=3.5355,
        # power = Phi(ncp - 1.96) = Phi(1.5755) ~ 0.9424.
        from scipy import stats

        r = sp.power_rct(n=200, effect_size=0.5)
        ncp = 0.5 * np.sqrt(100 * 100 / 200)
        expected = stats.norm.cdf(ncp - stats.norm.ppf(0.975))
        assert r.power == pytest.approx(expected, abs=2e-3)

    def test_power_increases_with_n(self):
        powers = [sp.power_rct(n=n, effect_size=0.3).power for n in (50, 100, 200, 400)]
        assert powers == sorted(powers)
        assert powers[-1] > powers[0]

    def test_power_increases_with_effect_size(self):
        powers = [
            sp.power_rct(n=200, effect_size=es).power for es in (0.1, 0.3, 0.5, 0.8)
        ]
        assert powers == sorted(powers)

    def test_power_in_unit_interval(self):
        for n in (10, 100, 1000):
            for es in (0.05, 0.5, 1.5):
                p = sp.power_rct(n=n, effect_size=es).power
                assert 0.0 <= p <= 1.0

    def test_power_ols_covariates_help_via_r2(self):
        # Explaining residual variance (higher r2_other) raises power.
        base = sp.power_ols(n=200, effect_size=0.2, n_covariates=3, r2_other=0.0).power
        better = sp.power_ols(
            n=200, effect_size=0.2, n_covariates=3, r2_other=0.5
        ).power
        assert better >= base


# ---------------------------------------------------------------------------
# Survey weighted estimators (analytical).
# ---------------------------------------------------------------------------


class TestSurveyEstimators:
    def _design(self):
        df = pd.DataFrame({"y": [1.0, 2, 3, 4, 5, 6], "w": [1, 1, 1, 2, 2, 2]})
        return sp.svydesign(data=df, weights="w")

    def test_svymean_is_weighted_mean(self):
        # sum(w*y)/sum(w) = 36/9 = 4.0
        r = sp.svymean("y", self._design())
        assert float(np.atleast_1d(r.estimate)[0]) == pytest.approx(4.0)

    def test_svytotal_is_weighted_total(self):
        # sum(w*y) = 36
        r = sp.svytotal("y", self._design())
        assert float(np.atleast_1d(r.estimate)[0]) == pytest.approx(36.0)

    def test_svymean_positive_se(self):
        r = sp.svymean("y", self._design())
        assert float(np.atleast_1d(r.std_error)[0]) > 0


# ---------------------------------------------------------------------------
# Regression models: cloglog & zero-inflated NB (sign recovery + smoke).
# ---------------------------------------------------------------------------


class TestRareRegressions:
    def test_cloglog_recovers_positive_slope(self):
        rng = np.random.default_rng(0)
        n = 2000
        x = rng.normal(size=n)
        # cloglog link: P(y=1) = 1 - exp(-exp(b0 + b1 x)), b1 > 0
        eta = -0.5 + 1.0 * x
        p = 1 - np.exp(-np.exp(eta))
        y = (rng.uniform(size=n) < p).astype(int)
        df = pd.DataFrame({"y": y, "x": x})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.cloglog(data=df, y="y", x=["x"])
        coefs = res.params if hasattr(res, "params") else res.coefficients
        bx = (
            float(np.asarray(coefs)[list(res.param_names).index("x")])
            if hasattr(res, "param_names")
            else float(pd.Series(coefs)["x"])
        )
        assert bx > 0.3  # recovers the positive sign with the right magnitude

    def test_zinb_runs_and_returns_finite_estimates(self):
        rng = np.random.default_rng(1)
        n = 1500
        x = rng.normal(size=n)
        # NB-ish count with structural zeros
        mu = np.exp(0.3 + 0.5 * x)
        y = rng.poisson(mu)
        zero_infl = rng.uniform(size=n) < 0.3
        y[zero_infl] = 0
        df = pd.DataFrame({"y": y, "x": x})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.zinb(data=df, y="y", x=["x"])
        # Smoke + sanity: a result object with finite count-model estimates.
        assert res is not None
        coefs = res.params if hasattr(res, "params") else res.coefficients
        assert np.all(np.isfinite(np.asarray(coefs, dtype=float)))
