"""Tier D P2 known-truth upgrades — Kaplan-Meier, log-rank, parametric survival.

Part of the P1/P2 "Tier D analytic special-cases" campaign (see
``.tierd_campaign/CAMPAIGN.md``). All three were graded ``weak`` by
``scripts/tierd_classify.py`` (only provenance asserts). Each anchors to a
known truth:

    sp.kaplan_meier   with no censoring the product-limit estimate equals the
                      empirical survival (n - cumulative events) / n exactly.
    sp.logrank_test   detects a known hazard difference; under equal hazards it
                      does not reject.
    sp.survreg        Weibull AFT recovers the known log-time coefficient.

Purely additive — no estimator numerics changed (campaign red line).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# sp.kaplan_meier
# ---------------------------------------------------------------------------
class TestKaplanMeierAnalytic:

    @staticmethod
    def _no_censoring(seed=0, n=200, scale=5.0):
        rng = np.random.default_rng(seed)
        dur = rng.exponential(scale, n)
        return pd.DataFrame({"t": dur, "e": np.ones(n, dtype=int)}), dur

    def test_no_censoring_equals_empirical_product_limit(self):
        df, dur = self._no_censoring()
        n = len(dur)
        km = sp.kaplan_meier(df, duration="t", event="e")
        st = km.survival_table
        # With no censoring the product-limit telescopes to
        # S(t_i) = (n - cumulative events up to t_i) / n.
        expected = (n - st["n_event"].cumsum()) / n
        np.testing.assert_allclose(st["survival"].values, expected.values, atol=1e-12)

    def test_survival_is_a_valid_monotone_curve(self):
        df, _ = self._no_censoring()
        st = sp.kaplan_meier(df, duration="t", event="e").survival_table
        s = st["survival"].values
        assert s[0] == pytest.approx(1.0)
        assert np.all(np.diff(s) <= 1e-12)  # non-increasing
        assert np.all((s >= -1e-12) & (s <= 1 + 1e-12))

    def test_median_equals_sample_median_without_censoring(self):
        df, dur = self._no_censoring()
        km = sp.kaplan_meier(df, duration="t", event="e")
        # No censoring -> KM median is the empirical median duration.
        assert km.median_survival == pytest.approx(np.median(dur), abs=0.15)


# ---------------------------------------------------------------------------
# sp.logrank_test
# ---------------------------------------------------------------------------
class TestLogRankAnalytic:

    def test_detects_hazard_difference(self):
        rng = np.random.default_rng(0)
        n = 600
        g = rng.integers(0, 2, n)
        dur = rng.exponential(np.where(g == 1, 3.0, 8.0), n)  # different hazards
        df = pd.DataFrame({"t": dur, "e": np.ones(n, dtype=int), "g": g})
        res = sp.logrank_test(df, duration="t", event="e", group="g")
        assert res["p_value"] < 0.01

    def test_equal_hazards_not_rejected(self):
        rng = np.random.default_rng(1)
        n = 600
        g = rng.integers(0, 2, n)
        dur = rng.exponential(5.0, n)  # same hazard in both groups
        df = pd.DataFrame({"t": dur, "e": np.ones(n, dtype=int), "g": g})
        res = sp.logrank_test(df, duration="t", event="e", group="g")
        assert res["p_value"] > 0.10

    def test_observed_events_sum_to_total(self):
        rng = np.random.default_rng(2)
        n = 400
        g = rng.integers(0, 2, n)
        e = rng.integers(0, 2, n)  # some censoring
        df = pd.DataFrame({"t": rng.exponential(5, n), "e": e, "g": g})
        res = sp.logrank_test(df, duration="t", event="e", group="g")
        # observed_events is a {group: count} dict; the counts partition the
        # total events exactly.
        assert sum(res["observed_events"].values()) == int(e.sum())


# ---------------------------------------------------------------------------
# sp.survreg — parametric AFT
# ---------------------------------------------------------------------------
class TestSurvregAnalytic:

    def test_weibull_recovers_aft_coefficient(self):
        # AFT model: log T = 1 + 0.5 x + EV error -> the x coefficient is 0.5.
        rng = np.random.default_rng(0)
        n = 2000
        x = rng.normal(0, 1, n)
        T = np.exp(1.0 + 0.5 * x + rng.gumbel(0, 0.3, n))
        df = pd.DataFrame({"t": T, "e": np.ones(n, dtype=int), "x": x})
        res = sp.survreg(data=df, duration="t", event="e", x=["x"], dist="weibull")
        assert float(res.params["x"]) == pytest.approx(0.5, abs=0.1)

    def test_recovers_a_larger_coefficient(self):
        # A second, larger AFT slope (1.2) is also recovered — the slope is the
        # cleanly identified target (the intercept absorbs the error-law
        # location, which is parameterisation-dependent).
        rng = np.random.default_rng(3)
        n = 4000
        x = rng.normal(0, 1, n)
        T = np.exp(1.0 + 1.2 * x + rng.gumbel(0, 0.3, n))
        df = pd.DataFrame({"t": T, "e": np.ones(n, dtype=int), "x": x})
        res = sp.survreg(data=df, duration="t", event="e", x=["x"], dist="weibull")
        assert float(res.params["x"]) == pytest.approx(1.2, abs=0.1)
