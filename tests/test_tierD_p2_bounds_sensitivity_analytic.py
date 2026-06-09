"""Tier D P2 known-truth upgrades — partial identification & E-value sensitivity.

Part of the P1/P2 "Tier D analytic special-cases" campaign (see
``.tierd_campaign/CAMPAIGN.md``). These three entry points were graded ``weak``
by ``scripts/tierd_classify.py`` — referenced by tests only via ``hasattr`` /
provenance checks, with no numerical assertion. Each now anchors to an exact
closed form:

    sp.evalue_rr      VanderWeele-Ding (2017): E = RR + sqrt(RR (RR-1))
    sp.manski_bounds  Manski (1990) no-assumption ATE bound width = outcome range
    sp.lee_bounds     Lee (2009) trimming bounds bracket the always-observed effect

Purely additive — no estimator numerics changed (campaign red line).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# sp.evalue_rr — VanderWeele & Ding (2017) E-value
# ---------------------------------------------------------------------------
class TestEValueRRAnalytic:

    @staticmethod
    def _closed_form(rr):
        rr = 1.0 / rr if rr < 1 else rr
        return rr + np.sqrt(rr * (rr - 1.0))

    @pytest.mark.parametrize("rr", [1.5, 2.0, 3.0, 4.5])
    def test_point_matches_closed_form(self, rr):
        # E-value = RR + sqrt(RR (RR - 1))  (VanderWeele-Ding 2017, eq. 1).
        got = sp.evalue_rr(rr)["evalue_estimate"]
        assert got == pytest.approx(self._closed_form(rr), abs=1e-9)

    def test_protective_rr_is_symmetric(self):
        # A protective RR is reflected: E(RR) == E(1/RR).
        assert sp.evalue_rr(0.5)["evalue_estimate"] == pytest.approx(
            sp.evalue_rr(2.0)["evalue_estimate"], abs=1e-9
        )

    def test_ci_evalue_uses_lower_bound(self):
        # With a CI whose lower bound exceeds 1, the CI E-value uses that bound
        # (the bound closest to the null). Both CI ends must be supplied.
        out = sp.evalue_rr(2.0, rr_lower=1.5, rr_upper=2.8)
        assert out["evalue_ci"] == pytest.approx(self._closed_form(1.5), abs=1e-9)

    def test_evalue_monotone_in_rr(self):
        e2 = sp.evalue_rr(2.0)["evalue_estimate"]
        e3 = sp.evalue_rr(3.0)["evalue_estimate"]
        assert e3 > e2 > 1.0  # stronger association -> larger E-value


# ---------------------------------------------------------------------------
# sp.manski_bounds — Manski (1990) worst-case ATE bounds
# ---------------------------------------------------------------------------
class TestManskiBoundsAnalytic:

    @staticmethod
    def _data(seed=0, n=2000):
        rng = np.random.default_rng(seed)
        # Treatment independent of outcome -> true ATE = 0.
        return pd.DataFrame({"y": rng.uniform(0, 1, n), "t": rng.integers(0, 2, n)})

    @pytest.mark.parametrize("lo,hi", [(0.0, 1.0), (-2.0, 3.0), (0.0, 10.0)])
    def test_no_assumption_width_equals_outcome_range(self, lo, hi):
        # The no-assumption bound width is exactly the support range hi - lo,
        # independent of the data (the counterfactual arm is unrestricted).
        df = self._data()
        res = sp.manski_bounds(df, y="y", treat="t", y_lower=lo, y_upper=hi)
        assert res.diagnostics["bound_width"] == pytest.approx(hi - lo, abs=1e-9)

    def test_bounds_bracket_true_ate(self):
        df = self._data()
        d = sp.manski_bounds(df, y="y", treat="t", y_lower=0.0, y_upper=1.0).diagnostics
        assert d["lower_bound"] <= 0.0 <= d["upper_bound"]
        assert d["upper_bound"] >= d["lower_bound"]


# ---------------------------------------------------------------------------
# sp.lee_bounds — Lee (2009) trimming bounds under selection
# ---------------------------------------------------------------------------
class TestLeeBoundsAnalytic:

    @staticmethod
    def _selection_dgp(seed=0, n=4000, effect=0.8, p_sel_t=0.8, p_sel_c=0.8):
        rng = np.random.default_rng(seed)
        t = rng.integers(0, 2, n)
        latent = 1.0 + effect * t + rng.normal(0, 1, n)
        p_sel = np.where(t == 1, p_sel_t, p_sel_c)
        s = (rng.uniform(size=n) < p_sel).astype(int)
        y = np.where(s == 1, latent, np.nan)
        return pd.DataFrame({"y": y, "t": t, "s": s}), effect

    def test_equal_selection_brackets_observed_effect_tightly(self):
        # No differential selection -> trimming fraction ~ 0 -> the bounds
        # collapse around the identified selected-sample effect, and their
        # midpoint recovers the true effect (0.8) up to sampling noise.
        df, effect = self._selection_dgp()
        obs_diff = np.nanmean(df["y"][df["t"] == 1]) - np.nanmean(df["y"][df["t"] == 0])
        d = sp.lee_bounds(df, y="y", treat="t", selection="s").diagnostics
        assert d["lower_bound"] <= obs_diff <= d["upper_bound"]
        assert d["trimming_fraction"] < 0.05
        assert d["bound_width"] < 0.1
        midpoint = 0.5 * (d["lower_bound"] + d["upper_bound"])
        assert midpoint == pytest.approx(effect, abs=0.1)

    def test_bounds_ordered(self):
        df, _ = self._selection_dgp(seed=2)
        d = sp.lee_bounds(df, y="y", treat="t", selection="s").diagnostics
        assert d["upper_bound"] >= d["lower_bound"]

    def test_differential_selection_widens_bounds(self):
        # Treatment lifts the retention rate -> positive trimming -> the
        # identified set is strictly wider than under equal selection.
        equal, _ = self._selection_dgp(p_sel_t=0.8, p_sel_c=0.8)
        diff, _ = self._selection_dgp(p_sel_t=0.9, p_sel_c=0.6)
        w_equal = sp.lee_bounds(equal, y="y", treat="t", selection="s").diagnostics[
            "bound_width"
        ]
        w_diff = sp.lee_bounds(diff, y="y", treat="t", selection="s").diagnostics[
            "bound_width"
        ]
        assert w_diff > w_equal
