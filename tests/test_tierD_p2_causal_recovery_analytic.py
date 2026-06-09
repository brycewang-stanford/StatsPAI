"""Tier D P2 known-truth upgrades — changes-in-changes & dose-response.

Part of the P1/P2 "Tier D analytic special-cases" campaign (see
``.tierd_campaign/CAMPAIGN.md``). Both were graded ``weak`` by
``scripts/tierd_classify.py`` (only provenance asserts). Each anchors to a
known-DGP recovery / limiting identity:

    sp.cic            Changes-in-Changes (Athey-Imbens 2006) reduces to DiD and
                      recovers a constant additive treatment effect.
    sp.dose_response  the average marginal effect recovers a known linear
                      dose slope (and is ~0 when the dose has no effect).

Purely additive — no estimator numerics changed (campaign red line).
NB: ``sp.cic(n_boot=0)`` raises (percentile over an empty bootstrap array);
tests pass a small positive ``n_boot``. Reported as a minor robustness finding.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# sp.cic — Changes-in-Changes
# ---------------------------------------------------------------------------
class TestCICAnalytic:

    @staticmethod
    def _additive_did_dgp(seed=1, n=6000, effect=2.0):
        rng = np.random.default_rng(seed)
        g = rng.integers(0, 2, n)
        t = rng.integers(0, 2, n)
        # Constant additive treatment effect on the treated-post cell.
        y = 1.0 * g + 0.5 * t + effect * (g * t) + rng.normal(0, 1, n)
        return pd.DataFrame({"y": y, "g": g, "t": t})

    def test_recovers_constant_additive_effect(self):
        df = self._additive_did_dgp(effect=2.0)
        est = sp.cic(df, y="y", group="g", time="t", n_boot=50).estimate
        assert est == pytest.approx(2.0, abs=0.15)

    def test_agrees_with_did_under_additivity(self):
        # Under a constant additive effect the CiC estimand coincides with the
        # 2x2 DiD estimand (Athey & Imbens 2006, Sec. 3).
        df = self._additive_did_dgp(effect=2.0)
        cic_est = sp.cic(df, y="y", group="g", time="t", n_boot=50).estimate
        did_est = float(sp.did_2x2(df, y="y", treat="g", time="t").estimate)
        assert cic_est == pytest.approx(did_est, abs=0.1)


# ---------------------------------------------------------------------------
# sp.dose_response — continuous-treatment dose-response curve
# ---------------------------------------------------------------------------
class TestDoseResponseAnalytic:

    @staticmethod
    def _linear_dgp(seed=2, n=3000, slope=1.5):
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 1, n)
        dose = 0.5 * x + rng.normal(0, 1, n)  # confounded by x
        y = slope * dose + 1.0 * x + rng.normal(0, 1, n)
        return pd.DataFrame({"y": y, "dose": dose, "x": x})

    def test_average_marginal_effect_recovers_slope(self):
        # Adjusting for the confounder x, the average marginal effect of the
        # dose recovers the true structural slope.
        df = self._linear_dgp(slope=1.5)
        dr = sp.dose_response(
            df,
            y="y",
            treat="dose",
            covariates=["x"],
            n_dose_points=10,
            n_bootstrap=0,
        )
        assert dr.diagnostics["avg_marginal_effect"] == pytest.approx(1.5, abs=0.2)

    def test_zero_effect_dose_recovered(self):
        df = self._linear_dgp(slope=0.0)
        dr = sp.dose_response(
            df,
            y="y",
            treat="dose",
            covariates=["x"],
            n_dose_points=10,
            n_bootstrap=0,
        )
        assert dr.diagnostics["avg_marginal_effect"] == pytest.approx(0.0, abs=0.2)
