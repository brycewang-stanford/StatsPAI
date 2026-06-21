"""Tier D P2 known-truth upgrades — QTE, multi-valued treatment, distributional TE.

Part of the P1/P2 "Tier D analytic special-cases" campaign (see
``.tierd_campaign/CAMPAIGN.md``). All three were graded ``weak`` by
``scripts/tierd_classify.py``. Each anchors to a known-DGP recovery:

    sp.qte                a pure location shift Y(1)=Y(0)+tau gives a constant
                          quantile treatment effect tau at every quantile.
    sp.multi_treatment    AIPW recovers each arm's known effect vs the reference.
    sp.distributional_te  an upward location shift makes the treated CDF lie
                          below the control CDF (stochastic dominance) and the
                          KS statistic is large; under no effect it is ~0.

Purely additive — no estimator numerics changed (campaign red line).

NB (Tier D edge findings, reported not fixed):
- ``sp.qte(n_boot=0)`` raises (np.percentile over an empty bootstrap array),
  same edge as ``sp.cic``; tests pass a small positive bootstrap count.
- ``sp.distributional_te.ks_pvalue`` is unreliable (ks_stat=0.69 reported with
  ks_pvalue=0.70, where scipy's KS p-value is ~1e-170). Tests anchor on the
  correct ``ks_stat`` and the CDF-dominance ``dte`` instead. See
  ``.tierd_campaign/FINDINGS_minor_edge_cases.md``.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# Bootstrap only sizes the CIs; every assertion below is on a point estimate /
# statistic, so a small n_boot keeps these fast without weakening the anchor.
N = 2000
NBOOT = 20


# ---------------------------------------------------------------------------
# sp.qte — quantile treatment effects
# ---------------------------------------------------------------------------
class TestQTEAnalytic:

    def test_location_shift_is_constant_across_quantiles(self):
        rng = np.random.default_rng(0)
        t = rng.integers(0, 2, N)
        y = rng.normal(0, 1, N) + 2.0 * t  # constant shift tau = 2
        res = sp.qte(
            pd.DataFrame({"y": y, "t": t}),
            y="y",
            treatment="t",
            quantiles=[0.25, 0.5, 0.75],
            n_boot=NBOOT,
        )
        np.testing.assert_allclose(res.effects, 2.0, atol=0.25)

    def test_ate_recovers_mean_shift(self):
        rng = np.random.default_rng(1)
        t = rng.integers(0, 2, N)
        y = rng.normal(0, 1, N) + 2.0 * t
        res = sp.qte(
            pd.DataFrame({"y": y, "t": t}),
            y="y",
            treatment="t",
            quantiles=[0.5],
            n_boot=NBOOT,
        )
        assert res.ate == pytest.approx(2.0, abs=0.2)

    def test_no_effect_gives_zero_qte(self):
        rng = np.random.default_rng(2)
        t = rng.integers(0, 2, N)
        y = rng.normal(0, 1, N)  # treatment unrelated to outcome
        res = sp.qte(
            pd.DataFrame({"y": y, "t": t}),
            y="y",
            treatment="t",
            quantiles=[0.25, 0.5, 0.75],
            n_boot=NBOOT,
        )
        np.testing.assert_allclose(res.effects, 0.0, atol=0.25)


# ---------------------------------------------------------------------------
# sp.multi_treatment — multi-valued treatment via AIPW
# ---------------------------------------------------------------------------
class TestMultiTreatmentAnalytic:

    @staticmethod
    def _three_arm_dgp(seed=0, n=N):
        rng = np.random.default_rng(seed)
        T = rng.integers(0, 3, n)
        x = rng.normal(0, 1, n)
        # Known effects vs arm 0: arm 1 = +1.0, arm 2 = +2.5.
        y = 1.0 * (T == 1) + 2.5 * (T == 2) + 0.5 * x + rng.normal(0, 1, n)
        return pd.DataFrame({"y": y, "T": T, "x": x})

    def test_recovers_per_arm_effects(self):
        df = self._three_arm_dgp()
        res = sp.multi_treatment(
            df, y="y", treat="T", covariates=["x"], reference=0, n_bootstrap=NBOOT
        )
        eff = res.detail.set_index("treatment")["estimate"]
        assert eff[1] == pytest.approx(1.0, abs=0.25)
        assert eff[2] == pytest.approx(2.5, abs=0.25)

    def test_reference_arm_excluded_and_ordering(self):
        df = self._three_arm_dgp()
        res = sp.multi_treatment(
            df, y="y", treat="T", covariates=["x"], reference=0, n_bootstrap=NBOOT
        )
        assert 0 not in set(res.detail["treatment"])
        eff = res.detail.set_index("treatment")["estimate"]
        assert eff[2] > eff[1]  # arm 2 effect exceeds arm 1


# ---------------------------------------------------------------------------
# sp.distributional_te — distributional treatment effects
# ---------------------------------------------------------------------------
class TestDistributionalTEAnalytic:

    def test_upward_shift_gives_stochastic_dominance(self):
        # Treated distribution shifted up by 2 -> treated CDF lies below the
        # control CDF (DTE = F_treated - F_control <= 0 on average), and the KS
        # distance between the two distributions is large.
        rng = np.random.default_rng(0)
        t = rng.integers(0, 2, N)
        y = rng.normal(0, 1, N) + 2.0 * t
        res = sp.distributional_te(
            pd.DataFrame({"y": y, "t": t}),
            y="y",
            treatment="t",
            n_grid=40,
            n_boot=NBOOT,
        )
        assert np.nanmean(res.dte) < 0  # treated stochastically dominates
        assert res.ks_stat > 0.3  # distributions clearly differ

    def test_no_effect_has_small_ks_distance(self):
        rng = np.random.default_rng(3)
        t = rng.integers(0, 2, N)
        y = rng.normal(0, 1, N)  # identical distributions across arms
        res = sp.distributional_te(
            pd.DataFrame({"y": y, "t": t}),
            y="y",
            treatment="t",
            n_grid=40,
            n_boot=NBOOT,
        )
        assert res.ks_stat < 0.1
