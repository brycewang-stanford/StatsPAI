"""Tier D analytic special-case tests — front-door ID & PS balance diagnostics.

Part of the P1 "Tier D analytic special-cases" campaign (see
``.tierd_campaign/CAMPAIGN.md``). Both entry points were graded ``untested`` by
``scripts/tierd_classify.py``. Each test anchors to a known truth: the front-door
ATE on a linear mediator DGP equals the product of the D->M and M->Y effects,
and the standardized mean difference follows the exact Austin (2011) formula.

Entry points covered:
    sp.frontdoor    front-door adjustment (Pearl 1995), binary D
    sp.ps_balance   propensity-score balance table (Austin 2011 SMD)

Purely additive — no estimator numerics changed (campaign red line).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# sp.frontdoor — Pearl (1995) front-door adjustment
# ---------------------------------------------------------------------------
class TestFrontdoorAnalytic:
    """On a linear DGP  U -> D -> M -> Y  with U also -> Y (back-door open,
    front-door valid), the do-effect of D on Y is exactly (D->M) * (M->Y)."""

    @staticmethod
    def _mediation_dgp(seed=7, n=6000, a=2.0, b=1.0, gamma=1.5):
        rng = np.random.default_rng(seed)
        u = rng.normal(0, 1, n)  # unobserved confounder
        pd_ = 1.0 / (1.0 + np.exp(-(0.9 * u)))  # U confounds treatment
        d = (rng.uniform(size=n) < pd_).astype(int)
        m = a * d + rng.normal(0, 1, n)  # mediator: parent = D only
        y = b * m + gamma * u + rng.normal(0, 1, n)  # no direct D->Y path
        df = pd.DataFrame({"Y": y, "D": d, "M": m})
        return df, a * b

    def test_recovers_product_of_path_effects(self):
        df, true_effect = self._mediation_dgp()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.frontdoor(df, y="Y", d="D", m="M", n_boot=20, seed=0)
        # do-effect = a*b = 2.0; front-door identifies it despite U-confounding.
        assert res.estimate == pytest.approx(true_effect, rel=0.10)

    def test_beats_naive_ols_under_confounding(self):
        df, true_effect = self._mediation_dgp()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.frontdoor(df, y="Y", d="D", m="M", n_boot=20, seed=0)
        Y = df["Y"].values
        Xd = np.column_stack([np.ones(len(df)), df["D"].values])
        ols = np.linalg.lstsq(Xd, Y, rcond=None)[0][1]
        # Naive OLS is biased up by the open back-door; front-door is closer.
        assert abs(res.estimate - true_effect) < abs(ols - true_effect)
        assert ols > true_effect + 0.5  # confirm the back-door bias is real


# ---------------------------------------------------------------------------
# sp.ps_balance — Austin (2011) standardized mean difference
# ---------------------------------------------------------------------------
class TestPSBalanceAnalytic:

    def test_smd_raw_matches_austin_formula(self):
        # Re-derive SMD = (mean_t - mean_c) / sqrt((var_t + var_c) / 2) with
        # ddof=1 variances, exactly as Austin (2011) / the implementation.
        rng = np.random.default_rng(3)
        n = 3000
        t = rng.integers(0, 2, n)
        x = rng.normal(0, 1, n) + 0.4 * t  # imbalanced covariate
        df = pd.DataFrame({"t": t, "x": x})
        res = sp.ps_balance(df, treatment="t", covariates=["x"])
        xt, xc = x[t == 1], x[t == 0]
        denom = np.sqrt((xt.var(ddof=1) + xc.var(ddof=1)) / 2.0)
        expected = (xt.mean() - xc.mean()) / denom
        row = res.table.loc["x"]
        assert row["mean_treat"] == pytest.approx(xt.mean(), abs=1e-9)
        assert row["mean_control"] == pytest.approx(xc.mean(), abs=1e-9)
        assert row["smd_raw"] == pytest.approx(expected, abs=1e-9)

    def test_balanced_covariate_has_near_zero_smd(self):
        # Identically distributed across arms -> SMD ~ 0 (Austin's <0.1 rule).
        rng = np.random.default_rng(4)
        n = 4000
        df = pd.DataFrame({"t": rng.integers(0, 2, n), "x": rng.normal(0, 1, n)})
        res = sp.ps_balance(df, treatment="t", covariates=["x"])
        assert abs(res.table.loc["x", "smd_raw"]) < 0.08

    def test_ipw_reduces_imbalance(self):
        # IPW from the (correct) propensity model balances a confounded
        # covariate: |weighted SMD| < |raw SMD|.
        rng = np.random.default_rng(5)
        n = 5000
        x = rng.normal(0, 1, n)
        p = 1.0 / (1.0 + np.exp(-(1.2 * x)))  # x drives treatment
        t = (rng.uniform(size=n) < p).astype(int)
        df = pd.DataFrame({"t": t, "x": x})
        res = sp.ps_balance(df, treatment="t", covariates=["x"], method="logit")
        row = res.table.loc["x"]
        assert abs(row["smd_raw"]) > 0.3  # genuine initial imbalance
        assert abs(row["smd_weighted"]) < abs(row["smd_raw"])

    def test_variance_ratio_near_one_for_equal_variance(self):
        rng = np.random.default_rng(6)
        n = 4000
        t = rng.integers(0, 2, n)
        x = rng.normal(0, 2.0, n)  # same variance in both arms
        df = pd.DataFrame({"t": t, "x": x})
        res = sp.ps_balance(df, treatment="t", covariates=["x"])
        assert res.table.loc["x", "variance_ratio"] == pytest.approx(1.0, abs=0.1)
