"""Tier D P2 known-truth upgrades — Granger causality.

Part of the P1/P2 "Tier D analytic special-cases" campaign (see
``.tierd_campaign/CAMPAIGN.md``). ``sp.granger_causality`` was graded ``weak``
and the Tier D probe surfaced a HIGH-severity correctness bug (placeholder Wald
variance ``V = sigma2*I`` ignoring ``(X'X)^-1``; F off by ~factor T·Var(X)),
reported in ``.tierd_campaign/BUG_granger_causality_wald_variance.md`` and fixed
per CLAUDE.md §12 (CHANGELOG + MIGRATION, ``⚠️ Correctness fix``).

These tests are the regression guard: on a DGP where x[t-1] drives y, the
Granger F matches the restricted-vs-unrestricted OLS F-test and detects the
causal direction (x->y) but not the reverse (y->x).

Purely additive test file (the estimator fix is logged separately).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


class TestGrangerCausalityAnalytic:

    @staticmethod
    def _causal_dgp(seed=0, T=800):
        rng = np.random.default_rng(seed)
        x = np.zeros(T)
        y = np.zeros(T)
        for k in range(1, T):
            x[k] = 0.5 * x[k - 1] + rng.normal()
            y[k] = 0.3 * y[k - 1] + 0.8 * x[k - 1] + rng.normal()  # x -> y
        return pd.DataFrame({"x": x, "y": y})

    @staticmethod
    def _hand_f(df, cause, effect, p=2):
        # Restricted (effect on its own lags) vs unrestricted (+ cause lags).
        v = df[effect].values
        c = df[cause].values
        T = len(v)
        Y = v[p:]
        n = len(Y)

        def lagmat(s):
            return np.column_stack([s[p - 1 - j : T - 1 - j][:n] for j in range(p)])

        Xr = np.column_stack([np.ones(n), lagmat(v)])
        Xu = np.column_stack([np.ones(n), lagmat(v), lagmat(c)])

        def rss(X, z):
            b = np.linalg.lstsq(X, z, rcond=None)[0]
            res = z - X @ b
            return res @ res

        return ((rss(Xr, Y) - rss(Xu, Y)) / p) / (rss(Xu, Y) / (n - Xu.shape[1]))

    def test_detects_true_causal_direction(self):
        df = self._causal_dgp()
        g = sp.granger_causality(data=df, causing="x", caused="y", lags=2)
        assert g["p_value"] < 0.01
        assert g["reject"] is True or g["reject"] == np.True_

    def test_does_not_detect_reverse_direction(self):
        df = self._causal_dgp()
        g = sp.granger_causality(data=df, causing="y", caused="x", lags=2)
        assert g["p_value"] > 0.05

    def test_f_statistic_matches_ols_f_test(self):
        # The fixed Wald F equals the restricted-vs-unrestricted OLS F-test up
        # to the MLE-vs-unbiased sigma^2 factor T/(T - n_params) (~0.6% here).
        df = self._causal_dgp()
        g = sp.granger_causality(data=df, causing="x", caused="y", lags=2)
        f_hand = self._hand_f(df, cause="x", effect="y", p=2)
        assert g["F_stat"] == pytest.approx(f_hand, rel=0.02)
        assert g["F_stat"] > 50  # the placeholder-variance bug gave ~0.36


class TestVARStandardErrorConventions:

    @staticmethod
    def _var_dgp(seed=7, T=240):
        rng = np.random.default_rng(seed)
        y1 = np.zeros(T)
        y2 = np.zeros(T)
        for t in range(1, T):
            y1[t] = 0.45 * y1[t - 1] + 0.20 * y2[t - 1] + rng.normal(scale=0.5)
            y2[t] = -0.25 * y1[t - 1] + 0.55 * y2[t - 1] + rng.normal(scale=0.5)
        return pd.DataFrame({"y1": y1, "y2": y2})

    def test_var_exposes_stata_and_r_lm_se_denominators(self):
        df = self._var_dgp()
        v_stata = sp.var(df, variables=["y1", "y2"], lags=2, se_df="stata")
        v_r = sp.var(df, variables=["y1", "y2"], lags=2, se_df="r")

        n_params = 2 * 2 + 1  # two variables, two lags, constant
        expected_ratio = np.sqrt(v_stata.n_obs / (v_stata.n_obs - n_params))
        assert v_stata.se_df == "stata"
        assert v_r.se_df == "r"

        for eq in ("y1", "y2"):
            ratio = v_r.coefs[eq]["se"] / v_stata.coefs[eq]["se"]
            assert ratio.to_numpy() == pytest.approx(expected_ratio, rel=1e-12)

    def test_var_rejects_unknown_se_df_convention(self):
        df = self._var_dgp()
        with pytest.raises(ValueError, match="se_df"):
            sp.var(df, variables=["y1", "y2"], lags=2, se_df="mystery")
