"""
Tests for the rewritten ``statspai.multilevel`` toolkit.

The suite cross-validates against statsmodels' ``MixedLM`` wherever
possible (Gaussian LMMs) and against analytic / closed-form truths
otherwise (GLMMs, variance components, ICC, LR test, three-level
nesting).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Data-generation helpers
# ---------------------------------------------------------------------------


def _random_intercept_panel(seed: int = 0, n_groups: int = 40, n_per: int = 20,
                             sigma_u: float = 1.0, sigma_e: float = 0.5,
                             beta0: float = 2.0, beta1: float = 0.5):
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(n_groups), n_per)
    u = np.repeat(rng.normal(0, sigma_u, n_groups), n_per)
    x = rng.normal(0, 1, n_groups * n_per)
    y = beta0 + beta1 * x + u + rng.normal(0, sigma_e, n_groups * n_per)
    return pd.DataFrame({"y": y, "x": x, "g": group})


def _random_slope_panel(seed: int = 7, n_groups: int = 60, n_per: int = 20,
                         sigma_int: float = 1.0, sigma_slope: float = 0.5,
                         rho: float = 0.4, sigma_e: float = 0.5):
    rng = np.random.default_rng(seed)
    n = n_groups * n_per
    group = np.repeat(np.arange(n_groups), n_per)
    L = np.array([[sigma_int, 0.0],
                  [rho * sigma_slope, sigma_slope * np.sqrt(1 - rho ** 2)]])
    U = rng.normal(size=(n_groups, 2)) @ L.T
    u_int = np.repeat(U[:, 0], n_per)
    u_sl = np.repeat(U[:, 1], n_per)
    x = rng.normal(0, 1, n)
    y = 1.0 + 0.7 * x + u_int + u_sl * x + rng.normal(0, sigma_e, n)
    return pd.DataFrame({"y": y, "x": x, "g": group})


def _three_level_panel(seed: int = 1, n_schools: int = 80,
                       classes_per_school: int = 6,
                       students_per_class: int = 15,
                       sigma_s: float = 0.6, sigma_c: float = 0.4,
                       sigma_e: float = 0.5):
    rng = np.random.default_rng(seed)
    records = []
    school_u = rng.normal(0, sigma_s, n_schools)
    n_classes_total = n_schools * classes_per_school
    class_u = rng.normal(0, sigma_c, n_classes_total)
    cid = 0
    for s in range(n_schools):
        for _ in range(classes_per_school):
            for _ in range(students_per_class):
                x = rng.normal()
                y = (1.0 + 0.5 * x + school_u[s] + class_u[cid]
                     + rng.normal(0, sigma_e))
                records.append({"school": s, "klass": cid, "x": x, "y": y})
            cid += 1
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# LMM — basic random intercept
# ---------------------------------------------------------------------------


class TestRandomIntercept:
    def test_fixed_effects_recover(self):
        df = _random_intercept_panel()
        r = sp.mixed(df, "y", ["x"], "g")
        # Recovery within 3×SE is a very loose sanity check.
        assert abs(r.fixed_effects["_cons"] - 2.0) < 3 * r._se_fixed["_cons"]
        assert abs(r.fixed_effects["x"] - 0.5) < 3 * r._se_fixed["x"]

    def test_variance_components(self):
        df = _random_intercept_panel()
        r = sp.mixed(df, "y", ["x"], "g")
        assert 0.5 < r.variance_components["var(_cons)"] < 1.5
        assert 0.2 < r.variance_components["var(Residual)"] < 0.3

    def test_icc(self):
        df = _random_intercept_panel()
        r = sp.mixed(df, "y", ["x"], "g")
        icc = sp.icc(r)
        # True ICC = 1 / (1 + 0.25) = 0.8
        assert 0.6 < icc.estimate < 0.9
        assert icc.ci_lower < icc.estimate < icc.ci_upper

    def test_predict_matches_fit(self):
        df = _random_intercept_panel()
        r = sp.mixed(df, "y", ["x"], "g")
        p = r.predict()
        resid = df["y"].values - p.values
        # Residual variance ≈ σ²_ε = 0.25 (same sample)
        assert 0.2 < np.var(resid) < 0.32

    def test_predict_is_row_aligned_with_training_frame(self):
        """Regression test: predict(data=None) must not reorder rows."""
        rng = np.random.default_rng(0)
        n_g, n_per = 6, 5
        group = np.repeat(np.arange(n_g), n_per)
        # Scramble row order so groups are interleaved.
        idx = rng.permutation(len(group))
        group = group[idx]
        x = rng.normal(0, 1, len(group))
        u = np.array([rng.normal(0, 1, n_g)[g] for g in group])
        y = 2.0 + 0.5 * x + u + rng.normal(0, 0.3, len(group))
        df = pd.DataFrame({"y": y, "x": x, "g": group}).reset_index(drop=True)
        r = sp.mixed(df, "y", ["x"], "g")
        p_null = r.predict().values
        p_df = r.predict(df).values
        np.testing.assert_allclose(p_null, p_df)
        # Independent alignment check: the correlation with y should be
        # positive and not ~0 (which is what mis-alignment produces).
        assert np.corrcoef(df["y"].values, p_null)[0, 1] > 0.4

    def test_rejects_unhashable_group_values(self):
        df = _random_intercept_panel()
        df["g_bad"] = [list(range(3)) for _ in range(len(df))]  # unhashable
        with pytest.raises(TypeError):
            sp.mixed(df, "y", ["x"], "g_bad")

    def test_predict_marginal_vs_conditional(self):
        df = _random_intercept_panel()
        r = sp.mixed(df, "y", ["x"], "g")
        p_cond = r.predict(df, include_random=True)
        p_marg = r.predict(df, include_random=False)
        # The residual of the marginal prediction should include the
        # random-intercept variance, hence be larger.
        assert np.var(df["y"] - p_marg) > np.var(df["y"] - p_cond)

    def test_r_squared(self):
        df = _random_intercept_panel()
        r = sp.mixed(df, "y", ["x"], "g")
        r2 = r.r_squared()
        assert 0.0 < r2["marginal"] < r2["conditional"] < 1.0
        assert r2["conditional"] > 0.7  # conditional R² dominated by u_j

    def test_aic_bic_penalty(self):
        df = _random_intercept_panel()
        r = sp.mixed(df, "y", ["x"], "g")
        # AIC < BIC since N > exp(2).
        assert r.aic < r.bic


# ---------------------------------------------------------------------------
# LMM — random slope with unstructured G
# ---------------------------------------------------------------------------


class TestRandomSlopeUnstructured:
    def test_matches_statsmodels(self):
        pytest.importorskip("statsmodels")
        import statsmodels.formula.api as smf

        df = _random_slope_panel()
        r = sp.mixed(df, "y", ["x"], "g", x_random=["x"],
                     cov_type="unstructured")
        ref = smf.mixedlm("y ~ x", df, groups=df["g"],
                          re_formula="~x").fit(reml=True)
        np.testing.assert_allclose(
            r.fixed_effects.values, ref.fe_params.values, atol=1e-3
        )
        np.testing.assert_allclose(
            r._se_fixed.values, ref.bse_fe.values, atol=1e-3
        )
        # Var components: compare the 2×2 covariance block.
        our_G = np.array([
            [r.variance_components["var(_cons)"],
             r.variance_components["cov(_cons,x)"]],
            [r.variance_components["cov(_cons,x)"],
             r.variance_components["var(x)"]],
        ])
        np.testing.assert_allclose(our_G, ref.cov_re.values, atol=1e-2)
        assert abs(r.variance_components["var(Residual)"] - ref.scale) < 1e-2

    def test_diagonal_cov_type_drops_correlation(self):
        df = _random_slope_panel()
        r = sp.mixed(df, "y", ["x"], "g", x_random=["x"],
                     cov_type="diagonal")
        assert not any("corr(" in k for k in r.variance_components)

    def test_ranef_se_shape(self):
        df = _random_slope_panel()
        r = sp.mixed(df, "y", ["x"], "g", x_random=["x"])
        eff, se = r.ranef(conditional_se=True)
        assert eff.shape == (60, 2)
        assert se.shape == (60, 2)
        # SEs should be strictly positive.
        assert (se.values > 0).all()


# ---------------------------------------------------------------------------
# LMM — three-level nested
# ---------------------------------------------------------------------------


class TestThreeLevelNested:
    def test_separates_variance_components(self):
        pytest.importorskip("statsmodels")
        import statsmodels.formula.api as smf
        df = _three_level_panel()
        r = sp.mixed(df, "y", ["x"], group=["school", "klass"])
        # Cross-check against statsmodels with explicit re_formula="1".
        ref = smf.mixedlm(
            "y ~ x", df, groups=df["school"],
            re_formula="1",
            vc_formula={"klass": "0 + C(klass)"},
        ).fit(reml=True)
        np.testing.assert_allclose(
            r.fixed_effects.values, ref.fe_params.values, atol=1e-3
        )
        # statsmodels stores the scaled school variance in ``params``;
        # ``cov_re`` holds the raw covariance element.
        sm_school = float(ref.cov_re.values[0, 0])
        sm_class = float(ref.vcomp[0])
        assert abs(r.variance_components["var(_cons|school)"]
                   - sm_school) < 5e-2
        assert abs(r.variance_components["var(_cons|klass)"]
                   - sm_class) < 5e-2
        assert abs(r.variance_components["var(Residual)"] - ref.scale) < 1e-3

    def test_rejects_random_slope(self):
        df = _three_level_panel(n_schools=20)
        with pytest.raises(NotImplementedError):
            sp.mixed(df, "y", ["x"], group=["school", "klass"], x_random=["x"])

    def test_variance_components_keys(self):
        df = _three_level_panel(n_schools=30)
        r = sp.mixed(df, "y", ["x"], group=["school", "klass"])
        # Three-level fit must expose all three variance components and
        # both ICCs via the documented key names.
        assert "var(_cons|school)" in r.variance_components
        assert "var(_cons|klass)" in r.variance_components
        assert "var(Residual)" in r.variance_components
        assert "icc(school)" in r.variance_components
        assert "icc(school+klass)" in r.variance_components

    def test_singleton_inner_warns(self):
        # Build a panel where some schools have only one class.
        rng = np.random.default_rng(0)
        records = []
        cid = 0
        for s in range(20):
            n_classes = 1 if s < 5 else 4
            for _ in range(n_classes):
                for _ in range(12):
                    x = rng.normal()
                    y = 1 + 0.4 * x + rng.normal(0, 0.5)
                    records.append({"school": s, "klass": cid, "x": x, "y": y})
                cid += 1
        df = pd.DataFrame(records)
        with pytest.warns(RuntimeWarning):
            sp.mixed(df, "y", ["x"], group=["school", "klass"])


# ---------------------------------------------------------------------------
# GLMM — logit and Poisson
# ---------------------------------------------------------------------------


class TestMELogit:
    def test_recovers_truth(self):
        rng = np.random.default_rng(42)
        n_groups, n_per = 80, 25
        n = n_groups * n_per
        group = np.repeat(np.arange(n_groups), n_per)
        x = rng.normal(0, 1, n)
        u = np.repeat(rng.normal(0, 0.8, n_groups), n_per)
        eta = -0.5 + 0.9 * x + u
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, p)
        df = pd.DataFrame({"y": y, "x": x, "g": group})
        r = sp.melogit(df, "y", ["x"], "g")
        assert r.family == "binomial" and r.link == "logit"
        # Slope and variance should be in the right neighbourhood.
        assert 0.6 < r.fixed_effects["x"] < 1.2
        assert 0.3 < r.variance_components["var(_cons)"] < 1.5

    def test_odds_ratios(self):
        rng = np.random.default_rng(3)
        n = 1000
        g = np.repeat(np.arange(40), 25)
        u = np.repeat(rng.normal(0, 0.5, 40), 25)
        x = rng.normal(0, 1, n)
        p = 1 / (1 + np.exp(-(0.2 + 0.7 * x + u)))
        df = pd.DataFrame({"y": rng.binomial(1, p), "x": x, "g": g})
        r = sp.melogit(df, "y", ["x"], "g")
        orr = r.odds_ratios()
        assert "OR" in orr.columns and (orr["OR"] > 0).all()
        assert (orr["lower"] < orr["OR"]).all()
        assert (orr["upper"] > orr["OR"]).all()


class TestMEGLMResultContract:
    """Contract tests — every result class must expose the same surface."""

    def _fit(self):
        rng = np.random.default_rng(0)
        n_g, n_per = 40, 20
        g = np.repeat(np.arange(n_g), n_per)
        x = rng.normal(0, 1, n_g * n_per)
        u = np.repeat(rng.normal(0, 0.5, n_g), n_per)
        eta = 0.1 + 0.5 * x + u
        p = 1 / (1 + np.exp(-eta))
        y = rng.binomial(1, p)
        df = pd.DataFrame({"y": y, "x": x, "g": g})
        return sp.melogit(df, "y", ["x"], "g")

    def test_to_latex(self):
        tex = self._fit().to_latex()
        assert r"\begin{table}" in tex and r"\end{table}" in tex

    def test_plot_caterpillar(self):
        fig, ax = self._fit().plot(kind="caterpillar")
        # matplotlib artists created — basic smoke.
        assert fig is not None and ax is not None

    def test_plot_invalid_kind(self):
        with pytest.raises(ValueError):
            self._fit().plot(kind="not-a-kind")


class TestMEPoisson:
    def test_recovers_truth(self):
        rng = np.random.default_rng(2026)
        n_groups, n_per = 100, 10
        n = n_groups * n_per
        group = np.repeat(np.arange(n_groups), n_per)
        x = rng.normal(0, 1, n)
        u = np.repeat(rng.normal(0, 0.5, n_groups), n_per)
        mu = np.exp(1.0 + 0.4 * x + u)
        y = rng.poisson(mu)
        df = pd.DataFrame({"y": y, "x": x, "g": group})
        r = sp.mepoisson(df, "y", ["x"], "g")
        assert r.family == "poisson" and r.link == "log"
        assert abs(r.fixed_effects["x"] - 0.4) < 0.1
        assert 0.1 < r.variance_components["var(_cons)"] < 0.5
        # Incidence rate ratio for x is about exp(0.4) ≈ 1.49.
        irr = r.incidence_rate_ratios()
        assert 1.3 < irr.loc["x", "IRR"] < 1.7


# ---------------------------------------------------------------------------
# Adaptive Gauss-Hermite quadrature (nAGQ > 1)
# ---------------------------------------------------------------------------


class TestAGHQ:
    """AGHQ should match Laplace at nAGQ=1, beat it on small clusters."""

    def _small_cluster_binary(self, seed: int = 2026,
                              n_groups: int = 200, n_per: int = 3,
                              sigma_u: float = 1.5, beta_x: float = 0.8):
        rng = np.random.default_rng(seed)
        g = np.repeat(np.arange(n_groups), n_per)
        x = rng.normal(0, 1, n_groups * n_per)
        u = np.repeat(rng.normal(0, sigma_u, n_groups), n_per)
        eta = -0.5 + beta_x * x + u
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, p)
        return pd.DataFrame({"y": y, "x": x, "g": g}), sigma_u, beta_x

    def test_nagq1_equals_laplace(self):
        df, _, _ = self._small_cluster_binary()
        r1 = sp.melogit(df, "y", ["x"], "g", nAGQ=1)
        # nAGQ=1 path uses the Laplace branch — verify by method label.
        assert r1._method == "laplace"

    def test_nagq_label(self):
        df, _, _ = self._small_cluster_binary()
        r7 = sp.melogit(df, "y", ["x"], "g", nAGQ=7)
        assert r7._method == "AGHQ(nAGQ=7)"

    def test_aghq_logL_at_least_laplace(self):
        # The integrated log-likelihood under AGHQ must be >= Laplace
        # (more accurate quadrature, higher likelihood at the optimum).
        df, _, _ = self._small_cluster_binary()
        r1 = sp.melogit(df, "y", ["x"], "g", nAGQ=1)
        r7 = sp.melogit(df, "y", ["x"], "g", nAGQ=7)
        # Allow tiny numerical slack — AGHQ should not be materially worse.
        assert r7.log_likelihood >= r1.log_likelihood - 1e-3

    def test_aghq_var_less_biased(self):
        # On small clusters, Laplace severely underestimates var(u).
        # AGHQ should be closer to truth than Laplace.
        df, sigma_u, _ = self._small_cluster_binary()
        true_var = sigma_u ** 2
        r1 = sp.melogit(df, "y", ["x"], "g", nAGQ=1)
        r7 = sp.melogit(df, "y", ["x"], "g", nAGQ=7)
        v1 = r1.variance_components["var(_cons)"]
        v7 = r7.variance_components["var(_cons)"]
        assert abs(v7 - true_var) < abs(v1 - true_var)

    def test_aghq_converges_in_nodes(self):
        # nAGQ=15 should be ~indistinguishable from nAGQ=7 (4 sig figs).
        df, _, _ = self._small_cluster_binary()
        r7 = sp.melogit(df, "y", ["x"], "g", nAGQ=7)
        r15 = sp.melogit(df, "y", ["x"], "g", nAGQ=15)
        # log-likelihood and beta should agree to 3 decimals.
        assert abs(r7.log_likelihood - r15.log_likelihood) < 1e-2
        assert abs(r7.fixed_effects["x"] - r15.fixed_effects["x"]) < 1e-2

    def test_aghq_rejects_random_slopes(self):
        # AGHQ is restricted to scalar random effects.
        df = _random_slope_panel(n_groups=20, n_per=20)
        # Map y to binary so this exercises the GLMM path.
        df["yb"] = (df["y"] > df["y"].median()).astype(int)
        with pytest.raises(ValueError, match="random-intercept"):
            sp.melogit(df, "yb", ["x"], "g", x_random=["x"], nAGQ=7)

    def test_invalid_nagq(self):
        df, _, _ = self._small_cluster_binary()
        with pytest.raises(ValueError, match="nAGQ"):
            sp.melogit(df, "y", ["x"], "g", nAGQ=0)


# ---------------------------------------------------------------------------
# Gamma family (megamma)
# ---------------------------------------------------------------------------


class TestMEGamma:
    def _gamma_panel(self, seed: int = 2026, n_g: int = 60, n_per: int = 25,
                      phi: float = 0.4, sigma_u: float = 0.5,
                      beta0: float = 0.5, beta_x: float = 0.7):
        rng = np.random.default_rng(seed)
        g = np.repeat(np.arange(n_g), n_per)
        x = rng.normal(0, 1, n_g * n_per)
        u = np.repeat(rng.normal(0, sigma_u, n_g), n_per)
        mu = np.exp(beta0 + beta_x * x + u)
        y = rng.gamma(shape=1.0 / phi, scale=mu * phi)
        return pd.DataFrame({"y": y, "x": x, "g": g})

    def test_recovers_truth(self):
        df = self._gamma_panel()
        r = sp.megamma(df, "y", ["x"], "g")
        assert r.family == "gamma" and r.link == "log"
        assert abs(r.fixed_effects["x"] - 0.7) < 0.05
        assert abs(r.fixed_effects["_cons"] - 0.5) < 0.1
        assert 0.15 < r.variance_components["var(_cons)"] < 0.40
        # Dispersion within ~25% of truth on this sample size.
        assert 0.30 < r.dispersion < 0.50

    def test_n_params_includes_dispersion(self):
        df = self._gamma_panel(n_g=20, n_per=10)
        r = sp.megamma(df, "y", ["x"], "g")
        # 2 fixed (cons + x) + 1 cov (var(cons)) + 1 dispersion = 4
        assert r.n_params == 4
        assert r.dispersion is not None

    def test_summary_lists_dispersion(self):
        df = self._gamma_panel(n_g=20, n_per=10)
        r = sp.megamma(df, "y", ["x"], "g")
        s = r.summary()
        assert "phi" in s.lower() or "dispersion" in s.lower()


# ---------------------------------------------------------------------------
# Negative binomial family (menbreg)
# ---------------------------------------------------------------------------


class TestMENegBin:
    def _nb_panel(self, seed: int = 2026, n_g: int = 60, n_per: int = 25,
                   alpha: float = 0.5, sigma_u: float = 0.4,
                   beta0: float = 1.0, beta_x: float = 0.4):
        rng = np.random.default_rng(seed)
        g = np.repeat(np.arange(n_g), n_per)
        x = rng.normal(0, 1, n_g * n_per)
        u = np.repeat(rng.normal(0, sigma_u, n_g), n_per)
        mu = np.exp(beta0 + beta_x * x + u)
        r_param = 1.0 / alpha
        y = rng.negative_binomial(r_param, r_param / (r_param + mu))
        return pd.DataFrame({"y": y, "x": x, "g": g})

    def test_recovers_truth(self):
        df = self._nb_panel()
        r = sp.menbreg(df, "y", ["x"], "g")
        assert r.family == "nbinomial" and r.link == "log"
        assert abs(r.fixed_effects["x"] - 0.4) < 0.05
        assert 0.05 < r.variance_components["var(_cons)"] < 0.30
        assert 0.30 < r.dispersion < 0.70

    def test_irr_available(self):
        df = self._nb_panel(n_g=30, n_per=15)
        r = sp.menbreg(df, "y", ["x"], "g")
        irr = r.incidence_rate_ratios()
        assert "IRR" in irr.columns and (irr["IRR"] > 0).all()

    def test_alias_negbin(self):
        df = self._nb_panel(n_g=30, n_per=15)
        r1 = sp.menbreg(df, "y", ["x"], "g")
        r2 = sp.meglm(df, "y", ["x"], "g", family="negbin")
        # Same fit (alias resolution).
        assert abs(r1.log_likelihood - r2.log_likelihood) < 1e-6


# ---------------------------------------------------------------------------
# Ordinal logit (meologit)
# ---------------------------------------------------------------------------


class TestMEOLogit:
    def _ordinal_panel(self, seed: int = 7, n_g: int = 80, n_per: int = 25,
                        beta_x: float = 0.8, sigma_u: float = 0.6,
                        kappa=(-1.0, 0.0, 1.5)):
        rng = np.random.default_rng(seed)
        g = np.repeat(np.arange(n_g), n_per)
        x = rng.normal(0, 1, n_g * n_per)
        u = np.repeat(rng.normal(0, sigma_u, n_g), n_per)
        eta = beta_x * x + u
        kappa_arr = np.asarray(kappa, dtype=float)
        F = lambda t: 1.0 / (1.0 + np.exp(-t))
        P_le = np.column_stack([F(k - eta) for k in kappa_arr])
        K = len(kappa_arr) + 1
        n = len(eta)
        probs = np.zeros((n, K))
        probs[:, 0] = P_le[:, 0]
        for k in range(1, K - 1):
            probs[:, k] = P_le[:, k] - P_le[:, k - 1]
        probs[:, K - 1] = 1.0 - P_le[:, K - 2]
        y = np.array([rng.choice(K, p=p) + 1 for p in probs])
        return pd.DataFrame({"y": y, "x": x, "g": g})

    def test_recovers_truth(self):
        df = self._ordinal_panel()
        r = sp.meologit(df, "y", ["x"], "g")
        assert r.family == "ordinal" and r.link == "logit"
        assert abs(r.fixed_effects["x"] - 0.8) < 0.1
        # Thresholds in correct order and close to truth.
        thr = r.thresholds.values
        assert thr[0] < thr[1] < thr[2]
        assert abs(thr[0] - (-1.0)) < 0.2
        assert abs(thr[2] - 1.5) < 0.2

    def test_threshold_ordering_strict(self):
        # The δ → κ reparam guarantees strict ordering for any δ.
        df = self._ordinal_panel(n_g=30, n_per=15)
        r = sp.meologit(df, "y", ["x"], "g")
        thr = r.thresholds.values
        assert np.all(np.diff(thr) > 0)

    def test_no_intercept_in_fixed_effects(self):
        df = self._ordinal_panel(n_g=30, n_per=15)
        r = sp.meologit(df, "y", ["x"], "g")
        assert "_cons" not in r.fixed_effects.index
        assert list(r.fixed_effects.index) == ["x"]

    def test_summary_lists_thresholds(self):
        df = self._ordinal_panel(n_g=30, n_per=15)
        r = sp.meologit(df, "y", ["x"], "g")
        s = r.summary()
        assert "Thresholds" in s or "cutpoint" in s.lower() or "cut" in s.lower()

    def test_K2_rejected(self):
        # meologit needs K >= 3 categories.
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "y": rng.binomial(1, 0.5, 200) + 1,  # 1 or 2
            "x": rng.normal(0, 1, 200),
            "g": np.repeat(np.arange(20), 10),
        })
        with pytest.raises(ValueError, match="3 outcome"):
            sp.meologit(df, "y", ["x"], "g")


# ---------------------------------------------------------------------------
# lrtest
# ---------------------------------------------------------------------------


class TestLRTest:
    def test_variance_only_boundary(self):
        # Simulate with NO random effect: LR stat should be small,
        # p-value should be large.  (The previous `or` assertion was
        # vacuous; this enforces both conditions.)
        rng = np.random.default_rng(11)
        g = np.repeat(np.arange(30), 20)
        x = rng.normal(0, 1, 600)
        y = 1.0 + 0.5 * x + rng.normal(0, 0.5, 600)
        df = pd.DataFrame({"y": y, "x": x, "g": g})

        r_full = sp.mixed(df, "y", ["x"], "g")
        assert r_full._lr_test["chi2"] < 3.84
        assert r_full._lr_test["p"] > 0.05

    def test_nested_models(self):
        df = _random_slope_panel(n_groups=100, n_per=30, sigma_slope=0.3,
                                 rho=0.0, sigma_int=0.6, sigma_e=0.5)
        r_full = sp.mixed(df, "y", ["x"], "g", x_random=["x"],
                          cov_type="diagonal")
        r_restricted = sp.mixed(df, "y", ["x"], "g")
        lr = sp.lrtest(r_restricted, r_full, boundary=True)
        # A random slope is present → we should strongly reject.
        assert lr.chi2 > 0
        assert lr.p_value < 0.05

    def test_multi_component_boundary_warns(self):
        # Unstructured → diagonal adds 2 free params on the boundary.
        df = _random_slope_panel(n_groups=80, n_per=30, sigma_slope=0.3,
                                 rho=0.2, sigma_int=0.6, sigma_e=0.5)
        r_full = sp.mixed(df, "y", ["x"], "g", x_random=["x"],
                          cov_type="unstructured")
        r_restricted = sp.mixed(df, "y", ["x"], "g")
        with pytest.warns(RuntimeWarning):
            sp.lrtest(r_restricted, r_full, boundary=True)

    def test_rejects_cross_family(self):
        rng = np.random.default_rng(3)
        n = 400
        g = np.repeat(np.arange(20), 20)
        x = rng.normal(0, 1, n)
        u = np.repeat(rng.normal(0, 0.3, 20), 20)
        df = pd.DataFrame({
            "y_bin": rng.binomial(1, 1 / (1 + np.exp(-(0.1 + 0.5 * x + u)))),
            "y_ct": rng.poisson(np.exp(0.1 + 0.3 * x + u)),
            "x": x, "g": g,
        })
        r_b = sp.melogit(df, "y_bin", ["x"], "g")
        r_p = sp.mepoisson(df, "y_ct", ["x"], "g")
        with pytest.raises(ValueError):
            sp.lrtest(r_b, r_p)

    def test_rejects_reml_with_fixed_effect_change(self):
        df = _random_intercept_panel()
        df["z"] = np.random.default_rng(0).normal(size=len(df))
        r_small = sp.mixed(df, "y", ["x"], "g", method="reml")
        r_big = sp.mixed(df, "y", ["x", "z"], "g", method="reml")
        with pytest.raises(ValueError):
            sp.lrtest(r_small, r_big)


# ---------------------------------------------------------------------------
# ICC helper
# ---------------------------------------------------------------------------


class TestICC:
    def test_bounded_estimate(self):
        df = _random_intercept_panel()
        r = sp.mixed(df, "y", ["x"], "g")
        icc = sp.icc(r)
        assert 0.0 <= icc.estimate <= 1.0
        assert icc.ci_lower <= icc.estimate <= icc.ci_upper

    def test_float_coercion(self):
        df = _random_intercept_panel()
        r = sp.mixed(df, "y", ["x"], "g")
        assert isinstance(float(sp.icc(r)), float)

    def test_n_boot_not_implemented(self):
        df = _random_intercept_panel()
        r = sp.mixed(df, "y", ["x"], "g")
        with pytest.raises(NotImplementedError):
            sp.icc(r, n_boot=100)

    def test_small_n_groups_warns(self):
        df = _random_intercept_panel(n_groups=10, n_per=20)
        r = sp.mixed(df, "y", ["x"], "g")
        with pytest.warns(RuntimeWarning):
            sp.icc(r)


# ---------------------------------------------------------------------------
# Export formatters
# ---------------------------------------------------------------------------


class TestExportMethods:
    def test_markdown(self):
        df = _random_intercept_panel()
        r = sp.mixed(df, "y", ["x"], "g")
        md = r.to_markdown()
        assert "Fixed effects" in md and "Variance components" in md

    def test_latex(self):
        df = _random_intercept_panel()
        r = sp.mixed(df, "y", ["x"], "g")
        tex = r.to_latex()
        assert r"\begin{table}" in tex and r"\end{table}" in tex

    def test_html_repr(self):
        df = _random_intercept_panel()
        r = sp.mixed(df, "y", ["x"], "g")
        html = r._repr_html_()
        assert "Linear Mixed Model" in html and "<table>" in html


# ---------------------------------------------------------------------------
# Wald restrictions
# ---------------------------------------------------------------------------


class TestWaldTest:
    def test_single_coefficient(self):
        df = _random_intercept_panel()
        r = sp.mixed(df, "y", ["x"], "g")
        w = r.wald_test(["x"])
        assert w["df"] == 1
        assert w["chi2"] > 0
        assert w["p_value"] < 0.05  # slope is significant by design
