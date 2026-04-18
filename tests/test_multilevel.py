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
# lrtest
# ---------------------------------------------------------------------------


class TestLRTest:
    def test_variance_only_boundary(self):
        # Simulate with NO random effect: LR stat should be small.
        rng = np.random.default_rng(11)
        g = np.repeat(np.arange(30), 20)
        x = rng.normal(0, 1, 600)
        y = 1.0 + 0.5 * x + rng.normal(0, 0.5, 600)
        df = pd.DataFrame({"y": y, "x": x, "g": g})

        r_full = sp.mixed(df, "y", ["x"], "g")
        # Restricted: pooled OLS captured via the full model's _lr_test dict.
        assert r_full._lr_test["chi2"] < 3.84 or r_full._lr_test["p"] > 0.01

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
