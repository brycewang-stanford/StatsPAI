"""Option and degenerate-input tests for the ``sp.iv`` dispatcher,
``sp.iv.iv_diag``, the plausibly-exogenous bounds and ``mte`` (scan round 3).

Numerical assertions compare with 2SLS algebra written here (one endogenous
regressor after partialling out the constant and the control, so every
quantity is a ratio of inner products) or with an identity.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility, StatsPAIError
from statspai.iv.mte import mte
from statspai.iv.plausibly_exogenous import (
    plausibly_exogenous_ltz,
    plausibly_exogenous_uci,
)
from statspai.iv.weak_identification import kleibergen_paap_rk
from statspai.iv.weak_iv_ci import conditional_lr_ci

RTOL = 1e-9
N = 500
Z_COLS = ["z0", "z1", "z2"]
FORMULA = "y ~ (d1 ~ z0 + z1 + z2) + x"


def _data(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(N, 3))
    x = rng.normal(size=N)
    het = np.sqrt(1 + Z[:, 0] ** 2)
    d1 = Z @ np.array([0.6, 0.4, 0.2]) + 0.3 * x + rng.normal(size=N) * het
    y = d1 + rng.normal(size=N) * het
    df = pd.DataFrame(
        {
            "y": y,
            "d1": d1,
            "x": x,
            "g": rng.integers(0, 25, N),
            "w": rng.uniform(0.5, 2.0, N),
        }
    )
    for j in range(3):
        df[f"z{j}"] = Z[:, j]
    return df


def _partial(df, cols, extra=None):
    blocks = [np.ones(len(df)), df["x"].to_numpy()]
    if extra is not None:
        blocks.append(extra)
    W = np.column_stack(blocks)
    M = df[cols].to_numpy()
    return M - W @ np.linalg.lstsq(W, M, rcond=None)[0]


def _tsls(df):
    """beta, the fitted endogenous regressor, the residual, and d'P_Z d."""
    Y, D, Z = (
        _partial(df, ["y"])[:, 0],
        _partial(df, ["d1"])[:, 0],
        _partial(df, Z_COLS),
    )
    Dh = Z @ np.linalg.lstsq(Z, D, rcond=None)[0]
    beta = (Dh @ Y) / (Dh @ D)
    return beta, Dh, Y - beta * D, Dh @ D


def _cluster_ss(scores, groups):
    s = pd.Series(scores).groupby(np.asarray(groups)).sum().to_numpy()
    return float(s @ s)


def _iv(formula, df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.iv(formula, df, **kw)


def _diag(df, **kw):
    kw.setdefault("n_boot", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.iv.iv_diag(
            df, y="y", endog="d1", instruments=Z_COLS, exog=["x"], **kw
        )


# ------------------------------------------------------------------ dispatcher


class TestKClassOptionsReachTheEstimator:
    def test_2sls_three_variances_by_hand(self):
        df = _data()
        beta, Dh, e, dd = _tsls(df)
        dof = N - 3
        res = _iv(FORMULA, df)
        np.testing.assert_allclose(res.params["d1"], beta, rtol=RTOL)
        np.testing.assert_allclose(
            res.std_errors["d1"] ** 2, e @ e / dof / dd, rtol=RTOL
        )
        robust = _iv(FORMULA, df, robust="robust")
        np.testing.assert_allclose(
            robust.std_errors["d1"] ** 2,
            ((Dh * e) @ (Dh * e)) / dd**2 * N / dof,
            rtol=RTOL,
        )
        G = df.g.nunique()
        clustered = _iv(FORMULA, df, cluster="g")
        np.testing.assert_allclose(
            clustered.std_errors["d1"] ** 2,
            _cluster_ss(Dh * e, df.g) / dd**2 * G / (G - 1) * (N - 1) / dof,
            rtol=RTOL,
        )

    def test_vce_spelling_is_the_same_request_as_robust(self):
        df = _data()
        a = _iv(FORMULA, df, robust="robust")
        b = _iv(FORMULA, df, vce="robust")
        np.testing.assert_allclose(
            a.std_errors.to_numpy(), b.std_errors.to_numpy(), rtol=RTOL
        )

    def test_just_identified_is_the_ratio_form(self):
        df = _data()
        z, d, y = (_partial(df, [c])[:, 0] for c in ("z0", "d1", "y"))
        res = _iv("y ~ (d1 ~ z0) + x", df)
        np.testing.assert_allclose(res.params["d1"], (z @ y) / (z @ d), rtol=RTOL)

    @pytest.mark.parametrize("method", ["2sls", "liml", "fuller", "jive"])
    def test_just_identified_k_class_members_agree(self, method):
        """One instrument: LIML's kappa is one, and 2SLS, LIML and JIVE-type
        estimators differ only when the model is over-identified (Fuller
        shifts kappa by alpha/(n-K), JIVE by leave-one-out leverage; both are
        O(1/n) on this design, hence the loose bound for those two)."""
        df = _data()
        base = _iv("y ~ (d1 ~ z0) + x", df).params["d1"]
        other = _iv("y ~ (d1 ~ z0) + x", df, method=method).params["d1"]
        tol = 1e-8 if method in ("2sls", "liml") else 2e-2
        np.testing.assert_allclose(other, base, rtol=tol)

    def test_weights_are_weighted_2sls(self):
        df = _data()
        w = df["w"].to_numpy()
        X = np.column_stack([np.ones(N), df.x, df.d1])
        Z = np.column_stack([np.ones(N), df.x, df[Z_COLS]])
        Xh = Z @ np.linalg.solve((Z * w[:, None]).T @ Z, (Z * w[:, None]).T @ X)
        beta = np.linalg.solve((Xh * w[:, None]).T @ X, (Xh * w[:, None]).T @ df.y)
        res = _iv(FORMULA, df, weights="w")
        np.testing.assert_allclose(
            res.params[["Intercept", "x", "d1"]].to_numpy(), beta, rtol=1e-8
        )

    def test_weight_two_equals_duplicated_rows(self):
        df = _data()
        df = df.assign(w2=np.where(df.g < 8, 2.0, 1.0))
        stacked = pd.concat([df, df[df.g < 8]], ignore_index=True)
        for method in ("2sls", "liml"):
            np.testing.assert_allclose(
                _iv(FORMULA, df, weights="w2", method=method).params.to_numpy(),
                _iv(FORMULA, stacked, method=method).params.to_numpy(),
                rtol=1e-8,
            )

    def test_reported_kp_f_follows_the_cluster_option(self):
        df = _data()
        res = _iv(FORMULA, df, cluster="g")
        direct = kleibergen_paap_rk(
            df[["d1"]], df[Z_COLS], exog=["x"], data=df,
            cov_type="cluster", cluster=df.g.to_numpy(),
        )  # fmt: skip
        np.testing.assert_allclose(
            res.diagnostics["KP rk Wald F"], direct.rk_f, rtol=RTOL
        )

    @pytest.mark.parametrize(
        "kw,exc",
        [
            ({"bogus": 1}, TypeError),
            ({"robust": "zzz"}, MethodIncompatibility),
            ({"vce": "zzz"}, MethodIncompatibility),
            ({"robust": "robust", "vce": "hc3"}, MethodIncompatibility),
            ({"method": "liml", "vce": "cr2", "cluster": "g"}, MethodIncompatibility),
            ({"method": "fuller", "small": False}, MethodIncompatibility),
            ({"method": "jive", "robust": "hc2"}, MethodIncompatibility),
            ({"method": "nonsense"}, ValueError),
        ],
    )
    def test_unsupported_requests_are_refused(self, kw, exc):
        with pytest.raises(exc):
            sp.iv(FORMULA, _data(), **kw)

    @pytest.mark.parametrize(
        "method", ["jive1", "ujive", "ijive", "rjive", "mte", "kernel", "npiv"]
    )
    @pytest.mark.parametrize("option", [{"cluster": "g"}, {"weights": "w"}])
    def test_options_a_method_does_not_have_are_refused_not_dropped(
        self, method, option
    ):
        with pytest.raises(TypeError, match="unexpected keyword"):
            sp.iv(FORMULA, _data(), method=method, **option)

    def test_method_needs_formula_and_data(self):
        with pytest.raises(ValueError, match="requires `formula` and `data`"):
            sp.iv(method="liml")

    def test_ivqreg_through_the_formula_interface(self):
        df = _data()
        res = _iv("y ~ (d1 ~ z0) + x", df, method="ivqreg", n_grid=11)
        direct = sp.iv(
            data=df, method="ivqreg", y="y", endog="d1", instruments="z0",
            exog=["x"], n_grid=11,
        )  # fmt: skip
        np.testing.assert_allclose(res.params["d1"], direct.params["d1"], rtol=1e-8)


# --------------------------------------------------------------------- iv_diag


class TestIVDiag:
    def test_point_estimate_and_hc1_se_by_hand(self):
        df = _data()
        beta, Dh, e, dd = _tsls(df)
        res = _diag(df)
        np.testing.assert_allclose(res.beta_2sls, beta, rtol=RTOL)
        se = np.sqrt(((Dh * e) @ (Dh * e)) / dd**2 * N / (N - 3))
        np.testing.assert_allclose(res.se_2sls, se, rtol=RTOL)
        z = stats.norm.ppf(0.975)
        np.testing.assert_allclose(
            res.ci_analytic_2sls, (beta - z * se, beta + z * se), rtol=RTOL
        )

    def test_cluster_column_se_by_hand(self):
        df = _data()
        _, Dh, e, dd = _tsls(df)
        G = df.g.nunique()
        se = np.sqrt(
            _cluster_ss(Dh * e, df.g) / dd**2 * G / (G - 1) * (N - 1) / (N - 3)
        )
        np.testing.assert_allclose(_diag(df, cluster="g").se_2sls, se, rtol=RTOL)

    def test_first_stage_f_by_hand(self):
        df = _data()
        D, Z = _partial(df, ["d1"])[:, 0], _partial(df, Z_COLS)
        fit = Z @ np.linalg.lstsq(Z, D, rcond=None)[0]
        f_hand = (fit @ fit / 3) / ((D - fit) @ (D - fit) / (N - 5))
        np.testing.assert_allclose(_diag(df).first_stage_F, f_hand, rtol=RTOL)

    def test_missing_cluster_column_rows_are_dropped(self):
        df = _data()
        g = df["g"].astype(float).where(df.index > 9)
        a = _diag(df.assign(g=g), cluster="g")
        b = _diag(df.iloc[10:], cluster="g")
        assert a.n == b.n == N - 10
        np.testing.assert_allclose(a.se_2sls, b.se_2sls, rtol=RTOL)
        np.testing.assert_allclose(a.kp_rk_f, b.kp_rk_f, rtol=RTOL)

    def test_vcov_strings(self):
        df = _data()
        for bad in ("hc1", "HC3", "zzz"):
            with pytest.raises(MethodIncompatibility, match="vcov must be"):
                _diag(df, vcov=bad)
        # HC0 and HC1 differ by the factor n / (n - K), K = 3
        ratio = (_diag(df, vcov="HC1").se_2sls / _diag(df, vcov="HC0").se_2sls) ** 2
        np.testing.assert_allclose(ratio, N / (N - 3), rtol=RTOL)

    def test_irrelevant_instrument_gives_an_unbounded_ar_set(self):
        df = _data()
        d = df.assign(zz=np.random.default_rng(1).normal(size=N))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.iv.iv_diag(d, y="y", endog="d1", instruments=["zz"], n_boot=0)
        assert res.ar_ci == (-np.inf, np.inf)
        assert res.effective_F < 10

    def test_cluster_array_equals_cluster_column(self):
        df = _data()
        by_name = _diag(df, cluster="g")
        by_array = _diag(df, cluster=df.g.to_numpy())
        np.testing.assert_allclose(by_array.se_2sls, by_name.se_2sls, rtol=RTOL)
        np.testing.assert_allclose(by_array.effective_F, by_name.effective_F, rtol=RTOL)
        np.testing.assert_allclose(by_array.ar_stat, by_name.ar_stat, rtol=RTOL)

    @pytest.mark.parametrize("index", ["shuffled", "offset"])
    def test_cluster_array_is_aligned_by_position(self, index):
        df = _data()
        d = (
            df.sample(frac=1.0, random_state=0)
            if index == "shuffled"
            else df.set_index(df.index + 1000)
        )
        by_name = _diag(d, cluster="g")
        by_array = _diag(d, cluster=d.g.to_numpy())
        np.testing.assert_allclose(by_array.kp_rk_f, by_name.kp_rk_f, rtol=RTOL)
        np.testing.assert_allclose(by_array.se_2sls, by_name.se_2sls, rtol=RTOL)

    def test_clr_set_uses_the_absorbed_fixed_effects(self):
        df = _data()
        res = _diag(df, absorb="g", include_clr_ci=True)
        dummies = pd.get_dummies(df.g, dtype=float).iloc[:, 1:]
        dummies.columns = [f"g{c}" for c in dummies.columns]
        wide = pd.concat([df, dummies], axis=1)
        ref = conditional_lr_ci(
            "y", "d1", Z_COLS, exog=["x"] + list(dummies.columns), data=wide, n_grid=401
        )
        # grid inversion on two different grids, endpoints bisected
        np.testing.assert_allclose(res.clr_ci, (ref.lower, ref.upper), rtol=1e-6)

    def test_single_cluster_is_refused(self):
        with pytest.raises((StatsPAIError, ValueError)):
            _diag(_data().assign(g=1), cluster="g")

    @pytest.mark.parametrize("redundant", ["constant", "multiple"])
    def test_collinear_instruments_are_refused(self, redundant):
        df = _data()
        d = df.assign(zz=1.0 if redundant == "constant" else 2.0 * df.z0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises((StatsPAIError, ValueError)):
                sp.iv.iv_diag(d, y="y", endog="d1", instruments=["z0", "zz"], n_boot=0)

    def test_alpha_outside_the_unit_interval_is_refused(self):
        with pytest.raises((StatsPAIError, ValueError)):
            _diag(_data(), alpha=2)


# --------------------------------------------------------- plausibly exogenous


class TestPlausiblyExogenous:
    KW = dict(y="y", endog="d1", instruments=Z_COLS, exog=["x"])

    def test_no_violation_is_the_classical_2sls_interval(self):
        df = _data()
        beta, _, e, dd = _tsls(df)
        se = np.sqrt(e @ e / (N - 3) / dd)
        z = stats.norm.ppf(0.975)
        ltz = plausibly_exogenous_ltz(data=df, **self.KW)
        uci = plausibly_exogenous_uci(gamma_grid=np.zeros((1, 3)), data=df, **self.KW)
        for res in (ltz, uci):
            np.testing.assert_allclose(res.beta_hat, beta, rtol=RTOL)
            np.testing.assert_allclose(
                (res.ci_lower, res.ci_upper), (beta - z * se, beta + z * se), rtol=RTOL
            )

    def test_local_to_zero_prior_shifts_and_widens_by_the_sensitivity(self):
        """beta(gamma) = beta - a'gamma with a = Z'P_Z d / d'P_Z d, so a prior
        N(mu, omega I) moves the centre by a'mu and adds omega a'a to the
        variance."""
        df = _data()
        beta, Dh, e, dd = _tsls(df)
        a = (Dh @ _partial(df, Z_COLS)) / dd
        mu, omega = np.array([0.1, 0.0, -0.05]), 0.01
        res = plausibly_exogenous_ltz(
            gamma_mean=mu, gamma_var=omega, data=df, **self.KW
        )
        centre = beta - a @ mu
        se = np.sqrt(e @ e / (N - 3) / dd + omega * a @ a)
        np.testing.assert_allclose(res.extra["beta_ltz"], centre, rtol=RTOL)
        np.testing.assert_allclose(res.extra["se_ltz"], se, rtol=RTOL)
        z = stats.norm.ppf(0.975)
        np.testing.assert_allclose(
            (res.ci_lower, res.ci_upper), (centre - z * se, centre + z * se), rtol=RTOL
        )

    def test_union_of_intervals_is_the_hull_over_the_grid(self):
        df = _data()
        grid = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, -0.05], [-0.1, 0.05, 0.0]])
        res = plausibly_exogenous_uci(gamma_grid=grid, data=df, **self.KW)
        z = stats.norm.ppf(0.975)
        np.testing.assert_allclose(
            res.ci_lower, np.min(res.beta_at_gamma - z * res.se_at_gamma), rtol=RTOL
        )
        np.testing.assert_allclose(
            res.ci_upper, np.max(res.beta_at_gamma + z * res.se_at_gamma), rtol=RTOL
        )
        beta, Dh, _, dd = _tsls(df)
        a = (Dh @ _partial(df, Z_COLS)) / dd
        np.testing.assert_allclose(res.beta_at_gamma, beta - grid @ a, rtol=RTOL)

    def test_wrong_length_prior_mean_is_refused(self):
        with pytest.raises(ValueError, match="n_instruments"):
            plausibly_exogenous_ltz(
                gamma_mean=np.array([0.1, 0.2]), data=_data(), **self.KW
            )

    def test_negative_prior_variance_is_refused(self):
        with pytest.raises((StatsPAIError, ValueError)):
            plausibly_exogenous_ltz(gamma_var=-1.0, data=_data(), **self.KW)

    def test_level_outside_the_unit_interval_is_refused(self):
        with pytest.raises((StatsPAIError, ValueError)):
            plausibly_exogenous_ltz(ci_level=95, data=_data(), **self.KW)


# ------------------------------------------------------------------------- MTE


def _binary(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z0, z1, x = rng.normal(size=(3, N))
    d = (0.8 * z0 + 0.5 * z1 + rng.normal(size=N) > 0).astype(int)
    y = d * (1 + 0.5 * x) + rng.normal(size=N)
    return pd.DataFrame({"y": y, "d": d, "z0": z0, "z1": z1, "x": x})


class TestMTE:
    def test_constant_mtr_without_covariates_is_the_difference_in_means(self):
        """Degree 0 and no covariates: each marginal treatment response is a
        constant, so ATE = ATT = ATU = the raw difference in means (on the
        trimmed sample, which here is everything: trim=0)."""
        df = _binary()
        res = mte(
            y="y", treatment="d", instruments=["z0", "z1"], data=df,
            poly_degree=0, trim=0.0,
        )  # fmt: skip
        diff = df.y[df.d == 1].mean() - df.y[df.d == 0].mean()
        for value in (res.ate, res.att, res.atu):
            np.testing.assert_allclose(value, diff, rtol=1e-8)

    def test_bootstrap_is_reproducible_and_leaves_the_point_estimate(self):
        df = _binary()
        kw = dict(y="y", treatment="d", instruments=["z0", "z1"], exog=["x"], data=df)
        plain = mte(**kw)
        a = mte(bootstrap=20, random_state=1, **kw)
        b = mte(bootstrap=20, random_state=1, **kw)
        assert a.ate == plain.ate and a.att == plain.att
        assert a.ate_se == b.ate_se

    def test_non_binary_treatment_is_refused(self):
        df = _binary()
        with pytest.raises(ValueError, match="binary"):
            mte(y="y", treatment="x", instruments=["z0"], data=df)

    def test_unknown_propensity_model_is_refused(self):
        df = _binary()
        with pytest.raises((StatsPAIError, ValueError)):
            mte(
                y="y", treatment="d", instruments=["z0", "z1"], data=df,
                propensity_model="zzz",
            )  # fmt: skip


class TestIVDiagClusterArrayRows:
    def test_missing_cluster_id_in_an_array_drops_the_row_like_a_column(self):
        df = _data()
        g = df.g.astype(float).where(df.index >= 10)
        by_name = _diag(df.assign(g=g), cluster="g")
        by_array = _diag(df, cluster=g.to_numpy())
        for field in ("se_2sls", "effective_F", "ar_stat", "kp_rk_f", "kp_rk_lm"):
            np.testing.assert_allclose(
                getattr(by_array, field), getattr(by_name, field), rtol=RTOL
            )

    def test_cluster_array_under_absorb(self):
        df = _data().assign(f=lambda d: d.index % 10)
        by_name = _diag(df, cluster="g", absorb="f")
        by_array = _diag(df, cluster=df.g.to_numpy(), absorb="f")
        for field in ("se_2sls", "effective_F", "ar_stat", "kp_rk_f"):
            np.testing.assert_allclose(
                getattr(by_array, field), getattr(by_name, field), rtol=RTOL
            )


class TestMissingValuesAreRefused:
    """One NaN used to turn every statistic into NaN, reported as an
    all-NaN result."""

    def _holed(self):
        df = _data()
        return df.assign(y=df.y.where(df.index > 3))

    def test_plausibly_exogenous(self):
        kw = dict(y="y", endog="d1", instruments=Z_COLS, exog=["x"], data=self._holed())
        with pytest.raises(MethodIncompatibility, match="missing"):
            plausibly_exogenous_ltz(**kw)
        with pytest.raises(MethodIncompatibility, match="missing"):
            plausibly_exogenous_uci(gamma_grid=np.zeros((1, 3)), **kw)

    def test_mte(self):
        df = _binary()
        d = df.assign(y=df.y.where(df.index > 3))
        with pytest.raises(MethodIncompatibility, match="missing"):
            mte(y="y", treatment="d", instruments=["z0", "z1"], data=d)

    def test_post_lasso(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(MethodIncompatibility, match="missing"):
                sp.iv.bch_post_lasso_iv(
                    "y", "d1", Z_COLS, exog=["x"], data=self._holed()
                )

    def test_nan_beta0(self):
        from statspai.iv.weak_identification import conditional_lr_test

        with pytest.raises(MethodIncompatibility, match="beta0"):
            conditional_lr_test("y", "d1", Z_COLS, data=_data(), beta0=np.nan)


class TestScanThreeReviewFollowUps:
    def test_alpha_hint_names_a_valid_alpha(self):
        with pytest.raises(MethodIncompatibility) as err:
            _diag(_data(), alpha=95)
        assert "alpha=0.05" in err.value.recovery_hint
        assert "alpha=0.95" not in err.value.recovery_hint

    def test_level_hint_names_a_valid_level(self):
        df = _data()
        with pytest.raises(MethodIncompatibility) as err:
            conditional_lr_ci(y="y", endog="d1", instruments=Z_COLS, data=df, level=95)
        assert "level=0.95" in err.value.recovery_hint

    @pytest.mark.parametrize("cluster", [np.arange(7), 3.5])
    def test_bad_cluster_array_is_a_typed_refusal(self, cluster):
        df = _data()
        with pytest.raises(MethodIncompatibility, match="cluster"):
            sp.iv.iv_diag(
                df, y="y", endog="d1", instruments=Z_COLS, exog=["x"],
                cluster=cluster, n_boot=0,
            )  # fmt: skip

    def test_uncomputed_rank_diagnostics_are_announced(self):
        """The augmenting step keeps the fit when the rank test refuses its
        input (here: one cluster), and says which keys are absent."""
        from statspai.exceptions import StatsPAIWarning
        from statspai.iv import IVRegression, _attach_augmented_diagnostics

        df = _data().assign(one=1)
        model = IVRegression(formula=FORMULA, data=df, method="2sls")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(robust="nonrobust", cluster="g")
        before = result.params.copy()
        result.model_info["cluster"] = "one"
        for key in [k for k in result.diagnostics if k.startswith("KP rk")]:
            del result.diagnostics[key]
        with pytest.warns(StatsPAIWarning, match="KP rk LM.*at least two clusters"):
            _attach_augmented_diagnostics(model, result)
        assert "augmented_diagnostics_error" in result.diagnostics
        assert "KP rk LM" not in result.diagnostics
        np.testing.assert_array_equal(result.params.to_numpy(), before.to_numpy())

    @pytest.mark.parametrize("cluster", [None, "g"])
    def test_ordinary_fit_has_its_rank_diagnostics_and_no_such_warning(self, cluster):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = sp.iv.fit(FORMULA, _data(), cluster=cluster)
        assert not any("could not be computed" in str(w.message) for w in caught)
        assert "augmented_diagnostics_error" not in res.diagnostics
        assert np.isfinite(res.diagnostics["KP rk Wald F"])

    def test_weakrobust_announces_a_refused_rank_test(self):
        from statspai.exceptions import StatsPAIWarning

        df = _data()
        df["z3"] = 2 * df["z0"]
        with pytest.warns(StatsPAIWarning, match="Kleibergen-Paap.*rank deficient"):
            sp.weakrobust(
                data=df, y="y", endog="d1", instruments=["z0", "z3"],
                include_clr=False, include_k=False,
            )  # fmt: skip
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sp.weakrobust(
                data=df, y="y", endog="d1", instruments=Z_COLS,
                include_clr=False, include_k=False,
            )  # fmt: skip
        assert not any("could not be computed" in str(w.message) for w in caught)
