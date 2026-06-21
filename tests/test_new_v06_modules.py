"""
Tests for all new v0.6 modules:
- GLM, Logit/Probit, Multinomial, Count Data, Zero-Inflated
- Survival, Nonparametric, Time Series
- Experimental Design, Imputation, Mendelian Randomization
- Multi-Cutoff RD, Continuous DID, Advanced IV
"""

import numpy as np
import pandas as pd
import pytest
import warnings

# ====================================================================
# Helper: generate test data
# ====================================================================


def _binary_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    xb = -1 + 0.5 * x1 + 0.3 * x2
    p = 1 / (1 + np.exp(-xb))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _count_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    mu = np.exp(0.5 + 0.3 * x1 + 0.2 * x2)
    y = rng.poisson(mu)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _survival_data(n=300, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.binomial(1, 0.5, n)
    hazard = np.exp(0.3 * x1 + 0.5 * x2)
    time = rng.exponential(1 / hazard)
    censor_time = rng.exponential(3, n)
    duration = np.minimum(time, censor_time)
    event = (time <= censor_time).astype(int)
    return pd.DataFrame({"duration": duration, "event": event, "x1": x1, "group": x2})


def _panel_data(n_units=100, n_periods=6, seed=42):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)
    dose = np.repeat(rng.exponential(2, n_units), n_periods)
    dose[np.repeat(rng.random(n_units) > 0.5, n_periods)] = 0  # 50% untreated
    post = (times >= 3).astype(int)
    y = 2 + 0.5 * dose * post + rng.normal(0, 1, n_units * n_periods)
    return pd.DataFrame({"id": ids, "time": times, "y": y, "dose": dose, "post": post})


# ====================================================================
# Test GLM
# ====================================================================


class TestGLM:
    def test_glm_gaussian(self):
        from statspai.regression.glm import glm

        rng = np.random.default_rng(42)
        df = pd.DataFrame({"y": rng.normal(0, 1, 200), "x1": rng.normal(0, 1, 200)})
        result = glm(data=df, y="y", x=["x1"], family="gaussian")
        assert result is not None
        assert len(result.params) >= 2

    def test_glm_binomial(self):
        from statspai.regression.glm import glm

        df = _binary_data()
        result = glm(data=df, y="y", x=["x1", "x2"], family="binomial")
        assert result is not None
        assert "Pseudo_R2" in result.diagnostics or "Deviance" in result.diagnostics

    def test_glm_poisson(self):
        from statspai.regression.glm import glm

        df = _count_data()
        result = glm(data=df, y="y", x=["x1", "x2"], family="poisson")
        assert result is not None


# ====================================================================
# Test Logit / Probit
# ====================================================================


class TestLogitProbit:
    def test_logit_basic(self):
        from statspai.regression.logit_probit import logit

        df = _binary_data()
        result = logit(data=df, y="y", x=["x1", "x2"])
        assert result is not None
        assert len(result.params) >= 3  # intercept + 2 vars
        # Coefficients should have correct sign
        assert result.params["x1"] > 0  # positive effect

    def test_probit_basic(self):
        from statspai.regression.logit_probit import probit

        df = _binary_data()
        result = probit(data=df, y="y", x=["x1", "x2"])
        assert result is not None
        assert len(result.params) >= 3

    def test_logit_summary(self):
        from statspai.regression.logit_probit import logit

        df = _binary_data()
        result = logit(data=df, y="y", x=["x1", "x2"])
        s = result.summary()
        assert "Logit" in s or "logit" in s or len(s) > 50


# ====================================================================
# Test Multinomial / Ordered
# ====================================================================


class TestMultinomial:
    def test_mlogit(self):
        from statspai.regression.multinomial import mlogit

        rng = np.random.default_rng(42)
        n = 400
        x1 = rng.normal(0, 1, n)
        # 3 categories
        p = np.column_stack([np.ones(n), np.exp(0.5 * x1), np.exp(-0.3 * x1)])
        p = p / p.sum(axis=1, keepdims=True)
        y = np.array([rng.choice(3, p=p[i]) for i in range(n)])
        df = pd.DataFrame({"y": y, "x1": x1})
        result = mlogit(data=df, y="y", x=["x1"])
        assert result is not None

    def test_ologit(self):
        from statspai.regression.multinomial import ologit

        rng = np.random.default_rng(42)
        n = 400
        x1 = rng.normal(0, 1, n)
        latent = 0.5 * x1 + rng.logistic(0, 1, n)
        y = np.digitize(latent, bins=[-1, 0, 1])
        df = pd.DataFrame({"y": y, "x1": x1})
        result = ologit(data=df, y="y", x=["x1"])
        assert result is not None


# ====================================================================
# Test Count Data
# ====================================================================


class TestCountData:
    def test_poisson(self):
        from statspai.regression.count import poisson

        df = _count_data()
        result = poisson(data=df, y="y", x=["x1", "x2"])
        assert result is not None
        assert result.params["x1"] > 0  # positive coefficient

    def test_nbreg(self):
        from statspai.regression.count import nbreg

        df = _count_data()
        result = nbreg(data=df, y="y", x=["x1", "x2"])
        assert result is not None

    def test_ppmlhdfe(self):
        from statspai.regression.count import ppmlhdfe

        df = _count_data()
        result = ppmlhdfe(data=df, y="y", x=["x1", "x2"])
        assert result is not None


# ====================================================================
# Test Zero-Inflated
# ====================================================================


class TestZeroInflated:
    def test_zip_model(self):
        from statspai.regression.zeroinflated import zip_model

        # Create data with excess zeros
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.normal(0, 1, n)
        zero_inflate = rng.random(n) < 0.3
        mu = np.exp(0.5 + 0.3 * x1)
        y = np.where(zero_inflate, 0, rng.poisson(mu))
        df = pd.DataFrame({"y": y, "x1": x1})
        result = zip_model(data=df, y="y", x=["x1"])
        assert result is not None

    def test_hurdle(self):
        from statspai.regression.zeroinflated import hurdle

        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.normal(0, 1, n)
        zero_inflate = rng.random(n) < 0.3
        mu = np.exp(0.5 + 0.3 * x1)
        y = np.where(zero_inflate, 0, rng.poisson(mu))
        df = pd.DataFrame({"y": y, "x1": x1})
        result = hurdle(data=df, y="y", x=["x1"])
        assert result is not None


# ====================================================================
# Test Survival
# ====================================================================


class TestSurvival:
    def test_kaplan_meier(self):
        from statspai.survival.models import kaplan_meier

        df = _survival_data()
        result = kaplan_meier(data=df, duration="duration", event="event")
        assert result is not None
        assert hasattr(result, "survival_table")
        s = result.summary()
        assert len(s) > 20

    def test_kaplan_meier_groups(self):
        from statspai.survival.models import kaplan_meier

        df = _survival_data()
        result = kaplan_meier(
            data=df, duration="duration", event="event", group="group"
        )
        assert result is not None

    def test_logrank(self):
        from statspai.survival.models import logrank_test

        df = _survival_data()
        result = logrank_test(
            data=df, duration="duration", event="event", group="group"
        )
        assert "test_statistic" in result or "p_value" in result

    def test_cox(self):
        from statspai.survival.models import cox

        df = _survival_data()
        result = cox(data=df, duration="duration", event="event", x=["x1"])
        assert result is not None
        assert len(result.params) >= 1

    def test_survreg(self):
        from statspai.survival.models import survreg

        df = _survival_data()
        result = survreg(data=df, duration="duration", event="event", x=["x1"])
        assert result is not None


# ====================================================================
# Test Nonparametric
# ====================================================================


class TestNonparametric:
    def test_lpoly(self):
        from statspai.nonparametric.lpoly import lpoly

        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 10, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)
        df = pd.DataFrame({"y": y, "x": x})
        result = lpoly(data=df, y="y", x="x")
        assert result is not None
        assert len(result.grid) == 100
        assert len(result.fitted) == 100
        s = result.summary()
        assert "Local Polynomial" in s

    def test_kdensity(self):
        from statspai.nonparametric.kdensity import kdensity

        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(0, 1, 500)})
        result = kdensity(data=df, x="x")
        assert result is not None
        assert len(result.density) == 512
        assert np.all(result.density >= 0)
        s = result.summary()
        assert "Kernel Density" in s


# ====================================================================
# Test Time Series
# ====================================================================


class TestTimeSeries:
    def test_var(self):
        from statspai.timeseries.var import var

        rng = np.random.default_rng(42)
        n = 200
        y1 = np.cumsum(rng.normal(0, 1, n))
        y2 = np.cumsum(rng.normal(0, 1, n))
        df = pd.DataFrame({"y1": y1, "y2": y2})
        result = var(df, variables=["y1", "y2"], lags=2)
        assert result is not None
        assert result.lags == 2
        s = result.summary()
        assert "VAR" in s

    def test_granger(self):
        from statspai.timeseries.var import var, granger_causality

        rng = np.random.default_rng(42)
        n = 200
        y1 = np.cumsum(rng.normal(0, 1, n))
        y2 = np.cumsum(rng.normal(0, 1, n))
        df = pd.DataFrame({"y1": y1, "y2": y2})
        v = var(df, variables=["y1", "y2"], lags=2)
        gc = granger_causality(v, caused="y1", causing="y2")
        assert "F_stat" in gc
        assert "p_value" in gc

    def test_structural_break(self):
        from statspai.timeseries.structural_break import structural_break

        rng = np.random.default_rng(42)
        n = 200
        y = np.concatenate([rng.normal(0, 1, 100), rng.normal(2, 1, 100)])
        df = pd.DataFrame({"y": y})
        result = structural_break(data=df, y="y", method="sup-f")
        assert result is not None
        s = result.summary()
        assert "Structural Break" in s

    def test_cusum(self):
        from statspai.timeseries.structural_break import cusum_test

        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        y = 2 * x + np.concatenate([rng.normal(0, 0.5, 100), rng.normal(0, 2, 100)])
        df = pd.DataFrame({"y": y, "x": x})
        result = cusum_test(df, y="y", x=["x"])
        X_data = np.column_stack([np.ones(n), df[["x"]].to_numpy(dtype=float)])
        y_data = df["y"].to_numpy(dtype=float)
        rec_resid = []
        for t in range(X_data.shape[1], n):
            Xt = X_data[:t]
            yt = y_data[:t]
            bt = np.linalg.lstsq(Xt, yt, rcond=None)[0]
            resid = y_data[t] - X_data[t] @ bt
            ft = 1 + X_data[t] @ np.linalg.solve(Xt.T @ Xt, X_data[t])
            rec_resid.append(resid / np.sqrt(ft))
        rec_resid = np.asarray(rec_resid)
        manual_cusum = np.cumsum(rec_resid) / (
            np.std(rec_resid, ddof=1) * np.sqrt(n - X_data.shape[1])
        )
        np.testing.assert_allclose(result["max_cusum"], np.max(np.abs(manual_cusum)))
        assert "max_cusum" in result


# ====================================================================
# Test Experimental Design
# ====================================================================


class TestExperimental:
    def test_randomize(self):
        from statspai.experimental.design import randomize

        df = pd.DataFrame(
            {
                "id": range(12),
                "age": np.linspace(20.0, 42.0, 12),
                "income": np.linspace(40000.0, 62000.0, 12),
                "district": np.repeat(["A", "B", "C"], 4),
            }
        )
        result = randomize(
            df, strata="district", balance_vars=["age", "income"], seed=42
        )
        by_stratum = result.data.groupby("district")["treatment"].sum().to_numpy()
        np.testing.assert_allclose(by_stratum, [2, 2, 2])
        np.testing.assert_allclose(result.n_treated, 6)
        np.testing.assert_allclose(result.n_control, 6)
        assert result is not None
        assert result.treatment_col == "treatment"
        assert "treatment" in result.data.columns
        s = result.summary()
        assert "Randomization" in s

    def test_balance_check(self):
        from statspai.experimental.design import balance_check

        df = pd.DataFrame(
            {
                "treated": [0, 0, 1, 1],
                "age": [20.0, 22.0, 24.0, 26.0],
                "income": [40.0, 44.0, 48.0, 52.0],
            }
        )
        result = balance_check(df, treatment="treated", covariates=["age", "income"])
        age_t = df.loc[df["treated"] == 1, "age"]
        age_c = df.loc[df["treated"] == 0, "age"]
        age_diff = age_t.mean() - age_c.mean()
        age_pooled_sd = np.sqrt((age_t.std(ddof=1) ** 2 + age_c.std(ddof=1) ** 2) / 2)
        np.testing.assert_allclose(
            result.normalized_diffs["age"], age_diff / age_pooled_sd
        )
        np.testing.assert_allclose(result.table.loc[0, "diff"], 4.0)
        assert result is not None
        s = result.summary()
        assert "Balance" in s

    def test_attrition(self):
        from statspai.experimental.attrition import attrition_test

        df = pd.DataFrame(
            {
                "treated": [0, 0, 0, 0, 1, 1, 1, 1],
                "observed": [1, 1, 0, 0, 1, 1, 1, 0],
                "age": [20.0, 22.0, 24.0, 26.0, 21.0, 23.0, 25.0, 27.0],
            }
        )
        result = attrition_test(
            df, treatment="treated", observed="observed", covariates=["age"]
        )
        np.testing.assert_allclose(result.overall_rate, 3 / 8)
        np.testing.assert_allclose(result.control_rate, 2 / 4)
        np.testing.assert_allclose(result.treat_rate, 1 / 4)
        np.testing.assert_allclose(result.n_attrit, 3)
        assert result is not None
        s = result.summary()
        assert "Attrition" in s

    def test_optimal_design(self):
        from statspai.experimental.optimal import optimal_design
        from scipy import stats

        result = optimal_design(design="individual", mde=0.2, sigma=1.0)
        z_alpha = stats.norm.ppf(1 - 0.05 / 2)
        z_beta = stats.norm.ppf(0.8)
        expected_n_per_arm = np.ceil(((z_alpha + z_beta) ** 2) / (0.2**2 * 0.5 * 0.5))
        np.testing.assert_allclose(result.n_per_arm, expected_n_per_arm)
        np.testing.assert_allclose(result.n_total, np.ceil(expected_n_per_arm / 0.5))
        assert result is not None
        assert result.n_total > 0
        s = result.summary()
        assert "Optimal" in s

        result_cl = optimal_design(
            design="cluster", mde=0.2, sigma=1.0, icc=0.05, cluster_size=20
        )
        deff = 1 + (20 - 1) * 0.05
        expected_cluster_ind_per_arm = np.ceil(
            ((z_alpha + z_beta) ** 2 * deff) / (0.2**2 * 0.5 * 0.5)
        )
        expected_clusters = np.ceil(expected_cluster_ind_per_arm / 20) * 2
        np.testing.assert_allclose(result_cl.n_clusters, expected_clusters)
        assert result_cl.n_clusters > 0


# ====================================================================
# Test Imputation (MICE)
# ====================================================================


class TestImputation:
    def test_mice_basic(self):
        from statspai.imputation.mice import mice

        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "y": rng.normal(0, 1, n),
                "x1": rng.normal(0, 1, n),
                "x2": rng.normal(0, 1, n),
            }
        )
        # Introduce missing data
        df.loc[rng.choice(n, 30, replace=False), "x1"] = np.nan
        df.loc[rng.choice(n, 20, replace=False), "x2"] = np.nan

        result = mice(df, m=3, max_iter=5, seed=42)
        assert result is not None
        assert result.n_imputations == 3
        completed = result.complete(0)
        assert completed["x1"].isna().sum() == 0
        assert completed["x2"].isna().sum() == 0
        s = result.summary()
        assert "MICE" in s


# ====================================================================
# Test Mendelian Randomization
# ====================================================================


class TestMendelianRandomization:
    def test_mr_basic(self):
        from statspai.mendelian.mr import mendelian_randomization

        rng = np.random.default_rng(42)
        n_snps = 20
        true_effect = 0.5
        beta_x = rng.normal(0.3, 0.1, n_snps)
        beta_y = true_effect * beta_x + rng.normal(0, 0.05, n_snps)
        se_x = np.abs(rng.normal(0.05, 0.01, n_snps))
        se_y = np.abs(rng.normal(0.08, 0.02, n_snps))

        df = pd.DataFrame(
            {
                "beta_x": beta_x,
                "beta_y": beta_y,
                "se_x": se_x,
                "se_y": se_y,
            }
        )

        result = mendelian_randomization(
            data=df,
            beta_exposure="beta_x",
            beta_outcome="beta_y",
            se_exposure="se_x",
            se_outcome="se_y",
            exposure_name="BMI",
            outcome_name="T2D",
        )
        assert result is not None
        assert len(result.estimates) == 3  # IVW, Egger, Weighted Median
        # IVW estimate should be close to 0.5
        ivw_est = result.estimates.loc[
            result.estimates["method"] == "IVW", "estimate"
        ].values[0]
        assert abs(ivw_est - true_effect) < 0.3
        s = result.summary()
        assert "Mendelian" in s


# ====================================================================
# Test Multi-Cutoff RD
# ====================================================================


class TestMultiCutoffRD:
    def test_rdmc(self):
        from statspai.rd.rdmulti import rdmc

        rng = np.random.default_rng(42)
        n = 1000
        x = rng.uniform(0, 100, n)
        y = 2 + 0.1 * x + 3 * (x >= 50) + 2 * (x >= 75) + rng.normal(0, 1, n)
        df = pd.DataFrame({"y": y, "x": x})
        result = rdmc(df, y="y", x="x", cutoffs=[50, 75])
        assert result is not None
        assert result.n_cutoffs == 2
        s = result.summary()
        assert "Multi-Cutoff" in s

    def test_rdms(self):
        from statspai.rd.rdmulti import rdms

        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        treat = (x1 >= 0).astype(int)
        y = 1 + 2 * treat + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 1, n)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        result = rdms(df, y="y", x1="x1", x2="x2", bandwidth=2.0)
        assert result is not None
        # Geographic RD produces an estimate (may be noisy with small sample)
        assert np.isfinite(result.estimate)


# ====================================================================
# Test Continuous DID
# ====================================================================


class TestContinuousDID:
    def test_continuous_did_twfe(self):
        from statspai.did.continuous_did import continuous_did

        df = _panel_data()
        result = continuous_did(
            df,
            y="y",
            dose="dose",
            time="time",
            id="id",
            post="post",
            method="twfe",
        )
        assert result is not None
        # True effect is 0.5
        assert abs(result.estimate - 0.5) < 0.5


# ====================================================================
# Test Advanced IV
# ====================================================================


class TestAdvancedIV:
    def test_liml(self):
        from statspai.regression.advanced_iv import liml

        rng = np.random.default_rng(42)
        n = 500
        z = rng.normal(0, 1, n)
        u = rng.normal(0, 1, n)
        x = 0.5 * z + 0.5 * u + rng.normal(0, 0.5, n)
        y = 1 + 0.8 * x + u
        df = pd.DataFrame({"y": y, "x": x, "z": z})
        result = liml(data=df, y="y", x_endog=["x"], z=["z"])
        assert result is not None
        # LIML estimate should be close to 0.8
        x_coef = result.params["x"]
        assert abs(x_coef - 0.8) < 0.5

    def test_liml_fuller(self):
        from statspai.regression.advanced_iv import liml

        rng = np.random.default_rng(42)
        n = 500
        z = rng.normal(0, 1, n)
        u = rng.normal(0, 1, n)
        x = 0.5 * z + 0.5 * u + rng.normal(0, 0.5, n)
        y = 1 + 0.8 * x + u
        df = pd.DataFrame({"y": y, "x": x, "z": z})
        result = liml(data=df, y="y", x_endog=["x"], z=["z"], fuller=1)
        assert result is not None
        assert "Fuller" in result.model_info["model_type"]

    def test_jive(self):
        from statspai.regression.advanced_iv import jive

        rng = np.random.default_rng(42)
        n = 500
        z1 = rng.normal(0, 1, n)
        z2 = rng.normal(0, 1, n)
        u = rng.normal(0, 1, n)
        x = 0.3 * z1 + 0.3 * z2 + 0.5 * u + rng.normal(0, 0.5, n)
        y = 1 + 0.8 * x + u
        df = pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2})
        result = jive(data=df, y="y", x_endog=["x"], z=["z1", "z2"])
        assert result is not None
        assert "JIVE" in result.model_info["model_type"]
