"""Meta-analysis and prediction-powered inference: identities and edge cases."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.meta_analysis import combine_effects  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.exceptions import (  # noqa: E402
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)

EFFECTS = np.array([0.10, 0.25, -0.05, 0.30, 0.15, 0.42])
SES = np.array([0.05, 0.10, 0.08, 0.12, 0.06, 0.20])


class TestMetaAnalysis:
    def test_fixed_effect_is_the_inverse_variance_mean(self):
        res = sp.meta_analysis(EFFECTS, SES, method="fixed")
        w = 1 / SES**2
        assert res.estimate == pytest.approx(np.sum(w * EFFECTS) / w.sum(), rel=1e-12)
        assert res.se == pytest.approx(np.sqrt(1 / w.sum()), rel=1e-12)
        np.testing.assert_allclose(res.weights, w / w.sum(), rtol=1e-12)
        z = stats.norm.ppf(0.975)
        assert res.ci[1] == pytest.approx(res.estimate + z * res.se, rel=1e-12)
        assert res.p_value == pytest.approx(
            2 * stats.norm.sf(abs(res.estimate / res.se)), rel=1e-10
        )

    def test_dersimonian_laird_against_statsmodels(self):
        res = sp.meta_analysis(EFFECTS, SES)
        ref = combine_effects(EFFECTS, SES**2, method_re="dl")
        frame = ref.summary_frame()
        assert res.tau2 == pytest.approx(ref.tau2, rel=1e-12)
        assert res.q == pytest.approx(ref.q, rel=1e-12)
        assert res.i2 == pytest.approx(ref.i2, rel=1e-12)
        assert res.random_estimate == pytest.approx(
            frame.loc["random effect", "eff"], rel=1e-12
        )
        assert res.random_se == pytest.approx(
            frame.loc["random effect", "sd_eff"], rel=1e-12
        )
        assert res.fixed_estimate == pytest.approx(
            frame.loc["fixed effect", "eff"], rel=1e-12
        )
        k = len(EFFECTS)
        assert res.q_pvalue == pytest.approx(stats.chi2.sf(res.q, k - 1), rel=1e-10)
        assert res.h2 == pytest.approx(res.q / (k - 1), rel=1e-12)
        # prediction interval: t with k - 2 df on sqrt(se^2 + tau^2)
        half = stats.t.ppf(0.975, k - 2) * np.sqrt(res.random_se**2 + res.tau2)
        assert res.prediction_interval == pytest.approx(
            (res.random_estimate - half, res.random_estimate + half), rel=1e-12
        )

    def test_homogeneous_studies_make_the_two_models_coincide(self):
        # Q below its degrees of freedom truncates tau^2 at zero.
        res = sp.meta_analysis([0.20, 0.21, 0.19, 0.20], [0.1, 0.2, 0.15, 0.3])
        assert res.tau2 == 0.0
        assert res.i2 == 0.0
        assert res.random_estimate == pytest.approx(res.fixed_estimate, rel=1e-14)
        assert res.random_se == pytest.approx(res.fixed_se, rel=1e-14)

    def test_scale_and_shift_equivariance(self):
        a = sp.meta_analysis(EFFECTS, SES)
        b = sp.meta_analysis(3 * EFFECTS + 1, 3 * SES)
        assert b.estimate == pytest.approx(3 * a.estimate + 1, rel=1e-12)
        assert b.se == pytest.approx(3 * a.se, rel=1e-12)
        assert b.tau2 == pytest.approx(9 * a.tau2, rel=1e-10)
        assert b.q == pytest.approx(a.q, rel=1e-10)
        assert b.i2 == pytest.approx(a.i2, rel=1e-10)

    def test_egger_is_the_regression_of_z_on_precision(self):
        res = sp.meta_analysis(EFFECTS, SES)
        fit = sm.OLS(EFFECTS / SES, sm.add_constant(1 / SES)).fit()
        egger = res.egger_test()
        assert egger["intercept"] == pytest.approx(fit.params[0], rel=1e-9)
        assert egger["se"] == pytest.approx(fit.bse[0], rel=1e-9)
        assert egger["p_value"] == pytest.approx(fit.pvalues[0], rel=1e-8)
        assert egger["df"] == len(EFFECTS) - 2

    def test_two_studies_have_no_prediction_interval_or_egger_test(self):
        res = sp.meta_analysis([0.2, 0.5], [0.1, 0.1])
        assert res.prediction_interval is None
        assert np.isnan(res.egger_test()["p_value"])
        assert res.estimate == pytest.approx(0.35)
        assert "Egger" not in res.summary()

    def test_plots_carry_the_study_data(self):
        res = sp.meta_analysis(EFFECTS, SES, labels=list("abcdef"))
        ax = res.forest_plot()
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels == list("abcdef") + ["Pooled"]
        ax2 = res.funnel_plot()
        np.testing.assert_allclose(
            ax2.collections[0].get_offsets(), np.column_stack([EFFECTS, SES])
        )
        plt.close("all")
        assert "k=6" in repr(res)

    @pytest.mark.parametrize(
        "effects, ses, kwargs",
        [
            ([0.1], [0.1], {}),
            ([0.1, 0.2], [0.1, 0.0], {}),
            ([0.1, 0.2], [0.1, -0.1], {}),
            ([0.1, 0.2], [0.1], {}),
            ([[0.1, 0.2]], [[0.1, 0.1]], {}),
            ([0.1, 0.2], [0.1, 0.1], {"method": "REML"}),
        ],
    )
    def test_invalid_inputs_are_refused(self, effects, ses, kwargs):
        with pytest.raises(ValueError):
            sp.meta_analysis(effects, ses, **kwargs)

    @pytest.mark.parametrize(
        "effects, ses",
        [
            ([0.1, np.nan, 0.3], [0.1, 0.1, 0.1]),
            ([0.1, 0.2, 0.3], [0.1, np.nan, 0.1]),
            ([0.1, 0.2, 0.3], [0.1, np.inf, 0.1]),
        ],
    )
    def test_non_finite_inputs_are_refused(self, effects, ses):
        with pytest.raises(ValueError):
            sp.meta_analysis(effects, ses)

    def test_summary_works_when_all_standard_errors_are_equal(self):
        res = sp.meta_analysis([0.1, 0.2, 0.3], [0.1, 0.1, 0.1])
        assert "Fixed-effect pooled" in res.summary()
        assert np.isnan(res.egger_test()["p_value"])

    def test_labels_of_the_wrong_length_are_refused(self):
        with pytest.raises(ValueError):
            sp.meta_analysis(EFFECTS, SES, labels=["a", "b"])

    def test_alpha_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ValueError):
            sp.meta_analysis(EFFECTS, SES, alpha=1.2)


# --------------------------------------------------------------------- #
#  prediction-powered inference
# --------------------------------------------------------------------- #


def _ppi_data():
    rng = np.random.default_rng(3)
    n, N = 50, 400
    y = rng.normal(1, 1, n)
    f = y + rng.normal(0, 0.3, n)
    fu = rng.normal(1, 1.04, N)
    return y, f, fu


class TestPpiMean:
    def test_estimate_and_variance_are_the_rectified_formula(self):
        y, f, fu = _ppi_data()
        n, N = len(y), len(fu)
        res = sp.ppi_mean(y=y, yhat=f, yhat_unlabeled=fu)
        lam = np.cov(y, f, ddof=1)[0, 1] / (np.var(f, ddof=1) * (1 + n / N))
        assert res.model_info["lambda"] == pytest.approx(lam, rel=1e-12)
        est = lam * fu.mean() + np.mean(y - lam * f)
        var = lam**2 * fu.var(ddof=1) / N + np.var(y - lam * f, ddof=1) / n
        assert res.estimate == pytest.approx(est, rel=1e-12)
        assert res.se == pytest.approx(np.sqrt(var), rel=1e-12)
        plain = sp.ppi_mean(y=y, yhat=f, yhat_unlabeled=fu, tune=False)
        assert plain.model_info["lambda"] == 1.0
        assert plain.estimate == pytest.approx(fu.mean() + np.mean(y - f), rel=1e-12)
        assert res.n_obs == n + N

    def test_perfect_predictor_gives_the_pooled_mean(self):
        # With yhat = y the tuned weight is N / (n + N), which turns the
        # estimate into the mean of all n + N values.
        y, _, fu = _ppi_data()
        n, N = len(y), len(fu)
        res = sp.ppi_mean(y=y, yhat=y, yhat_unlabeled=fu)
        assert res.model_info["lambda"] == pytest.approx(N / (n + N), rel=1e-12)
        assert res.estimate == pytest.approx((y.sum() + fu.sum()) / (n + N), rel=1e-12)

    @pytest.mark.parametrize("tune", [True, False])
    def test_constant_predictor_gives_the_classical_answer(self, tune):
        y, _, _ = _ppi_data()
        res = sp.ppi_mean(
            y=y, yhat=np.full(50, 0.1), yhat_unlabeled=np.full(400, 0.1), tune=tune
        )
        assert res.estimate == pytest.approx(y.mean(), rel=1e-12)
        assert res.se == pytest.approx(y.std(ddof=1) / np.sqrt(50), rel=1e-10)
        assert res.model_info["classical_se"] == pytest.approx(res.se, rel=1e-10)

    def test_harmful_predictor_is_switched_off_by_tuning(self):
        y, f, fu = _ppi_data()
        res = sp.ppi_mean(y=y, yhat=-f, yhat_unlabeled=-fu)
        # negative covariance: the weight is clipped at zero
        assert res.model_info["lambda"] == 0.0
        assert res.estimate == pytest.approx(y.mean(), rel=1e-12)
        assert res.model_info["var_ratio_vs_classical"] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "kwargs, exc",
        [
            (dict(y=[1.0, np.nan, 2, 3], yhat=[1.0, 2, 3, 4]), MethodIncompatibility),
            (dict(y=[1.0, 2, 3], yhat=[1.0, 2, 3]), DataInsufficient),
            (dict(y=[1.0, 2, 3, 4], yhat=[1.0, 2, 3]), MethodIncompatibility),
            (dict(y=[], yhat=[]), DataInsufficient),
            (dict(y=[1.0, 2, 3, 4], yhat=[1.0, 2, 3, 4], N=3), DataInsufficient),
            (dict(y=[2.0] * 6, yhat=[2.0] * 6), NumericalInstability),
        ],
    )
    def test_invalid_inputs_are_refused(self, kwargs, exc):
        n_unlabeled = kwargs.pop("N", 8)
        with pytest.raises(exc):
            sp.ppi_mean(yhat_unlabeled=np.full(n_unlabeled, 2.0), **kwargs)

    def test_alpha_outside_the_unit_interval_is_refused(self):
        y, f, fu = _ppi_data()
        with pytest.raises(ValueError):
            sp.ppi_mean(y=y, yhat=f, yhat_unlabeled=fu, alpha=1.5)


class TestPpiOls:
    def _data(self):
        rng = np.random.default_rng(7)
        n, N = 60, 500
        X = rng.normal(size=(n, 2))
        XU = rng.normal(size=(N, 2))
        b = np.array([1.0, -0.5])
        y = X @ b + rng.normal(size=n)
        f = y + rng.normal(0, 0.3, n)
        fu = XU @ b + rng.normal(size=N)
        return X, y, f, XU, fu

    def test_untuned_estimate_is_imputed_plus_rectifier(self):
        X, y, f, XU, fu = self._data()
        res = sp.ppi_ols(
            X=X, y=y, yhat=f, X_unlabeled=XU, yhat_unlabeled=fu, tune=False
        )
        b_unl = sm.OLS(fu, sm.add_constant(XU)).fit().params
        b_y = sm.OLS(y, sm.add_constant(X)).fit()
        b_f = sm.OLS(f, sm.add_constant(X)).fit().params
        np.testing.assert_allclose(
            res.detail["estimate"], b_unl + b_y.params - b_f, rtol=1e-10
        )
        np.testing.assert_allclose(res.detail["classical_estimate"], b_y.params)
        # the classical column is the HC0 sandwich up to the ddof of np.var
        hc0 = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC0").bse
        np.testing.assert_allclose(
            res.detail["classical_se"], hc0 * np.sqrt(60 / 59), rtol=1e-10
        )
        assert res.estimand == "coef[x0]"
        assert res.estimate == pytest.approx(res.detail["estimate"].iloc[1])

    def test_perfect_predictions_with_tuning_pool_the_two_samples_weights(self):
        # With yhat = y on the labelled rows the rectifier is (1 - lambda)
        # times the labelled OLS, so each coefficient is the lambda-weighted
        # mix of the unlabelled and the labelled fit, lambda = N / (n + N).
        X, y, _, XU, fu = self._data()
        res = sp.ppi_ols(X=X, y=y, yhat=y, X_unlabeled=XU, yhat_unlabeled=fu)
        lam = 500 / 560
        np.testing.assert_allclose(res.detail["lambda"], lam, rtol=1e-12)
        b_unl = sm.OLS(fu, sm.add_constant(XU)).fit().params
        b_lab = sm.OLS(y, sm.add_constant(X)).fit().params
        np.testing.assert_allclose(
            res.detail["estimate"], lam * b_unl + (1 - lam) * b_lab, rtol=1e-10
        )

    def test_target_selection_and_intercept(self):
        X, y, f, XU, fu = self._data()
        kw = dict(y=y, yhat=f, yhat_unlabeled=fu)
        frame = pd.DataFrame(X, columns=["a", "b"])
        frame_u = pd.DataFrame(XU, columns=["a", "b"])
        by_name = sp.ppi_ols(X=frame, X_unlabeled=frame_u, target="b", **kw)
        by_pos = sp.ppi_ols(X=X, X_unlabeled=XU, target=2, **kw)
        assert by_name.estimand == "coef[b]"
        assert by_name.estimate == pytest.approx(by_pos.estimate, rel=1e-12)
        bare = sp.ppi_ols(X=X, X_unlabeled=XU, add_intercept=False, **kw)
        assert list(bare.detail["term"]) == ["x0", "x1"]
        assert bare.estimand == "coef[x0]"

    def test_invalid_inputs_are_refused(self):
        X, y, f, XU, fu = self._data()
        kw = dict(y=y, yhat=f, yhat_unlabeled=fu)
        with pytest.raises(MethodIncompatibility, match="same columns"):
            sp.ppi_ols(
                X=pd.DataFrame(X, columns=["a", "b"]),
                X_unlabeled=pd.DataFrame(XU, columns=["b", "a"]),
                **kw,
            )
        with pytest.raises(MethodIncompatibility, match="out of range"):
            sp.ppi_ols(X=X, X_unlabeled=XU, target=3, **kw)
        with pytest.raises(MethodIncompatibility, match="not among terms"):
            sp.ppi_ols(X=X, X_unlabeled=XU, target="zz", **kw)
        with pytest.raises(NumericalInstability, match="singular"):
            sp.ppi_ols(
                X=np.column_stack([X, 2 * X[:, 0]]),
                X_unlabeled=np.column_stack([XU, 2 * XU[:, 0]]),
                **kw,
            )
        with pytest.raises(MethodIncompatibility, match="rows"):
            sp.ppi_ols(X=X, X_unlabeled=XU[:-1], **kw)
        with pytest.raises(DataInsufficient):
            sp.ppi_ols(X=X[:4], y=y[:4], yhat=f[:4], X_unlabeled=XU, yhat_unlabeled=fu)
        bad = X.copy()
        bad[0, 0] = np.nan
        with pytest.raises(MethodIncompatibility, match="non-finite"):
            sp.ppi_ols(X=bad, X_unlabeled=XU, **kw)

    def test_numpy_integer_target_is_a_position(self):
        X, y, f, XU, fu = self._data()
        res = sp.ppi_ols(
            X=X, y=y, yhat=f, X_unlabeled=XU, yhat_unlabeled=fu, target=np.int64(2)
        )
        assert res.estimand == "coef[x1]"
