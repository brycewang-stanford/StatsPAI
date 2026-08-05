"""Tests for prediction-powered inference (``sp.ppi_mean`` / ``sp.ppi_ols``)."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def _mean_data(seed=7, n=100, N=5000, theta=2.0, pred_noise=0.5):
    rng = np.random.default_rng(seed)
    y_all = theta + rng.normal(0, 1, n + N)
    f_all = y_all + rng.normal(0, pred_noise, n + N)
    return y_all[:n], f_all[:n], f_all[n:]


class TestPPIMeanCorrectness:
    def test_matches_manual_formula(self):
        y, f, fu = _mean_data()
        r = sp.ppi_mean(y=y, yhat=f, yhat_unlabeled=fu, tune=False)
        # Classic PPI (lambda = 1): mean(fu) + mean(y - f)
        manual = fu.mean() + (y - f).mean()
        assert r.estimate == pytest.approx(manual, rel=1e-12)
        n, N = len(y), len(fu)
        manual_se = np.sqrt(np.var(fu, ddof=1) / N + np.var(y - f, ddof=1) / n)
        assert r.se == pytest.approx(manual_se, rel=1e-12)

    def test_beats_classical_with_good_predictions(self):
        y, f, fu = _mean_data()
        r = sp.ppi_mean(y=y, yhat=f, yhat_unlabeled=fu)
        assert r.se < r.model_info["classical_se"]
        assert r.ci[0] < 2.0 < r.ci[1]

    def test_junk_predictions_recover_classical(self):
        """PPI++ safety: with uninformative predictions, lambda -> 0 and
        the interval matches labeled-only inference."""
        rng = np.random.default_rng(11)
        n, N = 100, 5000
        y = 2.0 + rng.normal(0, 1, n)
        f = rng.normal(0, 1, n)
        fu = rng.normal(0, 1, N)
        r = sp.ppi_mean(y=y, yhat=f, yhat_unlabeled=fu)
        assert r.model_info["lambda"] < 0.2
        assert r.se == pytest.approx(r.model_info["classical_se"], rel=0.05)

    def test_coverage_simulation(self):
        """95% CI should cover the truth in the vast majority of draws."""
        cover = 0
        n_sim = 200
        for s in range(n_sim):
            y, f, fu = _mean_data(seed=s, n=80, N=2000)
            r = sp.ppi_mean(y=y, yhat=f, yhat_unlabeled=fu)
            if r.ci[0] < 2.0 < r.ci[1]:
                cover += 1
        assert cover / n_sim > 0.90


class TestPPIMeanBoundaries:
    def test_length_mismatch_raises(self):
        with pytest.raises(MethodIncompatibility, match="paired rows"):
            sp.ppi_mean(y=[1.0, 2.0], yhat=[1.0], yhat_unlabeled=[1.0] * 10)

    def test_nan_raises(self):
        with pytest.raises(MethodIncompatibility, match="non-finite"):
            sp.ppi_mean(
                y=[1.0, np.nan, 2.0, 3.0],
                yhat=[1.0, 2.0, 2.0, 3.0],
                yhat_unlabeled=[1.0] * 10,
            )

    def test_too_few_rows_raise(self):
        with pytest.raises(DataInsufficient):
            sp.ppi_mean(y=[1.0, 2.0], yhat=[1.0, 2.0], yhat_unlabeled=[1.0] * 10)
        with pytest.raises(DataInsufficient):
            sp.ppi_mean(
                y=[1.0, 2.0, 3.0, 4.0],
                yhat=[1.0, 2.0, 3.0, 4.0],
                yhat_unlabeled=[1.0, 2.0],
            )


def _ols_data(seed=3, n=150, N=4000, beta=2.0, pred_noise=0.5):
    rng = np.random.default_rng(seed)
    x_all = rng.normal(size=n + N)
    y_all = 1.0 + beta * x_all + rng.normal(0, 1, n + N)
    f_all = y_all + rng.normal(0, pred_noise, n + N)
    return (
        pd.DataFrame({"x": x_all[:n]}),
        y_all[:n],
        f_all[:n],
        pd.DataFrame({"x": x_all[n:]}),
        f_all[n:],
    )


class TestPPIOLS:
    def test_recovers_slope_and_beats_classical(self):
        X, y, f, Xu, fu = _ols_data()
        r = sp.ppi_ols(X=X, y=y, yhat=f, X_unlabeled=Xu, yhat_unlabeled=fu)
        assert r.estimand == "coef[x]"
        assert r.ci[0] < 2.0 < r.ci[1]
        assert r.se < r.model_info["classical_se"]
        assert list(r.detail["term"]) == ["const", "x"]

    def test_lambda_zero_equals_classical(self):
        X, y, f, Xu, fu = _ols_data()
        rng = np.random.default_rng(0)
        junk_f = rng.normal(size=len(y))
        junk_fu = rng.normal(size=len(fu))
        r = sp.ppi_ols(X=X, y=y, yhat=junk_f, X_unlabeled=Xu, yhat_unlabeled=junk_fu)
        row = r.detail.set_index("term").loc["x"]
        assert row["lambda"] < 0.25
        assert row["estimate"] == pytest.approx(
            row["classical_estimate"], abs=4 * row["classical_se"]
        )

    def test_target_selection(self):
        X, y, f, Xu, fu = _ols_data()
        r = sp.ppi_ols(
            X=X,
            y=y,
            yhat=f,
            X_unlabeled=Xu,
            yhat_unlabeled=fu,
            target="const",
        )
        assert r.estimand == "coef[const]"
        with pytest.raises(MethodIncompatibility, match="not among terms"):
            sp.ppi_ols(
                X=X,
                y=y,
                yhat=f,
                X_unlabeled=Xu,
                yhat_unlabeled=fu,
                target="zzz",
            )

    def test_column_mismatch_raises(self):
        X, y, f, Xu, fu = _ols_data()
        Xu_bad = Xu.rename(columns={"x": "z"})
        with pytest.raises(MethodIncompatibility, match="same columns"):
            sp.ppi_ols(X=X, y=y, yhat=f, X_unlabeled=Xu_bad, yhat_unlabeled=fu)

    def test_too_few_labeled_rows_raise(self):
        X, y, f, Xu, fu = _ols_data()
        with pytest.raises(DataInsufficient):
            sp.ppi_ols(
                X=X.iloc[:3],
                y=y[:3],
                yhat=f[:3],
                X_unlabeled=Xu,
                yhat_unlabeled=fu,
            )
