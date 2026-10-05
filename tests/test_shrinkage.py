"""``sp.shrinkage``: ridge, lasso and principal-components prediction.

The fits are compared with scikit-learn on the same standardised design
(ridge and principal components are closed forms, so the tolerance is
numerical; the lasso is an iterative solve on both sides). The
cross-validated error is compared with a loop written out here, which
standardises inside each training fold.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(20261005)
    n, k = 240, 25
    common = rng.normal(size=(n, 1))
    X = 0.6 * common + rng.normal(size=(n, k)) * rng.uniform(0.5, 3.0, size=k)
    beta = np.zeros(k)
    beta[:4] = [1.5, -2.0, 0.8, 0.5]
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(k)])
    df["y"] = 3.0 + X @ beta + rng.normal(scale=2.0, size=n)
    return df


def _standardised(df, cols):
    Z = (df[cols] - df[cols].mean()) / df[cols].std(ddof=1)
    return Z.to_numpy(), (df["y"] - df["y"].mean()).to_numpy()


def test_ridge_is_sklearn_ridge_on_the_standardised_design(frame):
    cols = [c for c in frame if c != "y"]
    Z, yc = _standardised(frame, cols)
    fit = sp.shrinkage(frame, "y", cols, method="ridge", penalty=37.0, n_folds=0)
    ref = Ridge(alpha=37.0, fit_intercept=False, solver="svd").fit(Z, yc)
    # closed form on both sides
    np.testing.assert_allclose(fit.params.to_numpy(), ref.coef_, rtol=1e-10)
    assert fit.intercept == pytest.approx(frame["y"].mean())
    assert fit.penalty == 37.0 and fit.cv is None and fit.cv_rmspe is None


def test_lasso_penalty_is_on_the_scale_of_the_sum_of_squares(frame):
    cols = [c for c in frame if c != "y"]
    Z, yc = _standardised(frame, cols)
    n = len(frame)
    alpha = 0.15  # scikit-learn's per-observation penalty
    fit = sp.shrinkage(
        frame, "y", cols, method="lasso", penalty=2 * n * alpha, n_folds=0
    )
    ref = Lasso(alpha=alpha, fit_intercept=False, tol=1e-12, max_iter=10**6)
    ref.fit(Z, yc)
    # two coordinate-descent solves of one convex problem
    np.testing.assert_allclose(fit.params.to_numpy(), ref.coef_, atol=1e-6)
    assert fit.n_nonzero == int(np.count_nonzero(ref.coef_)) < len(cols)


def test_pcr_is_ols_on_the_leading_principal_components(frame):
    cols = [c for c in frame if c != "y"]
    Z, yc = _standardised(frame, cols)
    fit = sp.shrinkage(frame, "y", cols, method="pcr", n_components=6, n_folds=0)
    pca = PCA(n_components=6, svd_solver="full").fit(Z)
    gamma = np.linalg.lstsq(pca.transform(Z), yc, rcond=None)[0]
    np.testing.assert_allclose(
        fit.params.to_numpy(), pca.components_.T @ gamma, rtol=1e-9, atol=1e-12
    )
    # every component kept: least squares
    full = sp.shrinkage(
        frame, "y", cols, method="pcr", n_components=len(cols), n_folds=0
    )
    ols = sp.shrinkage(frame, "y", cols, method="ols", n_folds=0)
    np.testing.assert_allclose(full.params, ols.params, rtol=1e-8)


def test_cross_validation_standardises_inside_each_training_fold(frame):
    cols = [c for c in frame if c != "y"]
    grid = [5.0, 50.0, 500.0]
    fit = sp.shrinkage(frame, "y", cols, method="ridge", penalty=grid, n_folds=5)
    n = len(frame)
    sse = np.zeros(len(grid))
    for held in np.array_split(np.arange(n), 5):
        train = frame.drop(index=frame.index[held])
        test = frame.iloc[held]
        mu, sd, ybar = train[cols].mean(), train[cols].std(ddof=1), train["y"].mean()
        Zt, Zh = ((train[cols] - mu) / sd).to_numpy(), (
            (test[cols] - mu) / sd
        ).to_numpy()
        for i, lam in enumerate(grid):
            ref = Ridge(alpha=lam, fit_intercept=False, solver="svd")
            ref.fit(Zt, (train["y"] - ybar).to_numpy())
            sse[i] += float(((test["y"] - ybar - Zh @ ref.coef_) ** 2).sum())
    np.testing.assert_allclose(fit.cv["mspe"].to_numpy(), sse / n, rtol=1e-10)
    assert fit.penalty == grid[int(np.argmin(sse))]
    assert fit.cv_rmspe == pytest.approx(np.sqrt(sse.min() / n))
    assert fit.selected_by_cv and fit.n_folds == 5


def test_shrinkage_beats_ols_out_of_sample_when_predictors_are_many():
    # 60 predictors, 80 observations, 3 of them matter: least squares
    # overfits and the cross-validated estimators do not.
    rng = np.random.default_rng(7)
    n, k = 280, 60
    X = rng.normal(size=(n, k))
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(k)])
    df["y"] = X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.normal(size=n)
    cols = [f"x{j}" for j in range(k)]
    train, hold = df.iloc[:80], df.iloc[80:]
    ols = sp.shrinkage(train, "y", cols, method="ols")
    for method in ("ridge", "lasso"):
        fit = sp.shrinkage(train, "y", cols, method=method)
        assert fit.rmspe(hold) < 0.75 * ols.rmspe(hold)
        # the cross-validated error is an honest guide to the hold-out error
        assert abs(fit.cv_rmspe - fit.rmspe(hold)) < 0.35
    assert ols.rmspe_in < 0.6  # and the in-sample fit of OLS is not


def test_predict_uses_original_units_and_rmspe_is_its_error(frame):
    cols = [c for c in frame if c != "y"]
    fit = sp.shrinkage(frame, "y", cols, method="ridge", penalty=10.0, n_folds=0)
    pred = fit.predict(frame)
    assert fit.rmspe(frame) == pytest.approx(fit.rmspe_in)
    assert fit.rmspe(frame) == pytest.approx(np.sqrt(np.mean((frame.y - pred) ** 2)))
    # rescaling a predictor leaves standardised predictions unchanged
    scaled = frame.assign(x0=frame.x0 * 1000.0)
    again = sp.shrinkage(scaled, "y", cols, method="ridge", penalty=10.0, n_folds=0)
    np.testing.assert_allclose(again.predict(scaled), pred, rtol=1e-9)
    assert "Ridge" in fit.summary() and fit.to_dict()["penalty"] == 10.0


def test_refusals(frame):
    cols = [c for c in frame if c != "y"]
    with pytest.raises(MethodIncompatibility, match="unknown method"):
        sp.shrinkage(frame, "y", cols, method="elastic")
    with pytest.raises(MethodIncompatibility, match="pcr"):
        sp.shrinkage(frame, "y", cols, method="ridge", n_components=3)
    with pytest.raises(MethodIncompatibility, match="penalty applies"):
        sp.shrinkage(frame, "y", cols, method="pcr", penalty=1.0)
    with pytest.raises(MethodIncompatibility, match="do not vary"):
        sp.shrinkage(frame.assign(k=1.0), "y", cols + ["k"])
    with pytest.raises(MethodIncompatibility, match="not in the data"):
        sp.shrinkage(frame, "y", cols + ["nope"])
    with pytest.raises(MethodIncompatibility, match="need cross-validation"):
        sp.shrinkage(frame, "y", cols, penalty=[1.0, 2.0], n_folds=0)
    with pytest.raises(MethodIncompatibility, match="whole numbers"):
        sp.shrinkage(frame, "y", cols, method="pcr", n_components=0)
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        sp.shrinkage(frame, "y", cols, penalty=-1.0)
