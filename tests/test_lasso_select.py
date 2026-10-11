"""Tests for ``sp.lasso_select`` — LASSO variable selection with BIC/AIC/CV
penalty choice (previously untested public function).

On a sparse linear DGP the selected set must contain the true support (no
false negatives), the recovered coefficients must be close to the truth, and
BIC — the most aggressive criterion — must not over-select relative to AIC.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp

TRUE_SUPPORT = {"x0", "x3", "x7"}


def _sparse_data(n=400, p=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 3] + 0.8 * X[:, 7] + rng.normal(0, 0.5, n)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    df["y"] = y
    return df, [f"x{i}" for i in range(p)]


@pytest.mark.parametrize("method", ["bic", "aic", "cv"])
def test_lasso_select_recovers_true_support(method):
    df, x = _sparse_data()
    res = sp.lasso_select(df, y="y", x=x, method=method, verbose=False, seed=0)
    # No false negatives: every truly-relevant variable is retained.
    assert TRUE_SUPPORT <= set(res.selected)


def test_lasso_select_coefficients_close_to_truth():
    df, x = _sparse_data()
    res = sp.lasso_select(df, y="y", x=x, method="bic", verbose=False, seed=0)
    coef = res.coefficients
    assert coef["x0"] == pytest.approx(2.0, abs=0.2)
    assert coef["x3"] == pytest.approx(-1.5, abs=0.2)
    assert coef["x7"] == pytest.approx(0.8, abs=0.2)


def test_lasso_select_result_contract():
    df, x = _sparse_data()
    res = sp.lasso_select(df, y="y", x=x, method="bic", verbose=False, seed=0)
    assert set(res.selected) <= set(x)
    # selected and dropped partition the candidate set.
    assert set(res.selected).isdisjoint(set(res.dropped))
    assert set(res.selected) | set(res.dropped) == set(x)
    assert res.method == "lasso_bic"


def test_lasso_select_bic_no_larger_than_aic():
    df, x = _sparse_data()
    bic = sp.lasso_select(df, y="y", x=x, method="bic", verbose=False, seed=0)
    aic = sp.lasso_select(df, y="y", x=x, method="aic", verbose=False, seed=0)
    # BIC's heavier penalty -> at most as many variables as AIC.
    assert len(bic.selected) <= len(aic.selected)


def test_lasso_select_is_seed_reproducible():
    df, x = _sparse_data()
    a = sp.lasso_select(df, y="y", x=x, method="cv", verbose=False, seed=42)
    b = sp.lasso_select(df, y="y", x=x, method="cv", verbose=False, seed=42)
    assert a.selected == b.selected


def test_cv_error_curve_is_the_one_of_glmnet_on_the_same_folds():
    # Each training fold is centred and scaled on its own rows and the
    # error is pooled over observations: on the same folds and penalties
    # that is the cross-validated error of sp.glmnet (itself pinned to
    # cv.glmnet). Up to 1.39.4 the folds reused the full-sample centring
    # and scaling and had no intercept of their own; on this design the
    # two curves then differed by several percent.
    import importlib

    sw = importlib.import_module("statspai.selection.stepwise")
    rng = np.random.default_rng(3)
    n, p = 45, 12
    X = rng.normal(size=(n, p)) * rng.uniform(0.5, 4.0, size=p) + rng.normal(size=p)
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(p)])
    df["y"] = 5.0 + X[:, 0] - 0.5 * X[:, 3] + rng.normal(size=n)
    x = list(df.columns[:-1])
    res = sp.lasso_select(
        df,
        y="y",
        x=x,
        method="cv",
        n_folds=5,
        n_lambda=25,
        eps=0.01,
        max_iter=20000,
        tol=1e-10,
        seed=7,
        verbose=False,
    )
    folds = np.empty(n, dtype=int)
    for k, (_, test_idx) in enumerate(sw._kfold_indices(n, 5, 7)):
        folds[test_idx] = k + 1
    ref = sp.glmnet(
        df, y="y", x=x, lambda_=list(res.lasso_path["lambdas"]), foldid=folds
    )
    np.testing.assert_allclose(
        res.lasso_path["cv_mse"], ref.cv["cvm"].to_numpy(), rtol=1e-6
    )
    assert res.lasso_path["lambda_best"] == pytest.approx(ref.lambda_min, rel=1e-12)
    assert sorted(res.selected) == sorted(ref.params.index[ref.params != 0])
