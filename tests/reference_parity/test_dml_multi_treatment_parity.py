"""``sp.dml(treat=[d1, d2])``: several treatments vs Python DoubleML (review §4.5).

DoubleML's multi-``d`` PLR fits each treatment with the other treatments as
controls on one cross-fitting split; StatsPAI does the same and also returns
the covariance *across* treatments from the stacked scores. With identical
folds and learners (``LinearRegression``), coefficients and SEs equal
DoubleML's, and the joint covariance equals the one DoubleML's own ``psi`` /
``psi_deriv`` arrays imply: observed 3e-15, asserted at 1e-10 (T2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

dml = pytest.importorskip("doubleml")
from sklearn.linear_model import LinearRegression as LR  # noqa: E402

X_COLS = [f"x{j}" for j in range(5)]


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(size=(n, 5))
    d1 = X[:, 0] + rng.normal(size=n)
    d2 = 0.5 * d1 + X[:, 1] + rng.normal(size=n)
    y = 0.5 * d1 - 0.3 * d2 + X[:, 0] + X[:, 2] + rng.normal(size=n)
    out = pd.DataFrame(X, columns=X_COLS)
    out["d1"], out["d2"], out["y"] = d1, d2, y
    return out


@pytest.fixture(scope="module")
def fits(df):
    from statspai.dml.plr import DoubleMLPLR

    ours = sp.dml(
        df,
        y="y",
        treat=["d1", "d2"],
        covariates=X_COLS,
        ml_g=LR(),
        ml_m=LR(),
        n_folds=5,
        random_state=7,
    )
    probe = DoubleMLPLR(
        data=df,
        y="y",
        treat="d1",
        covariates=X_COLS + ["d2"],
        ml_g=LR(),
        ml_m=LR(),
        n_folds=5,
        n_rep=1,
        random_state=7,
    )
    splits = probe._make_splits(df[X_COLS].to_numpy(), rng_seed=7)
    ref = dml.DoubleMLPLR(
        dml.DoubleMLData(df, "y", ["d1", "d2"], X_COLS),
        LR(),
        LR(),
        n_folds=5,
        draw_sample_splitting=False,
    )
    ref.set_sample_splitting([[(tr, te) for tr, te in splits]])
    ref.fit()
    return ours, ref


def test_coefficients_and_ses_match_doubleml(fits):
    ours, ref = fits
    np.testing.assert_allclose(ours.params.values, ref.coef, rtol=1e-10)
    np.testing.assert_allclose(ours.std_errors.values, ref.se, rtol=1e-10)


def test_joint_covariance_matches_doubleml_scores(fits, df):
    ours, ref = fits
    n = len(df)
    psi = ref.psi[:, 0, :]
    J = ref.psi_deriv[:, 0, :].mean(axis=0)
    V_ref = (psi.T @ psi) / n / np.outer(J, J) / n
    np.testing.assert_allclose(ours.data_info["var_cov"], V_ref, rtol=1e-10)


def test_joint_hypotheses_use_the_cross_covariance(fits):
    ours, _ = fits
    V = ours.data_info["var_cov"]
    w = sp.test(ours, "d1 + d2 = 0")
    diff = ours.params["d1"] + ours.params["d2"]
    var = V[0, 0] + V[1, 1] + 2 * V[0, 1]
    assert w["statistic"] == pytest.approx(diff**2 / var, rel=1e-10)


def test_unsupported_options_are_refused(df):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="model='plr'"):
        sp.dml(df, y="y", treat=["d1", "d2"], covariates=X_COLS, model="irm")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="n_rep"):
        sp.dml(df, y="y", treat=["d1", "d2"], covariates=X_COLS, n_rep=3)
