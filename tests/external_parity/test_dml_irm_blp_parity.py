"""``sp.best_linear_projection`` on a DML IRM fit, against ``DoubleML.cate()``.

The CATE-inference chapter of *Applied Causal Inference Powered by ML and
AI* regresses the cross-fitted doubly-robust score on a basis of
covariates to get the best linear predictor of the conditional effect
(Semenova and Chernozhukov 2021). ``DoubleMLIRM.cate(basis)`` and
``.gate(groups)`` do the same. Before this test ``sp.best_linear_projection``
accepted forests only.

Shared folds and closed-form learners make the scores identical on the two
sides, so coefficients agree to 1e-10. The standard errors differ by a
documented convention: StatsPAI follows ``sandwich::vcovCL`` and applies
the ``n / (n - 1)`` adjustment under every ``vce``, ``DoubleML`` uses
statsmodels' plain HC0. Removing the factor reproduces ``DoubleML`` to
1e-10, which is what is asserted.

References
----------
[@semenova2021debiased], [@bach2022doubleml]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import statspai as sp

doubleml = pytest.importorskip("doubleml")

N, K = 2000, 4
COVARIATES = ["x1", "x2", "x3"]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, 3))
    p = 1.0 / (1.0 + np.exp(-X[:, 0]))
    d = (rng.uniform(size=N) < p).astype(float)
    y = (1.0 + X[:, 1]) * d + X[:, 0] + rng.normal(size=N)
    df = pd.DataFrame(X, columns=COVARIATES)
    df["d"], df["y"] = d, y
    return df


@pytest.fixture(scope="module")
def fits(data):
    folds = np.arange(N) % K
    ours = sp.dml(
        data, y="y", treat="d", covariates=COVARIATES, model="irm",
        ml_g=LinearRegression(), ml_m=LogisticRegression(),
        n_folds=K, fold_indices=folds,
    )  # fmt: skip
    dd = doubleml.DoubleMLData(data, y_col="y", d_cols="d", x_cols=COVARIATES)
    ref = doubleml.DoubleMLIRM(
        dd, ml_g=LinearRegression(), ml_m=LogisticRegression(),
        n_folds=K, draw_sample_splitting=False,
    )  # fmt: skip
    ref.set_sample_splitting(
        [[(np.flatnonzero(folds != f), np.flatnonzero(folds == f)) for f in range(K)]]
    )
    ref.fit()
    return ours, ref


def _reference(blp):
    model = blp.blp_model[0] if isinstance(blp.blp_model, list) else blp.blp_model
    return np.asarray(model.params), np.sqrt(np.diag(np.squeeze(blp.blp_omega)))


def test_projection_on_a_basis_matches_doubleml_cate(data, fits):
    ours, ref = fits
    A = pd.DataFrame({"x2": data["x2"], "x2sq": data["x2"] ** 2})
    table = sp.best_linear_projection(ours, A=A, vce="HC0")
    coef, se = _reference(ref.cate(A.assign(const=1.0)[["const", "x2", "x2sq"]]))
    np.testing.assert_allclose(table["coef"], coef, rtol=1e-10)
    np.testing.assert_allclose(table["se"] * np.sqrt((N - 1) / N), se, rtol=1e-10)
    assert list(table.index) == ["Intercept", "x2", "x2sq"]
    vcov = table.attrs["vcov"]
    np.testing.assert_allclose(np.sqrt(np.diag(vcov)), table["se"], rtol=1e-12)


def test_group_effects_match_doubleml_gate(data, fits):
    ours, ref = fits
    hi = (data["x2"] >= 0).astype(float).rename("hi")
    table = sp.best_linear_projection(ours, A=hi, vce="HC0")
    groups = pd.DataFrame({"lo": data["x2"] < 0, "hi": data["x2"] >= 0})
    coef, _ = _reference(ref.gate(groups))
    assert table.loc["Intercept", "coef"] == pytest.approx(coef[0], rel=1e-10)
    assert table["coef"].sum() == pytest.approx(coef[1], rel=1e-10)


def test_no_covariates_gives_the_average_effect(fits):
    ours, _ = fits
    table = sp.best_linear_projection(ours, vce="HC0")
    assert table.loc["Intercept", "coef"] == pytest.approx(ours.estimate, rel=1e-12)
    assert table.loc["Intercept", "se"] * np.sqrt((N - 1) / N) == pytest.approx(
        ours.se, rel=1e-10
    )


def test_recovers_a_known_linear_conditional_effect(data, fits):
    """tau(x) = 1 + x2 in the DGP: slope 1, no curvature (3-se band)."""
    ours, _ = fits
    A = pd.DataFrame({"x2": data["x2"], "x2sq": data["x2"] ** 2})
    table = sp.best_linear_projection(ours, A=A)
    assert abs(table.loc["x2", "coef"] - 1.0) < 3 * table.loc["x2", "se"]
    assert abs(table.loc["x2sq", "coef"]) < 3 * table.loc["x2sq", "se"]


def test_other_dml_fits_are_refused(data):
    plr = sp.dml(
        data, y="y", treat="d", covariates=COVARIATES, model="plr",
        ml_g=LinearRegression(), ml_m=LinearRegression(), n_folds=2,
    )  # fmt: skip
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="irm"):
        sp.best_linear_projection(plr, A=data[["x2"]])
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="one row"):
        irm = sp.dml(
            data, y="y", treat="d", covariates=COVARIATES, model="irm",
            ml_g=LinearRegression(), ml_m=LogisticRegression(), n_folds=2,
        )  # fmt: skip
        sp.best_linear_projection(irm, A=data[["x2"]].iloc[:100])
