"""GRF-inspired extensions for CausalForest: variable_importance, BLP, ate, att."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.forest.causal_forest import CausalForest


@pytest.fixture(scope="module")
def fitted_cf():
    rng = np.random.default_rng(42)
    n = 600
    X = rng.standard_normal((n, 3))
    T = rng.binomial(1, 0.5, n)
    # CATE = X[:, 0]  (heterogeneous along dim 0 only)
    Y = X[:, 0] * T + X[:, 1] + rng.standard_normal(n)
    data = pd.DataFrame(
        {
            "Y": Y,
            "T": T,
            "X0": X[:, 0],
            "X1": X[:, 1],
            "X2": X[:, 2],
        }
    )
    cf = CausalForest(n_estimators=50, random_state=42)
    cf.fit("Y ~ T | X0 + X1 + X2", data=data)
    return cf


def test_variable_importance_shape_and_norm(fitted_cf):
    vi = fitted_cf.variable_importance()
    assert len(vi) == 3
    np.testing.assert_allclose(vi.sum(), 1.0, atol=1e-8)
    assert all(v >= 0 for v in vi.values)


def test_variable_importance_sums_to_one(fitted_cf):
    vi = fitted_cf.variable_importance()
    np.testing.assert_allclose(vi.sum(), 1.0, atol=1e-8)


def test_blp_detects_heterogeneity(fitted_cf):
    blp = fitted_cf.best_linear_projection()
    # X0 should have a significant t-stat (drives CATE)
    assert abs(blp.loc["X0", "t"]) > 2.0
    assert blp.loc["X0", "p"] < 0.05


def test_blp_returns_full_table(fitted_cf):
    blp = fitted_cf.best_linear_projection()
    # v1.15 ML+causal polish: BLP rewritten to AIPW pseudo-outcome with
    # HC1 SEs and now reports a 95% CI (ci_lower / ci_upper) alongside
    # the legacy coef / se / t / p columns.
    assert set(blp.columns) == {
        "coef",
        "se",
        "t",
        "p",
        "ci_lower",
        "ci_upper",
    }
    assert "Intercept" in blp.index
    assert "X0" in blp.index


def test_grf_helpers_raise_taxonomy_when_unfitted():
    cf = CausalForest()

    with pytest.raises(MethodIncompatibility, match="variable_importance"):
        cf.variable_importance()
    with pytest.raises(MethodIncompatibility, match="best_linear_projection"):
        cf.best_linear_projection()
    with pytest.raises(MethodIncompatibility, match="ate"):
        cf.ate()
    with pytest.raises(MethodIncompatibility, match="att"):
        cf.att()


def test_blp_validates_prediction_inputs_and_scalar_contracts(fitted_cf):
    with pytest.raises(MethodIncompatibility, match="alpha"):
        fitted_cf.best_linear_projection(alpha=np.nan)

    with pytest.raises(MethodIncompatibility, match="clip"):
        fitted_cf.best_linear_projection(clip=0.5)

    with pytest.raises(MethodIncompatibility) as wrong_shape:
        fitted_cf.best_linear_projection(np.ones((2, 2)))
    assert wrong_shape.value.diagnostics["expected_features"] == 3

    with pytest.raises(MethodIncompatibility, match="numeric"):
        fitted_cf.best_linear_projection(np.array([["bad", "data", "x"]]))

    with pytest.raises(DataInsufficient):
        fitted_cf.best_linear_projection(np.empty((0, 3)))

    with pytest.warns(UserWarning, match="different sample size"):
        out = fitted_cf.best_linear_projection([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert "Intercept" in out.index


def test_ate_finite(fitted_cf):
    ate = fitted_cf.ate()
    assert np.isfinite(ate)
    # With n=600 and honest splitting, ATE should be in a reasonable range
    assert abs(ate) < 2.0


def test_att_runs(fitted_cf):
    att = fitted_cf.att()
    assert np.isfinite(att)


def test_att_validates_external_treatment_vector(fitted_cf):
    with pytest.raises(MethodIncompatibility) as wrong_len:
        fitted_cf.att(T=np.ones(2))
    assert wrong_len.value.diagnostics["n_effects"] == 600

    with pytest.raises(MethodIncompatibility, match="numeric"):
        fitted_cf.att(T=np.array(["bad"] * 600, dtype=object))

    with pytest.raises(DataInsufficient, match="no treated"):
        fitted_cf.att(T=np.zeros(600))


def test_leaf_effect_uses_ols_slope_not_mixed_ddof_covariance():
    cf = CausalForest(discrete_treatment=True)
    tree = DecisionTreeRegressor(max_depth=1, random_state=0)
    X = np.zeros((4, 1))
    tree.fit(X, np.zeros(4))

    t_resid = np.array([-0.5, 0.5, -0.5, 0.5])
    y_resid = 2.0 * t_resid
    cf._replace_leaf_values_with_causal_effects(tree, X, t_resid, y_resid)

    np.testing.assert_allclose(tree.predict(X), np.full(4, 2.0), atol=1e-12)


def _small_cf_data(n: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    age = rng.normal(size=n)
    score = rng.normal(size=n)
    T = rng.binomial(1, 0.5, size=n)
    Y = (1.0 + 0.4 * age - 0.2 * score) * T + age + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"Y": Y, "T": T, "age": age, "score": score})


def _small_formula_causal_forest() -> tuple[CausalForest, pd.DataFrame]:
    data = _small_cf_data()
    cf = CausalForest(
        n_estimators=8,
        min_samples_leaf=2,
        max_samples=0.8,
        model_y=RandomForestRegressor(n_estimators=8, random_state=1),
        model_t=RandomForestClassifier(n_estimators=8, random_state=2),
        random_state=3,
    )
    cf.fit("Y ~ T | age + score", data=data)
    return cf, data


def test_predict_formula_uses_original_effect_modifier_names():
    cf, data = _small_formula_causal_forest()

    pred = cf.predict(data[["age", "score"]].head(5))

    assert pred.shape == (5,)


def test_predict_raises_taxonomy_for_missing_or_bad_effect_modifiers():
    cf, data = _small_formula_causal_forest()

    with pytest.raises(MethodIncompatibility) as missing:
        cf.predict(data[["age"]].head(2))
    assert missing.value.diagnostics["missing_columns"] == ["score"]

    bad = data[["age", "score"]].head(2).copy()
    bad["score"] = ["bad", "worse"]
    with pytest.raises(MethodIncompatibility, match="must be numeric"):
        cf.predict(bad)

    nonfinite = data[["age", "score"]].head(2).copy()
    nonfinite.loc[nonfinite.index[0], "age"] = np.inf
    with pytest.raises(MethodIncompatibility, match="NaN or infinite"):
        cf.predict(nonfinite)


def test_effect_validates_shape_and_empty_prediction_samples(fitted_cf):
    with pytest.raises(MethodIncompatibility) as wrong_shape:
        fitted_cf.effect(np.ones((2, 2)))
    assert wrong_shape.value.diagnostics["expected_features"] == 3

    with pytest.raises(DataInsufficient):
        fitted_cf.effect(np.empty((0, 3)))

    pred = fitted_cf.effect(np.array([0.0, 0.0, 0.0]))
    assert pred.shape == (1,)


def test_effect_interval_uses_prediction_validation_and_alpha_contract(fitted_cf):
    lo, hi = fitted_cf.effect_interval(np.array([0.0, 0.0, 0.0]), alpha=0.1)
    assert lo.shape == (1,)
    assert hi.shape == (1,)

    with pytest.raises(MethodIncompatibility, match="alpha"):
        fitted_cf.effect_interval(np.zeros((2, 3)), alpha=np.nan)

    with pytest.raises(MethodIncompatibility) as wrong_shape:
        fitted_cf.effect_interval(np.ones((2, 2)))
    assert wrong_shape.value.diagnostics["expected_features"] == 3


def test_fit_rejects_invalid_controls_before_sklearn():
    data = _small_cf_data()

    with pytest.raises(MethodIncompatibility, match="n_estimators"):
        CausalForest(n_estimators=0).fit(
            "Y ~ T | age + score",
            data=data,
        )

    with pytest.raises(MethodIncompatibility, match="max_samples"):
        CausalForest(max_samples=np.nan).fit(
            "Y ~ T | age + score",
            data=data,
        )


def test_fit_rejects_bad_numeric_inputs_with_taxonomy():
    data = _small_cf_data()
    bad = data.copy()
    bad["score"] = ["bad"] * len(bad)

    with pytest.raises(MethodIncompatibility, match="numeric"):
        CausalForest().fit("Y ~ T | age + score", data=bad)

    bad = data.copy()
    bad.loc[bad.index[0], "age"] = np.inf
    with pytest.raises(MethodIncompatibility, match="NaN or infinite"):
        CausalForest().fit("Y ~ T | age + score", data=bad)


def test_fit_rejects_unsupported_or_sparse_discrete_treatment():
    data = _small_cf_data()

    multiclass = data.copy()
    multiclass["T"] = np.arange(len(multiclass)) % 3
    with pytest.raises(MethodIncompatibility, match="binary treatment"):
        CausalForest().fit("Y ~ T | age + score", data=multiclass)

    recode_needed = data.copy()
    recode_needed["T"] = recode_needed["T"] + 1
    with pytest.raises(MethodIncompatibility, match="coded as 0/1"):
        CausalForest().fit("Y ~ T | age + score", data=recode_needed)

    sparse = data.copy()
    sparse["T"] = 0
    sparse.loc[sparse.index[:2], "T"] = 1
    with pytest.raises(DataInsufficient, match="at least 3 observations"):
        CausalForest().fit("Y ~ T | age + score", data=sparse)
