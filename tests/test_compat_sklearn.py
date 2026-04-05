"""Tests for sklearn-compatible wrappers."""

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip("sklearn")


def _make_data(n=200, k=3, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, k)
    beta = np.array([2.0, -1.0, 0.5])
    y = X @ beta + rng.randn(n) * 0.5
    return X, y


class TestSklearnOLS:
    def test_fit_predict(self):
        from statspai.compat import SklearnOLS
        X, y = _make_data()
        model = SklearnOLS(robust="hc1")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (200,)
        assert hasattr(model, "results_")
        assert hasattr(model, "coef_")
        assert hasattr(model, "intercept_")

    def test_get_set_params(self):
        from statspai.compat import SklearnOLS
        model = SklearnOLS(robust="hc1")
        params = model.get_params()
        assert params["robust"] == "hc1"
        model.set_params(robust="hc3")
        assert model.robust == "hc3"

    def test_cross_val_score(self):
        from statspai.compat import SklearnOLS
        from sklearn.model_selection import cross_val_score
        X, y = _make_data(n=100)
        model = SklearnOLS()
        scores = cross_val_score(model, X, y, cv=3, scoring="r2")
        assert len(scores) == 3

    def test_pipeline(self):
        from statspai.compat import SklearnOLS
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        X, y = _make_data()
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ols", SklearnOLS(robust="hc1")),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (200,)


class TestSklearnDML:
    def test_fit_predict(self):
        from statspai.compat import SklearnDML
        rng = np.random.RandomState(42)
        n = 300
        confounders = rng.randn(n, 2)
        treatment = (rng.randn(n) + confounders[:, 0] > 0).astype(float)
        y = 2.0 * treatment + confounders @ [1.0, -0.5] + rng.randn(n) * 0.5
        X = np.column_stack([treatment, confounders])

        model = SklearnDML(n_folds=3)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (n,)
        assert hasattr(model, "ate_")
