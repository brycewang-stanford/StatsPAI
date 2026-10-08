"""The learners that follow the scikit-learn contract without inheriting it.

scikit-learn 1.7 stopped inferring estimator tags for classes that do not
define ``__sklearn_tags__``; ``clone`` and the cross-fitting utilities then
raise ``AttributeError``. The nightly matrix caught it on 2026-10-07 as
``sp.dml`` failing with an ``RlassoRegressor`` learner.
"""

import importlib

import numpy as np
import pytest

import statspai as sp

sklearn_base = pytest.importorskip("sklearn.base")
_hal = importlib.import_module("statspai.tmle.hal_tmle")

LEARNERS = [
    (sp.RlassoRegressor, "regressor"),
    (sp.RlassoClassifier, "classifier"),
    (sp.RlassologitClassifier, "classifier"),
    (_hal.HALRegressor, "regressor"),
    (_hal.HALClassifier, "classifier"),
]


@pytest.mark.parametrize("cls, kind", LEARNERS)
def test_learner_is_recognised_and_cross_fits(cls, kind):
    from sklearn.model_selection import cross_val_predict

    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 4))
    y = X[:, 0] + rng.normal(size=150)
    target = y if kind == "regressor" else (y > 0).astype(int)

    learner = sklearn_base.clone(cls())
    assert sklearn_base.is_regressor(learner) == (kind == "regressor")
    assert sklearn_base.is_classifier(learner) == (kind == "classifier")
    method = "predict" if kind == "regressor" else "predict_proba"
    out = cross_val_predict(learner, X, target, cv=3, method=method)
    assert out.shape[0] == 150 and np.isfinite(out).all()


@pytest.mark.parametrize("cls, kind", LEARNERS)
def test_tags_name_the_estimator_type(cls, kind):
    pytest.importorskip("sklearn", minversion="1.6")
    tags = cls().__sklearn_tags__()
    assert tags.estimator_type == kind
    assert tags.target_tags.required
    assert (tags.regressor_tags is not None) == (kind == "regressor")
    assert (tags.classifier_tags is not None) == (kind == "classifier")
