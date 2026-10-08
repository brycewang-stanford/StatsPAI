"""Estimator tags for the duck-typed scikit-learn learners.

``RlassoRegressor`` / ``RlassoClassifier`` / ``RlassologitClassifier`` and
the HAL learners follow the scikit-learn estimator contract without
inheriting ``sklearn.base.BaseEstimator`` (inheriting it would import
scikit-learn on every ``import statspai``). scikit-learn 1.6 moved estimator
tags to ``__sklearn_tags__`` and 1.7 stopped inferring them for classes that
do not define it, so ``clone``, ``is_classifier`` and the cross-fitting
utilities raise ``AttributeError`` on such a class. This module builds the
tags; scikit-learn is imported only when they are asked for.
"""

from __future__ import annotations

from typing import Any


def duck_typed_tags(estimator: Any) -> Any:
    """Return ``sklearn.utils.Tags`` for a learner with ``_estimator_type``."""
    from sklearn.utils import ClassifierTags, RegressorTags, Tags, TargetTags

    kind = getattr(estimator, "_estimator_type", None)
    return Tags(
        estimator_type=kind,
        target_tags=TargetTags(required=True),
        classifier_tags=ClassifierTags() if kind == "classifier" else None,
        regressor_tags=RegressorTags() if kind == "regressor" else None,
    )
