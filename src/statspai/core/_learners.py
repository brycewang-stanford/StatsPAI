"""User-supplied nuisance learners: cloning and positive-class probabilities.

The treatment-effect estimators that accept a learner (``sp.aipw`` /
``sp.ipw`` ``ps_learner=``, ``sp.ltmle`` ``g_learner=`` / ``q_learner=``,
``sp.lmtp`` ``ratio_learner=`` / ``outcome_learner=``) take any
scikit-learn style estimator, ``sp.SuperLearner`` included. Two things
differ across those objects and are handled here once:

* **Cloning.** ``sklearn.base.clone`` needs ``get_params``; an object
  without it is deep-copied instead (it is refitted right after, so any
  fitted state in the copy is overwritten).
* **The probability of class 1.** scikit-learn classifiers return an
  ``(n, 2)`` array ordered by ``classes_``; ``sp.SuperLearner`` and some
  hand-written learners return the ``(n,)`` vector of P(Y = 1). Indexing
  ``[:, 1]`` blindly breaks on the latter and, for a classifier whose
  ``classes_`` is not ``[0, 1]``, picks the wrong column. A classifier
  trained on a single class (a cross-fitting fold without treated rows)
  returns ``(n, 1)``; its one class is P = 1 or P = 0 everywhere.

Every estimator that reads a probability from a learner it did not build
itself goes through :func:`positive_class_proba` (or
:func:`proba_or_predict` when a regressor is also allowed). The remaining
``predict_proba(X)[:, 1]`` reads in the package are on scikit-learn
classifiers the module constructs and fits on a validated 0/1 target, for
which column 1 is class 1 by construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..exceptions import MethodIncompatibility


def clone_learner(learner: Any) -> Any:
    """Unfitted copy of ``learner`` (``sklearn.base.clone``, else a deep copy)."""
    try:
        from sklearn.base import clone

        return clone(learner)
    except (TypeError, ValueError, RuntimeError, AttributeError):
        import copy

        return copy.deepcopy(learner)


def positive_class_proba(model: Any, X: Any, *, context: str) -> np.ndarray:
    """P(Y = 1 | X) from a fitted binary classifier.

    Accepts ``predict_proba`` output of shape ``(n,)`` (already P(Y = 1)),
    ``(n, 2)`` (the column whose ``classes_`` entry is 1; the second
    column when the model has no ``classes_``), or ``(n, 1)`` from a model
    fitted on one class (ones if that class is 1, else zeros). Any other
    shape is refused.
    """
    if not hasattr(model, "predict_proba"):
        raise MethodIncompatibility(
            f"{context}: the learner has no predict_proba method, so it "
            "cannot give treatment probabilities.",
            recovery_hint="Pass a classifier (e.g. sklearn LogisticRegression "
            "or sp.SuperLearner(task='classification')).",
            diagnostics={"learner": type(model).__name__},
        )
    prob = np.asarray(model.predict_proba(X), dtype=float)
    n = np.shape(X)[0]
    if prob.ndim == 1 and prob.shape[0] == n:
        return prob
    if prob.ndim == 2 and prob.shape == (n, 2):
        classes = getattr(model, "classes_", None)
        if classes is not None:
            hit = np.flatnonzero(np.asarray(classes) == 1)
            if len(classes) == 2 and hit.size == 1:
                return prob[:, int(hit[0])]
        return prob[:, 1]
    if prob.ndim == 2 and prob.shape == (n, 1):
        classes = getattr(model, "classes_", None)
        if classes is not None and len(classes) == 1:
            return np.full(n, 1.0 if np.asarray(classes)[0] == 1 else 0.0)
    raise MethodIncompatibility(
        f"{context}: the learner's predict_proba returned shape "
        f"{prob.shape} for {n} rows; a binary classifier must return "
        "(n,) or (n, 2).",
        recovery_hint="Use a binary classifier fitted on a 0/1 target.",
        diagnostics={"learner": type(model).__name__, "shape": list(prob.shape)},
    )


def proba_or_predict(model: Any, X: Any, *, context: str) -> np.ndarray:
    """P(Y = 1 | X) when ``model`` has ``predict_proba``, else ``predict``.

    The nuisance slots that take either a classifier or a regressor for a
    0/1 target (a propensity in ``sp.dml(model='irm')``, a metalearner's
    ``propensity_model``) read the conditional mean this way.
    """
    if hasattr(model, "predict_proba"):
        return positive_class_proba(model, X, context=context)
    return np.asarray(model.predict(X), dtype=float).ravel()
