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
  ``classes_`` is not ``[0, 1]``, picks the wrong column.
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

    Accepts ``predict_proba`` output of shape ``(n,)`` (already P(Y = 1))
    or ``(n, 2)`` (the column whose ``classes_`` entry is 1; the second
    column when the model has no ``classes_``). Any other shape is refused.
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
    raise MethodIncompatibility(
        f"{context}: the learner's predict_proba returned shape "
        f"{prob.shape} for {n} rows; a binary classifier must return "
        "(n,) or (n, 2).",
        recovery_hint="Use a binary classifier fitted on a 0/1 target.",
        diagnostics={"learner": type(model).__name__, "shape": list(prob.shape)},
    )
