"""User-supplied propensity-score learners for :func:`statspai.ipw`.

A learner is any scikit-learn style classifier (``fit`` /
``predict_proba``), ``sp.SuperLearner(task='classification')`` included.
With ``folds`` the score of each row is predicted by a learner trained on
the other folds (cross-fitting); without, the learner is trained and
evaluated on the same rows.
"""

from __future__ import annotations

import numbers
import warnings
from typing import Any, Optional

import numpy as np

from ..exceptions import DataInsufficient, MethodIncompatibility

#: The clip ``_estimate_propensity`` applies to a fitted logit / probit.
_BOUND = 1e-8


def learner_pscore(
    learner: Any,
    X: np.ndarray,
    T: np.ndarray,
    sw: Optional[np.ndarray],
    folds: Optional[np.ndarray],
) -> np.ndarray:
    """Propensity scores from ``learner``, out of fold when ``folds`` is given.

    ``folds`` holds one fold label per row. In a bootstrap resample the
    copies of a row carry that row's label, so no copy is ever predicted
    by a learner that saw another copy of it.
    """
    from .aipw import _learner_predict

    if folds is None:
        e = _learner_predict(learner, X, T, X, sw, proba=True)
    else:
        e = np.empty(len(T))
        for f in np.unique(folds):
            test = folds == f
            train = ~test
            if T[train].min() == T[train].max():
                raise DataInsufficient(
                    "ipw: a cross-fitting training set holds a single "
                    "treatment arm, so the propensity learner cannot be "
                    "trained.",
                    recovery_hint="Use fewer folds (n_folds=), or cross_fit=False.",
                    diagnostics={"fold": int(f), "n_train": int(train.sum())},
                )
            e[test] = _learner_predict(
                learner,
                X[train],
                T[train],
                X[test],
                None if sw is None else sw[train],
                proba=True,
            )
    return np.clip(e, _BOUND, 1 - _BOUND)


def check_learner_options(
    se_method: str,
    design: Any,
    propensity: Any,
    ps_model: str,
    cross_fit: bool,
    n_folds: int,
) -> None:
    """Refuse the options a user propensity learner cannot serve."""
    if se_method == "sandwich" or design is not None:
        raise MethodIncompatibility(
            "ipw: the sandwich and design-based variances stack the score "
            "of a parametric propensity model; a ps_learner has none.",
            recovery_hint="Use se_method='bootstrap' (each replicate refits "
            "the learner), or sp.aipw(ps_learner=) for an influence-function "
            "standard error.",
        )
    if propensity is not None:
        raise MethodIncompatibility(
            "ipw: propensity= gives the score, ps_learner= estimates it; "
            "pass one of them.",
            recovery_hint="Drop ps_learner= when the design probability is known.",
        )
    if ps_model != "logit":
        raise MethodIncompatibility(
            "ipw: ps_model= names the parametric propensity model, which "
            "ps_learner= replaces.",
            recovery_hint="Drop ps_model=, or pass the probit as the learner.",
        )
    if cross_fit and not (isinstance(n_folds, numbers.Integral) and n_folds >= 2):
        raise MethodIncompatibility(
            f"ipw: cross-fitting needs an integer n_folds >= 2, got {n_folds!r}.",
            recovery_hint="Use n_folds=5 (the default), or cross_fit=False.",
        )
    if not cross_fit:
        warnings.warn(
            "ipw: a propensity learner trained and evaluated on the same rows "
            "(cross_fit=False) overfits the scores of its training units, "
            "which pulls the weighted contrast towards the raw difference "
            "in means.",
            RuntimeWarning,
            stacklevel=5,
        )
