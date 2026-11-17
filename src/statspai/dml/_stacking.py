"""Short stacking of nuisance learners inside ``sp.dml(model='plr')``.

A nuisance given as a roster of learners is cross-fitted once per
candidate on the partition of the fit, and the candidates' out-of-fold
predictions are combined with the constrained least-squares weights of
Ahrens, Hansen, Schaffer and Wiemann (2025, eq. 7) [@ahrens2025model]:
non-negative, summing to one, chosen on the full sample. The stacked
out-of-fold prediction then takes the place of a single learner's, so
every score and variance of the PLR class applies unchanged (partialling
out and IV-type, sample weights, one- and two-way clusters, survey
designs, repeated cross-fitting, the sensitivity bound).

``sp.dml_model_averaging(weight_rule='short_stacking')`` is the same
estimator; ``tests/test_dml_short_stack.py`` pins the two against each
other.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..exceptions import MethodIncompatibility
from ._learners import _is_estimator_like, resolve_learner

#: Strings that ask for the default roster.
STACK_ALIASES = frozenset({"stack", "stacking", "short_stacking", "short-stacking"})


class ShortStack:
    """A roster of candidate learners for one nuisance function.

    Not an estimator: short stacking chooses its weights from the
    cross-fitted predictions of the whole sample, so it cannot be fitted
    inside one training fold. The PLR class recognises the object and
    cross-fits its members.
    """

    def __init__(self, learners: Sequence[Any], labels: Sequence[str]) -> None:
        self.learners: List[Any] = list(learners)
        self.labels: List[str] = [str(lab) for lab in labels]

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {"learners": self.learners, "labels": self.labels}

    def __repr__(self) -> str:
        return f"ShortStack({', '.join(self.labels)})"


def _default_roster(role: str) -> ShortStack:
    """Lasso, Ridge, random forest and gradient boosting: the roster of
    ``sp.dml_model_averaging`` (one source, so the two cannot drift)."""
    from .model_averaging import _default_candidates

    side = 1 if role == "ml_m" else 0
    cand = _default_candidates()
    return ShortStack([c[side] for c in cand], [c[2] for c in cand])


def as_short_stack(spec: Any, role: str) -> Optional[ShortStack]:
    """``ShortStack`` for a stacking request, ``None`` for anything else.

    A request is ``'stack'`` (the default roster), a list of learners --
    estimators, string aliases, or ``(label, learner)`` pairs -- or a
    ``ShortStack`` itself.
    """
    if isinstance(spec, ShortStack):
        return spec
    if isinstance(spec, str):
        return _default_roster(role) if spec.lower() in STACK_ALIASES else None
    if not isinstance(spec, (list, tuple)):
        return None
    if len(spec) == 0:
        raise MethodIncompatibility(
            f"dml: {role} is an empty list of learners.",
            recovery_hint=f"Pass at least one learner, or {role}='stack'.",
        )
    learners: List[Any] = []
    labels: List[str] = []
    for item in spec:
        label: Optional[str] = None
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], str)
            and not isinstance(item[1], str)
        ):
            label, item = item[0], item[1]
        if isinstance(item, str):
            if item.lower() in STACK_ALIASES:
                raise MethodIncompatibility(
                    f"dml: {role} lists {item!r} as a candidate; a roster "
                    "cannot contain another roster.",
                    recovery_hint=f"Pass {role}='stack' on its own.",
                )
            learner = resolve_learner(item, kind="regressor", role=role)
            label = label or item.lower()
        elif _is_estimator_like(item):
            learner = item
            label = label or type(item).__name__
        else:
            raise MethodIncompatibility(
                f"dml: {role} lists {item!r}, which is neither a "
                "scikit-learn estimator nor a learner alias.",
                recovery_hint="List estimators, aliases such as 'lasso' or "
                "'rf', or (label, estimator) pairs.",
            )
        base, k = label, 2
        while label in labels:
            label = f"{base}_{k}"
            k += 1
        learners.append(learner)
        labels.append(label)
    return ShortStack(learners, labels)


def crossfit_short_stack(
    stack: ShortStack,
    crossfit: Callable[[Any], np.ndarray],
    target: np.ndarray,
    sample_weight: Optional[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Stacked out-of-fold prediction of ``target`` and what produced it.

    ``crossfit(learner)`` returns one candidate's out-of-fold prediction
    for every row. The weights minimise the (sample-weighted) squared
    error of the combination on the full sample.
    """
    from .model_averaging import _solve_cls_weights

    preds = np.column_stack([crossfit(learner) for learner in stack.learners])
    weights = _solve_cls_weights(target, preds, sample_weight=sample_weight)
    stacked = preds @ weights
    err = (target[:, None] - preds) ** 2
    if sample_weight is None:
        mse = err.mean(axis=0)
    else:
        mse = (sample_weight[:, None] * err).sum(axis=0) / float(sample_weight.sum())
    info = {
        "candidates": list(stack.labels),
        "weights": dict(zip(stack.labels, weights.tolist())),
        "mse": dict(zip(stack.labels, mse.tolist())),
    }
    return stacked, info


def aggregate_stack_info(per_rep: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """``model_info['stacking']``: weights and risks averaged over the
    repetitions of the cross-fitting, per stacked nuisance."""
    out: Dict[str, Any] = {"rule": "short_stacking"}
    for name in per_rep[0]:
        labels = per_rep[0][name]["candidates"]
        out[name] = {
            "candidates": list(labels),
            **{
                key: {
                    lab: float(np.mean([rep[name][key][lab] for rep in per_rep]))
                    for lab in labels
                }
                for key in ("weights", "mse")
            },
        }
    return out


__all__ = [
    "STACK_ALIASES",
    "ShortStack",
    "aggregate_stack_info",
    "as_short_stack",
    "crossfit_short_stack",
]
