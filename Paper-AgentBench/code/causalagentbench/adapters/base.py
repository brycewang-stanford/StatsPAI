"""Adapter base class + the shared estimate-extraction helper."""

from __future__ import annotations

import abc
from typing import Any

from ..conditions import Condition
from ..schema import Task, Trajectory


class AgentAdapter(abc.ABC):
    """An adapter turns one Task (at one seed) into one Trajectory."""

    def __init__(self, condition: Condition, **kwargs: Any) -> None:
        self.condition = condition
        self.kwargs = kwargs

    @abc.abstractmethod
    def run(self, task: Task, seed: int = 0) -> Trajectory:  # pragma: no cover
        ...


def extract_effect(result: Any, treatment_name: str = "treatment") -> float:
    """Pull a scalar causal estimate out of any StatsPAI result object.

    StatsPAI returns two families:

    * ``CausalResult`` (did / rd / staggered / matching) exposes a uniform
      ``.estimate`` scalar.
    * ``EconometricResults`` (regress / ivreg) exposes a ``.params`` Series
      indexed by coefficient name.

    This helper tries both so the oracle dispatch stays small.
    """
    est = getattr(result, "estimate", None)
    if isinstance(est, (int, float)):
        return float(est)
    params = getattr(result, "params", None)
    if params is not None:
        # named coefficient (regress / ivreg)
        try:
            return float(params[treatment_name])
        except Exception:
            pass
        # single-row params Series (e.g. RD "RD Effect")
        try:
            return float(params.iloc[0])
        except Exception:
            pass
    raise ValueError(
        f"Could not extract a scalar effect from {type(result).__name__}; "
        f"checked .estimate and .params['{treatment_name}']."
    )
