"""Decision-ready causal-effect summaries.

This module gives StatsPAI a small, stable analogue of CausalPy's
``EffectSummary``: every supported effect result can return both a
machine-readable table and concise prose.  The implementation deliberately
adapts to StatsPAI's function-first, frequentist-first API instead of imposing
an experiment-class hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Dict, Optional, cast

import numpy as np
import pandas as pd

from .results import _to_jsonable


@dataclass
class EffectSummary:
    """Container for an effect summary table and human-readable text.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> table = pd.DataFrame([{"estimand": "ATT", "estimate": 2.0}])
    >>> summary = sp.EffectSummary(table=table, text="ATT is positive.")
    >>> summary.to_dict()["kind"]
    'effect_summary'
    """

    table: pd.DataFrame
    text: str
    kind: str = "effect_summary"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation for agents and MCP tools."""
        return {
            "kind": self.kind,
            "table": _to_jsonable(self.table),
            "text": self.text,
            "metadata": _to_jsonable(self.metadata),
        }

    def __str__(self) -> str:  # pragma: no cover - trivial display hook
        return self.text


def effect_summary(
    result: Any,
    *,
    rope: Any = None,
    sesoi: Optional[float] = None,
    alpha: Optional[float] = None,
    direction: str = "two-sided",
) -> EffectSummary:
    """Build a decision-ready summary for a fitted causal result.

    Parameters
    ----------
    result : object
        A StatsPAI result object.  ``CausalResult`` and
        ``BayesianCausalResult`` are handled natively; any object exposing an
        ``effect_summary`` method is delegated to.
    rope, sesoi, alpha : optional
        Forwarded to frequentist ``CausalResult.decision_summary`` or used to
        describe Bayesian posterior ROPE probability when available.
    direction : {"increase", "decrease", "two-sided"}, default "two-sided"
        Directional posterior probability reported for Bayesian results.

    Returns
    -------
    EffectSummary
        A table-plus-text summary with JSON-safe ``to_dict`` support.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.CausalResult(
    ...     method="DID",
    ...     estimand="ATT",
    ...     estimate=2.0,
    ...     se=0.2,
    ...     pvalue=0.001,
    ...     ci=(1.6, 2.4),
    ...     alpha=0.05,
    ...     n_obs=1000,
    ... )
    >>> summary = sp.effect_summary(result, rope=0.5)
    >>> summary.kind
    'effect_summary'
    """
    own = getattr(result, "effect_summary", None)
    if callable(own) and getattr(own, "__self__", None) is result:
        return cast(
            EffectSummary,
            own(rope=rope, sesoi=sesoi, alpha=alpha, direction=direction),
        )

    if _looks_like_bayesian_causal_result(result):
        return bayesian_effect_summary(
            result, rope=rope, sesoi=sesoi, alpha=alpha, direction=direction
        )

    if hasattr(result, "decision_summary"):
        return causal_effect_summary(result, rope=rope, sesoi=sesoi, alpha=alpha)

    raise TypeError(
        "effect_summary() supports StatsPAI CausalResult, BayesianCausalResult, "
        "or objects exposing an effect_summary() method."
    )


def causal_effect_summary(
    result: Any,
    *,
    rope: Any = None,
    sesoi: Optional[float] = None,
    alpha: Optional[float] = None,
) -> EffectSummary:
    """Build an ``EffectSummary`` from ``CausalResult.decision_summary``."""
    decision = result.decision_summary(rope=rope, sesoi=sesoi, alpha=alpha)
    return EffectSummary(
        table=decision.table.copy(),
        text=decision.text,
        metadata={
            "source": "decision_summary",
            "verdict": decision.verdict,
            "label": decision.label,
            "statistically_significant": decision.statistically_significant,
            "practically_significant": decision.practically_significant,
            "ci_vs_rope": decision.ci_vs_rope,
        },
    )


def bayesian_effect_summary(
    result: Any,
    *,
    rope: Any = None,
    sesoi: Optional[float] = None,
    alpha: Optional[float] = None,
    direction: str = "two-sided",
) -> EffectSummary:
    """Build an ``EffectSummary`` from ``BayesianCausalResult`` fields."""
    if direction not in {"increase", "decrease", "two-sided"}:
        raise ValueError(
            "direction must be 'increase', 'decrease', or 'two-sided'; "
            f"got {direction!r}."
        )
    if rope is not None and sesoi is not None:
        raise ValueError("Pass either rope= or sesoi=, not both.")

    hdi_prob = float(getattr(result, "hdi_prob", 0.95))
    if alpha is not None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1); got {alpha!r}.")
        hdi_prob = 1 - float(alpha)

    mean = _as_float(getattr(result, "posterior_mean", None))
    median = _as_float(getattr(result, "posterior_median", None))
    sd = _as_float(getattr(result, "posterior_sd", None))
    hdi_low = _as_float(getattr(result, "hdi_lower", None))
    hdi_high = _as_float(getattr(result, "hdi_upper", None))
    prob_positive = _as_float(getattr(result, "prob_positive", None))
    prob_rope = _as_float(getattr(result, "prob_rope", None))

    if sesoi is not None and rope is None:
        rope = float(sesoi)
    rope_desc = _rope_description(rope)

    if direction == "increase":
        tail_label = f"P({result.estimand} > 0)"
        tail_prob = prob_positive
    elif direction == "decrease":
        tail_label = f"P({result.estimand} < 0)"
        tail_prob = 1 - prob_positive if prob_positive is not None else None
    else:
        tail_label = f"P({result.estimand} != 0)"
        tail_prob = (
            1 - 2 * min(prob_positive, 1 - prob_positive)
            if prob_positive is not None
            else None
        )

    table = pd.DataFrame(
        [
            {
                "method": getattr(result, "method", ""),
                "estimand": getattr(result, "estimand", ""),
                "posterior_mean": mean,
                "posterior_median": median,
                "posterior_sd": sd,
                "hdi_low": hdi_low,
                "hdi_high": hdi_high,
                "hdi_prob": hdi_prob,
                "prob_positive": prob_positive,
                "tail_probability": tail_prob,
                "tail_probability_label": tail_label,
                "prob_rope": prob_rope,
                "rope": rope_desc,
                "rhat": _as_float(getattr(result, "rhat", None)),
                "ess": _as_float(getattr(result, "ess", None)),
                "n_obs": getattr(result, "n_obs", None),
            }
        ]
    )

    pct = f"{100 * hdi_prob:g}%"
    text = (
        f"{getattr(result, 'method', 'Bayesian estimator')} estimates "
        f"{getattr(result, 'estimand', 'the effect')} at {_fmt(mean)} "
        f"({pct} HDI [{_fmt(hdi_low)}, {_fmt(hdi_high)}]). "
        f"{tail_label} = {_fmt(tail_prob, digits=3)}."
    )
    if prob_rope is not None:
        text += f" Posterior probability in the ROPE is {_fmt(prob_rope, digits=3)}."
    if rope_desc is not None and prob_rope is None:
        text += (
            " A ROPE was supplied, but this result does not store posterior "
            "ROPE mass; re-fit with rope= on the Bayesian estimator to report it."
        )

    return EffectSummary(
        table=table,
        text=text,
        metadata={
            "source": "bayesian_causal_result",
            "direction": direction,
            "tail_probability_label": tail_label,
        },
    )


def _looks_like_bayesian_causal_result(result: Any) -> bool:
    return all(
        hasattr(result, attr)
        for attr in (
            "posterior_mean",
            "posterior_median",
            "hdi_lower",
            "hdi_upper",
            "prob_positive",
        )
    )


def _as_float(value: Any) -> Optional[float]:
    if value is None or not isinstance(value, Real):
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def _fmt(value: Optional[float], *, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}g}"


def _rope_description(rope: Any) -> Any:
    if rope is None:
        return None
    if np.isscalar(rope):
        r = float(rope)
        return [-r, r]
    if isinstance(rope, (list, tuple)) and len(rope) == 2:
        lo, hi = rope
        return [float(lo), float(hi)]
    return str(rope)
