"""Agent adapters: map a Condition to something that runs a Task.

* ``OracleAdapter`` — deterministic StatsPAI reference pipeline, no LLM.
  Runs with the core deps alone; used for dry-runs, CI, and as an
  upper-bound calibration cell.
* ``LLMAdapter`` — drives a real Claude/GPT agent over one of the three
  stacks (statspai / pythonic / r). Requires the relevant SDK + API key.
"""

from .base import AgentAdapter
from .statspai_oracle import OracleAdapter

__all__ = ["AgentAdapter", "OracleAdapter", "build_adapter"]


def build_adapter(condition_code: str, **kwargs):
    """Factory: return the adapter that implements a condition cell.

    ``oracle`` -> OracleAdapter. C1..C6 -> LLMAdapter (imported lazily so
    the package works without the LLM SDKs installed).
    """
    from ..conditions import get

    cond = get(condition_code)
    if cond.stack == "oracle":
        return OracleAdapter(condition=cond, **kwargs)
    from .llm import LLMAdapter  # lazy: avoids hard dep on anthropic/openai

    return LLMAdapter(condition=cond, **kwargs)
