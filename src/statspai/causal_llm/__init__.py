"""
LLM × Causal Inference (StatsPAI v0.10).

Three integration points where large language models help causal
analysis without replacing the formal estimator:

* :func:`llm_dag_propose` — propose candidate DAGs from variable
  names + domain description (Kiciman-Sharma 2025, arXiv 2402.11068).
* :func:`llm_unobserved_confounders` — generate plausible unobserved
  confounder candidates for E-value sensitivity analysis
  (arXiv 2603.14273).
* :func:`llm_sensitivity_priors` — propose Cornfield-style sensitivity
  parameter priors based on the substantive context.

All three are **offline** by default — they ship with deterministic
heuristic backends so they work without an API key. If a real LLM
client (OpenAI / Anthropic / local) is available via the optional
``[llm]`` extra, set the ``client`` keyword argument.

The deterministic backends are designed to be **transparent**: they
return reproducible candidates derived from variable-name pattern
matching and domain heuristics, not silent fabrications.
"""

from .llm_dag import llm_dag_propose, LLMDAGProposal
from .llm_evalue import (
    llm_unobserved_confounders, UnobservedConfounderProposal,
)
from .llm_sensitivity import (
    llm_sensitivity_priors, SensitivityPriorProposal,
)
from .causal_mas import causal_mas, CausalMASResult
from .llm_clients import (
    LLMClient, openai_client, anthropic_client, echo_client,
)

__all__ = [
    'llm_dag_propose', 'LLMDAGProposal',
    'llm_unobserved_confounders', 'UnobservedConfounderProposal',
    'llm_sensitivity_priors', 'SensitivityPriorProposal',
    'causal_mas', 'CausalMASResult',
    'LLMClient', 'openai_client', 'anthropic_client', 'echo_client',
]
