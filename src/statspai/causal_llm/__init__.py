"""
LLM × Causal Inference (StatsPAI v0.10).

Three integration points where large language models help causal
analysis without replacing the formal estimator:

* :func:`llm_dag_propose` — propose candidate DAGs from variable
  names + domain description (kiciman2023causal, arXiv 2305.00050).
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

from ._config import DEFAULT_MODELS as DEFAULT_LLM_MODELS
from ._config import config_path as llm_config_path
from ._config import load_config as load_llm_config
from ._resolver import (
    LLMConfigurationError,
    configure_llm,
    get_llm_client,
    list_available_providers,
)
from .causal_mas import CausalMASResult, causal_mas
from .llm_clients import LLMClient, anthropic_client, echo_client, openai_client
from .llm_dag import LLMDAGProposal, llm_dag_propose
from .llm_dag_loop import (
    DAGValidationResult,
    LLMConstrainedDAGResult,
    llm_dag_constrained,
    llm_dag_validate,
)
from .llm_evalue import UnobservedConfounderProposal, llm_unobserved_confounders
from .llm_sensitivity import SensitivityPriorProposal, llm_sensitivity_priors

__all__ = [
    "llm_dag_propose",
    "LLMDAGProposal",
    "llm_unobserved_confounders",
    "UnobservedConfounderProposal",
    "llm_sensitivity_priors",
    "SensitivityPriorProposal",
    "causal_mas",
    "CausalMASResult",
    "LLMClient",
    "openai_client",
    "anthropic_client",
    "echo_client",
    "llm_dag_constrained",
    "llm_dag_validate",
    "LLMConstrainedDAGResult",
    "DAGValidationResult",
    # v1.7.2: layered config + resolver.
    "get_llm_client",
    "list_available_providers",
    "configure_llm",
    "LLMConfigurationError",
    "llm_config_path",
    "load_llm_config",
    "DEFAULT_LLM_MODELS",
]
