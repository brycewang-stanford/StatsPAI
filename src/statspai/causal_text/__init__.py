"""Causal inference with text data — MVP (P1-B, v1.6 experimental).

Two methods to start the family:

* :func:`text_treatment_effect` — Veitch-Wang-Blei (2020) text-as-
  treatment via embedding adjustment.  Estimates the ATE of a
  user-supplied treatment indicator while controlling for the
  confounding variation captured in the text embedding.

* :func:`llm_annotator_correct` — Egami-Hinck-Stewart-Wei (2024)
  measurement-error correction for LLM-derived treatment labels.
  Given a small validation subset where both LLM and human labels
  exist, debias the downstream regression coefficient via Hausman-
  style correction.

Status
------
**Experimental.**  Both estimators ship with conservative defaults and
validation-set-estimated noise parameters; consume them as a starting
point, not a final answer.  Future versions will add text-as-confounder
(Roberts-Stewart-Nielsen) and text-as-outcome (Egami et al. 2018)
methods alongside topic-model integration.

References
----------
- Veitch, V., Wang, Y., & Blei, D. M. (2020). "Adapting text embeddings
  for causal inference."  *UAI*.  arXiv:1905.12741.
- Egami, N., Hinck, M., Stewart, B., & Wei, H. (2024). "Using
  imperfect surrogates for downstream inference: Design-based
  supervised learning for social science applications of large
  language models."  *NeurIPS*.  arXiv:2306.04746.
"""
from .text_treatment import (
    text_treatment_effect, TextTreatmentResult,
)
from .llm_annotator import (
    llm_annotator_correct, LLMAnnotatorResult,
)

__all__ = [
    'text_treatment_effect', 'TextTreatmentResult',
    'llm_annotator_correct', 'LLMAnnotatorResult',
]
