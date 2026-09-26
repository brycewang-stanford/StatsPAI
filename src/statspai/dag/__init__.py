"""
DAG (Directed Acyclic Graph) module for causal reasoning.

Declare causal graphs, compute adjustment sets, check for collider bias,
enumerate paths, detect bad controls, and visualize causal structures —
the Python equivalent of R's ``dagitty`` and ``ggdag``.

>>> import statspai as sp
>>> g = sp.dag('X -> Y; Z -> X; Z -> Y')
>>> g.adjustment_sets('X', 'Y')
[{'Z'}]
>>> g.backdoor_paths('X', 'Y')
[['X', 'Z', 'Y']]
>>> g.bad_controls('X', 'Y')
{}
>>> print(g.summary('X', 'Y').splitlines()[0])
DAG Summary: effect of X on Y
>>> g.do('X')  # interventional graph
DAG(3 nodes, 2 edges)
>>> sp.dag_example('discrimination')  # classic textbook DAG
DAG(4 nodes, 5 edges)
"""

from __future__ import annotations

from collections.abc import Sequence

from .counterfactual import SCM
from .do_calculus import RuleCheck, apply_rules, rule1, rule2, rule3
from .graph import (
    DAG,
    dag,
    dag_example,
    dag_example_positions,
    dag_examples,
    dag_simulate,
)
from .identification import IdentificationResult, identify
from .llm_dag import LLMDAGResult, llm_dag
from .llm_evaluator import (
    LLMCausalAssessResult,
    PairwiseBenchmarkResult,
    llm_causal_assess,
    pairwise_causal_benchmark,
)
from .recommend import EstimatorRecommendation, recommend_estimator
from .swig import SWIGGraph, swig


# Attach recommend_estimator as a DAG method for the fluent API.
def _dag_recommend_estimator(
    self: DAG,
    exposure: str,
    outcome: str,
    candidate_instruments: Sequence[str] | None = None,
) -> EstimatorRecommendation:
    """See :func:`statspai.dag.recommend_estimator`."""
    return recommend_estimator(
        self,
        exposure,
        outcome,
        candidate_instruments=candidate_instruments,
    )


setattr(DAG, "recommend_estimator", _dag_recommend_estimator)

__all__ = [
    "DAG",
    "dag",
    "dag_example",
    "dag_examples",
    "dag_example_positions",
    "dag_simulate",
    "identify",
    "IdentificationResult",
    "rule1",
    "rule2",
    "rule3",
    "apply_rules",
    "RuleCheck",
    "swig",
    "SWIGGraph",
    "SCM",
    "llm_dag",
    "LLMDAGResult",
    "llm_causal_assess",
    "pairwise_causal_benchmark",
    "LLMCausalAssessResult",
    "PairwiseBenchmarkResult",
    "recommend_estimator",
    "EstimatorRecommendation",
]
