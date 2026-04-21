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
>>> g.bad_controls('X', 'Y')
>>> g.summary('X', 'Y')
>>> g.do('X')  # interventional graph
>>> sp.dag_example('discrimination')  # classic textbook DAG
"""

from .graph import DAG, dag, dag_example, dag_examples, dag_example_positions, dag_simulate
from .identification import identify, IdentificationResult
from .do_calculus import rule1, rule2, rule3, apply_rules, RuleCheck
from .swig import swig, SWIGGraph
from .counterfactual import SCM
from .llm_dag import llm_dag, LLMDAGResult

__all__ = [
    "DAG", "dag", "dag_example", "dag_examples",
    "dag_example_positions", "dag_simulate",
    "identify", "IdentificationResult",
    "rule1", "rule2", "rule3", "apply_rules", "RuleCheck",
    "swig", "SWIGGraph",
    "SCM",
    "llm_dag", "LLMDAGResult",
]
