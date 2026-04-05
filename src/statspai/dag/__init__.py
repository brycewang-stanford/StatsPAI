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

__all__ = ["DAG", "dag", "dag_example", "dag_examples", "dag_example_positions", "dag_simulate"]
