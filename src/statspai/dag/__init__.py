"""
DAG (Directed Acyclic Graph) module for causal reasoning.

Declare causal graphs, compute adjustment sets, check for collider bias,
and visualize causal structures — the Python equivalent of R's ``dagitty``
and ``ggdag``.

>>> import statspai as sp
>>> g = sp.dag('X -> Y; Z -> X; Z -> Y')
>>> g.adjustment_sets('X', 'Y')
[{'Z'}]
>>> g.is_collider('M', ['X', 'Y'])
>>> g.plot()
"""

from .graph import DAG, dag

__all__ = ["DAG", "dag"]
