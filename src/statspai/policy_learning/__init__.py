"""
Policy Learning: Optimal treatment assignment from heterogeneous effects.

Learns an interpretable treatment assignment policy that maximises
the expected welfare (value) of the population. Given estimated CATE,
finds the optimal tree-based policy: "who should be treated?"

Components
----------
- **PolicyTree** : Optimal depth-limited decision tree for treatment
  assignment (Athey & Wager 2021).
- **policy_value** : Evaluate the expected value of a treatment policy
  using doubly robust scores.

References
----------
Athey, S. & Wager, S. (2021).
Policy Learning with Observational Data.
Econometrica, 89(1), 133-161.

Zhou, Z., Athey, S., & Wager, S. (2023).
Offline Multi-Action Policy Learning: Generalization and Optimization.
Operations Research, 71(1), 148-183.
"""

from .policy_tree import policy_tree, PolicyTree, policy_value

__all__ = [
    'policy_tree',
    'PolicyTree',
    'policy_value',
]
