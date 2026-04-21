"""
Causal Reinforcement Learning (StatsPAI v0.10).

Bridges between RL and causal inference for offline / batch learning
scenarios with unobserved confounding.

References
----------
* Fu-Zhou (2025), arXiv 2510.21110 — Confounding-Robust Deep RL.
* Zhou-Bareinboim (2025), arXiv 2512.18135 — Unifying Causal RL.
* Chemingui et al. (2025), arXiv 2510.22027 — Online Optimization
  for Offline Safe RL.
"""

from .causal_dqn import causal_dqn, CausalDQNResult
from .benchmarks import causal_rl_benchmark, BanditBenchmarkResult
from .offline_safe import offline_safe_policy, OfflineSafeResult

__all__ = [
    'causal_dqn', 'CausalDQNResult',
    'causal_rl_benchmark', 'BanditBenchmarkResult',
    'offline_safe_policy', 'OfflineSafeResult',
]
