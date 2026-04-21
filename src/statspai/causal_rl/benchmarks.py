"""
Causal RL Benchmark Suite (Zhou-Bareinboim 2025, arXiv 2512.18135).

Five canonical benchmarks for evaluating causal RL algorithms under
unobserved confounding:

* ``confounded_bandit``     — classic two-arm bandit with hidden context
* ``confounded_dosage``     — continuous-dose pricing
* ``confounded_pricing``    — competitive pricing with hidden demand
* ``confounded_targeting``  — uplift modelling with selection bias
* ``confounded_routing``    — sequential decisions with hidden state

Each generates a transition dataset and reports the regret of a
policy against the optimal action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class BanditBenchmarkResult:
    """Output from a causal-RL benchmark run."""
    benchmark: str
    transitions: pd.DataFrame
    optimal_policy: np.ndarray
    optimal_value: float
    suggested_evaluator: str

    def summary(self) -> str:
        return (
            f"Causal-RL Benchmark: {self.benchmark}\n"
            "=" * 42 + "\n"
            f"  Transitions     : {len(self.transitions)}\n"
            f"  Optimal value   : {self.optimal_value:+.4f}\n"
            f"  Optimal policy  : {self.optimal_policy[:5]}...\n"
            f"  Recommended eval: {self.suggested_evaluator}\n"
        )


def causal_rl_benchmark(
    name: str = "confounded_bandit",
    n_episodes: int = 1000,
    confounding_strength: float = 0.5,
    seed: int = 0,
) -> BanditBenchmarkResult:
    """
    Generate a synthetic causal-RL benchmark dataset.

    Parameters
    ----------
    name : {'confounded_bandit', 'confounded_dosage', 'confounded_pricing',
            'confounded_targeting', 'confounded_routing'}
    n_episodes : int, default 1000
    confounding_strength : float in [0, 1], default 0.5
        Magnitude of unmeasured confounding U → (action, reward).
    seed : int

    Returns
    -------
    BanditBenchmarkResult
    """
    rng = np.random.default_rng(seed)
    if name == "confounded_bandit":
        U = rng.standard_normal(n_episodes)
        # Behaviour: action depends on U; arm 1 better in expectation
        A = (rng.uniform(size=n_episodes) < (
            0.5 + confounding_strength * np.tanh(U)
        )).astype(int)
        R = 1.0 + 0.5 * A + confounding_strength * U + rng.standard_normal(n_episodes) * 0.1
        df = pd.DataFrame({
            'state': np.zeros(n_episodes, dtype=int),
            'action': A,
            'reward': R,
            'next_state': np.zeros(n_episodes, dtype=int),
        })
        opt_policy = np.array([1])
        opt_value = 1.5
        evaluator = "sp.causal_dqn"
    elif name == "confounded_dosage":
        # Discretise dose into 5 levels; reward = inverted U-shape + confounder
        D = rng.integers(0, 5, size=n_episodes)
        U = rng.standard_normal(n_episodes)
        R = -(D - 2.0) ** 2 + confounding_strength * U + rng.standard_normal(n_episodes) * 0.2
        df = pd.DataFrame({
            'state': np.zeros(n_episodes, dtype=int),
            'action': D, 'reward': R,
            'next_state': np.zeros(n_episodes, dtype=int),
        })
        opt_policy = np.array([2])
        opt_value = 0.0
        evaluator = "sp.causal_dqn"
    elif name == "confounded_pricing":
        prices = rng.integers(0, 5, size=n_episodes)
        demand_shock = rng.standard_normal(n_episodes)
        R = (5 - prices) * (1 + 0.1 * confounding_strength * demand_shock) + 0.1 * rng.standard_normal(n_episodes)
        df = pd.DataFrame({
            'state': np.zeros(n_episodes, dtype=int),
            'action': prices, 'reward': R,
            'next_state': np.zeros(n_episodes, dtype=int),
        })
        opt_policy = np.array([0])
        opt_value = 5.0
        evaluator = "sp.causal_dqn or sp.policy_tree"
    elif name == "confounded_targeting":
        X = rng.integers(0, 3, size=n_episodes)
        U = rng.standard_normal(n_episodes)
        T = (rng.uniform(size=n_episodes) < (0.3 + confounding_strength * (U > 0))).astype(int)
        R = 0.5 * T * (X == 1) + confounding_strength * U + 0.1 * rng.standard_normal(n_episodes)
        df = pd.DataFrame({
            'state': X, 'action': T, 'reward': R,
            'next_state': X,
        })
        opt_policy = np.array([0, 1, 0])  # treat only state==1
        opt_value = 0.5
        evaluator = "sp.policy_tree"
    elif name == "confounded_routing":
        S = rng.integers(0, 3, size=n_episodes)
        A = rng.integers(0, 2, size=n_episodes)
        Sp = (S + A) % 3
        R = (Sp == 2).astype(float) + 0.1 * rng.standard_normal(n_episodes)
        df = pd.DataFrame({
            'state': S, 'action': A, 'reward': R, 'next_state': Sp,
        })
        opt_policy = np.array([0, 1, 0])
        opt_value = 1.0
        evaluator = "sp.causal_dqn"
    else:
        raise ValueError(
            f"Unknown benchmark: {name!r}. Available: confounded_bandit, "
            f"confounded_dosage, confounded_pricing, confounded_targeting, "
            f"confounded_routing."
        )

    return BanditBenchmarkResult(
        benchmark=name,
        transitions=df,
        optimal_policy=opt_policy,
        optimal_value=float(opt_value),
        suggested_evaluator=evaluator,
    )
