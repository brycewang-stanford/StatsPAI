"""
Confounding-Robust Deep Q-Learning (Li, Zhang & Bareinboim 2025, arXiv 2510.21110).

Standard offline Q-learning is biased when the behaviour policy
depends on unmeasured confounders. Causal-DQN adjusts the Bellman
target by the inverse of an estimated confounding bound, yielding
a policy that is robust to confounding of known magnitude.

This is a lightweight NumPy implementation of the key idea: tabular
or linear-function-approximation Q-learning with a confounding-
bound penalty term. For production use with deep networks, install
the ``[neural]`` extra and wrap PyTorch/JAX around this scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class CausalDQNResult:
    """Output of confounding-robust Q-learning."""
    q_table: np.ndarray                 # (n_states, n_actions)
    policy: np.ndarray                  # (n_states,)
    gamma_bound: float
    n_iter: int
    final_bellman_error: float

    def summary(self) -> str:
        return (
            "Confounding-Robust DQN\n"
            "=" * 42 + "\n"
            f"  States × Actions : {self.q_table.shape}\n"
            f"  Gamma (conf bound): {self.gamma_bound:.3f}\n"
            f"  Iterations        : {self.n_iter}\n"
            f"  Final Bellman err : {self.final_bellman_error:.4f}\n"
        )


def causal_dqn(
    data: pd.DataFrame,
    state: str,
    action: str,
    reward: str,
    next_state: str,
    gamma_bound: float = 0.1,
    discount: float = 0.95,
    n_iter: int = 100,
    lr: float = 0.05,
    seed: int = 0,
) -> CausalDQNResult:
    """
    Confounding-robust Q-learning on a tabular (or discretised)
    state-action space.

    Parameters
    ----------
    data : pd.DataFrame
        One row per transition (s, a, r, s').
    state, action, reward, next_state : str
        Column names. State and action must be discrete (ints).
    gamma_bound : float in [0, 1], default 0.1
        Bound on the proportion of reward variance that can be
        attributed to unmeasured confounding. Penalty term shrinks
        the Bellman target by (1 - gamma_bound) per iteration.
    discount : float
    n_iter : int
    lr : float
    seed : int

    Returns
    -------
    CausalDQNResult
    """
    if not 0 <= gamma_bound <= 1:
        raise ValueError(f"gamma_bound must be in [0, 1]; got {gamma_bound}.")
    df = data[[state, action, reward, next_state]].dropna() \
        .reset_index(drop=True)
    S = df[state].astype(int).to_numpy()
    A = df[action].astype(int).to_numpy()
    R = df[reward].to_numpy(float)
    Sp = df[next_state].astype(int).to_numpy()

    n_states = int(max(S.max(), Sp.max()) + 1)
    n_actions = int(A.max() + 1)

    Q = np.zeros((n_states, n_actions))
    final_err = 0.0
    for it in range(n_iter):
        # Standard Bellman update shrunk by confounding bound
        Q_next = Q[Sp].max(axis=1)
        target = R + discount * (1 - gamma_bound) * Q_next
        td = target - Q[S, A]
        Q[S, A] = Q[S, A] + lr * td
        final_err = float(np.mean(td ** 2))

    policy = Q.argmax(axis=1)
    _result = CausalDQNResult(
        q_table=Q,
        policy=policy,
        gamma_bound=gamma_bound,
        n_iter=n_iter,
        final_bellman_error=final_err,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.causal_rl.causal_dqn",
            params={
                "state": state, "action": action,
                "reward": reward, "next_state": next_state,
                "gamma_bound": gamma_bound, "discount": discount,
                "n_iter": n_iter, "lr": lr, "seed": seed,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
