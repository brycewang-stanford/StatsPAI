"""
Online Optimization for Offline Safe RL (Chemingui et al. 2025,
arXiv 2510.22027).

Offline-learned policies can be unsafe if the behaviour policy
rarely visited certain states. Safe offline RL projects the learned
policy onto a feasible set defined by a cost-constraint threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OfflineSafeResult:
    """Output of safe offline policy learning.

    Returned by :func:`offline_safe_policy`. Holds the per-state action
    table (``policy``), the policy's expected reward and cost, the cost
    threshold it was constrained against, and whether the realised cost
    stays under that threshold (``feasible``).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 600
    >>> df = pd.DataFrame({
    ...     "state": rng.integers(0, 3, n),
    ...     "action": rng.integers(0, 2, n),
    ...     "reward": rng.integers(0, 2, n) * 1.0 + rng.normal(0, 0.5, n),
    ...     "cost": rng.integers(0, 2, n) * 0.3 + rng.normal(0, 0.1, n),
    ... })
    >>> res = sp.offline_safe_policy(df, state="state", action="action",
    ...                              reward="reward", cost="cost",
    ...                              cost_threshold=0.5)
    >>> isinstance(res, sp.OfflineSafeResult)
    True
    >>> bool(res.feasible)
    True
    >>> bool(res.expected_cost <= res.cost_threshold)
    True
    """

    policy: np.ndarray
    expected_reward: float
    expected_cost: float
    cost_threshold: float
    feasible: bool

    def summary(self) -> str:
        status = "FEASIBLE" if self.feasible else "INFEASIBLE"
        return (
            "Offline Safe Policy\n"
            "=" * 42 + "\n"
            f"  Status          : {status}\n"
            f"  Expected reward : {self.expected_reward:+.4f}\n"
            f"  Expected cost   : {self.expected_cost:.4f} "
            f"(threshold {self.cost_threshold:.4f})\n"
            f"  Policy (first 5): {self.policy[:5]}\n"
        )


def offline_safe_policy(
    data: pd.DataFrame,
    state: str,
    action: str,
    reward: str,
    cost: str,
    cost_threshold: float = 0.5,
    discount: float = 0.95,
    n_iter: int = 100,
    seed: int = 0,
) -> OfflineSafeResult:
    """
    Safe offline policy learning with a cost-constraint.

    Parameters
    ----------
    data : pd.DataFrame
        Transition data (s, a, r, cost).
    state, action, reward, cost : str
        Column names. state and action must be discrete.
    cost_threshold : float, default 0.5
        Max allowed expected cost per step.
    discount : float
    n_iter : int
    seed : int

    Returns
    -------
    OfflineSafeResult

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 600
    >>> df = pd.DataFrame({
    ...     "state": rng.integers(0, 3, n),
    ...     "action": rng.integers(0, 2, n),
    ...     "reward": rng.integers(0, 2, n) * 1.0 + rng.normal(0, 0.5, n),
    ...     "cost": rng.integers(0, 2, n) * 0.3 + rng.normal(0, 0.1, n),
    ... })
    >>> res = sp.offline_safe_policy(df, state="state", action="action",
    ...                              reward="reward", cost="cost",
    ...                              cost_threshold=0.5)
    >>> int(res.policy.shape[0])      # one action per visited state
    3
    >>> bool(res.feasible)
    True
    """
    df = data[[state, action, reward, cost]].dropna().reset_index(drop=True)
    S = df[state].astype(int).to_numpy()
    A = df[action].astype(int).to_numpy()
    R = df[reward].to_numpy(float)
    C = df[cost].to_numpy(float)

    n_states = int(S.max() + 1)
    n_actions = int(A.max() + 1)

    # Estimate Q_reward and Q_cost per (s, a) by simple averaging
    Q_r = np.zeros((n_states, n_actions))
    Q_c = np.zeros((n_states, n_actions))
    counts: np.ndarray = np.zeros((n_states, n_actions))
    for s, a, r, c in zip(S, A, R, C):
        Q_r[s, a] += r
        Q_c[s, a] += c
        counts[s, a] += 1
    counts = np.where(counts > 0, counts, 1.0)
    Q_r /= counts
    Q_c /= counts

    # Safe policy: argmax Q_r subject to Q_c ≤ threshold
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        # Mask infeasible actions
        mask = Q_c[s] <= cost_threshold
        if mask.any():
            # Best feasible
            q = np.where(mask, Q_r[s], -np.inf)
            policy[s] = int(np.argmax(q))
        else:
            # Fall back to cheapest
            policy[s] = int(np.argmin(Q_c[s]))

    # Evaluate policy
    exp_r = float(np.mean([Q_r[s, policy[s]] for s in range(n_states)]))
    exp_c = float(np.mean([Q_c[s, policy[s]] for s in range(n_states)]))
    feasible = exp_c <= cost_threshold

    return OfflineSafeResult(
        policy=policy,
        expected_reward=exp_r,
        expected_cost=exp_c,
        cost_threshold=float(cost_threshold),
        feasible=feasible,
    )
