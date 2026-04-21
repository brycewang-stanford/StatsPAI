"""
Core causal-RL primitives added in v1.0.

Three building blocks on top of :mod:`statspai.causal_rl`:

- :func:`causal_bandit` — Bareinboim-Pearl contextual causal bandit.
- :func:`counterfactual_policy_optimization` — Oberst-Sontag 2019 style
  counterfactual policy evaluation via SCM noise inversion (Gaussian
  linear special case).
- :func:`structural_mdp` — fit a linear SVAR from logged trajectories
  and support counterfactual rollouts under alternative policies.

These complement the existing :func:`causal_dqn`,
:func:`causal_rl_benchmark`, and :func:`offline_safe_policy`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


__all__ = [
    "causal_bandit",
    "counterfactual_policy_optimization",
    "structural_mdp",
    "CausalBanditResult",
    "CFPolicyResult",
    "StructuralMDPResult",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class CausalBanditResult:
    optimal_arm: int
    expected_rewards: np.ndarray
    arm_labels: List[str]
    context: Optional[dict]

    def summary(self) -> str:
        lines = [
            "Causal Contextual Bandit — Optimal Arm Recommendation",
            "=" * 60,
            f"  Optimal arm       : {self.arm_labels[self.optimal_arm]}",
            f"  Expected rewards  :",
        ]
        for a, r in zip(self.arm_labels, self.expected_rewards):
            marker = " *" if a == self.arm_labels[self.optimal_arm] else ""
            lines.append(f"    {a:<12s} : {r:+.6f}{marker}")
        if self.context:
            lines.append(f"  Context           : {self.context}")
        return "\n".join(lines)


@dataclass
class CFPolicyResult:
    expected_value_logged: float
    expected_value_target: float
    improvement: float
    n_trajectories: int

    def summary(self) -> str:
        return "\n".join([
            "Counterfactual Policy Optimisation",
            "=" * 60,
            f"  E[V | logged policy]  : {self.expected_value_logged:+.6f}",
            f"  E[V | target policy]  : {self.expected_value_target:+.6f}",
            f"  Improvement           : {self.improvement:+.6f}",
            f"  # trajectories        : {self.n_trajectories}",
        ])


@dataclass
class StructuralMDPResult:
    state_dim: int
    action_dim: int
    A: np.ndarray  # state transition: s_{t+1} = A s_t + B a_t + noise
    B: np.ndarray
    reward_coef: np.ndarray  # r_t = coef_s @ s_t + coef_a @ a_t

    def counterfactual_rollout(
        self,
        initial_state: np.ndarray,
        policy: Callable[[np.ndarray], np.ndarray],
        horizon: int = 10,
    ) -> Dict[str, np.ndarray]:
        """Roll out the fitted SVAR under a new ``policy`` to get a
        counterfactual (state, action, reward) trajectory.
        """
        s = np.array(initial_state, dtype=float).reshape(self.state_dim)
        states, actions, rewards = [s.copy()], [], []
        for _ in range(horizon):
            a = np.array(policy(s), dtype=float).reshape(self.action_dim)
            r = float(self.reward_coef @ np.concatenate([s, a]))
            s = self.A @ s + self.B @ a
            states.append(s.copy())
            actions.append(a)
            rewards.append(r)
        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
        }


# ---------------------------------------------------------------------------
# 1) Causal bandit
# ---------------------------------------------------------------------------


def causal_bandit(
    arms: Sequence[str],
    *,
    reward_fn: Callable[[str, dict], float],
    context: Optional[dict] = None,
    n_samples: int = 500,
    rng_seed: int = 0,
) -> CausalBanditResult:
    """Bareinboim-Pearl contextual causal bandit.

    Given a callable ``reward_fn(arm, context)`` that samples the
    potential outcome of an arm under the current context, Monte Carlo
    estimates ``E[Y(a) | context]`` for each arm and returns the
    argmax.

    Parameters
    ----------
    arms : sequence of str
        Arm labels.
    reward_fn : callable
        Stochastic reward sampler. Must accept (arm, context) and return
        a scalar reward.
    context : dict, optional
    n_samples : int, default 500
        Monte Carlo draws per arm.
    rng_seed : int, default 0

    Returns
    -------
    CausalBanditResult
    """
    if len(arms) < 2:
        raise ValueError("Need >= 2 arms.")
    rng = np.random.default_rng(rng_seed)
    expected = np.zeros(len(arms))
    for i, a in enumerate(arms):
        draws = np.array([
            reward_fn(a, context) for _ in range(n_samples)
        ])
        expected[i] = float(draws.mean())
    best = int(expected.argmax())
    return CausalBanditResult(
        optimal_arm=best,
        expected_rewards=expected,
        arm_labels=list(arms),
        context=dict(context) if context else None,
    )


# ---------------------------------------------------------------------------
# 2) Counterfactual policy optimisation
# ---------------------------------------------------------------------------


def counterfactual_policy_optimization(
    data: pd.DataFrame,
    *,
    state: str,
    action: str,
    reward: str,
    target_policy: Callable[[float], float],
    noise_sd: float = 1.0,
) -> CFPolicyResult:
    """Counterfactual policy evaluation under a linear-Gaussian SCM.

    Assumes a one-step SCM

        r = alpha * s + beta * a + eps,  eps ~ Normal(0, noise_sd²)

    so that fixing ``s`` and changing ``a`` uniquely determines a new
    reward via noise inversion.

    Parameters
    ----------
    data : DataFrame
        One row per trajectory; must contain numeric ``state``,
        ``action``, and ``reward`` columns.
    state, action, reward : str
    target_policy : callable(float) -> float
        Proposed policy ``a_new = π(s)``.
    noise_sd : float, default 1.0

    Returns
    -------
    CFPolicyResult
    """
    cols = {state, action, reward}
    missing = cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    s = data[state].to_numpy(dtype=float)
    a = data[action].to_numpy(dtype=float)
    r = data[reward].to_numpy(dtype=float)
    # Fit linear model r ~ s + a
    X = np.column_stack([np.ones_like(s), s, a])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    intercept, alpha, beta_a = beta
    # Residual / "noise" per trajectory (for noise inversion).
    eps = r - (intercept + alpha * s + beta_a * a)
    # Counterfactual reward under the target policy, keeping eps fixed.
    a_new = np.array([target_policy(si) for si in s], dtype=float)
    r_cf = intercept + alpha * s + beta_a * a_new + eps
    logged_value = float(r.mean())
    target_value = float(r_cf.mean())
    return CFPolicyResult(
        expected_value_logged=logged_value,
        expected_value_target=target_value,
        improvement=target_value - logged_value,
        n_trajectories=int(len(data)),
    )


# ---------------------------------------------------------------------------
# 3) Structural MDP
# ---------------------------------------------------------------------------


def structural_mdp(
    data: pd.DataFrame,
    *,
    state_cols: Sequence[str],
    action_cols: Sequence[str],
    reward: str,
    next_state_cols: Optional[Sequence[str]] = None,
    time: Optional[str] = None,
    trajectory: Optional[str] = None,
) -> StructuralMDPResult:
    """Fit a linear SVAR for a Markov decision process.

    Estimates:

        s_{t+1} = A s_t + B a_t + noise
        r_t     = coef_s @ s_t + coef_a @ a_t

    from logged tuples. Supports per-trajectory data (``trajectory``
    column groups consecutive transitions) or single-stream data with a
    ``time`` column.

    Parameters
    ----------
    data : DataFrame
    state_cols, action_cols : sequence of str
    reward : str
    next_state_cols : sequence of str, optional
        If present, each row is a complete (s, a, r, s') tuple. If
        omitted, the function derives ``s_{t+1}`` by shifting ``s`` within
        each trajectory.
    time : str, optional
        Required if ``next_state_cols`` is None — used to order rows
        within each trajectory.
    trajectory : str, optional
        Trajectory identifier for multi-episode data.

    Returns
    -------
    StructuralMDPResult
    """
    state_cols = list(state_cols)
    action_cols = list(action_cols)
    required = set(state_cols) | set(action_cols) | {reward}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = data.copy()
    if next_state_cols is not None:
        # Explicit next-state columns
        ns_cols = list(next_state_cols)
        if set(ns_cols) - set(df.columns):
            raise ValueError(
                f"Missing next-state columns: {set(ns_cols) - set(df.columns)}"
            )
    else:
        if time is None:
            raise ValueError(
                "Either `next_state_cols` or `time` must be provided."
            )
        # Construct next-state columns within each trajectory.
        group_cols = [trajectory] if trajectory is not None else []
        df = df.sort_values(group_cols + [time])
        if trajectory is None:
            ns = df[state_cols].shift(-1)
        else:
            ns = df.groupby(trajectory)[state_cols].shift(-1)
        ns_cols = [f"__next_{c}" for c in state_cols]
        ns.columns = ns_cols
        df = pd.concat([df.reset_index(drop=True), ns.reset_index(drop=True)], axis=1)
        df = df.dropna(subset=ns_cols).reset_index(drop=True)

    S = df[state_cols].to_numpy(dtype=float)
    A = df[action_cols].to_numpy(dtype=float)
    Sn = df[ns_cols].to_numpy(dtype=float)
    r = df[reward].to_numpy(dtype=float)
    n = len(df)
    if n < len(state_cols) + len(action_cols) + 5:
        raise ValueError(
            f"Too few transitions (n={n}) for SVAR identification."
        )

    # s_{t+1} = A_mat @ s_t + B_mat @ a_t
    #   X has shape (n, ds + da); Sn has shape (n, ds).  lstsq(X, Sn)
    #   returns AB of shape (ds + da, ds) satisfying S_next = X @ AB.
    #   For the column-vector rollout we need:
    #       s_next[j] = sum_k A_mat[j, k] * s[k] + sum_k B_mat[j, k] * a[k]
    #   Expanding S_next = X @ AB elementwise gives
    #       s_next[j] = sum_k s[k] * AB[k, j] + sum_k a[k] * AB[ds+k, j]
    #   so A_mat[j, k] = AB[k, j], i.e. A_mat = AB[:ds, :].T and
    #   B_mat = AB[ds:, :].T.  The assertions below pin this invariant
    #   so any future refactor that changes the slice semantics will
    #   fail loudly instead of silently returning a transposed matrix.
    X = np.column_stack([S, A])  # n × (ds + da)
    AB, *_ = np.linalg.lstsq(X, Sn, rcond=None)
    ds = len(state_cols)
    da = len(action_cols)
    A_mat = AB[:ds, :].T  # shape (ds, ds): s_next = A_mat @ s
    B_mat = AB[ds:, :].T  # shape (ds, da): s_next += B_mat @ a
    assert A_mat.shape == (ds, ds), (
        f"A_mat shape mismatch: expected {(ds, ds)}, got {A_mat.shape}"
    )
    assert B_mat.shape == (ds, da), (
        f"B_mat shape mismatch: expected {(ds, da)}, got {B_mat.shape}"
    )

    # r_t = coef @ [s_t; a_t]
    coef, *_ = np.linalg.lstsq(X, r, rcond=None)
    return StructuralMDPResult(
        state_dim=ds,
        action_dim=len(action_cols),
        A=A_mat,
        B=B_mat,
        reward_coef=coef,
    )
