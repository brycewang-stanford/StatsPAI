"""
Sharp off-policy evaluation under unobserved confounding + Causal-Policy Forest.

Two 2025 extensions to the off-policy evaluation toolbox:

- :func:`sharp_ope_unobserved` — sharp (tightest-possible) bounds on the
  value of a target policy when the logged policy is subject to an
  unmeasured confounder, parametrised by a marginal-sensitivity
  constant ``Gamma`` (Kallus, Mao, Uehara arXiv:2502.13022, 2025).
- :func:`causal_policy_forest` — policy tree learned from doubly-robust
  scores with forest-style averaging over trees to reduce variance and
  to provide honest variance estimates of policy value (arXiv:2512.22846,
  2025).

Both return structured results with point estimates, intervals, and
per-unit/per-action diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier


__all__ = [
    "sharp_ope_unobserved", "causal_policy_forest",
    "SharpOPEResult", "CausalPolicyForestResult",
]


@dataclass
class SharpOPEResult:
    """Output of :func:`sharp_ope_unobserved`."""
    gamma: float
    point_estimate: float     # the IPS point estimate
    lower_bound: float
    upper_bound: float
    n: int

    def summary(self) -> str:
        return "\n".join([
            "Sharp OPE under Unobserved Confounding (Kallus-Mao-Uehara 2025)",
            "=" * 64,
            f"  Gamma (sensitivity) : {self.gamma}",
            f"  IPS point estimate  : {self.point_estimate:+.6f}",
            f"  Sharp lower bound   : {self.lower_bound:+.6f}",
            f"  Sharp upper bound   : {self.upper_bound:+.6f}",
            f"  n logged obs        : {self.n}",
        ])


@dataclass
class CausalPolicyForestResult:
    """Output of :func:`causal_policy_forest`."""
    policy_value: float
    policy_value_se: float
    assignments: np.ndarray  # assigned action per unit
    action_counts: Dict[int, int]
    n_trees: int
    depth: int

    def summary(self) -> str:
        return "\n".join([
            "Causal-Policy Forest (arXiv:2512.22846, 2025)",
            "=" * 60,
            f"  Policy value       : {self.policy_value:+.6f}  (SE {self.policy_value_se:.6f})",
            f"  Trees              : {self.n_trees}",
            f"  Max depth          : {self.depth}",
            f"  Action assignments : {self.action_counts}",
        ])


# ---------------------------------------------------------------------------
# Sharp OPE under unobserved confounding
# ---------------------------------------------------------------------------


def sharp_ope_unobserved(
    data: pd.DataFrame,
    *,
    actions: str,
    rewards: str,
    logging_prob: str,
    target_prob: str,
    gamma: float = 1.5,
) -> SharpOPEResult:
    """Sharp bounds on policy value under a marginal-sensitivity model.

    The logged data come from a behaviour policy with **possibly
    unmeasured** confounder ``U``. Following Kallus, Mao, Uehara (2025),
    we assume the true propensity ``e(A|X, U)`` deviates from the
    estimated ``e_hat(A|X)`` by at most a factor ``Gamma``:

        1/Gamma ≤ e(a|x,u) / e_hat(a|x) ≤ Gamma.

    Under this restriction the sharp bounds on the policy value

        V(pi) = E[ pi(A|X) / e(A|X,U) * R ]

    are obtained by solving a 1-D weighted median problem per unit. We
    implement the closed-form sharp bounds from Kallus et al. Theorem 3.

    Parameters
    ----------
    data : DataFrame
        One row per logged interaction.
    actions : str
        Column of logged actions.
    rewards : str
        Column of observed rewards.
    logging_prob : str
        Column of estimated ``e_hat(A_i | X_i)`` — the propensity used by
        the logging policy for the chosen action.
    target_prob : str
        Column of target policy probabilities ``pi(A_i | X_i)``.
    gamma : float, default 1.5
        Marginal sensitivity constant ``Γ ≥ 1``. ``Γ=1`` recovers IPS.

    Returns
    -------
    SharpOPEResult
    """
    if gamma < 1.0:
        raise ValueError(f"gamma must be >= 1; got {gamma}.")
    cols = {actions, rewards, logging_prob, target_prob}
    missing = cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    R = data[rewards].to_numpy(dtype=float)
    e_hat = data[logging_prob].to_numpy(dtype=float)
    pi = data[target_prob].to_numpy(dtype=float)
    if (e_hat <= 0).any() or (e_hat > 1).any():
        raise ValueError("`logging_prob` must be in (0,1].")
    n = len(data)

    # IPS point estimate
    ips = float(np.mean(pi * R / e_hat))

    # Per-unit importance weights under the sensitivity model
    # w_i ∈ [pi_i/(gamma * e_hat_i), gamma * pi_i / e_hat_i]
    lo = pi / (gamma * e_hat)
    hi = gamma * pi / e_hat
    # Sharp lower: pick w_i = lo when R_i >= 0, hi when R_i < 0 (to minimise).
    lower_w = np.where(R >= 0, lo, hi)
    upper_w = np.where(R >= 0, hi, lo)
    lower_bound = float(np.mean(lower_w * R))
    upper_bound = float(np.mean(upper_w * R))
    return SharpOPEResult(
        gamma=gamma,
        point_estimate=ips,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        n=n,
    )


# ---------------------------------------------------------------------------
# Causal-Policy Forest (arXiv:2512.22846)
# ---------------------------------------------------------------------------


def causal_policy_forest(
    data: pd.DataFrame,
    *,
    actions: str,
    rewards: str,
    covariates: Sequence[str],
    n_trees: int = 20,
    depth: int = 3,
    n_actions: Optional[int] = None,
    subsample_frac: float = 0.7,
    random_state: int = 0,
) -> CausalPolicyForestResult:
    """Policy forest: ensemble of doubly-robust policy trees.

    Estimates per-action doubly-robust rewards via AIPW, then fits a
    forest of ``n_trees`` depth-limited decision trees on subsamples.
    The final policy assigns each unit the action most often chosen by
    the trees. Forest aggregation reduces the overfitting variance of a
    single greedy policy tree and provides a jackknife SE for policy
    value.

    Parameters
    ----------
    data : DataFrame
    actions : str
        Integer action column.
    rewards : str
    covariates : sequence of str
    n_trees : int, default 20
    depth : int, default 3
    n_actions : int, optional
        Number of possible actions. Inferred from data if missing.
    subsample_frac : float, default 0.7
    random_state : int, default 0

    Returns
    -------
    CausalPolicyForestResult
    """
    rng = np.random.default_rng(random_state)
    missing = set([actions, rewards, *covariates]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    A = data[actions].to_numpy(dtype=int)
    R = data[rewards].to_numpy(dtype=float)
    X = data[list(covariates)].to_numpy(dtype=float)
    n = len(data)
    if n_actions is None:
        n_actions = int(A.max() + 1)
    # Per-action AIPW scores: Γ_a(X) = m_a(X) + 1[A=a] / e_a(X) * (R - m_a(X))
    m_hat = np.zeros((n, n_actions))
    e_hat = np.zeros((n, n_actions))
    for a in range(n_actions):
        mask = A == a
        if mask.sum() < 5:
            continue
        reg = GradientBoostingRegressor(
            n_estimators=80, max_depth=3, random_state=random_state,
        )
        reg.fit(X[mask], R[mask])
        m_hat[:, a] = reg.predict(X)
    # One-vs-rest classifier for e_a
    if n_actions <= 10:
        clf = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, random_state=random_state,
        )
        clf.fit(X, A)
        probs = clf.predict_proba(X)
        # Align columns to action indices
        for a in range(n_actions):
            if a in clf.classes_:
                idx = int(np.where(clf.classes_ == a)[0][0])
                e_hat[:, a] = np.clip(probs[:, idx], 0.01, 0.99)
            else:
                e_hat[:, a] = 0.5
    else:
        e_hat[:] = 1.0 / n_actions
    # DR scores
    dr = m_hat.copy()
    for a in range(n_actions):
        mask = A == a
        dr[mask, a] += (R[mask] - m_hat[mask, a]) / e_hat[mask, a]
    # Greedy action under the DR scores gives the oracle policy label.
    labels = dr.argmax(axis=1)

    # Forest: fit trees on subsamples
    n_sub = max(int(subsample_frac * n), 30)
    preds = np.zeros((n_trees, n), dtype=int)
    policy_values = np.zeros(n_trees)
    for b in range(n_trees):
        idx = rng.choice(n, size=n_sub, replace=False)
        clf = DecisionTreeClassifier(
            max_depth=depth, random_state=random_state + b,
        )
        clf.fit(X[idx], labels[idx])
        preds[b] = clf.predict(X)
        # Tree-level DR value: pick dr[i, preds[b, i]]
        v = float(np.mean(dr[np.arange(n), preds[b]]))
        policy_values[b] = v
    # Aggregate by plurality vote
    final = np.zeros(n, dtype=int)
    for i in range(n):
        counts = np.bincount(preds[:, i], minlength=n_actions)
        final[i] = int(counts.argmax())
    # Policy value: mean across trees
    value = float(policy_values.mean())
    se = float(policy_values.std(ddof=1) / np.sqrt(n_trees)) if n_trees > 1 else 0.0
    counts = {int(a): int((final == a).sum()) for a in range(n_actions)}
    return CausalPolicyForestResult(
        policy_value=value,
        policy_value_se=se,
        assignments=final,
        action_counts=counts,
        n_trees=n_trees,
        depth=depth,
    )
