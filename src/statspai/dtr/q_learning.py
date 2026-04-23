"""
Q-learning for dynamic treatment regimes (Murphy 2005; Chakraborty &
Moodie 2013).

Two-stage setting
-----------------
At stage :math:`k = 1, ..., K`, the decision maker observes history
:math:`H_k = (X_1, A_1, X_2, A_2, ..., X_k)` and chooses action
:math:`A_k \\in \\{0, 1\\}`. A final outcome :math:`Y` is observed.

Q-learning fits a *Q-function* at each stage by backward induction:

.. math::

    Q_K(H_K, A_K)     = E[Y \\mid H_K, A_K]
    Q_{k}(H_k, A_k)   = E[ \\max_{a} Q_{k+1}(H_{k+1}, a) \\mid H_k, A_k ]

The optimal regime is :math:`d_k^*(H_k) = \\arg\\max_{a} Q_k(H_k, a)`.

The implementation uses linear regression at each stage (so ``value``
interpretation is the linear-model expectation). Non-linear bases are
easy to pass via ``basis=`` (callable returning an augmented design
matrix), allowing tree / neural extensions.

References
----------
Murphy, S. A. (2005). "A generalization error for Q-learning."
*JMLR*, 6, 1073-1097.

Chakraborty, B., & Moodie, E. E. M. (2013).
*Statistical Methods for Dynamic Treatment Regimes*. Springer. [@chakraborty2013statistical]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Any

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


@dataclass
class QLearningResult:
    stage_coefs: List[Dict[str, float]]
    value: float
    optimal_actions: np.ndarray  # (n, K)
    K: int
    n_obs: int
    outcome: str
    actions: Sequence[str]
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        lines = [
            "Q-learning DTR",
            "---------------",
            f"  stages    : {self.K}",
            f"  N         : {self.n_obs}",
            f"  V (value) : {self.value:.4f}",
            "",
            "Optimal-rule coefficients per stage (linear Q):",
        ]
        for k, coefs in enumerate(self.stage_coefs):
            lines.append(f"  stage {k+1}: {coefs}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return f"QLearningResult(K={self.K}, V={self.value:.4f})"


def _build_design(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    if cols:
        X = df[list(cols)].to_numpy(dtype=float)
    else:
        X = np.zeros((len(df), 0))
    return np.column_stack([np.ones(len(df)), X])


def q_learning(
    data: pd.DataFrame,
    outcome: str,
    actions: Sequence[str],
    stage_covariates: Sequence[Sequence[str]],
    baseline: Optional[Sequence[str]] = None,
    include_interactions: bool = True,
) -> QLearningResult:
    """
    Backward-induction Q-learning for a K-stage binary-action DTR.

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
        Final-stage outcome (higher is better).
    actions : sequence of str, length K
        Column names for the stage-k actions (binary 0/1).
    stage_covariates : sequence of sequences of str, length K
        Covariates observed *before* the stage-k decision (may be empty).
    baseline : sequence of str, optional
        Baseline covariates in the history at all stages.
    include_interactions : bool, default True
        If True, Q-models include :math:`A_k \\times H_k` interactions,
        enabling nontrivial optimal rules.

    Returns
    -------
    QLearningResult
    """
    actions = list(actions)
    stage_covariates = [list(c) for c in stage_covariates]
    if len(actions) != len(stage_covariates):
        raise ValueError("actions and stage_covariates must have equal length")
    K = len(actions)
    df = data.copy().reset_index(drop=True)
    n = len(df)
    baseline = list(baseline or [])

    stage_coefs: List[Dict[str, float]] = []
    optimal_actions = np.zeros((n, K), dtype=int)

    # Pseudo-outcome updated backward
    Y_tilde = df[outcome].to_numpy(dtype=float).copy()

    # Build each stage's history columns
    histories: List[List[str]] = []
    for k in range(K):
        hist = list(baseline)
        for j in range(k):
            hist += [actions[j]] + stage_covariates[j]
        hist += stage_covariates[k]
        histories.append(hist)

    for k in reversed(range(K)):
        hist = histories[k]
        H = _build_design(df, hist)
        A = df[actions[k]].to_numpy(dtype=int).astype(float)
        if include_interactions and H.shape[1] > 0:
            design = np.column_stack([H, A[:, None] * H])
            # coefficient layout: H-block, H*A-block
        else:
            design = np.column_stack([H, A])
        reg = LinearRegression(fit_intercept=False).fit(design, Y_tilde)
        coef = reg.coef_

        # Extract "advantage" A_k = Q(H, A=1) - Q(H, A=0)
        if include_interactions and H.shape[1] > 0:
            # advantage = H @ coef_interaction
            adv_coef = coef[H.shape[1]:]
            advantage = H @ adv_coef
        else:
            advantage = np.full(n, coef[-1])

        optimal_actions[:, k] = (advantage > 0).astype(int)
        # Optimal Q value at (H_k, optimal action)
        if include_interactions and H.shape[1] > 0:
            # Q(H, a=0) = H @ coef_H ; Q(H, a=1) = Q0 + advantage
            Q0 = H @ coef[:H.shape[1]]
            Y_tilde = np.maximum(Q0, Q0 + advantage)
        else:
            Q0 = H @ coef[:-1]
            Y_tilde = np.maximum(Q0, Q0 + coef[-1])

        stage_coefs.append({
            "intercept_block_dim": int(H.shape[1]),
            "advantage_norm": float(np.linalg.norm(advantage)),
            "fraction_treat_optimal": float(optimal_actions[:, k].mean()),
        })

    stage_coefs = list(reversed(stage_coefs))
    value = float(np.mean(Y_tilde))
    return QLearningResult(
        stage_coefs=stage_coefs,
        value=value,
        optimal_actions=optimal_actions,
        K=K,
        n_obs=n,
        outcome=outcome,
        actions=actions,
    )


__all__ = ["q_learning", "QLearningResult"]
