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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .._aliases import accepts_aliases
from .._result_serialize import ResultProtocolMixin


@dataclass
class QLearningResult(ResultProtocolMixin):
    """Result of :func:`statspai.q_learning`.

    Fields: ``stage_coefs``, ``value``, ``optimal_actions``, ``K``,
    ``n_obs``, ``outcome``, ``actions``, ``detail``.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x1 = rng.normal(size=n); a1 = rng.integers(0, 2, n)
    >>> x2 = x1 + rng.normal(size=n); a2 = rng.integers(0, 2, n)
    >>> y = x1 + a1 * (1 + x1) + a2 * (0.5 - x2) + rng.normal(size=n)
    >>> df = pd.DataFrame({"x1": x1, "a1": a1, "x2": x2, "a2": a2, "y": y})
    >>> res = sp.q_learning(df, y="y", actions=["a1", "a2"],
    ...     stage_covariates=[["x1"], ["x2"]])
    >>> isinstance(res, sp.QLearningResult)
    True
    """

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
            lines.append(f"  stage {k + 1}: {coefs}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return f"QLearningResult(K={self.K}, V={self.value:.4f})"


def _build_design(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    if cols:
        X = df[list(cols)].to_numpy(dtype=float)
    else:
        X = np.zeros((len(df), 0))
    return np.column_stack([np.ones(len(df)), X])


@accepts_aliases(_strict=True, outcome="y")
def q_learning(
    data: pd.DataFrame,
    y: str,
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
    y : str
        Final-stage outcome column, higher is better (``outcome=`` is
        accepted as an alias).
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

    Notes
    -----
    Each stage's Q-function is linear in the history ``H_k`` (baseline,
    earlier actions and covariates as main effects) and in ``A_k * H_k``.
    The pseudo-outcome passed to the previous stage is the fitted maximum,
    so the earlier rule is only as good as the later working model. If the
    outcome depends on an interaction among history variables that the
    model has no term for, such as an earlier action times an earlier
    covariate, that component is lost on the way back and the earlier rule
    is biased; on ``y = x1 + a1 (1 + x1) + a2 (0.5 - x2)`` the stage-1 rule
    comes out as "always treat" where the optimum treats 84%.
    ``sp.a_learning`` and ``sp.snmm`` model the blip only and recover both
    stages on that design.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x1 = rng.normal(size=n); a1 = rng.integers(0, 2, n)
    >>> x2 = x1 + rng.normal(size=n); a2 = rng.integers(0, 2, n)
    >>> y = x1 + a1 * (1 + x1) + a2 * (0.5 - x2) + rng.normal(size=n)
    >>> df = pd.DataFrame({"x1": x1, "a1": a1, "x2": x2, "a2": a2, "y": y})
    >>> res = sp.q_learning(df, y="y", actions=["a1", "a2"],
    ...     stage_covariates=[["x1"], ["x2"]])
    >>> res.K
    2
    """
    actions = list(actions)
    stage_covs: List[List[str]] = [list(c) for c in stage_covariates]
    if len(actions) != len(stage_covs):
        raise ValueError("actions and stage_covariates must have equal length")
    K = len(actions)
    df = data.copy().reset_index(drop=True)
    n = len(df)
    baseline = list(baseline or [])

    stage_coefs: List[Dict[str, float]] = []
    optimal_actions = np.zeros((n, K), dtype=int)

    # Pseudo-outcome updated backward
    Y_tilde = df[y].to_numpy(dtype=float).copy()

    # Build each stage's history columns
    histories: List[List[str]] = []
    for k in range(K):
        hist = list(baseline)
        for j in range(k):
            hist += [actions[j]] + stage_covs[j]
        hist += stage_covs[k]
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
            adv_coef = coef[H.shape[1] :]
            advantage = H @ adv_coef
        else:
            advantage = np.full(n, coef[-1])

        optimal_actions[:, k] = (advantage > 0).astype(int)
        # Optimal Q value at (H_k, optimal action)
        if include_interactions and H.shape[1] > 0:
            # Q(H, a=0) = H @ coef_H ; Q(H, a=1) = Q0 + advantage
            Q0 = H @ coef[: H.shape[1]]
            Y_tilde = np.maximum(Q0, Q0 + advantage)
        else:
            Q0 = H @ coef[:-1]
            Y_tilde = np.maximum(Q0, Q0 + coef[-1])

        stage_coefs.append(
            {
                "intercept_block_dim": int(H.shape[1]),
                "advantage_norm": float(np.linalg.norm(advantage)),
                "fraction_treat_optimal": float(optimal_actions[:, k].mean()),
            }
        )

    stage_coefs = list(reversed(stage_coefs))
    value = float(np.mean(Y_tilde))
    return QLearningResult(
        stage_coefs=stage_coefs,
        value=value,
        optimal_actions=optimal_actions,
        K=K,
        n_obs=n,
        outcome=y,
        actions=actions,
    )


__all__ = ["q_learning", "QLearningResult"]
