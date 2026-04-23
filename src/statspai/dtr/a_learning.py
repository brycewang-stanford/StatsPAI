"""
A-learning for dynamic treatment regimes (Murphy 2003; Schulte et al.
2014).

Unlike Q-learning, A-learning models **only** the contrast function
:math:`C_k(H_k) = E[Y(1) - Y(0) \\mid H_k]` at each stage, sidestepping
the need to specify the (often nuisance) main-effect model of the Q-
function. The cost is additional moment conditions requiring a
propensity model :math:`\\pi_k(H_k) = P(A_k = 1 \\mid H_k)`.

The two-stage estimator follows Robins (2004) G-estimation with
propensity-weighted residuals.

At each stage k (backward):

.. math::

    C_k(H_k; \\psi) = H_k^\\top \\psi_k

and we solve

.. math::

    \\sum_i (A_{k,i} - \\pi_{k,i})\\, H_{k,i}\\,
    (\\tilde Y_i - C_k(H_{k,i}; \\psi_k) A_{k,i}) = 0.

After getting :math:`\\hat\\psi_k`, we update
:math:`\\tilde Y_i \\leftarrow \\tilde Y_i - A_{k,i} \\cdot C_k(H_{k,i}; \\hat\\psi_k)`
plus the max-advantage term.

References
----------
Murphy, S. A. (2003). "Optimal dynamic treatment regimes." *JRSS B*,
65(2), 331-355. [@murphy2003optimal]

Schulte, P. J., Tsiatis, A. A., Laber, E. B., & Davidian, M. (2014).
"Q- and A-learning methods for estimating optimal dynamic treatment
regimes." *Statistical Science*, 29(4), 640-661. [@schulte2014mathbf]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Any

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression


@dataclass
class ALearningResult:
    psi: List[np.ndarray]
    value: float
    optimal_actions: np.ndarray
    K: int
    n_obs: int
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        lines = [
            "A-learning DTR",
            "---------------",
            f"  stages    : {self.K}",
            f"  N         : {self.n_obs}",
            f"  value V   : {self.value:.4f}",
            "",
        ]
        for k, psi_k in enumerate(self.psi):
            lines.append(
                f"  stage {k+1} psi (contrast coef): {np.round(psi_k, 4).tolist()}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ALearningResult(K={self.K}, V={self.value:.4f})"


def a_learning(
    data: pd.DataFrame,
    outcome: str,
    actions: Sequence[str],
    stage_covariates: Sequence[Sequence[str]],
    baseline: Optional[Sequence[str]] = None,
    propensity_bounds: tuple = (0.05, 0.95),
) -> ALearningResult:
    """
    G-estimation / A-learning for a K-stage binary-action DTR.

    Parameters
    ----------
    data : pd.DataFrame
    outcome : str
    actions : sequence of str, length K
    stage_covariates : sequence of sequences of str
        Covariates observed just before stage k's decision.
    baseline : sequence of str, optional
    propensity_bounds : (float, float), default (0.05, 0.95)
        Clip propensity to avoid division blow-up.

    Returns
    -------
    ALearningResult
    """
    actions = list(actions)
    stage_covariates = [list(c) for c in stage_covariates]
    if len(actions) != len(stage_covariates):
        raise ValueError("actions and stage_covariates must have equal length")
    K = len(actions)
    df = data.copy().reset_index(drop=True)
    n = len(df)
    baseline = list(baseline or [])

    def build_hist(k: int) -> np.ndarray:
        hist = list(baseline)
        for j in range(k):
            hist += [actions[j]] + stage_covariates[j]
        hist += stage_covariates[k]
        if not hist:
            return np.ones((n, 1))
        X = df[hist].to_numpy(dtype=float)
        return np.column_stack([np.ones(n), X])

    Y_tilde = df[outcome].to_numpy(dtype=float).copy()
    psis: List[np.ndarray] = []
    optimal_actions = np.zeros((n, K), dtype=int)

    for k in reversed(range(K)):
        H = build_hist(k)
        A = df[actions[k]].to_numpy(dtype=int).astype(float)

        # Propensity model
        if len(np.unique(A)) == 1:
            pi = np.full(n, np.mean(A))
        else:
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
            lr.fit(H, A.astype(int))
            pi = lr.predict_proba(H)[:, 1]
        pi = np.clip(pi, *propensity_bounds)

        # G-estimation moment: sum (A - pi) * H * (Y - A * H psi) = 0
        #   => sum (A-pi) H Y = sum (A-pi) A H H^T psi
        weight = (A - pi)
        LHS = (weight * A)[:, None, None] * (H[:, :, None] * H[:, None, :])
        LHS_mat = LHS.sum(axis=0)
        RHS_vec = (weight * Y_tilde)[:, None] * H
        RHS_sum = RHS_vec.sum(axis=0)
        try:
            psi_k = np.linalg.solve(LHS_mat + 1e-6 * np.eye(LHS_mat.shape[0]), RHS_sum)
        except np.linalg.LinAlgError:
            psi_k = np.linalg.lstsq(LHS_mat, RHS_sum, rcond=None)[0]
        psis.append(psi_k)

        contrast = H @ psi_k
        optimal_actions[:, k] = (contrast > 0).astype(int)
        # Update pseudo-outcome:
        # Y_tilde <- Y_tilde - (A - d*(H)) * C(H; psi)
        best = optimal_actions[:, k].astype(float)
        Y_tilde = Y_tilde - (A - best) * contrast

    psis = list(reversed(psis))
    value = float(np.mean(Y_tilde))
    return ALearningResult(
        psi=psis,
        value=value,
        optimal_actions=optimal_actions,
        K=K,
        n_obs=n,
    )


__all__ = ["a_learning", "ALearningResult"]
