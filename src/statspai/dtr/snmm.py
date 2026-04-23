"""
Structural Nested Mean Model (SNMM) via G-estimation.

Robins (1994) proposed SNMMs as a way to encode the *causal* effect of
a time-varying treatment at each stage while leaving the marginal mean
unspecified. For a K-stage setting and a binary treatment:

.. math::

    γ_k(H_k, A_k; ψ_k) = A_k \\cdot H_k^\\top ψ_k

The SNMM posits

.. math::

    E\\left[Y^{(\\bar A_k, 0, \\ldots, 0)} - Y^{(\\bar A_{k-1}, 0, \\ldots, 0)}
    \\;\\big|\\; H_k, A_k\\right] = γ_k(H_k, A_k; ψ_k).

G-estimation proceeds backward, constructing "blipped-down" pseudo-
outcomes

.. math::

    U_k = Y - \\sum_{j \\ge k} γ_j(H_j, A_j; ψ_j)

and solving moment equations

.. math::

    \\sum_i S(H_{k,i}) (A_{k,i} - π_k(H_{k,i}))
        (U_{k,i} - γ_k(H_{k,i}, A_{k,i}; ψ_k)) = 0.

The closed-form solution for a linear blip is

.. math::

    \\hat ψ_k = \\left[ \\sum_i (A_{k,i} - π_{k,i}) A_{k,i} H_{k,i} H_{k,i}^\\top \\right]^{-1}
               \\sum_i (A_{k,i} - π_{k,i}) H_{k,i} U_{k,i}.

Under consistency, positivity, sequential exchangeability and correct
propensity specification, :math:`\\hat ψ_k` is consistent for the true
blip coefficients.

References
----------
Robins, J. M. (1994). "Correcting for non-compliance in randomized
trials using structural nested mean models." *Communications in
Statistics*, 23(8), 2379-2412. [@robins1994correcting]

Vansteelandt, S. & Joffe, M. (2014). "Structural nested models and
G-estimation." *Statistical Science*, 29(4), 707-731. [@vansteelandt2014structural]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Any

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression


@dataclass
class SNMMResult:
    blip_params: List[np.ndarray]
    optimal_actions: np.ndarray
    value: float
    K: int
    n_obs: int
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        lines = [
            "SNMM via G-estimation",
            "----------------------",
            f"  stages    : {self.K}",
            f"  N         : {self.n_obs}",
            f"  V (blip-adjusted mean) : {self.value:.4f}",
        ]
        for k, p in enumerate(self.blip_params):
            lines.append(f"  stage {k+1} psi: {np.round(p, 4).tolist()}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SNMMResult(K={self.K}, V={self.value:.4f})"


def snmm(
    data: pd.DataFrame,
    outcome: str,
    actions: Sequence[str],
    stage_covariates: Sequence[Sequence[str]],
    baseline: Optional[Sequence[str]] = None,
    propensity_bounds: tuple = (0.05, 0.95),
) -> SNMMResult:
    """
    G-estimation of a linear structural nested mean model.

    Parameters mirror :func:`a_learning`. The returned ``blip_params``
    are the stage-wise :math:`\\hat ψ_k`.
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

    U = df[outcome].to_numpy(dtype=float).copy()
    psis: List[np.ndarray] = []
    optimal_actions = np.zeros((n, K), dtype=int)

    for k in reversed(range(K)):
        H = build_hist(k)
        A = df[actions[k]].to_numpy(dtype=int).astype(float)
        if len(np.unique(A)) == 1:
            pi = np.full(n, np.mean(A))
        else:
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
            lr.fit(H, A.astype(int))
            pi = lr.predict_proba(H)[:, 1]
        pi = np.clip(pi, *propensity_bounds)

        # Solve closed-form: psi = [sum (A-pi)A H H^T]^-1 sum (A-pi) H U
        w = (A - pi) * A
        LHS = (w[:, None, None] * (H[:, :, None] * H[:, None, :])).sum(axis=0)
        RHS = ((A - pi) * U)[:, None] * H
        RHS = RHS.sum(axis=0)
        try:
            psi_k = np.linalg.solve(LHS + 1e-6 * np.eye(LHS.shape[0]), RHS)
        except np.linalg.LinAlgError:
            psi_k = np.linalg.lstsq(LHS, RHS, rcond=None)[0]
        psis.append(psi_k)

        contrast = H @ psi_k
        optimal_actions[:, k] = (contrast > 0).astype(int)
        # Blip-down: U <- U - A * contrast
        U = U - A * contrast

    psis = list(reversed(psis))
    return SNMMResult(
        blip_params=psis,
        optimal_actions=optimal_actions,
        value=float(np.mean(U)),
        K=K,
        n_obs=n,
    )


__all__ = ["snmm", "SNMMResult"]
