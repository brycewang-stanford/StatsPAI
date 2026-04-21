"""
Linear-in-means peer-effects model (Manski 1993; Bramoullé, Djebbari
& Fortin 2009; Lee 2007).

For a network :math:`G` with row-normalised adjacency :math:`W`:

.. math::

    y_i = \\alpha + \\beta\\, \\bar y_{N(i)}
          + \\gamma\\, \\bar x_{N(i)} + \\delta\\, x_i + \\varepsilon_i.

The parameters are identified under the Bramoullé-Djebbari-Fortin
2009 exclusion restriction: there must exist a pair of nodes at
network distance 2 who are not directly connected (so :math:`W^2 x`
is not colinear with :math:`W x`).

We estimate by 2SLS with instruments :math:`W^2 x, W^3 x, \\ldots` for
the endogenous :math:`W y`.

References
----------
Manski, C. F. (1993). "Identification of endogenous social effects:
the reflection problem." *Review of Economic Studies*, 60(3), 531-542.

Bramoullé, Y., Djebbari, H., & Fortin, B. (2009). "Identification of
peer effects through social networks." *Journal of Econometrics*, 150(1), 41-55.

Lee, L. F. (2007). "Identification and estimation of econometric
models with group interactions, contextual factors and fixed effects."
*Journal of Econometrics*, 140(2), 333-374.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PeerEffectsResult:
    endogenous_peer: float  # beta (mean of neighbors' y)
    contextual_peer: Dict[str, float]  # gamma per covariate
    direct: Dict[str, float]  # delta per covariate
    se: Dict[str, float]
    ci: Dict[str, tuple]
    n_obs: int
    coefficients: pd.DataFrame
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        return (
            "Linear-in-Means Peer Effects (Bramoullé-Djebbari-Fortin 2009)\n"
            "-------------------------------------------------------------\n"
            f"  endogenous peer beta : {self.endogenous_peer:+.4f}\n"
            f"  N                    : {self.n_obs}\n"
            f"  coefficients:\n{self.coefficients.to_string(index=False)}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"PeerEffectsResult(beta={self.endogenous_peer:+.4f})"


def _row_normalise(W: np.ndarray) -> np.ndarray:
    rs = W.sum(axis=1, keepdims=True)
    rs = np.where(rs == 0, 1.0, rs)
    return W / rs


def peer_effects(
    data: pd.DataFrame,
    y: str,
    covariates: Sequence[str],
    W,
    include_contextual: bool = True,
    alpha: float = 0.05,
) -> PeerEffectsResult:
    """
    Fit the linear-in-means peer-effects model via 2SLS.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    covariates : sequence of str
    W : ndarray or W-like with ``.full()`` / ``.toarray()``.
    include_contextual : bool, default True
        Include the contextual peer effect (W X) as regressors.
    alpha : float, default 0.05

    Returns
    -------
    PeerEffectsResult
    """
    cov = list(covariates)
    df = data[[y] + cov].dropna().reset_index(drop=True)
    n = len(df)
    Y = df[y].to_numpy(dtype=float)
    X = df[cov].to_numpy(dtype=float) if cov else np.zeros((n, 0))

    if hasattr(W, "full"):
        W_mat = np.asarray(W.full()[0], dtype=float)
    elif hasattr(W, "toarray"):
        W_mat = np.asarray(W.toarray(), dtype=float)
    else:
        W_mat = np.asarray(W, dtype=float)
    W_mat = _row_normalise(W_mat)

    WY = W_mat @ Y
    WX = W_mat @ X if X.shape[1] > 0 else np.zeros((n, 0))
    W2X = W_mat @ WX if WX.shape[1] > 0 else np.zeros((n, 0))
    W3X = W_mat @ W2X if W2X.shape[1] > 0 else np.zeros((n, 0))

    # Build design: [1, WY, X, optionally WX]
    cols = ["(Intercept)", "WY"]
    endog_cols = ["WY"]
    full = np.column_stack([np.ones(n), WY])
    if X.shape[1] > 0:
        full = np.column_stack([full, X])
        cols += cov
    if include_contextual and X.shape[1] > 0:
        full = np.column_stack([full, WX])
        cols += [f"W*{c}" for c in cov]

    # Instruments: [1, X, WX, W^2 X, W^3 X]
    instr_list = [np.ones(n)]
    if X.shape[1] > 0:
        instr_list += [X, WX, W2X, W3X]
    instruments = np.column_stack(instr_list) if instr_list else np.ones((n, 1))

    # 2SLS
    PZ = instruments @ np.linalg.pinv(instruments.T @ instruments) @ instruments.T
    X_hat = PZ @ full
    beta = np.linalg.pinv(X_hat.T @ full) @ X_hat.T @ Y
    resid = Y - full @ beta
    vcov = (
        np.linalg.pinv(X_hat.T @ full)
        @ X_hat.T @ np.diag(resid ** 2) @ X_hat
        @ np.linalg.pinv(X_hat.T @ full).T
    )
    se = np.sqrt(np.diag(vcov))

    crit = float(stats.norm.ppf(1 - alpha / 2))
    ci_map: Dict[str, tuple] = {}
    se_map: Dict[str, float] = {}
    for i, name in enumerate(cols):
        se_map[name] = float(se[i])
        ci_map[name] = (float(beta[i] - crit * se[i]), float(beta[i] + crit * se[i]))

    coef_df = pd.DataFrame({"variable": cols, "coef": beta, "se": se})

    # Extract beta (WY coefficient) and gamma/delta
    beta_peer = float(beta[cols.index("WY")])
    contextual = {}
    direct = {}
    for i, name in enumerate(cols):
        if name.startswith("W*"):
            contextual[name[2:]] = float(beta[i])
        elif name in cov:
            direct[name] = float(beta[i])

    return PeerEffectsResult(
        endogenous_peer=beta_peer,
        contextual_peer=contextual,
        direct=direct,
        se=se_map,
        ci=ci_map,
        n_obs=n,
        coefficients=coef_df,
        detail={"include_contextual": include_contextual},
    )


__all__ = ["peer_effects", "PeerEffectsResult"]
